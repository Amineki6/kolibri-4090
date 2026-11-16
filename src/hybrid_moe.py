"""Step 3.2: Hybrid Decode + Prefill MoE Dispatcher over HybridExpertCache."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401 - registers torch.ops.trtllm
from tensorrt_llm._torch.moe.fused_moe.marlin.base import sum_topk_expert_outputs
from tensorrt_llm.quantization.utils.marlin_utils import marlin_make_workspace

from .config import E, H, I_MOE, MOE_BLOCK_SIZE, NUM_LAYERS, TOPK
from .expert_cache import BYTES_PER_EXPERT, HybridExpertCache
from .kolibri_kernels import flashinfer_silu_and_mul


@dataclass
class HybridMoEStats:
    total_calls: int = 0
    decode_0miss_calls: int = 0
    decode_miss_calls: int = 0
    prefill_split_calls: int = 0
    total_pair_evals: int = 0
    cache_hit_pairs: int = 0
    experts_transferred: int = 0
    layer_pair_evals: List[int] = field(default_factory=lambda: [0] * NUM_LAYERS)
    layer_hit_pairs: List[int] = field(default_factory=lambda: [0] * NUM_LAYERS)
    layer_experts_transferred: List[int] = field(default_factory=lambda: [0] * NUM_LAYERS)

    @property
    def hit_rate(self) -> float:
        return self.cache_hit_pairs / max(self.total_pair_evals, 1)

    @property
    def pcie_mb_transferred(self) -> float:
        return (self.experts_transferred * BYTES_PER_EXPERT) / 1e6

    def layer_hit_rates_pct(self) -> List[float]:
        return [
            round((hits / evals) * 100.0, 2) if evals > 0 else 100.0
            for hits, evals in zip(self.layer_hit_pairs, self.layer_pair_evals)
        ]


class HybridMarlinMoE:
    """Routes MoE layers across 0-miss decode, <=6-miss single-launch decode, and Hit/Miss prefill."""

    def __init__(self, cache: HybridExpertCache, max_tokens: int = 1024) -> None:
        self.cache = cache
        self.device = cache.device
        self.max_tokens = max_tokens
        self.workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        max_pairs = max_tokens * TOPK
        max_pad = max_pairs + E * MOE_BLOCK_SIZE
        max_blocks = (max_pad + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE
        self._s_ids1 = torch.empty(max_pad, dtype=torch.int32, device=self.device)
        self._e_ids1 = torch.empty(max_blocks, dtype=torch.int32, device=self.device)
        self._n_pad1 = torch.empty(1, dtype=torch.int32, device=self.device)
        self._s_ids2 = torch.empty(max_pad, dtype=torch.int32, device=self.device)
        self._e_ids2 = torch.empty(max_blocks, dtype=torch.int32, device=self.device)
        self._n_pad2 = torch.empty(1, dtype=torch.int32, device=self.device)
        self._pin_slots = torch.empty(max_pairs, dtype=torch.int32).pin_memory()
        self._gpu_slots = torch.empty(max_pairs, dtype=torch.int32, device=self.device)

        self._buf_cursor = 0
        self._step_id = 0
        self.stats = HybridMoEStats()

    def reset_stats(self) -> None:
        self.stats = HybridMoEStats()

    def _run_marlin_gemm1_gemm2(
        self,
        x_in: torch.Tensor,
        ids_2d: torch.Tensor,
        weights_2d: torch.Tensor,
        slab: Dict[str, torch.Tensor],
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        """Execute GEMM1 -> SwiGLU -> GEMM2 on `slab`, returning `[num_rows * top_k, H]`."""
        n_rows = x_in.shape[0]
        assert n_rows * top_k <= self.max_tokens * TOPK, (
            f"Token-expert pairs {n_rows * top_k} exceed max_tokens*TOPK={self.max_tokens * TOPK}"
        )
        max_pad1 = n_rows * top_k + num_experts * MOE_BLOCK_SIZE
        s_ids1 = self._s_ids1[:max_pad1]
        e_ids1 = self._e_ids1[: (max_pad1 + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE]
        torch.ops.trtllm.moe_align_block_size(
            ids_2d, num_experts, MOE_BLOCK_SIZE, s_ids1, e_ids1, self._n_pad1
        )

        g1 = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            x_in.contiguous(),
            slab["w13_marlin"],
            b_scales=slab["w13_scale_marlin"],
            global_scale=slab["w13_gs_marlin"],
            workspace=self.workspace,
            sorted_token_ids=s_ids1,
            expert_ids=e_ids1,
            num_tokens_past_padded=self._n_pad1,
            topk_weights=weights_2d,
            moe_block_size=MOE_BLOCK_SIZE,
            top_k=top_k,
            mul_topk_weights=False,
            size_n=2 * I_MOE,
            size_k=H,
            out_dtype=torch.bfloat16,
            use_fp32_reduce=False,
        )

        hidden = flashinfer_silu_and_mul(g1)

        n_g2 = n_rows * top_k
        ids_g2 = ids_2d.reshape(-1, 1).contiguous()
        w_g2 = weights_2d.reshape(-1, 1).contiguous()
        if top_k == 1:
            s_ids2, e_ids2, n_pad2 = s_ids1, e_ids1, self._n_pad1
        else:
            max_pad2 = n_g2 + num_experts * MOE_BLOCK_SIZE
            s_ids2 = self._s_ids2[:max_pad2]
            e_ids2 = self._e_ids2[: (max_pad2 + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE]
            n_pad2 = self._n_pad2
            torch.ops.trtllm.moe_align_block_size(
                ids_g2, num_experts, MOE_BLOCK_SIZE, s_ids2, e_ids2, n_pad2
            )

        return torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            hidden,
            slab["w2_marlin"],
            b_scales=slab["w2_scale_marlin"],
            global_scale=slab["w2_gs_marlin"],
            workspace=self.workspace,
            sorted_token_ids=s_ids2,
            expert_ids=e_ids2,
            num_tokens_past_padded=n_pad2,
            topk_weights=w_g2,
            moe_block_size=MOE_BLOCK_SIZE,
            top_k=1,
            mul_topk_weights=True,
            size_n=H,
            size_k=I_MOE,
            out_dtype=torch.bfloat16,
            use_fp32_reduce=False,
        )

    def forward_layer(
        self,
        moe_in: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_idx: int,
        shared_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        promote_misses: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute routed MoE (and optional overlapped shared expert) for one layer.

        Returns `(routed_out, shared_out)` where `shared_out` is `None` if `shared_fn` is `None`.
        """
        n = moe_in.shape[0]
        self._step_id += 1
        step_id = self._step_id
        self.stats.total_calls += 1
        pair_evals = n * TOPK
        self.stats.total_pair_evals += pair_evals
        self.stats.layer_pair_evals[layer_idx] += pair_evals

        # Fast small-batch / decode path (n <= 4): single D2H of <=24 ints avoids CUB unique & double sync.
        if n <= 4:
            e_list = topk_ids.reshape(-1).tolist()
            state = self.cache.cache_states[layer_idx]
            e2s = state.expert_to_slot
            miss_map: Dict[int, int] = {}
            miss_unique_list = []
            pin_slots = self._pin_slots
            hit_pairs = 0
            k_persist = self.cache.persistent_slots

            for idx, eid in enumerate(e_list):
                slot = int(e2s[eid])
                if slot >= 0:
                    hit_pairs += 1
                    state.last_used[slot] = step_id
                    state.hit_count[slot] += 1
                    pin_slots[idx] = slot
                else:
                    m_idx = miss_map.get(eid)
                    if m_idx is None:
                        m_idx = len(miss_unique_list)
                        miss_map[eid] = m_idx
                        miss_unique_list.append(eid)
                    pin_slots[idx] = k_persist + m_idx

            self.stats.cache_hit_pairs += hit_pairs
            self.stats.layer_hit_pairs[layer_idx] += hit_pairs
            n_miss_unique = len(miss_unique_list)

            if n_miss_unique == 0:
                self.stats.decode_0miss_calls += 1
                n_pairs = n * TOPK
                self._gpu_slots[:n_pairs].copy_(pin_slots[:n_pairs], non_blocking=True)
                g2 = self._run_marlin_gemm1_gemm2(
                    moe_in,
                    self._gpu_slots[:n_pairs].view(n, TOPK),
                    topk_weights,
                    self.cache.gpu_caches[layer_idx],
                    num_experts=k_persist,
                    top_k=TOPK,
                )
                routed = sum_topk_expert_outputs(g2, n, TOPK, H, torch.bfloat16)
                shared = shared_fn(moe_in) if shared_fn is not None else None
                return routed, shared

            if n_miss_unique <= self.cache.decode_miss_slots:
                self.stats.decode_miss_calls += 1
                self.stats.experts_transferred += n_miss_unique
                self.stats.layer_experts_transferred[layer_idx] += n_miss_unique
                buf = self._buf_cursor
                self._buf_cursor ^= 1

                miss_t = torch.tensor(miss_unique_list, dtype=torch.long)
                self.cache.decode_stage_misses_async(layer_idx, miss_t, buf_idx=buf)
                shared = shared_fn(moe_in) if shared_fn is not None else None

                n_pairs = n * TOPK
                self._gpu_slots[:n_pairs].copy_(pin_slots[:n_pairs], non_blocking=True)
                self.cache.wait_copy(buf)

                g2 = self._run_marlin_gemm1_gemm2(
                    moe_in,
                    self._gpu_slots[:n_pairs].view(n, TOPK),
                    topk_weights,
                    self.cache.gpu_caches[layer_idx],
                    num_experts=self.cache.cache_slots,
                    top_k=TOPK,
                )
                routed = sum_topk_expert_outputs(g2, n, TOPK, H, torch.bfloat16)
                if promote_misses:
                    self.cache.promote_decode_misses(layer_idx, miss_unique_list, step_id)
                return routed, shared

        # Vectorized GPU lookup path (n > 4, or n <= 4 with > decode_miss_slots unique misses)
        slot_map = self.cache.gpu_expert_to_slot[layer_idx]
        flat_ids = topk_ids.reshape(-1).to(torch.long)
        flat_slots = slot_map[flat_ids]
        miss_mask = flat_slots < 0
        miss_count_pairs = int(miss_mask.sum().item())
        hit_pairs_vec = n * TOPK - miss_count_pairs
        self.stats.cache_hit_pairs += hit_pairs_vec
        self.stats.layer_hit_pairs[layer_idx] += hit_pairs_vec

        if miss_count_pairs == 0:
            self.stats.decode_0miss_calls += 1
            state = self.cache.cache_states[layer_idx]
            for eid in torch.unique(flat_ids).tolist():
                state.record_access(eid, step_id)
            g2 = self._run_marlin_gemm1_gemm2(
                moe_in,
                flat_slots.view(n, TOPK).contiguous(),
                topk_weights,
                self.cache.gpu_caches[layer_idx],
                num_experts=self.cache.persistent_slots,
                top_k=TOPK,
            )
            routed = sum_topk_expert_outputs(g2, n, TOPK, H, torch.bfloat16)
            shared = shared_fn(moe_in) if shared_fn is not None else None
            return routed, shared

        miss_eids = flat_ids[miss_mask]
        unique_miss_eids, miss_inv = torch.unique(miss_eids, sorted=False, return_inverse=True)
        n_miss_unique = int(unique_miss_eids.numel())

        if n_miss_unique <= self.cache.decode_miss_slots:
            self.stats.decode_miss_calls += 1
            self.stats.experts_transferred += n_miss_unique
            self.stats.layer_experts_transferred[layer_idx] += n_miss_unique
            buf = self._buf_cursor
            self._buf_cursor ^= 1

            self.cache.decode_stage_misses_async(layer_idx, unique_miss_eids, buf_idx=buf)
            shared = shared_fn(moe_in) if shared_fn is not None else None

            remapped = flat_slots.clone()
            remapped[miss_mask] = (self.cache.persistent_slots + miss_inv).to(torch.int32)

            state = self.cache.cache_states[layer_idx]
            hit_mask = ~miss_mask
            if hit_mask.any():
                for eid in torch.unique(flat_ids[hit_mask]).tolist():
                    state.record_access(eid, step_id)

            self.cache.wait_copy(buf)
            g2 = self._run_marlin_gemm1_gemm2(
                moe_in,
                remapped.view(n, TOPK).contiguous(),
                topk_weights,
                self.cache.gpu_caches[layer_idx],
                num_experts=self.cache.cache_slots,
                top_k=TOPK,
            )
            routed = sum_topk_expert_outputs(g2, n, TOPK, H, torch.bfloat16)

            if promote_misses:
                self.cache.promote_decode_misses(layer_idx, unique_miss_eids.tolist(), step_id)
            return routed, shared

        # Path C: > 6 unique misses (Prefill / large batch) -> Hit/Miss split + overlapped DMA & compute
        self.stats.prefill_split_calls += 1
        self.stats.experts_transferred += n_miss_unique
        self.stats.layer_experts_transferred[layer_idx] += n_miss_unique
        buf = self._buf_cursor
        self._buf_cursor ^= 1

        m_staged, stage_slab = self.cache.stage_experts_async(
            layer_idx, unique_miss_eids, buf_idx=buf
        )

        # Compute stream executes shared_expert + cached Hit-MoE while PCIe transfers misses
        shared = shared_fn(moe_in) if shared_fn is not None else None

        flat_tok = torch.arange(n, device=self.device, dtype=torch.long).repeat_interleave(TOPK)
        flat_w = topk_weights.reshape(-1)
        out_acc = torch.zeros((n, H), dtype=torch.float32, device=self.device)

        hit_mask = ~miss_mask
        if hit_mask.any():
            hit_tok = flat_tok[hit_mask]
            hit_slots = flat_slots[hit_mask].view(-1, 1).contiguous()
            hit_w = flat_w[hit_mask].view(-1, 1).contiguous()
            hit_out = self._run_marlin_gemm1_gemm2(
                moe_in[hit_tok],
                hit_slots,
                hit_w,
                self.cache.gpu_caches[layer_idx],
                num_experts=self.cache.persistent_slots,
                top_k=1,
            )
            out_acc.index_add_(0, hit_tok, hit_out.float())

        self.cache.wait_copy(buf)
        miss_tok = flat_tok[miss_mask]
        miss_slots = miss_inv.to(torch.int32).view(-1, 1).contiguous()
        miss_w = flat_w[miss_mask].view(-1, 1).contiguous()
        miss_out = self._run_marlin_gemm1_gemm2(
            moe_in[miss_tok],
            miss_slots,
            miss_w,
            stage_slab,
            num_experts=m_staged,
            top_k=1,
        )
        out_acc.index_add_(0, miss_tok, miss_out.float())
        return out_acc.to(torch.bfloat16), shared
