"""Step 3.3: Hybrid 40-layer SWA (513-token ring) + 10-layer Full-RNoPE KV Cache (NHD Paged Decode)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import flashinfer
import torch
import torch.nn.functional as F

from .config import (
    HD,
    NKV,
    NQ,
    NUM_LAYERS,
    RMS_EPS,
    SLIDING_WINDOW,
    is_full_attention_layer,
)
from .kolibri_kernels import (
    flashinfer_apply_rope_inplace,
    kv_append_decode_cuda,
    make_rope_cos_sin_cache,
)

KV_PAGE_SIZE = 16


@dataclass
class KVCacheStats:
    max_seq_len: int
    num_swa_layers: int
    num_full_layers: int
    swa_kv_mib: float
    full_kv_mib: float
    total_kv_mib: float
    total_kv_gib: float


class HybridKVCache:
    """Manages BF16 KV cache for 40 SWA layers (capped at 513 tokens) + 10 Full-RNoPE layers."""

    def __init__(
        self,
        max_seq_len: int = 100_000,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.seq_len = 0
        self.page_size = KV_PAGE_SIZE

        self.layer_is_full: List[bool] = [is_full_attention_layer(i) for i in range(NUM_LAYERS)]
        self.layer_sub_idx: List[int] = []
        swa_count = 0
        full_count = 0
        for is_full in self.layer_is_full:
            if is_full:
                self.layer_sub_idx.append(full_count)
                full_count += 1
            else:
                self.layer_sub_idx.append(swa_count)
                swa_count += 1

        self.swa_pages = (SLIDING_WINDOW + self.page_size - 1) // self.page_size
        self.swa_capacity = self.swa_pages * self.page_size
        self.full_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self.full_capacity = self.full_pages * self.page_size

        # NHD layout: [num_layers, capacity, NKV, HD] (zero-copy view into [pages, page_size, NKV, HD])
        self.k_swa = torch.empty((swa_count, self.swa_capacity, NKV, HD), dtype=dtype, device=device)
        self.v_swa = torch.empty((swa_count, self.swa_capacity, NKV, HD), dtype=dtype, device=device)

        self.k_full = torch.empty((full_count, self.full_capacity, NKV, HD), dtype=dtype, device=device)
        self.v_full = torch.empty((full_count, self.full_capacity, NKV, HD), dtype=dtype, device=device)

        self._swa_kv_paged: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                self.k_swa[i].view(self.swa_pages, self.page_size, NKV, HD),
                self.v_swa[i].view(self.swa_pages, self.page_size, NKV, HD),
            )
            for i in range(swa_count)
        ]
        self._full_kv_paged: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                self.k_full[i].view(self.full_pages, self.page_size, NKV, HD),
                self.v_full[i].view(self.full_pages, self.page_size, NKV, HD),
            )
            for i in range(full_count)
        ]

        # Separate workspaces & static metadata buffers for CUDA-graph-compatible FlashInfer decode
        self._fi_ws_swa = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._fi_ws_full = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

        self._swa_indptr_gpu = torch.empty(2, dtype=torch.int32, device=device)
        self._swa_indices_gpu = torch.arange(self.swa_pages, dtype=torch.int32, device=device)
        self._swa_last_gpu = torch.empty(1, dtype=torch.int32, device=device)
        self._swa_indptr_cpu = torch.zeros(2, dtype=torch.int32)
        self._swa_indices_cpu = torch.arange(self.swa_pages, dtype=torch.int32)
        self._swa_last_cpu = torch.ones(1, dtype=torch.int32)

        self._full_indptr_gpu = torch.empty(2, dtype=torch.int32, device=device)
        self._full_indices_gpu = torch.arange(self.full_pages, dtype=torch.int32, device=device)
        self._full_last_gpu = torch.empty(1, dtype=torch.int32, device=device)
        self._full_indptr_cpu = torch.zeros(2, dtype=torch.int32)
        self._full_indices_cpu = torch.arange(self.full_pages, dtype=torch.int32)
        self._full_last_cpu = torch.ones(1, dtype=torch.int32)

        # Group size = 48/4 = 12 requires use_tensor_cores=True in FlashInfer
        self._fi_swa = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_ws_swa,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self._swa_indptr_gpu,
            paged_kv_indices_buffer=self._swa_indices_gpu,
            paged_kv_last_page_len_buffer=self._swa_last_gpu,
        )
        self._fi_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_ws_full,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self._full_indptr_gpu,
            paged_kv_indices_buffer=self._full_indices_gpu,
            paged_kv_last_page_len_buffer=self._full_last_gpu,
        )

        # Device-side decode position buffer + precomputed RoPE cos/sin table
        self._decode_pos_cpu = torch.zeros((1,), dtype=torch.int64).pin_memory()
        self._decode_pos_gpu = torch.zeros((1,), dtype=torch.int64, device=device)
        self._planned_pos: int = -1
        self._planned_swa_len: int = -1
        self._planned_full_len: int = -1

        self.rope_cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, device)
        self.pos_range = torch.arange(max_seq_len, dtype=torch.int64, device=device)

        swa_bytes = 2 * self.k_swa.numel() * self.k_swa.element_size()
        full_bytes = 2 * self.k_full.numel() * self.k_full.element_size()
        total_bytes = swa_bytes + full_bytes
        mib = 1024.0 * 1024.0
        self.stats = KVCacheStats(
            max_seq_len=max_seq_len,
            num_swa_layers=swa_count,
            num_full_layers=full_count,
            swa_kv_mib=swa_bytes / mib,
            full_kv_mib=full_bytes / mib,
            total_kv_mib=total_bytes / mib,
            total_kv_gib=total_bytes / (1024.0 ** 3),
        )

    def reset(self) -> None:
        self.seq_len = 0
        self._planned_pos = -1

    def advance(self, n_tokens: int) -> None:
        assert self.seq_len + n_tokens <= self.max_seq_len, (
            f"Sequence length {self.seq_len + n_tokens} exceeds max_seq_len={self.max_seq_len}"
        )
        self.seq_len += n_tokens

    def prepare_decode_step(self, pos: Optional[int] = None) -> None:
        """Update device decode position and FlashInfer plans once before a 50-layer n=1 step."""
        p = self.seq_len if pos is None else int(pos)
        assert p + 1 <= self.max_seq_len, f"KV cache overflow: {p + 1} > {self.max_seq_len}"

        self._decode_pos_cpu[0] = p
        self._decode_pos_gpu.copy_(self._decode_pos_cpu, non_blocking=True)

        valid_swa = min(p + 1, SLIDING_WINDOW)
        if valid_swa != self._planned_swa_len:
            np_swa = (valid_swa + self.page_size - 1) // self.page_size
            rem_swa = valid_swa - (np_swa - 1) * self.page_size
            self._swa_indptr_cpu[1] = np_swa
            self._swa_last_cpu[0] = rem_swa
            self._fi_swa.plan(
                self._swa_indptr_cpu,
                self._swa_indices_cpu[:np_swa],
                self._swa_last_cpu,
                NQ,
                NKV,
                HD,
                self.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            self._planned_swa_len = valid_swa

        valid_full = p + 1
        if valid_full != self._planned_full_len:
            np_full = (valid_full + self.page_size - 1) // self.page_size
            rem_full = valid_full - (np_full - 1) * self.page_size
            self._full_indptr_cpu[1] = np_full
            self._full_last_cpu[0] = rem_full
            self._fi_full.plan(
                self._full_indptr_cpu,
                self._full_indices_cpu[:np_full],
                self._full_last_cpu,
                NQ,
                NKV,
                HD,
                self.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            self._planned_full_len = valid_full

        self._planned_pos = p

    def attention_decode_step(
        self, x: torch.Tensor, L: Dict[str, torch.Tensor], layer_idx: int
    ) -> torch.Tensor:
        """Static-shape n=1 decode attention suitable for CUDA Graph capture."""
        qkv = x @ L["w_qkv"]
        q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
        q = F.rms_norm(q.view(1, NQ, HD), (HD,), L["q_norm"], RMS_EPS)
        k = F.rms_norm(k.view(1, NKV, HD), (HD,), L["k_norm"], RMS_EPS)
        v_nhd = v.view(1, NKV, HD)

        is_full = self.layer_is_full[layer_idx]
        sub_idx = self.layer_sub_idx[layer_idx]

        if is_full:
            kv_append_decode_cuda(
                k, v_nhd, self.k_full[sub_idx], self.v_full[sub_idx], self._decode_pos_gpu, 0
            )
            o = self._fi_full.run(q, self._full_kv_paged[sub_idx])
        else:
            flashinfer_apply_rope_inplace(
                self._decode_pos_gpu,
                q.view(1, NQ * HD),
                k.view(1, NKV * HD),
                self.rope_cos_sin_cache,
            )
            kv_append_decode_cuda(
                k,
                v_nhd,
                self.k_swa[sub_idx],
                self.v_swa[sub_idx],
                self._decode_pos_gpu,
                SLIDING_WINDOW,
            )
            o = self._fi_swa.run(q, self._swa_kv_paged[sub_idx])

        return o.view(1, NQ * HD) @ L["w_o"]

    def _gather_swa_tail(self, s_idx: int, pos: int, tail_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the last `tail_len` (`<= 512`) chronological SWA keys/values `[tail_len, NKV, HD]` before `pos`."""
        if tail_len == 0:
            empty = torch.empty((0, NKV, HD), dtype=self.dtype, device=self.device)
            return empty, empty
        if pos <= SLIDING_WINDOW:
            start = pos - tail_len
            return self.k_swa[s_idx, start:pos, :, :], self.v_swa[s_idx, start:pos, :, :]
        head = pos % SLIDING_WINDOW
        k_chrono = torch.cat(
            [self.k_swa[s_idx, head:SLIDING_WINDOW, :, :], self.k_swa[s_idx, :head, :, :]], dim=0
        )
        v_chrono = torch.cat(
            [self.v_swa[s_idx, head:SLIDING_WINDOW, :, :], self.v_swa[s_idx, :head, :, :]], dim=0
        )
        return k_chrono[-tail_len:, :, :], v_chrono[-tail_len:, :, :]

    def _write_swa(self, s_idx: int, pos: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        n = k_new.shape[0]
        if n >= SLIDING_WINDOW:
            # Keep only the most recent SLIDING_WINDOW tokens anchored at (pos + n) % SLIDING_WINDOW
            k_tail = k_new[-SLIDING_WINDOW:, :, :]
            v_tail = v_new[-SLIDING_WINDOW:, :, :]
            start_pos = pos + n - SLIDING_WINDOW
        else:
            k_tail, v_tail, start_pos = k_new, v_new, pos

        m = k_tail.shape[0]
        slot_start = start_pos % SLIDING_WINDOW
        first = min(m, SLIDING_WINDOW - slot_start)
        self.k_swa[s_idx, slot_start : slot_start + first, :, :].copy_(k_tail[:first, :, :])
        self.v_swa[s_idx, slot_start : slot_start + first, :, :].copy_(v_tail[:first, :, :])
        rem = m - first
        if rem > 0:
            self.k_swa[s_idx, :rem, :, :].copy_(k_tail[first:, :, :])
            self.v_swa[s_idx, :rem, :, :].copy_(v_tail[first:, :, :])

    def attention_forward(
        self,
        x: torch.Tensor,
        L: Dict[str, torch.Tensor],
        layer_idx: int,
        inv_freq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention for layer `layer_idx` at current `self.seq_len` (does NOT advance `seq_len`)."""
        del inv_freq  # kept for signature compatibility; uses self.rope_cos_sin_cache
        n = x.shape[0]
        pos = self.seq_len
        assert pos + n <= self.max_seq_len, f"KV cache overflow: {pos + n} > {self.max_seq_len}"

        if n == 1:
            if self._planned_pos != pos:
                self.prepare_decode_step(pos)
            return self.attention_decode_step(x, L, layer_idx)

        qkv = x @ L["w_qkv"]
        q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
        q = F.rms_norm(q.view(n, NQ, HD), (HD,), L["q_norm"], RMS_EPS)
        k = F.rms_norm(k.view(n, NKV, HD), (HD,), L["k_norm"], RMS_EPS)
        v_nhd = v.view(n, NKV, HD)

        is_full = self.layer_is_full[layer_idx]
        sub_idx = self.layer_sub_idx[layer_idx]

        if not is_full:
            flashinfer_apply_rope_inplace(
                self.pos_range[pos : pos + n],
                q.view(n, NQ * HD),
                k.view(n, NKV * HD),
                self.rope_cos_sin_cache,
            )

        q_h = q.transpose(0, 1)

        if is_full:
            # Full Attention (RNoPE: no RoPE)
            self.k_full[sub_idx, pos : pos + n, :, :].copy_(k)
            self.v_full[sub_idx, pos : pos + n, :, :].copy_(v_nhd)
            k_ctx = self.k_full[sub_idx, : pos + n, :, :].transpose(0, 1)
            v_ctx = self.v_full[sub_idx, : pos + n, :, :].transpose(0, 1)

            if pos == 0:
                o = F.scaled_dot_product_attention(q_h, k_ctx, v_ctx, is_causal=True, enable_gqa=True)
            else:
                q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                k_pos = self.pos_range[: pos + n].unsqueeze(0)
                mask = q_pos >= k_pos
                o = F.scaled_dot_product_attention(q_h, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
        else:
            # SWA Layer (RoPE + 513-token ring buffer)
            if pos + n <= SLIDING_WINDOW:
                self._write_swa(sub_idx, pos, k, v_nhd)
                k_ctx = self.k_swa[sub_idx, : pos + n, :, :].transpose(0, 1)
                v_ctx = self.v_swa[sub_idx, : pos + n, :, :].transpose(0, 1)
                if pos == 0:
                    o = F.scaled_dot_product_attention(q_h, k_ctx, v_ctx, is_causal=True, enable_gqa=True)
                else:
                    q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                    k_pos = self.pos_range[: pos + n].unsqueeze(0)
                    mask = q_pos >= k_pos
                    o = F.scaled_dot_product_attention(q_h, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
            else:
                # Chunked prefill crossing or past SLIDING_WINDOW: gather prior 512 tokens before overwrite
                tail_len = min(pos, SLIDING_WINDOW - 1)
                k_tail, v_tail = self._gather_swa_tail(sub_idx, pos, tail_len)
                k_ctx = torch.cat([k_tail, k], dim=0).transpose(0, 1)
                v_ctx = torch.cat([v_tail, v_nhd], dim=0).transpose(0, 1)
                q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                k_pos = self.pos_range[pos - tail_len : pos + n].unsqueeze(0)
                diff = q_pos - k_pos
                mask = (diff >= 0) & (diff < SLIDING_WINDOW)
                o = F.scaled_dot_product_attention(q_h, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
                self._write_swa(sub_idx, pos, k, v_nhd)

        return o.transpose(0, 1).reshape(n, NQ * HD) @ L["w_o"]
