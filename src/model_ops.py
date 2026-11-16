"""Kolibri-1 decoder, attention, router, shared expert, and MoE execution ops."""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401 - registers torch.ops.trtllm
from tensorrt_llm._torch.moe.fused_moe.marlin.base import sum_topk_expert_outputs

from .config import (
    E,
    H,
    HD,
    I_MOE,
    MOE_BLOCK_SIZE,
    NKV,
    NQ,
    RMS_EPS,
    ROPE_THETA,
    SLIDING_WINDOW,
    TOPK,
    is_full_attention_layer,
)
from .kolibri_kernels import (
    flashinfer_apply_rope_inplace,
    flashinfer_fused_add_rmsnorm,
    flashinfer_silu_and_mul,
    fused_add_post_moe_norm,
    fused_kolibri_router,
    fused_sandwich_norm,
    make_rope_cos_sin_cache,
)

_ROPE_CACHE_BY_DEV: Dict[int, torch.Tensor] = {}


def _get_shared_rope_cache(dev: torch.device, min_len: int) -> torch.Tensor:
    key = dev.index or 0
    cached = _ROPE_CACHE_BY_DEV.get(key)
    if cached is None or cached.shape[0] < min_len:
        alloc = max(min_len, 4096)
        cached = make_rope_cos_sin_cache(alloc, dev)
        _ROPE_CACHE_BY_DEV[key] = cached
    return cached


def make_rope_inv_freq(dev: torch.device) -> torch.Tensor:
    return 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))


def apply_rope_neox(x: torch.Tensor, inv_freq: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
    """Apply GPT-NeoX style RoPE to [num_heads, seq_len, head_dim] tensor starting at `pos_offset`."""
    seq_len, hd = x.shape[1], x.shape[2]
    pos = torch.arange(pos_offset, pos_offset + seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(x.dtype).unsqueeze(0)
    sin = freqs.sin().to(x.dtype).unsqueeze(0)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def kolibri_attention(
    x: torch.Tensor,
    L: Dict[str, torch.Tensor],
    inv_freq: Optional[torch.Tensor] = None,
    is_full_attention: bool = False,
    cos_sin_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kolibri-1 GQA attention with QK RMSNorm; SWA layers apply RoPE+window, full layers use RNoPE."""
    del inv_freq  # kept for signature compatibility; uses FlashInfer cos/sin cache
    n = x.shape[0]
    qkv = x @ L["w_qkv"]
    q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
    q = F.rms_norm(q.view(n, NQ, HD), (HD,), L["q_norm"], RMS_EPS)
    k = F.rms_norm(k.view(n, NKV, HD), (HD,), L["k_norm"], RMS_EPS)

    if not is_full_attention:
        cache = cos_sin_cache if cos_sin_cache is not None else _get_shared_rope_cache(x.device, n)
        pos_ids = torch.arange(n, device=x.device, dtype=torch.int64)
        flashinfer_apply_rope_inplace(pos_ids, q.view(n, NQ * HD), k.view(n, NKV * HD), cache)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.view(n, NKV, HD).transpose(0, 1)

    if is_full_attention or n <= SLIDING_WINDOW:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    else:
        idx = torch.arange(n, device=x.device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        mask = (diff >= 0) & (diff < SLIDING_WINDOW)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    return o.transpose(0, 1).reshape(n, NQ * HD) @ L["w_o"]


def kolibri_route(moe_in: torch.Tensor, L: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid Top-6 routing using Kolibri's fused CUDA router kernel (`norm_topk_prob=False`)."""
    return fused_kolibri_router(moe_in @ L["w_gate"], L["gate_bias"], k=TOPK)


def exact_bf16_routed_moe(
    moe_in: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_nk: torch.Tensor,
    w2_nk: torch.Tensor,
) -> torch.Tensor:
    """Exact per-token BF16 reference (loops over active experts for mathematical ground truth)."""
    n = moe_in.shape[0]
    flat_ids = topk_ids.reshape(-1)
    flat_tok = torch.arange(n, device=moe_in.device).repeat_interleave(TOPK)
    flat_w = topk_weights.reshape(-1, 1)

    out_acc = torch.zeros((n, H), dtype=torch.float32, device=moe_in.device)
    for eid in torch.unique(flat_ids).tolist():
        mask = flat_ids == eid
        tok_idx = flat_tok[mask]
        x_e = moe_in[tok_idx]
        g1 = F.linear(x_e, w13_nk[eid])
        act = F.silu(g1[:, :I_MOE]) * g1[:, I_MOE:]
        g2 = F.linear(act, w2_nk[eid]).float() * flat_w[mask]
        out_acc.index_add_(0, tok_idx, g2)
    return out_acc.to(torch.bfloat16)


def marlin_nvfp4_routed_moe(
    moe_in: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    L: Dict[str, torch.Tensor],
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Execute W4A16 NVFP4 Marlin MoE (`marlin_nvfp4_moe_gemm` GEMM1 -> SwiGLU -> GEMM2 -> reduce)."""
    n = moe_in.shape[0]
    dev = moe_in.device

    max_pad1 = n * TOPK + E * MOE_BLOCK_SIZE
    s_ids1 = torch.empty(max_pad1, dtype=torch.int32, device=dev)
    e_ids1 = torch.empty((max_pad1 + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE, dtype=torch.int32, device=dev)
    n_pad1 = torch.empty(1, dtype=torch.int32, device=dev)
    torch.ops.trtllm.moe_align_block_size(topk_ids, E, MOE_BLOCK_SIZE, s_ids1, e_ids1, n_pad1)

    gemm1_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
        moe_in.contiguous(),
        L["w13_marlin"],
        b_scales=L["w13_scale_marlin"],
        global_scale=L["w13_gs_marlin"],
        workspace=workspace,
        sorted_token_ids=s_ids1,
        expert_ids=e_ids1,
        num_tokens_past_padded=n_pad1,
        topk_weights=topk_weights,
        moe_block_size=MOE_BLOCK_SIZE,
        top_k=TOPK,
        mul_topk_weights=False,
        size_n=2 * I_MOE,
        size_k=H,
        out_dtype=torch.bfloat16,
        use_fp32_reduce=False,
    )

    hidden = flashinfer_silu_and_mul(gemm1_out)

    n_g2 = n * TOPK
    tk_ids2 = topk_ids.reshape(-1, 1).contiguous()
    tk_w2 = topk_weights.reshape(-1, 1).contiguous()
    max_pad2 = n_g2 + E * MOE_BLOCK_SIZE
    s_ids2 = torch.empty(max_pad2, dtype=torch.int32, device=dev)
    e_ids2 = torch.empty((max_pad2 + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE, dtype=torch.int32, device=dev)
    n_pad2 = torch.empty(1, dtype=torch.int32, device=dev)
    torch.ops.trtllm.moe_align_block_size(tk_ids2, E, MOE_BLOCK_SIZE, s_ids2, e_ids2, n_pad2)

    gemm2_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
        hidden,
        L["w2_marlin"],
        b_scales=L["w2_scale_marlin"],
        global_scale=L["w2_gs_marlin"],
        workspace=workspace,
        sorted_token_ids=s_ids2,
        expert_ids=e_ids2,
        num_tokens_past_padded=n_pad2,
        topk_weights=tk_w2,
        moe_block_size=MOE_BLOCK_SIZE,
        top_k=1,
        mul_topk_weights=True,
        size_n=H,
        size_k=I_MOE,
        out_dtype=torch.bfloat16,
        use_fp32_reduce=False,
    )
    return sum_topk_expert_outputs(gemm2_out, n, TOPK, H, torch.bfloat16)


def shared_expert(moe_in: torch.Tensor, L: Dict[str, torch.Tensor]) -> torch.Tensor:
    return flashinfer_silu_and_mul(moe_in @ L["sh_w13"]) @ L["sh_w2"]


def decoder_block_forward(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    L: Dict[str, torch.Tensor],
    inv_freq: Optional[torch.Tensor],
    routed_fn: Callable,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual is None:
        residual = x
        h = F.rms_norm(residual, (H,), L["input_ln"], RMS_EPS)
    else:
        h = x
        flashinfer_fused_add_rmsnorm(h, residual, L["input_ln"], RMS_EPS)

    attn_out = kolibri_attention(h, L, inv_freq, is_full_attention=is_full_attention_layer(layer_idx))
    residual, moe_in = fused_sandwich_norm(attn_out, residual, L["post_attn_ln"], L["pre_moe_ln"], RMS_EPS)

    topk_ids, topk_weights = kolibri_route(moe_in, L)
    routed = routed_fn(moe_in, topk_ids, topk_weights, L)
    shared = shared_expert(moe_in, L)
    ffn_out = fused_add_post_moe_norm(routed, shared, L["post_ffn_ln"], RMS_EPS)
    return ffn_out, residual, topk_ids, routed


def run_layers_forward(
    x: torch.Tensor,
    layers: List[Dict[str, torch.Tensor]],
    inv_freq: torch.Tensor,
    routed_fn: Callable,
    start_layer_idx: int = 0,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    residual = None
    cur = x
    all_ids, all_routed = [], []
    for offset, L in enumerate(layers):
        cur, residual, ids, routed = decoder_block_forward(
            cur, residual, L, inv_freq, routed_fn, layer_idx=start_layer_idx + offset
        )
        all_ids.append(ids)
        all_routed.append(routed)
    return cur + residual, all_ids, all_routed
