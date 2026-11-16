"""W4A16 NVFP4 quantization, evaluation, and Marlin repacking on SM89+."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401 - registers torch.ops.trtllm
from tensorrt_llm.quantization.utils.marlin_utils import prepare_nvfp4_moe_weights_for_marlin

from .config import E, H, I_MOE

E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def metrics(ref: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    """Compute cosine similarity, relative L2 error, max abs diff, and std ratio."""
    r = ref.float().flatten()
    p = pred.float().flatten()
    cos = F.cosine_similarity(r, p, dim=0).item()
    r_norm = r.norm().item()
    diff_norm = (r - p).norm().item()
    rel_l2 = diff_norm / max(r_norm, 1e-12)
    max_abs = (r - p).abs().max().item()
    std_ratio = p.std().item() / max(r.std().item(), 1e-12) if r.numel() > 1 else 1.0
    return {
        "cosine": cos,
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "std_ratio": std_ratio,
    }


def quantize_and_eval_nvfp4_chunked(
    w_cpu_hk: torch.Tensor,
    dev: torch.device,
    chunk_size: int = 64,
    compute_stats: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Quantize [E, K_in, N_out] BF16 weights to GPU NVFP4 in bounded-VRAM chunks.

    Transposes each chunk to [chunk, N_out, K_in], applies exact E2M1 round-to-nearest-even
    quantization with 16-element FP8 E4M3 block scales and per-expert FP32 scale_2, and
    optionally accumulates exact dequantization fidelity metrics without materializing
    full-layer FP32 buffers in VRAM.

    Note: TRT-LLM's CUDA `fp4_quantize` op relies on SM100+ PTX intrinsics that yield
    invalid scale bytes on SM89 (RTX 4090), so this vectorized CUDA implementation runs
    natively on SM89.
    """
    num_e, k_in, n_out = w_cpu_hk.shape
    assert k_in % 16 == 0, f"K_in={k_in} must be divisible by 16"

    packed_all = torch.empty((num_e, n_out, k_in // 2), dtype=torch.uint8, device=dev)
    s1_all = torch.empty((num_e, n_out, k_in // 16), dtype=torch.float8_e4m3fn, device=dev)
    s2_all = torch.empty((num_e,), dtype=torch.float32, device=dev)
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=dev) if compute_stats else None

    dot = 0.0
    norm_r2 = 0.0
    norm_p2 = 0.0
    diff2 = 0.0
    max_abs = 0.0

    for start in range(0, num_e, chunk_size):
        end = min(start + chunk_size, num_e)
        ce = end - start
        wf = w_cpu_hk[start:end].to(dev).transpose(1, 2).contiguous().float()

        amax_e = wf.abs().amax(dim=(1, 2)).clamp(min=1e-12)
        scale_2 = amax_e / (448.0 * 6.0)
        s2_all[start:end] = scale_2

        blocks = wf.view(ce, n_out, k_in // 16, 16)
        block_amax = blocks.abs().amax(dim=-1)
        s1_fp8 = (block_amax / (6.0 * scale_2.view(ce, 1, 1))).clamp(max=448.0).to(torch.float8_e4m3fn)
        s1_all[start:end] = s1_fp8

        s1_eff = (s1_fp8.float() * scale_2.view(ce, 1, 1)).unsqueeze(-1).clamp(min=1e-12)
        scaled = blocks / s1_eff
        sign = (scaled < 0).to(torch.uint8)
        ax = scaled.abs()

        mag = torch.zeros_like(ax, dtype=torch.uint8)
        mag = torch.where(ax > 0.25, torch.tensor(1, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax >= 0.75, torch.tensor(2, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax > 1.25, torch.tensor(3, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax >= 1.75, torch.tensor(4, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax > 2.50, torch.tensor(5, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax >= 3.50, torch.tensor(6, dtype=torch.uint8, device=dev), mag)
        mag = torch.where(ax > 5.00, torch.tensor(7, dtype=torch.uint8, device=dev), mag)

        nibbles = ((sign << 3) | mag).view(ce, n_out, k_in // 2, 2)
        packed = (nibbles[..., 0] | (nibbles[..., 1] << 4)).contiguous()
        packed_all[start:end] = packed

        if compute_stats:
            lo = lut[(packed & 0x0F).long()]
            hi = lut[((packed >> 4) & 0x0F).long()]
            deq = (torch.stack([lo, hi], dim=-1).view(ce, n_out, k_in // 16, 16) * s1_eff).view(ce, n_out, k_in)

            dot += float((wf * deq).sum(dtype=torch.float64).item())
            norm_r2 += float((wf * wf).sum(dtype=torch.float64).item())
            norm_p2 += float((deq * deq).sum(dtype=torch.float64).item())
            diff = wf - deq
            diff2 += float((diff * diff).sum(dtype=torch.float64).item())
            max_abs = max(max_abs, float(diff.abs().max().item()))

    stats: Dict[str, float] = {}
    if compute_stats:
        stats = {
            "cosine": dot / max((norm_r2 ** 0.5) * (norm_p2 ** 0.5), 1e-12),
            "rel_l2": (diff2 ** 0.5) / max(norm_r2 ** 0.5, 1e-12),
            "max_abs": max_abs,
            "std_ratio": (norm_p2 ** 0.5) / max(norm_r2 ** 0.5, 1e-12),
        }
    return packed_all, s1_all, s2_all, stats


def quantize_and_pack_marlin_layer(
    w13_hk: torch.Tensor,
    w2_hk: torch.Tensor,
    dev: torch.device,
    hidden_size: int = H,
    inter_size: int = I_MOE,
    num_experts: int = E,
    chunk_size: int = 64,
    compute_stats: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    """Quantize routed expert matrices [E, K_in, N_out] to NVFP4 and repack into Marlin tiled format."""
    w13_q, w13_sf, w13_gs, w13_stats = quantize_and_eval_nvfp4_chunked(
        w13_hk, dev, chunk_size=chunk_size, compute_stats=compute_stats
    )
    w2_q, w2_sf, w2_gs, w2_stats = quantize_and_eval_nvfp4_chunked(
        w2_hk, dev, chunk_size=chunk_size, compute_stats=compute_stats
    )

    (
        w13_m,
        w13_s_m,
        w13_gs_m,
        w2_m,
        w2_s_m,
        w2_gs_m,
    ) = prepare_nvfp4_moe_weights_for_marlin(
        w13=w13_q,
        w13_scale=w13_sf,
        w13_global_scale=w13_gs,
        w2=w2_q,
        w2_scale=w2_sf,
        w2_global_scale=w2_gs,
        hidden_size=hidden_size,
        intermediate_size_per_partition=inter_size,
        num_experts=num_experts,
        is_act_and_mul=True,
        param_dtype=torch.bfloat16,
    )
    del w13_q, w13_sf, w13_gs, w2_q, w2_sf, w2_gs

    marlin_tensors = {
        "w13_marlin": w13_m.view(torch.int64).contiguous(),
        "w13_scale_marlin": w13_s_m.view(torch.int32).contiguous(),
        "w13_gs_marlin": w13_gs_m.contiguous(),
        "w2_marlin": w2_m.view(torch.int64).contiguous(),
        "w2_scale_marlin": w2_s_m.view(torch.int32).contiguous(),
        "w2_gs_marlin": w2_gs_m.contiguous(),
    }
    return marlin_tensors, {"w13": w13_stats, "w2": w2_stats}
