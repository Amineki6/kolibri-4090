"""Kolibri-1 specialized CUDA kernels and TensorRT-LLM / FlashInfer fast ops.

Bundles standalone compilation of Kolibri's fused sandwich RMSNorm and fused
post-MoE add+RMSNorm kernels from `serving/csrc/`, alongside TensorRT-LLM's
compiled `kolibri_router_op` and FlashInfer's RMSNorm / RoPE / SiLU+Mul ops.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure conda env bin dir (`ninja`, etc.) is on PATH before JIT/extension imports.
_ENV_BIN = str(Path(sys.executable).resolve().parent)
if _ENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import flashinfer  # noqa: E402
import tensorrt_llm  # noqa: E402, F401 - registers torch.ops.trtllm custom ops
import torch  # noqa: E402
from tensorrt_llm._torch.moe.custom_ops.kolibri_router_custom_op import (  # noqa: E402
    fused_kolibri_router as _trtllm_fused_kolibri_router,
    is_kolibri_router_available,
)
from torch.utils.cpp_extension import load as _load_cpp_ext  # noqa: E402

from .config import HD, ROPE_THETA, TOPK

_CSRC_DIR = Path(__file__).resolve().parent.parent / "csrc"
_BUILD_DIR = _CSRC_DIR / "build"
_EXT_MODULE = None


def _get_ext():
    """Compile (once) and return the standalone Kolibri norm CUDA extension."""
    global _EXT_MODULE
    if _EXT_MODULE is not None:
        return _EXT_MODULE

    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _EXT_MODULE = _load_cpp_ext(
        name="kolibri_kernels_ext",
        sources=[
            str(_CSRC_DIR / "kolibri_kernels_ext.cpp"),
            str(_CSRC_DIR / "kolibriSandwichNormKernel.cu"),
            str(_CSRC_DIR / "kolibriPostMoeNormKernel.cu"),
        ],
        extra_include_paths=[str(_CSRC_DIR / "include")],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-DENABLE_BF16", "-DNDEBUG"],
        build_directory=str(_BUILD_DIR),
        verbose=False,
    )
    return _EXT_MODULE


def fused_sandwich_norm(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    gamma1: torch.Tensor,
    gamma2: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute `(residual_in + rmsnorm(attn_out, gamma1), rmsnorm(residual_out, gamma2))` in 1 kernel."""
    return _get_ext().fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps)


def fused_add_post_moe_norm(
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute `rmsnorm(routed_out + shared_out, gamma)` in 1 kernel."""
    return _get_ext().fused_add_post_moe_norm(routed_out, shared_out, gamma, eps)


assert is_kolibri_router_available(), "TensorRT-LLM kolibri_router_op is required"


def fused_kolibri_router(
    router_logits: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    k: int = TOPK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Kolibri's fused CUDA sigmoid Top-K router (`norm_topk_prob=False`, unnormalized weights)."""
    return _trtllm_fused_kolibri_router(
        router_logits,
        bias=gate_bias,
        k=k,
        norm_topk_prob=False,
        output_dtype=torch.float32,
    )


def flashinfer_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FlashInfer RMSNorm (direct call, supports 2D `[N, D]` CUDA tensors)."""
    return flashinfer.rmsnorm(x, weight, eps)


def flashinfer_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """In-place FlashInfer fused residual add + RMSNorm (`residual += x; x = rmsnorm(residual)`)."""
    flashinfer.fused_add_rmsnorm(x, residual, weight, eps)


def make_rope_cos_sin_cache(
    max_seq_len: int,
    device: torch.device,
    head_dim: int = HD,
    theta: float = ROPE_THETA,
) -> torch.Tensor:
    """Precompute FlashInfer NeoX RoPE `[max_seq_len, head_dim]` float32 `[cos, sin]` cache."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def flashinfer_apply_rope_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int = HD,
    is_neox: bool = True,
) -> None:
    """In-place FlashInfer NeoX RoPE using precomputed `[max_seq, HD]` float32 `cos_sin_cache`."""
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        positions, query, key, head_size, cos_sin_cache, is_neox=is_neox
    )


def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused FlashInfer SwiGLU (`silu(x[..., :d]) * x[..., d:]`)."""
    return flashinfer.silu_and_mul(x)


def warmup_kernels(device: torch.device = torch.device("cuda:0")) -> None:
    """Eagerly compile/load all CUDA extensions and FlashInfer JIT modules before timing."""
    _get_ext()
    d = torch.empty((1, 2560), dtype=torch.bfloat16, device=device)
    g = torch.ones((2560,), dtype=torch.bfloat16, device=device)
    fused_sandwich_norm(d, d, g, g, 1e-6)
    fused_add_post_moe_norm(d, d, g, 1e-6)
    r = torch.empty((1, 384), dtype=torch.bfloat16, device=device)
    b = torch.zeros((384,), dtype=torch.float32, device=device)
    fused_kolibri_router(r, b)
    flashinfer_rmsnorm(d, g, 1e-6)
    flashinfer_fused_add_rmsnorm(d.clone(), d.clone(), g, 1e-6)
    flashinfer_silu_and_mul(torch.empty((1, 1024), dtype=torch.bfloat16, device=device))
    q = torch.empty((1, 48 * HD), dtype=torch.bfloat16, device=device)
    k = torch.empty((1, 4 * HD), dtype=torch.bfloat16, device=device)
    pos = torch.zeros((1,), dtype=torch.int64, device=device)
    cs = make_rope_cos_sin_cache(8, device)
    flashinfer_apply_rope_inplace(pos, q, k, cs)
    torch.cuda.synchronize(device)
