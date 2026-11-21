"""Kolibri-1 specialized CUDA kernels and TensorRT-LLM / FlashInfer fast ops.

Bundles standalone compilation of Kolibri's fused sandwich RMSNorm and fused
post-MoE add+RMSNorm kernels from `serving/csrc/`, alongside TensorRT-LLM's
compiled `kolibri_router_op` and FlashInfer's RMSNorm / RoPE / SiLU+Mul ops.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Ensure conda env bin dir (`ninja`, etc.) is on PATH before JIT/extension imports.
_ENV_BIN = str(Path(sys.executable).resolve().parent)
if _ENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import flashinfer  # noqa: E402
import tensorrt_llm  # noqa: E402, F401 - registers torch.ops.trtllm custom ops
import torch  # noqa: E402
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
            str(_CSRC_DIR / "kolibriCacheOps.cu"),
            str(_CSRC_DIR / "kolibriGatherExperts.cpp"),
            str(_CSRC_DIR / "kolibriSandwichNormKernel.cu"),
            str(_CSRC_DIR / "kolibriPostMoeNormKernel.cu"),
            str(_CSRC_DIR / "kolibriRouterKernel.cu"),
        ],
        extra_include_paths=[str(_CSRC_DIR), str(_CSRC_DIR / "include")],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-fopenmp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-DENABLE_BF16", "-DNDEBUG"],
        extra_ldflags=["-fopenmp"],
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


def fused_kolibri_router(
    router_logits: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    k: int = TOPK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Kolibri's standalone CUDA sigmoid Top-6 router (`norm_topk_prob=False`)."""
    assert k == TOPK, f"Standalone kolibri_router expects k={TOPK}, got {k}"
    bias = gate_bias.float() if gate_bias is not None and gate_bias.dtype != torch.float32 else gate_bias
    return _get_ext().fused_kolibri_router(router_logits, bias)


def detected_pcore_count() -> int:
    """Return the detected Linux performance (P) core count."""
    return int(_get_ext().detected_pcore_count())


def gather_experts_omp(
    src_tensors: Sequence[torch.Tensor],
    dst_tensors: Sequence[torch.Tensor],
    expert_ids: torch.Tensor,
    num_threads: int = 0,
) -> None:
    """Gather expert rows across all slabs into pinned host memory using P-core pinned OpenMP."""
    _get_ext().gather_experts_omp(list(src_tensors), list(dst_tensors), expert_ids, int(num_threads))


def dma_slabs_async(
    src_slabs: Sequence[torch.Tensor],
    dst_slabs: Sequence[torch.Tensor],
    count: int,
    dst_offset: int,
    stream_ptr: int,
    event_ptr: int,
) -> None:
    """Issue async H2D cudaMemcpyAsync across all slabs on `stream_ptr` and record `event_ptr`."""
    _get_ext().dma_slabs_async(
        list(src_slabs), list(dst_slabs), int(count), int(dst_offset), int(stream_ptr), int(event_ptr)
    )


def promote_experts_cuda(
    cache_slabs: Sequence[torch.Tensor],
    gpu_expert_to_slot: torch.Tensor,
    src_slots: Sequence[int],
    dst_slots: Sequence[int],
    old_eids: Sequence[int],
    new_eids: Sequence[int],
) -> None:
    """Copy promoted experts and update `gpu_expert_to_slot` in one fused CUDA kernel."""
    _get_ext().promote_experts_cuda(
        list(cache_slabs),
        gpu_expert_to_slot,
        list(src_slots),
        list(dst_slots),
        list(old_eids),
        list(new_eids),
    )


def kv_append_decode_cuda(
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos_buf: torch.Tensor,
    ring_mod: int = 0,
) -> None:
    """Append single-token K/V into NHD cache using a device-side position tensor."""
    _get_ext().kv_append_decode_cuda(k_in, v_in, k_cache, v_cache, pos_buf, int(ring_mod))


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


def configure_cuda_mempool(device: torch.device = torch.device("cuda:0")) -> int:
    """Keep stream-ordered CUDA mempool blocks cached across syncs (`ReleaseThreshold=UINT64_MAX`)."""
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    return int(_get_ext().configure_cuda_mempool(int(dev_idx)))


def warmup_kernels(device: torch.device = torch.device("cuda:0")) -> None:
    """Eagerly compile/load all CUDA extensions and FlashInfer JIT modules before timing."""
    configure_cuda_mempool(device)
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
