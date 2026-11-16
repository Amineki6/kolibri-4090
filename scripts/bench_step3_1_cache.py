#!/usr/bin/env python3
"""Step 3.1 Benchmark & Verification: Full 50-layer Hybrid Cache Load + PCIe 4.0 x16 DMA."""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import PHASE2_MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.expert_cache import HybridExpertCache, MARLIN_KEYS  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "step3_1_cache_bench.json"
BYTES_PER_EXPERT = 2211844  # Verified: 2.1094 MiB per Marlin NVFP4 routed expert


def bench_cuda_fn(fn, warmup: int = 5, iters: int = 25) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=np.float64)
    return {
        "median_ms": float(np.median(ms)),
        "mean_ms": float(np.mean(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(np.min(ms)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    print(f"=== Step 3.1: Kolibri-1 50-Layer Hybrid Cache & PCIe DMA Verification ===")
    print(f"GPU: {torch.cuda.get_device_name(dev)} | Model dir: {PHASE2_MODEL_DIR}")

    cache = HybridExpertCache(
        model_dir=PHASE2_MODEL_DIR,
        device=dev,
        persistent_slots_per_layer=110,
        decode_miss_slots=6,
        bounce_slots=274,
        stage_slots=384,
    )
    st = cache.stats
    free_b, total_b = torch.cuda.mem_get_info(dev)

    print("\n[1] 50-Layer Memory Breakdown (Live Allocated)")
    print(f"  Load time (50 layers + globals -> GPU + Host RAM): {st.load_time_s:.2f} s")
    print(f"  GPU globals (embed + norm + lm_head BF16):        {st.globals_mib:8.2f} MiB")
    print(f"  GPU 50x non-experts (BF16 + FP32 bias):           {st.non_experts_mib:8.2f} MiB")
    print(f"  GPU persistent expert cache (50 x 116 slots):     {st.gpu_cache_mib:8.2f} MiB ({st.gpu_cache_mib / 1024:.2f} GiB)")
    print(f"  GPU double-buffered stage (2 x 384 slots):        {st.gpu_stage_mib:8.2f} MiB ({st.gpu_stage_mib / 1024:.2f} GiB)")
    print(f"  Total PyTorch CUDA allocated / reserved:          {st.total_gpu_allocated_mib / 1024:.2f} GiB / {st.total_gpu_reserved_mib / 1024:.2f} GiB")
    print(f"  CUDA driver free / total VRAM:                    {free_b / (1024**3):.2f} GiB / {total_b / (1024**3):.2f} GiB")
    print(f"  Host pageable routed experts store (50 x 384):    {st.host_routed_gib:8.2f} GiB")
    print(f"  Host pinned DMA bounce arena (2 x 274):           {st.host_pinned_bounce_gib:8.2f} GiB")

    # 2. Bit-exact DMA verification across cold misses on Layers 0, 24, 49
    print("\n[2] Bit-Exact DMA Verification (Host Store -> Pinned Bounce -> GPU Stage)")
    for li, m_count in ((0, 1), (0, 6), (24, 64), (49, 256)):
        gen = torch.Generator().manual_seed(100 + li + m_count)
        # Pick expert IDs from cold range [118..383] (and wrap within 0..383)
        eids = torch.randperm(384, generator=gen)[:m_count]
        buf_idx = li & 1
        m_copied, stage_slab = cache.stage_experts_async(li, eids, buf_idx=buf_idx)
        cache.wait_copy(buf_idx)
        assert m_copied == m_count
        host_layer = cache.host_routed[li]
        for key in MARLIN_KEYS:
            expected = host_layer[key][eids.long()]
            got = stage_slab[key][:m_count].cpu()
            assert torch.equal(expected, got), f"DMA mismatch on layer {li} key {key} (m={m_count})"
        print(f"  Layer {li:02d} (m={m_count:3d} experts, {m_count * BYTES_PER_EXPERT / (1024**2):6.2f} MiB): BIT-EXACT PASS")

    # Also test decode scratch copy + LRU promotion on Layer 0
    miss_eids = torch.tensor([200, 201, 350], dtype=torch.int64)
    m_dec = cache.decode_stage_misses_async(0, miss_eids, buf_idx=0)
    cache.wait_copy(0)
    assert m_dec == 3
    for idx, eid in enumerate(miss_eids.tolist()):
        slot = cache.persistent_slots + idx
        for key in MARLIN_KEYS:
            assert torch.equal(cache.host_routed[0][key][eid], cache.gpu_caches[0][key][slot].cpu())
    promotions = cache.promote_decode_misses(0, miss_eids.tolist(), step_id=1)
    for eid, slot in promotions:
        assert cache.cache_states[0].expert_to_slot[eid] == slot
        for key in MARLIN_KEYS:
            assert torch.equal(cache.host_routed[0][key][eid], cache.gpu_caches[0][key][slot].cpu())
    print("  Decode miss staging + LRU GPU promotion: BIT-EXACT PASS")

    # 3. PCIe 4.0 x16 H2D transfer latency benchmark
    print("\n[3] PCIe 4.0 x16 H2D Transfer Benchmark")
    print("  Count | Payload (MiB) | Pure Pinned DMA (ms | GB/s) | CPU Gather + DMA (ms | eff. GB/s)")
    dma_results = {}
    for m_count in (1, 6, 64, 256):
        eids = torch.randperm(384)[:m_count]
        payload_bytes = m_count * BYTES_PER_EXPERT
        payload_mib = payload_bytes / (1024.0 * 1024.0)

        # Pre-populate pinned buffer once for pure PCIe DMA measurement
        cache.gather_to_pinned(12, eids, buf_idx=0)

        def run_pure_dma(m=m_count):
            cache.dma_pinned_to_gpu_slab(m, 0, cache.gpu_stage[0], dst_offset=0)
            cache.wait_copy(0)

        def run_e2e_miss(eids_t=eids):
            cache.stage_experts_async(12, eids_t, buf_idx=1)
            cache.wait_copy(1)

        pure_t = bench_cuda_fn(run_pure_dma, warmup=5, iters=30)
        e2e_t = bench_cuda_fn(run_e2e_miss, warmup=5, iters=30)

        pure_gbps = (payload_bytes / 1e9) / (pure_t["median_ms"] / 1e3)
        e2e_gbps = (payload_bytes / 1e9) / (e2e_t["median_ms"] / 1e3)

        print(
            f"  {m_count:5d} |    {payload_mib:7.2f}    |    {pure_t['median_ms']:6.3f} ms | {pure_gbps:5.2f}    |       "
            f"{e2e_t['median_ms']:6.3f} ms | {e2e_gbps:5.2f}"
        )
        dma_results[str(m_count)] = {
            "payload_mib": payload_mib,
            "pure_dma": {**pure_t, "gbps": pure_gbps},
            "gather_plus_dma": {**e2e_t, "gbps": e2e_gbps},
        }

    report = {
        "gpu": torch.cuda.get_device_name(dev),
        "driver_free_gib": free_b / (1024.0 ** 3),
        "driver_total_gib": total_b / (1024.0 ** 3),
        "memory_stats": {
            "load_time_s": st.load_time_s,
            "globals_mib": st.globals_mib,
            "non_experts_mib": st.non_experts_mib,
            "gpu_cache_mib": st.gpu_cache_mib,
            "gpu_stage_mib": st.gpu_stage_mib,
            "total_gpu_allocated_mib": st.total_gpu_allocated_mib,
            "total_gpu_reserved_mib": st.total_gpu_reserved_mib,
            "host_routed_gib": st.host_routed_gib,
            "host_pinned_bounce_gib": st.host_pinned_bounce_gib,
        },
        "pcie_h2d_bench": dma_results,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved Step 3.1 results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
