#!/usr/bin/env python3
"""Step 3.3 Verification & Benchmark: Hybrid 100k KV Cache Footprint, SWA Wrap Parity & Speed."""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import H, PHASE2_MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.expert_cache import HybridExpertCache  # noqa: E402
from src.kv_cache import HybridKVCache  # noqa: E402
from src.model_ops import kolibri_attention, make_rope_inv_freq  # noqa: E402
from src.nvfp4_quant import metrics  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "step3_3_kv_cache.json"


def bench_cuda_fn(fn, warmup: int = 10, iters: int = 40) -> Dict[str, float]:
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
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    inv_freq = make_rope_inv_freq(dev)

    print("=== Step 3.3: 100k Hybrid KV Cache (40 SWA@513 + 10 Full RNoPE) Verification ===")
    cache = HybridExpertCache(
        model_dir=PHASE2_MODEL_DIR,
        device=dev,
        persistent_slots_per_layer=110,
        decode_miss_slots=6,
        bounce_slots=274,
        stage_slots=384,
    )
    kv = HybridKVCache(max_seq_len=100_000, device=dev)
    torch.cuda.synchronize(dev)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    alloc_gib = torch.cuda.memory_allocated(dev) / (1024.0 ** 3)
    res_gib = torch.cuda.memory_reserved(dev) / (1024.0 ** 3)

    print("\n[1] Co-Resident 50-Layer Experts + 100k KV Cache Memory")
    print(f"  SWA KV cache (40 layers x 513 tokens, BF16): {kv.stats.swa_kv_mib:7.2f} MiB")
    print(f"  Full RNoPE KV (10 layers x 100k tokens):     {kv.stats.full_kv_mib:7.2f} MiB")
    print(f"  Total 100k Hybrid KV Cache:                  {kv.stats.total_kv_mib:7.2f} MiB ({kv.stats.total_kv_gib:.4f} GiB)")
    print(f"  Total GPU allocated / reserved (with model): {alloc_gib:.2f} GiB / {res_gib:.2f} GiB")
    print(f"  Remaining free driver VRAM headroom:         {free_b / (1024**3):.2f} GiB / {total_b / (1024**3):.2f} GiB")

    # 2. Parity test across the SWA=513 wraparound boundary
    # Test both Layer 0 (SWA@513 + RoPE) and Layer 4 (Full RNoPE)
    print("\n[2] Incremental KV Cache vs Full-Context Reference (Crossing SWA=513 Boundary)")
    parity = {}
    with torch.inference_mode():
        total_seq = 650
        gen = torch.Generator(device=dev).manual_seed(202603)
        x_full = torch.empty((total_seq, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=gen)

        for li, label in ((0, "layer_00_swa513_rope"), (4, "layer_04_full_rnope")):
            L = cache.gpu_non_experts[li]
            ref_out = kolibri_attention(x_full, L, inv_freq, is_full_attention=kv.layer_is_full[li])

            # Mode A: Chunked prefill (256 + 256 + 128 = 640) + 10 single-token decode steps (crossing 513!)
            kv.reset()
            chunks = []
            for start, end in ((0, 256), (256, 512), (512, 640)):
                out_c = kv.attention_forward(x_full[start:end], L, li, inv_freq)
                kv.advance(end - start)
                chunks.append(out_c)
            for pos in range(640, total_seq):
                out_1 = kv.attention_forward(x_full[pos : pos + 1], L, li, inv_freq)
                kv.advance(1)
                chunks.append(out_1)

            inc_out = torch.cat(chunks, dim=0)
            m_all = metrics(ref_out, inc_out)
            m_tail = metrics(ref_out[510:], inc_out[510:])
            print(
                f"  {label:22s}: full[0:650] cos={m_all['cosine']:.6f} rel_l2={m_all['rel_l2']:.6f} | "
                f"post-wrap[510:650] cos={m_tail['cosine']:.6f} rel_l2={m_tail['rel_l2']:.6f}"
            )
            assert m_all["cosine"] >= 0.99995 and m_tail["cosine"] >= 0.99995, f"KV parity fail on {label}: {m_tail}"
            parity[label] = {"full_0_650": m_all, "post_wrap_510_650": m_tail}

    # 3. Attention Decode Latency Across Context Lengths (up to 100k!)
    print("\n[3] Single-Token Decode Attention Latency vs Context Length (up to 99,999)")
    print("  Context Pos | Layer 0 SWA@513 (ms) | Layer 4 Full RNoPE (ms)")
    attn_bench = {}
    with torch.inference_mode():
        x_dec = torch.randn((1, H), dtype=torch.bfloat16, device=dev)
        L0 = cache.gpu_non_experts[0]
        L4 = cache.gpu_non_experts[4]
        for ctx_len in (1000, 16000, 64000, 99990):
            kv.seq_len = ctx_len
            t_swa = bench_cuda_fn(lambda: kv.attention_forward(x_dec, L0, 0, inv_freq), warmup=10, iters=40)
            t_full = bench_cuda_fn(lambda: kv.attention_forward(x_dec, L4, 4, inv_freq), warmup=10, iters=40)
            print(f"  {ctx_len:11d} |        {t_swa['median_ms']:6.4f}        |         {t_full['median_ms']:6.4f}")
            attn_bench[str(ctx_len)] = {"swa513_ms": t_swa, "full_rnope_ms": t_full}

    report = {
        "kv_stats": {
            "max_seq_len": kv.stats.max_seq_len,
            "swa_kv_mib": kv.stats.swa_kv_mib,
            "full_kv_mib": kv.stats.full_kv_mib,
            "total_kv_mib": kv.stats.total_kv_mib,
            "total_kv_gib": kv.stats.total_kv_gib,
        },
        "vram_with_full_model_and_100k_kv": {
            "allocated_gib": alloc_gib,
            "reserved_gib": res_gib,
            "driver_free_gib": free_b / (1024.0 ** 3),
            "driver_total_gib": total_b / (1024.0 ** 3),
        },
        "parity": parity,
        "decode_attn_latency_ms": attn_bench,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved Step 3.3 results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
