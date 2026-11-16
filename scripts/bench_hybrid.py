#!/usr/bin/env python3
"""Step 3.4 End-to-End 50-Layer Benchmark: Real Text Generation, Warmup Cache & 100k Context."""

import json
import sys
import time
from pathlib import Path

import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import PHASE2_MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.engine import KolibriHybridEngine, format_chat_prompt  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "step3_4_e2e_bench.json"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")

    print("=== Step 3.4: Full 50-Layer Kolibri-1 Hybrid Engine E2E Verification & Benchmark ===")
    engine = KolibriHybridEngine(
        model_dir=PHASE2_MODEL_DIR,
        device=dev,
        persistent_slots_per_layer=110,
        max_seq_len=100_000,
        prefill_chunk_size=256,
    )
    free_b, total_b = torch.cuda.mem_get_info(dev)
    print(
        f"Engine loaded in {engine.cache.stats.load_time_s:.2f}s | "
        f"VRAM allocated={torch.cuda.memory_allocated(dev)/(1024**3):.2f} GiB | "
        f"driver free={free_b/(1024**3):.2f} / {total_b/(1024**3):.2f} GiB"
    )

    # 1. Cold-cache real generation (prompt 1)
    p1 = format_chat_prompt(
        "Explain in 3 concise sentences how MoE (Mixture of Experts) models reduce compute per token.",
        reasoning_effort="none",
    )
    print("\n[1] Cold-Cache Generation (Prompt 1, reasoning_effort=none)")
    r1 = engine.generate(p1, max_new_tokens=64, temperature=0.0, promote_misses=True)
    print(f"  Output: {r1.text!r}")
    print(
        f"  Prefill: {r1.prompt_tokens} toks | {r1.prefill_ms:.1f} ms ({r1.prefill_tok_per_s:.1f} tok/s, hit={r1.prefill_hit_pct:.1f}%)"
    )
    print(
        f"  Decode:  {r1.generated_tokens} toks | {r1.decode_ms:.1f} ms ({r1.decode_tok_per_s:.2f} tok/s, "
        f"hit={r1.decode_hit_pct:.1f}%, experts_dma={r1.decode_experts_transferred})"
    )

    # 2. Warmed-cache real generation (re-run Prompt 1 + Prompt 2 after LRU warmup)
    print("\n[2] Warmed-Cache Generation (Prompt 1 repeat -> tests LRU warm hit rate)")
    r1_warm = engine.generate(p1, max_new_tokens=64, temperature=0.0, promote_misses=True)
    print(
        f"  Prefill: {r1_warm.prompt_tokens} toks | {r1_warm.prefill_ms:.1f} ms ({r1_warm.prefill_tok_per_s:.1f} tok/s, hit={r1_warm.prefill_hit_pct:.1f}%)"
    )
    print(
        f"  Decode:  {r1_warm.generated_tokens} toks | {r1_warm.decode_ms:.1f} ms ({r1_warm.decode_tok_per_s:.2f} tok/s, "
        f"hit={r1_warm.decode_hit_pct:.1f}%, experts_dma={r1_warm.decode_experts_transferred})"
    )

    p2 = format_chat_prompt(
        "Write a short Python function `fibonacci(n)` that returns the n-th Fibonacci number iteratively.",
        reasoning_effort="none",
    )
    print("\n[3] New Coding Prompt (Prompt 2, sharing warm LRU cache)")
    r2 = engine.generate(p2, max_new_tokens=64, temperature=0.0, promote_misses=True)
    print(f"  Output:\n{r2.text}")
    print(
        f"  Prefill: {r2.prompt_tokens} toks | {r2.prefill_ms:.1f} ms ({r2.prefill_tok_per_s:.1f} tok/s, hit={r2.prefill_hit_pct:.1f}%)"
    )
    print(
        f"  Decode:  {r2.generated_tokens} toks | {r2.decode_ms:.1f} ms ({r2.decode_tok_per_s:.2f} tok/s, "
        f"hit={r2.decode_hit_pct:.1f}%, experts_dma={r2.decode_experts_transferred})"
    )

    # 3. Calibration pre-warm + Long Prefill N=1024 + Decode at ~99.5k context stress test
    cal_stats = engine.calibrate_hot_cache([p1, p2])
    print(f"\n[4] Prefill Throughput Test (N=1024 tokens, chunk=256, calibrated cache)")
    long_ids = (engine.tokenizer.encode(p1).ids * 30)[:1024]
    engine.moe.reset_stats()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    logits_1k = engine.prefill(long_ids, promote_misses=False)
    torch.cuda.synchronize(dev)
    ms_1k = (time.perf_counter() - t0) * 1000.0
    hit_1k = engine.moe.stats.hit_rate * 100.0
    print(f"  N=1024 Prefill: {ms_1k:.1f} ms ({1024 / (ms_1k / 1000.0):.1f} tok/s, hit={hit_1k:.1f}%)")

    # Decode latency check at context_pos = 99,900 inside the 100k KV buffer
    print("\n[5] Decode Speed at Context Position 99,900 (inside 100k KV buffer)")
    engine.kv.seq_len = 99_900
    engine.moe.reset_stats()
    tok_id = int(torch.argmax(logits_1k[0]).item())
    # Warm 2 steps at 99,900 then time 16 decode steps
    for _ in range(2):
        logits_next = engine.step(tok_id, promote_misses=True)
        tok_id = int(torch.argmax(logits_next[0]).item())
    engine.moe.reset_stats()
    torch.cuda.synchronize(dev)
    t_99k = time.perf_counter()
    for _ in range(16):
        logits_next = engine.step(tok_id, promote_misses=True)
        tok_id = int(torch.argmax(logits_next[0]).item())
    torch.cuda.synchronize(dev)
    ms_99k = (time.perf_counter() - t_99k) * 1000.0
    toks_99k = 16 / (ms_99k / 1000.0)
    hit_99k = engine.moe.stats.hit_rate * 100.0
    peak_gib = torch.cuda.max_memory_allocated(dev) / (1024.0 ** 3)
    print(
        f"  16 tokens @99.9k ctx: {ms_99k:.1f} ms ({toks_99k:.2f} tok/s, hit={hit_99k:.1f}%, "
        f"experts_dma={engine.moe.stats.experts_transferred}) | Peak VRAM={peak_gib:.2f} GiB"
    )

    report = {
        "gpu": torch.cuda.get_device_name(dev),
        "load_time_s": engine.cache.stats.load_time_s,
        "peak_vram_gib": peak_gib,
        "driver_free_after_load_gib": free_b / (1024.0 ** 3),
        "prompt1_cold": r1.__dict__,
        "prompt1_warm": r1_warm.__dict__,
        "prompt2_coding": r2.__dict__,
        "calibration": cal_stats,
        "prefill_1024": {
            "tokens": 1024,
            "ms": ms_1k,
            "tok_per_s": 1024 / (ms_1k / 1000.0),
            "hit_pct": hit_1k,
        },
        "decode_99900_ctx": {
            "tokens": 16,
            "ms": ms_99k,
            "tok_per_s": toks_99k,
            "hit_pct": hit_99k,
            "experts_transferred": engine.moe.stats.experts_transferred,
        },
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved Step 3.4 results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
