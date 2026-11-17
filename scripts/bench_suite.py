#!/usr/bin/env python3
"""Step 2: Deterministic HF-Backed Regression Benchmark Suite for Kolibri-1.

Uses a stratified random sample from `HuggingFaceTB/smoltalk` (`config="all"`) with
strict split isolation:
  - Calibration (`calibrate_hot_cache`): sampled exclusively from `train` split
  - Evaluation (`cold`, `warmed`, `calibrated`, `prefill`, `long_ctx`): sampled
    exclusively from held-out `test` split (zero prompt/token overlap)

Outputs `results/baseline.json` and supports `--compare <baseline.json>` to report exact
deltas in Prefill tok/s, Decode tok/s (mean, p50, p95), per-layer cache hit %,
PCIe MB transferred, peak VRAM, and token parity.
"""

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.bench_dataset import load_or_create_benchmark_dataset  # noqa: E402
from src.config import NUM_LAYERS, PHASE2_MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.engine import KolibriHybridEngine, format_messages_prompt  # noqa: E402

DEFAULT_OUT_PATH = RESULTS_DIR / "baseline.json"


def _set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _summarize_stats(engine: KolibriHybridEngine) -> Dict[str, Any]:
    s = engine.moe.stats
    layer_hits = s.layer_hit_rates_pct()
    return {
        "cache_hit_pct": round(s.hit_rate * 100.0, 2),
        "layer_hit_summary": {
            "min_pct": float(min(layer_hits)),
            "p50_pct": float(round(float(np.median(layer_hits)), 2)),
            "max_pct": float(max(layer_hits)),
        },
        "layer_hit_pct": layer_hits,
        "experts_transferred": int(s.experts_transferred),
        "pcie_mb_transferred": round(s.pcie_mb_transferred, 3),
    }


def _decode_with_step_timings(
    engine: KolibriHybridEngine,
    first_logits: torch.Tensor,
    n_steps: int,
    promote_misses: bool,
) -> Tuple[List[int], List[float]]:
    """Run `n_steps` greedy decode steps, returning `(token_ids, step_ms_list)`."""
    dev = engine.device
    token_ids: List[int] = []
    step_ms: List[float] = []
    next_id = int(torch.argmax(first_logits[0]).item())

    gc.collect()
    gc.disable()
    try:
        for idx in range(n_steps):
            token_ids.append(next_id)
            if idx + 1 >= n_steps:
                break
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            logits = engine.step(next_id, promote_misses=promote_misses)
            torch.cuda.synchronize(dev)
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            next_id = int(torch.argmax(logits[0]).item())
    finally:
        gc.enable()
    return token_ids, step_ms


def _pack_decode_metrics(step_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(step_ms, dtype=np.float64)
    total_ms = float(arr.sum())
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    return {
        "decode_steps": int(len(arr)),
        "decode_ms": round(total_ms, 2),
        "decode_tok_per_s": round(len(arr) / max(total_ms / 1000.0, 1e-9), 2),
        "decode_tok_per_s_p50": round(1000.0 / max(p50_ms, 1e-9), 2),
        "decode_tok_per_s_p95": round(1000.0 / max(p95_ms, 1e-9), 2),
        "decode_step_ms_p50": round(p50_ms, 3),
        "decode_step_ms_p95": round(p95_ms, 3),
    }


def _run_multi_prompt_decode(
    engine: KolibriHybridEngine,
    prompt_id_lists: Sequence[Sequence[int]],
    steps_per_prompt: int,
    promote_misses: bool = True,
) -> Dict[str, Any]:
    """Evaluate prefill + decode across multiple unseen `test` prompts without resetting between them."""
    dev = engine.device
    torch.cuda.reset_peak_memory_stats(dev)

    all_prefill_ms = 0.0
    all_prefill_toks = 0
    all_step_ms: List[float] = []
    all_toks: List[List[int]] = []

    # Accumulate decode stats across all eval prompts manually via a dedicated counter snapshot
    dec_pair_evals = np.zeros(NUM_LAYERS, dtype=np.int64)
    dec_hit_pairs = np.zeros(NUM_LAYERS, dtype=np.int64)
    dec_xfer = np.zeros(NUM_LAYERS, dtype=np.int64)
    dec_total_bytes_experts = 0

    for p_ids in prompt_id_lists:
        torch.cuda.synchronize(dev)
        t_p0 = time.perf_counter()
        logits = engine.prefill(p_ids, promote_misses=False)
        torch.cuda.synchronize(dev)
        all_prefill_ms += (time.perf_counter() - t_p0) * 1000.0
        all_prefill_toks += len(p_ids)

        engine.moe.reset_stats()
        toks, step_ms = _decode_with_step_timings(
            engine, logits, n_steps=steps_per_prompt, promote_misses=promote_misses
        )
        all_toks.append(toks)
        all_step_ms.extend(step_ms)

        s = engine.moe.stats
        dec_pair_evals += np.asarray(s.layer_pair_evals, dtype=np.int64)
        dec_hit_pairs += np.asarray(s.layer_hit_pairs, dtype=np.int64)
        dec_xfer += np.asarray(s.layer_experts_transferred, dtype=np.int64)
        dec_total_bytes_experts += int(s.experts_transferred)

    from src.expert_cache import BYTES_PER_EXPERT

    layer_hits = [
        round((float(h) / float(e)) * 100.0, 2) if e > 0 else 100.0
        for h, e in zip(dec_hit_pairs.tolist(), dec_pair_evals.tolist())
    ]
    total_hit_pct = round((float(dec_hit_pairs.sum()) / max(float(dec_pair_evals.sum()), 1.0)) * 100.0, 2)
    return {
        "num_eval_prompts": len(prompt_id_lists),
        "prompt_tokens": all_prefill_toks,
        "prefill_ms": round(all_prefill_ms, 2),
        "prefill_tok_per_s": round(all_prefill_toks / max(all_prefill_ms / 1000.0, 1e-9), 2),
        **_pack_decode_metrics(all_step_ms),
        "cache_hit_pct": total_hit_pct,
        "layer_hit_summary": {
            "min_pct": float(min(layer_hits)),
            "p50_pct": float(round(float(np.median(layer_hits)), 2)),
            "max_pct": float(max(layer_hits)),
        },
        "layer_hit_pct": layer_hits,
        "experts_transferred": dec_total_bytes_experts,
        "pcie_mb_transferred": round((dec_total_bytes_experts * BYTES_PER_EXPERT) / 1e6, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 3), 3),
        "token_ids": all_toks,
    }


def _bench_prefill_only(
    engine: KolibriHybridEngine, token_ids: Sequence[int], reps: int = 3
) -> Dict[str, Any]:
    """Time chunked prefill over `reps` runs (after 1 warmup) with invariant cache (`promote_misses=False`)."""
    dev = engine.device
    engine.prefill(token_ids, promote_misses=False)
    torch.cuda.synchronize(dev)

    rep_ms: List[float] = []
    last_stats: Optional[Dict[str, Any]] = None
    torch.cuda.reset_peak_memory_stats(dev)
    gc.collect()
    gc.disable()
    try:
        for _ in range(reps):
            engine.moe.reset_stats()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            engine.prefill(token_ids, promote_misses=False)
            torch.cuda.synchronize(dev)
            rep_ms.append((time.perf_counter() - t0) * 1000.0)
            last_stats = _summarize_stats(engine)
    finally:
        gc.enable()

    med_ms = float(np.median(rep_ms))
    assert last_stats is not None
    return {
        "prompt_tokens": len(token_ids),
        "prefill_ms": round(med_ms, 2),
        "prefill_tok_per_s": round(len(token_ids) / max(med_ms / 1000.0, 1e-9), 2),
        **last_stats,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 3), 3),
    }


def _bench_long_context_decode(
    engine: KolibriHybridEngine,
    ctx_pos: int,
    fixed_token_stream: Sequence[int],
    n_steps: int,
) -> Dict[str, Any]:
    """Decode `n_steps` fixed token IDs at `ctx_pos` so routing stays comparable across depths."""
    dev = engine.device
    torch.cuda.reset_peak_memory_stats(dev)
    engine.kv.seq_len = ctx_pos

    for w in range(2):
        engine.step(int(fixed_token_stream[w % len(fixed_token_stream)]), promote_misses=True)

    engine.moe.reset_stats()
    step_ms: List[float] = []
    gc.collect()
    gc.disable()
    try:
        for idx in range(n_steps):
            tok = int(fixed_token_stream[(idx + 2) % len(fixed_token_stream)])
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            engine.step(tok, promote_misses=True)
            torch.cuda.synchronize(dev)
            step_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        gc.enable()

    return {
        "context_pos": ctx_pos,
        **_pack_decode_metrics(step_ms),
        **_summarize_stats(engine),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 3), 3),
    }


def _seed_kv_deterministically(engine: KolibriHybridEngine, seed: int) -> None:
    """Fill KV tensors with finite small deterministic values before artificial ctx_pos jumps."""
    gen = torch.Generator(device=engine.device).manual_seed(seed + 99)
    engine.kv.k_full.normal_(0.0, 0.02, generator=gen)
    engine.kv.v_full.normal_(0.0, 0.02, generator=gen)
    engine.kv.k_swa.normal_(0.0, 0.02, generator=gen)
    engine.kv.v_swa.normal_(0.0, 0.02, generator=gen)


def _reset_cold_engine(engine: KolibriHybridEngine) -> None:
    engine.cache.reset_default_slots()
    engine.moe._step_id = 0
    engine.moe._buf_cursor = 0
    engine.moe.reset_stats()
    engine.kv.reset()


def _seed_calibrated_slots(
    engine: KolibriHybridEngine, calibrated_slot_ids: Sequence[Sequence[int]]
) -> None:
    for li in range(NUM_LAYERS):
        engine.cache.seed_layer_cache(li, calibrated_slot_ids[li], step_id=0)
    engine.moe._step_id = 0
    engine.moe._buf_cursor = 0
    engine.moe.reset_stats()
    engine.kv.reset()
    torch.cuda.synchronize(engine.device)


def run_suite(
    seed: int,
    decode_steps: int,
    long_decode_steps: int,
    cal_samples: int,
    resample: bool,
    scenarios_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    _set_deterministic(seed)
    dev = torch.device("cuda:0")
    selected = set(scenarios_filter) if scenarios_filter else {"cold", "calibrated", "prefill", "long_ctx"}

    ds = load_or_create_benchmark_dataset(
        seed=seed, resample=resample, n_calibration=cal_samples
    )

    engine = KolibriHybridEngine(
        model_dir=PHASE2_MODEL_DIR,
        device=dev,
        persistent_slots_per_layer=110,
        max_seq_len=100_064,
        prefill_chunk_size=256,
    )

    def fmt_eval(rows: Sequence[Dict[str, Any]]) -> List[str]:
        return [format_messages_prompt(r["prompt_messages"], reasoning_effort="none") for r in rows]

    def fmt_cal(rows: Sequence[Dict[str, Any]]) -> List[str]:
        return [format_messages_prompt(r["prompt_messages"], reasoning_effort="none") for r in rows]

    cal_prompts = fmt_cal(ds["calibration"])
    ids_decode_eval = [engine.tokenizer.encode(p).ids for p in fmt_eval(ds["eval_calibrated_decode"])]

    prefill_stream: List[int] = []
    for p in fmt_eval(ds["eval_prefill"]):
        prefill_stream.extend(engine.tokenizer.encode(p).ids)
    assert len(prefill_stream) >= 1024, f"Held-out prefill stream only {len(prefill_stream)} tokens"

    long_stream: List[int] = []
    for p in fmt_eval(ds["eval_long_ctx_stream"]):
        long_stream.extend(engine.tokenizer.encode(p).ids)

    # Untimed cuBLAS/SDPA graph warmup
    engine.prefill(ids_decode_eval[0][:16], promote_misses=False)
    engine.step(ids_decode_eval[0][0], promote_misses=False)
    torch.cuda.synchronize(dev)

    scenarios: Dict[str, Any] = {}
    eval_steps = max(decode_steps // 2, 24)

    # 1. Cold decode (0..109 init, 3 held-out test prompts)
    if "cold" in selected:
        _reset_cold_engine(engine)
        scenarios["cold_decode"] = _run_multi_prompt_decode(
            engine, ids_decode_eval, steps_per_prompt=eval_steps, promote_misses=True
        )

    cal_stats = None
    calibrated_slots: Optional[List[List[int]]] = None
    if selected & {"calibrated", "prefill", "long_ctx"}:
        # Calibrate hot cache ONCE strictly on `train` split and snapshot slot IDs
        _reset_cold_engine(engine)
        cal_stats = engine.calibrate_hot_cache(cal_prompts)
        calibrated_slots = [engine.cache.cache_states[li].slot_to_expert.tolist() for li in range(NUM_LAYERS)]

    # 2. Calibrated decode (same 3 held-out test prompts -> exact apples-to-apples vs cold)
    if "calibrated" in selected and calibrated_slots is not None:
        _seed_calibrated_slots(engine, calibrated_slots)
        scenarios["calibrated_decode"] = _run_multi_prompt_decode(
            engine, ids_decode_eval, steps_per_prompt=eval_steps, promote_misses=True
        )

    # 3. Prefill N=256 and N=1024 on calibrated cache using disjoint held-out `test` tokens
    if "prefill" in selected and calibrated_slots is not None:
        _seed_calibrated_slots(engine, calibrated_slots)
        scenarios["prefill_n256"] = _bench_prefill_only(engine, prefill_stream[:256])
        scenarios["prefill_n1024"] = _bench_prefill_only(engine, prefill_stream[:1024])

    # 4. Long-context decode at 16k, 64k, 100k (fresh calibrated slots per depth + held-out stream)
    if "long_ctx" in selected and calibrated_slots is not None:
        _seed_kv_deterministically(engine, seed)
        for name, ctx_pos in (
            ("long_ctx_16k", 16_000),
            ("long_ctx_64k", 64_000),
            ("long_ctx_100k", 100_000),
        ):
            _seed_calibrated_slots(engine, calibrated_slots)
            scenarios[name] = _bench_long_context_decode(
                engine, ctx_pos=ctx_pos, fixed_token_stream=long_stream, n_steps=long_decode_steps
            )

    free_b, total_b = torch.cuda.mem_get_info(dev)
    return {
        "metadata": {
            "gpu": torch.cuda.get_device_name(dev),
            "seed": seed,
            "decode_steps": decode_steps,
            "long_decode_steps": long_decode_steps,
            "persistent_slots_per_layer": engine.cache.persistent_slots,
            "max_seq_len": engine.kv.max_seq_len,
            "prefill_chunk_size": engine.prefill_chunk_size,
            "load_time_s": round(engine.cache.stats.load_time_s, 3),
            "driver_free_gib": round(free_b / (1024.0 ** 3), 3),
            "driver_total_gib": round(total_b / (1024.0 ** 3), 3),
            "dataset": {
                "name": ds["dataset"],
                "config": ds["config"],
                "calibration_split": ds["calibration_split"],
                "eval_split": ds["eval_split"],
                "calibration_sources": dict(Counter(r["source"] for r in ds["calibration"])),
                "calibration": cal_stats,
                "held_out_prefill_pool_tokens": len(prefill_stream),
            },
        },
        "scenarios": scenarios,
    }


def _fmt_delta(cur: Optional[float], base: Optional[float], unit: str = "", pct: bool = True) -> str:
    if cur is None:
        return "-"
    if base is None:
        return f"{cur:.2f}{unit}"
    diff = cur - base
    sign = "+" if diff >= 0 else ""
    if pct and abs(base) > 1e-9:
        rel = (diff / base) * 100.0
        return f"{cur:.2f}{unit} ({sign}{diff:.2f}, {sign}{rel:.1f}%)"
    return f"{cur:.2f}{unit} ({sign}{diff:.2f})"


def print_summary_and_compare(
    current: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None
) -> None:
    meta = current["metadata"]
    ds_info = meta.get("dataset", {})
    print(
        f"\n=== Kolibri-1 Regression Suite | GPU={meta['gpu']} | "
        f"seed={meta['seed']} | slots={meta['persistent_slots_per_layer']} | "
        f"dataset={ds_info.get('name', 'custom')} ({ds_info.get('calibration_split', '?')}->{ds_info.get('eval_split', '?')}) ==="
    )
    if baseline is not None:
        print(f"Comparing against baseline (seed={baseline['metadata'].get('seed')})")

    header = (
        f"{'Scenario':24s} | {'Prefill tok/s':26s} | {'Decode mean (p50/p95)':32s} | "
        f"{'Cache Hit % [min-max]':26s} | {'PCIe MB':19s} | {'Peak VRAM':15s} | {'Parity'}"
    )
    print(header)
    print("-" * len(header))

    cur_sc = current["scenarios"]
    base_sc = baseline["scenarios"] if baseline is not None else {}

    for name, c in cur_sc.items():
        b = base_sc.get(name) if isinstance(base_sc, dict) else None
        pref_str = _fmt_delta(c.get("prefill_tok_per_s"), b.get("prefill_tok_per_s") if b else None)

        if "decode_tok_per_s" in c:
            dec_base = b.get("decode_tok_per_s") if b else None
            dec_diff = ""
            if dec_base is not None:
                d = c["decode_tok_per_s"] - dec_base
                rel = (d / max(dec_base, 1e-9)) * 100.0
                dec_diff = f" ({'+' if d >= 0 else ''}{rel:.1f}%)"
            dec_str = (
                f"{c['decode_tok_per_s']:.2f} "
                f"({c['decode_tok_per_s_p50']:.1f}/{c['decode_tok_per_s_p95']:.1f}){dec_diff}"
            )
        else:
            dec_str = "-"

        ls = c["layer_hit_summary"]
        hit_diff = ""
        if b is not None:
            hd = c["cache_hit_pct"] - b["cache_hit_pct"]
            hit_diff = f" ({'+' if hd >= 0 else ''}{hd:.1f})"
        hit_str = f"{c['cache_hit_pct']:.1f}% [{ls['min_pct']:.0f}-{ls['max_pct']:.0f}]{hit_diff}"

        pcie_str = _fmt_delta(
            c["pcie_mb_transferred"], b.get("pcie_mb_transferred") if b else None, pct=False
        )
        vram_str = _fmt_delta(
            c["peak_vram_gib"], b.get("peak_vram_gib") if b else None, unit="G", pct=False
        )

        parity = "-"
        if b is not None and "token_ids" in c and "token_ids" in b:
            parity = "EXACT" if c["token_ids"] == b["token_ids"] else "DIFF"

        print(
            f"{name:24s} | {pref_str:26s} | {dec_str:32s} | "
            f"{hit_str:26s} | {pcie_str:19s} | {vram_str:15s} | {parity}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kolibri-1 HF-Backed Deterministic Regression Suite")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output JSON path")
    parser.add_argument("--compare", type=Path, default=None, help="Baseline JSON to compare against")
    parser.add_argument(
        "--compare-only",
        nargs=2,
        type=Path,
        metavar=("BASELINE_JSON", "CURRENT_JSON"),
        default=None,
        help="Compare two existing JSON files without running the GPU suite",
    )
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--cal-samples", type=int, default=16)
    parser.add_argument("--resample", action="store_true", help="Force fresh HF stream sample")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--long-decode-steps", type=int, default=32)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("cold", "calibrated", "prefill", "long_ctx"),
        default=None,
        help="Subset of scenarios to run (default: all)",
    )
    args = parser.parse_args()

    if args.compare_only is not None:
        base_doc = json.loads(args.compare_only[0].read_text())
        cur_doc = json.loads(args.compare_only[1].read_text())
        print_summary_and_compare(cur_doc, base_doc)
        return

    report = run_suite(
        seed=args.seed,
        decode_steps=args.decode_steps,
        long_decode_steps=args.long_decode_steps,
        cal_samples=args.cal_samples,
        resample=args.resample,
        scenarios_filter=args.scenarios,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    baseline = json.loads(args.compare.read_text()) if args.compare is not None else None
    print_summary_and_compare(report, baseline)
    print(f"\nSaved benchmark report to {args.out}")


if __name__ == "__main__":
    main()
