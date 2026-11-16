"""Reproducible HF dataset sampler (`HuggingFaceTB/smoltalk`) with strict train/test isolation.

Samples calibration prompts exclusively from the `train` split and held-out evaluation
prompts/prefill streams exclusively from the `test` split, caching the exact frozen
sample JSON under `serving/data/` so regression runs have zero network variance and zero
calibration-to-eval data leakage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset

DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "all"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _prompt_messages_until_user(messages: Sequence[Dict[str, Any]], max_turns: int = 3) -> List[Dict[str, Any]]:
    """Slice a conversation cleanly so the final turn is a `user` message ready for generation."""
    sliced = list(messages[:max_turns])
    while sliced and sliced[-1].get("role") != "user":
        sliced.pop()
    return sliced


def _stratified_stream_sample(
    split: str,
    target_count: int,
    seed: int,
    scan_limit: int = 400,
    min_chars: int = 40,
    max_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """Shuffle streamed HF rows deterministically and pick diverse sources in round-robin."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split, streaming=True)
    shuffled = ds.shuffle(seed=seed, buffer_size=4096)

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    seen_texts = set()
    scanned = 0
    for row in shuffled:
        scanned += 1
        msgs = row.get("messages") or []
        prompt_msgs = _prompt_messages_until_user(msgs, max_turns=3)
        if not prompt_msgs:
            continue
        user_txt = str(prompt_msgs[-1].get("content", ""))
        if not (min_chars <= len(user_txt) <= max_chars):
            continue
        key = user_txt[:200]
        if key in seen_texts:
            continue
        seen_texts.add(key)

        src = str(row.get("source", "unknown"))
        by_source.setdefault(src, []).append(
            {
                "source": src,
                "full_messages": msgs,
                "prompt_messages": prompt_msgs,
            }
        )
        if scanned >= scan_limit and sum(len(v) for v in by_source.values()) >= target_count * 2:
            break

    # Round-robin across sorted sources for deterministic multi-domain coverage
    sources = sorted(by_source.keys())
    picked: List[Dict[str, Any]] = []
    idx_by_src = {s: 0 for s in sources}
    while len(picked) < target_count:
        progressed = False
        for s in sources:
            i = idx_by_src[s]
            if i < len(by_source[s]):
                picked.append(by_source[s][i])
                idx_by_src[s] = i + 1
                progressed = True
                if len(picked) >= target_count:
                    break
        if not progressed:
            break

    assert len(picked) == target_count, (
        f"Requested {target_count} samples from {DATASET_NAME}/{split}, got {len(picked)}"
    )
    return picked


def load_or_create_benchmark_dataset(
    seed: int = 20261116,
    cache_path: Optional[Path] = None,
    resample: bool = False,
    n_calibration: int = 16,
    n_eval_total: int = 24,
) -> Dict[str, Any]:
    """Load frozen HF sample JSON, or stream & freeze a fresh disjoint `train`/`test` sample."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cache_path or (DATA_DIR / f"smoltalk_bench_seed{seed}_cal{n_calibration}.json")

    if out_path.exists() and not resample:
        return json.loads(out_path.read_text())

    cal_rows = _stratified_stream_sample("train", target_count=n_calibration, seed=seed)
    eval_rows = _stratified_stream_sample("test", target_count=n_eval_total, seed=seed + 1)

    # Assert zero prompt leakage between train calibration and held-out test rows
    cal_first_turns = {str(r["prompt_messages"][-1]["content"]).strip() for r in cal_rows}
    eval_first_turns = {str(r["prompt_messages"][-1]["content"]).strip() for r in eval_rows}
    overlap = cal_first_turns & eval_first_turns
    assert not overlap, f"Unexpected train/test prompt overlap: {len(overlap)}"

    payload: Dict[str, Any] = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "seed": seed,
        "calibration_split": "train",
        "eval_split": "test",
        "calibration": cal_rows,
        "eval_cold": eval_rows[0:1],
        "eval_multiturn_warmup": eval_rows[1:3],
        "eval_multiturn_measure": eval_rows[3:4],
        "eval_calibrated_decode": eval_rows[4:7],
        "eval_prefill": eval_rows[7:19],
        "eval_long_ctx_stream": eval_rows[19:24],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
