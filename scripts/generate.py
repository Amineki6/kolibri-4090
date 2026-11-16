#!/usr/bin/env python3
"""Interactive / single-prompt CLI generator for Kolibri-1 W4A16_NVFP4 Hybrid Engine."""

import argparse
import sys
from pathlib import Path

import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import PHASE2_MODEL_DIR  # noqa: E402
from src.engine import KolibriHybridEngine, format_chat_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text locally with 50-layer Kolibri-1 NVFP4 Marlin.")
    parser.add_argument("--prompt", type=str, default="Explain in 3 concise sentences what GPU Tensor Cores do.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--reasoning-effort", type=str, default="none", choices=["none", "low", "medium", "high"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--persistent-slots", type=int, default=110)
    parser.add_argument("--raw", action="store_true", help="Do not wrap prompt with Kolibri ChatML template")
    args = parser.parse_args()

    engine = KolibriHybridEngine(
        model_dir=PHASE2_MODEL_DIR,
        device=torch.device("cuda:0"),
        persistent_slots_per_layer=args.persistent_slots,
        max_seq_len=args.max_seq_len,
    )

    prompt = args.prompt if args.raw else format_chat_prompt(args.prompt, reasoning_effort=args.reasoning_effort)
    print(f"=== Prompt ({engine.tokenizer.encode(prompt).ids.__len__()} tokens) ===")
    res = engine.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stream_callback=lambda piece: print(piece, end="", flush=True),
    )
    print("\n\n=== Metrics ===")
    print(
        f"Prefill: {res.prompt_tokens} toks in {res.prefill_ms:.1f} ms ({res.prefill_tok_per_s:.1f} tok/s, "
        f"cache hit={res.prefill_hit_pct:.1f}%)"
    )
    print(
        f"Decode:  {res.generated_tokens} toks in {res.decode_ms:.1f} ms ({res.decode_tok_per_s:.2f} tok/s, "
        f"cache hit={res.decode_hit_pct:.1f}%, misses_dma={res.decode_experts_transferred})"
    )
    print(f"Peak CUDA allocated: {res.peak_vram_gib:.2f} GiB")


if __name__ == "__main__":
    main()
