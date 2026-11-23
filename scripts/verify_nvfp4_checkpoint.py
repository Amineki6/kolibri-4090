#!/usr/bin/env python3
"""Verify integrity, tensor contracts, and GPU forward parity of the Phase 2 checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from tensorrt_llm.quantization.utils.marlin_utils import marlin_make_workspace

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import H, NUM_LAYERS, PHASE1_CKPT_PATH, PHASE2_MODEL_DIR, RESULTS_DIR, TOPK  # noqa: E402
from src.model_ops import (  # noqa: E402
    exact_bf16_routed_moe,
    make_rope_inv_freq,
    marlin_nvfp4_routed_moe,
    run_layers_forward,
)
from src.nvfp4_quant import metrics  # noqa: E402

EXPECTED_SHAPES = {
    "input_ln": ((2560,), torch.bfloat16),
    "post_attn_ln": ((2560,), torch.bfloat16),
    "pre_moe_ln": ((2560,), torch.bfloat16),
    "post_ffn_ln": ((2560,), torch.bfloat16),
    "q_norm": ((128,), torch.bfloat16),
    "k_norm": ((128,), torch.bfloat16),
    "w_qkv": ((2560, 7168), torch.bfloat16),
    "w_o": ((6144, 2560), torch.bfloat16),
    "w_gate": ((2560, 384), torch.bfloat16),
    "gate_bias": ((384,), torch.float32),
    "sh_w13": ((2560, 1024), torch.bfloat16),
    "sh_w2": ((512, 2560), torch.bfloat16),
    "w13_marlin": ((384, 160, 1024), torch.int64),
    "w13_scale_marlin": ((384, 160, 256), torch.int32),
    "w13_gs_marlin": ((384,), torch.bfloat16),
    "w2_marlin": ((384, 32, 2560), torch.int64),
    "w2_scale_marlin": ((384, 32, 640), torch.int32),
    "w2_gs_marlin": ((384,), torch.bfloat16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 2 Kolibri-1 NVFP4 Marlin checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=PHASE2_MODEL_DIR)
    parser.add_argument(
        "--bf16-ref",
        type=Path,
        default=PHASE1_CKPT_PATH,
        help="Optional 2-block BF16 reference .pt for Step 2 parity check",
    )
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    print(f"=== Verifying Kolibri-1 NVFP4 Marlin Checkpoint: {args.model_dir} ===")

    # 1. Header check across all 50 layer files + globals
    globals_file = args.model_dir / "globals.safetensors"
    assert globals_file.exists(), f"Missing {globals_file}"
    with safe_open(str(globals_file), framework="pt", device="cpu") as f:
        g_keys = set(f.keys())
        assert g_keys == {"embed_tokens", "model_norm", "lm_head"}, f"Unexpected globals keys: {g_keys}"
        for k in g_keys:
            sl = f.get_slice(k)
            print(f"  globals.{k}: shape={tuple(sl.get_shape())} dtype={sl.get_dtype()}")

    layer_files = [args.model_dir / f"layer_{i:02d}.safetensors" for i in range(NUM_LAYERS)]
    missing = [p.name for p in layer_files if not p.exists()]
    assert not missing, f"Missing layer files: {missing}"

    st_dtype_map = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "I64": torch.int64,
        "I32": torch.int32,
    }
    total_bytes = globals_file.stat().st_size
    for lf in layer_files:
        total_bytes += lf.stat().st_size
        with safe_open(str(lf), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            assert keys == set(EXPECTED_SHAPES), f"{lf.name} key mismatch: {keys ^ set(EXPECTED_SHAPES)}"
            for k, (exp_shape, exp_dtype) in EXPECTED_SHAPES.items():
                sl = f.get_slice(k)
                assert tuple(sl.get_shape()) == exp_shape, f"{lf.name} {k} shape {sl.get_shape()} != {exp_shape}"
                got_dtype = st_dtype_map[sl.get_dtype()]
                assert got_dtype == exp_dtype, f"{lf.name} {k} dtype {got_dtype} != {exp_dtype}"
    print(f"[+] Header shape/dtype/key verification PASSED for all {NUM_LAYERS} layers ({total_bytes / (1024**3):.2f} GiB total)")

    # 2. Functional GPU execution parity check: Layers 0 & 1 vs optional Phase 1 BF16 checkpoint
    inv_freq = make_rope_inv_freq(dev)
    workspace = marlin_make_workspace(dev, max_blocks_per_sm=4)

    parity_results = {}
    if args.bf16_ref is not None and args.bf16_ref.exists():
        bf16_2blk = torch.load(args.bf16_ref, map_location="cpu")
        bf16_layers_gpu = [{k: v.to(dev) for k, v in L.items()} for L in bf16_2blk]
        del bf16_2blk
        w13_refs = [L["w13"].transpose(1, 2).contiguous() for L in bf16_layers_gpu]
        w2_refs = [L["w2"].transpose(1, 2).contiguous() for L in bf16_layers_gpu]

        nvfp4_2blk = [
            {k: v.to(dev) for k, v in load_file(str(layer_files[i]), device="cpu").items()}
            for i in (0, 1)
        ]

        bf16_idx_by_id = {id(L): idx for idx, L in enumerate(bf16_layers_gpu)}
        with torch.inference_mode():
            for n in (1, 64):
                gen = torch.Generator(device=dev).manual_seed(9000 + n)
                x = torch.empty((n, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=gen)

                out_ref, ids_ref, _ = run_layers_forward(
                    x,
                    bf16_layers_gpu,
                    inv_freq,
                    lambda mx, mids, mwts, mL: exact_bf16_routed_moe(
                        mx, mids, mwts,
                        w13_refs[bf16_idx_by_id[id(mL)]],
                        w2_refs[bf16_idx_by_id[id(mL)]],
                    ),
                )
                out_nvfp4, ids_nvfp4, _ = run_layers_forward(
                    x,
                    nvfp4_2blk,
                    inv_freq,
                    lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, workspace),
                )
                m = metrics(out_ref, out_nvfp4)
                l1_overlap = sum(
                    len(set(ids_ref[1][t].tolist()) & set(ids_nvfp4[1][t].tolist())) / TOPK
                    for t in range(n)
                ) / n
                parity_results[str(n)] = {**m, "l1_router_topk_overlap": l1_overlap}
                print(
                    f"[+] Saved Phase2 Layers 0-1 vs Phase1 BF16 (N={n}): "
                    f"cos={m['cosine']:.6f}, rel_l2={m['rel_l2']:.6f}, l1_overlap={l1_overlap*100:.1f}%"
                )

        del bf16_layers_gpu, w13_refs, w2_refs, nvfp4_2blk
        torch.cuda.empty_cache()
    else:
        print("[skip] Skipping 2-block BF16 reference parity (no --bf16-ref checkpoint provided)")

    # 3. Smoke check sampled deep layers (including Layer 49 full-attention RNoPE)
    with torch.inference_mode():
        sample_ids = [24, 49]
        for li in sample_ids:
            L = {k: v.to(dev) for k, v in load_file(str(layer_files[li]), device="cpu").items()}
            x = torch.randn((8, H), dtype=torch.bfloat16, device=dev)
            out, _, _ = run_layers_forward(
                x,
                [L],
                inv_freq,
                lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, workspace),
                start_layer_idx=li,
            )
            assert torch.isfinite(out).all(), f"Non-finite output in Layer {li}"
            print(f"[+] Deep layer smoke test PASSED: Layer {li:02d} (std={out.float().std():.4f}, finite=True)")
            del L
            torch.cuda.empty_cache()

    # 4. Smoke check token embedding -> Layer 0 -> model_norm -> lm_head
    with torch.inference_mode():
        g = load_file(str(globals_file), device="cpu")
        tok = torch.tensor([1, 42, 1000], dtype=torch.long, device=dev)
        h = F.embedding(tok, g["embed_tokens"].to(dev)).to(torch.bfloat16)
        L0 = {k: v.to(dev) for k, v in load_file(str(layer_files[0]), device="cpu").items()}
        h, _, _ = run_layers_forward(
            h, [L0], inv_freq, lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, workspace)
        )
        h_norm = F.rms_norm(h, (H,), g["model_norm"].to(dev), 1e-6)
        logits = F.linear(h_norm.to(g["lm_head"].dtype), g["lm_head"].to(dev))
        assert logits.shape == (3, 128000) and torch.isfinite(logits).all()
        print(f"[+] End-to-end token->logits smoke check PASSED: logits shape={tuple(logits.shape)} dtype={logits.dtype}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    verif_payload = {
        "model_dir": str(args.model_dir),
        "num_layers_verified": NUM_LAYERS,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024 ** 3),
    }
    if parity_results:
        verif_payload["layers_0_1_parity"] = parity_results
    verif_path = RESULTS_DIR / "phase2_verification.json"
    verif_path.write_text(json.dumps(verif_payload, indent=2))
    print(f"Saved verification report to {verif_path}")


if __name__ == "__main__":
    main()
