# Kolibri-1 Serving Workspace

Local hosting and quantization pipeline for **Aleph-Alpha/Kolibri-1** (`50` layers, `384` routed experts/layer, `top-6`) targeting **RTX 4090 (24 GiB VRAM, SM 8.9)** with **W4A16 NVFP4 Marlin**.

## Directory Structure

```text
serving/
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                     # Model dimensions, paths, SWA/RNoPE layer map
│   ├── nvfp4_quant.py                # SM89 CUDA E2M1+FP8 quantizer & Marlin repacker
│   └── model_ops.py                  # Kolibri attention, router, shared expert, Marlin MoE
├── scripts/
│   ├── phase1_marlin_nvfp4_2block.py # Phase 1: 2-block accuracy & CUTLASS vs Marlin bench
│   ├── quantize_kolibri_nvfp4.py     # Phase 2: Full 50-layer streaming quantization & packing
│   └── verify_nvfp4_checkpoint.py    # Checkpoint integrity & Layer 0..49 GPU sanity check
├── checkpoints/
│   └── kolibri-1-nvfp4-marlin/       # Quantized 50-layer safetensors + globals + tokenizer
├── results/
│   ├── phase1_results.json
│   ├── phase2_quant_report.json
│   └── phase2_verification.json
└── scratch/
    └── hf_cache/                     # Temporary rolling HF shard cache (evicted after pack)
```

## Strategy Overview
- **Dense / Attention / Shared Expert / Router Gate**: Kept in `BF16` (router `expert_bias` in `FP32`, `lm_head` preserved in source dtype).
- **Routed Experts (`384 × 50` = 19,200 experts)**: Quantized to **W4A16 NVFP4** (E2M1 nibbles + 16-element FP8 E4M3 group scales + per-expert `scale_2`) and pre-packed per expert into **Marlin tiled format** (`layer_00.safetensors` .. `layer_49.safetensors`).
