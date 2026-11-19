# Kolibri-4090: Hybrid RTX 4090 Serving Engine

Local quantization, hybrid CPU/GPU offloading, and OpenAI-compatible serving engine for **Aleph-Alpha/Kolibri-1** (`50` layers, `384` routed experts/layer, `top-6` routing, `100k` context) targeting a single **NVIDIA GeForce RTX 4090 (24 GiB VRAM, SM 8.9)** with **W4A16 NVFP4 Marlin**.

## Tracked Performance (`results/omp_pcore.json`)

Measured on **NVIDIA GeForce RTX 4090 (24 GiB)** + **Intel Core Ultra 7 265K (8 P-cores + 12 E-cores)** with `seed=20261116`, `persistent_slots_per_layer=110`, `prefill_chunk_size=256`, `max_seq_len=100064`, `load_time_s=3.109`, and `HuggingFaceTB/smoltalk` (`train` calibration → held-out `test` evaluation):

| Scenario | Prefill (tok/s) | Decode mean (`p50` / `p95`) (tok/s) | Cache Hit % `[min-max]` | PCIe MB | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cold_decode` | 185.58 | 17.46 (18.11 / 15.59) | 76.15% `[53.76–94.80]` | 14719.82 | 20.52 GiB |
| `calibrated_decode` | 212.32 | 18.03 (18.23 / 16.54) | 77.68% `[57.71–100.00]` | 13773.15 | 20.52 GiB |
| `prefill_n256` | 353.88 | - | 61.70% `[38.48–100.00]` | 9400.34 | 20.52 GiB |

## Architecture & Serving Design

- **Precision & Quantization (`src/nvfp4_quant.py`):**
  - Dense layers, attention projections (`w_qkv`, `w_o`), norms, shared experts (`sh_w13`, `sh_w2`), and router gates (`w_gate` in `BF16`, `gate_bias` in `FP32`) remain resident in VRAM.
  - All `384 × 50 = 19,200` routed experts are quantized natively on SM89 to **W4A16 NVFP4** (E2M1 nibbles + 16-element `float8_e4m3fn` block scales + per-expert FP32 `scale_2`) and pre-packed per expert into Marlin tiled format (`layer_00.safetensors` .. `layer_49.safetensors`).
- **Three-Tier Expert Cache & P-Core OpenMP Gather (`src/expert_cache.py`, `csrc/kolibriGatherExperts.cpp`):**
  - **VRAM Persistent + Decode Scratch:** `110` persistent LRU slots + `6` decode miss scratch slots per layer (`gpu_caches`).
  - **Double-Buffered Staging:** `2 × 274` pinned host bounce slabs (`pinned_bounce`) and `2 × 384` GPU staging slabs (`gpu_stage`) fed over a dedicated CUDA `copy_stream`.
  - **Host Store Pre-faulting:** `host_routed` (`mmap`-backed safetensors) is pre-faulted on load via Linux `MADV_POPULATE_READ` (with `MADV_WILLNEED` + 4 KB strided byte-touch fallback) with zero extra RAM allocation.
  - **P-Core Pinned OpenMP Gather:** `gather_experts_omp` releases the GIL, copies all 6 Marlin tensors per layer in one C++ call, partitions large slabs into 64-byte-aligned byte ranges across all threads (balancing even single-expert `m=1` decode misses), and pins worker threads to performance cores (`cpuinfo_max_freq >= 95%` of max) without mutating PyTorch's global `torch.get_num_threads()`.
- **Hybrid MoE & KV Execution (`src/hybrid_moe.py`, `src/kv_cache.py`, `csrc/`):**
  - Routes each layer across `0-miss` decode, `<=6-miss` async DMA overlapped with `shared_expert`, or `>6-miss` Hit/Miss split prefill.
  - Alternates `40` Sliding-Window Attention layers (`[40, 4, 513, 128]` ring buffers with FlashInfer NeoX RoPE) and `10` Full-Attention RNoPE layers (`(layer_idx + 1) % 5 == 0`, up to `100k` context).
  - Uses custom CUDA kernels for `fused_sandwich_norm`, `fused_add_post_moe_norm`, and standalone `fused_kolibri_router`, plus FlashInfer `fused_add_rmsnorm` and `silu_and_mul`.



## Usage

```bash
# Run the full regression benchmark and compare against the tracked OMP P-core report
/home/mkina/miniconda3/envs/thesis/bin/python scripts/bench_suite.py --compare results/omp_pcore.json

# Single-prompt CLI generation
/home/mkina/miniconda3/envs/thesis/bin/python scripts/generate.py --prompt "Explain MoE offloading on an RTX 4090."

# Start OpenAI-compatible HTTP/SSE server + Strata Web UI
/home/mkina/miniconda3/envs/thesis/bin/python scripts/web_server.py --host 0.0.0.0 --port 8080
```
