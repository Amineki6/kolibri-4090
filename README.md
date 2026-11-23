![Kolibri-1 78B Inference Engine](assets/banner.jpg)

# Kolibri-4090: Hybrid RTX 4090 Serving Engine

Standalone streaming quantizer, hybrid CPU/GPU expert-offloading engine, and OpenAI-compatible HTTP server for **[Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)** (`50` layers, `384` routed experts/layer, `top-6` routing, `100k` context) targeting a single **NVIDIA GeForce RTX 4090 (24 GiB VRAM, SM 8.9)** with **W4A16 NVFP4 Marlin**.

- **Zero pre-quantized checkpoint download required:** `scripts/quantize_kolibri_nvfp4.py` streams the official Apache-2.0 `Aleph-Alpha/Kolibri-1-BF16` shards directly from Hugging Face, quantizes routed experts natively on your 4090, and evicts finished BF16 shards on the fly.
- **Full 100k context on 24 GiB VRAM:** Keeps `110` hot expert slots/layer + attention + shared experts + 100k paged KV cache inside **20.6 GiB peak VRAM**.
- **~51–53 tok/s calibrated decode:** Uses `mmap` pre-faulting, P-core pinned OpenMP slab gather, CUDA mempool retention, fused C++/CUDA DMA & promotion ops, FlashInfer paged NHD decode, and per-layer CUDA Graphs (Pre-MoE Graph A + MoE Graph B). See [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) for the full 5.1×–7.9× optimization write-up.

---

## Performance (`results/omp_mempool_cacheops_graphab.json`)

Measured on **NVIDIA GeForce RTX 4090 (24 GiB)** + **Intel Core Ultra 7 265K (8 P-cores + 12 E-cores)** with `seed=20261116`, `persistent_slots_per_layer=110`, `prefill_chunk_size=256`, `max_seq_len=100064`, `load_time_s=3.093`, and `HuggingFaceTB/smoltalk` (`train` calibration → held-out `test` evaluation):

| Scenario | Prefill (tok/s) | Decode mean (`p50` / `p95`) (tok/s) | Cache Hit % `[min–max]` | PCIe MB | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cold_decode` | 194.56 | 47.05 (`50.5` / `33.1`) | 76.0% `[53–95]` | 14777.33 | 20.61G |
| `calibrated_decode` | 226.09 | 51.09 (`52.9` / `38.0`) | 78.4% `[60–100]` | 13330.78 | 20.61G |
| `prefill_n256` | 381.06 | - | 61.8% `[39–100]` | 9422.46 | 20.61G |
| `prefill_n1024` | 349.92 | - | 60.6% `[45–100]` | 40678.02 | 20.61G |
| `long_ctx_16k` | - | 52.58 (`54.4` / `41.1`) | 80.9% `[60–100]` | 4056.52 | 20.58G |
| `long_ctx_64k` | - | 52.57 (`53.5` / `45.4`) | 83.3% `[59–100]` | 3554.43 | 20.58G |
| `long_ctx_100k` | - | 51.42 (`53.9` / `43.5`) | 84.1% `[61–100]` | 3375.27 | 20.58G |

---

## Requirements

- **GPU:** NVIDIA GeForce RTX 4090 (24 GiB VRAM, SM 8.9 Ada Lovelace). If your GPU also drives a desktop display/compositor, see [VRAM Headroom Tuning](#vram-headroom-tuning).
- **System RAM:** 64 GiB recommended (`44.4 GiB` quantized checkpoint; `~31 GiB` of cold host routed experts are pre-faulted from `mmap`-backed safetensors).
- **Disk Space:** ~60 GiB free during initial quantization (`44.4 GiB` final checkpoint + `9–14 GiB` temporary rolling download scratch; BF16 shards are automatically deleted as each layer completes).
- **OS & Build Tools:** Linux `x86_64` with NVIDIA driver + CUDA Toolkit 12.8+ (`nvcc`), `g++` with C++20 & OpenMP support (`libgomp`), and Python `3.10–3.12`.
- **Tested Python Stack:** `torch==2.14.0+cu130`, `tensorrt_llm==1.3.0rc29`, `flashinfer-python==0.6.18`, plus the packages in [`requirements.txt`](requirements.txt).

---

## Setup Guide: From Zero to Running Engine

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Amineki6/kolibri-4090.git
cd kolibri-4090

# Create and activate an isolated environment (conda or venv)
conda create -n kolibri python=3.10 -y
conda activate kolibri
```

Install CUDA-enabled **PyTorch**, **TensorRT-LLM**, **FlashInfer**, and the remaining dependencies (adjust CUDA wheel tags if your local CUDA toolkit differs):

```bash
# 1. Install PyTorch with CUDA support (tested with torch 2.14.0+cu130)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. Install TensorRT-LLM (provides Marlin NVFP4 kernels and moe_align_block_size)
pip install tensorrt-llm

# 3. Install FlashInfer (provides RMSNorm, NeoX RoPE, SiLU+Mul, and paged GQA decode)
pip install flashinfer-python

# 4. Install remaining runtime, quantization, and dataset packages
pip install -r requirements.txt
```

Verify your toolchain before proceeding (`nvcc` and `g++` are required on first run to compile the custom CUDA/OpenMP extension):

```bash
nvcc --version
g++ --version
python -c "import torch, tensorrt_llm, flashinfer; print(torch.cuda.get_device_name(0), torch.__version__)"
```

### 2. Download & Quantize Kolibri-1 Weights

`Aleph-Alpha/Kolibri-1-BF16` is ungated (Apache-2.0). The quantization script streams BF16 shards from Hugging Face with 1-shard background prefetching, quantizes all `384 × 50 = 19,200` routed experts natively on SM89 to **W4A16 NVFP4 Marlin**, keeps dense/attention/shared-expert/embedding weights in BF16 (router bias FP32), and evicts each BF16 shard as soon as it is no longer needed:

```bash
# Optional: set HF_TOKEN for faster Hugging Face downloads
# export HF_TOKEN=hf_...

python scripts/quantize_kolibri_nvfp4.py
```

- **Output location:** `checkpoints/kolibri-1-nvfp4-marlin/` (`44.40 GiB` across `globals.safetensors` + `layer_00.safetensors`..`layer_49.safetensors`, tokenizer/config metadata, and `quant_manifest.json`).
- **Duration:** ~20–25 minutes on a fast connection + RTX 4090 (`21.7 min` measured end-to-end).
- **Resumable:** If interrupted, re-running `python scripts/quantize_kolibri_nvfp4.py` skips existing `layer_XX.safetensors` files unless `--force` is passed.
- **Custom checkpoint path:** Pass `--out-dir /path/to/checkpoint` and export `KOLIBRI_MODEL_DIR=/path/to/checkpoint` (or pass `--model-dir` to runtime scripts).

### 3. Verify the Checkpoint

Run the integrity and GPU smoke test across all 50 layers:

```bash
python scripts/verify_nvfp4_checkpoint.py
```

> **Note:** On the very first execution, PyTorch JIT-compiles the custom CUDA/OpenMP extension (`csrc/build/kolibri_kernels_ext.so`) and FlashInfer kernels (~30–60 seconds once). Subsequent engine startups take ~3 seconds.

---

## Launching the Engine

### Option A: Single-Prompt CLI Generation

```bash
python scripts/generate.py \
  --prompt "Explain how Mixture-of-Experts offloading works on a single 24 GB GPU." \
  --reasoning-effort none \
  --max-new-tokens 256
```

Useful flags:
- `--reasoning-effort {none,low,medium,high}` — Controls Kolibri-1's built-in reasoning budget.
- `--max-seq-len 16384` (up to `100000`) — Controls preallocated KV cache length.
- `--persistent-slots 110` — Persistent routed expert slots kept in VRAM per layer.
- `--model-dir /path/to/ckpt` — Override checkpoint directory.

### Option B: OpenAI-Compatible API Server

Start the threaded HTTP/SSE server (`POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`, `POST /fast-mode`):

```bash
python scripts/web_server.py --host 0.0.0.0 --port 8080 --max-seq-len 100000 --persistent-slots 110
```

Once loaded, point any OpenAI-compatible client (e.g., Open WebUI, LibreChat, or the `openai` Python SDK with `base_url="http://localhost:8080/v1"`) at port `8080`, or test directly with `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kolibri-1 (W4A16-NVFP4 Marlin)",
    "messages": [{"role": "user", "content": "Give 3 bullet points on PCIe bandwidth in MoE serving."}],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": false
  }'
```

To maximize decode speed during serving, enable **Fast Decode** mode (pauses background `nvidia-smi` telemetry polling, disables Python GC during decode, uses O(1) incremental UTF-8 token decoding, and batches SSE flushes):

```bash
curl -X POST http://localhost:8080/fast-mode \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

Opening `http://127.0.0.1:8080` in a browser serves a built-in API status page with a `Fast Decode` toggle (or you can point `--web-dir /path/to/strata/serve/web` / `KOLIBRI_WEB_DIR` at a Strata frontend checkout).

### Option C: Deterministic Benchmark Suite

Reproduce the full 7-scenario benchmark on the frozen `HuggingFaceTB/smoltalk` (`train` → `test`) split and compare against the Stage 6 reference report:

```bash
python scripts/bench_suite.py \
  --out results/my_run.json \
  --compare results/omp_mempool_cacheops_graphab.json
```

You can also run a subset with `--scenarios cold calibrated prefill long_ctx`.

---

## Performance Tips & Troubleshooting

### CPU Governor (`powersave` Throttling)
During long multi-second generations on hybrid Intel/AMD CPUs, Linux's default `powersave` governor can downclock performance cores to ~800 MHz, slowing host expert gathering over P-cores. Setting the governor to `performance` avoids mid-generation slowdowns:

```bash
sudo cpupower frequency-set -g performance
```

### VRAM Headroom Tuning
With `--persistent-slots 110` and `--max-seq-len 100000`, Kolibri-1 uses ~20.61 GiB peak VRAM (~2.0 GiB driver headroom on a headless 24 GiB RTX 4090). If your GPU also drives a desktop session or fails with CUDA OOM at startup, lower `--persistent-slots` and/or `--max-seq-len`:

```bash
python scripts/web_server.py --persistent-slots 96 --max-seq-len 32768
```

---

## Architecture Overview

- **Precision & Quantization ([`src/nvfp4_quant.py`](src/nvfp4_quant.py)):** Attention (`w_qkv`, `w_o`), norms, shared experts (`sh_w13`, `sh_w2`), embeddings, LM head, and router weights (`w_gate` BF16, `gate_bias` FP32) stay in VRAM. All `384 × 50` routed experts are quantized natively on SM89 to **W4A16 NVFP4** (E2M1 nibbles + 16-element `float8_e4m3fn` block scales + per-expert FP32 `scale_2`) and pre-packed into Marlin tiled slabs (`layer_00.safetensors`..`layer_49.safetensors`).
- **Three-Tier Expert Cache & Native Host/CUDA Ops ([`src/expert_cache.py`](src/expert_cache.py), [`csrc/kolibriGatherExperts.cpp`](csrc/kolibriGatherExperts.cpp), [`csrc/kolibriCacheOps.cu`](csrc/kolibriCacheOps.cu)):**
  - `110` persistent LRU slots + `6` decode miss scratch slots per layer in VRAM (`gpu_caches`), double-buffered pinned host bounce slabs (`pinned_bounce`), and GPU staging slabs (`gpu_stage`) fed asynchronously via `dma_slabs_async` on a dedicated CUDA `copy_stream`.
  - Cold host slabs (`mmap`-backed safetensors) are pre-faulted on load with zero extra RAM allocation via `MADV_POPULATE_READ` (`MADV_WILLNEED` + 4 KB strided byte touch fallback).
  - `gather_experts_omp` releases the Python GIL, partitions all 6 Marlin slabs into 64-byte-aligned byte ranges across `#pragma omp parallel`, and pins worker threads exclusively to Linux performance cores (`cpuinfo_max_freq >= 95%` of max).
  - `promote_experts_cuda` promotes evicted cache misses across all 6 slabs (`int4`-vectorized) and updates `gpu_expert_to_slot` in a single CUDA kernel launch.
- **CUDA-Graphed Decode & Paged 100k KV Cache ([`src/engine.py`](src/engine.py), [`src/hybrid_moe.py`](src/hybrid_moe.py), [`src/kv_cache.py`](src/kv_cache.py), [`csrc/`](csrc/)):**
  - Alternates `40` Sliding-Window Attention layers (FlashInfer NeoX RoPE, `SLIDING_WINDOW=513`) and `10` Full-Attention RNoPE layers (`100k` max context) using NHD paged KV layout (`page_size=16`) and FlashInfer `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True, use_tensor_cores=True)`.
  - Captures per-layer **Pre-MoE CUDA Graph A** (`input_ln → attention_decode_step → fused_sandwich_norm → kolibri_route`) and **MoE CUDA Graph B** (`moe_align_block_size → Marlin GEMM1 → silu_and_mul → Marlin GEMM2 → sum_topk_expert_outputs`), sets `cudaMemPoolAttrReleaseThreshold = UINT64_MAX`, and reuses GEMM1 alignment indices for GEMM2.

## Repository Layout

```text
kolibri-4090/
├── assets/
│   └── banner.jpg
├── csrc/
│   ├── include/tensorrt_llm/...     # Minimal vendored TRT-LLM headers for standalone compilation
│   ├── kolibri_kernels_ext.cpp      # PyTorch C++/CUDA extension bindings & CUDA mempool tuning
│   ├── kolibriCacheOps.{h,cu}       # Async slab DMA, fused LRU promotion, NHD KV append kernels
│   ├── kolibriGatherExperts.{h,cpp} # P-core pinned OpenMP host slab gather
│   ├── kolibriPostMoeNormKernel.cu  # Fused post-MoE add + RMSNorm
│   ├── kolibriRouterKernel.{h,cu}   # Standalone CUDA sigmoid Top-6 router
│   └── kolibriSandwichNormKernel.cu # Fused attention residual add + pre-MoE RMSNorm
├── data/
│   └── smoltalk_bench_seed20261116_cal16.json
├── results/                         # Tracked regression progression JSONs (baseline -> Graph A+B)
├── scripts/
│   ├── quantize_kolibri_nvfp4.py    # Streaming BF16 -> W4A16 NVFP4 Marlin quantizer
│   ├── verify_nvfp4_checkpoint.py   # 50-layer integrity & GPU forward smoke checks
│   ├── generate.py                  # Single-prompt CLI generator
│   ├── web_server.py                # OpenAI-compatible HTTP/SSE server & status/Strata adapter
│   └── bench_suite.py               # Reproducible HF-backed benchmark & regression comparator
├── src/
│   ├── config.py                    # Kolibri-1 constants & env-overridable paths
│   ├── nvfp4_quant.py               # SM89 W4A16 NVFP4 quantizer & Marlin repacker
│   ├── kolibri_kernels.py           # JIT compiler wrapper & FlashInfer / custom kernel wrappers
│   ├── expert_cache.py              # Three-tier LRU expert cache & mmap pre-faulting
│   ├── hybrid_moe.py                # Hybrid MoE execution & MoE Graph B capture
│   ├── kv_cache.py                  # NHD paged SWA + Full KV cache & FlashInfer wrappers
│   ├── model_ops.py                 # Attention, RoPE, router, and shared expert ops
│   ├── engine.py                    # End-to-end 50-layer engine & Pre-MoE Graph A capture
│   └── bench_dataset.py             # Deterministic SmolTalk train/test sampler
├── OPTIMIZATION_LOG.md              # Detailed stage-by-stage optimization & nsys trace write-up
├── requirements.txt
└── README.md
```

---

## License & Acknowledgments

- **Base Model:** [Aleph-Alpha/Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) (Apache License 2.0).
- **Kernels & Runtime:** Built on PyTorch, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (Marlin NVFP4 & vendored Kolibri kernels), and [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
