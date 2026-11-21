# Kolibri-1 (RTX 4090) Iterative Optimization Log

Summary of the iterative performance optimizations, profiling findings, and engineering changes made to the standalone **Aleph-Alpha/Kolibri-1** W4A16 NVFP4 Marlin hybrid CPU/GPU inference engine in `kolibri-4090`.

---

## Executive Summary

Across six optimization stages—guided by page-fault instrumentation, CPU thread sweeps, and four Nsight Systems traces—single-stream decode throughput improved by **5.1×–7.9×** and prefill throughput by **3.5×–10.6×** while keeping peak VRAM at **20.58–20.61 GiB** (110 persistent expert slots/layer + 100k context KV cache) and preserving greedy token parity.

| Metric (`bench_suite.py`, `seed=20261116`) | Baseline | Final (Graph A+B) | Speedup |
|---|---:|---:|---:|
| `cold_decode` — Prefill (tok/s) | 18.41 | **194.56** | **10.57×** |
| `cold_decode` — Decode (tok/s) | 5.96 | **47.05** (`p50 50.5`) | **7.89×** |
| `calibrated_decode` — Prefill (tok/s) | 60.35 | **226.09** | **3.75×** |
| `calibrated_decode` — Decode (tok/s) | 9.99 | **51.09** (`p50 52.9`) | **5.11×** |
| `prefill_n256` (tok/s) | 93.86 | **381.06** | **4.06×** |
| `prefill_n1024` (tok/s) | 101.30 | **349.92** | **3.45×** |
| `long_ctx_16k` — Decode (tok/s) | 10.01 | **52.58** (`p50 54.4`) | **5.25×** |
| `long_ctx_64k` — Decode (tok/s) | 8.64 | **52.57** (`p50 53.5`) | **6.08×** |
| `long_ctx_100k` — Decode (tok/s) | 8.67 | **51.42** (`p50 53.9`) | **5.93×** |
| Peak VRAM (`cold` / `100k`) | 20.52 / 20.49 GiB | **20.61 / 20.58 GiB** | +0.09 GiB |

---

## Hardware & Model Configuration

- **Model:** `Aleph-Alpha/Kolibri-1` (`50` layers, `H=2560`, `NQ=48`, `NKV=4`, `GQA=12`, `HD=128`, `E=384` routed experts, `TOPK=6`, `I_MOE=512`, `I_SH=512`, 40 Sliding-Window Attention layers with `SLIDING_WINDOW=513` + 10 Full-RNoPE layers, `100k` max context, W4A16 NVFP4 Marlin routed experts, `110` persistent GPU slots/layer + `6` staging slots).
- **GPU:** NVIDIA GeForce RTX 4090 (24 GiB VRAM, SM 8.9, PCIe Gen 4 x16).
- **Host:** Linux `6.8.0-136-generic`, Intel Core Ultra 7 265K (20 cores: 8 P-cores `cpu0..cpu7` + 12 E-cores `cpu8..cpu19`, `scaling_governor=powersave`), 62 GiB RAM.
- **Software:** Python (`thesis` conda env), `torch 2.14.0+cu130`, `tensorrt_llm==1.3.0rc29`, `flashinfer-python==0.6.18`, `g++ 13.3.0`, `nsys 2024.6.2`.
- **Benchmark Suite:** `scripts/bench_suite.py` (`seed=20261116`, `slots=110`, `HuggingFaceTB/smoltalk` `train→test`, `decode_steps=64`, `long_decode_steps=32`).

---

## End-to-End Benchmark Progression

All numbers come directly from the saved JSON runs in `results/`:

| Stage | JSON Artifact | `cold_decode` Prefill / Decode | `calibrated_decode` Prefill / Decode | `prefill_n256` | `prefill_n1024` | `long_ctx_16k` | `long_ctx_64k` | `long_ctx_100k` | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0. Baseline** | `results/baseline.json` | 18.41 / 5.96 | 60.35 / 9.99 | 93.86 | 101.30 | 10.01 | 8.64 | 8.67 | 20.52 G |
| **1–2. mmap Pre-fault + P-Core OMP** | `results/omp_pcore.json` | 185.58 / 17.46 | 212.32 / 18.03 | 353.88 | 335.63 | 18.38 | 18.19 | 17.96 | 20.52 G |
| **3. + CUDA Mempool Threshold** | `results/omp_mempool.json` | 189.72 / 33.53 | 226.47 / 37.36 | 380.25 | 349.75 | 35.51 | 34.09 | 32.74 | 20.52 G |
| **4. + Fused Cache Ops (DMA/Promote)** | `results/omp_mempool_cacheops.json` | 199.04 / 40.38 | 231.89 / 41.19 | 374.12 | 349.68 | 42.63 | 39.53 | 39.83 | 20.52 G |
| **5. + Phase 1 (Align Reuse + Graph B)** | `results/omp_mempool_cacheops_graphb.json` | 200.72 / 47.08 | 215.86 / 48.85 | 380.76 | 350.25 | 49.62 | 48.27 | 48.77 | 20.52 G |
| **6. + Phase 2 (FlashInfer + Graph A)** | `results/omp_mempool_cacheops_graphab.json` | 194.56 / 47.05 | 226.09 / 51.09 | 381.06 | 349.92 | 52.58 | 52.57 | 51.42 | 20.61 G |

*(Decode p50 / p95 in Stage 6: `cold_decode` 50.5 / 33.1, `calibrated_decode` 52.9 / 38.0, `long_ctx_16k` 54.4 / 41.1, `long_ctx_64k` 53.5 / 45.4, `long_ctx_100k` 53.9 / 43.5 tok/s.)*

---

## Nsight Systems Trace Progression

Captured with `scratch/profile_nsys.py` (`prefill_n196` + 6 calibrated decode steps `decode_step_0..5`, NVTX-annotated):

| Trace (`results/*.nsys-rep`) | `decode_step_0..5` Total | `decode_step_5` | Key Bottleneck Eliminated |
|---|---:|---:|---|
| 1. `kolibri_decode.nsys-rep` (Stage 2: OMP) | 395.03 ms | 61.91 ms | Baseline trace: `cudaMallocAsync` burned **242.37 ms** across 700 calls (`22.15 ms` in `decode_step_5`) while total GPU kernel time was only `9.39 ms/step`. |
| 2. `kolibri_decode_mempool.nsys-rep` (Stage 3: Mempool) | 249.09 ms | 34.55 ms | `cudaMallocAsync` dropped **242.37 → 1.93 ms (125×)**; `marlin_gemm1_gemm2` dropped **173.69 → 36.45 ms**. |
| 3. `kolibri_decode_cacheops.nsys-rep` (Stage 4: CacheOps) | 198.42 ms | 30.24 ms | `promote_misses` dropped **27.74 → 6.39 ms (4.3×)**; `dma_h2d_async` dropped **19.90 → 6.47 ms (3.1×)**; `cudaMemcpyAsync` calls halved (**6,282 → 2,920**). |
| 4. `kolibri_decode_graphab.nsys-rep` (Stage 6: Graph A+B) | **144.26 ms** | **22.23 ms** | `pre_moe_graph_a` launch **57.39 → 3.27 ms (17.5×)** (`10.9 µs/layer`); `moe_graph_b` launch **35.45 → 3.68 ms (9.6×)** (`12.3 µs/layer`). |

---

## Stage-by-Stage Iterative Improvements

### Stage 0 — Baseline & Root-Cause Diagnosis
- **Observation:** Initial `bench_suite.py` (`results/baseline.json`) ran `cold_decode` at `18.41` prefill / `5.96` decode tok/s and `calibrated_decode` at `60.35` prefill / `9.99` decode tok/s, with large cold-vs-warm variance.
- **Root Causes Identified:**
  1. `safetensors.torch.load_file(..., device="cpu")` returns `mmap`-backed tensors where `.contiguous()` is a no-op (`same storage? True`). Cold host experts `[110:384]` (~31 GiB) stayed un-faulted at load time (`291,215` minor page faults in cold decode vs `819` in calibrated decode).
  2. PyTorch defaulted to `torch.get_num_threads() == 20` across the Ultra 7 265K's hybrid 8 P-cores + 12 E-cores under `powersave`, causing `gather_to_pinned` (`torch.index_select` × 6 slabs × 50 layers) to spike up to `82.9 ms/step` while GPU compute stayed steady at `~48–50 ms/step`.

### Stage 1 — Zero-Allocation `mmap` Pre-Faulting
- **Files:** `src/expert_cache.py`
- **What Changed:**
  - Added `_get_libc_madvise()`, `_strided_page_touch()`, and `_prefault_host_routed()`, called inside `HybridExpertCache._load_all()` (`prefault_host: bool = True`).
  - Page-aligns each tensor's `[addr, addr + nbytes)` down to `mmap.PAGESIZE` (safetensors `data_ptr()` is 8-byte aligned, not 4 KB page-aligned), invokes Linux `MADV_POPULATE_READ` (`22`), and falls back cleanly on `EINVAL`/`ENOSYS`/`EOPNOTSUPP`/non-Linux to `MADV_WILLNEED` + zero-copy `t.reshape(-1).view(torch.uint8)` 4 KB strided byte touch (plus `byte_view[-1]`).
- **Impact & Verification:**
  - Verified `0` minor/major page faults on `[110:384]` `index_select` across all 3 code paths with `data_ptr()` unchanged.
  - 50-layer cold minor page faults dropped ~71× (`291,215 → 4,095`); cold prefill jumped `18.41 → 53.74 tok/s`.

### Stage 2 — P-Core Pinned C++ OpenMP Expert Gather
- **Commit:** `6994f48` (`feat: implement C++ OpenMP expert gather with CPU core pinning`)
- **Files:** `csrc/kolibriGatherExperts.h`, `csrc/kolibriGatherExperts.cpp`, `csrc/kolibri_kernels_ext.cpp`, `src/kolibri_kernels.py`, `src/expert_cache.py`
- **What Changed:**
  - Sweeping `torch.set_num_threads(nt)` (`nt=1..20`) revealed a sharp performance cliff above 8 threads (the 8 P-cores: `nt=6` achieved `183.00` prefill / `18.19` decode vs `nt=20` at `62.15` prefill / `10.26` decode).
  - Implemented a native C++ OpenMP extension `gather_experts_omp` (compiled with `-fopenmp`):
    - Auto-detects P-cores from `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq` (`>= 95%` of max → `8` P-cores on Ultra 7 265K).
    - Releases the Python GIL, copies tiny slabs (`< 4 KB`) single-threaded, and partitions large Marlin slabs into 64-byte-aligned byte ranges across `#pragma omp parallel num_threads(eff_threads)` so even single-expert (`m=1`) decode misses parallelize across P-cores.
    - Saves/restores each worker thread's `pthread_setaffinity_np` mask without mutating `torch.get_num_threads()` or main-thread affinity.
- **Impact & Verification:**
  - Verified bit-exact `torch.equal` parity against `torch.index_select`.
  - `results/omp_pcore.json` vs `baseline.json`: `cold_decode` **185.58 / 17.46 tok/s** (`+908% / +193%`), `calibrated_decode` **212.32 / 18.03 tok/s** (`+252% / +80.5%`), `prefill_n256` **353.88 tok/s** (`+277%`).

### Stage 3 — CUDA Mempool Release Threshold Fix
- **Files:** `csrc/kolibri_kernels_ext.cpp`, `src/kolibri_kernels.py`, `src/hybrid_moe.py`
- **What Changed:**
  - The first Nsight Systems trace (`results/kolibri_decode.nsys-rep`) revealed `cudaMallocAsync_v11020` inside `torch.ops.trtllm.moe_align_block_size` (`cumsum_buffer`) was taking **242.37 ms** across 700 calls (`med=412.5 µs`) because CUDA's default `cudaMemPoolAttrReleaseThreshold == 0` trimmed cached blocks back to the OS driver on every layer's `tolist()` sync.
  - Added `configure_cuda_mempool(device_idx)` setting `cudaMemPoolAttrReleaseThreshold = UINT64_MAX`, invoked in `warmup_kernels()` and `HybridMarlinMoE.__init__()`.
- **Impact & Verification:**
  - `cudaMallocAsync` dropped **125×** (`242.37 → 1.93 ms`).
  - `results/omp_mempool.json` vs `omp_pcore.json`: `calibrated_decode` decode jumped **18.03 → 37.36 tok/s (+107.2%)**, `cold_decode` decode jumped **17.46 → 33.53 tok/s (+92.0%)**, with `0` extra VRAM.

### Stage 4 — Fused C++/CUDA Cache Operations (`dma_slabs_async` + `promote_experts_cuda`)
- **Commit:** `047709e` (`feat: add CUDA kernels for fused expert cache operations and asynchronous slab DMA`)
- **Files:** `csrc/kolibriCacheOps.h`, `csrc/kolibriCacheOps.cu`, `csrc/kolibri_kernels_ext.cpp`, `src/kolibri_kernels.py`, `src/expert_cache.py`
- **What Changed:**
  - `kolibri_decode_mempool.nsys-rep` showed `promote_misses` (`27.74 ms` / 6 steps) and `dma_h2d_async` (`19.90 ms` / 6 steps) were dominated by Python/PyTorch dispatch (`6 × m` `.copy_()` calls + `from_numpy` H2D overhead).
  - Implemented:
    - `dma_slabs_async`: single GIL-released C++ call issuing all 6 `cudaMemcpyAsync` H2D transfers + `cudaEventRecord` on `copy_stream`.
    - `promote_experts_cuda`: single fused `int4`-vectorized CUDA kernel copying all `m` misses × 6 slabs in VRAM and updating `gpu_expert_to_slot` LUT in-kernel via pass-by-value `PromoteParams` (zero temporary GPU/H2D allocations).
    - Pre-cached per-layer tensor lists and `copy_stream` / `copy_events` handles in `HybridExpertCache`.
- **Impact & Verification:**
  - Verified bit-exact via unit checks and `scripts/bench_step3_1_cache.py` (`m=6` CPU Gather + DMA: `0.549 ms` / `24.19 GB/s`).
  - `promote_misses` dropped **4.3×** (`27.74 → 6.39 ms`), `dma_h2d_async` dropped **3.1×** (`19.90 → 6.47 ms`).
  - `results/omp_mempool_cacheops.json`: `cold_decode` decode **40.38 tok/s (+20.4%)**, `calibrated_decode` decode **41.19 tok/s (+10.3%)**, `long_ctx_16k` **42.63 tok/s**.

### Stage 5 — Phase 1 CUDA Graphs: GEMM1 Align Reuse + Per-Layer MoE Graph B
- **Files:** `src/hybrid_moe.py`
- **What Changed:**
  - Eliminated the redundant second `moe_align_block_size` call for GEMM2 by reusing GEMM1's `s_ids1`, `e_ids1`, `_n_pad1` (100 → 50 calls/token; removed `_s_ids2`, `_e_ids2`, `_n_pad2`).
  - Captured per-layer **CUDA Graph B** (`_run_decode_moe_graphed`) for `n=1` decode covering `moe_align_block_size → GEMM1 → silu_and_mul → GEMM2 → sum_topk_expert_outputs` using static buffers (`_graph_moe_in`, `_graph_topk_w`, `_graph_slots`, `num_experts=cache_slots=116`) and a shared `torch.cuda.graph_pool_handle()`.
- **Impact & Verification:**
  - Bit-exact (`cos=1.000000, max_abs=0.0` on 100% hit in `scripts/bench_step3_2_hybrid_moe.py`; `EXACT` greedy tokens on `cold_decode`).
  - `results/omp_mempool_cacheops_graphb.json`: `cold_decode` decode **47.08 tok/s (+16.6%)**, `calibrated_decode` decode **48.85 tok/s (+18.6%)**, `long_ctx_100k` **48.77 tok/s**.

### Stage 6 — Phase 2 CUDA Graphs: NHD Paged KV Cache + FlashInfer Decode + Per-Layer Pre-MoE Graph A
- **Files:** `csrc/kolibriCacheOps.h`, `csrc/kolibriCacheOps.cu`, `csrc/kolibri_kernels_ext.cpp`, `src/kolibri_kernels.py`, `src/kv_cache.py`, `src/engine.py`, `src/hybrid_moe.py`
- **What Changed:**
  - Added `kv_append_decode_cuda` (`int4`-vectorized GPU KV slot writer reading device `pos_buf`, supporting both SWA `ring_mod=513` and Full `0`).
  - Switched `HybridKVCache` to NHD paged layout (`page_size=16`, zero-copy paged views) with two `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True, use_tensor_cores=True)` instances (`32 MiB` separate workspaces for SWA & Full; `use_tensor_cores=True` required for `GQA=12`, and `page_size=16` cuts 100k `.plan()` time ~20× from `1,249.5 µs` to `65.1 µs`). Planned once per step in `prepare_decode_step()` (SWA `.plan()` skipped once `pos >= 512`).
  - Captured per-layer **CUDA Graph A** (`_run_pre_moe_graphed`) in `KolibriHybridEngine` covering `input_ln → attention_decode_step → fused_sandwich_norm → kolibri_route`, aliasing `_dec_moe_in` / `_dec_wts` directly into `HybridMarlinMoE` static buffers (`data_ptr()` copy-skip guard), saving/restoring `_dec_cur` / `_dec_res` around capture (due to in-place `flashinfer_fused_add_rmsnorm`), and using a pinned scalar token buffer in `step()`.
- **Impact & Verification:**
  - `scripts/bench_step3_3_kv_cache.py` passed (`cos=0.999992 / 0.999985` across SWA=513 wraparound; `0.086 ms` SWA / `0.091 ms` Full at 16k).
  - 10-step eager vs graphed parity: `10/10` exact token match.
  - `results/omp_mempool_cacheops_graphab.json`: `calibrated_decode` **51.09 tok/s** (`p50 52.9`), `long_ctx_16k` **52.58 tok/s**, `long_ctx_64k` **52.57 tok/s (+8.9%)**, `long_ctx_100k` **51.42 tok/s** (`+0.09 GiB` VRAM).

### Additional Engineering — Web Server Fast Decode & Host Frequency Finding
- **Files:** `scripts/web_server.py`, `.gitignore`
- **What Changed:**
  - Added a `Fast Decode: OFF / ON` toggle button in the Strata web UI header + `GET/POST /fast-mode` endpoint: pauses `HardwareSampler` `nvidia-smi` polling + history loop during generation, disables GC during decode, switches to O(1) incremental `pending_ids` decoding, and coalesces SSE token flushes (4-token chunks, flushing immediately on `</think>` transitions).
  - Updated `.gitignore` to exclude Nsight Systems artifacts (`traces/`, `*.nsys-rep`, `*.sqlite`, `*.qdrep`, `*.qdstrm`).
- **Host Frequency Finding:** Diagnosed why sustained multi-second web runs plateaued below 32-step `bench_suite.py`: under Linux `scaling_governor=powersave`, 7 of the 8 P-cores downclocked to `~800 MHz` after ~2 seconds (`[800, 803, 1154, 808, 800, 800, 895, 5299] MHz`) while the RTX 4090 stayed at `2565 MHz / 114 W / PCIe Gen 4`.

---

## Current Bottleneck Profile (`decode_step_5`: 22.23 ms / token)

With Graph A and Graph B in place (`results/kolibri_decode_graphab.nsys-rep`), CPU kernel-launch overhead is largely eliminated (`10.9 µs/layer` for Graph A, `12.3 µs/layer` for Graph B). The remaining per-step budget is:

1. **GPU Pre-MoE + MoE compute + per-layer sync (`<moe_layer/tolist>`):** `10.74 ms/step` CPU wait for `pre_moe_graph_a` to finish so the CPU can read the `top-6` routed expert IDs.
2. **PCIe H2D expert transfers (`dma_h2d_async`):** `7.34 ms` GPU copy time (`168.1 MB/step` across cache misses; `0.96 ms/step` CPU launch + `0.47 ms/step` `wait_copy`).
3. **Host P-core slab gather (`gather_to_pinned_omp`):** `3.37 ms/step` (sensitive to Linux `powersave` P-core downclocking on sustained runs).
4. **VRAM cache promotion (`promote_misses`):** `0.79 ms/step`.

### Next High-Leverage Directions
- **Eliminate per-layer `tolist()` host sync** (or overlap next-layer routing / async prefetching with current-layer MoE compute).
- **Raise cache hit rate** (e.g., adaptive/LFU-weighted eviction or speculative layer-to-layer prefetching) to cut the `168 MB/step` PCIe H2D traffic.
- **Switch host CPU governor to `performance`** during sustained serving to prevent P-core downclocking to `800 MHz`.
