#!/usr/bin/env python3
"""Phase 1: Kolibri-1 2-Block W4A16 NVFP4 Quantization & Marlin Benchmark."""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tensorrt_llm.quantization.utils.marlin_utils import marlin_make_workspace

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import E, H, I_MOE, PHASE1_CKPT_PATH, PROFILING_BENCH_DIR, RESULTS_DIR, TOPK  # noqa: E402
from src.model_ops import (  # noqa: E402
    exact_bf16_routed_moe,
    kolibri_attention,
    kolibri_route,
    make_rope_inv_freq,
    marlin_nvfp4_routed_moe,
    run_layers_forward,
    shared_expert,
)
from src.nvfp4_quant import metrics, quantize_and_pack_marlin_layer  # noqa: E402

if PROFILING_BENCH_DIR not in sys.path:
    sys.path.insert(0, PROFILING_BENCH_DIR)
import cutlass_grouped_moe as cutlass_moe  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "phase1_results.json"


def reset_and_measure_vram_mb(dev: torch.device) -> float:
    torch.cuda.synchronize(dev)
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def layer_expert_bytes_bf16(L: Dict[str, torch.Tensor]) -> int:
    if "w13" in L:
        return tensor_bytes(L["w13"]) + tensor_bytes(L["w2"])
    return tensor_bytes(L["w31_col"]) + tensor_bytes(L["w2_col"])


def layer_non_expert_bytes(L: Dict[str, torch.Tensor]) -> int:
    skip = {
        "w13", "w2", "w31_col", "w2_col",
        "w13_marlin", "w13_scale_marlin", "w13_gs_marlin",
        "w2_marlin", "w2_scale_marlin", "w2_gs_marlin",
    }
    return sum(tensor_bytes(v) for k, v in L.items() if k not in skip and torch.is_tensor(v))


def layer_expert_bytes_marlin(L: Dict[str, torch.Tensor]) -> int:
    keys = (
        "w13_marlin", "w13_scale_marlin", "w13_gs_marlin",
        "w2_marlin", "w2_scale_marlin", "w2_gs_marlin",
    )
    return sum(tensor_bytes(L[k]) for k in keys)


def cutlass_bf16_routed_moe(
    moe_in: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    L: Dict[str, torch.Tensor],
) -> torch.Tensor:
    n = moe_in.shape[0]
    flat_ids = topk_ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[order]
    src_tok = order // TOPK

    permuted = moe_in[src_tok].contiguous()
    counts = torch.bincount(sorted_ids.to(torch.int64), minlength=E)
    offsets = torch.empty(E + 1, dtype=torch.int64, device=moe_in.device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])

    expert_out = cutlass_moe.cutlass_grouped_moe_gemm(
        permuted, offsets, L["w31_col"], L["w2_col"]
    )
    weighted = expert_out.float() * topk_weights.reshape(-1)[order].unsqueeze(-1)
    routed = torch.zeros((n, H), dtype=torch.float32, device=moe_in.device)
    routed.index_add_(0, src_tok, weighted)
    return routed.to(torch.bfloat16)


def benchmark_fn(fn, warmup: int = 10, iters: int = 40) -> Dict[str, float]:
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
        "min_ms": float(np.min(ms)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    inv_freq = make_rope_inv_freq(dev)

    print("=== Kolibri-1 Phase 1: W4A16 NVFP4 Marlin 2-Block Validation ===")
    print(f"GPU: {gpu_name} | Checkpoint: {PHASE1_CKPT_PATH}")

    raw_layers = torch.load(PHASE1_CKPT_PATH, map_location="cpu")
    layers = [{k: v.to(dev) for k, v in layer.items()} for layer in raw_layers]
    del raw_layers

    seq_lens_bench = [1, 8, 16, 64, 256, 512]
    seq_lens_acc = [1, 16, 64, 256, 512]
    test_inputs: Dict[int, torch.Tensor] = {}
    for n in sorted(set(seq_lens_bench + seq_lens_acc)):
        gen = torch.Generator(device=dev).manual_seed(2026 + n)
        test_inputs[n] = torch.empty((n, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=gen)

    bf16_expert_mb = sum(layer_expert_bytes_bf16(L) for L in layers) / (1024.0 * 1024.0)
    non_expert_mb = sum(layer_non_expert_bytes(L) for L in layers) / (1024.0 * 1024.0)

    ref_outputs: Dict[int, Dict[str, object]] = {}
    with torch.inference_mode():
        w13_nk_refs = [L["w13"].transpose(1, 2).contiguous() for L in layers]
        w2_nk_refs = [L["w2"].transpose(1, 2).contiguous() for L in layers]
        layer_idx_by_id = {id(L): idx for idx, L in enumerate(layers)}

        for n in seq_lens_acc:
            x = test_inputs[n]
            out, ids_list, routed_list = run_layers_forward(
                x,
                layers,
                inv_freq,
                lambda mx, mids, mwts, mL: exact_bf16_routed_moe(
                    mx, mids, mwts,
                    w13_nk_refs[layer_idx_by_id[id(mL)]],
                    w2_nk_refs[layer_idx_by_id[id(mL)]],
                ),
            )
            # Capture exact L0 MoE input + routing for isolated L0 MoE error comparison
            res0 = x
            h0 = F.rms_norm(res0, (H,), layers[0]["input_ln"], 1e-6)
            a0 = kolibri_attention(h0, layers[0], inv_freq, is_full_attention=False)
            res_mid0 = res0 + F.rms_norm(a0, (H,), layers[0]["post_attn_ln"], 1e-6)
            moe_in0 = F.rms_norm(res_mid0, (H,), layers[0]["pre_moe_ln"], 1e-6)
            ids0, wts0 = kolibri_route(moe_in0, layers[0])
            ref_outputs[n] = {
                "out_2block": out.cpu(),
                "l0_moe_in": moe_in0.cpu(),
                "l0_ids": ids0.cpu(),
                "l0_wts": wts0.cpu(),
                "l0_routed": routed_list[0].cpu(),
                "l1_ids": ids_list[1].cpu(),
            }
        del w13_nk_refs, w2_nk_refs

    for L in layers:
        cutlass_moe.prepare_cutlass_moe_weights_inplace(L, inter_size=I_MOE)
    cutlass_vram_mb = reset_and_measure_vram_mb(dev)

    bf16_moe_timings: Dict[int, Dict[str, float]] = {}
    bf16_2blk_timings: Dict[int, Dict[str, float]] = {}
    with torch.inference_mode():
        cutlass_check_out, _, _ = run_layers_forward(
            test_inputs[64], layers, inv_freq, cutlass_bf16_routed_moe
        )
        cutlass_vs_exact = metrics(ref_outputs[64]["out_2block"].to(dev), cutlass_check_out)
        print(f"CUTLASS BF16 vs Exact BF16 (N=64, 2-block): cos={cutlass_vs_exact['cosine']:.6f}, rel_l2={cutlass_vs_exact['rel_l2']:.6f}")

        for n in seq_lens_bench:
            x = test_inputs[n]
            ids, wts = kolibri_route(x, layers[0])
            bf16_moe_timings[n] = benchmark_fn(
                lambda x=x, ids=ids, wts=wts: cutlass_bf16_routed_moe(x, ids, wts, layers[0])
            )
            bf16_2blk_timings[n] = benchmark_fn(
                lambda x=x: run_layers_forward(x, layers, inv_freq, cutlass_bf16_routed_moe)
            )

    for L in layers:
        del L["w31_col"], L["w2_col"]
    reset_and_measure_vram_mb(dev)

    raw_layers = torch.load(PHASE1_CKPT_PATH, map_location="cpu")
    weight_quant_stats = []
    t_q0 = time.perf_counter()
    with torch.inference_mode():
        for idx, (L, raw_L) in enumerate(zip(layers, raw_layers)):
            marlin_dict, stats = quantize_and_pack_marlin_layer(
                raw_L["w13"], raw_L["w2"], dev, chunk_size=64, compute_stats=True
            )
            L.update(marlin_dict)
            weight_quant_stats.append({"layer": idx, **stats})

    quant_elapsed_s = time.perf_counter() - t_q0
    del raw_layers
    marlin_vram_mb = reset_and_measure_vram_mb(dev)
    marlin_expert_mb = sum(layer_expert_bytes_marlin(L) for L in layers) / (1024.0 * 1024.0)

    workspace = marlin_make_workspace(dev, max_blocks_per_sm=4)
    accuracy_results: Dict[int, Dict[str, object]] = {}
    marlin_moe_timings: Dict[int, Dict[str, float]] = {}
    marlin_2blk_timings: Dict[int, Dict[str, float]] = {}

    with torch.inference_mode():
        for n in seq_lens_acc:
            ref = ref_outputs[n]
            l0_marlin = marlin_nvfp4_routed_moe(
                ref["l0_moe_in"].to(dev),
                ref["l0_ids"].to(dev),
                ref["l0_wts"].to(dev),
                layers[0],
                workspace,
            )
            out_2blk, ids_list, _ = run_layers_forward(
                test_inputs[n],
                layers,
                inv_freq,
                lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, workspace),
            )
            ref_l1_ids = ref["l1_ids"].cpu().numpy()
            pred_l1_ids = ids_list[1].cpu().numpy()
            overlaps = [
                len(set(ref_l1_ids[t].tolist()) & set(pred_l1_ids[t].tolist())) / TOPK
                for t in range(n)
            ]
            accuracy_results[n] = {
                "l0_routed_moe": metrics(ref["l0_routed"].to(dev), l0_marlin),
                "end_to_end_2block": metrics(ref["out_2block"].to(dev), out_2blk),
                "l1_router_topk_overlap": float(np.mean(overlaps)),
            }

        for n in seq_lens_bench:
            x = test_inputs[n]
            ids, wts = kolibri_route(x, layers[0])
            marlin_moe_timings[n] = benchmark_fn(
                lambda x=x, ids=ids, wts=wts: marlin_nvfp4_routed_moe(x, ids, wts, layers[0], workspace)
            )
            marlin_2blk_timings[n] = benchmark_fn(
                lambda x=x: run_layers_forward(
                    x, layers, inv_freq, lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, workspace)
                )
            )

    print(f"\nQuantize + Marlin repack time (2 blocks): {quant_elapsed_s:.2f} s")
    print(f"Expert memory: {bf16_expert_mb:.2f} MiB (BF16) -> {marlin_expert_mb:.2f} MiB (Marlin NVFP4)")
    report = {
        "gpu": gpu_name,
        "checkpoint": str(PHASE1_CKPT_PATH),
        "quant_pack_time_s": quant_elapsed_s,
        "memory_mib": {
            "non_expert_2block": non_expert_mb,
            "experts_bf16_2block": bf16_expert_mb,
            "experts_marlin_nvfp4_2block": marlin_expert_mb,
            "resident_cuda_cutlass_bf16": cutlass_vram_mb,
            "resident_cuda_marlin_nvfp4": marlin_vram_mb,
            "expert_compression_ratio": bf16_expert_mb / marlin_expert_mb,
        },
        "weight_quant_stats": weight_quant_stats,
        "cutlass_vs_exact_bf16_n64": cutlass_vs_exact,
        "accuracy": {str(k): v for k, v in accuracy_results.items()},
        "timings_moe_only_ms": {
            str(n): {"bf16_cutlass": bf16_moe_timings[n], "nvfp4_marlin": marlin_moe_timings[n]}
            for n in seq_lens_bench
        },
        "timings_2block_ms": {
            str(n): {"bf16_cutlass": bf16_2blk_timings[n], "nvfp4_marlin": marlin_2blk_timings[n]}
            for n in seq_lens_bench
        },
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
