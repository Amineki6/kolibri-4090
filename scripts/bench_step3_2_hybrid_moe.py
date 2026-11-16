#!/usr/bin/env python3
"""Step 3.2 Verification & Benchmark: Hybrid Decode/Prefill MoE vs Full-GPU Marlin & Exact BF16."""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tensorrt_llm.quantization.utils.marlin_utils import marlin_make_workspace

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import H, PHASE1_CKPT_PATH, PHASE2_MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.expert_cache import HybridExpertCache  # noqa: E402
from src.hybrid_moe import HybridMarlinMoE  # noqa: E402
from src.model_ops import (  # noqa: E402
    exact_bf16_routed_moe,
    kolibri_attention,
    kolibri_route,
    make_rope_inv_freq,
    marlin_nvfp4_routed_moe,
    run_layers_forward,
    shared_expert,
)
from src.nvfp4_quant import metrics  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "step3_2_hybrid_moe.json"


def bench_cuda_fn(fn, warmup: int = 8, iters: int = 30) -> Dict[str, float]:
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
    }


def hybrid_2block_forward(
    x: torch.Tensor,
    cache: HybridExpertCache,
    hybrid_moe: HybridMarlinMoE,
    inv_freq: torch.Tensor,
    promote_misses: bool = False,
) -> torch.Tensor:
    residual = None
    cur = x
    for li in (0, 1):
        L = cache.gpu_non_experts[li]
        residual = cur if residual is None else cur + residual
        h = F.rms_norm(residual, (H,), L["input_ln"], 1e-6)
        attn_out = kolibri_attention(h, L, inv_freq, is_full_attention=False)
        residual = residual + F.rms_norm(attn_out, (H,), L["post_attn_ln"], 1e-6)
        moe_in = F.rms_norm(residual, (H,), L["pre_moe_ln"], 1e-6)
        ids, wts = kolibri_route(moe_in, L)
        routed, shared = hybrid_moe.forward_layer(
            moe_in,
            ids,
            wts,
            li,
            shared_fn=lambda m_in, layer=L: shared_expert(m_in, layer),
            promote_misses=promote_misses,
        )
        cur = F.rms_norm(routed + shared, (H,), L["post_ffn_ln"], 1e-6)
    return cur + residual


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    inv_freq = make_rope_inv_freq(dev)

    print("=== Step 3.2: HybridMarlinMoE Numerical Parity & Latency Verification ===")
    cache = HybridExpertCache(
        model_dir=PHASE2_MODEL_DIR,
        device=dev,
        persistent_slots_per_layer=110,
        decode_miss_slots=6,
        bounce_slots=274,
        stage_slots=384,
    )
    hybrid_moe = HybridMarlinMoE(cache, max_tokens=512)

    # Compute exact Phase 2 Full-GPU Marlin + Phase 1 BF16 references on Layers 0 & 1 cleanly
    # Note: we load full L0/L1 into temporary CPU tensors and move to GPU only for reference, OR run layer-by-layer!
    # Actually 2 layers full GPU is ~1.8 GiB, fits cleanly inside our 3.85 GiB free VRAM!
    p2_ws = marlin_make_workspace(dev, max_blocks_per_sm=4)
    p2_2blk = [
        {k: v.to(dev) for k, v in load_file(str(PHASE2_MODEL_DIR / f"layer_{i:02d}.safetensors"), device="cpu").items()}
        for i in (0, 1)
    ]

    parity_summary = {}
    with torch.inference_mode():
        # 1. Test Layer 0 MoE-only across all 3 execution paths:
        #    (a) Decode 0% miss (100% cache hit, N=1)
        #    (b) Decode 100% miss (6/6 cold miss via decode [K+6] slab, N=1)
        #    (c) Mixed Hit/Miss Prefill Split (N=64, N=256)
        gen = torch.Generator(device=dev).manual_seed(4242)
        x1 = torch.empty((1, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=gen)
        ids1, wts1 = kolibri_route(x1, p2_2blk[0])
        ref1 = marlin_nvfp4_routed_moe(x1, ids1, wts1, p2_2blk[0], p2_ws)

        # Force 100% hit on Layer 0 by seeding cache with ids1 + remaining experts
        active1 = [int(e) for e in torch.unique(ids1).tolist()]
        others = [e for e in range(384) if e not in active1]
        cache.seed_layer_cache(0, active1 + others[: cache.persistent_slots - len(active1)])
        out_hit, _ = hybrid_moe.forward_layer(x1, ids1, wts1, 0, promote_misses=False)
        m_hit = metrics(ref1, out_hit)
        print(f"[Parity] L0 Decode 100% Hit (N=1):        cos={m_hit['cosine']:.6f}, max_abs={m_hit['max_abs']:.6e}")
        assert torch.equal(ref1, out_hit), "100% Hit path should be bit-exact with Full-GPU Marlin!"

        # Force 100% cold miss (0 hits) on Layer 0 by seeding cache strictly with non-active experts
        cache.seed_layer_cache(0, others[: cache.persistent_slots])
        out_miss, _ = hybrid_moe.forward_layer(x1, ids1, wts1, 0, promote_misses=False)
        m_miss = metrics(ref1, out_miss)
        print(f"[Parity] L0 Decode 100% Cold Miss (N=1):  cos={m_miss['cosine']:.6f}, max_abs={m_miss['max_abs']:.6e}")
        assert m_miss["cosine"] >= 0.99999 and m_miss["max_abs"] < 1e-4, f"Unexpected miss drift: {m_miss}"

        # Test LRU promotion: running a second time should now be a 100% Hit!
        out_miss_prom, _ = hybrid_moe.forward_layer(x1, ids1, wts1, 0, promote_misses=True)
        prev_hits = hybrid_moe.stats.decode_0miss_calls
        out_warm, _ = hybrid_moe.forward_layer(x1, ids1, wts1, 0, promote_misses=True)
        assert hybrid_moe.stats.decode_0miss_calls == prev_hits + 1, "Expected 0-miss cache hit after LRU promotion!"
        m_warm = metrics(ref1, out_warm)
        assert m_warm["cosine"] >= 0.99999 and m_warm["max_abs"] < 1e-4, f"Unexpected warm drift: {m_warm}"
        print(f"[Parity] LRU promotion verified: 2nd decode call hit 0-miss fast path (cos={m_warm['cosine']:.6f})!")

        parity_summary["l0_decode_100pct_hit"] = m_hit
        parity_summary["l0_decode_100pct_miss"] = m_miss

        # Reset caches to [0..109] for realistic prefill split tests on N=64 and N=256
        cache.seed_layer_cache(0, list(range(cache.persistent_slots)))
        cache.seed_layer_cache(1, list(range(cache.persistent_slots)))

        for n in (64, 256):
            g_n = torch.Generator(device=dev).manual_seed(777 + n)
            xn = torch.empty((n, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=g_n)
            ids_n, wts_n = kolibri_route(xn, p2_2blk[0])
            ref_moe_n = marlin_nvfp4_routed_moe(xn, ids_n, wts_n, p2_2blk[0], p2_ws)
            hyb_moe_n, _ = hybrid_moe.forward_layer(xn, ids_n, wts_n, 0, promote_misses=False)
            m_moe = metrics(ref_moe_n, hyb_moe_n)

            # End-to-end 2-block Hybrid vs Full-GPU Phase 2 Marlin
            ref_2b, _, _ = run_layers_forward(
                xn, p2_2blk, inv_freq, lambda mx, mids, mwts, mL: marlin_nvfp4_routed_moe(mx, mids, mwts, mL, p2_ws)
            )
            hyb_2b = hybrid_2block_forward(xn, cache, hybrid_moe, inv_freq, promote_misses=False)
            m_2b = metrics(ref_2b, hyb_2b)

            print(
                f"[Parity] Prefill Split (N={n:3d}) L0 MoE cos={m_moe['cosine']:.6f} rel_l2={m_moe['rel_l2']:.6f} | "
                f"2-Block E2E cos={m_2b['cosine']:.6f} rel_l2={m_2b['rel_l2']:.6f}"
            )
            parity_summary[f"prefill_n{n}"] = {"l0_moe_vs_full_marlin": m_moe, "e2e_2blk_vs_full_marlin": m_2b}

        # Verify Hybrid 2-Block vs Exact Phase 1 BF16 checkpoint at N=64 (1 layer at a time in <=2.9 GiB headroom)
        del p2_2blk
        torch.cuda.empty_cache()

        bf16_raw = torch.load(PHASE1_CKPT_PATH, map_location="cpu")
        x64 = torch.empty((64, H), dtype=torch.bfloat16, device=dev).normal_(
            0.0, 1.0, generator=torch.Generator(device=dev).manual_seed(777 + 64)
        )
        cur_ref, res_ref = x64, None
        for li in (0, 1):
            L = cache.gpu_non_experts[li]
            res_ref = cur_ref if res_ref is None else cur_ref + res_ref
            h = F.rms_norm(res_ref, (H,), L["input_ln"], 1e-6)
            attn_out = kolibri_attention(h, L, inv_freq, is_full_attention=False)
            res_ref = res_ref + F.rms_norm(attn_out, (H,), L["post_attn_ln"], 1e-6)
            moe_in = F.rms_norm(res_ref, (H,), L["pre_moe_ln"], 1e-6)
            ids, wts = kolibri_route(moe_in, L)
            w13_nk = bf16_raw[li]["w13"].transpose(1, 2).contiguous().to(dev)
            w2_nk = bf16_raw[li]["w2"].transpose(1, 2).contiguous().to(dev)
            routed = exact_bf16_routed_moe(moe_in, ids, wts, w13_nk, w2_nk)
            del w13_nk, w2_nk
            torch.cuda.empty_cache()
            sh = shared_expert(moe_in, L)
            cur_ref = F.rms_norm(routed + sh, (H,), L["post_ffn_ln"], 1e-6)
        ref_bf16_64 = cur_ref + res_ref
        del bf16_raw

        hyb_64 = hybrid_2block_forward(x64, cache, hybrid_moe, inv_freq, promote_misses=False)
        m_bf16 = metrics(ref_bf16_64, hyb_64)
        print(f"[Parity] Hybrid 2-Block vs Exact Phase1 BF16 (N=64): cos={m_bf16['cosine']:.6f}, rel_l2={m_bf16['rel_l2']:.6f}")
        parity_summary["hybrid_vs_exact_bf16_n64"] = m_bf16

        # 2. Latency Benchmarks across scenarios (L0 MoE+Shared & 2-Block)
        print("\n[Latency Benchmarks]")
        bench_results = {}

        # A. Decode N=1 (100% cache hit)
        cache.seed_layer_cache(0, active1 + others[: cache.persistent_slots - len(active1)])
        t_dec_hit = bench_cuda_fn(
            lambda: hybrid_moe.forward_layer(
                x1, ids1, wts1, 0, shared_fn=lambda m_in: shared_expert(m_in, cache.gpu_non_experts[0]), promote_misses=False
            ),
            warmup=10,
            iters=50,
        )
        print(f"  L0 MoE+Shared Decode N=1 (0 misses / 100% hit): {t_dec_hit['median_ms']:.4f} ms")
        bench_results["l0_decode_0miss"] = t_dec_hit

        # B. Decode N=1 (100% cold miss: 6/6 experts transferred over PCIe)
        cache.seed_layer_cache(0, others[: cache.persistent_slots])
        t_dec_miss = bench_cuda_fn(
            lambda: hybrid_moe.forward_layer(
                x1, ids1, wts1, 0, shared_fn=lambda m_in: shared_expert(m_in, cache.gpu_non_experts[0]), promote_misses=False
            ),
            warmup=10,
            iters=40,
        )
        print(f"  L0 MoE+Shared Decode N=1 (6 misses / 0% hit):   {t_dec_miss['median_ms']:.4f} ms")
        bench_results["l0_decode_6miss"] = t_dec_miss

        # C. Prefill Hit/Miss Split (N=64, N=256)
        cache.seed_layer_cache(0, list(range(cache.persistent_slots)))
        cache.seed_layer_cache(1, list(range(cache.persistent_slots)))
        for n in (64, 256):
            g_n = torch.Generator(device=dev).manual_seed(777 + n)
            xn = torch.empty((n, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=g_n)
            ids_n, wts_n = kolibri_route(xn, cache.gpu_non_experts[0])
            hybrid_moe.reset_stats()
            t_layer = bench_cuda_fn(
                lambda xn=xn, ids_n=ids_n, wts_n=wts_n: hybrid_moe.forward_layer(
                    xn, ids_n, wts_n, 0, shared_fn=lambda m_in: shared_expert(m_in, cache.gpu_non_experts[0]), promote_misses=False
                ),
                warmup=5,
                iters=20,
            )
            hit_pct = hybrid_moe.stats.hit_rate * 100.0
            t_2blk = bench_cuda_fn(
                lambda xn=xn: hybrid_2block_forward(xn, cache, hybrid_moe, inv_freq, promote_misses=False),
                warmup=4,
                iters=15,
            )
            print(
                f"  Prefill N={n:3d} (cache hit={hit_pct:4.1f}%): L0 MoE+Shared={t_layer['median_ms']:.2f} ms | "
                f"2-Block E2E={t_2blk['median_ms']:.2f} ms ({2 * n / (t_2blk['median_ms'] / 1e3):.0f} layer-tok/s)"
            )
            bench_results[f"prefill_n{n}"] = {
                "cache_hit_pct": hit_pct,
                "l0_moe_shared_ms": t_layer,
                "e2e_2blk_ms": t_2blk,
            }

        # D. Warmed-Cache Prefill N=256 (seed top-110 most frequent experts for L0 & L1)
        g256 = torch.Generator(device=dev).manual_seed(777 + 256)
        x256 = torch.empty((256, H), dtype=torch.bfloat16, device=dev).normal_(0.0, 1.0, generator=g256)
        ids256_0, wts256_0 = kolibri_route(x256, cache.gpu_non_experts[0])
        counts0 = torch.bincount(ids256_0.reshape(-1), minlength=384)
        top110_0 = torch.topk(counts0, k=cache.persistent_slots).indices.tolist()
        cache.seed_layer_cache(0, top110_0)
        # Also seed L1 using routed ids after L0
        h0 = F.rms_norm(x256, (H,), cache.gpu_non_experts[0]["input_ln"], 1e-6)
        a0 = kolibri_attention(h0, cache.gpu_non_experts[0], inv_freq, is_full_attention=False)
        r0 = x256 + F.rms_norm(a0, (H,), cache.gpu_non_experts[0]["post_attn_ln"], 1e-6)
        m0 = F.rms_norm(r0, (H,), cache.gpu_non_experts[0]["pre_moe_ln"], 1e-6)
        ro0, sh0 = hybrid_moe.forward_layer(m0, ids256_0, wts256_0, 0, shared_fn=lambda m: shared_expert(m, cache.gpu_non_experts[0]), promote_misses=False)
        x_l1 = r0 + F.rms_norm(ro0 + sh0, (H,), cache.gpu_non_experts[0]["post_ffn_ln"], 1e-6)
        m1 = F.rms_norm(x_l1 + F.rms_norm(kolibri_attention(F.rms_norm(x_l1, (H,), cache.gpu_non_experts[1]["input_ln"], 1e-6), cache.gpu_non_experts[1], inv_freq, False), (H,), cache.gpu_non_experts[1]["post_attn_ln"], 1e-6), (H,), cache.gpu_non_experts[1]["pre_moe_ln"], 1e-6)
        ids256_1, _ = kolibri_route(m1, cache.gpu_non_experts[1])
        counts1 = torch.bincount(ids256_1.reshape(-1), minlength=384)
        cache.seed_layer_cache(1, torch.topk(counts1, k=cache.persistent_slots).indices.tolist())

        hybrid_moe.reset_stats()
        t_layer_warm = bench_cuda_fn(
            lambda: hybrid_moe.forward_layer(
                x256, ids256_0, wts256_0, 0, shared_fn=lambda m_in: shared_expert(m_in, cache.gpu_non_experts[0]), promote_misses=False
            ),
            warmup=5,
            iters=20,
        )
        warm_hit_pct = hybrid_moe.stats.hit_rate * 100.0
        t_2blk_warm = bench_cuda_fn(
            lambda: hybrid_2block_forward(x256, cache, hybrid_moe, inv_freq, promote_misses=False),
            warmup=4,
            iters=15,
        )
        print(
            f"  Prefill N=256 (warmed hit={warm_hit_pct:4.1f}%): L0 MoE+Shared={t_layer_warm['median_ms']:.2f} ms | "
            f"2-Block E2E={t_2blk_warm['median_ms']:.2f} ms ({2 * 256 / (t_2blk_warm['median_ms'] / 1e3):.0f} layer-tok/s)"
        )
        bench_results["prefill_n256_warmed"] = {
            "cache_hit_pct": warm_hit_pct,
            "l0_moe_shared_ms": t_layer_warm,
            "e2e_2blk_ms": t_2blk_warm,
        }

    RESULTS_PATH.write_text(
        json.dumps({"parity": parity_summary, "benchmarks": bench_results}, indent=2)
    )
    print(f"\nSaved Step 3.2 results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
