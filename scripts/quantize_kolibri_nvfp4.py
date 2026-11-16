#!/usr/bin/env python3
"""Phase 2: Streaming BF16 -> W4A16 NVFP4 Marlin quantization for full 50-layer Kolibri-1.

Downloads shards from `Aleph-Alpha/Kolibri-1-BF16` with 1-shard ahead prefetching,
quantizes all 384 routed experts per layer to NVFP4 and repacks them into Marlin
tiled format on GPU, preserves all non-expert layers in BF16 (router bias FP32),
writes one safetensors file per layer (`layer_00.safetensors` .. `layer_49.safetensors`)
plus `globals.safetensors`, and optionally evicts completed BF16 shards to keep
peak scratch disk usage minimal (~9-14 GB).
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import (  # noqa: E402
    E,
    HF_CACHE_DIR,
    HF_REPO_ID,
    I_MOE,
    NUM_LAYERS,
    PHASE2_MODEL_DIR,
    RESULTS_DIR,
)
from src.nvfp4_quant import quantize_and_pack_marlin_layer  # noqa: E402

AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
    "README.md",
]


def parse_layer_range(spec: str) -> List[int]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        start = int(a) if a else 0
        end = int(b) if b else NUM_LAYERS
        return list(range(start, end))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def atomic_save_safetensors(tensors: Dict[str, torch.Tensor], dst: Path) -> None:
    tmp = dst.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, dst)


def evict_hf_shard(symlink_path: Path, cache_root: Path) -> None:
    """Remove both snapshot symlink and backing blob inside `cache_root` to reclaim disk immediately."""
    try:
        blob_path = Path(os.path.realpath(symlink_path))
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink(missing_ok=True)
        if blob_path.exists() and blob_path.is_file():
            # Safety guard: only delete blobs strictly inside our scratch HF cache
            if cache_root.resolve() in blob_path.resolve().parents:
                blob_path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[warn] Could not evict shard {symlink_path}: {exc}")


class RollingShardReader:
    """Downloads shards with 1-shard background prefetch and evicts consumed shards."""

    def __init__(
        self,
        repo_id: str,
        cache_dir: Path,
        shard_sequence: List[str],
        shard_last_use: Dict[str, int],
        clean_bf16: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.shard_sequence = shard_sequence
        self.shard_last_use = shard_last_use
        self.clean_bf16 = clean_bf16

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._paths: Dict[str, Path] = {}
        self._handles: Dict[str, object] = {}
        self._evicted: set = set()
        self._prefetch_idx = 0

    def _download(self, filename: str) -> Path:
        p = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            cache_dir=str(self.cache_dir),
        )
        return Path(p)

    def _schedule_prefetch(self) -> None:
        while self._prefetch_idx < len(self.shard_sequence):
            cand = self.shard_sequence[self._prefetch_idx]
            self._prefetch_idx += 1
            if cand not in self._evicted and cand not in self._futures and cand not in self._paths:
                self._futures[cand] = self._executor.submit(self._download, cand)
                break

    def ensure_shards(self, shard_names: List[str]) -> None:
        for s in shard_names:
            if s not in self._paths:
                if s not in self._futures:
                    self._futures[s] = self._executor.submit(self._download, s)
                self._paths[s] = self._futures[s].result()
            if s not in self._handles:
                self._handles[s] = safe_open(str(self._paths[s]), framework="pt", device="cpu")
        self._schedule_prefetch()

    def get_tensor(self, shard_name: str, key: str) -> torch.Tensor:
        return self._handles[shard_name].get_tensor(key)

    def release_finished_before_or_at(self, step_idx: int) -> None:
        if not self.clean_bf16:
            return
        for s, last_step in list(self.shard_last_use.items()):
            if last_step <= step_idx and s not in self._evicted and s in self._paths:
                handle = self._handles.pop(s, None)
                if handle is not None:
                    del handle
                path = self._paths.pop(s)
                evict_hf_shard(path, self.cache_dir)
                self._evicted.add(s)

    def close(self) -> None:
        self._handles.clear()
        self._executor.shutdown(wait=True)


def extract_layer_tensors(
    layer_idx: int,
    weight_map: Dict[str, str],
    reader: RollingShardReader,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Assemble non-expert BF16 weights and routed expert [E, K_in, N_out] CPU tensors."""
    prefix = f"model.layers.{layer_idx}."
    needed_shards = sorted({
        shard for k, shard in weight_map.items() if k.startswith(prefix)
    })
    reader.ensure_shards(needed_shards)

    def get(suffix: str) -> torch.Tensor:
        key = prefix + suffix
        return reader.get_tensor(weight_map[key], key)

    non_expert = {
        "input_ln": get("input_layernorm.weight").to(compute_dtype).contiguous(),
        "post_attn_ln": get("post_attn_norm.weight").to(compute_dtype).contiguous(),
        "pre_moe_ln": get("post_attention_layernorm.weight").to(compute_dtype).contiguous(),
        "post_ffn_ln": get("post_ffn_norm.weight").to(compute_dtype).contiguous(),
        "q_norm": get("self_attn.q_norm.weight").to(compute_dtype).contiguous(),
        "k_norm": get("self_attn.k_norm.weight").to(compute_dtype).contiguous(),
        "w_qkv": torch.cat(
            [
                get("self_attn.q_proj.weight").t().to(compute_dtype),
                get("self_attn.k_proj.weight").t().to(compute_dtype),
                get("self_attn.v_proj.weight").t().to(compute_dtype),
            ],
            dim=-1,
        ).contiguous(),
        "w_o": get("self_attn.o_proj.weight").t().to(compute_dtype).contiguous(),
        "w_gate": get("mlp.gate.weight").t().to(compute_dtype).contiguous(),
        "gate_bias": get("moe.router.expert_bias").float().contiguous(),
        "sh_w13": torch.cat(
            [
                get("mlp.shared_experts.gate_proj.weight").t().to(compute_dtype),
                get("mlp.shared_experts.up_proj.weight").t().to(compute_dtype),
            ],
            dim=-1,
        ).contiguous(),
        "sh_w2": get("mlp.shared_experts.down_proj.weight").t().to(compute_dtype).contiguous(),
    }

    w13_list = []
    w2_list = []
    for e in range(E):
        g_t = get(f"mlp.experts.{e}.gate_proj.weight").t().to(compute_dtype)
        u_t = get(f"mlp.experts.{e}.up_proj.weight").t().to(compute_dtype)
        d_t = get(f"mlp.experts.{e}.down_proj.weight").t().to(compute_dtype)
        w13_list.append(torch.cat([g_t, u_t], dim=-1))
        w2_list.append(d_t)

    w13_hk = torch.stack(w13_list, dim=0).contiguous()  # [384, 2560, 1024]
    w2_hk = torch.stack(w2_list, dim=0).contiguous()    # [384, 512, 2560]
    return non_expert, w13_hk, w2_hk


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize full Kolibri-1-BF16 to Marlin W4A16_NVFP4.")
    parser.add_argument("--layers", type=str, default=f"0:{NUM_LAYERS}", help="Layer slice (e.g. 0:50 or 0,1)")
    parser.add_argument("--out-dir", type=Path, default=PHASE2_MODEL_DIR)
    parser.add_argument("--keep-bf16-shards", action="store_true", help="Keep downloaded BF16 shards in scratch cache")
    parser.add_argument("--force", action="store_true", help="Overwrite existing layer safetensors")
    args = parser.parse_args()

    target_layers = parse_layer_range(args.layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda:0")
    print(f"=== Kolibri-1 Phase 2: Streaming W4A16_NVFP4 Marlin Quantization ===")
    print(f"GPU: {torch.cuda.get_device_name(dev)} | Target layers: {target_layers[0]}..{target_layers[-1]} ({len(target_layers)} layers)")
    print(f"Output dir: {args.out_dir} | Clean BF16 shards: {not args.keep_bf16_shards}")

    # Copy tokenizer/config metadata directly into output checkpoint directory
    for fname in AUX_FILES:
        dst = args.out_dir / fname
        if not dst.exists() or args.force:
            src = hf_hub_download(HF_REPO_ID, fname, cache_dir=str(HF_CACHE_DIR))
            shutil.copy2(src, dst)

    idx_path = hf_hub_download(HF_REPO_ID, "model.safetensors.index.json", cache_dir=str(HF_CACHE_DIR))
    index_data = json.load(open(idx_path))
    weight_map: Dict[str, str] = index_data["weight_map"]

    globals_path = args.out_dir / "globals.safetensors"
    need_globals = args.force or not globals_path.exists()

    # Determine shard order & last-use index so consumed BF16 shards can be evicted cleanly
    layer_shards: Dict[int, List[str]] = {}
    shard_last_use: Dict[str, int] = {}
    ordered_shards: List[str] = []

    if need_globals:
        g_shards = sorted({
            weight_map[k]
            for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
        })
        for s in g_shards:
            if s not in ordered_shards:
                ordered_shards.append(s)
            shard_last_use[s] = -1

    for pos, li in enumerate(target_layers):
        layer_file = args.out_dir / f"layer_{li:02d}.safetensors"
        if layer_file.exists() and not args.force:
            continue
        prefix = f"model.layers.{li}."
        shards = sorted({s for k, s in weight_map.items() if k.startswith(prefix)})
        layer_shards[li] = shards
        for s in shards:
            if s not in ordered_shards:
                ordered_shards.append(s)
            shard_last_use[s] = pos

    reader = RollingShardReader(
        repo_id=HF_REPO_ID,
        cache_dir=HF_CACHE_DIR,
        shard_sequence=ordered_shards,
        shard_last_use=shard_last_use,
        clean_bf16=not args.keep_bf16_shards,
    )

    t_start = time.perf_counter()
    globals_info = {}
    layer_reports = []

    try:
        if need_globals:
            g_shards = sorted({
                weight_map[k]
                for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
            })
            reader.ensure_shards(g_shards)
            embed = reader.get_tensor(weight_map["model.embed_tokens.weight"], "model.embed_tokens.weight").to(torch.bfloat16).contiguous()
            norm = reader.get_tensor(weight_map["model.norm.weight"], "model.norm.weight").to(torch.bfloat16).contiguous()
            lm_head = reader.get_tensor(weight_map["lm_head.weight"], "lm_head.weight").contiguous()
            atomic_save_safetensors(
                {"embed_tokens": embed, "model_norm": norm, "lm_head": lm_head},
                globals_path,
            )
            globals_info = {
                "file": globals_path.name,
                "size_bytes": globals_path.stat().st_size,
                "embed_tokens": [list(embed.shape), str(embed.dtype)],
                "model_norm": [list(norm.shape), str(norm.dtype)],
                "lm_head": [list(lm_head.shape), str(lm_head.dtype)],
            }
            del embed, norm, lm_head
            print(f"[+] Saved {globals_path.name} ({globals_info['size_bytes'] / (1024**2):.2f} MiB)")
            reader.release_finished_before_or_at(-1)

        with torch.inference_mode():
            for pos, li in enumerate(target_layers):
                layer_file = args.out_dir / f"layer_{li:02d}.safetensors"
                if layer_file.exists() and not args.force:
                    print(f"[skip] Layer {li:02d} already exists at {layer_file.name}")
                    continue

                t0 = time.perf_counter()
                non_expert, w13_hk, w2_hk = extract_layer_tensors(li, weight_map, reader)
                t_extract = time.perf_counter() - t0

                t_q0 = time.perf_counter()
                marlin_gpu, qstats = quantize_and_pack_marlin_layer(
                    w13_hk, w2_hk, dev, chunk_size=64, compute_stats=True
                )
                torch.cuda.synchronize(dev)
                t_quant = time.perf_counter() - t_q0
                del w13_hk, w2_hk

                marlin_cpu = {k: v.cpu() for k, v in marlin_gpu.items()}
                del marlin_gpu
                torch.cuda.empty_cache()

                payload = {**non_expert, **marlin_cpu}
                t_s0 = time.perf_counter()
                atomic_save_safetensors(payload, layer_file)
                t_save = time.perf_counter() - t_s0

                size_bytes = layer_file.stat().st_size
                total_t = time.perf_counter() - t0
                print(
                    f"[{li+1:02d}/{NUM_LAYERS}] {layer_file.name}: {size_bytes / (1024**2):.2f} MiB | "
                    f"w13 cos={qstats['w13']['cosine']:.5f} w2 cos={qstats['w2']['cosine']:.5f} | "
                    f"extract={t_extract:.2f}s quant={t_quant:.2f}s save={t_save:.2f}s total={total_t:.2f}s"
                )
                layer_reports.append({
                    "layer": li,
                    "file": layer_file.name,
                    "size_bytes": size_bytes,
                    "w13": qstats["w13"],
                    "w2": qstats["w2"],
                    "timing_s": {
                        "extract": t_extract,
                        "quant_pack": t_quant,
                        "save": t_save,
                        "total": total_t,
                    },
                })
                del payload, non_expert, marlin_cpu
                reader.release_finished_before_or_at(pos)

    finally:
        reader.close()

    total_elapsed = time.perf_counter() - t_start

    # Build final manifest + Phase 2 report covering ALL saved files in out_dir
    saved_layers = sorted(args.out_dir.glob("layer_*.safetensors"))
    total_layer_bytes = sum(p.stat().st_size for p in saved_layers)
    globals_bytes = globals_path.stat().st_size if globals_path.exists() else 0
    total_ckpt_bytes = total_layer_bytes + globals_bytes

    manifest = {
        "source_repo": HF_REPO_ID,
        "quant_format": "W4A16_NVFP4_MARLIN",
        "routed_experts_only": True,
        "num_layers_present": len(saved_layers),
        "total_checkpoint_bytes": total_ckpt_bytes,
        "total_checkpoint_gib": total_ckpt_bytes / (1024 ** 3),
        "globals": globals_info or ({"file": "globals.safetensors", "size_bytes": globals_bytes} if globals_bytes else {}),
        "layers": [p.name for p in saved_layers],
    }
    (args.out_dir / "quant_manifest.json").write_text(json.dumps(manifest, indent=2))

    report_path = RESULTS_DIR / "phase2_quant_report.json"
    merged_layers: Dict[int, dict] = {}
    if report_path.exists() and not args.force:
        try:
            prev = json.loads(report_path.read_text())
            for item in prev.get("per_layer", []):
                merged_layers[int(item["layer"])] = item
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    for item in layer_reports:
        merged_layers[int(item["layer"])] = item
    all_layer_reports = [merged_layers[k] for k in sorted(merged_layers)]

    report = {
        "source_repo": HF_REPO_ID,
        "output_dir": str(args.out_dir),
        "elapsed_s": total_elapsed,
        "num_layers_processed_this_run": len(layer_reports),
        "num_layers_total_on_disk": len(saved_layers),
        "total_checkpoint_bytes": total_ckpt_bytes,
        "total_checkpoint_gib": total_ckpt_bytes / (1024 ** 3),
        "globals_bytes": globals_bytes,
        "per_layer": all_layer_reports,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(
        f"\n=== Phase 2 Complete in {total_elapsed / 60.0:.2f} min ===\n"
        f"Saved layers: {len(saved_layers)} | Total checkpoint size: {total_ckpt_bytes / (1024**3):.2f} GiB\n"
        f"Manifest: {args.out_dir / 'quant_manifest.json'}\n"
        f"Report:   {report_path}"
    )


if __name__ == "__main__":
    main()
