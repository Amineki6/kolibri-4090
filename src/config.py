"""Kolibri-1 architecture constants and serving directory layout."""

import os
from pathlib import Path
from typing import Optional

SERVING_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = SERVING_ROOT / "checkpoints"
RESULTS_DIR = SERVING_ROOT / "results"
SCRATCH_DIR = SERVING_ROOT / "scratch"
HF_CACHE_DIR = SCRATCH_DIR / "hf_cache"

_PHASE1_CKPT_ENV = os.environ.get("KOLIBRI_PHASE1_CKPT")
PHASE1_CKPT_PATH: Optional[Path] = Path(_PHASE1_CKPT_ENV) if _PHASE1_CKPT_ENV else None
PHASE2_MODEL_DIR = Path(
    os.environ.get("KOLIBRI_MODEL_DIR", str(CHECKPOINTS_DIR / "kolibri-1-nvfp4-marlin"))
)

HF_REPO_ID = "Aleph-Alpha/Kolibri-1-BF16"

# Kolibri-1 architecture dimensions (verified from config.json & model.safetensors.index.json)
H = 2560
NQ = 48
NKV = 4
HD = 128
E = 384
TOPK = 6
I_MOE = 512
I_SH = 512
NUM_LAYERS = 50
VOCAB_SIZE = 128000
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
SLIDING_WINDOW = 513
MOE_BLOCK_SIZE = 16


def is_full_attention_layer(layer_idx: int) -> bool:
    """Kolibri-1 alternates 4 sliding-window layers + 1 full-attention RNoPE layer every 5 blocks."""
    return (layer_idx + 1) % 5 == 0
