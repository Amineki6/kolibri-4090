"""Step 3.1: Three-tier host/VRAM expert cache, pinned DMA bounce buffers, and LRU/LFU tracker."""

import ctypes
import errno
import math
import mmap
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file

from .config import E, NUM_LAYERS, PHASE2_MODEL_DIR, TOPK
from .kolibri_kernels import gather_experts_omp

_MADV_WILLNEED: int = getattr(mmap, "MADV_WILLNEED", 3)
_MADV_POPULATE_READ: int = getattr(mmap, "MADV_POPULATE_READ", 22)
_TOUCH_STRIDE_BYTES: int = 4096
_UNSET = object()

NON_EXPERT_KEYS = (
    "input_ln",
    "post_attn_ln",
    "pre_moe_ln",
    "post_ffn_ln",
    "q_norm",
    "k_norm",
    "w_qkv",
    "w_o",
    "w_gate",
    "gate_bias",
    "sh_w13",
    "sh_w2",
)

# (key_name, per_expert_shape, dtype)
MARLIN_TENSOR_SPECS: Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...] = (
    ("w13_marlin", (160, 1024), torch.int64),
    ("w13_scale_marlin", (160, 256), torch.int32),
    ("w13_gs_marlin", (), torch.bfloat16),
    ("w2_marlin", (32, 2560), torch.int64),
    ("w2_scale_marlin", (32, 640), torch.int32),
    ("w2_gs_marlin", (), torch.bfloat16),
)

MARLIN_KEYS = tuple(spec[0] for spec in MARLIN_TENSOR_SPECS)
BYTES_PER_EXPERT: int = sum(
    math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    for _, shape, dtype in MARLIN_TENSOR_SPECS
)


@dataclass
class CacheMemoryStats:
    load_time_s: float
    globals_mib: float
    non_experts_mib: float
    gpu_cache_mib: float
    gpu_stage_mib: float
    total_gpu_allocated_mib: float
    total_gpu_reserved_mib: float
    host_routed_gib: float
    host_pinned_bounce_gib: float


def _alloc_marlin_slab(
    num_slots: int, device: torch.device, pin_memory: bool = False
) -> Dict[str, torch.Tensor]:
    slab: Dict[str, torch.Tensor] = {}
    for key, shape_tail, dtype in MARLIN_TENSOR_SPECS:
        t = torch.empty((num_slots, *shape_tail), dtype=dtype, device=device)
        if pin_memory and device.type == "cpu":
            t = t.pin_memory()
        slab[key] = t
    return slab


def _slab_bytes(slab: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in slab.values())


def _get_libc_madvise() -> Optional[Callable[[int, int, int], int]]:
    """Return ctypes handle to `libc.madvise`, or `None` if unsupported."""
    if sys.platform == "win32":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise_fn = libc.madvise
        madvise_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise_fn.restype = ctypes.c_int
        return madvise_fn
    except (OSError, AttributeError):
        return None


def _strided_page_touch(t: torch.Tensor, stride_bytes: int = _TOUCH_STRIDE_BYTES) -> None:
    """Fault all pages of a contiguous CPU tensor via zero-copy 4 KB strided reads."""
    byte_view = t.reshape(-1).view(torch.uint8)
    _ = int(byte_view[::stride_bytes].sum().item()) + int(byte_view[-1].item())


def _prefault_host_routed(
    host_routed: Sequence[Dict[str, torch.Tensor]],
    *,
    _madvise_fn: object = _UNSET,
    _allow_populate_read: Optional[bool] = None,
) -> str:
    """Pre-fault mmap-backed CPU expert tensors with zero extra allocation.

    Attempts Linux `MADV_POPULATE_READ` first, falling back to `MADV_WILLNEED`
    plus a 4 KB strided byte touch on `EINVAL`/`ENOSYS`/`EOPNOTSUPP` or non-Linux OSes.
    """
    madvise_fn = _get_libc_madvise() if _madvise_fn is _UNSET else _madvise_fn
    page_size = mmap.PAGESIZE
    use_populate_read = (
        sys.platform.startswith("linux")
        if _allow_populate_read is None
        else _allow_populate_read
    ) and (madvise_fn is not None)

    used_populate_read = False
    used_fallback = False

    for slab in host_routed:
        for t in slab.values():
            if t.device.type != "cpu" or t.numel() == 0:
                continue
            if not t.is_contiguous():
                t = t.contiguous()

            addr = t.data_ptr()
            nbytes = t.numel() * t.element_size()
            aligned_addr = addr & ~(page_size - 1)
            aligned_len = (addr + nbytes) - aligned_addr

            if use_populate_read:
                ctypes.set_errno(0)
                rc = madvise_fn(aligned_addr, aligned_len, _MADV_POPULATE_READ)  # type: ignore[operator]
                if rc == 0:
                    used_populate_read = True
                    continue
                if ctypes.get_errno() in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    use_populate_read = False

            used_fallback = True
            if madvise_fn is not None:
                madvise_fn(aligned_addr, aligned_len, _MADV_WILLNEED)  # type: ignore[operator]
            _strided_page_touch(t)

    if used_populate_read and used_fallback:
        return "mixed"
    if used_populate_read:
        return "madv_populate_read"
    if used_fallback:
        return "madv_willneed_touch"
    return "noop"


class LayerCacheState:
    """CPU-side O(1) LRU tracking metadata for one layer's persistent GPU slots [0 .. K-1]."""

    def __init__(self, num_persistent: int, total_experts: int = E) -> None:
        self.k = num_persistent
        self.expert_to_slot = np.full((total_experts,), -1, dtype=np.int32)
        self.slot_to_expert = np.full((num_persistent,), -1, dtype=np.int32)
        self.last_used = np.zeros((num_persistent,), dtype=np.int64)
        self.hit_count = np.zeros((num_persistent,), dtype=np.int64)

    def init_slots(self, initial_experts: Sequence[int]) -> None:
        assert len(initial_experts) == self.k
        self.expert_to_slot.fill(-1)
        for slot, eid in enumerate(initial_experts):
            eid_int = int(eid)
            self.slot_to_expert[slot] = eid_int
            self.expert_to_slot[eid_int] = slot
            self.last_used[slot] = 0
            self.hit_count[slot] = 1

    def record_access(self, expert_id: int, step_id: int) -> int:
        slot = int(self.expert_to_slot[expert_id])
        if slot >= 0:
            self.last_used[slot] = step_id
            self.hit_count[slot] += 1
        return slot

    def select_lru_victim(self) -> int:
        return int(np.argmin(self.last_used))

    def assign_slot(self, slot: int, new_expert: int, step_id: int) -> int:
        old_expert = int(self.slot_to_expert[slot])
        if old_expert >= 0:
            self.expert_to_slot[old_expert] = -1
        self.slot_to_expert[slot] = new_expert
        self.expert_to_slot[new_expert] = slot
        self.last_used[slot] = step_id
        self.hit_count[slot] = 1
        return old_expert


class HybridExpertCache:
    """Manages 50-layer Kolibri-1 resident weights, host routed store, and GPU cache/staging slabs."""

    def __init__(
        self,
        model_dir: Path = PHASE2_MODEL_DIR,
        device: Optional[torch.device] = None,
        persistent_slots_per_layer: int = 110,
        decode_miss_slots: int = TOPK,
        bounce_slots: int = 274,
        stage_slots: int = E,
        prefault_host: bool = True,
        gather_threads: int = 0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda:0")
        self.persistent_slots = persistent_slots_per_layer
        self.decode_miss_slots = decode_miss_slots
        self.cache_slots = persistent_slots_per_layer + decode_miss_slots  # e.g. 118 + 6 = 124
        self.bounce_slots = bounce_slots
        self.stage_slots = stage_slots
        self.prefault_host = prefault_host
        self.prefault_mode: str = "disabled"
        self.gather_threads = gather_threads

        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.globals: Dict[str, torch.Tensor] = {}
        self.gpu_non_experts: List[Dict[str, torch.Tensor]] = []
        self.host_routed: List[Dict[str, torch.Tensor]] = []
        self.gpu_caches: List[Dict[str, torch.Tensor]] = []
        self.cache_states: List[LayerCacheState] = []
        self.gpu_expert_to_slot = torch.full(
            (NUM_LAYERS, E), -1, dtype=torch.int32, device=self.device
        )

        # Shared across all 50 layers: 2 pinned host bounce slabs + 2 GPU staging slabs
        self.pinned_bounce: List[Dict[str, torch.Tensor]] = []
        self.gpu_stage: List[Dict[str, torch.Tensor]] = []

        self.stats: Optional[CacheMemoryStats] = None
        self._load_all()

    def _load_all(self) -> None:
        t0 = time.perf_counter()
        torch.cuda.synchronize(self.device)
        base_alloc = torch.cuda.memory_allocated(self.device)

        # 1. Load globals directly onto GPU
        g_cpu = load_file(str(self.model_dir / "globals.safetensors"), device="cpu")
        self.globals = {k: v.to(self.device, non_blocking=False).contiguous() for k, v in g_cpu.items()}
        del g_cpu
        after_globals = torch.cuda.memory_allocated(self.device)

        # 2. Load 50 layers: non-experts -> GPU, routed Marlin slices -> pageable CPU host store
        for li in range(NUM_LAYERS):
            lf = self.model_dir / f"layer_{li:02d}.safetensors"
            layer_cpu = load_file(str(lf), device="cpu")

            non_exp_gpu = {
                k: layer_cpu[k].to(self.device, non_blocking=False).contiguous()
                for k in NON_EXPERT_KEYS
            }
            self.gpu_non_experts.append(non_exp_gpu)

            routed_cpu = {k: layer_cpu[k].contiguous() for k in MARLIN_KEYS}
            self.host_routed.append(routed_cpu)
            del layer_cpu

        if self.prefault_host:
            self.prefault_mode = _prefault_host_routed(self.host_routed)

        after_non_exp = torch.cuda.memory_allocated(self.device)

        # 3. Allocate GPU persistent caches [50, K + 6, ...] and populate initial K slots
        # Copy slice-by-slice directly from CPU without creating intermediate GPU tensors.
        init_ids = list(range(self.persistent_slots))
        for li in range(NUM_LAYERS):
            slab = _alloc_marlin_slab(self.cache_slots, self.device, pin_memory=False)
            host_layer = self.host_routed[li]
            for key in MARLIN_KEYS:
                slab[key][: self.persistent_slots].copy_(host_layer[key][: self.persistent_slots])
            self.gpu_caches.append(slab)

            state = LayerCacheState(self.persistent_slots, total_experts=E)
            state.init_slots(init_ids)
            self.cache_states.append(state)
            self.gpu_expert_to_slot[li].copy_(torch.from_numpy(state.expert_to_slot))

        after_cache = torch.cuda.memory_allocated(self.device)

        # 4. Allocate double-buffered GPU staging arena [2, 384, ...] and pinned host bounce [2, bounce_slots, ...]
        for _ in range(2):
            self.gpu_stage.append(_alloc_marlin_slab(self.stage_slots, self.device, pin_memory=False))
            self.pinned_bounce.append(_alloc_marlin_slab(self.bounce_slots, torch.device("cpu"), pin_memory=True))

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        after_stage = torch.cuda.memory_allocated(self.device)

        host_routed_bytes = sum(_slab_bytes(slab) for slab in self.host_routed)
        bounce_bytes = sum(_slab_bytes(slab) for slab in self.pinned_bounce)
        mib = 1024.0 * 1024.0
        gib = 1024.0 ** 3

        self.stats = CacheMemoryStats(
            load_time_s=time.perf_counter() - t0,
            globals_mib=(after_globals - base_alloc) / mib,
            non_experts_mib=(after_non_exp - after_globals) / mib,
            gpu_cache_mib=(after_cache - after_non_exp) / mib,
            gpu_stage_mib=(after_stage - after_cache) / mib,
            total_gpu_allocated_mib=after_stage / mib,
            total_gpu_reserved_mib=torch.cuda.memory_reserved(self.device) / mib,
            host_routed_gib=host_routed_bytes / gib,
            host_pinned_bounce_gib=bounce_bytes / gib,
        )

    def gather_to_pinned(
        self, layer_idx: int, expert_ids: torch.Tensor, buf_idx: int
    ) -> int:
        """Gather `expert_ids` from pageable host store into `pinned_bounce[buf_idx][:m]`."""
        m = int(expert_ids.numel())
        assert m <= self.bounce_slots, f"Miss count {m} exceeds bounce_slots={self.bounce_slots}"
        if m == 0:
            return 0
        idx_cpu = expert_ids.to(dtype=torch.long, device="cpu").contiguous()
        host_layer = self.host_routed[layer_idx]
        bounce = self.pinned_bounce[buf_idx]
        gather_experts_omp(
            [host_layer[k] for k in MARLIN_KEYS],
            [bounce[k] for k in MARLIN_KEYS],
            idx_cpu,
            num_threads=self.gather_threads,
        )
        return m

    def dma_pinned_to_gpu_slab(
        self,
        m: int,
        buf_idx: int,
        dst_slab: Dict[str, torch.Tensor],
        dst_offset: int = 0,
    ) -> None:
        """Issue async H2D DMA from `pinned_bounce[buf_idx][:m]` into `dst_slab[dst_offset:dst_offset+m]`."""
        if m == 0:
            return
        bounce = self.pinned_bounce[buf_idx]
        with torch.cuda.stream(self.copy_stream):
            for key in MARLIN_KEYS:
                dst_slab[key][dst_offset : dst_offset + m].copy_(bounce[key][:m], non_blocking=True)
            self.copy_events[buf_idx].record(self.copy_stream)

    def stage_experts_async(
        self, layer_idx: int, expert_ids: torch.Tensor, buf_idx: int
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Full pipeline: host gather -> pinned bounce -> async DMA onto `gpu_stage[buf_idx]`."""
        m = self.gather_to_pinned(layer_idx, expert_ids, buf_idx)
        self.dma_pinned_to_gpu_slab(m, buf_idx, self.gpu_stage[buf_idx], dst_offset=0)
        return m, self.gpu_stage[buf_idx]

    def wait_copy(self, buf_idx: int, stream: Optional[torch.cuda.Stream] = None) -> None:
        target = stream or torch.cuda.current_stream(self.device)
        target.wait_event(self.copy_events[buf_idx])

    def decode_stage_misses_async(
        self, layer_idx: int, miss_expert_ids: torch.Tensor, buf_idx: int = 0
    ) -> int:
        """DMA up to `decode_miss_slots` missing experts into `gpu_caches[layer_idx][K : K + m]`."""
        m = int(miss_expert_ids.numel())
        assert m <= self.decode_miss_slots, f"Decode miss count {m} > {self.decode_miss_slots}"
        self.gather_to_pinned(layer_idx, miss_expert_ids, buf_idx)
        self.dma_pinned_to_gpu_slab(
            m, buf_idx, self.gpu_caches[layer_idx], dst_offset=self.persistent_slots
        )
        return m

    def promote_decode_misses(
        self, layer_idx: int, miss_expert_ids: Sequence[int], step_id: int
    ) -> List[Tuple[int, int]]:
        """Promote missed experts from decode scratch slots [K : K+m] into persistent LRU slots [0..K-1] on GPU."""
        state = self.cache_states[layer_idx]
        cache_slab = self.gpu_caches[layer_idx]
        promotions: List[Tuple[int, int]] = []
        for idx, eid in enumerate(miss_expert_ids):
            eid_int = int(eid)
            victim_slot = state.select_lru_victim()
            src_slot = self.persistent_slots + idx
            for key in MARLIN_KEYS:
                cache_slab[key][victim_slot].copy_(cache_slab[key][src_slot])
            state.assign_slot(victim_slot, eid_int, step_id)
            promotions.append((eid_int, victim_slot))
        if promotions:
            self.gpu_expert_to_slot[layer_idx].copy_(
                torch.from_numpy(state.expert_to_slot), non_blocking=True
            )
        return promotions

    def seed_layer_cache(self, layer_idx: int, expert_ids: Sequence[int], step_id: int = 0) -> None:
        """Overwrite layer persistent slots [0..K-1] with specific expert IDs (used for testing & warm-up)."""
        assert len(expert_ids) == self.persistent_slots
        state = self.cache_states[layer_idx]
        cache_slab = self.gpu_caches[layer_idx]
        host_layer = self.host_routed[layer_idx]
        idx_t = torch.tensor(list(expert_ids), dtype=torch.long)
        for key in MARLIN_KEYS:
            cache_slab[key][: self.persistent_slots].copy_(host_layer[key][idx_t])
        state.init_slots(expert_ids)
        state.last_used.fill(step_id)
        self.gpu_expert_to_slot[layer_idx].copy_(torch.from_numpy(state.expert_to_slot))

    def reset_default_slots(self) -> None:
        """Reset all 50 layers back to deterministic cold `[0 .. K-1]` expert slots."""
        init_ids = list(range(self.persistent_slots))
        for li in range(NUM_LAYERS):
            self.seed_layer_cache(li, init_ids, step_id=0)
        torch.cuda.synchronize(self.device)
