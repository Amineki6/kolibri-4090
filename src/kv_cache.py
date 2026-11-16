"""Step 3.3: Hybrid 40-layer SWA (513-token ring) + 10-layer Full-RNoPE KV Cache."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import (
    HD,
    NKV,
    NQ,
    NUM_LAYERS,
    RMS_EPS,
    SLIDING_WINDOW,
    is_full_attention_layer,
)
from .kolibri_kernels import flashinfer_apply_rope_inplace, make_rope_cos_sin_cache


@dataclass
class KVCacheStats:
    max_seq_len: int
    num_swa_layers: int
    num_full_layers: int
    swa_kv_mib: float
    full_kv_mib: float
    total_kv_mib: float
    total_kv_gib: float


class HybridKVCache:
    """Manages BF16 KV cache for 40 SWA layers (capped at 513 tokens) + 10 Full-RNoPE layers."""

    def __init__(
        self,
        max_seq_len: int = 100_000,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.seq_len = 0

        self.layer_is_full: List[bool] = [is_full_attention_layer(i) for i in range(NUM_LAYERS)]
        self.layer_sub_idx: List[int] = []
        swa_count = 0
        full_count = 0
        for is_full in self.layer_is_full:
            if is_full:
                self.layer_sub_idx.append(full_count)
                full_count += 1
            else:
                self.layer_sub_idx.append(swa_count)
                swa_count += 1

        # 40 SWA ring buffers: [40, NKV, 513, HD]
        self.k_swa = torch.empty((swa_count, NKV, SLIDING_WINDOW, HD), dtype=dtype, device=device)
        self.v_swa = torch.empty((swa_count, NKV, SLIDING_WINDOW, HD), dtype=dtype, device=device)

        # 10 Full-attention RNoPE buffers: [10, NKV, max_seq_len, HD]
        self.k_full = torch.empty((full_count, NKV, max_seq_len, HD), dtype=dtype, device=device)
        self.v_full = torch.empty((full_count, NKV, max_seq_len, HD), dtype=dtype, device=device)

        # Precomputed RoPE cos/sin table and integer positions for FlashInfer RoPE + prefill masks
        self.rope_cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, device)
        self.pos_range = torch.arange(max_seq_len, dtype=torch.int64, device=device)

        swa_bytes = 2 * self.k_swa.numel() * self.k_swa.element_size()
        full_bytes = 2 * self.k_full.numel() * self.k_full.element_size()
        total_bytes = swa_bytes + full_bytes
        mib = 1024.0 * 1024.0
        self.stats = KVCacheStats(
            max_seq_len=max_seq_len,
            num_swa_layers=swa_count,
            num_full_layers=full_count,
            swa_kv_mib=swa_bytes / mib,
            full_kv_mib=full_bytes / mib,
            total_kv_mib=total_bytes / mib,
            total_kv_gib=total_bytes / (1024.0 ** 3),
        )

    def reset(self) -> None:
        self.seq_len = 0

    def advance(self, n_tokens: int) -> None:
        assert self.seq_len + n_tokens <= self.max_seq_len, (
            f"Sequence length {self.seq_len + n_tokens} exceeds max_seq_len={self.max_seq_len}"
        )
        self.seq_len += n_tokens

    def _gather_swa_tail(self, s_idx: int, pos: int, tail_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the last `tail_len` (`<= 512`) chronological SWA keys/values before `pos`."""
        if tail_len == 0:
            empty = torch.empty((NKV, 0, HD), dtype=self.dtype, device=self.device)
            return empty, empty
        if pos <= SLIDING_WINDOW:
            start = pos - tail_len
            return self.k_swa[s_idx, :, start:pos, :], self.v_swa[s_idx, :, start:pos, :]
        head = pos % SLIDING_WINDOW
        k_chrono = torch.cat([self.k_swa[s_idx, :, head:, :], self.k_swa[s_idx, :, :head, :]], dim=1)
        v_chrono = torch.cat([self.v_swa[s_idx, :, head:, :], self.v_swa[s_idx, :, :head, :]], dim=1)
        return k_chrono[:, -tail_len:, :], v_chrono[:, -tail_len:, :]

    def _write_swa(self, s_idx: int, pos: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        n = k_new.shape[1]
        if n >= SLIDING_WINDOW:
            # Keep only the most recent SLIDING_WINDOW tokens anchored at (pos + n) % SLIDING_WINDOW
            k_tail = k_new[:, -SLIDING_WINDOW:, :]
            v_tail = v_new[:, -SLIDING_WINDOW:, :]
            start_pos = pos + n - SLIDING_WINDOW
        else:
            k_tail, v_tail, start_pos = k_new, v_new, pos

        m = k_tail.shape[1]
        slot_start = start_pos % SLIDING_WINDOW
        first = min(m, SLIDING_WINDOW - slot_start)
        self.k_swa[s_idx, :, slot_start : slot_start + first, :].copy_(k_tail[:, :first, :])
        self.v_swa[s_idx, :, slot_start : slot_start + first, :].copy_(v_tail[:, :first, :])
        rem = m - first
        if rem > 0:
            self.k_swa[s_idx, :, :rem, :].copy_(k_tail[:, first:, :])
            self.v_swa[s_idx, :, :rem, :].copy_(v_tail[:, first:, :])

    def attention_forward(
        self,
        x: torch.Tensor,
        L: Dict[str, torch.Tensor],
        layer_idx: int,
        inv_freq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention for layer `layer_idx` at current `self.seq_len` (does NOT advance `seq_len`)."""
        del inv_freq  # kept for signature compatibility; uses self.rope_cos_sin_cache
        n = x.shape[0]
        pos = self.seq_len
        assert pos + n <= self.max_seq_len, f"KV cache overflow: {pos + n} > {self.max_seq_len}"

        qkv = x @ L["w_qkv"]
        q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
        q = F.rms_norm(q.view(n, NQ, HD), (HD,), L["q_norm"], RMS_EPS)
        k = F.rms_norm(k.view(n, NKV, HD), (HD,), L["k_norm"], RMS_EPS)

        is_full = self.layer_is_full[layer_idx]
        sub_idx = self.layer_sub_idx[layer_idx]

        if not is_full:
            flashinfer_apply_rope_inplace(
                self.pos_range[pos : pos + n],
                q.view(n, NQ * HD),
                k.view(n, NKV * HD),
                self.rope_cos_sin_cache,
            )

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.view(n, NKV, HD).transpose(0, 1)

        if is_full:
            # Full Attention (RNoPE: no RoPE)
            self.k_full[sub_idx, :, pos : pos + n, :].copy_(k)
            self.v_full[sub_idx, :, pos : pos + n, :].copy_(v)
            k_ctx = self.k_full[sub_idx, :, : pos + n, :]
            v_ctx = self.v_full[sub_idx, :, : pos + n, :]

            if n == 1:
                o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, is_causal=False, enable_gqa=True)
            elif pos == 0:
                o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, is_causal=True, enable_gqa=True)
            else:
                q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                k_pos = self.pos_range[: pos + n].unsqueeze(0)
                mask = q_pos >= k_pos
                o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
        else:
            # SWA Layer (RoPE + 513-token ring buffer)
            if n == 1:
                slot = pos % SLIDING_WINDOW
                self.k_swa[sub_idx, :, slot : slot + 1, :].copy_(k)
                self.v_swa[sub_idx, :, slot : slot + 1, :].copy_(v)
                valid_len = min(pos + 1, SLIDING_WINDOW)
                k_ctx = self.k_swa[sub_idx, :, :valid_len, :]
                v_ctx = self.v_swa[sub_idx, :, :valid_len, :]
                o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, is_causal=False, enable_gqa=True)
            elif pos + n <= SLIDING_WINDOW:
                self._write_swa(sub_idx, pos, k, v)
                k_ctx = self.k_swa[sub_idx, :, : pos + n, :]
                v_ctx = self.v_swa[sub_idx, :, : pos + n, :]
                if pos == 0:
                    o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, is_causal=True, enable_gqa=True)
                else:
                    q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                    k_pos = self.pos_range[: pos + n].unsqueeze(0)
                    mask = q_pos >= k_pos
                    o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
            else:
                # Chunked prefill crossing or past SLIDING_WINDOW: gather prior 512 tokens before overwrite
                tail_len = min(pos, SLIDING_WINDOW - 1)
                k_tail, v_tail = self._gather_swa_tail(sub_idx, pos, tail_len)
                k_ctx = torch.cat([k_tail, k], dim=1)
                v_ctx = torch.cat([v_tail, v], dim=1)
                q_pos = self.pos_range[pos : pos + n].unsqueeze(1)
                k_pos = self.pos_range[pos - tail_len : pos + n].unsqueeze(0)
                diff = q_pos - k_pos
                mask = (diff >= 0) & (diff < SLIDING_WINDOW)
                o = F.scaled_dot_product_attention(q, k_ctx, v_ctx, attn_mask=mask, enable_gqa=True)
                self._write_swa(sub_idx, pos, k, v)

        return o.transpose(0, 1).reshape(n, NQ * HD) @ L["w_o"]
