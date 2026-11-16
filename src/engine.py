"""Step 3.4: Full 50-layer Kolibri-1 Hybrid Inference Engine (W4A16 NVFP4 Marlin)."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import H, NUM_LAYERS, PHASE2_MODEL_DIR, RMS_EPS, TOPK
from .expert_cache import HybridExpertCache
from .hybrid_moe import HybridMarlinMoE
from .kolibri_kernels import (
    flashinfer_fused_add_rmsnorm,
    fused_add_post_moe_norm,
    fused_sandwich_norm,
    warmup_kernels,
)
from .kv_cache import HybridKVCache
from .model_ops import kolibri_route, make_rope_inv_freq, shared_expert

EOS_TOKEN_ID = 127906   # <|im_end|>
PAD_TOKEN_ID = 127901   # <|endoftext|>

REASONING_SENTENCES = {
    "none": "Reasoning is disabled. Proceed straight to answering according to the user's instructions.",
    "low": "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed directly to the answer.",
    "medium": "Reasoning effort is set to medium. Think through the task methodically in the user's language, check key assumptions, and provide a well-supported answer.",
    "high": "Reasoning effort is set to high. Think carefully through the task in the user's language, validate key assumptions, consider plausible alternatives, and prioritize correctness and clarity.",
}


@dataclass
class GenerationResult:
    prompt_tokens: int
    generated_tokens: int
    text: str
    token_ids: List[int]
    prefill_ms: float
    prefill_tok_per_s: float
    prefill_hit_pct: float
    decode_ms: float
    decode_tok_per_s: float
    decode_hit_pct: float
    decode_experts_transferred: int
    peak_vram_gib: float


def format_messages_prompt(
    messages: Sequence[Dict[str, object]],
    reasoning_effort: str = "none",
) -> str:
    """Format an OpenAI-style message list using Kolibri-1's ChatML + reasoning effort template."""
    sentence = REASONING_SENTENCES.get(reasoning_effort, REASONING_SENTENCES["none"])
    parts: List[str] = []

    has_first_sys = bool(messages) and messages[0].get("role") == "system"
    first_sys_text = ""
    if has_first_sys:
        c = messages[0].get("content", "")
        first_sys_text = c if isinstance(c, str) else "".join(
            p.get("text", "") for p in c if isinstance(p, dict)
        )
    sys_body = (
        f"{first_sys_text}\n\n# Reasoning effort\n\n{sentence}"
        if first_sys_text
        else f"# Reasoning effort\n\n{sentence}"
    )
    parts.append(f"<|im_start|>system\n{sys_body}<|im_end|>\n")

    start_idx = 1 if has_first_sys else 0
    for msg in messages[start_idx:]:
        role = str(msg.get("role", "user"))
        raw = msg.get("content", "")
        content = raw if isinstance(raw, str) else "".join(
            p.get("text", "") for p in raw if isinstance(p, dict)
        )
        if role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content.lstrip(chr(10))}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    if reasoning_effort == "none":
        parts.append("<think>\n\n</think>\n\n")
    else:
        parts.append("<think>\n")
    return "".join(parts)


def format_chat_prompt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    reasoning_effort: str = "none",
) -> str:
    """Format a single user prompt using Kolibri-1's ChatML + reasoning effort template."""
    msgs: List[Dict[str, object]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return format_messages_prompt(msgs, reasoning_effort=reasoning_effort)


class KolibriHybridEngine:
    """Full 50-layer Kolibri-1 serving engine combining HybridExpertCache, HybridMarlinMoE, and HybridKVCache."""

    def __init__(
        self,
        model_dir: Path = PHASE2_MODEL_DIR,
        device: Optional[torch.device] = None,
        persistent_slots_per_layer: int = 110,
        max_seq_len: int = 100_000,
        prefill_chunk_size: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda:0")
        self.prefill_chunk_size = prefill_chunk_size

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.cache = HybridExpertCache(
            model_dir=self.model_dir,
            device=self.device,
            persistent_slots_per_layer=persistent_slots_per_layer,
            decode_miss_slots=TOPK,
            bounce_slots=384 - persistent_slots_per_layer,
            stage_slots=384,
        )
        self.kv = HybridKVCache(max_seq_len=max_seq_len, device=self.device)
        self.moe = HybridMarlinMoE(self.cache, max_tokens=max(prefill_chunk_size, 512))
        self.inv_freq = make_rope_inv_freq(self.device)
        warmup_kernels(self.device)

    @torch.inference_mode()
    def _forward_layers(
        self,
        token_ids: torch.Tensor,
        promote_misses: bool = True,
        counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one contiguous chunk [n] through all 50 layers, advance KV cache, return (cur, residual)."""
        n = int(token_ids.shape[0])
        cur = F.embedding(token_ids, self.cache.globals["embed_tokens"]).to(torch.bfloat16)
        residual: Optional[torch.Tensor] = None

        for li in range(NUM_LAYERS):
            L = self.cache.gpu_non_experts[li]
            if residual is None:
                residual = cur
                h = F.rms_norm(residual, (H,), L["input_ln"], RMS_EPS)
            else:
                flashinfer_fused_add_rmsnorm(cur, residual, L["input_ln"], RMS_EPS)
                h = cur

            attn_out = self.kv.attention_forward(h, L, li, self.inv_freq)
            residual, moe_in = fused_sandwich_norm(
                attn_out, residual, L["post_attn_ln"], L["pre_moe_ln"], RMS_EPS
            )

            ids, wts = kolibri_route(moe_in, L)
            if counts is not None:
                counts[li] += torch.bincount(ids.reshape(-1), minlength=384)
            routed, shared = self.moe.forward_layer(
                moe_in,
                ids,
                wts,
                li,
                shared_fn=lambda m_in, layer=L: shared_expert(m_in, layer),
                promote_misses=promote_misses,
            )
            cur = fused_add_post_moe_norm(routed, shared, L["post_ffn_ln"], RMS_EPS)

        self.kv.advance(n)
        return cur, residual  # type: ignore[return-value]

    @torch.inference_mode()
    def _forward_chunk(
        self, token_ids: torch.Tensor, promote_misses: bool = True
    ) -> torch.Tensor:
        """Run one contiguous chunk [n] of tokens through all 50 layers and return last-token logits."""
        cur, residual = self._forward_layers(token_ids, promote_misses=promote_misses)
        last_cur = cur[-1:].contiguous()
        last_res = residual[-1:].contiguous()
        flashinfer_fused_add_rmsnorm(last_cur, last_res, self.cache.globals["model_norm"], RMS_EPS)
        return F.linear(last_cur, self.cache.globals["lm_head"])

    @torch.inference_mode()
    def prefill(
        self,
        token_ids: Sequence[int],
        promote_misses: bool = False,
        reset_kv: bool = True,
        progress_cb=None,
    ) -> torch.Tensor:
        """Prefill prompt tokens in bounded chunks and return last-token logits `[1, vocab]`."""
        if reset_kv:
            self.kv.reset()
        ids_t = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        total = int(ids_t.shape[0])
        assert total > 0, "Prompt must contain at least 1 token"

        logits = None
        for start in range(0, total, self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, total)
            logits = self._forward_chunk(ids_t[start:end], promote_misses=promote_misses)
            if progress_cb is not None:
                progress_cb(end, total)
        return logits

    @torch.inference_mode()
    def step(self, token_id: int, promote_misses: bool = True) -> torch.Tensor:
        """Execute one autoregressive decode step and return logits `[1, vocab]`."""
        t = torch.tensor([token_id], dtype=torch.long, device=self.device)
        return self._forward_chunk(t, promote_misses=promote_misses)

    @torch.inference_mode()
    def calibrate_hot_cache(self, prompts: Sequence[str]) -> Dict[str, float]:
        """Profile expert selection frequencies on calibration prompts and seed top-K experts per layer."""
        counts = torch.zeros((NUM_LAYERS, 384), dtype=torch.int64, device=self.device)
        self.moe.reset_stats()
        total_toks = 0

        for text in prompts:
            ids = self.tokenizer.encode(text).ids
            total_toks += len(ids)
            self.kv.reset()
            ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
            for start in range(0, len(ids), self.prefill_chunk_size):
                chunk = ids_t[start : min(start + self.prefill_chunk_size, len(ids))]
                self._forward_layers(chunk, promote_misses=False, counts=counts)

        k = self.cache.persistent_slots
        top_lists = torch.topk(counts, k=k, dim=-1).indices.cpu().tolist()
        for li in range(NUM_LAYERS):
            self.cache.seed_layer_cache(li, top_lists[li])
        return {"calibration_tokens": total_toks, "seeded_slots_per_layer": k}

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits[0]).item())
        scaled = logits[0].float() / temperature
        if top_k > 0 and top_k < scaled.shape[-1]:
            vals, idx = torch.topk(scaled, k=top_k)
        else:
            vals, idx = scaled, None
        probs = F.softmax(vals, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_order = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
            choice = int(torch.multinomial(sorted_probs, num_samples=1, generator=generator).item())
            chosen_local = int(sorted_order[choice].item())
        else:
            chosen_local = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        return int(idx[chosen_local].item()) if idx is not None else chosen_local

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 1,
        promote_misses: bool = True,
        stop_on_eos: bool = True,
        stream_callback=None,
    ) -> GenerationResult:
        prompt_ids = self.tokenizer.encode(prompt).ids
        torch.cuda.reset_peak_memory_stats(self.device)

        # Prefill phase
        self.moe.reset_stats()
        torch.cuda.synchronize(self.device)
        t_p0 = time.perf_counter()
        logits = self.prefill(prompt_ids, promote_misses=False)
        torch.cuda.synchronize(self.device)
        prefill_ms = (time.perf_counter() - t_p0) * 1000.0
        prefill_hit_pct = self.moe.stats.hit_rate * 100.0

        # Decode phase
        self.moe.reset_stats()
        generated: List[int] = []
        next_id = self._sample(logits, temperature, top_k)

        torch.cuda.synchronize(self.device)
        t_d0 = time.perf_counter()
        for step_idx in range(max_new_tokens):
            if stop_on_eos and next_id in (EOS_TOKEN_ID, PAD_TOKEN_ID):
                break
            generated.append(next_id)
            if stream_callback is not None:
                stream_callback(self.tokenizer.decode([next_id], skip_special_tokens=False))
            if step_idx + 1 >= max_new_tokens:
                break
            logits = self.step(next_id, promote_misses=promote_misses)
            next_id = self._sample(logits, temperature, top_k)

        torch.cuda.synchronize(self.device)
        decode_ms = (time.perf_counter() - t_d0) * 1000.0

        decode_steps = max(len(generated) - 1, 0)
        decode_tok_s = (decode_steps / (decode_ms / 1000.0)) if decode_ms > 0 and decode_steps > 0 else 0.0
        decode_hit_pct = self.moe.stats.hit_rate * 100.0 if self.moe.stats.total_pair_evals > 0 else 100.0

        return GenerationResult(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=generated,
            prefill_ms=prefill_ms,
            prefill_tok_per_s=len(prompt_ids) / max(prefill_ms / 1000.0, 1e-9),
            prefill_hit_pct=prefill_hit_pct,
            decode_ms=decode_ms,
            decode_tok_per_s=decode_tok_s,
            decode_hit_pct=decode_hit_pct,
            decode_experts_transferred=self.moe.stats.experts_transferred,
            peak_vram_gib=torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3),
        )
