#!/usr/bin/env python3
"""OpenAI-compatible HTTP server & Strata Web UI adapter for Kolibri-1 NVFP4 Marlin.

Serves Strata's static frontend assets read-only from `/home/mkina/profiling/Strata/serve/web`
on top of `KolibriHybridEngine` with SSE streaming, `<think>...</think>` splitting, KV turn
continuation reuse, `/load` + `/unload`, and live RTX 4090 + host telemetry at `/metrics`.
"""

import argparse
import collections
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import torch

SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from src.config import NUM_LAYERS, PHASE2_MODEL_DIR  # noqa: E402
from src.engine import (  # noqa: E402
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    KolibriHybridEngine,
    format_messages_prompt,
)

STRATA_WEB = Path("/home/mkina/profiling/Strata/serve/web")
MODEL_NAME = "Kolibri-1 (W4A16-NVFP4 Marlin)"

LOCALE_RE = re.compile(r"^/web/locales/(?P<name>[a-z]{2,3}(?:-[A-Z]{2})?\.json)$")
VENDOR_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
}
FLAT_WEB_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}

FAST_MODE_BTN_HTML = (
    '<button class="st-btn st-btn--secondary" id="fast-mode-btn" type="button" '
    'title="Pause background nvidia-smi polling, disable Python GC during decode, '
    'use O(1) incremental decoding, and batch SSE flushes" aria-pressed="false">'
    "Fast Decode: OFF</button>\n  "
)

FAST_MODE_SCRIPT_HTML = """<script>
(() => {
  const btn = document.getElementById("fast-mode-btn");
  if (!btn) return;
  let enabled = false;
  function render() {
    btn.textContent = enabled ? "Fast Decode: ON" : "Fast Decode: OFF";
    btn.setAttribute("aria-pressed", enabled ? "true" : "false");
    btn.classList.toggle("st-btn--primary", enabled);
    btn.classList.toggle("st-btn--secondary", !enabled);
  }
  async function sync() {
    try {
      const r = await fetch("/fast-mode", { cache: "no-store" });
      if (r.ok) {
        const d = await r.json();
        enabled = !!d.fast_mode;
        render();
      }
    } catch (e) {}
  }
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const r = await fetch("/fast-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !enabled }),
      });
      if (r.ok) {
        const d = await r.json();
        enabled = !!d.fast_mode;
        render();
      }
    } catch (e) {
    } finally {
      btn.disabled = false;
    }
  });
  sync();
})();
</script>
"""


def render_index_html(html_text: str) -> bytes:
    """Inject the Fast Decode header toggle button and script into Strata's index.html."""
    target_btn = '<button class="st-btn st-btn--icon" id="theme-btn"'
    if target_btn in html_text:
        html_text = html_text.replace(target_btn, FAST_MODE_BTN_HTML + target_btn, 1)
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", FAST_MODE_SCRIPT_HTML + "</body>", 1)
    return html_text.encode("utf-8")


def resolve_subtree_file(path: str) -> Optional[Tuple[Path, str]]:
    if any(c in path for c in ("\\", ":", "%", "\0")) or ".." in path:
        return None
    m = LOCALE_RE.match(path)
    if m:
        f = STRATA_WEB / "locales" / m.group("name")
        return (f, "application/json; charset=utf-8") if f.is_file() else None
    prefix = "/web/vendor/katex/"
    if path.startswith(prefix):
        vendor = (STRATA_WEB / "vendor" / "katex").resolve()
        f = (vendor / path[len(prefix) :]).resolve()
        ctype = VENDOR_TYPES.get(f.suffix)
        if ctype and f.is_relative_to(vendor) and f.is_file():
            return f, ctype
    return None


class HardwareSampler:
    """Lightweight background sampler for RTX 4090 + host RAM/CPU telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prev_cpu: Optional[Tuple[int, int]] = None
        self._snapshot: Dict[str, Any] = {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU",
            "gpu_util": None,
            "gpu_mem_used": None,
            "gpu_mem_total": None,
            "gpu_temp": None,
            "gpu_power": None,
            "gpu_power_limit": None,
            "gpu_pcie_gen": None,
            "gpu_pcie_gen_max": None,
            "gpu_pcie_width": None,
            "cpu": None,
            "ram_used": None,
            "ram_total": None,
            "cpu_name": self._read_cpu_name(),
            "threads": os.cpu_count(),
        }
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @staticmethod
    def _read_cpu_name() -> str:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return ""

    def _sample_cpu_mem(self) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        cpu_pct = None
        try:
            fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:8]]
            total = sum(fields)
            idle = fields[3] + fields[4]
            if self._prev_cpu is not None:
                dt = total - self._prev_cpu[0]
                di = idle - self._prev_cpu[1]
                if dt > 0:
                    cpu_pct = round(100.0 * (1.0 - di / dt), 1)
            self._prev_cpu = (total, idle)
        except (OSError, IndexError, ValueError):
            pass

        ram_used = ram_total = None
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) * 1024
            ram_total = info.get("MemTotal")
            avail = info.get("MemAvailable")
            if ram_total and avail is not None:
                ram_used = ram_total - avail
        except (OSError, ValueError):
            pass
        return cpu_pct, ram_used, ram_total

    def _sample_gpu(self) -> Dict[str, Any]:
        fields = [
            "name",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
            "power.limit",
            "pcie.link.gen.current",
            "pcie.link.gen.max",
            "pcie.link.width.current",
        ]
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", "0", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if out.returncode != 0:
                return {}
            vals = [v.strip() for v in out.stdout.strip().split(",")]
            if len(vals) != len(fields):
                return {}

            def num(s: str, fn=float):
                try:
                    return None if s in ("", "[N/A]", "N/A") else fn(s)
                except ValueError:
                    return None

            mem_used_mib = num(vals[2])
            mem_tot_mib = num(vals[3])
            return {
                "gpu_name": vals[0],
                "gpu_util": num(vals[1]),
                "gpu_mem_used": int(mem_used_mib * 1048576) if mem_used_mib is not None else None,
                "gpu_mem_total": int(mem_tot_mib * 1048576) if mem_tot_mib is not None else None,
                "gpu_temp": num(vals[4]),
                "gpu_power": num(vals[5]),
                "gpu_power_limit": num(vals[6]),
                "gpu_pcie_gen": num(vals[7], int),
                "gpu_pcie_gen_max": num(vals[8], int),
                "gpu_pcie_width": num(vals[9], int),
            }
        except (OSError, subprocess.SubprocessError):
            return {}

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.25)
                continue
            cpu, ram_u, ram_t = self._sample_cpu_mem()
            gpu = self._sample_gpu()
            with self._lock:
                self._snapshot.update(gpu)
                self._snapshot["cpu"] = cpu
                self._snapshot["ram_used"] = ram_u
                self._snapshot["ram_total"] = ram_t
            self._stop.wait(1.0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)


class KolibriService:
    """Thread-safe lifecycle, streaming inference, KV prefix tracking, and metrics state."""

    def __init__(
        self,
        model_dir: Path,
        persistent_slots: int,
        max_seq_len: int,
        prefill_chunk_size: int,
    ) -> None:
        self.model_dir = model_dir
        self.persistent_slots = persistent_slots
        self.max_seq_len = max_seq_len
        self.prefill_chunk_size = prefill_chunk_size

        self.infer_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.hw = HardwareSampler()
        self.fast_mode: bool = False
        self.shared_settings: Optional[Dict[str, Any]] = None

        self.engine: Optional[KolibriHybridEngine] = None
        self.last_tokens: List[int] = []

        self.live: Dict[str, Any] = {
            "state": "unloaded",
            "queued": 0,
            "prompt_tokens": 0,
            "prompt_read": 0,
            "prompt_total": 0,
            "generated": 0,
            "max_tokens": 0,
            "tok_s": 0.0,
            "prefill_tok_s_mean": None,
        }
        self.requests: collections.deque = collections.deque(maxlen=200)
        self.totals: Dict[str, Any] = {
            "since": time.time(),
            "requests": 0,
            "prompt_tokens": 0,
            "reused": 0,
            "prompt_ms": 0.0,
            "output_tokens": 0,
            "decode_ms": 0.0,
        }
        self.conv_stats: Dict[str, Any] = {
            "enabled": False,
            "requests": 0,
            "requests_reused": 0,
            "prompt_tokens": 0,
            "reused_tokens": 0,
            "last_prompt": None,
            "last_reused": 0,
        }
        self.history: Dict[str, collections.deque] = {
            k: collections.deque(maxlen=60)
            for k in ("tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "cpu")
        }
        self._hist_thread = threading.Thread(target=self._record_history_loop, daemon=True)
        self._hist_thread.start()

    def set_fast_mode(self, enabled: bool) -> bool:
        self.fast_mode = bool(enabled)
        self.hw.set_paused(self.fast_mode)
        return self.fast_mode

    def _record_history_loop(self) -> None:
        while True:
            if self.fast_mode:
                time.sleep(1.0)
                continue
            hw = self.hw.snapshot()
            with self.state_lock:
                live_tok_s = self.live["tok_s"] if self.live["state"] == "generating" else None
                pref = self.live["prefill_tok_s_mean"]
            self.history["tok_s"].append(live_tok_s)
            self.history["prefill_tok_s_mean"].append(pref)
            self.history["gpu_util"].append(hw.get("gpu_util"))
            self.history["gpu_mem_used"].append(hw.get("gpu_mem_used"))
            self.history["gpu_temp"].append(hw.get("gpu_temp"))
            self.history["gpu_power"].append(hw.get("gpu_power"))
            self.history["cpu"].append(hw.get("cpu"))
            time.sleep(1.0)

    def load(self) -> None:
        with self.infer_lock:
            if self.engine is not None:
                return
            self.engine = KolibriHybridEngine(
                model_dir=self.model_dir,
                device=torch.device("cuda:0"),
                persistent_slots_per_layer=self.persistent_slots,
                max_seq_len=self.max_seq_len,
                prefill_chunk_size=self.prefill_chunk_size,
            )
            self.last_tokens = []
            with self.state_lock:
                self.live["state"] = "idle"

    def unload(self) -> str:
        if not self.infer_lock.acquire(blocking=False):
            return "busy"
        try:
            if self.engine is not None:
                del self.engine
                self.engine = None
                self.last_tokens = []
                torch.cuda.empty_cache()
            with self.state_lock:
                self.live["state"] = "unloaded"
            return "unloaded"
        finally:
            self.infer_lock.release()

    def loaded(self) -> bool:
        return self.engine is not None

    def health_payload(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "max_context": self.max_seq_len,
            "images": False,
            "image_generation": False,
            "api_key": False,
            "loaded": self.loaded(),
            "service": "kolibri-hybrid",
        }

    def metrics_payload(self, all_requests: bool = False) -> Dict[str, Any]:
        hw = self.hw.snapshot()
        with self.state_lock:
            live = dict(self.live)
            reqs = list(self.requests) if all_requests else list(self.requests)[:12]
            totals = dict(self.totals)
            cc = dict(self.conv_stats)
            hist = {k: list(v) for k, v in self.history.items()}

        expert_slots = NUM_LAYERS * self.persistent_slots
        expert_cache_mib = self.engine.cache.stats.gpu_cache_mib if self.engine is not None else 0.0
        return {
            "live": live,
            "hardware": {
                **hw,
                "gpus": [
                    {
                        "index": 0,
                        "name": hw.get("gpu_name"),
                        "util": hw.get("gpu_util"),
                        "mem_used": hw.get("gpu_mem_used"),
                        "mem_total": hw.get("gpu_mem_total"),
                        "temp": hw.get("gpu_temp"),
                        "power": hw.get("gpu_power"),
                    }
                ],
            },
            "static": {
                "gpu_name": hw.get("gpu_name"),
                "cpu_name": hw.get("cpu_name"),
                "threads": hw.get("threads"),
                "psutil": False,
            },
            "engine": {
                "model": MODEL_NAME,
                "version": "kolibri-nvfp4-marlin",
                "max_context": self.max_seq_len,
                "kv": "bf16",
                "expert_slots": expert_slots,
                "expert_cache_mib": expert_cache_mib,
                "images": False,
                "cvec": "0",
            },
            "history": hist,
            "requests": reqs,
            "requests_kept": len(self.requests),
            "totals": totals,
            "conversation_cache": cc,
        }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(svc: KolibriService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep terminal output clean; inference stats print concisely per turn.
            pass

        def _json(self, status: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, body: bytes, ctype: str, cache: str = "no-cache") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            raw_path = self.path.split("?")[0]
            path = raw_path.rstrip("/")

            if path == "" or path == "/index.html":
                f = STRATA_WEB / "index.html"
                self._send_bytes(200, render_index_html(f.read_text("utf-8")), "text/html; charset=utf-8")
                return
            if path.startswith("/fonts/"):
                name = path[len("/fonts/") :]
                f = STRATA_WEB / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                self._send_bytes(200, f.read_bytes(), "font/woff2", cache="max-age=86400")
                return
            if path.startswith(("/web/locales/", "/web/vendor/")):
                sub = resolve_subtree_file(path)
                if sub is None:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                self._send_bytes(200, sub[0].read_bytes(), sub[1])
                return
            if path.startswith("/web/"):
                name = path[len("/web/") :]
                ext = os.path.splitext(name)[1]
                f = STRATA_WEB / name
                if "/" in name or "\\" in name or ext not in FLAT_WEB_TYPES or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                self._send_bytes(200, f.read_bytes(), FLAT_WEB_TYPES[ext])
                return

            if path in ("/health", "/api/health"):
                self._json(200, svc.health_payload())
                return
            if path == "/metrics":
                self._json(200, svc.metrics_payload(all_requests="requests=all" in self.path))
                return
            if path in ("/v1/models", "/models"):
                status_val = "loaded" if svc.loaded() else "unloaded"
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL_NAME,
                                "object": "model",
                                "status": {"value": status_val},
                                "meta": {"n_ctx": svc.max_seq_len},
                            }
                        ],
                    },
                )
                return
            if path == "/slots":
                with svc.state_lock:
                    busy = svc.live["state"] in ("reading", "generating")
                    in_use = int(svc.engine.kv.seq_len) if svc.engine is not None else 0
                self._json(
                    200,
                    [{"id": 0, "n_ctx": svc.max_seq_len, "is_processing": busy, "n_prompt_tokens": in_use}],
                )
                return
            if path == "/settings":
                self._json(200, {"shared": bool(svc.shared_settings), "defaults": svc.shared_settings})
                return
            if path == "/fast-mode":
                self._json(200, {"fast_mode": svc.fast_mode})
                return
            if path == "/mcp":
                self._json(200, {"servers": [], "tools": 0})
                return
            if path == "/config":
                self._json(404, {"error": {"message": "no run config"}})
                return
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body.strip() else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": {"message": f"invalid JSON: {exc}"}})
                return

            if path == "/load":
                svc.load()
                self._json(200, {"status": "loaded"})
                return
            if path == "/unload":
                res = svc.unload()
                self._json(409 if res == "busy" else 200, {"status": res})
                return
            if path == "/settings":
                svc.shared_settings = body.get("defaults")
                self._json(200, {"shared": bool(svc.shared_settings), "defaults": svc.shared_settings})
                return
            if path == "/fast-mode":
                enabled = body.get("enabled")
                new_val = svc.set_fast_mode(not svc.fast_mode if enabled is None else bool(enabled))
                self._json(200, {"fast_mode": new_val})
                return
            if path == "/v1/chat/completions":
                self._handle_chat(body)
                return
            self._json(404, {"error": {"message": "not found"}})

        def _write_sse(self, payload: Any) -> bool:
            chunk = (
                f"data: {json.dumps(payload)}\n\n"
                if not isinstance(payload, str)
                else f"data: {payload}\n\n"
            ).encode("utf-8")
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        def _handle_chat(self, body: Dict[str, Any]) -> None:
            messages = body.get("messages") or []
            stream = bool(body.get("stream", False))
            effort = str(body.get("reasoning_effort") or "none")
            temperature = float(body.get("temperature", 0.0) or 0.0)
            top_p = float(body.get("top_p", 1.0) or 1.0)
            top_k = int(body.get("top_k", 40) or 0)
            raw_max = body.get("max_tokens")
            explicit_max = int(raw_max) if raw_max else 0
            max_tokens = explicit_max if explicit_max > 0 else svc.max_seq_len
            seed = body.get("seed")

            with svc.state_lock:
                svc.live["queued"] += 1
            try:
                with svc.infer_lock:
                    with svc.state_lock:
                        svc.live["queued"] = max(0, svc.live["queued"] - 1)

                    if svc.engine is None:
                        svc.engine = KolibriHybridEngine(
                            model_dir=svc.model_dir,
                            device=torch.device("cuda:0"),
                            persistent_slots_per_layer=svc.persistent_slots,
                            max_seq_len=svc.max_seq_len,
                            prefill_chunk_size=svc.prefill_chunk_size,
                        )
                        svc.last_tokens = []

                    eng = svc.engine
                    prompt_text = format_messages_prompt(messages, reasoning_effort=effort)
                    prompt_ids = eng.tokenizer.encode(prompt_text).ids
                    n_prompt = len(prompt_ids)
                    if n_prompt + max_tokens > svc.max_seq_len:
                        max_tokens = max(1, svc.max_seq_len - n_prompt)

                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()

                    # Exact continuation KV reuse: if prompt extends last_tokens, prefill only new tokens
                    reused = 0
                    if (
                        svc.last_tokens
                        and n_prompt > len(svc.last_tokens)
                        and prompt_ids[: len(svc.last_tokens)] == svc.last_tokens
                        and eng.kv.seq_len == len(svc.last_tokens)
                    ):
                        reused = len(svc.last_tokens)
                        to_prefill = prompt_ids[reused:]
                        reset_kv = False
                    else:
                        to_prefill = prompt_ids
                        reset_kv = True
                        svc.last_tokens = []

                    req_start = time.perf_counter()
                    with svc.state_lock:
                        svc.live.update(
                            state="reading",
                            prompt_tokens=n_prompt,
                            prompt_read=0,
                            prompt_total=len(to_prefill),
                            generated=0,
                            max_tokens=explicit_max,
                            tok_s=0.0,
                        )

                    eng.moe.reset_stats()
                    torch.cuda.synchronize(eng.device)
                    t_p0 = time.perf_counter()

                    def on_chunk(done: int, total: int) -> None:
                        with svc.state_lock:
                            svc.live["prompt_read"] = done
                            svc.live["prompt_total"] = total

                    logits = eng.prefill(
                        to_prefill,
                        promote_misses=False,
                        reset_kv=reset_kv,
                        progress_cb=on_chunk,
                    )
                    torch.cuda.synchronize(eng.device)
                    prompt_ms = (time.perf_counter() - t_p0) * 1000.0
                    prefill_rate = len(to_prefill) / max(prompt_ms / 1000.0, 1e-6)
                    prefill_hits = eng.moe.stats.cache_hit_pairs
                    prefill_pairs = eng.moe.stats.total_pair_evals

                    gen_rng = None
                    if seed is not None:
                        gen_rng = torch.Generator(device=eng.device).manual_seed(int(seed))

                    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    created = int(time.time())
                    eng.moe.reset_stats()

                    gen_ids: List[int] = []
                    emitted_len = 0
                    pending_ids: List[int] = []
                    sse_r_buf: List[str] = []
                    sse_c_buf: List[str] = []
                    sse_tok_cnt = 0
                    in_think = effort != "none"
                    full_answer: List[str] = []
                    full_reasoning: List[str] = []
                    finish_reason = "length"
                    fast = svc.fast_mode

                    def flush_sse_buf() -> bool:
                        nonlocal sse_tok_cnt
                        if not sse_r_buf and not sse_c_buf:
                            sse_tok_cnt = 0
                            return True
                        delta_obj: Dict[str, str] = {}
                        if sse_r_buf:
                            delta_obj["reasoning_content"] = "".join(sse_r_buf)
                            sse_r_buf.clear()
                        if sse_c_buf:
                            delta_obj["content"] = "".join(sse_c_buf)
                            sse_c_buf.clear()
                        sse_tok_cnt = 0
                        return self._write_sse(
                            {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}],
                            }
                        )

                    with svc.state_lock:
                        svc.live.update(state="generating", prefill_tok_s_mean=prefill_rate)

                    if fast:
                        gc.collect()
                        gc.disable()
                    try:
                        t_d0 = time.perf_counter()
                        next_id = eng._sample(logits, temperature, top_k, top_p=top_p, generator=gen_rng)

                        for step_idx in range(max_tokens):
                            if next_id in (EOS_TOKEN_ID, PAD_TOKEN_ID):
                                finish_reason = "stop"
                                break

                            gen_ids.append(next_id)
                            if fast:
                                pending_ids.append(next_id)
                                dec_piece = eng.tokenizer.decode(pending_ids, skip_special_tokens=True)
                                if dec_piece and dec_piece.endswith("\ufffd"):
                                    delta_text = ""
                                else:
                                    delta_text = dec_piece
                                    pending_ids.clear()
                            else:
                                decoded = eng.tokenizer.decode(gen_ids, skip_special_tokens=True)
                                # Hold trailing U+FFFD replacement characters until multi-byte UTF-8 completes
                                clean_len = len(decoded)
                                while clean_len > emitted_len and decoded[clean_len - 1] == "\ufffd":
                                    clean_len -= 1
                                delta_text = decoded[emitted_len:clean_len]
                                emitted_len = clean_len

                            if delta_text:
                                think_ended = False
                                if in_think:
                                    if "</think>" in delta_text:
                                        r_part, c_part = delta_text.split("</think>", 1)
                                        in_think = False
                                        think_ended = True
                                        r_part = r_part.rstrip("\n")
                                        c_part = c_part.lstrip("\n")
                                        if r_part:
                                            sse_r_buf.append(r_part)
                                            full_reasoning.append(r_part)
                                        if c_part:
                                            sse_c_buf.append(c_part)
                                            full_answer.append(c_part)
                                    else:
                                        sse_r_buf.append(delta_text)
                                        full_reasoning.append(delta_text)
                                else:
                                    sse_c_buf.append(delta_text)
                                    full_answer.append(delta_text)

                                if stream:
                                    sse_tok_cnt += 1
                                    if not fast or sse_tok_cnt >= 4 or think_ended:
                                        if not flush_sse_buf():
                                            finish_reason = "disconnect"
                                            break

                            if not fast or (step_idx & 7) == 7:
                                elapsed_dec = max(time.perf_counter() - t_d0, 1e-6)
                                with svc.state_lock:
                                    svc.live["generated"] = len(gen_ids)
                                    svc.live["tok_s"] = len(gen_ids) / elapsed_dec

                            if step_idx + 1 >= max_tokens:
                                break
                            logits = eng.step(next_id, promote_misses=True)
                            next_id = eng._sample(logits, temperature, top_k, top_p=top_p, generator=gen_rng)

                        if fast and pending_ids and finish_reason != "disconnect":
                            rem_text = eng.tokenizer.decode(pending_ids, skip_special_tokens=True)
                            pending_ids.clear()
                            if rem_text:
                                if in_think:
                                    sse_r_buf.append(rem_text)
                                    full_reasoning.append(rem_text)
                                else:
                                    sse_c_buf.append(rem_text)
                                    full_answer.append(rem_text)
                        if stream and finish_reason != "disconnect":
                            if not flush_sse_buf():
                                finish_reason = "disconnect"

                        torch.cuda.synchronize(eng.device)
                        decode_ms = (time.perf_counter() - t_d0) * 1000.0
                    finally:
                        if fast:
                            gc.enable()
                    total_s = time.perf_counter() - req_start

                    # Track exact KV sequence for fast turn continuation
                    svc.last_tokens = list(prompt_ids) + gen_ids if finish_reason != "disconnect" else []

                    total_hits = prefill_hits + eng.moe.stats.cache_hit_pairs
                    total_pairs = prefill_pairs + eng.moe.stats.total_pair_evals
                    hit_rate = total_hits / max(total_pairs, 1)
                    dec_tok_s = (len(gen_ids) / max(decode_ms / 1000.0, 1e-6)) if gen_ids else 0.0

                    usage = {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": len(gen_ids),
                        "total_tokens": n_prompt + len(gen_ids),
                    }
                    if stream:
                        if finish_reason != "disconnect":
                            self._write_sse(
                                {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": MODEL_NAME,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                    "usage": usage,
                                }
                            )
                            self._write_sse("[DONE]")
                    else:
                        self._json(
                            200,
                            {
                                "id": req_id,
                                "object": "chat.completion",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "".join(full_answer),
                                            **(
                                                {"reasoning_content": "".join(full_reasoning)}
                                                if full_reasoning
                                                else {}
                                            ),
                                        },
                                        "finish_reason": finish_reason,
                                    }
                                ],
                                "usage": usage,
                            },
                        )

                    with svc.state_lock:
                        svc.live.update(state="idle", tok_s=dec_tok_s)
                        svc.requests.appendleft(
                            {
                                "time": time.time(),
                                "finish": finish_reason,
                                "prompt_tokens": n_prompt,
                                "reused": reused,
                                "output_tokens": len(gen_ids),
                                "prompt_ms": prompt_ms,
                                "decode_tok_s": dec_tok_s,
                                "hit_rate": hit_rate,
                                "duration_s": total_s,
                            }
                        )
                        svc.totals["requests"] += 1
                        svc.totals["prompt_tokens"] += n_prompt
                        svc.totals["reused"] += reused
                        svc.totals["prompt_ms"] += prompt_ms
                        svc.totals["output_tokens"] += len(gen_ids)
                        svc.totals["decode_ms"] += decode_ms

                        svc.conv_stats["requests"] += 1
                        if reused > 0:
                            svc.conv_stats["requests_reused"] += 1
                        svc.conv_stats["prompt_tokens"] += n_prompt
                        svc.conv_stats["reused_tokens"] += reused
                        svc.conv_stats["last_prompt"] = n_prompt
                        svc.conv_stats["last_reused"] = reused

                    print(
                        f"[{finish_reason}] in={n_prompt} (reused={reused}) out={len(gen_ids)} | "
                        f"prefill={prefill_rate:.1f} tok/s | decode={dec_tok_s:.2f} tok/s | hit={hit_rate*100:.1f}%"
                    )
            except Exception as exc:
                with svc.state_lock:
                    svc.live["state"] = "idle" if svc.loaded() else "unloaded"
                if stream:
                    self._write_sse({"error": {"message": str(exc)}})
                    self._write_sse("[DONE]")
                else:
                    self._json(500, {"error": {"message": str(exc)}})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Kolibri-1 NVFP4 Marlin with Strata's Web UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--persistent-slots", type=int, default=110)
    parser.add_argument("--max-seq-len", type=int, default=100_000)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    args = parser.parse_args()

    svc = KolibriService(
        model_dir=PHASE2_MODEL_DIR,
        persistent_slots=args.persistent_slots,
        max_seq_len=args.max_seq_len,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    print(f"Loading Kolibri-1 W4A16_NVFP4_MARLIN (max_seq_len={args.max_seq_len}, slots/layer={args.persistent_slots})...")
    svc.load()
    print(f"Model loaded in {svc.engine.cache.stats.load_time_s:.2f}s.")
    print(f"Strata Web UI ready at http://{args.host}:{args.port}")
    print(f"For SSH access from local Chrome, run on your local machine:")
    print(f"  ssh -N -L {args.port}:127.0.0.1:{args.port} <user>@<vm-host>")

    server = ThreadedHTTPServer((args.host, args.port), make_handler(svc))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")
        server.server_close()


if __name__ == "__main__":
    main()
