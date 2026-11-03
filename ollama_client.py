"""
ollama_client.py

Thin wrapper around the Ollama HTTP API used by every agent in the pipeline.

Responsibilities:
  - Send chat/generate requests to one or more Ollama instances (for simple
    multi-instance load spreading, e.g. two systemd services on different
    ports bound to different GPUs).
  - Dynamically size num_ctx per-request so we never truncate the important
    input, but also don't ask for a bigger context window than we need.
  - Retry transient failures (connection errors, timeouts, 5xx) with
    exponential backoff.
  - Rough token estimation (no tokenizer dependency required, good enough
    for sizing context windows -- swap in tiktoken/model tokenizer if you
    want exact counts).
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import requests

logger = logging.getLogger("ollama_client")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class OllamaEndpoint:
    """One Ollama instance (e.g. one systemd service bound to a GPU pair)."""
    host: str  # e.g. "http://127.0.0.1:11434"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.host


@dataclass
class OllamaClientConfig:
    endpoints: list[OllamaEndpoint] = field(default_factory=lambda: [
        OllamaEndpoint("http://127.0.0.1:11434", "primary"),
    ])
    model: str = "qwen2.5:32b"
    request_timeout: int = 600
    max_retries: int = 4
    backoff_base_seconds: float = 2.0
    # Model's real max context window (check `ollama show <model>`).
    model_max_context: int = 32768
    # Tokens reserved for the model's response.
    response_reserve_tokens: int = 2048
    # Safety margin added on top of the naive char/4 estimate.
    estimate_buffer_pct: float = 0.05
    # How often (seconds) to log a "still waiting" heartbeat during a
    # request. Set to 0 to disable.
    heartbeat_interval_seconds: int = 30
    # Circuit breaker: after this many consecutive failures on one
    # endpoint, stop sending it new requests for circuit_breaker_cooldown_seconds
    # (as long as at least one other endpoint is available). Prevents every
    # other request from paying a full request_timeout on a broken server.
    circuit_breaker_failure_threshold: int = 2
    circuit_breaker_cooldown_seconds: int = 120
    # Explicit control over CPU/RAM vs GPU/VRAM usage. Ollama already
    # auto-offloads layers to CPU+RAM when a model doesn't fully fit in
    # VRAM -- these let you override that instead of relying on the
    # automatic guess.
    #   num_gpu: how many model layers to put on GPU. None = let Ollama
    #     decide automatically (default). 0 = force pure CPU inference,
    #     no GPU at all. A specific number = put exactly that many layers
    #     on GPU, the rest on CPU/RAM (partial offload).
    #   num_thread: CPU threads to use for whatever runs on CPU (the
    #     offloaded layers, or everything if num_gpu=0). None = Ollama's
    #     default (usually all physical cores).
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    # Which server API to speak. "ollama" = Ollama's native /api/chat.
    # "vllm" = vLLM's OpenAI-compatible /v1/chat/completions endpoint.
    # Endpoints must all be running the same backend type.
    backend: str = "ollama"


# --------------------------------------------------------------------------- #
# Token / context sizing helpers
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """
    Cheap, dependency-free token estimate (~4 chars/token for English text,
    a bit worse for code -- we bias slightly high on purpose since
    under-sizing num_ctx is the expensive mistake, not over-sizing it).
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


def compute_num_ctx(
    prompt_text: str,
    cfg: OllamaClientConfig,
    response_reserve_tokens: Optional[int] = None,
) -> int:
    """
    Dynamic context calculation, mirroring the approach described in the
    HOSTKEY writeup:

        max_input = max_context - response_reserve
        needed    = estimated_input_tokens * (1 + buffer)
        actual    = min(needed, max_input)
        num_ctx   = min(actual + response_reserve, max_context)

    This keeps num_ctx as small as possible (faster inference, less VRAM)
    while guaranteeing we never truncate the input we actually need.
    """
    reserve = response_reserve_tokens or cfg.response_reserve_tokens
    max_context = cfg.model_max_context

    max_input = max_context - reserve
    estimated = estimate_tokens(prompt_text)
    needed = int(estimated * (1 + cfg.estimate_buffer_pct))

    actual = min(needed, max_input)
    num_ctx = min(actual + reserve, max_context)

    # Ollama context windows are typically most efficient in round numbers;
    # round up to the nearest 512 without exceeding model_max_context.
    num_ctx = min(((num_ctx + 511) // 512) * 512, max_context)
    return max(num_ctx, 1024)


def single_call_input_budget(
    cfg: OllamaClientConfig,
    prompt_overhead_tokens: int = 400,
    safety_factor: float = 0.85,
) -> int:
    """
    How many tokens of *content* can safely go into one prompt for this
    model, after leaving room for the response and a safety margin.

    Used to decide whether a batch of file content fits in a single call,
    or needs to be split (see chunk_text / pack_into_batches below).
    """
    max_input = cfg.model_max_context - cfg.response_reserve_tokens - prompt_overhead_tokens
    return max(int(max_input * safety_factor), 512)


def chunk_text(text: str, max_tokens: int) -> list[str]:
    """
    Split text into chunks that each fit within max_tokens, breaking on
    line boundaries so we never cut a command or code block mid-line.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append("".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append("".join(current))

    return chunks or [text]


def pack_into_batches(
    entries: Iterable[tuple[str, str]],
    budget_tokens: int,
) -> list[list[tuple[str, str]]]:
    """
    Greedy bin-packing of (label, content) entries into batches that each
    fit within budget_tokens. Any single entry bigger than the budget on
    its own gets split first via chunk_text, with parts labeled
    "<label> (part i/n)" so downstream agents can tell they're related.
    """
    normalized: list[tuple[str, str]] = []
    for label, content in entries:
        if estimate_tokens(content) <= budget_tokens:
            normalized.append((label, content))
            continue
        parts = chunk_text(content, budget_tokens)
        for i, part in enumerate(parts, 1):
            normalized.append((f"{label} (part {i}/{len(parts)})", part))

    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0

    for label, content in normalized:
        t = estimate_tokens(content)
        if current and current_tokens + t > budget_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((label, content))
        current_tokens += t

    if current:
        batches.append(current)

    return batches


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class OllamaClient:
    """
    Round-robins across configured endpoints (useful if you're running two
    Ollama systemd services on separate GPU pairs, each reachable on its own
    port, sharing one model directory).

    Includes a simple circuit breaker: an endpoint that fails
    circuit_breaker_failure_threshold times in a row is skipped for
    circuit_breaker_cooldown_seconds (as long as another endpoint is
    available), so one broken/slow server doesn't cost every other request
    a full request_timeout. It's automatically retried after the cooldown
    in case it recovers.

    Thread-safe: the same OllamaClient instance can be shared across a
    ThreadPoolExecutor. itertools.cycle and the health-tracking dict are
    both guarded by a lock since multiple worker threads pick endpoints
    concurrently.
    """

    def __init__(self, cfg: OllamaClientConfig):
        self.cfg = cfg
        self._endpoint_cycle = itertools.cycle(cfg.endpoints)
        self._cycle_lock = threading.Lock()
        # host -> {"failures": int, "unhealthy_until": float (monotonic time)}
        self._health: dict[str, dict[str, float]] = {
            ep.host: {"failures": 0, "unhealthy_until": 0.0} for ep in cfg.endpoints
        }

    def _is_healthy(self, endpoint: OllamaEndpoint, now: float) -> bool:
        return self._health[endpoint.host]["unhealthy_until"] <= now

    def _record_success(self, endpoint: OllamaEndpoint):
        with self._cycle_lock:
            h = self._health[endpoint.host]
            if h["failures"] or h["unhealthy_until"]:
                logger.info("%s recovered -- clearing circuit breaker state", endpoint.name)
            h["failures"] = 0
            h["unhealthy_until"] = 0.0

    def _record_failure(self, endpoint: OllamaEndpoint):
        with self._cycle_lock:
            h = self._health[endpoint.host]
            h["failures"] += 1
            if h["failures"] >= self.cfg.circuit_breaker_failure_threshold:
                cooldown = self.cfg.circuit_breaker_cooldown_seconds
                h["unhealthy_until"] = time.monotonic() + cooldown
                logger.warning(
                    "%s failed %d times in a row -- marking unhealthy for %ds "
                    "(new requests will prefer other endpoints until then)",
                    endpoint.name, h["failures"], cooldown,
                )

    def _next_endpoint(self) -> OllamaEndpoint:
        """
        Round-robins to the next endpoint, skipping ones currently marked
        unhealthy by the circuit breaker -- unless ALL endpoints are
        unhealthy, in which case we have to try something, so we fall back
        to whichever came up first in the cycle rather than refusing to
        make a request at all.
        """
        with self._cycle_lock:
            now = time.monotonic()
            first_seen = None
            for _ in range(len(self.cfg.endpoints)):
                candidate = next(self._endpoint_cycle)
                if first_seen is None:
                    first_seen = candidate
                if self._is_healthy(candidate, now):
                    return candidate
            logger.warning("All endpoints are marked unhealthy -- retrying %s anyway", first_seen.name)
            return first_seen

    def _post_with_heartbeat(self, url: str, payload: dict, endpoint_name: str, heartbeat_interval: int = 30):
        """
        Runs the actual POST in a background thread while the calling
        thread logs a heartbeat every `heartbeat_interval` seconds, so long
        requests show visible progress ("still waiting, 90s so far...")
        instead of going silent until they complete or hit the timeout.
        """
        result: dict[str, Any] = {}
        done = threading.Event()

        def _do_request():
            try:
                result["response"] = requests.post(url, json=payload, timeout=self.cfg.request_timeout)
            except Exception as e:  # noqa: BLE001 -- re-raised on the calling thread below
                result["error"] = e
            finally:
                done.set()

        start = time.monotonic()
        worker = threading.Thread(target=_do_request, daemon=True)
        worker.start()

        while not done.wait(timeout=heartbeat_interval):
            elapsed = time.monotonic() - start
            logger.info("Still waiting on %s... %.0fs elapsed so far (timeout at %ds)",
                        endpoint_name, elapsed, self.cfg.request_timeout)

        worker.join()
        elapsed = time.monotonic() - start

        if "error" in result:
            raise result["error"]
        logger.info("Response from %s in %.1fs", endpoint_name, elapsed)
        return result["response"]

    def _build_request(
        self, endpoint: OllamaEndpoint, model: str, messages: list[dict[str, str]],
        temperature: float, num_ctx: int, json_mode: bool,
    ) -> tuple[str, dict]:
        """Builds the (url, payload) for this backend's actual wire format."""
        if self.cfg.backend in ("vllm", "llamacpp"):
            # Both vLLM and llama.cpp's server (llama-server) expose the
            # same OpenAI-compatible /v1/chat/completions API, so they
            # share this code path -- only the URL and log wording differ.
            url = f"{endpoint.host.rstrip('/')}/v1/chat/completions"
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                # Neither backend takes a context-window size per request
                # (vLLM: --max-model-len at startup; llama.cpp: -c at
                # startup) -- but both respect max_tokens as a
                # response-length cap, so response_reserve_tokens covers that.
                "max_tokens": self.cfg.response_reserve_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self.cfg.num_gpu is not None or self.cfg.num_thread is not None:
                logger.warning(
                    "num_gpu/num_thread are ignored with backend=%s -- GPU/thread "
                    "placement for that backend is controlled via its own server "
                    "startup flags, not per-request.",
                    self.cfg.backend,
                )
            return url, payload

        # Default: Ollama's native API.
        url = f"{endpoint.host.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                **({"num_gpu": self.cfg.num_gpu} if self.cfg.num_gpu is not None else {}),
                **({"num_thread": self.cfg.num_thread} if self.cfg.num_thread is not None else {}),
            },
        }
        if json_mode:
            payload["format"] = "json"
        return url, payload

    @staticmethod
    def _extract_content(backend: str, data: dict) -> str:
        """Pulls the assistant's text out of this backend's response shape."""
        if backend in ("vllm", "llamacpp"):
            # OpenAI-compatible shape: {"choices": [{"message": {"content": "..."}}]}
            choices = data.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")
        # Ollama's native shape: {"message": {"content": "..."}}
        return data.get("message", {}).get("content", "")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        num_ctx: Optional[int] = None,
        json_mode: bool = False,
        heartbeat_interval: Optional[int] = None,
    ) -> str:
        """
        Send a chat request. Returns the assistant's text content.
        Retries with exponential backoff on network errors / 5xx / timeouts.
        Logs a heartbeat every `heartbeat_interval` seconds while waiting,
        so long requests are visible in progress, not just after the fact.
        Defaults to cfg.heartbeat_interval_seconds; pass 0 to disable.

        Works against either backend (cfg.backend = "ollama" or "vllm") --
        the wire format differs but this method's behavior and signature
        don't change either way.
        """
        if heartbeat_interval is None:
            heartbeat_interval = self.cfg.heartbeat_interval_seconds
        model = model or self.cfg.model

        if num_ctx is None:
            full_text = "\n".join(m.get("content", "") for m in messages)
            num_ctx = compute_num_ctx(full_text, self.cfg)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            endpoint = self._next_endpoint()
            url, payload = self._build_request(endpoint, model, messages, temperature, num_ctx, json_mode)
            try:
                logger.info(
                    "POST %s model=%s num_ctx=%d attempt=%d/%d",
                    url, model, num_ctx, attempt, self.cfg.max_retries,
                )
                if heartbeat_interval > 0:
                    resp = self._post_with_heartbeat(url, payload, endpoint.name, heartbeat_interval)
                else:
                    resp = requests.post(url, json=payload, timeout=self.cfg.request_timeout)
                if resp.status_code >= 500:
                    raise RuntimeError(f"{endpoint.name} returned {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                self._record_success(endpoint)
                return self._extract_content(self.cfg.backend, data)
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as e:
                last_err = e
                self._record_failure(endpoint)
                if attempt == self.cfg.max_retries:
                    break
                wait = self.cfg.backoff_base_seconds ** attempt
                logger.warning(
                    "Request to %s failed (%s). Retrying in %.0fs...",
                    endpoint.name, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"All {self.cfg.max_retries} attempts failed: {last_err}")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: Optional[int] = None,
    ) -> Any:
        """Convenience wrapper: request JSON mode and parse the result."""
        raw = self.chat(messages, model=model, temperature=temperature, num_ctx=num_ctx, json_mode=True)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
