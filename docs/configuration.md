# config.yaml

All fields live under the top-level `ollama:` key.

```yaml
ollama:
  backend: "ollama"
  endpoints:
    - host: "http://127.0.0.1:11434"
      name: "gpu01"
    - host: "http://127.0.0.1:11435"
      name: "gpu23"
  model: "qwen2.5-coder:32b"
  model_max_context: 32768
  response_reserve_tokens: 2048
  request_timeout: 600
  max_retries: 4
  backoff_base_seconds: 2.0
  heartbeat_interval_seconds: 30
  circuit_breaker_failure_threshold: 2
  circuit_breaker_cooldown_seconds: 120
  # num_gpu: 0
  # num_thread: 8
```

| Field | Default | What it does |
|---|---|---|
| `backend` | `"ollama"` | `"ollama"`, `"vllm"`, or `"llamacpp"`. Which server API to speak — see "Using vLLM or llama.cpp instead of Ollama" below. |
| `endpoints` | one localhost entry | One or more servers matching the chosen backend. See "Running on multiple Ollama servers". |
| `model` | — | The model tag to use, e.g. `qwen2.5-coder:32b`. |
| `model_max_context` | `32768` | The model's real max context window — check with `ollama show <model>`. Used to dynamically size `num_ctx` per request. |
| `response_reserve_tokens` | `2048` | Tokens reserved for the response when sizing context. |
| `request_timeout` | `600` | Seconds to wait for one response before giving up and retrying. Raise this if your hardware is just slow rather than actually stuck — see "Diagnosing slow requests" below. |
| `max_retries` | `4` | Attempts before giving up on one call. Each failed attempt costs a full `request_timeout` — 4 retries at 600s is up to 40 minutes on one stuck call. Fewer retries + longer timeout usually beats many retries + short timeout on consistently-slow hardware. |
| `backoff_base_seconds` | `2.0` | Exponential backoff base between retries (`wait = base ** attempt`). Minor next to a 600s timeout. |
| `heartbeat_interval_seconds` | `30` | Log "still waiting on `<endpoint>`... Xs elapsed" this often while a request is in flight. `0` disables it. |
| `circuit_breaker_failure_threshold` | `2` | Mark an endpoint unhealthy after this many failures in a row. |
| `circuit_breaker_cooldown_seconds` | `120` | How long to route around an unhealthy endpoint before retrying it. |
| `num_gpu` | unset (Ollama decides) | Explicit GPU-layer count. `0` forces pure CPU/RAM inference. See "CPU vs GPU" below. |
| `num_thread` | unset (Ollama decides) | CPU threads for whatever runs on CPU. |
