# vLLM and llama.cpp

Set `backend: "vllm"` or `backend: "llamacpp"` in `config.yaml` to talk to
a [vLLM](https://github.com/vllm-project/vllm) or
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) server
instead of Ollama. Both expose the same OpenAI-compatible
`/v1/chat/completions` API, so they share identical request/response
handling in this pipeline -- only the config value and default port differ.
Everything else (all the agents, `--workers`, the circuit breaker) works
identically regardless of backend.

```yaml
# vLLM
ollama:
  backend: "vllm"
  endpoints:
    - host: "http://127.0.0.1:8000"   # vLLM's default port
      name: "vllm0"
  model: "Qwen2.5-Coder-32B-Instruct"   # HF repo id/path, not an Ollama tag

# llama.cpp
ollama:
  backend: "llamacpp"
  endpoints:
    - host: "http://127.0.0.1:8080"   # llama-server's default port
      name: "llamacpp0"
  model: "qwen2.5-coder-32b-instruct-q4_k_m.gguf"   # mostly ignored by llama-server (it serves whatever .gguf was loaded at startup), but keep it descriptive
```

Start the server separately -- this pipeline doesn't launch either for you:
```bash
# vLLM
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000

# llama.cpp
llama-server -m qwen2.5-coder-32b-instruct-q4_k_m.gguf --port 8080
```

**What's different from the Ollama backend, for both:**
- Requests go to `/v1/chat/completions` (OpenAI-compatible), not `/api/chat`.
- `num_gpu` / `num_thread` are Ollama-only concepts and are ignored (with a
  logged warning) under `backend: "vllm"`/`"llamacpp"` -- GPU/thread
  placement is controlled via each server's own startup flags instead
  (vLLM: `--tensor-parallel-size`, `--gpu-memory-utilization`; llama.cpp:
  `-ngl`, `--threads`), not per request.
- `model_max_context` is still used client-side to size how much content
  gets packed into a prompt, but the real context limit is set server-side
  at startup (vLLM: `--max-model-len`; llama.cpp: `-c`) -- make sure the
  two agree, or you may hit a server-side context error instead of a
  clean local one.

Worth knowing: **llama.cpp is the inference engine Ollama itself is built
on** (GGUF format, similar underlying engine) -- Ollama is a friendlier
wrapper around it, not a separate technology. Using `backend: "llamacpp"`
directly only really pays off if you specifically need `llama-server`'s
own flags/features that Ollama doesn't expose; otherwise Ollama's
automatic model management (pull/CPU-fallback/etc.) is usually less setup
for the same underlying engine.

`test_vllm_backend.py` covers the URL/payload/response-parsing for both,
and confirms the default Ollama backend is unaffected.
