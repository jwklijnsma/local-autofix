# Files

| File | Role |
|---|---|
| `ollama_client.py` | HTTP client — retries, dynamic context sizing, multi-endpoint round-robin, circuit breaker, heartbeat logging, CPU/GPU offload control. Supports Ollama's native API and the OpenAI-compatible API shared by vLLM and llama.cpp (`backend: "ollama"` / `"vllm"` / `"llamacpp"` in `config.yaml`). |
| `pipeline/repo_scanner.py` | Clones a repo or uses a local dir; finds `.py`/`.yml`/`.yaml` files (tests included). |
| `pipeline/static_analysis.py` | Runs `ruff`, `bandit`, `yamllint`, `ansible-lint` as real subprocesses and parses their output into a common `Finding` format. Any missing tool is skipped with a warning, never a crash. Also handles Jinja2-templated YAML correctly (see below). |
| `pipeline/code_reviewer.py` | **CodeReviewer** (LLM) — triages static findings, adds semantic/security issues, marks each `auto_fixable` true/false. |
| `pipeline/code_fixer.py` | **CodeFixer** (LLM) — fixes only the issues it's given, preserving formatting/indentation exactly elsewhere. |
| `pipeline/verifier.py` | **Verifier** (no LLM) — syntax check (`py_compile` / YAML / Jinja2, whichever applies) + re-run linter; rejects any fix that doesn't parse or that regresses. |
| `orchestrator.py` | Ties it all together — the CLI entrypoint. |
| `config.yaml` | Ollama endpoint(s), model, and all tuning knobs. |
| `test_pipeline_offline.py` | Mocks only the LLM calls — linters and the Verifier are real. Proves a safe fix gets applied, a risky issue gets left alone, and a broken fix gets rejected. |
| `test_circuit_breaker.py` | Proves a consistently-failing endpoint gets routed around, and retried after it recovers. |
| `test_jinja_templates.py` | Proves Jinja2-templated YAML (Ansible `templates/`) is validated correctly instead of always failing plain YAML parsing. |
| `test_vllm_backend.py` | Proves the vLLM/llama.cpp backend hits the right endpoint with the right payload shape, and that the default Ollama backend is unaffected. |
