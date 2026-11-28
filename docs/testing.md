# Testing

```bash
python test_pipeline_offline.py       # safe fix applied / risky issue withheld / broken fix rejected
python test_circuit_breaker.py         # broken endpoint routed around + auto-recovers
python test_jinja_templates.py         # Jinja2 templates validated correctly
python test_vllm_backend.py             # vLLM backend URL/payload/parsing correct
```

All three mock only the Ollama HTTP layer — linters, the Verifier, and
(for the circuit breaker test) real timing are exercised for real.
