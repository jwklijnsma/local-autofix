# Several Ollama servers

Two things both need to be true to get real parallel throughput:

1. **List every endpoint** in `config.yaml` (see reference above).
2. **Pass `--workers N`**, roughly matching your endpoint count:
   ```bash
   python orchestrator.py --local-dir . --workers 2 --apply
   ```

Listing endpoints alone isn't enough — with the default `--workers 1`,
files are reviewed one at a time, so a second server only gets used on the
*next* file, never simultaneously with the first. `--workers` controls how
many files are processed concurrently (a thread pool); the client
round-robins across configured endpoints per request, so `--workers 2`
with 2 endpoints genuinely reviews two files on two different GPUs at
once.

If `--workers` is higher than your endpoint count, requests queue on
whichever endpoint is free — safe, just less parallel benefit. With only
one endpoint, `--workers > 1` still works but you're pipelining requests
against one server (check `OLLAMA_NUM_PARALLEL` server-side if you want it
to actually handle concurrent requests well).

## If one server is broken/slow while others work fine

The circuit breaker (`circuit_breaker_failure_threshold` /
`circuit_breaker_cooldown_seconds` in `config.yaml`) handles this
automatically — an endpoint that fails repeatedly gets skipped for a
cooldown period instead of costing every other request a full
`request_timeout`, and is retried automatically after the cooldown in
case it recovered:

```
[WARNING] gpu1 failed 2 times in a row -- marking unhealthy for 120s ...
[INFO] gpu1 recovered -- clearing circuit breaker state
```

If *every* endpoint is currently unhealthy, it still tries one rather than
refusing outright — an outage isn't assumed permanent.
