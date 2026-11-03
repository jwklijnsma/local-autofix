# Code Review & Auto-Fix Pipeline (Ollama edition)

Reviews Python and Ansible/YAML code using real linters plus an LLM for
semantic/security review, and automatically fixes what's safe to fix —
verifying every fix before it's written, and never silently touching
anything flagged as risky.

Runs entirely against a local [Ollama](https://ollama.com) server (or
several) — no code leaves your machine.

## Why this is different from just asking an LLM to fix your code

1. **Grounded findings, not hallucinated ones.** `ruff`, `bandit`,
   `yamllint`, and `ansible-lint` run first, as real subprocesses. The LLM
   (CodeReviewer) triages *their* output and adds what they can't catch
   (hardcoded secrets, shell injection, non-idempotent Ansible tasks) — it
   isn't inventing issues from nothing.
2. **Fixes are verified before they're written, and risky ones are never
   auto-applied by default.** CodeReviewer marks every issue
   `auto_fixable: true/false`. Only `true` issues reach CodeFixer, unless
   you explicitly opt out with `--include-risky`. After a fix, the
   Verifier (plain Python, no LLM) checks the result still parses and
   re-runs the linter — if the finding count went up instead of down, the
   fix is rejected and the original file is left alone untouched.

## How a file flows through the pipeline

```
RepoScanner ─▶ static_analysis (ruff/bandit/yamllint/ansible-lint)
           ─▶ CodeReviewer (LLM: triage findings + add semantic issues,
                             mark each auto_fixable true/false)
           ─▶ CodeFixer (LLM: fix only the auto_fixable ones)
           ─▶ Verifier (syntax check + regression check, no LLM)
           ─▶ write back — only if --apply, and only if Verifier accepted it
```

## Files

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

## Setup

```bash
pip install -r requirements.txt
```

`ruff` / `bandit` / `yamllint` / `ansible-lint` are all optional — install
whichever match what you're reviewing. A missing tool is skipped with a
log warning, never a crash.

Edit `config.yaml` with your Ollama host(s) and model — see the full
reference below. A coder-tuned model (`qwen2.5-coder`, `qwen3-coder`,
etc.) does noticeably better at this than a general chat model.

## Quick start

```bash
# Dry-run against a local repo -- writes a report, touches NO files
python orchestrator.py --local-dir /path/to/repo

# Actually apply verified fixes (backs up originals first)
python orchestrator.py --local-dir /path/to/repo --apply

# Against a git repo instead of a local path
python orchestrator.py --repo-url https://gitlab.example.com/team/infra.git --branch main --apply
```

**Always dry-run first** on a repo you haven't used this tool on before,
and read `review_output/review_report.json` before turning on `--apply`.
Treat this like any other automated refactor: review the diffs, run your
test suite, and don't point `--apply` at something without version control
under it.

## CLI reference

| Flag | Default | What it does |
|---|---|---|
| `--local-dir PATH` | — | Review a local directory. Mutually exclusive with `--repo-url`. |
| `--repo-url URL` | — | Clone and review a git repo instead. |
| `--branch NAME` | `master` | Branch to clone, with `--repo-url`. |
| `--config PATH` | `config.yaml` | Path to the config file. |
| `--work-dir PATH` | `./_clones` | Scratch directory for git clones. |
| `--output-dir PATH` | `./review_output` | Where reports/backups are written. |
| `--apply` | off | Actually write verified fixes to disk. Without it, nothing on disk changes — you get a report only. |
| `--include-risky` | off | Also attempt fixes for issues marked `auto_fixable: false` (hardcoded secrets, injection risk, logic bugs, etc). The Verifier still applies. See below. |
| `--only-python` | off | Skip YAML/Ansible files. |
| `--only-ansible` | off | Skip Python files. |
| `--workers N` | `1` | Review N files concurrently. See "Running on multiple Ollama servers" below — listing endpoints alone isn't enough, this is the other half. |
| `--keep-clone` | off | Don't delete a git clone after processing (no effect with `--local-dir`, which is never deleted anyway). |

## config.yaml reference

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

## Output

Everything lands in `--output-dir` (default `./review_output/`):

- **`review_report.json`** — one entry per file: static finding count,
  full issue list (with `auto_fixable`), verification result, and a
  unified diff for anything that was fixed.
- **`manual_review_needed.json`** — every issue marked
  `auto_fixable: false`, flattened across all files, for a human to work
  through. This is usually where the important stuff ends up — hardcoded
  secrets, injection risk, logic bugs the LLM wasn't confident enough to
  touch on its own.
- **`backups/`** — original content of every file that was actually
  modified (only created with `--apply`), mirroring the repo's directory
  structure. Restore with a plain `cp`.

## How auto-fixable is decided

CodeReviewer is deliberately conservative:

- **Usually auto-fixable**: unused imports, unsorted imports, missing
  quoting, YAML indentation, `0777`-style permission fixes, pinning an
  unpinned version, swapping a `shell:`/`command:` task for the equivalent
  proper Ansible module in simple cases.
- **Usually NOT auto-fixable**: hardcoded secrets (removing the value
  doesn't tell you where it *should* come from), shell/command injection
  (the real fix is often architectural), logic bugs, anything touching
  business logic the file alone doesn't fully explain.

Tune this by editing the `SYSTEM_PROMPT` in `pipeline/code_reviewer.py` if
it's too conservative or too aggressive for your codebase.

### Fixing everything anyway, including risky issues

```bash
python orchestrator.py --local-dir . --apply --include-risky
```

This bypasses CodeReviewer's `auto_fixable` gate — every issue found gets
sent to CodeFixer, including hardcoded secrets and injection risk. It does
**not** disable the Verifier: a fix still has to pass its syntax check and
can't make the finding count worse, or it's still rejected. What this
removes is only the upstream judgment call that an issue is too risky to
*attempt* automatically — read the diffs and check `backups/` carefully
afterward. This is for someone who'd rather review a larger batch of
changes after the fact than have some skipped up front, not for
unattended use on a repo you haven't checked this tool's output on before.

## Running on multiple Ollama servers

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

### If one server is broken/slow while others work fine

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

## Diagnosing slow requests

A slow request isn't necessarily a crash or a bug — check what's actually
happening before changing anything:

```bash
# Is it actually computing, or stuck?
nvidia-smi          # near 0% utilization for a long stretch = stuck, not just slow
ollama ps            # shows the CPU/GPU split Ollama is actually using

# Is the model cold-loading? First request after a restart can take
# 1-3 minutes just to load into VRAM -- pre-warm before a real run:
ollama run <model> "say hi"
```

If it's genuinely computing (high GPU util) but just slow, raise
`request_timeout` and lower `max_retries` rather than retrying — retrying
a request that's slow-but-working just repeats the same wait.

## Using vLLM or llama.cpp instead of Ollama

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

## CPU vs GPU

Ollama already auto-offloads model layers to system RAM/CPU when a model
doesn't fully fit in VRAM — this happens by default, no configuration
needed, and is a common reason some requests are much slower than others.

Override the automatic guess explicitly if needed:

```yaml
ollama:
  num_gpu: 0        # force pure CPU+RAM, no GPU at all
  num_thread: 8      # CPU threads to use (omit for Ollama's default)
```

- `num_gpu: 0` forces the entire model onto CPU/RAM. Never OOMs regardless
  of model size, but is typically 10-50x slower than GPU. Fine for a
  handful of files, not practical for hundreds.
- `num_gpu: <N>` puts exactly N layers on GPU, the rest on CPU (partial
  offload) — useful for deliberate control instead of the automatic guess.
- Leave both unset (default) to let Ollama decide, which is usually right.

If you're routinely needing `num_gpu: 0` to avoid OOM, that's generally a
sign to pick a smaller model rather than run a bigger one slowly on CPU.

## Ansible Jinja2 templates

Files under `templates/` (or any YAML containing `{{ }}` / `{% %}`) are
handled specially. Plain `yaml.safe_load` cannot parse Jinja2 syntax at
all — `{{ var }}` reads as an invalid YAML flow mapping — so without this,
both the Verifier's syntax check and `yamllint` would reject every fix to
this class of file regardless of correctness. These files are now
validated as Jinja2 templates (via the `jinja2` package) instead of as
YAML; `yamllint` is skipped for them entirely, while `ansible-lint`
(which is Jinja-aware) still covers them via its normal repo-wide scan.

## Known limitations

- **Large files**: unlike token-budget-aware pipelines, this one doesn't
  chunk oversized single files. `RepoScanner.max_file_bytes` (default
  500KB) just skips anything bigger, logged as a warning. Rare for source
  code; raise the limit or split the file if you hit it.
- **`ansible-lint`** needs your repo to look like an actual Ansible
  project (playbooks/roles structure) to give useful output, so it runs
  once against the whole repo rather than per-file. `yamllint` runs
  per-file regardless, so plain YAML repos still get signal without
  `ansible-lint` installed.
- **`--workers`** parallelizes across *files*, not within a single file's
  review→fix→verify sequence — those three steps for one file still run
  sequentially, just possibly alongside other files' sequences on other
  threads.

## Testing

```bash
python test_pipeline_offline.py       # safe fix applied / risky issue withheld / broken fix rejected
python test_circuit_breaker.py         # broken endpoint routed around + auto-recovers
python test_jinja_templates.py         # Jinja2 templates validated correctly
python test_vllm_backend.py             # vLLM backend URL/payload/parsing correct
```

All three mock only the Ollama HTTP layer — linters, the Verifier, and
(for the circuit breaker test) real timing are exercised for real.
