# Setup and quick start

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
