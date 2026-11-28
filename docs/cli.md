# CLI reference

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
