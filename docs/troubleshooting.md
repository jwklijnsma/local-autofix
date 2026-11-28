# Troubleshooting

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
