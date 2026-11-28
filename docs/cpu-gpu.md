# CPU vs GPU

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
