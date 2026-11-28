# Output

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
