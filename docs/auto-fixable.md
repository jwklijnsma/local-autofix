# Auto-fixable or not

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

## Fixing everything anyway, including risky issues

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
