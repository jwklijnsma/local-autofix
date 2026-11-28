# How it works

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
