"""
code_reviewer.py -- CodeReviewer agent

Given one file's content plus the static-analysis findings already detected
by real tools (ruff/bandit/yamllint/ansible-lint), this agent:

  1. Correlates and prioritizes the static findings.
  2. Adds semantic/logic/security issues the linters can't catch: hardcoded
     secrets, missing error handling, non-idempotent Ansible tasks, shell
     injection risk via unsanitized variables, unpinned image/package
     versions, missing 'become'/permissions issues, etc.
  3. Marks each issue as auto_fixable or not -- CodeFixer only touches
     issues marked auto_fixable, so anything genuinely risky (large
     architectural concerns, ambiguous business logic) gets surfaced for a
     human instead of silently rewritten.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ollama_client import OllamaClient
from pipeline.static_analysis import Finding

logger = logging.getLogger("code_reviewer")

SYSTEM_PROMPT = """\
You are CodeReviewer, a senior engineer reviewing a single source file \
({language}). You are given the file's content and a list of findings \
already detected by static analysis tools (ruff/bandit for Python, \
yamllint/ansible-lint for YAML/Ansible).

Your job:
1. Keep every static finding as an issue (don't drop them), but assign \
each a clear severity and a one-line explanation in your own words if the \
tool's message is cryptic.
2. Add any additional issues the static tools would not catch: hardcoded \
secrets/credentials, missing error handling, shell/command injection risk \
via unsanitized input, non-idempotent Ansible tasks (e.g. shell/command \
modules that should be a proper module, missing 'changed_when'), unpinned \
dependency/image versions, overly broad file permissions, missing \
'become'/privilege issues, or logic bugs you can see directly in the code.
3. For EVERY issue, decide auto_fixable:
   - true: the fix is mechanical/local and safe to apply automatically \
(e.g. remove unused import, pin a version, fix YAML indentation, add \
quoting, replace shell module with proper Ansible module for a simple \
case, add a missing default value).
   - false: fixing it requires understanding intent/business logic beyond \
the file, could change behavior in a way you're not certain about, or is a \
larger architectural concern. Set this whenever you're not fully confident \
an automatic fix is safe.

Return ONLY a JSON object of this exact shape, nothing else:
{{
  "issues": [
    {{
      "line": <int or null>,
      "severity": "low|medium|high",
      "source": "static|review",
      "rule": "<tool rule id or null>",
      "problem": "<description>",
      "auto_fixable": true|false
    }}
  ]
}}
"""


class CodeReviewer:
    def __init__(self, client: OllamaClient):
        self.client = client

    def review(self, file_path: Path, content: str, static_findings: list[Finding]) -> list[dict]:
        language = "Python" if file_path.suffix.lower() == ".py" else "YAML/Ansible"
        system_prompt = SYSTEM_PROMPT.format(language=language)

        findings_text = "\n".join(
            f"- [{f.tool}:{f.rule or '?'}] line {f.line}: {f.message} (severity: {f.severity})"
            for f in static_findings
        ) or "(no static findings)"

        user_prompt = (
            f"FILE: {file_path.name}\n\n"
            f"STATIC ANALYSIS FINDINGS:\n{findings_text}\n\n"
            f"FILE CONTENT:\n```\n{content}\n```\n\n"
            "Review this file now."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result = self.client.chat_json(messages, temperature=0.0)
            issues = result.get("issues", [])
            if not isinstance(issues, list):
                raise ValueError("issues is not a list")
        except Exception as e:
            logger.warning("CodeReviewer LLM call failed for %s (%s); falling back to static findings only", file_path, e)
            issues = [
                {
                    "line": f.line, "severity": f.severity, "source": "static",
                    "rule": f.rule, "problem": f.message, "auto_fixable": False,
                }
                for f in static_findings
            ]

        logger.info("CodeReviewer found %d issue(s) in %s (%d auto-fixable)",
                    len(issues), file_path.name, sum(1 for i in issues if i.get("auto_fixable")))
        return issues
