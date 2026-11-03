"""
code_fixer.py -- CodeFixer agent

Applies fixes for the issues CodeReviewer marked auto_fixable, returning
the full corrected file content. Unlike prose, code has semantically
significant whitespace (Python, YAML) -- the prompt is deliberately strict
about preserving exact formatting/indentation style outside of what's
being fixed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ollama_client import OllamaClient

logger = logging.getLogger("code_fixer")

SYSTEM_PROMPT = """\
You are CodeFixer, an engineer making the smallest possible correct change \
to a {language} file to resolve a specific list of issues.

Rules:
- Fix ONLY the listed issues. Do not refactor, reformat, rename things, or \
"improve" anything not explicitly listed.
- Preserve exact indentation style, line endings, quoting style, and \
comments that aren't directly related to a fix.
- {language} is whitespace/indentation-sensitive -- be exact.
- If applying a fix would require information not present in this file \
(e.g. a variable defined elsewhere, external context), make the most \
conservative safe change or leave that specific issue unfixed rather than \
guessing.
- Return ONLY the complete corrected file content, nothing else -- no \
explanation, no markdown code fences wrapping the whole file, no preamble.
"""


class CodeFixer:
    def __init__(self, client: OllamaClient):
        self.client = client

    def fix(self, file_path: Path, content: str, issues: list[dict]) -> str:
        if not issues:
            return content

        language = "Python" if file_path.suffix.lower() == ".py" else "YAML/Ansible"
        system_prompt = SYSTEM_PROMPT.format(language=language)

        issues_text = "\n".join(
            f"- [{i.get('severity', 'medium')}] line {i.get('line')}: {i.get('problem', '')}"
            for i in issues
        )

        user_prompt = (
            f"FILE: {file_path.name}\n\n"
            f"CURRENT CONTENT:\n```\n{content}\n```\n\n"
            f"ISSUES TO FIX:\n{issues_text}\n\n"
            "Return the corrected file content in full."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        fixed = self.client.chat(messages, temperature=0.0)

        # Model sometimes wraps the whole response in a fence despite
        # instructions -- strip a single outer fence if present.
        stripped = fixed.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            stripped = "\n".join(lines)

        logger.info("CodeFixer applied %d issue(s) to %s", len(issues), file_path.name)
        return stripped + ("\n" if not stripped.endswith("\n") else "")
