"""
verifier.py -- Verifier (not an LLM agent -- deterministic safety check)

Before any fix from CodeFixer is written back to the real file, this module
checks:

  1. Syntax validity -- `py_compile` for Python, `yaml.safe_load` for YAML.
     A syntactically broken "fix" is rejected outright, no exceptions.
  2. Regression check -- re-runs the relevant static analyzer(s) against
     the fixed content and compares the finding count to before. If the
     fix introduced MORE issues than it resolved, it's rejected.

This is deliberately not an LLM call: correctness of "does this parse" and
"did the finding count go down" should never depend on a model's opinion.
"""

from __future__ import annotations

import logging
import py_compile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from pipeline.static_analysis import Finding, is_jinja_template, run_single_file

logger = logging.getLogger("verifier")


@dataclass
class VerificationResult:
    accepted: bool
    reason: str
    findings_before: int
    findings_after: int


def _check_syntax(file_path: Path, content: str) -> tuple[bool, str]:
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            py_compile.compile(str(tmp_path), doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        finally:
            tmp_path.unlink(missing_ok=True)

    elif suffix == ".j2" or is_jinja_template(content):
        try:
            import jinja2
        except ImportError:
            logger.warning(
                "%s looks like a Jinja2 template but the 'jinja2' package isn't installed "
                "(pip install jinja2) -- skipping syntax check, only the regression check applies.",
                file_path,
            )
            return True, ""
        try:
            jinja2.Environment().parse(content)
            return True, ""
        except jinja2.TemplateSyntaxError as e:
            return False, f"Jinja2 syntax error: {e}"

    else:
        try:
            # safe_load_all so multi-document YAML (common in Ansible) is fine.
            list(yaml.safe_load_all(content))
            return True, ""
        except yaml.YAMLError as e:
            return False, str(e)


def verify_fix(
    repo_root: Path,
    file_path: Path,
    original_content: str,
    fixed_content: str,
    findings_before: list[Finding],
) -> VerificationResult:
    if fixed_content.strip() == original_content.strip():
        return VerificationResult(
            accepted=False, reason="fixer made no changes",
            findings_before=len(findings_before), findings_after=len(findings_before),
        )

    ok, err = _check_syntax(file_path, fixed_content)
    if not ok:
        logger.error("Verifier: %s failed syntax check after fix: %s", file_path, err)
        return VerificationResult(
            accepted=False, reason=f"syntax error after fix: {err}",
            findings_before=len(findings_before), findings_after=-1,
        )

    findings_after = run_single_file(repo_root, file_path, fixed_content)

    before_n, after_n = len(findings_before), len(findings_after)
    if after_n > before_n:
        logger.warning(
            "Verifier: rejecting fix for %s -- findings went from %d to %d",
            file_path, before_n, after_n,
        )
        return VerificationResult(
            accepted=False, reason=f"regression: findings increased from {before_n} to {after_n}",
            findings_before=before_n, findings_after=after_n,
        )

    logger.info("Verifier: accepted fix for %s (findings %d -> %d)", file_path, before_n, after_n)
    return VerificationResult(
        accepted=True, reason="ok", findings_before=before_n, findings_after=after_n,
    )
