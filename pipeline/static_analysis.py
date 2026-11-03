"""
static_analysis.py

Wraps real linters/static analyzers so the LLM agents aren't inventing
issues from scratch -- they triage and act on tool output, the same way
FactExtractor grounds the doc pipeline in real file content instead of
hallucination.

Tools used (each optional -- missing tools are skipped with a warning,
never a crash):
  - ruff        (Python: style, bugs, unused imports, etc.)
  - bandit      (Python: security issues)
  - yamllint    (YAML: syntax/style, works standalone per-file)
  - ansible-lint (Ansible: best practices, needs repo-level context so it
                  runs once against the whole repo, not per-file)

Install what you want to use:
    pip install ruff bandit yamllint ansible-lint
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("static_analysis")


@dataclass
class Finding:
    file: str          # path relative to repo root, as a string
    line: int | None
    severity: str       # "low" | "medium" | "high"
    tool: str
    rule: str | None
    message: str

    def to_dict(self) -> dict:
        return {
            "file": self.file, "line": self.line, "severity": self.severity,
            "tool": self.tool, "rule": self.rule, "message": self.message,
        }


def is_jinja_template(content: str) -> bool:
    """
    Ansible YAML often contains Jinja2 syntax ({{ var }}, {% if %}) which
    is not valid standalone YAML -- yamllint isn't Jinja-aware and will
    report bogus "syntax error" findings on such files regardless of
    whether they're actually fine. ansible-lint (which runs repo-wide,
    separately) does understand Jinja, so it remains the real check for
    these files; yamllint is just skipped for them.
    """
    return "{{" in content or "{%" in content


def _tool_available(name: str) -> bool:
    available = shutil.which(name) is not None
    if not available:
        logger.warning("'%s' not found on PATH -- skipping that check (pip install %s)", name, name)
    return available


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        logger.error("Command timed out: %s", " ".join(cmd))
        return 1, "", "timeout"
    except FileNotFoundError:
        return 127, "", "not found"


def _parse_json_lenient(text: str):
    """
    Parses tool output as JSON, tolerating stray log lines that some tools
    (bandit especially) print to stdout before/after the actual JSON
    depending on version/config -- including log prefixes like "[main]"
    that themselves contain bracket characters, which a naive "find the
    first bracket" approach would misidentify as the start of the JSON.

    Tries a straight parse first, then tries every '{' and '[' position in
    the text as a possible start, preferring the earliest one that
    successfully parses to the end of a balanced structure.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = sorted(i for i, ch in enumerate(text) if ch in "{[")
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        end = text.rfind(closer)
        while end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try the next-earliest matching closer in case there's
                # trailing noise after the real JSON too.
                end = text.rfind(closer, start, end)
        # This start position never parsed; move on to the next candidate.
    return None


# --------------------------------------------------------------------------- #
# Python: ruff
# --------------------------------------------------------------------------- #

_RUFF_SEVERITY = {"E": "medium", "F": "medium", "W": "low", "S": "high", "B": "medium"}


def run_ruff(paths: list[Path], repo_root: Path) -> list[Finding]:
    if not paths or not _tool_available("ruff"):
        return []
    cmd = ["ruff", "check", "--output-format=json", *[str(p) for p in paths]]
    code, out, err = _run(cmd, cwd=repo_root)
    if not out.strip():
        if code not in (0, 1):
            logger.warning("ruff exited %d with no output: %s", code, err[:300])
        return []
    results = _parse_json_lenient(out)
    if results is None:
        logger.warning("Could not parse ruff JSON output (stdout started with: %s...) stderr: %s", out.strip()[:150], err[:150])
        return []

    findings = []
    for item in results:
        rule = item.get("code", "")
        severity = _RUFF_SEVERITY.get(rule[:1], "low") if rule else "low"
        findings.append(Finding(
            file=str(Path(item.get("filename", "")).resolve().relative_to(repo_root.resolve())),
            line=item.get("location", {}).get("row"),
            severity=severity,
            tool="ruff",
            rule=rule or None,
            message=item.get("message", ""),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Python: bandit
# --------------------------------------------------------------------------- #

_BANDIT_SEVERITY = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}


def run_bandit(paths: list[Path], repo_root: Path) -> list[Finding]:
    if not paths or not _tool_available("bandit"):
        return []
    # -q suppresses bandit's "[main] INFO ..." logging noise, which on some
    # bandit versions/environments leaks onto stdout and breaks JSON parsing.
    cmd = ["bandit", "-q", "-f", "json", *[str(p) for p in paths]]
    code, out, err = _run(cmd, cwd=repo_root)
    if not out.strip():
        return []

    data = _parse_json_lenient(out)
    if data is None:
        logger.warning(
            "Could not parse bandit JSON output (stdout started with: %s...) stderr: %s",
            out.strip()[:150], err.strip()[:150],
        )
        return []

    findings = []
    for item in data.get("results", []):
        try:
            rel = str(Path(item["filename"]).resolve().relative_to(repo_root.resolve()))
        except (KeyError, ValueError):
            rel = item.get("filename", "?")
        findings.append(Finding(
            file=rel,
            line=item.get("line_number"),
            severity=_BANDIT_SEVERITY.get(item.get("issue_severity", "LOW"), "low"),
            tool="bandit",
            rule=item.get("test_id"),
            message=item.get("issue_text", ""),
        ))
    return findings


# --------------------------------------------------------------------------- #
# YAML: yamllint
# --------------------------------------------------------------------------- #

_YAMLLINT_LINE_RE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*\[(?P<level>\w+)\]\s*(?P<msg>.+)$")


def run_yamllint(paths: list[Path], repo_root: Path) -> list[Finding]:
    if not paths or not _tool_available("yamllint"):
        return []

    plain_yaml_paths = []
    for p in paths:
        try:
            if is_jinja_template(p.read_text(errors="replace")):
                continue
        except OSError:
            continue
        plain_yaml_paths.append(p)
    if not plain_yaml_paths:
        return []

    cmd = ["yamllint", "-f", "parsable", *[str(p) for p in plain_yaml_paths]]
    code, out, err = _run(cmd, cwd=repo_root)
    if not out.strip():
        return []

    findings = []
    for line in out.splitlines():
        m = _YAMLLINT_LINE_RE.match(line.strip())
        if not m:
            continue
        level = m.group("level").lower()
        severity = "medium" if level == "error" else "low"
        try:
            rel = str(Path(m.group("file")).resolve().relative_to(repo_root.resolve()))
        except ValueError:
            rel = m.group("file")
        findings.append(Finding(
            file=rel, line=int(m.group("line")), severity=severity,
            tool="yamllint", rule=level, message=m.group("msg"),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Ansible: ansible-lint (repo-level -- needs role/playbook context)
# --------------------------------------------------------------------------- #

_ANSIBLE_LINT_SEVERITY = {"VERY_LOW": "low", "LOW": "low", "MEDIUM": "medium", "HIGH": "high", "VERY_HIGH": "high"}


def run_ansible_lint(repo_root: Path) -> list[Finding]:
    if not _tool_available("ansible-lint"):
        return []
    cmd = ["ansible-lint", "-f", "json"]
    code, out, err = _run(cmd, cwd=repo_root, timeout=300)
    if not out.strip():
        if code == 127:
            logger.warning("ansible-lint failed to run: %s", err[:300])
        return []
    data = _parse_json_lenient(out)
    if data is None:
        logger.warning("Could not parse ansible-lint JSON output (stdout started with: %s...) stderr: %s", out.strip()[:150], err[:150])
        return []

    findings = []
    for item in data:
        loc = item.get("location", {})
        path = loc.get("path") or item.get("filename", "?")
        line = loc.get("lines", {}).get("begin") if isinstance(loc.get("lines"), dict) else loc.get("line")
        severity_raw = (item.get("severity") or "MEDIUM").upper()
        findings.append(Finding(
            file=path,
            line=line,
            severity=_ANSIBLE_LINT_SEVERITY.get(severity_raw, "medium"),
            tool="ansible-lint",
            rule=item.get("check_name") or item.get("tag"),
            message=item.get("description") or item.get("message", ""),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Orchestration helper
# --------------------------------------------------------------------------- #

def run_all(repo_root: Path, python_files: list[Path], yaml_files: list[Path]) -> dict[str, list[Finding]]:
    """
    Runs every available tool once against the relevant file sets and
    returns findings grouped by file (relative path string).
    """
    all_findings: list[Finding] = []
    all_findings += run_ruff(python_files, repo_root)
    all_findings += run_bandit(python_files, repo_root)
    all_findings += run_yamllint(yaml_files, repo_root)
    all_findings += run_ansible_lint(repo_root)  # repo-wide; we filter per-file below

    by_file: dict[str, list[Finding]] = {}
    for f in all_findings:
        by_file.setdefault(f.file, []).append(f)
    return by_file


def run_single_file(repo_root: Path, file_path: Path, content: str) -> list[Finding]:
    """
    Re-runs the relevant tool(s) against a single piece of content (used by
    the Verifier to check a proposed fix before it's written back). Writes
    content to a temp file so we don't touch the real file on disk.
    """
    suffix = file_path.suffix.lower()
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".py":
            findings = run_ruff([tmp_path], tmp_path.parent) + run_bandit([tmp_path], tmp_path.parent)
        elif is_jinja_template(content):
            # yamllint isn't Jinja-aware -- see is_jinja_template(). Nothing
            # meaningful to check here at the single-file level; the
            # Verifier's Jinja2 syntax check handles correctness instead.
            findings = []
        else:
            findings = run_yamllint([tmp_path], tmp_path.parent)
        # Re-label with the real relative path for readability in reports.
        try:
            rel = str(file_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            rel = str(file_path)
        for f in findings:
            f.file = rel
        return findings
    finally:
        tmp_path.unlink(missing_ok=True)
