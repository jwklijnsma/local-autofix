#!/usr/bin/env python3
"""
orchestrator.py -- code review/fix pipeline entrypoint

Flow per file:
    RepoScanner -> static_analysis (ruff/bandit/yamllint/ansible-lint)
                -> CodeReviewer (LLM triage + semantic issues)
                -> CodeFixer (LLM fix, auto-fixable issues only)
                -> Verifier (syntax check + regression check, no LLM)
                -> write back (only if --apply; dry-run by default)

Safety by default:
    - Dry-run unless --apply is passed. Dry-run writes a report + diffs to
      review_output/ but never touches your actual files.
    - Every file that IS modified gets a .orig backup next to it inside
      backups/, plus a unified diff in the report, so anything can be
      reverted by hand even if you don't use git.
    - Only issues CodeReviewer marks auto_fixable are ever sent to
      CodeFixer. Everything else lands in the "needs manual review"
      section of the report instead of being touched.

Usage:
    python orchestrator.py --local-dir /path/to/repo               # dry-run report only
    python orchestrator.py --local-dir /path/to/repo --apply        # actually fix files
    python orchestrator.py --repo-url https://.../repo.git --branch main --apply
    python orchestrator.py --local-dir . --only-python               # skip YAML/Ansible
    python orchestrator.py --local-dir . --only-ansible               # skip Python
    python orchestrator.py --local-dir . --workers 4 --apply         # review files concurrently
                                                                       # across multiple Ollama
                                                                       # endpoints (see config.yaml)
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ollama_client import OllamaClient, OllamaClientConfig, OllamaEndpoint
from pipeline.code_fixer import CodeFixer
from pipeline.code_reviewer import CodeReviewer
from pipeline.repo_scanner import RepoScanner
from pipeline.static_analysis import run_all
from pipeline.verifier import verify_fix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client(cfg: dict) -> OllamaClient:
    oc = cfg["ollama"]
    endpoints = [OllamaEndpoint(host=e["host"], name=e.get("name", "")) for e in oc["endpoints"]]
    backend = oc.get("backend", "ollama")
    if backend not in ("ollama", "vllm", "llamacpp"):
        raise ValueError(f"ollama.backend must be 'ollama', 'vllm', or 'llamacpp', got: {backend!r}")
    client_cfg = OllamaClientConfig(
        endpoints=endpoints,
        model=oc["model"],
        model_max_context=oc.get("model_max_context", 32768),
        response_reserve_tokens=oc.get("response_reserve_tokens", 2048),
        request_timeout=oc.get("request_timeout", 600),
        max_retries=oc.get("max_retries", 4),
        backoff_base_seconds=oc.get("backoff_base_seconds", 2.0),
        heartbeat_interval_seconds=oc.get("heartbeat_interval_seconds", 30),
        circuit_breaker_failure_threshold=oc.get("circuit_breaker_failure_threshold", 2),
        circuit_breaker_cooldown_seconds=oc.get("circuit_breaker_cooldown_seconds", 120),
        num_gpu=oc.get("num_gpu"),
        num_thread=oc.get("num_thread"),
        backend=backend,
    )
    return OllamaClient(client_cfg)


def process_file(
    file_path: Path,
    repo_root: Path,
    static_findings: list,
    reviewer: CodeReviewer,
    fixer: CodeFixer,
    apply: bool,
    backups_dir: Path,
    include_risky: bool = False,
) -> dict:
    rel = str(file_path.relative_to(repo_root))
    original_content = file_path.read_text(errors="replace")

    issues = reviewer.review(file_path, original_content, static_findings)
    auto_fixable = [i for i in issues if i.get("auto_fixable")]
    manual_only = [i for i in issues if not i.get("auto_fixable")]

    # include_risky bypasses CodeReviewer's auto_fixable gate entirely --
    # every issue found gets sent to CodeFixer, including ones explicitly
    # flagged as unsafe to auto-fix (hardcoded secrets, shell injection,
    # logic bugs). The Verifier's syntax + regression check still applies
    # and can still reject a bad fix -- this only removes the "don't even
    # attempt it" gate, not the "don't write a broken result" gate.
    issues_to_fix = issues if include_risky else auto_fixable

    result = {
        "file": rel,
        "static_finding_count": len(static_findings),
        "issues": issues,
        "fixed": False,
        "verification": None,
        "diff": None,
    }

    if not issues_to_fix:
        logger.info("%s: no auto-fixable issues (%d flagged for manual review)", rel, len(manual_only))
        return result

    if include_risky and manual_only:
        logger.warning(
            "%s: --include-risky set -- attempting %d issue(s) normally left for manual review "
            "(e.g. hardcoded secrets, injection risk, logic bugs). Verifier still applies.",
            rel, len(manual_only),
        )

    fixed_content = fixer.fix(file_path, original_content, issues_to_fix)
    verdict = verify_fix(repo_root, file_path, original_content, fixed_content, static_findings)
    result["verification"] = {
        "accepted": verdict.accepted, "reason": verdict.reason,
        "findings_before": verdict.findings_before, "findings_after": verdict.findings_after,
    }

    if not verdict.accepted:
        logger.warning("%s: fix rejected (%s) -- left untouched", rel, verdict.reason)
        return result

    diff = "\n".join(difflib.unified_diff(
        original_content.splitlines(), fixed_content.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
    ))
    result["diff"] = diff
    result["fixed"] = True

    if apply:
        backup_path = backups_dir / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(original_content, encoding="utf-8")
        file_path.write_text(fixed_content, encoding="utf-8")
        logger.info("%s: fix applied (backup at %s)", rel, backup_path)
    else:
        logger.info("%s: fix verified but NOT applied (dry-run -- pass --apply to write it)", rel)

    return result


def main():
    parser = argparse.ArgumentParser(description="Automated Python/Ansible code review and fix pipeline (Ollama-powered).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-dir", help="Path to a local repo/directory to review")
    src.add_argument("--repo-url", help="Git URL to clone and review")

    parser.add_argument("--branch", default="master", help="Branch to clone (with --repo-url)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--work-dir", default="./_clones", help="Scratch dir for git clones")
    parser.add_argument("--output-dir", default="./review_output", help="Where reports/backups are written")
    parser.add_argument("--apply", action="store_true", help="Actually write verified fixes to disk (default: dry-run report only)")
    parser.add_argument(
        "--include-risky", action="store_true",
        help="Attempt fixes for ALL issues CodeReviewer finds, including ones marked "
             "NOT auto_fixable (hardcoded secrets, injection risk, logic bugs, etc). "
             "The Verifier's syntax/regression check still applies and can still reject "
             "a bad result -- but the 'don't even attempt this' safety gate is removed. "
             "Review the diffs in review_report.json carefully before trusting this.",
    )
    parser.add_argument("--only-python", action="store_true", help="Skip YAML/Ansible files")
    parser.add_argument("--only-ansible", action="store_true", help="Skip Python files")
    parser.add_argument("--keep-clone", action="store_true", help="Don't delete a git clone after processing")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of files to review/fix concurrently. Set this to roughly the number "
             "of Ollama endpoints in config.yaml (or a bit more) to actually use multiple "
             "Ollama servers in parallel -- with 1 worker, extra endpoints only get used "
             "sequentially across different files, never at the same time.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    client = build_client(cfg)

    scanner = RepoScanner(work_dir=Path(args.work_dir))
    if args.local_dir:
        repo_path = scanner.use_local_dir(args.local_dir)
        is_local = True
    else:
        repo_path = scanner.clone_or_update(args.repo_url, "review-target", branch=args.branch)
        is_local = False

    scan = scanner.scan(repo_path)
    python_files = [] if args.only_ansible else scan.python_files
    yaml_files = [] if args.only_python else scan.yaml_files

    if not python_files and not yaml_files:
        logger.error("No files to review after filtering. Nothing to do.")
        sys.exit(1)

    logger.info(
        "Reviewing %d Python file(s), %d YAML file(s) with %d worker(s)%s",
        len(python_files), len(yaml_files), args.workers,
        " (DRY RUN -- no files will be modified)" if not args.apply else "",
    )
    if args.include_risky:
        logger.warning(
            "--include-risky is set: fixes will be attempted for issues normally left "
            "for manual review (hardcoded secrets, injection risk, logic bugs, etc). "
            "The Verifier's syntax/regression check still applies, but this bypasses "
            "CodeReviewer's judgment about whether a fix is safe to attempt at all.%s",
            " Review backups/ and the diffs in review_report.json carefully after this run."
            if args.apply else "",
        )
    if args.workers > 1 and len(cfg["ollama"]["endpoints"]) == 1:
        logger.warning(
            "--workers %d requested but config.yaml only has 1 Ollama endpoint -- "
            "you'll get concurrent requests hitting the same server, not true "
            "multi-server parallelism. Add more entries under ollama.endpoints to "
            "actually spread load across multiple Ollama instances.",
            args.workers,
        )

    findings_by_file = run_all(repo_path, python_files, yaml_files)

    reviewer = CodeReviewer(client)
    fixer = CodeFixer(client)

    output_dir = Path(args.output_dir)
    backups_dir = output_dir / "backups"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = python_files + yaml_files
    results_by_rel: dict[str, dict] = {}

    def _process(file_path: Path) -> dict:
        rel = str(file_path.relative_to(repo_path))
        static_findings = findings_by_file.get(rel, [])
        try:
            return process_file(file_path, repo_path, static_findings, reviewer, fixer, args.apply, backups_dir, args.include_risky)
        except Exception:
            logger.exception("Unhandled error processing %s", rel)
            return {"file": rel, "error": "unhandled exception during processing"}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_process, fp): fp for fp in all_files}
        for future in as_completed(futures):
            file_path = futures[future]
            rel = str(file_path.relative_to(repo_path))
            results_by_rel[rel] = future.result()

    # Preserve the original scan order in the report regardless of which
    # thread finished first.
    results = [
        results_by_rel[str(fp.relative_to(repo_path))]
        for fp in all_files
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo_path) if is_local else args.repo_url,
        "mode": "apply" if args.apply else "dry-run",
        "files_reviewed": len(results),
        "files_fixed": sum(1 for r in results if r.get("fixed")),
        "files_with_manual_issues": sum(
            1 for r in results if any(not i.get("auto_fixable") for i in r.get("issues", []))
        ),
        "results": results,
    }
    report_path = output_dir / "review_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    manual_issues = [
        {"file": r["file"], "issue": i}
        for r in results for i in r.get("issues", [])
        if not i.get("auto_fixable")
    ]
    manual_path = output_dir / "manual_review_needed.json"
    manual_path.write_text(json.dumps(manual_issues, indent=2, ensure_ascii=False))

    logger.info(
        "Done. %d file(s) reviewed, %d fixed%s, %d issue(s) need manual review. Report: %s",
        report["files_reviewed"], report["files_fixed"],
        "" if args.apply else " (dry-run, not written)",
        len(manual_issues), report_path,
    )

    if not is_local and not args.keep_clone:
        scanner.cleanup(repo_path)


if __name__ == "__main__":
    main()
