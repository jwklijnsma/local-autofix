"""
repo_scanner.py -- RepoScanner agent (code-review edition)

Gets the source tree ready for review, either by cloning/updating a git
repository or by pointing straight at an existing local directory, then
collects Python and Ansible/YAML files worth reviewing.

Unlike the documentation pipeline's scanner, this one deliberately KEEPS
test files -- you want your tests reviewed too -- and skips the doc-style
priority-file logic since every matching file gets reviewed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("repo_scanner")

DEFAULT_INCLUDE_EXTS = {".py", ".yml", ".yaml"}

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", ".eggs",
}

DEFAULT_IGNORE_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
}


@dataclass
class ScanResult:
    repo_path: Path
    python_files: list[Path] = field(default_factory=list)
    yaml_files: list[Path] = field(default_factory=list)

    @property
    def all_files(self) -> list[Path]:
        return self.python_files + self.yaml_files


class RepoScanner:
    def __init__(
        self,
        work_dir: Path,
        include_exts: set[str] = None,
        ignore_dirs: set[str] = None,
        max_file_bytes: int = 500_000,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.include_exts = include_exts or DEFAULT_INCLUDE_EXTS
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.max_file_bytes = max_file_bytes

    def clone_or_update(self, repo_url: str, repo_name: str, branch: str = "master") -> Path:
        target = self.work_dir / repo_name
        if target.exists():
            logger.info("Repo %s already cloned, pulling latest %s", repo_name, branch)
            subprocess.run(
                ["git", "-C", str(target), "fetch", "origin", branch],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "reset", "--hard", f"origin/{branch}"],
                check=True, capture_output=True,
            )
        else:
            logger.info("Cloning %s (branch=%s) into %s", repo_url, branch, target)
            subprocess.run(
                ["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(target)],
                check=True, capture_output=True,
            )
        return target

    def use_local_dir(self, local_dir: str | Path) -> Path:
        """
        Point the scanner at an existing local directory instead of cloning
        anything. Read-only -- never copied, never written to.
        """
        target = Path(local_dir).expanduser().resolve()
        if not target.is_dir():
            raise NotADirectoryError(f"local_dir does not exist or is not a directory: {target}")
        logger.info("Using local directory as source: %s", target)
        return target

    def scan(self, repo_path: Path) -> ScanResult:
        repo_path = Path(repo_path)
        python_files: list[Path] = []
        yaml_files: list[Path] = []

        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.ignore_dirs for part in path.relative_to(repo_path).parts):
                continue
            if path.name in DEFAULT_IGNORE_FILENAMES:
                continue
            if path.suffix.lower() not in self.include_exts:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    logger.warning("Skipping oversized file: %s", path)
                    continue
                if path.stat().st_size == 0:
                    continue
            except OSError:
                continue

            if path.suffix.lower() == ".py":
                python_files.append(path)
            else:
                yaml_files.append(path)

        logger.info(
            "Scanned %s: %d Python file(s), %d YAML file(s)",
            repo_path, len(python_files), len(yaml_files),
        )
        return ScanResult(repo_path=repo_path, python_files=python_files, yaml_files=yaml_files)

    def cleanup(self, repo_path: Path):
        """Removes a cloned repo from work_dir. Never touches anything outside it."""
        repo_path = Path(repo_path).resolve()
        if self.work_dir.resolve() not in repo_path.parents:
            logger.warning("Refusing to clean up %s -- it is outside %s (likely a local_dir source)", repo_path, self.work_dir)
            return
        shutil.rmtree(repo_path, ignore_errors=True)
