"""Unit tests for every module, offline. Only the HTTP layer and, where noted,
the linter subprocesses are faked; ruff/bandit/yamllint run for real when
installed (CI installs them), and are skipped otherwise."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent))

import ollama_client as oc  # noqa: E402
import orchestrator  # noqa: E402
from pipeline import static_analysis as sa  # noqa: E402
from pipeline.code_fixer import CodeFixer  # noqa: E402
from pipeline.code_reviewer import CodeReviewer  # noqa: E402
from pipeline.repo_scanner import RepoScanner  # noqa: E402
from pipeline.verifier import verify_fix  # noqa: E402

needs = lambda tool: pytest.mark.skipif(not shutil.which(tool), reason=f"{tool} not installed")  # noqa: E731


class Resp:
    def __init__(self, payload=None, status=200, text=""):
        self._payload, self.status_code, self.text = payload, status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def client(**over):
    kw = {"heartbeat_interval_seconds": 0, "backoff_base_seconds": 0, **over}
    return oc.OllamaClient(oc.OllamaClientConfig(**kw))


# --------------------------------------------------------------------------- ollama_client: sizing

def test_estimate_tokens():
    assert oc.estimate_tokens("") == 0
    assert oc.estimate_tokens("a") == 1
    assert oc.estimate_tokens("x" * 360) == 100


def test_num_ctx_is_rounded_bounded_and_at_least_1024():
    cfg = oc.OllamaClientConfig(model_max_context=32768, response_reserve_tokens=2048)
    assert oc.compute_num_ctx("hi", cfg) == 2560                    # 1 + 2048 reserve, rounded up to 512
    assert oc.compute_num_ctx("x" * 36000, cfg) % 512 == 0
    assert oc.compute_num_ctx("x" * 10_000_000, cfg) == 32768        # never above the model max
    small = oc.OllamaClientConfig(model_max_context=8192, response_reserve_tokens=100)
    assert oc.compute_num_ctx("", small) == 1024
    assert oc.compute_num_ctx("x" * 3600, cfg, response_reserve_tokens=500) == 2048   # 1050 + 500 -> 2048


def test_single_call_budget():
    cfg = oc.OllamaClientConfig(model_max_context=32768, response_reserve_tokens=2048)
    assert oc.single_call_input_budget(cfg) == int((32768 - 2048 - 400) * 0.85)
    tiny = oc.OllamaClientConfig(model_max_context=1000, response_reserve_tokens=900)
    assert oc.single_call_input_budget(tiny) == 512


def test_chunk_text_splits_on_lines_and_keeps_everything():
    text = "".join(f"line {i:04d} " + "x" * 30 + "\n" for i in range(200))
    chunks = oc.chunk_text(text, 100)
    assert len(chunks) > 1 and "".join(chunks) == text
    assert all(c.endswith("\n") for c in chunks)
    # Per-line estimates are rounded down before summing, so a whole chunk can
    # estimate a few % over max_tokens; the 0.85 budget factor absorbs that.
    assert all(oc.estimate_tokens(c) <= 105 for c in chunks)
    assert oc.chunk_text("short", 100) == ["short"]


def test_pack_into_batches_splits_oversized_entries_with_part_labels():
    big = "y" * 40 + "\n"
    entries = [("a.py", "x" * 36), ("huge.py", big * 30), ("b.py", "z" * 36)]
    batches = oc.pack_into_batches(entries, budget_tokens=60)
    labels = [label for b in batches for label, _ in b]
    assert labels[0] == "a.py" and labels[-1] == "b.py"
    parts = [label for label in labels if label.startswith("huge.py")]
    assert len(parts) > 1 and parts[0] == f"huge.py (part 1/{len(parts)})"
    for b in batches:
        assert sum(oc.estimate_tokens(c) for _, c in b) <= 60 or len(b) == 1


# --------------------------------------------------------------------------- ollama_client: requests

def test_ollama_payload_options_and_json_mode():
    c = client(num_gpu=0, num_thread=8)
    with patch("ollama_client.requests.post", return_value=Resp({"message": {"content": "hi"}})) as post:
        assert c.chat([{"role": "user", "content": "x"}], json_mode=True, num_ctx=4096) == "hi"
    url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["format"] == "json" and payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_gpu": 0, "num_thread": 8}


def test_openai_backends_ignore_gpu_options_and_parse_choices(caplog):
    c = client(backend="vllm", num_gpu=0)
    with patch("ollama_client.requests.post",
               return_value=Resp({"choices": [{"message": {"content": "ok"}}]})) as post:
        assert c.chat([{"role": "user", "content": "x"}]) == "ok"
    assert "num_gpu" not in json.dumps(post.call_args.kwargs["json"])
    assert "ignored with backend=vllm" in caplog.text
    assert oc.OllamaClient._extract_content("llamacpp", {"choices": []}) == ""


def test_retries_5xx_and_bad_json_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)
    c = client(max_retries=4, backoff_base_seconds=2.0)
    responses = [Resp(status=503, text="busy"), Resp(json.JSONDecodeError("x", "", 0)),
                 Resp({"message": {"content": "third time"}})]
    with patch("ollama_client.requests.post", side_effect=responses):
        assert c.chat([{"role": "user", "content": "x"}]) == "third time"
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    c = client(max_retries=3)
    with patch("ollama_client.requests.post", side_effect=requests.ConnectionError("refused")) as post:
        with pytest.raises(RuntimeError, match="All 3 attempts failed: refused"):
            c.chat([{"role": "user", "content": "x"}])
    assert post.call_count == 3


def test_4xx_is_not_masked():
    c = client(max_retries=1)
    with patch("ollama_client.requests.post", return_value=Resp(status=404)):
        with pytest.raises(RuntimeError, match="404"):
            c.chat([{"role": "user", "content": "x"}])


def test_heartbeat_path_returns_and_reraises():
    c = client()

    def slow(url, json, timeout):
        time.sleep(0.15)
        return Resp({"message": {"content": "late"}})

    with patch("ollama_client.requests.post", side_effect=slow):
        assert c.chat([{"role": "user", "content": "x"}], heartbeat_interval=0.05) == "late"
    with patch("ollama_client.requests.post", side_effect=requests.Timeout("t")):
        c.cfg.max_retries = 1
        with pytest.raises(RuntimeError, match="t"):
            c.chat([{"role": "user", "content": "x"}], heartbeat_interval=1)


@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', '```\n{"a": 1}\n```'])
def test_chat_json_strips_fences(raw):
    c = client()
    with patch.object(c, "chat", return_value=raw):
        assert c.chat_json([]) == {"a": 1}


# --------------------------------------------------------------------------- static_analysis

def test_jinja_detection():
    assert sa.is_jinja_template("x: '{{ var }}'")
    assert sa.is_jinja_template("{% if a %}")
    assert not sa.is_jinja_template("x: 1")


@pytest.mark.parametrize("text,expected", [
    ("", None),
    ('[{"a": 1}]', [{"a": 1}]),
    ('[main] INFO running\n{"results": []}\n', {"results": []}),
    ('noise {"x": [1, 2]} trailing } junk', {"x": [1, 2]}),
    ("no json here", None),
])
def test_parse_json_lenient(text, expected):
    assert sa._parse_json_lenient(text) == expected


def test_run_survives_missing_tools_and_timeouts():
    assert sa._run(["definitely-not-a-tool-xyz"])[0] == 127
    assert sa._run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1) == (1, "", "timeout")


def test_missing_linters_are_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(sa.shutil, "which", lambda name: None)
    p = tmp_path / "a.py"
    p.write_text("import os\n")
    assert sa.run_ruff([p], tmp_path) == [] and sa.run_bandit([p], tmp_path) == []
    assert sa.run_yamllint([p], tmp_path) == [] and sa.run_ansible_lint(tmp_path) == []


@needs("ruff")
def test_ruff_findings_are_relative_with_severity(tmp_path):
    p = tmp_path / "pkg" / "a.py"
    p.parent.mkdir()
    p.write_text("import os\n")
    findings = sa.run_ruff([p], tmp_path)
    assert [(f.file, f.line, f.rule, f.severity, f.tool) for f in findings] == [("pkg/a.py", 1, "F401", "medium", "ruff")]


@needs("bandit")
def test_bandit_findings(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("import subprocess\nsubprocess.call('ls ' + x, shell=True)\n")
    findings = sa.run_bandit([p], tmp_path)
    assert any(f.rule == "B602" and f.severity == "high" and f.file == "a.py" for f in findings)


@needs("yamllint")
def test_yamllint_skips_jinja_files(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("a: [1, 2\n")
    jinja = tmp_path / "tpl.yml"
    jinja.write_text("a: {{ broken\n")
    findings = sa.run_yamllint([bad, jinja], tmp_path)
    assert {f.file for f in findings} == {"bad.yml"}
    assert any(f.severity == "medium" and f.rule == "error" for f in findings)
    assert sa.run_yamllint([jinja], tmp_path) == []


def test_ansible_lint_output_is_mapped(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_tool_available", lambda name: True)
    out = json.dumps([
        {"check_name": "no-changed-when", "severity": "major", "description": "Commands should not change things",
         "location": {"path": "roles/x/tasks/main.yml", "lines": {"begin": 7}}},
        {"tag": "yaml", "severity": "VERY_HIGH", "message": "bad", "filename": "site.yml",
         "location": {"line": 3}},
    ])
    monkeypatch.setattr(sa, "_run", lambda *a, **k: (2, "WARNING noise\n" + out, ""))
    f1, f2 = sa.run_ansible_lint(tmp_path)
    assert (f1.file, f1.line, f1.rule, f1.severity) == ("roles/x/tasks/main.yml", 7, "no-changed-when", "medium")
    assert (f2.file, f2.line, f2.rule, f2.severity, f2.message) == ("site.yml", 3, "yaml", "high", "bad")
    monkeypatch.setattr(sa, "_run", lambda *a, **k: (0, "not json at all", ""))
    assert sa.run_ansible_lint(tmp_path) == []


def test_run_all_groups_by_file(monkeypatch, tmp_path):
    F = sa.Finding
    monkeypatch.setattr(sa, "run_ruff", lambda p, r: [F("a.py", 1, "low", "ruff", "E1", "x")])
    monkeypatch.setattr(sa, "run_bandit", lambda p, r: [F("a.py", 2, "high", "bandit", "B1", "y")])
    monkeypatch.setattr(sa, "run_yamllint", lambda p, r: [F("b.yml", 1, "low", "yamllint", "w", "z")])
    monkeypatch.setattr(sa, "run_ansible_lint", lambda r: [])
    grouped = sa.run_all(tmp_path, [], [])
    assert sorted(grouped) == ["a.py", "b.yml"] and len(grouped["a.py"]) == 2
    assert grouped["a.py"][0].to_dict()["tool"] == "ruff"


@needs("ruff")
def test_run_single_file_relabels_and_leaves_disk_alone(tmp_path):
    real = tmp_path / "src" / "mod.py"
    real.parent.mkdir()
    real.write_text("print('original')\n")
    findings = sa.run_single_file(tmp_path, real, "import os\n")
    assert findings and all(f.file == "src/mod.py" for f in findings)
    assert real.read_text() == "print('original')\n"
    assert sa.run_single_file(tmp_path, tmp_path / "t.yml", "a: {{ x }}\n") == []


# --------------------------------------------------------------------------- repo_scanner

def test_scan_filters(tmp_path):
    repo = tmp_path / "repo"
    for rel, content in {
        "app.py": "x = 1\n", "roles/r/tasks/main.yml": "- debug: msg=hi\n", "ci.yaml": "a: 1\n",
        "empty.py": "", "README.md": "# x\n", ".venv/lib/site.py": "x\n", "node_modules/m/a.yml": "a\n",
        "big.py": "x" * 2000, "sub/__pycache__/c.py": "x\n", "Pipfile.lock": "{}",
    }.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    s = RepoScanner(tmp_path / "work", max_file_bytes=1000).scan(repo)
    assert sorted(p.name for p in s.python_files) == ["app.py"]
    assert sorted(str(p.relative_to(repo)) for p in s.yaml_files) == ["ci.yaml", "roles/r/tasks/main.yml"]
    assert len(s.all_files) == 3


def test_use_local_dir_and_safe_cleanup(tmp_path):
    scanner = RepoScanner(tmp_path / "work")
    with pytest.raises(NotADirectoryError):
        scanner.use_local_dir(tmp_path / "missing")
    mine = tmp_path / "my-repo"
    mine.mkdir()
    assert scanner.use_local_dir(mine) == mine.resolve()
    scanner.cleanup(mine)                          # outside work_dir: refused
    assert mine.exists()
    inside = tmp_path / "work" / "clone"
    inside.mkdir()
    scanner.cleanup(inside)
    assert not inside.exists()


@needs("git")
def test_clone_then_update(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git = lambda *a, cwd=origin: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)  # noqa: E731
    git("init", "-q", "-b", "main")
    (origin / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    scanner = RepoScanner(tmp_path / "work")
    target = scanner.clone_or_update(origin.as_uri(), "r", branch="main")
    assert (target / "a.py").read_text() == "x = 1\n"
    (origin / "a.py").write_text("x = 2\n")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    (target / "a.py").write_text("local edit\n")         # reset --hard wins
    scanner.clone_or_update(origin.as_uri(), "r", branch="main")
    assert (target / "a.py").read_text() == "x = 2\n"


# --------------------------------------------------------------------------- agents

class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, messages, temperature=0.2):
        self.calls.append(messages)
        return self.reply

    def chat_json(self, messages, temperature=0.0):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


def test_fixer_returns_content_untouched_without_issues():
    c = FakeClient("should not be called")
    assert CodeFixer(c).fix(Path("a.py"), "x = 1\n", []) == "x = 1\n" and c.calls == []


@pytest.mark.parametrize("reply,expected", [
    ("x = 2", "x = 2\n"),
    ("```python\nx = 2\n```", "x = 2\n"),
    ("```\nx = 2\n", "x = 2\n"),            # unterminated fence
    ("x = 2\n", "x = 2\n"),
])
def test_fixer_strips_one_outer_fence(reply, expected):
    assert CodeFixer(FakeClient(reply)).fix(Path("a.py"), "x=1\n", [{"line": 1, "problem": "spacing"}]) == expected


def test_fixer_prompt_names_language_and_issues():
    c = FakeClient("a: 1\n")
    CodeFixer(c).fix(Path("site.yml"), "a: 1\n", [{"severity": "high", "line": 3, "problem": "0777 mode"}])
    system, user = c.calls[0][0]["content"], c.calls[0][1]["content"]
    assert "YAML/Ansible" in system and "- [high] line 3: 0777 mode" in user


def test_reviewer_passes_static_findings_and_returns_issues():
    c = FakeClient({"issues": [{"line": 1, "auto_fixable": True, "problem": "p"}]})
    f = sa.Finding("a.py", 1, "medium", "ruff", "F401", "unused import")
    assert CodeReviewer(c).review(Path("a.py"), "import os\n", [f]) == [{"line": 1, "auto_fixable": True, "problem": "p"}]
    assert "[ruff:F401] line 1: unused import (severity: medium)" in c.calls[0][1]["content"]


@pytest.mark.parametrize("client_", [FakeClient(error=RuntimeError("down")), FakeClient({"issues": "nope"})])
def test_reviewer_falls_back_to_static_findings_never_auto_fixing(client_):
    f = sa.Finding("a.py", 4, "high", "bandit", "B602", "shell=True")
    issues = CodeReviewer(client_).review(Path("a.py"), "x\n", [f])
    assert issues == [{"line": 4, "severity": "high", "source": "static", "rule": "B602",
                       "problem": "shell=True", "auto_fixable": False}]


# --------------------------------------------------------------------------- verifier

def test_verifier_rejects_no_change_and_syntax_errors(tmp_path):
    r = verify_fix(tmp_path, tmp_path / "a.py", "x = 1\n", "x = 1\n  ", [])
    assert not r.accepted and r.reason == "fixer made no changes"
    r = verify_fix(tmp_path, tmp_path / "a.py", "x = 1\n", "def f(:\n", [])
    assert not r.accepted and r.reason.startswith("syntax error") and r.findings_after == -1
    r = verify_fix(tmp_path, tmp_path / "a.yml", "a: 1\n", "a: [1\n", [])
    assert not r.accepted and "syntax error" in r.reason


def test_verifier_rejects_regressions_and_accepts_improvements(monkeypatch, tmp_path):
    F = sa.Finding
    before = [F("a.py", 1, "low", "ruff", "E1", "x")]
    monkeypatch.setattr("pipeline.verifier.run_single_file", lambda *a: before * 2)
    r = verify_fix(tmp_path, tmp_path / "a.py", "x = 1\n", "x = 2\n", before)
    assert not r.accepted and r.reason == "regression: findings increased from 1 to 2"
    monkeypatch.setattr("pipeline.verifier.run_single_file", lambda *a: [])
    r = verify_fix(tmp_path, tmp_path / "a.py", "x = 1\n", "x = 2\n", before)
    assert r.accepted and (r.findings_before, r.findings_after) == (1, 0)


def test_verifier_accepts_multi_document_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.verifier.run_single_file", lambda *a: [])
    assert verify_fix(tmp_path, tmp_path / "k.yaml", "a: 1\n", "a: 1\n---\nb: 2\n", []).accepted


# --------------------------------------------------------------------------- orchestrator

def test_build_client_validates_backend():
    cfg = {"ollama": {"endpoints": [{"host": "http://h:1"}], "model": "m", "backend": "openai"}}
    with pytest.raises(ValueError, match="backend"):
        orchestrator.build_client(cfg)
    cfg["ollama"]["backend"] = "llamacpp"
    c = orchestrator.build_client(cfg)
    assert c.cfg.backend == "llamacpp" and c.cfg.endpoints[0].name == "http://h:1"


def fake_llm(url, json=None, timeout=None):
    system, user = json["messages"][0]["content"], json["messages"][1]["content"]
    if "CodeReviewer" in system:
        body = {"issues": [{"line": 1, "severity": "low", "problem": "unused import", "auto_fixable": True},
                           {"line": 2, "severity": "high", "problem": "hardcoded secret", "auto_fixable": False}]}
        return Resp({"message": {"content": __import__("json").dumps(body)}})
    if "play.yml" in user:
        return Resp({"message": {"content": "- hosts: all\n"}})      # no change -> rejected
    return Resp({"message": {"content": 'TOKEN = "abc"\n'}})


def run_main(monkeypatch, tmp_path, *argv):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ollama:\n  endpoints:\n    - host: http://127.0.0.1:1\n  model: m\n"
                   "  heartbeat_interval_seconds: 0\n  max_retries: 1\n")
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--config", str(cfg),
                                      "--output-dir", str(tmp_path / "out"), *argv])
    monkeypatch.setattr(orchestrator, "run_all", lambda *a: {})
    monkeypatch.setattr("pipeline.verifier.run_single_file", lambda *a: [])
    with patch("ollama_client.requests.post", side_effect=fake_llm):
        orchestrator.main()
    return json.loads((tmp_path / "out" / "review_report.json").read_text())


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "app.py").write_text('import os\nTOKEN = "abc"\n')
    (r / "play.yml").write_text("- hosts: all\n")
    return r


def test_main_dry_run_reports_but_does_not_write(monkeypatch, tmp_path, repo):
    report = run_main(monkeypatch, tmp_path, "--local-dir", str(repo), "--workers", "2")
    assert report["mode"] == "dry-run" and report["files_reviewed"] == 2
    assert [r["file"] for r in report["results"]] == ["app.py", "play.yml"]     # scan order kept
    assert report["files_fixed"] == 1 and report["files_with_manual_issues"] == 2
    assert (repo / "app.py").read_text() == 'import os\nTOKEN = "abc"\n'
    manual = json.loads((tmp_path / "out" / "manual_review_needed.json").read_text())
    assert {m["issue"]["problem"] for m in manual} == {"hardcoded secret"}


def test_main_apply_writes_fix_and_backup(monkeypatch, tmp_path, repo):
    report = run_main(monkeypatch, tmp_path, "--local-dir", str(repo), "--apply", "--only-python")
    assert report["mode"] == "apply" and report["files_reviewed"] == 1
    assert (repo / "app.py").read_text() == 'TOKEN = "abc"\n'
    assert (tmp_path / "out" / "backups" / "app.py").read_text() == 'import os\nTOKEN = "abc"\n'
    assert report["results"][0]["diff"].startswith("--- a/app.py\n+++ b/app.py")


def test_main_exits_when_nothing_to_review(monkeypatch, tmp_path, repo):
    (repo / "play.yml").unlink()
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, tmp_path, "--local-dir", str(repo), "--only-ansible")
    assert exc.value.code == 1


def test_main_keeps_going_when_one_file_crashes(monkeypatch, tmp_path, repo):
    real = orchestrator.process_file

    def flaky(file_path, *a, **k):
        if file_path.name == "play.yml":
            raise RuntimeError("boom")
        return real(file_path, *a, **k)

    monkeypatch.setattr(orchestrator, "process_file", flaky)
    report = run_main(monkeypatch, tmp_path, "--local-dir", str(repo))
    assert report["results"][1] == {"file": "play.yml", "error": "unhandled exception during processing"}


def test_include_risky_sends_manual_issues_to_the_fixer(monkeypatch, tmp_path, repo):
    seen = []

    class Fixer:
        def fix(self, path, content, issues):
            seen.append([i["problem"] for i in issues])
            return content.replace("import os\n", "")

    class Reviewer:
        def review(self, *a):
            return [{"problem": "unused import", "auto_fixable": True},
                    {"problem": "hardcoded secret", "auto_fixable": False}]

    monkeypatch.setattr("pipeline.verifier.run_single_file", lambda *a: [])
    orchestrator.process_file(repo / "app.py", repo, [], Reviewer(), Fixer(), False, tmp_path / "b")
    orchestrator.process_file(repo / "app.py", repo, [], Reviewer(), Fixer(), False, tmp_path / "b", include_risky=True)
    assert seen == [["unused import"], ["unused import", "hardcoded secret"]]


def test_offline_scenarios_script():
    """Runs the existing end-to-end scenario script (real linters, real verifier)."""
    if not shutil.which("ruff"):
        pytest.skip("ruff not installed")
    import test_pipeline_offline
    test_pipeline_offline.main()
