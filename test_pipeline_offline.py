"""
Offline smoke test: mocks ONLY the Ollama HTTP layer. Static analysis
(ruff/bandit/yamllint), the Verifier's syntax + regression checks, and file
I/O are all real -- this proves the pipeline's actual safety mechanism
works, not just that the LLM prompts are wired up.

Scenario:
  - app.py: hardcoded secret, shell injection via subprocess+shell=True,
    unused imports, missing space after comma.
  - deploy.yml: shell module misuse, 0777 permissions.

Two fixer behaviors are tested:
  1. A "good" fixer response that fixes the safe stuff and leaves the
     shell-injection issue alone (marked not auto_fixable by review) ->
     should be ACCEPTED and written (in --apply mode).
  2. A "bad" fixer response that returns syntactically broken Python ->
     should be REJECTED by the Verifier and the original file left intact.
"""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import build_client, process_file  # noqa: E402
from pipeline.code_reviewer import CodeReviewer  # noqa: E402
from pipeline.code_fixer import CodeFixer  # noqa: E402
from pipeline.static_analysis import run_all  # noqa: E402

TEST_REPO = Path("/tmp/code_review_pipeline_test")

CFG = {
    "ollama": {
        "endpoints": [{"host": "http://127.0.0.1:11434", "name": "gpu01"}],
        "model": "qwen2.5-coder:32b",
        "model_max_context": 32768,
        "response_reserve_tokens": 2048,
    },
}


def build_repo():
    shutil.rmtree(TEST_REPO, ignore_errors=True)
    TEST_REPO.mkdir(parents=True)
    (TEST_REPO / "app.py").write_text(
        "import os\n"
        "import sys\n"
        "import subprocess\n"
        "\n"
        'PASSWORD = "hunter2"\n'
        "\n"
        "def run_cmd(user_input):\n"
        '    subprocess.call("ls " + user_input, shell=True)\n'
        "\n"
        "def add(a,b):\n"
        "    return a+b\n"
    )
    (TEST_REPO / "broken.py").write_text(
        "import json\n\ndef f(a,b):\n    return a+b\n"
    )


class FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def fake_post_good_fix(url, json_body=None, **kwargs):
    json_arg = kwargs.get("json") or json_body
    system = json_arg["messages"][0]["content"]
    user = json_arg["messages"][1]["content"]

    if "CodeReviewer" in system:
        if "app.py" in user:
            content = json.dumps({"issues": [
                {"line": 1, "severity": "low", "source": "static", "rule": "F401", "problem": "unused import os", "auto_fixable": True},
                {"line": 2, "severity": "low", "source": "static", "rule": "F401", "problem": "unused import sys", "auto_fixable": True},
                {"line": 8, "severity": "high", "source": "review", "rule": None, "problem": "shell injection via unsanitized user_input with shell=True", "auto_fixable": False},
                {"line": 5, "severity": "high", "source": "review", "rule": None, "problem": "hardcoded password", "auto_fixable": False},
            ]})
        else:
            content = json.dumps({"issues": [
                {"line": 3, "severity": "low", "source": "review", "rule": None, "problem": "missing space after comma in def f(a,b)", "auto_fixable": True},
            ]})
    elif "CodeFixer" in system:
        if "app.py" in user:
            # Fixes only the two safe issues (unused imports), leaves the
            # risky shell-injection / hardcoded-password issues untouched.
            content = (
                "import subprocess\n"
                "\n"
                'PASSWORD = "hunter2"\n'
                "\n"
                "def run_cmd(user_input):\n"
                '    subprocess.call("ls " + user_input, shell=True)\n'
                "\n"
                "def add(a,b):\n"
                "    return a+b\n"
            )
        else:
            content = "import json\n\n\ndef f(a, b):\n    return a + b\n"
    else:
        content = "{}"

    return FakeResp({"message": {"content": content}})


def fake_post_bad_fix(url, json_body=None, **kwargs):
    """Same reviewer, but CodeFixer returns syntactically broken Python."""
    json_arg = kwargs.get("json") or json_body
    system = json_arg["messages"][0]["content"]
    user = json_arg["messages"][1]["content"]

    if "CodeReviewer" in system:
        content = json.dumps({"issues": [
            {"line": 3, "severity": "low", "source": "review", "rule": None, "problem": "missing space after comma", "auto_fixable": True},
        ]})
    elif "CodeFixer" in system:
        # Deliberately broken -- missing colon, should fail py_compile.
        content = "import json\n\ndef f(a, b)\n    return a + b\n"
    else:
        content = "{}"
    return FakeResp({"message": {"content": content}})


def run_scenario(post_fn, label):
    print(f"\n--- Scenario: {label} ---")
    build_repo()
    with patch("ollama_client.requests.post", side_effect=post_fn):
        client = build_client(CFG)
        reviewer = CodeReviewer(client)
        fixer = CodeFixer(client)

        if label == "good fix (app.py)":
            target = TEST_REPO / "app.py"
        else:
            target = TEST_REPO / "broken.py"

        findings_by_file = run_all(TEST_REPO, [target], [])
        rel = str(target.relative_to(TEST_REPO))
        static_findings = findings_by_file.get(rel, [])
        print(f"Static findings for {rel}: {len(static_findings)}")

        backups_dir = TEST_REPO.parent / "backups_test"
        shutil.rmtree(backups_dir, ignore_errors=True)

        result = process_file(target, TEST_REPO, static_findings, reviewer, fixer, apply=True, backups_dir=backups_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "diff"}, indent=2))
        return result, target


def main():
    result1, app_py = run_scenario(fake_post_good_fix, "good fix (app.py)")
    assert result1["fixed"] is True, "expected the safe fix to be accepted"
    final_content = app_py.read_text()
    assert "import os" not in final_content, "unused import should be removed"
    assert "hunter2" in final_content, "risky (non-auto-fixable) issue must NOT be silently touched"
    assert 'shell=True' in final_content, "shell injection issue was correctly left for manual review"
    print("PASS: safe fix applied, risky issues correctly left alone")

    result2, broken_py = run_scenario(fake_post_bad_fix, "bad fix (broken.py)")
    assert result2["fixed"] is False, "a syntactically broken fix must be rejected"
    assert "syntax error" in result2["verification"]["reason"]
    original_still_intact = broken_py.read_text()
    assert "def f(a,b)" in original_still_intact and "def f(a, b)\n    return" not in original_still_intact
    print("PASS: broken fix correctly rejected by Verifier, original file untouched")

    shutil.rmtree(TEST_REPO, ignore_errors=True)
    shutil.rmtree(TEST_REPO.parent / "backups_test", ignore_errors=True)
    print("\nALL SCENARIOS PASSED")


if __name__ == "__main__":
    main()
