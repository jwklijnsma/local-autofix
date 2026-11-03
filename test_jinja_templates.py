"""
Regression test for Jinja2-templated YAML handling (Ansible templates/
files with {{ var }} / {% if %} syntax).

Bug this covers: plain YAML parsing (yaml.safe_load) fails on {{ }} syntax
with "found unhashable key", because YAML interprets {{ }} as a flow
mapping. Before this fix, ANY proposed fix to a Jinja2-templated YAML file
was unconditionally rejected by the Verifier's syntax check, regardless of
whether the fix was actually correct -- because the check itself was using
the wrong grammar for that file type.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml  # noqa: E402

from pipeline.static_analysis import is_jinja_template  # noqa: E402
from pipeline.verifier import _check_syntax, verify_fix  # noqa: E402

JINJA_YAML = """apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: {{ oc_packagemanifests_operator_name }}
  namespace: {{ oc_packagemanifests_namespace }}
spec:
  channel: {{ oc_packagemanifests_channel }}
"""

FAKE_PATH = Path("/tmp/sub_regression_test.yaml")


def test_detection():
    assert is_jinja_template(JINJA_YAML)
    assert not is_jinja_template("plain:\n  yaml: content\n")
    print("PASS: is_jinja_template correctly distinguishes Jinja YAML from plain YAML")


def test_plain_yaml_parser_would_reject_this():
    """Documents WHY this bug existed -- plain yaml.safe_load genuinely can't parse this."""
    try:
        list(yaml.safe_load_all(JINJA_YAML))
        raise AssertionError("expected plain YAML parsing to fail on Jinja2 syntax")
    except yaml.YAMLError:
        print("PASS: confirmed plain yaml.safe_load cannot parse {{ }} syntax (root cause)")


def test_jinja_syntax_check_accepts_valid_template():
    ok, err = _check_syntax(FAKE_PATH, JINJA_YAML)
    assert ok, f"expected valid Jinja2 template to pass, got: {err}"
    print("PASS: valid Jinja2 template correctly passes the syntax check")


def test_jinja_syntax_check_rejects_broken_template():
    broken = "metadata:\n  name: {{ unterminated_var\n"
    ok, err = _check_syntax(FAKE_PATH, broken)
    assert not ok, "genuinely broken Jinja2 syntax should still be rejected"
    print("PASS: genuinely broken Jinja2 syntax is still correctly rejected")


def test_verify_fix_accepts_valid_template_fix():
    fixed = JINJA_YAML.replace(
        "namespace: {{ oc_packagemanifests_namespace }}",
        "namespace: {{ oc_packagemanifests_namespace | default('openshift-marketplace') }}",
    )
    result = verify_fix(Path("/tmp"), FAKE_PATH, JINJA_YAML, fixed, findings_before=[])
    assert result.accepted, f"valid Jinja2 template fix should be accepted, got reason: {result.reason}"
    print("PASS: end-to-end verify_fix accepts a valid fix to a Jinja2-templated file")


if __name__ == "__main__":
    test_detection()
    test_plain_yaml_parser_would_reject_this()
    test_jinja_syntax_check_accepts_valid_template()
    test_jinja_syntax_check_rejects_broken_template()
    test_verify_fix_accepts_valid_template_fix()
    print("\nALL JINJA2 TEMPLATE TESTS PASSED")
