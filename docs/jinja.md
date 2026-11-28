# Ansible Jinja2 templates

Files under `templates/` (or any YAML containing `{{ }}` / `{% %}`) are
handled specially. Plain `yaml.safe_load` cannot parse Jinja2 syntax at
all — `{{ var }}` reads as an invalid YAML flow mapping — so without this,
both the Verifier's syntax check and `yamllint` would reject every fix to
this class of file regardless of correctness. These files are now
validated as Jinja2 templates (via the `jinja2` package) instead of as
YAML; `yamllint` is skipped for them entirely, while `ansible-lint`
(which is Jinja-aware) still covers them via its normal repo-wide scan.
