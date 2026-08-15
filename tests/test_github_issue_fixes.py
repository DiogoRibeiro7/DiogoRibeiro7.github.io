import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import check_summary
import fix_date
import markdown_category_checker
import replace_latex


def test_fix_date_dry_run_does_not_write(tmp_path):
    path = tmp_path / "2024-03-04-example.md"
    path.write_text("---\ntitle: Example\ndate: 2020-01-01\n---\n\nBody\n", encoding="utf-8")

    changed = fix_date.process_markdown_file(str(path), dry_run=True)

    assert changed is True
    assert "2020-01-01" in path.read_text(encoding="utf-8")


def test_replace_latex_skip_code_blocks(tmp_path):
    path = tmp_path / "example.md"
    path.write_text("```python\nprint('\\(x\\)')\n```\n\n\\(y\\)\n", encoding="utf-8")

    changed = replace_latex.replace_latex_syntax_in_file(str(path), skip_code_blocks=True)

    assert changed is False
    assert "\\(y\\)" in path.read_text(encoding="utf-8")


def test_check_summary_json_output(tmp_path):
    posts = tmp_path / "posts"
    posts.mkdir()
    (posts / "missing.md").write_text("---\ntitle: Missing\n---\n\nBody\n", encoding="utf-8")
    output = tmp_path / "report.json"

    results = check_summary.check_front_matter(str(posts), str(output), output_format="json")

    assert results == [{
        "file": "missing.md",
        "valid_front_matter": True,
        "summary_present": False,
        "keywords_present": False,
    }]
    assert json.loads(output.read_text(encoding="utf-8")) == results


def test_markdown_category_checker_json_output(tmp_path):
    posts = tmp_path / "posts"
    posts.mkdir()
    (posts / "multi.md").write_text("---\ncategories:\n- A\n- B\n---\n\nBody\n", encoding="utf-8")
    output = tmp_path / "categories.json"

    results = markdown_category_checker.process_markdown_files(
        str(posts),
        str(output),
        output_format="json",
    )

    assert results == [{"file": "multi.md", "categories": ["A", "B"]}]
    assert json.loads(output.read_text(encoding="utf-8")) == results
