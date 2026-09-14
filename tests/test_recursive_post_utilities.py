import frontmatter

from check_summary import check_front_matter
from fix_date import process_markdown_files_in_directory
from markdown_category_checker import process_markdown_files
from replace_latex import process_markdown_files_in_folder


def _write_nested_post(root):
    subject = root / "statistics"
    subject.mkdir(parents=True)
    post = subject / "2026-01-02-nested-post.md"
    post.write_text(
        """---
title: Nested post
date: '2026-01-01'
categories:
- Statistics
- Software Engineering
---

An equation: \\(x + 1\\).
""",
        encoding="utf-8",
    )
    return post


def test_post_utilities_recurse_into_subject_folders(tmp_path):
    root = tmp_path / "_posts"
    post_path = _write_nested_post(root)
    relative = "statistics/2026-01-02-nested-post.md"

    process_markdown_files_in_directory(root)
    assert str(frontmatter.load(post_path)["date"]) == "2026-01-02"

    summary_report = tmp_path / "summary.json"
    summary_results = check_front_matter(root, summary_report, "json")
    assert [result["file"] for result in summary_results] == [relative]

    category_report = tmp_path / "categories.json"
    category_results = process_markdown_files(root, category_report, "json")
    assert category_results == [
        {
            "file": relative,
            "categories": ["Statistics", "Software Engineering"],
        }
    ]

    process_markdown_files_in_folder(root)
    assert "$$x + 1$$" in post_path.read_text(encoding="utf-8")
