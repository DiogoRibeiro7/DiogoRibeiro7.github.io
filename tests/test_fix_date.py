import os
import sys
import tempfile
import frontmatter
import pytest


# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fix_date


def test_extract_date_from_filename():
    assert fix_date.extract_date_from_filename('2023-12-01-post.md') == '2023-12-01'
    assert fix_date.extract_date_from_filename('no-date.md') is None


def create_markdown_file(path, front_matter):
    post = frontmatter.Post(content="Body", **front_matter)
    content = frontmatter.dumps(post)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def load_front_matter(path):
    with open(path, 'r', encoding='utf-8') as f:
        return frontmatter.load(f)


@pytest.mark.parametrize("initial_frontmatter, expected_date", [
    ({'title': 'Test'}, '2024-02-03'),
    ({'title': 'Test', 'date': '2020-01-01'}, '2024-02-03'),
])
def test_process_markdown_file_updates_date(tmp_path, initial_frontmatter, expected_date):
    file_path = tmp_path / '2024-02-03-test.md'
    create_markdown_file(file_path, initial_frontmatter)

    fix_date.process_markdown_file(str(file_path))

    updated = load_front_matter(file_path)
    assert updated['date'] == expected_date
