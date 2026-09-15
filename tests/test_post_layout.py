from pathlib import Path


def test_no_markdown_posts_live_directly_under_posts_root():
    flat_posts = sorted(Path("_posts").glob("*.md"))
    assert flat_posts == [], (
        "Published posts must live in subject folders under _posts/: "
        + ", ".join(str(path) for path in flat_posts)
    )
