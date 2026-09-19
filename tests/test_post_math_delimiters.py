"""Math in posts has to use delimiters this site typesets.

Inline math is ``$x$`` (or the older ``$$x$$`` inside a sentence). kramdown reads
``\\(`` and ``\\)`` as escaped parentheses and drops the backslashes, so the reader
sees "(x)" as plain text. A display block opened with ``$$`` on its own line and
never closed swallows the headings and paragraphs that follow it; two such blocks
in one post cancel out in a simple count, so the check is structural.
"""

import re
from pathlib import Path

POSTS = Path(__file__).resolve().parent.parent / "_posts"

# Rewritten in pull request 438; remove this entry once it is merged.
KNOWN = {"2026-09-16-why_exact_post_selection_confidence_intervals_can_be_enormous.md"}

FENCE = ("```", "~~~")
INLINE_CODE = re.compile(r"`[^`\n]*`")
LONGEST_BLOCK = 26


def prose_lines(path: Path):
    """Yield (line number, text) outside front matter and fenced code, inline code removed."""
    in_code = False
    front = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if number == 1 and stripped == "---":
            front = 1
            continue
        if front == 1:
            if stripped == "---":
                front = 2
            continue
        if stripped.startswith(FENCE):
            in_code = not in_code
            continue
        if not in_code:
            yield number, INLINE_CODE.sub("", line)


def posts():
    return sorted(p for p in POSTS.rglob("*.md") if p.name not in KNOWN)


def test_no_backslash_parenthesis_inline_math():
    offenders = [
        f"{path.relative_to(POSTS)}:{number}"
        for path in posts()
        for number, line in prose_lines(path)
        if "\\(" in line or "\\)" in line
    ]
    assert not offenders, "write inline math as $x$, not \\(x\\): " + ", ".join(offenders[:10])


def test_display_blocks_are_closed():
    offenders = []
    for path in posts():
        start = None
        body = []
        for number, line in prose_lines(path):
            if line.strip() == "$$":
                if start is None:
                    start, body = number, []
                else:
                    if any(text.startswith("#") for text in body) or len(body) > LONGEST_BLOCK:
                        offenders.append(f"{path.relative_to(POSTS)}:{start}")
                    start = None
            elif start is not None:
                body.append(line)
        if start is not None:
            offenders.append(f"{path.relative_to(POSTS)}:{start}")
    assert not offenders, "display block opened with $$, not closed: " + ", ".join(offenders[:10])


def test_no_half_typed_delimiters():
    typo = re.compile(r"\$%|%%[A-Za-z\\]")  # "$$z = 1.96$%" and "%%X$$", both seen in posts
    offenders = [
        f"{path.relative_to(POSTS)}:{number}"
        for path in posts()
        for number, line in prose_lines(path)
        if typo.search(line)
    ]
    assert not offenders, "mistyped math delimiter: " + ", ".join(offenders[:10])
