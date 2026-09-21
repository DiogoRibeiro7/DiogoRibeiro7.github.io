"""Article diagrams must carry their own colours.

The SVGs under assets/images/articles are embedded from markdown with an image
tag, so a browser loads each one as an isolated document. In that context
`currentColor` has no page to inherit from and falls back to its initial value,
black. On the light theme that happens to look right; on the dark theme, whose
surfaces run from #171717 to #2a2a2a, it is black ink on near-black and the
diagram cannot be read at all.

Four diagrams shipped that way before this was noticed, and a fifth was authored
the same way a day later, so the rule is checked here rather than remembered.
A diagram that draws with `currentColor` has to set an explicit colour, and it
has to paint its own surface, which is what guarantees the contrast between them.
"""
import re
from pathlib import Path

import pytest

DIAGRAMS = sorted(Path("assets/images/articles").rglob("*.svg"))

# The ink and surface the site's own figures are saved with, see
# assets/viz/house.mplstyle. Diagrams match them so a figure looks the same
# whether it came from Matplotlib or was drawn by hand.
INK = "#0b0b0b"
SURFACE = "#f7f9fa"

WHY = (
    "An <img>-embedded SVG is an isolated document: currentColor cannot inherit "
    "the page's colour and falls back to black, which is invisible on the dark "
    "theme. Add a <style> block setting `svg {{ color: {ink}; }}` and a "
    "`.diagram-surface {{ fill: {surface}; }}` rect covering the viewBox, as the "
    "other diagrams in {folder} do."
).format(ink=INK, surface=SURFACE, folder="assets/images/articles")


def test_there_are_diagrams_to_check():
    """Guard against the glob silently matching nothing."""
    assert DIAGRAMS, "no SVGs found under assets/images/articles"


@pytest.mark.parametrize("path", DIAGRAMS, ids=lambda p: p.name)
def test_diagram_sets_its_own_ink_and_surface(path):
    svg = path.read_text(encoding="utf-8")
    if "currentColor" not in svg:
        pytest.skip("draws with explicit colours, so it does not depend on inheritance")

    assert re.search(r"svg\s*\{[^}]*\bcolor\s*:", svg), f"{path}: no explicit colour. {WHY}"
    assert re.search(r"\.diagram-surface\s*\{[^}]*\bfill\s*:", svg), (
        f"{path}: no surface rule. {WHY}"
    )
    assert re.search(r'<rect[^>]*class="diagram-surface"', svg), (
        f"{path}: the surface rule is never painted onto a rect. {WHY}"
    )


@pytest.mark.parametrize("path", DIAGRAMS, ids=lambda p: p.name)
def test_diagram_surface_covers_the_whole_viewbox(path):
    """A partial surface would leave ink sitting straight on the page again."""
    svg = path.read_text(encoding="utf-8")
    rect = re.search(r'<rect[^>]*class="diagram-surface"[^>]*/?>', svg)
    if rect is None:
        pytest.skip("checked by the rule above")

    attrs = rect.group(0)
    for name in ("width", "height"):
        value = re.search(rf'{name}="([^"]+)"', attrs)
        assert value, f"{path}: the surface rect has no {name}"
        assert value.group(1) == "100%", (
            f"{path}: the surface rect uses {name}={value.group(1)!r}; use 100% so it "
            "keeps covering the viewBox when the diagram is relaid out"
        )
