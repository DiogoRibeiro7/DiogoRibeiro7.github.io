"""Validate internal hyperlinks in the generated Jekyll site.

The checker scans built HTML under _site and validates only navigational
anchor href links that resolve to this site. External links are deliberately
out of scope so CI remains deterministic and does not depend on third-party
availability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


SITE_HOST = "diogoribeiro7.github.io"


@dataclass(frozen=True)
class Link:
    """One hyperlink discovered in a generated HTML document."""

    source_file: Path
    source_url: str
    href: str
    line: int


class LinkParser(HTMLParser):
    """Collect navigational links and element identifiers from one HTML page."""

    def __init__(self, source_file: Path, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_file = source_file
        self.source_url = source_url
        self.links: list[Link] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect anchor hrefs and all HTML ids or named anchors."""
        values = dict(attrs)

        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)

        if tag == "a":
            anchor_name = values.get("name")
            if anchor_name:
                self.ids.add(anchor_name)

            href = values.get("href")
            if href:
                self.links.append(
                    Link(
                        source_file=self.source_file,
                        source_url=self.source_url,
                        href=href,
                        line=self.getpos()[0],
                    )
                )


def file_to_url(site_root: Path, html_file: Path) -> str:
    """Convert a generated HTML file path to its public URL path."""
    relative = html_file.relative_to(site_root).as_posix()
    if relative == "index.html":
        return "/"
    if relative.endswith("/index.html"):
        return "/" + relative[: -len("index.html")]
    return "/" + relative


def candidate_paths(site_root: Path, url_path: str) -> list[Path]:
    """Return generated-file candidates for one internal URL path."""
    path = unquote(url_path)
    if not path.startswith("/"):
        path = "/" + path

    relative = path.lstrip("/")
    if not relative:
        return [site_root / "index.html"]

    target = site_root / relative
    candidates = [target]

    if path.endswith("/"):
        candidates.append(target / "index.html")
    elif not Path(relative).suffix:
        candidates.extend(
            [
                site_root / f"{relative}.html",
                target / "index.html",
            ]
        )

    return list(dict.fromkeys(candidates))


def resolve_target(
    site_root: Path,
    source_url: str,
    href: str,
) -> tuple[Path | None, str]:
    """Resolve an internal href to a generated file and optional fragment."""
    stripped = href.strip()
    if not stripped or stripped.startswith(
        ("mailto:", "tel:", "javascript:", "data:")
    ):
        return None, ""

    absolute = urljoin(f"https://{SITE_HOST}{source_url}", stripped)
    parsed = urlparse(absolute)

    if parsed.scheme not in {"http", "https"}:
        return None, ""
    if parsed.netloc and parsed.netloc.lower() != SITE_HOST:
        return None, ""

    for candidate in candidate_paths(site_root, parsed.path):
        if candidate.is_file():
            return candidate, unquote(parsed.fragment)

    return Path("__MISSING__") / parsed.path.lstrip("/"), unquote(parsed.fragment)


def parse_site(site_root: Path) -> tuple[list[Link], dict[Path, set[str]]]:
    """Parse generated HTML and return links plus per-page anchors."""
    links: list[Link] = []
    anchors: dict[Path, set[str]] = {}

    for html_file in sorted(site_root.rglob("*.html")):
        source_url = file_to_url(site_root, html_file)
        parser = LinkParser(source_file=html_file, source_url=source_url)
        parser.feed(html_file.read_text(encoding="utf-8", errors="replace"))
        links.extend(parser.links)
        anchors[html_file.resolve()] = parser.ids

    return links, anchors


def validate(site_root: Path) -> list[str]:
    """Return human-readable validation failures."""
    links, anchors = parse_site(site_root)
    failures: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for link in links:
        target, fragment = resolve_target(site_root, link.source_url, link.href)
        if target is None:
            continue

        key = (link.source_url, link.href, str(target))
        if key in seen:
            continue
        seen.add(key)

        if "__MISSING__" in target.parts:
            failures.append(
                f"{link.source_url}:{link.line} -> {link.href} (missing target)"
            )
            continue

        if fragment and target.suffix.lower() == ".html":
            target_ids = anchors.get(target.resolve())
            if target_ids is None:
                parser = LinkParser(target, file_to_url(site_root, target))
                parser.feed(
                    target.read_text(encoding="utf-8", errors="replace")
                )
                target_ids = parser.ids
                anchors[target.resolve()] = target_ids

            if fragment not in target_ids:
                failures.append(
                    f"{link.source_url}:{link.line} -> {link.href} "
                    f"(missing anchor #{fragment})"
                )

    return failures


def main() -> int:
    """Run the checker from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_root",
        nargs="?",
        type=Path,
        default=Path("_site"),
        help="Generated Jekyll site directory (default: _site)",
    )
    args = parser.parse_args()

    site_root = args.site_root.resolve()
    if not site_root.is_dir():
        raise SystemExit(f"Generated site directory not found: {site_root}")

    failures = validate(site_root)
    if failures:
        print(f"Found {len(failures)} broken internal link(s):")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Internal link audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
