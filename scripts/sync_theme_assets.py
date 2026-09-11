"""Copy the parts of the DataLog theme that Jekyll cannot read from the submodule.

Jekyll reads layouts, includes, plugins and Sass straight from the
``vendor/datalog`` submodule through ``layouts_dir``, ``includes_dir``,
``plugins_dir`` and ``sass_dir`` in ``_config.yml``. It has no equivalent
setting for ``assets`` or for a second ``_data`` directory, so those files
are copied into the site by this script. Run it after updating the
submodule:

    git submodule update --remote vendor/datalog
    python scripts/sync_theme_assets.py

Use ``--dry-run`` to list what would be copied without writing anything.

Only theme infrastructure is copied. Data that describes the site itself
(navigation, social links, author) is owned by this repository and is not
touched.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THEME = ROOT / "vendor" / "datalog"

# (source directory or file, destination) pairs, relative to the theme and
# to the site root respectively. Directories are copied recursively.
SYNC_PATHS: tuple[tuple[str, str], ...] = (
    ("assets/css/main.scss", "assets/css/main.scss"),
    ("assets/js", "assets/js"),
    ("assets/img", "assets/img"),
    ("_data/i18n", "_data/i18n"),
    ("_data/js_manifest.json", "_data/js_manifest.json"),
    ("_data/js_meta.json", "_data/js_meta.json"),
    ("_data/cdn-integrity.yml", "_data/cdn-integrity.yml"),
)

# Files under the synced directories that belong to the theme's own tooling,
# not to a published site.
SKIP_NAMES = {"node_modules", ".DS_Store", "README.md"}


def iter_files(source: Path):
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and not (SKIP_NAMES & set(path.relative_to(source).parts)):
            yield path


def plan(theme: Path = THEME, root: Path = ROOT) -> list[tuple[Path, Path]]:
    """Return the (source, destination) file pairs that are missing or differ."""
    pairs: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in SYNC_PATHS:
        source = theme / src_rel
        if not source.exists():
            continue
        destination = root / dst_rel
        for src_file in iter_files(source):
            dst_file = destination / src_file.relative_to(source) if source.is_dir() else destination
            if dst_file.exists() and filecmp.cmp(src_file, dst_file, shallow=False):
                continue
            pairs.append((src_file, dst_file))
    return pairs


def sync(theme: Path = THEME, root: Path = ROOT, dry_run: bool = False) -> list[tuple[Path, Path]]:
    pairs = plan(theme, root)
    for src_file, dst_file in pairs:
        if dry_run:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list the files that would be copied")
    args = parser.parse_args(argv)

    if not THEME.exists():
        parser.error(f"theme submodule not found at {THEME}; run `git submodule update --init`")

    pairs = sync(dry_run=args.dry_run)
    verb = "would copy" if args.dry_run else "copied"
    for src_file, dst_file in pairs:
        print(f"{verb} {src_file.relative_to(THEME)} -> {dst_file.relative_to(ROOT)}")
    print(f"{len(pairs)} file(s) {verb}; {'nothing written' if args.dry_run else 'done'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
