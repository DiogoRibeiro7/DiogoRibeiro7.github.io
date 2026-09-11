import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sync_theme_assets as sync_module  # noqa: E402


def make_theme(tmp_path: Path) -> tuple[Path, Path]:
    theme = tmp_path / "theme"
    root = tmp_path / "site"
    (theme / "assets" / "css").mkdir(parents=True)
    (theme / "assets" / "css" / "main.scss").write_text('@use "theme";\n')
    (theme / "assets" / "js" / "core").mkdir(parents=True)
    (theme / "assets" / "js" / "core" / "nav.js").write_text("export const nav = 1;\n")
    (theme / "assets" / "js" / "node_modules").mkdir()
    (theme / "assets" / "js" / "node_modules" / "dep.js").write_text("ignored\n")
    (theme / "_data" / "i18n").mkdir(parents=True)
    (theme / "_data" / "i18n" / "en.yml").write_text("post:\n  toc_heading: On this page\n")
    root.mkdir()
    return theme, root


def test_plan_lists_missing_files_and_skips_tooling(tmp_path):
    theme, root = make_theme(tmp_path)
    pairs = sync_module.plan(theme, root)
    copied = sorted(str(dst.relative_to(root)).replace("\\", "/") for _, dst in pairs)
    assert copied == ["_data/i18n/en.yml", "assets/css/main.scss", "assets/js/core/nav.js"]


def test_sync_copies_and_is_idempotent(tmp_path):
    theme, root = make_theme(tmp_path)
    first = sync_module.sync(theme, root)
    assert len(first) == 3
    assert (root / "assets" / "js" / "core" / "nav.js").read_text() == "export const nav = 1;\n"
    assert not (root / "assets" / "js" / "node_modules").exists()
    assert sync_module.sync(theme, root) == []


def test_sync_refreshes_changed_files(tmp_path):
    theme, root = make_theme(tmp_path)
    sync_module.sync(theme, root)
    (theme / "assets" / "css" / "main.scss").write_text('@use "theme";\n// changed\n')
    pairs = sync_module.sync(theme, root)
    assert [dst.name for _, dst in pairs] == ["main.scss"]
    assert "changed" in (root / "assets" / "css" / "main.scss").read_text()


def test_dry_run_writes_nothing(tmp_path):
    theme, root = make_theme(tmp_path)
    pairs = sync_module.sync(theme, root, dry_run=True)
    assert len(pairs) == 3
    assert not (root / "assets").exists()
