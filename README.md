# Diogo Ribeiro Blog

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hosted with GH Pages](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://pages.github.com/)
[![Made with GH Actions](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github-actions&logoColor=white)](https://github.com/features/actions)

This repository contains the source for my personal website and technical blog.
It is built with Jekyll and the DataLog theme.

## Repository Layout

- `_posts/<subject>/` contains published blog posts grouped by primary subject.
- `_drafts/ideas/` and `_drafts/phd/` contain unpublished planning notes.
- `_pages/` contains standalone pages.
- `_data/`, `_includes/`, `_layouts/`, and `_sass/` contain Jekyll theme and site structure.
- `assets/images/` stores article images and generated figures.
- `assets/viz/` contains Python scripts used to regenerate custom figures.
- `code/` contains downloadable code examples linked from the site.
- `tests/` covers supported repository tooling and structural invariants.

The post-folder convention is documented in `docs/POST_ORGANIZATION.md`. No
Markdown post should live directly under `_posts/`.

## Theme

The site is built on the [DataLog](https://github.com/DiogoRibeiro7/analytics-blog-jekyll)
theme, checked out as the `vendor/datalog` git submodule. `_config.yml` points
`layouts_dir`, `includes_dir`, `plugins_dir` and `sass_dir` at the submodule, so
layouts, includes, plugins and styles are read from it directly. Jekyll cannot
redirect `assets` or a second `_data` directory, so those files are copied in by
`scripts/sync_theme_assets.py`; the theme's JS bundles are built with its own
Node toolchain first.

To update the theme:

```bash
git submodule update --remote vendor/datalog
(cd vendor/datalog && npm ci && npm run build:js)
python scripts/sync_theme_assets.py
```

Changes to the theme itself are made inside `vendor/datalog` and go upstream
through that repository; this repository only records which theme commit it
builds against.

## Setup

Clone with the submodule and install the toolchains:

```bash
git clone --recurse-submodules https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io
bundle install
(cd vendor/datalog && npm ci && npm run build:js)
python scripts/sync_theme_assets.py
python -m pip install -r requirements.txt
```

## Local Development

Run the site locally:

```bash
bundle exec rake serve
```

This wraps `jekyll serve --livereload` on http://127.0.0.1:4000/ and restarts
it when `_config.yml` or the theme under `vendor/datalog` changes, re-copying
the theme's assets first. Plain `jekyll serve` never watches either of those:
it ignores the config file and everything under `exclude`, so config and theme
changes would only show up after a manual restart. Posts, pages and `_data`
regenerate on their own. `bundle exec rake stop` stops the server; `JEKYLL_PORT`
and `JEKYLL_ARGS` override the port and extra `jekyll serve` flags.

Build the site without starting a server:

```bash
bundle exec rake build
```

The theme's JavaScript bundles are built inside the submodule (`npm run
build:js` in `vendor/datalog`, see Setup); the site itself has no Node build.

## Header Images

`assets/images/headers/` holds original 16:9 header graphics drawn by
`assets/viz/generate_headers.py` in the site palette on a dark ground, so a
white title stays readable in the post hero. Regenerate or extend the pool with:

```bash
cd assets/viz
python generate_headers.py             # all headers
python generate_headers.py walks cells # named headers only
```

`assets/viz/fetch_headers.py` adds photographs from Wikimedia Commons under CC0,
public-domain, CC BY or CC BY-SA licences and records each one's author, licence
and source page in `assets/images/headers/CREDITS.md` and `_data/image_credits.yml`,
which `_pages/image-credits.md` renders at `/image-credits/`.

Point a post at one through the `header` block (`image`, `overlay_image`,
`teaser`, `og_image`, `twitter_image`), for example
`/assets/images/headers/network.jpg`.

## Validation

Run the same checks used during routine maintenance:

```bash
pytest -q
npm test
bundle exec jekyll build
```

Python tests cover the supported theme-asset synchronisation tooling and the
repository layout invariant that published Markdown files belong in subject
folders rather than directly under `_posts/`.

The Jekyll build may report existing theme deprecation warnings. Treat build
failures as blockers; warnings should be reviewed when they point to content in
this repository rather than upstream theme code.

## Editorial Standard

New posts should have a reader-first reason to exist. Prefer articles built from
original analysis, code, experiments, worked examples, case studies, or practical
decision guidance. Avoid adding generic paraphrases of material already widely
available elsewhere.

Substantial posts can expose their contribution in front matter:

```yaml
why_this_exists: "What this post adds beyond a generic tutorial."
evidence: "Dataset, simulation, source material, or project experience used."
methodology: "How the analysis, comparison, or example was produced."
reviewed_at: 2026-08-16
```

Those fields render in the article provenance note and make review easier.

## Supported Python Tooling

The Python surface is intentionally small. Keep scripts only when they are
tested, linked from the site, or used to generate checked-in assets.

Supported repository tooling includes:

- `scripts/sync_theme_assets.py`: synchronise theme-owned assets into this site.
- `assets/viz/generate_figures.py`: regenerate custom article figures.
- `assets/viz/generate_2026_evidence_articles.py`: reproduce the autumn 2026 article tables and figures on longitudinal design, unlabelled monitoring, numerical verification, wearable alerts, microwave energy, and GDP release vintages. The GDP example reads a small, sourced CSV under `assets/data/`; the other examples use synthetic inputs or physical constants.
- `assets/viz/generate_science_communication_figures.py`: reproduce the seven figures and core calculations for the six science-communication archive articles, including KL divergence and accumulated climate evidence, seasonal heat storage, and overlapping random runs. Additional examples have executable snippets in the articles. Use `--dry-run` to print calculations without writing figures. The calculations use the standard library; plotting requires Matplotlib.
- `assets/viz/housestyle.py`: shared plotting style for generated figures.
- `code/michelson_morley.py`: downloadable example linked from the site.

Avoid adding broad one-off mutation scripts to the repository root. Prefer a
small tested utility only when it supports an ongoing repository workflow.
