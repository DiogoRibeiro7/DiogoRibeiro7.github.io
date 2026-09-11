# Diogo Ribeiro Blog

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hosted with GH Pages](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://pages.github.com/)
[![Made with GH Actions](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github-actions&logoColor=white)](https://github.com/features/actions)

This repository contains the source for my personal website and technical blog.
It is built with Jekyll, the
[Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme, and a
small set of Python utilities for validating Markdown front matter.

## Repository Layout

- `_posts/` contains blog posts.
- `_pages/` contains standalone pages.
- `_data/`, `_includes/`, `_layouts/`, and `_sass/` contain Jekyll theme and site structure.
- `assets/images/` stores article images and generated figures.
- `assets/viz/` contains Python scripts used to regenerate custom figures.
- `code/` contains downloadable code examples linked from the site.
- `tests/` covers the supported Python maintenance utilities.

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
regenerate on their own; with almost 400 posts that takes about two minutes.
`bundle exec rake stop` stops the server; `JEKYLL_PORT` and `JEKYLL_ARGS`
override the port and the extra `jekyll serve` flags.

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

Point a post at one through the `header` block (`image`, `overlay_image`,
`teaser`, `og_image`, `twitter_image`), for example
`/assets/images/headers/network.jpg`. The stock photographs under
`assets/images/` remain available; prefer a header no recent post already uses.

## Validation

Run the same checks used during routine maintenance:

```bash
pytest -q
npm test
bundle exec jekyll build
```

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

## Supported Python Utilities

The Python surface is intentionally small. Keep scripts only when they are
tested, linked from the site, or used to generate checked-in assets.

Supported root utilities:

- `fix_date.py`: sync post front-matter dates with `YYYY-MM-DD` filenames.
- `check_summary.py`: report posts missing `summary` or `keywords`.
- `markdown_category_checker.py`: report posts with multiple categories.
- `replace_latex.py`: convert legacy inline LaTeX delimiters where appropriate.

Figure and example code:

- `assets/viz/generate_figures.py`: regenerate custom article figures.
- `assets/viz/housestyle.py`: shared plotting style for generated figures.
- `code/michelson_morley.py`: downloadable example linked from the site.

Avoid adding broad one-off mutation scripts to the repository root. Prefer a
tested utility with a narrow purpose, a dry-run mode when it writes files, and a
short note in this README.
