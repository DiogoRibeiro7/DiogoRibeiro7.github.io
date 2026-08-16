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

## Setup

Install the three toolchains used by the project:

```bash
bundle install
npm install
python -m pip install -r requirements.txt
```

## Local Development

Run the site locally:

```bash
bundle exec jekyll serve
```

Build the site without starting a server:

```bash
bundle exec jekyll build
```

JavaScript and stylesheet tasks:

```bash
npm run build:js
npm run lint:css
```

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
