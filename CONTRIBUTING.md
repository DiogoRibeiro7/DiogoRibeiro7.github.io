# Contributing

This repository is a Jekyll site with a small set of Python utilities for
validating and maintaining Markdown posts. Keep changes focused: article edits,
site changes, and maintenance scripts should be easy to review separately.

## Workflow

Start from an up-to-date branch:

```bash
git checkout master
git pull
git checkout -b my-change
```

Before committing, review the diff:

```bash
git status
git diff
```

## Validation

Run the project checks before opening a pull request:

```bash
pytest -q
npm test
bundle exec jekyll build
```

## Python Utilities

Supported maintenance scripts are documented in `README.md`. Avoid adding broad
one-off scripts that mutate many posts. If a new script is necessary, give it a
narrow purpose, add tests, and include a dry-run mode when it writes files.
