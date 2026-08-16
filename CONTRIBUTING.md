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

## Content Quality Standard

This site should not accumulate generic articles just to cover keywords. Before
adding or expanding a post, make sure it has a reason to exist for a reader who
could already find a generic explanation elsewhere.

Publish or keep an article when it contributes at least one of the following:

- original analysis, code, figures, simulations, or experiments;
- a worked example with explicit assumptions and failure modes;
- a practical checklist, diagnostic, or reusable tool;
- a case study or professional judgment from applied work;
- a clear comparison of methods that helps someone make a decision.

Do not publish thin summaries, broad paraphrases, or mass-produced topic pages.
Revise, merge, mark `noindex`, or remove posts that do not add useful knowledge.

For substantial new posts, prefer front-matter fields that make the contribution
explicit:

```yaml
why_this_exists: "What this post adds beyond a generic tutorial."
evidence: "Dataset, simulation, source material, or project experience used."
methodology: "How the analysis, comparison, or example was produced."
reviewed_at: 2026-08-16
```

These fields are shown in the article provenance note when present. Follow
Google's own Search guidance by prioritizing helpful, reliable, people-first
content over search-engine-first pages:

- https://developers.google.com/search/docs/fundamentals/creating-helpful-content
- https://developers.google.com/search/docs/fundamentals/ai-optimization-guide
- https://developers.google.com/search/blog/2023/02/google-search-and-ai-content

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
