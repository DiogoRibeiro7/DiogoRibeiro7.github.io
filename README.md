# Personal Blog on Minimal Mistakes

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hosted with GH Pages](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://pages.github.com/)
[![Made with GH Actions](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github-actions&logoColor=white)](https://github.com/features/actions)

This repository contains the source code for my website built with the
[Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme.
It also includes a few helper scripts to clean up Markdown front matter and
run tests.

## Requirements

The site relies on Ruby, Node and Python tooling.  Install dependencies with:

```bash
# Python packages for helper scripts and tests
pip install -r requirements.txt

# JavaScript dependencies
npm install

# Ruby gems for Jekyll
bundle install
```

## Development

Use the following commands while working on the site:

```bash
# start a local server at http://localhost:4000/
bundle exec jekyll serve

# rebuild JavaScript when files change
npm run watch:js

# lint stylesheets
npm run lint:css
```

## Tests

Front matter utilities are covered by a small `pytest` suite.  Run the tests with:

```bash
pytest
```

GitHub Actions executes the same tests on every push.

## Roadmap

Planned improvements are organised as sprints in [ROADMAP.md](ROADMAP.md).
Highlights include:

- refining typography and the colour palette
- restructuring the homepage with card‑style articles
- adding search and dark mode
- optimising performance and accessibility

Contributions are welcome!
