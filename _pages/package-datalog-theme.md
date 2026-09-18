---
layout: page
title: "datalog-theme"
permalink: /packages/datalog-theme/
author_profile: true
seo_title: "datalog-theme Ruby Gem"
seo_description: "Project page for datalog-theme, the DataLog Jekyll theme for data science and research writing, published on RubyGems. This site runs on it."
---

`datalog-theme` is the gem distribution of DataLog, an academic-inspired Jekyll theme for data scientists, researchers and technical writers who publish reproducible analyses, papers, tutorials, datasets and project showcases. It is the theme this site runs on.

It ships research-ready layouts for posts, pages, datasets and portfolio case studies; a notebook converter that turns `.ipynb` files into sanitised, CSP-safe pages; MathJax tooling with accessible numbering and cross-references; syntax highlighting tuned for Python, R, SQL and Julia; lazy-loaded support for Plotly, D3.js, Bokeh and Observable visualisations; SEO metadata with Open Graph, canonical links and JSON-LD; citation exports through `CITATION.cff`, BibTeX, RIS and EndNote; a dark mode that remembers the reader's choice; and a responsive interface built to WCAG 2.1 AA.

## Install

Add the gem to the site's `Gemfile`:

```ruby
gem "datalog-theme", "~> 0.9.0"
```

Then enable it in `_config.yml`. Naming the theme under `plugins:` registers its Liquid tags:

```yml
theme: datalog-theme
plugins:
  - datalog-theme
```

Until 1.0 a minor release may include breaking changes, which the changelog lists.

## Project Links

- **RubyGems:** [datalog-theme](https://rubygems.org/gems/datalog-theme)
- **Documentation:** [README and guides](https://github.com/DiogoRibeiro7/analytics-blog-jekyll#readme)
- **Source:** [github.com/DiogoRibeiro7/analytics-blog-jekyll](https://github.com/DiogoRibeiro7/analytics-blog-jekyll)
- **Issues:** [github.com/DiogoRibeiro7/analytics-blog-jekyll/issues](https://github.com/DiogoRibeiro7/analytics-blog-jekyll/issues)
- **Changelog:** [GitHub releases](https://github.com/DiogoRibeiro7/analytics-blog-jekyll/releases)

## Package Metadata

- **Current release:** `0.9.0`
- **Requires Ruby:** `>= 3.2`
- **Requires Jekyll:** `~> 4.3`
- **License:** MIT

## Where It Fits

Use it for a technical blog or research site where mathematics, code, notebooks and figures are the content and the theme should stay out of their way. The repository doubles as a demo site that previews every feature, and it includes a guide for migrating from Minimal Mistakes that keeps existing front matter and URLs.
