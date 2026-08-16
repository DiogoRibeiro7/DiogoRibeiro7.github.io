---
layout: single
title: Projects & Packages
permalink: /packages/
author_profile: true
header:
  image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_1.jpg
  teaser: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  twitter_image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_1.jpg
seo_type: article
seo_title: "Open Source Projects and Packages"
seo_description: "R and Python packages, documentation and project notes by Diogo Ribeiro, including survival analysis and causal inference tools."
---

## Open Source Projects & Packages

This page collects R and Python projects that are more useful as technical assets than as generic blog posts: problem framing, methods, code and documentation.

## R Packages

### Available Packages

### [myrpackage](/packages/myrpackage/)

A multilingual greeting and farewell package that serves as an example of proper R package structure.

**Features:**

- Multilingual support (English, Spanish, French, Portuguese, German, Italian)
- Comprehensive documentation with roxygen2
- Full test coverage with testthat
- Continuous integration with GitHub Actions
- Proper package structure following R standards

[View Documentation →](/packages/myrpackage/)

--------------------------------------------------------------------------------

### [unconfoundedr](/packages/unconfoundedr/)

Test (un)confoundedness by comparing an effect from an RCT-like dataset to the same estimand from an observational dataset. Includes robust estimators, inference, and transportability tools.

**Features:**

- IPW and AIPW (doubly robust) estimators for the marginal ATE
- Bootstrap confidence intervals and Wald test
- Transport modes: `none`, `rct_to_obs`, and `auto` (KS/energy shift detection)
- Diagnostics for propensity overlap, stabilized weights, trimming, and transport ESS

[View Documentation →](/packages/unconfoundedr/)

--------------------------------------------------------------------------------

## Python Packages

This section includes Python libraries I've developed or contributed to, with a focus on statistical modeling, survival analysis, and reproducibility.

### Available Packages

### [genSurvPy](/packages/gensurvpy/)

A Python package for generalized survival analysis. It includes tools for simulation, estimation, and diagnostics in complex time-to-event models, including proportional hazards and accelerated failure time frameworks.

**Features:**

- Flexible simulation of survival datasets under user-defined models
- Support for right-censored and interval-censored data
- Estimation under both PH and AFT models using parametric or semi-parametric methods
- Modular API for custom hazard and survival functions
- Publication-ready plots using matplotlib and seaborn
- Documentation built with Sphinx, deployed to both Read the Docs and this site

[View Documentation →](/packages/gensurvpy/)

--------------------------------------------------------------------------------

For more R and Python packages, as well as general data science content, visit my [GitHub profile](https://github.com/DiogoRibeiro7).

