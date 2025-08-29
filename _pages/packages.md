---
layout: single
title: R Packages
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
---

# R Packages

This section contains documentation for R packages I've developed, demonstrating best practices in R package development, documentation, and testing.

## Available Packages

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

For more R packages and data science content, check out my [GitHub profile](https://github.com/DiogoRibeiro7).

