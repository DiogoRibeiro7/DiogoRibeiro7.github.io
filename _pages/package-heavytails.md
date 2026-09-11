---
layout: page
title: "heavytails"
permalink: /packages/heavytails/
author_profile: true
seo_title: "heavytails Python Package"
seo_description: "Project page for heavytails, a NumPy-vectorised library of heavy-tailed distributions, tail index estimators and extreme value diagnostics."
---

`heavytails` implements continuous and discrete heavy-tailed distributions, tail index estimators and diagnostic utilities with NumPy-backed vectorised evaluation. Every density, quantile and sampler is derived from first principles so the code can be read, checked and taught. Survival functions are computed directly rather than as `1 - cdf(x)`, so they stay accurate far into the tail where the subtraction has lost every significant digit.

Beyond the distribution interface it covers tail index estimation (Hill-family, robust, bias-reduced, threshold-averaged and peaks-over-threshold estimators), parameter fitting by maximum likelihood and method of moments with AIC and BIC comparison, log-log tail and QQ diagnostics, applied extreme value theory (threshold selection with mean residual life and parameter-stability plots, generalized Pareto fitting, return levels, tail-risk measures, actuarial frequency and severity models, streaming estimators), and dependent extremes (elliptical and multivariate Student-t models, tail dependence, Gaussian, Student-t, Gumbel and Galambos copulas, GARCH fitting, the extremal index and declustering). A command-line interface handles sampling, fitting, comparison and benchmarking, and the package ships type annotations with a `py.typed` marker.

## Install

```bash
pip install heavytails
```

## Project Links

- **PyPI:** [heavytails](https://pypi.org/project/heavytails/)
- **Documentation:** [diogoribeiro7.github.io/heavytails](https://diogoribeiro7.github.io/heavytails)
- **Source:** [github.com/DiogoRibeiro7/heavytails](https://github.com/DiogoRibeiro7/heavytails)
- **Issues:** [github.com/DiogoRibeiro7/heavytails/issues](https://github.com/DiogoRibeiro7/heavytails/issues)
- **Discussions:** [github.com/DiogoRibeiro7/heavytails/discussions](https://github.com/DiogoRibeiro7/heavytails/discussions)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/heavytails/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.6.3`
- **Requires Python:** `>=3.10,<3.14`
- **Status:** beta

## Where It Fits

Use it for research, teaching and simulation in risk, finance, insurance and extreme-value analysis: when a Gaussian assumption is the thing under test, when a tail index or a return level is the quantity of interest, or when a simulation needs samplers whose tails you can trust. It is the distribution layer behind several of the heavy-tail and risk articles on this site.
