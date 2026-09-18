---
layout: page
title: "imputation-methods"
permalink: /packages/imputation-methods/
author_profile: true
seo_title: "imputation-methods Python Package"
seo_description: "Project page for imputation-methods, a unified pandas API for more than forty missing-data imputation methods, from mean filling to MICE, Kalman filters and matrix completion."
---

`imputation-methods` puts 42 missing-data imputation methods behind one pandas API. Every imputer takes a numeric `DataFrame` and returns a new one with the same index and columns, leaving the input untouched, so swapping mean imputation for KNN, MICE, a Kalman filter or low-rank matrix completion is a one-line change and the evaluation code stays the same.

The methods are grouped into families: statistical (mean, median, quantile, trimmed mean, group mean, indicator), donor sampling (hot deck, cold deck), time series (LOCF, NOCB, interpolation, moving averages, trend, seasonal, Kalman filter), nearest neighbours, regression (stochastic regression, predictive mean matching, Bayesian ridge, Huber, RANSAC, Gaussian process), iterative (MICE, MissForest, EM-style chained equations), matrix completion (SoftImpute, probabilistic PCA), neural (autoencoder, GAIN) and ensembles (fallback chains, stacking, bagging). Each class also has a functional shortcut, imputers with a random component accept `random_state`, and `rmse` and `mae` helpers score an imputation on cells that were hidden on purpose. The core depends only on NumPy, pandas, SciPy and scikit-learn, and the code is type-checked with mypy in strict mode.

## Install

```bash
pip install imputation-methods
```

The optional `viz` extra adds matplotlib and seaborn for the example notebooks and scripts.

## Project Links

- **PyPI:** [imputation-methods](https://pypi.org/project/imputation-methods/)
- **Documentation:** [diogoribeiro7.github.io/imputation-methods](https://diogoribeiro7.github.io/imputation-methods/)
- **Source:** [github.com/DiogoRibeiro7/imputation-methods](https://github.com/DiogoRibeiro7/imputation-methods)
- **Issues:** [github.com/DiogoRibeiro7/imputation-methods/issues](https://github.com/DiogoRibeiro7/imputation-methods/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.2.0`
- **Requires Python:** `>=3.10`
- **License:** MIT
- **Status:** beta

## Where It Fits

Use it when the choice of imputation method is itself a question: hide a share of a complete dataset, run several imputers through the same loop and compare the error on the hidden cells before committing to one. Inputs must be numeric, so encode categorical columns first, and the time-series imputers rely on row order, so sort before imputing.
