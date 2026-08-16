---
permalink: '/time-series/hierarchical_forecasting_reconciliation/'
title: 'Hierarchical Forecasting: Making Forecasts Add Up'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Supply Chain
- Statistical Modeling
author_profile: false
seo_title: 'Hierarchical Forecasting and Reconciliation'
seo_description: 'Forecasts by region rarely sum to the national forecast. Bottom-up, top-down and optimal reconciliation compared.'
excerpt: >-
  Forecast every store separately and the total will not match the forecast
  you made for the company. Reconciliation is how you make a hierarchy of
  forecasts coherent.
summary: >-
  An introduction to hierarchical and grouped forecasting: why independently
  produced forecasts are incoherent, the trade-offs between bottom-up and
  top-down aggregation, and how optimal reconciliation improves accuracy at
  every level rather than merely enforcing consistency.
keywords:
  - hierarchical forecasting
  - forecast reconciliation
  - bottom-up forecasting
  - MinT
  - grouped time series
classes: wide
date: '2026-08-09'
header:
  image: /assets/images/data_science_4.jpg
  og_image: /assets/images/data_science_4.jpg
  overlay_image: /assets/images/data_science_4.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_4.jpg
  twitter_image: /assets/images/data_science_4.jpg
---
Forecast each store separately, sum the results, and compare against the forecast you made for the company as a whole. The two numbers will not match. They almost never do.

This is the coherence problem, and it appears wherever forecasts exist at more than one level of aggregation — by product and category, by store and region, by SKU and channel. Reconciliation is the set of methods that make a hierarchy of forecasts add up, and the good ones improve accuracy while doing it.

## Why Incoherence Happens

Each series in a hierarchy is forecast independently, using whatever model fits it best. Those models make different assumptions, are fitted on different noise, and have no knowledge of each other. Nothing constrains their sum.

The consequence is organisational as much as statistical. Finance plans against the top-level number, operations plan against the store-level numbers, and the two plans are quietly inconsistent. Someone reconciles them in a spreadsheet, usually by scaling the bottom to match the top, which is the crudest reconciliation method available and rarely the best one.

Formally, write $y_t$ for the vector of *all* series in the hierarchy — every level, stacked. It relates to the bottom-level series $b_t$ through a summing matrix $S$:

$$
y_t = S b_t .
$$

For two stores summing to one total, $S = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$, giving total, store A, store B. A set of forecasts is **coherent** if it satisfies this relationship. Independently produced ("base") forecasts $\hat{y}$ generally do not.

## Bottom-Up

Forecast only the bottom level and aggregate upward.

This is coherent by construction and uses the most granular information available. Its weakness is noise: bottom-level series are the sparsest and most erratic in the hierarchy, and errors that are individually small can accumulate. It also cannot exploit structure visible only at aggregate levels — a company-wide seasonal pattern may be obvious in the total and invisible in any single store.

## Top-Down

Forecast the total and split it downward using historical proportions.

The top level is the smoothest and easiest to forecast, so the aggregate number is often good. The difficulty is the disaggregation: proportions based on historical averages assume the mix is stable, and it rarely is. A store growing faster than the chain will be systematically under-forecast, indefinitely, because its historical share understates its current one.

Top-down is also provably unable to preserve information from the bottom level — the individual series' own dynamics are discarded entirely.

## Middle-Out

Forecast at an intermediate level, aggregate upward and disaggregate downward. This is a compromise, useful when one level is meaningfully more reliable than the others — often regional, where series are neither as noisy as individual stores nor as abstract as the total.

## Optimal Reconciliation

The modern approach treats reconciliation as a projection. Produce base forecasts at *every* level, then adjust them minimally to become coherent:

$$
\tilde{y} = S(S^\top W^{-1} S)^{-1} S^\top W^{-1} \hat{y},
$$

where $W$ is the covariance matrix of the base forecast errors. This maps the incoherent base forecasts onto the space of coherent ones, weighting each series by how reliable it is.

The result is not merely consistency. Because every level contributes information, reconciliation typically *improves* accuracy at every level compared with the base forecasts — including the levels that were already fine. That is the property that makes it worth doing rather than a bookkeeping exercise.

Estimating $W$ is the practical difficulty. **OLS** reconciliation sets $W = I$, ignoring error variances entirely, which is simple but treats a noisy SKU and a stable total as equally trustworthy. **WLS** uses only the diagonal, scaling by each series' own error variance. **MinT** ("minimum trace") estimates the full covariance, usually with shrinkage toward the diagonal, and is generally the best performer.

```python
import numpy as np

# hierarchy: total = A + B, and A = A1 + A2
#            rows: [total, A, B, A1, A2, B]  -> bottom = [A1, A2, B]
S = np.array([
    [1, 1, 1],   # total
    [1, 1, 0],   # A
    [0, 0, 1],   # B
    [1, 0, 0],   # A1
    [0, 1, 0],   # A2
    [0, 0, 1],   # B (bottom)
], dtype=float)

base = np.array([100.0, 62.0, 35.0, 30.0, 28.0, 35.0])   # incoherent on purpose
print("base total     :", base[0], " sum of bottom:", base[3:].sum())

def reconcile(S, yhat, W=None):
    W = np.eye(S.shape[0]) if W is None else W
    Wi = np.linalg.inv(W)
    P = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
    return S @ (P @ yhat)

ols = reconcile(S, base)
# WLS: trust aggregate levels more (their relative error is smaller)
wls = reconcile(S, base, np.diag([1.0, 2.0, 2.0, 4.0, 4.0, 4.0]))

for name, r in (("OLS", ols), ("WLS", wls)):
    print(f"{name}: total={r[0]:.2f}  bottom sum={r[3:].sum():.2f}  "
          f"coherent={np.allclose(r, S @ r[3:])}")
```

Both produce coherent forecasts; they differ in *where* the adjustment is absorbed. OLS spreads it evenly, while WLS pushes more of the correction onto the series it considers less reliable — which is usually what you want.

## Grouped Rather Than Hierarchical

A strict hierarchy is a tree: each series has exactly one parent. Many real structures are **grouped**, where series can be aggregated along several independent dimensions — product category *and* geography *and* channel — that cross rather than nest.

Grouped structures cannot be represented as a tree, and top-down methods have no well-defined meaning in them, since there is no unique path from top to bottom. Optimal reconciliation handles both cases identically, which is another argument for it.

## Practical Notes

**Reconciliation is not a substitute for good base forecasts.** It redistributes information; it does not create it. Poor base forecasts reconcile into coherent poor forecasts.

**Check whether coherence is actually required.** If nobody consumes both levels, the effort may not be warranted. The value appears when different teams plan against different levels and inconsistency causes real friction.

**Probabilistic reconciliation is harder than the point version.** Reconciling forecast *distributions* so that aggregated distributions are consistent is an active research area, and naively summing prediction intervals overstates uncertainty at aggregate levels because it ignores the diversification effect of imperfectly correlated errors.

**Watch the zeros.** Bottom-level series in retail hierarchies are frequently intermittent, and reconciliation can push negative values onto series that cannot be negative. Non-negative reconciliation variants exist and are worth using where the constraint is real.

## References

- Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., & Shang, H. L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis*, 55(9), 2579-2589.
- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804-819.
- Athanasopoulos, G., Ahmed, R. A., & Hyndman, R. J. (2009). Hierarchical forecasts for Australian domestic tourism. *International Journal of Forecasting*, 25(1), 146-166.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
