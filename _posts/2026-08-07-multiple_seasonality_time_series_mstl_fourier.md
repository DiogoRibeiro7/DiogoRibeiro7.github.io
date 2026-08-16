---
permalink: '/time-series/multiple_seasonality_time_series_mstl_fourier/'
title: 'Multiple Seasonality: MSTL, TBATS, and Fourier Terms'
categories:
- Time Series
tags:
- Time Series
- Forecasting
- Signal Processing
- Python
author_profile: false
seo_title: 'Multiple Seasonality: MSTL and Fourier Terms'
seo_description: 'Daily data usually has weekly and annual cycles at once. How MSTL, TBATS and Fourier terms handle more than one seasonal period.'
excerpt: >-
  Hourly and daily data rarely has one season. Electricity demand cycles
  daily, weekly and annually at the same time, and a single seasonal period
  cannot represent that.
summary: >-
  How to model series with several seasonal periods at once: why a single
  seasonal index fails on high-frequency data, how MSTL extends STL to
  multiple periods, where TBATS fits, and why Fourier terms are often the most
  practical answer.
keywords:
  - multiple seasonality
  - MSTL
  - TBATS
  - Fourier terms
  - high-frequency forecasting
classes: wide
date: '2026-08-07'
header:
  image: /assets/images/data_science_3.jpg
  og_image: /assets/images/data_science_3.jpg
  overlay_image: /assets/images/data_science_3.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_3.jpg
  twitter_image: /assets/images/data_science_3.jpg
---
Ask for the seasonal period of a daily electricity series and the question is already wrong. Consumption cycles across the week — weekdays differ from weekends — and across the year, as heating and cooling demand rises and falls. Hourly data adds a third cycle within the day.

Classical seasonal models take a single period $m$. That is adequate for monthly data with an annual cycle, and inadequate for almost anything sampled more frequently.

## Why One Seasonal Period Is Not Enough

Set $m = 7$ on daily data and the model captures the weekly rhythm while treating the annual cycle as trend or noise. Set $m = 365$ and you capture the annual pattern while the weekly one disappears into the residuals.

The problem compounds with SARIMA specifically. Its seasonal component requires estimating parameters at lag $m$, so an annual cycle in daily data means reaching back 365 observations. That demands years of history to estimate reliably, and the resulting model is unwieldy.

Two further complications appear at high frequency. Annual seasonality in daily data has a **non-integer period** — 365.25 days, thanks to leap years — which index-based seasonal methods cannot represent. And seasonal patterns interact: the shape of the daily cycle in summer differs from winter, which an additive combination of independent cycles cannot express.

## MSTL: Decomposition with Several Periods

MSTL extends STL to multiple seasonalities by applying STL iteratively, extracting one seasonal component at a time and passing the remainder to the next.

The result decomposes the series as

$$
y_t = T_t + \sum_{i} S_t^{(i)} + R_t,
$$

with one seasonal term per period. Because it inherits STL's machinery, each component may evolve gradually rather than repeating identically, and the robust variant downweights outliers so a single anomalous day does not distort every subsequent estimate of that weekday.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

rng = np.random.default_rng(0)
n = 365 * 3
idx = pd.date_range("2022-01-01", periods=n, freq="D")
t = np.arange(n)

series = pd.Series(
    50                                              # level
    + 0.01 * t                                      # slow trend
    + 6 * np.sin(2 * np.pi * t / 7)                 # weekly cycle
    + 15 * np.sin(2 * np.pi * t / 365.25)           # annual cycle
    + rng.normal(0, 1.5, n),
    index=idx)

res = MSTL(series, periods=(7, 365)).fit()
print("components:", [c for c in ("trend", "resid") ] + list(res.seasonal.columns))

var = {"seasonal_7": res.seasonal["seasonal_7"].var(),
       "seasonal_365": res.seasonal["seasonal_365"].var(),
       "trend": res.trend.var(),
       "resid": res.resid.var()}
total = sum(var.values())
for k, v in var.items():
    print(f"  {k:14} {100 * v / total:5.1f}% of variance")
```

Reporting the variance share of each component is the useful diagnostic. It tells you which cycle actually matters, and whether a residual share of 40% means there is structure the decomposition has not captured.

MSTL is additive, so apply it to log-transformed data when the seasonal amplitude grows with the level — which it usually does for demand series.

## Fourier Terms: The Practical Workhorse

The most flexible approach represents each seasonality as a small set of sine and cosine pairs used as regressors:

$$
x_{k,t}^{\sin} = \sin\!\left(\frac{2\pi k t}{m}\right), \qquad
x_{k,t}^{\cos} = \cos\!\left(\frac{2\pi k t}{m}\right),
$$

for harmonics $k = 1, \dots, K$. Adding several periods means simply adding more pairs.

The advantages are substantial. The number of parameters is two per harmonic, so $K$ harmonics cost $K$ sine and $K$ cosine columns per seasonality regardless of period length, so an annual cycle in daily data costs perhaps six parameters rather than 365. Non-integer periods are handled naturally, since $m$ appears only inside a division. And the terms are ordinary regressors, usable in any regression framework — including gradient boosting and linear models, not just time series packages.

```python
def fourier_terms(t, period, K):
    """2K columns of sin/cos harmonics for one seasonal period."""
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=idx)

X = pd.concat([fourier_terms(t, 7, 3),          # weekly, 3 harmonics
               fourier_terms(t, 365.25, 4)],    # annual, 4 harmonics
              axis=1)
X["trend"] = t

import numpy as np
beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X.values]),
                           series.values, rcond=None)
fitted = np.column_stack([np.ones(n), X.values]) @ beta
r2 = 1 - ((series.values - fitted) ** 2).sum() / ((series.values - series.mean()) ** 2).sum()
print(f"columns used : {X.shape[1]}  (vs 365 for an index-based annual season)")
print(f"R^2          : {r2:.4f}")
```

Choosing $K$ is the one real decision. More harmonics allow a more intricate seasonal shape and risk overfitting; the usual approach is to select by AIC or cross-validation. A smooth annual cycle rarely needs more than four or five harmonics, while a spiky weekly pattern may need three.

## TBATS

TBATS handles multiple seasonalities within a state space framework, combining a Box-Cox transformation, ARMA errors, and trigonometric seasonal terms. It selects the number of harmonics automatically and produces proper prediction intervals from the state space structure.

Its cost is computation. TBATS searches over a large model space and is slow on long series, which makes it awkward for forecasting thousands of series. It is worth reaching for when you have a handful of important series, complex seasonality, and need calibrated intervals more than you need speed.

## Choosing Between Them

| Need | Approach |
|---|---|
| Understand and visualise the components | MSTL |
| Seasonality as features in any model | Fourier terms |
| Automatic selection with prediction intervals | TBATS |
| Deseasonalise, then model the remainder | MSTL, then any method on the residual |

The last row is the pattern most often used in practice and worth naming: decompose, forecast the seasonally adjusted series with something simple, then add the seasonal components back. It keeps each part interpretable and lets you inspect where error is coming from.

## Practical Cautions

**Confirm the periods exist before assuming them.** A periodogram or the autocorrelation function will show which cycles are actually present. Fitting a weekly cycle to a process that has none adds parameters and noise.

**Holidays are not seasonality.** Easter moves, and no fixed period captures it. Treat major holidays as separate regressors rather than expecting the seasonal component to absorb them — this is one of the most common sources of large forecast errors in retail and energy.

**Watch for changing seasonal shape.** If the weekly pattern in 2024 differs materially from 2022, a fixed seasonal component will average the two and fit neither. MSTL's evolving components handle this; static Fourier terms do not, unless you interact them with time.

## References

- Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns. *arXiv:2107.13462*.
- De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *Journal of the American Statistical Association*, 106(496), 1513-1527.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
