---
permalink: '/time-series/feature_engineering_for_time_series/'
title: 'Feature Engineering for Time Series Without Leaking the Future'
categories:
- Time Series
tags:
- Feature Engineering
- Time Series
- Machine Learning
- Data Quality
author_profile: false
seo_title: 'Feature Engineering for Time Series'
seo_description: 'Lags, rolling statistics and calendar effects, and the leakage that makes tabular time series models look better than they are.'
excerpt: >-
  Turning a time series into a tabular problem unlocks powerful models and
  introduces a specific failure: features that quietly contain information
  from the future.
summary: >-
  How to build lag, rolling-window and calendar features for time series, why
  the shift before the roll is what prevents leakage, how forecast horizon
  determines which lags are legitimate, and why a suspiciously good validation
  score is usually a leak rather than a discovery.
keywords:
  - time series features
  - lag features
  - rolling window features
  - data leakage
  - calendar effects
classes: wide
date: '2026-08-08'
why_this_exists: >-
  Many strong tabular models fail in production because their time-series
  features accidentally include information from the future.
evidence: >-
  Uses lag, rolling-window, calendar, and forecast-horizon examples to expose
  the exact leakage points.
methodology: >-
  Treats each feature as an availability question: would this value have been
  known at the time the forecast was made?
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_2.avif
  og_image: /assets/images/data_science_2.avif
  overlay_image: /assets/images/data_science_2.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.avif
  twitter_image: /assets/images/data_science_2.avif
---
Gradient boosting and neural networks are not time series models. They accept a table of rows and columns with no notion that one row came before another. Turning a series into that table is where most of the work — and most of the mistakes — happen.

The mistakes share one shape: a feature that contains information which would not have been available at the moment the forecast was made. The result is a validation score that looks excellent and a production model that does not work.

## Lag Features

The basic move is to use past values as columns. For a target $y_t$, the lag-$k$ feature is $y_{t-k}$.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=400, freq="D")
y = (100 + np.arange(400) * 0.05
     + 8 * np.sin(2 * np.pi * np.arange(400) / 7)      # weekly cycle
     + rng.normal(0, 2, 400))
df = pd.DataFrame({"y": y}, index=idx)

for k in (1, 2, 7, 14):
    df[f"lag_{k}"] = df["y"].shift(k)
```

Which lags are legitimate depends entirely on the **forecast horizon**. If you predict one step ahead, `lag_1` is available. If you predict seven days ahead, it is not — at prediction time you do not yet know yesterday's value relative to the target date. A model forecasting seven days out may only use lags of 7 or more.

This is the most common leak in practice, and it is invisible in the metrics: the model trains happily, validates beautifully, and fails on deployment because the feature it relied on does not exist yet.

Choose lags from the data rather than by habit. The autocorrelation and partial autocorrelation functions show which lags carry signal, and for seasonal data the seasonal lag $m$, and its multiples, are usually the strongest.

## Rolling Statistics, and the Shift That Matters

Rolling means, standard deviations and extremes summarise recent behaviour. The danger is that pandas' `rolling` window **includes the current row**.

```python
# WRONG: window includes today's value, which is the thing being predicted
df["roll_bad"] = df["y"].rolling(7).mean()

# RIGHT: shift first, so the window ends yesterday
df["roll_7"] = df["y"].shift(1).rolling(7).mean()
df["roll_std_7"] = df["y"].shift(1).rolling(7).std()
df["roll_max_28"] = df["y"].shift(1).rolling(28).max()

check = df[["y", "roll_bad", "roll_7"]].dropna()
print("corr(y, leaky rolling mean) :", check["y"].corr(check["roll_bad"]).round(3))
print("corr(y, correct rolling mean):", check["y"].corr(check["roll_7"]).round(3))
```

The leaky version correlates more strongly with the target — 0.682 against 0.670 — because it literally contains one-seventh of it. The gap is small here precisely because this series is dominated by a clean weekly cycle that both versions capture. On a noisier series, where the current value carries a larger share of the window's variance, the inflation is far bigger.

That is what makes this mistake dangerous rather than obvious. The leak does not announce itself with an implausible correlation; it just makes the feature look slightly better, which is what you were hoping to see.

The rule is simple enough to apply mechanically: **shift, then roll.** Any window that touches time $t$ when predicting time $t$ is a leak.

Expanding windows have the same requirement, and they carry an extra caution: an expanding mean computed over the full dataset before splitting includes test-period values in every training row.

## Calendar and Cyclical Features

Date parts capture regular human patterns — day of week, month, quarter, week of year, and flags for weekends, holidays and month boundaries.

Encoding them naively creates a problem. Month 12 and month 1 are adjacent in reality but maximally distant as integers, and a model reading month as a number will treat December and January as opposites. Cyclical encoding fixes this by placing the period on a circle:

$$
x_{\sin} = \sin\!\left(\frac{2\pi k}{K}\right), \qquad
x_{\cos} = \cos\!\left(\frac{2\pi k}{K}\right),
$$

for value $k$ within a cycle of length $K$. Both terms are needed, since either alone maps two different points to the same value.

```python
df["dow"] = df.index.dayofweek
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
```

Tree models are a partial exception: they can split a raw integer month into arbitrary groups and so cope without the transform. Linear models and neural networks generally need it.

Holidays deserve separate treatment, because their effect usually extends beyond the day itself — retail demand shifts for a week around a public holiday, and the days *before* often matter more than the day. Distance-to-holiday is frequently a stronger feature than a binary flag.

## Detecting Leakage

Leakage announces itself if you know the symptoms. Validation error far below what the problem plausibly allows. A feature with implausibly high importance. Performance that collapses when the model reaches production. Accuracy that does not degrade as the forecast horizon grows — a strong tell, since genuine forecasting always gets harder further out.

The structural defence is to validate the way the model will be used. Random k-fold on a time series trains on future data to predict the past, which leaks regardless of how careful the features are. Use rolling-origin or expanding-window splits, where every training set ends before its test set begins.

```python
from sklearn.model_selection import TimeSeriesSplit

feats = ["lag_7", "lag_14", "roll_7", "roll_std_7", "dow_sin", "dow_cos"]
data = df.dropna()
X, target = data[feats], data["y"]

for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=4).split(X), 1):
    print(f"fold {i}: train {tr[0]:>3}-{tr[-1]:>3}  test {te[0]:>3}-{te[-1]:>3}")
```

Note that every training fold ends before its test fold starts, and the training set grows rather than sliding — the arrangement that mirrors deployment.

## Feature Choices That Actually Help

Beyond lags and calendar terms, a few additions repeatedly earn their place.

**Differences and ratios** relative to a lag ($y_{t-1} - y_{t-8}$, or the ratio to last week's mean) express change rather than level, and often generalise better when the series drifts.

**Time since an event** — last outage, last promotion, last maintenance — captures effects that decay, which no fixed-window statistic represents well.

**Exogenous variables must respect the same discipline.** Weather is a good predictor of energy demand, but at forecast time you have a weather *forecast*, not the actual. Training on actuals and deploying on forecasts is a subtle leak that silently degrades production accuracy.

Resist adding features indiscriminately. Each lag consumes rows at the start of the series through the resulting NaNs, and on a short series a 28-day rolling feature can remove a meaningful fraction of the training data.

## The Test to Apply

For every feature, ask one question: *at the moment this forecast is made, would this value be known?* If the answer is no, or only sometimes, the feature is a leak.

That question catches nearly everything. It is worth asking explicitly for each column rather than assuming, because leaks do not produce errors — they produce results that look better than they should, which is the hardest kind of bug to notice.

## References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21.
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press.
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
