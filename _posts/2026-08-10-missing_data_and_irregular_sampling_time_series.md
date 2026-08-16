---
permalink: '/time-series/missing_data_and_irregular_sampling_time_series/'
title: 'Missing Data and Irregular Sampling in Time Series'
categories:
- Time Series
tags:
- Missing Data
- Time Series
- Data Quality
- Python
author_profile: false
seo_title: 'Missing Data in Time Series'
seo_description: 'Gaps in a series are not the same as gaps in a table. Why interpolation choices change your conclusions.'
excerpt: >-
  A missing row in a table is a nuisance. A missing interval in a time series
  changes the meaning of every lag, window and seasonal index computed from
  it.
summary: >-
  How to handle gaps and irregular timestamps in time series: why the
  mechanism behind missingness matters more than the imputation method, what
  forward-fill and interpolation actually assume, how resampling silently
  invents data, and when a model built for irregular observations is the
  better answer.
keywords:
  - missing time series data
  - interpolation
  - resampling
  - irregular sampling
  - forward fill
classes: wide
date: '2026-08-10'
why_this_exists: >-
  Missing time-series intervals change the meaning of lags, windows, and
  seasonal features, so table-style imputation advice is not enough.
evidence: >-
  Uses sensor-style missingness, resampling, forward-fill, and interpolation
  examples to show what each repair assumes.
methodology: >-
  Separates missing values from irregular timestamps and evaluates each fix by
  the data-generating mechanism it implies.
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_8.avif
  og_image: /assets/images/data_science_8.avif
  overlay_image: /assets/images/data_science_8.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.avif
  twitter_image: /assets/images/data_science_8.avif
---
A missing cell in a table is a nuisance. A missing interval in a time series is something else: every lag, rolling window and seasonal index computed downstream silently assumes the observations are evenly spaced, and a gap breaks that assumption without raising an error.

## Two Different Problems

**Missing values** are gaps in an otherwise regular series — the sensor should have reported at 14:00 and did not. The timestamps are known, the values are absent.

**Irregular sampling** means observations arrive at arbitrary times. Clinical measurements, transaction logs and event streams are irregular by nature; nothing is "missing" because there was never a schedule.

The distinction determines the fix. Missing values in a regular series can be imputed. An irregular series is not incomplete, and forcing it onto a grid invents data that was never observed.

## Why the Mechanism Matters More Than the Method

Before choosing an imputation, ask why the value is absent.

**Missing at random** — a network hiccup dropped a reading — is benign, and most methods work.

**Missing because of the value itself** is not. If a sensor saturates and reports nothing above its range, every missing value is a high value. Imputing the mean does not just add noise; it systematically erases the extremes, which are usually the observations that mattered. In predictive maintenance this failure mode destroys exactly the signal the system exists to detect.

**Missing because nothing happened** is a third case, and the most commonly mishandled. In a sales series, a date with no row often means zero sales, not unknown sales. Imputing anything other than zero inflates the level and corrupts every downstream statistic.

The practical rule: add an indicator column recording that the value was imputed. If the missingness carries information, the model can use it; if not, it costs nothing.

## What the Common Methods Assume

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=200, freq="D")
t = np.arange(200)
truth = 50 + 10 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1, 200)

s = pd.Series(truth, index=idx)
gap = slice(80, 95)                       # a 15-day outage
observed = s.copy()
observed.iloc[gap] = np.nan

methods = {
    "forward fill": observed.ffill(),
    "linear interp": observed.interpolate("linear"),
    "time interp": observed.interpolate("time"),
    "seasonal (lag 30)": observed.fillna(observed.shift(30)),
}
for name, filled in methods.items():
    err = np.abs(filled.iloc[gap] - s.iloc[gap]).mean()
    print(f"{name:20} mean abs error over the gap: {err:5.2f}")
```

Each carries an assumption that is easy to forget.

**Forward fill** assumes the series holds its last value. Reasonable for a step-like quantity such as a thermostat setting, wrong for anything cyclical — across a long gap it produces a flat line that destroys the shape.

**Linear interpolation** assumes the series moves in a straight line between the endpoints. Fine for short gaps in a smooth series, and it will happily interpolate straight through an entire seasonal cycle, flattening it.

**Seasonal fill** uses the value one period earlier and is usually the strongest choice for seasonal data, because it preserves the shape the other methods destroy.

The general pattern: interpolation is safe when the gap is short relative to the dynamics and dangerous when it is not. A two-hour gap in hourly data is different from a two-month gap.

## Resampling Silently Invents Data

Converting an irregular series to a regular grid feels like cleaning. It is closer to modelling, and it makes claims.

Upsampling — moving to a finer grid than the observations — generates values that were never measured. Any variance estimate, autocorrelation or model fitted afterwards treats those inventions as evidence, and confidence intervals computed from them are too narrow because the effective sample size is smaller than the row count suggests.

Downsampling is safer but not free. Aggregating to a coarser grid requires choosing a statistic — mean, last, max — and each answers a different question. A daily *maximum* temperature series and a daily *mean* temperature series have different dynamics and different seasonality.

```python
# irregular observations, then a regular grid
times = np.sort(rng.choice(np.arange(500), size=120, replace=False))
vals = 20 + 5 * np.sin(2 * np.pi * times / 50) + rng.normal(0, 0.5, times.size)
irr = pd.Series(vals, index=pd.Timestamp("2024-01-01") + pd.to_timedelta(times, "D"))

grid = irr.resample("D").mean()
print(f"observations : {irr.size}")
print(f"grid rows    : {grid.size}  ({grid.isna().mean():.0%} of them empty)")
print(f"after interp : {grid.interpolate().size} rows, all filled")
```

The row count nearly quadruples and the information content does not change at all. That gap between apparent and actual sample size is what makes downstream statistics overconfident.

## Methods Built for Gaps

Some approaches handle missingness without imputing anything.

**State space models with a Kalman filter** treat missing observations natively: the prediction step runs, the update step is skipped, and uncertainty grows across the gap exactly as it should. This is the most principled option for a regular series with holes, and it propagates the uncertainty rather than hiding it.

**Gaussian processes** are defined on continuous time and never require a grid at all, which makes them a natural fit for genuinely irregular data — at the cost of scaling poorly to long series.

**Multiple imputation** creates several completed datasets, analyses each, and pools the results so that the extra uncertainty from imputing appears in the final standard errors. Single imputation always understates uncertainty; multiple imputation is the standard correction.

## Practical Guidance

Look at where the gaps are before deciding anything. Missingness concentrated in one period is a different problem from missingness scattered evenly, and the former often indicates an outage whose cause matters more than its imputation.

Set a limit on how much you will bridge — `interpolate(limit=3)` fills short gaps and leaves long ones visible, which is usually right. A gap longer than the seasonal period should generally not be imputed at all; treat the series as two segments instead.

Never impute across a train/test boundary. An interpolation that uses observations from both sides of the split leaks test information into training, and it does so invisibly.

Finally, report the missingness. A forecast built on a series that was 30% imputed carries uncertainty that no interval derived from the completed series will show.

## References

- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.
- Moritz, S., & Bartz-Beielstein, T. (2017). imputeTS: time series missing value imputation in R. *The R Journal*, 9(1), 207-218.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Shukla, S. N., & Marlin, B. M. (2021). A survey on principles, models and methods for learning from irregularly sampled time series. *arXiv:2012.00168*.
