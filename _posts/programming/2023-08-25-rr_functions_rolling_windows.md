---
permalink: '/programming/rr_functions_rolling_windows/'
author_profile: false
categories:
- Programming
classes: wide
date: '2023-08-25'
excerpt: A practical guide to rolling computations in R with runner, including fixed-size and time-indexed windows, lags, custom evaluation points, grouped data, and leakage-safe feature construction.
header:
  image: /assets/images/Rolling-window.jpg
  og_image: /assets/images/Rolling-window.jpg
  overlay_image: /assets/images/Rolling-window.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/Rolling-window.jpg
  twitter_image: /assets/images/Rolling-window.jpg
keywords:
- runner R package
- Rolling windows R
- Time based windows R
- Sliding windows
- Rolling regression
- Irregular time series
- R
redirect_from:
- '/r programming/rr_functions_rolling_windows/'
seo_description: How to use the runner package in R for cumulative, sliding, lagged, and time-indexed rolling computations.
seo_title: Rolling Window Functions in R with runner
seo_type: article
summary: The runner package applies arbitrary R functions over cumulative, fixed-size, and index-based rolling windows. The key statistical choices are window width, alignment, index semantics, and information availability.
tags:
- Time Series
- Statistical Modeling
- R
title: Applying R Functions on Rolling Windows with runner
---

![Rolling image - Applying R Functions on Rolling Windows with runner](/assets/images/rolling_image.png){: width="850" height="395" loading="lazy"}
<div align="center"><em>Rolling Window</em></div>

Rolling computations look simple until observations are irregularly spaced, windows are defined in calendar time rather than row counts, or features must respect a forecast origin. The runner package is useful because it makes those choices explicit. Its main function accepts a window size k, a lag, an optional index idx, and optional evaluation points at, while allowing an arbitrary R function to operate on each window.

## Cumulative and fixed-size windows

If k is omitted, runner uses an expanding window. For a sequence x_1,...,x_n, a cumulative statistic at time t is

$$
S_t = f(x_1,\ldots,x_t).
$$

~~~r
library(runner)

x <- 1:10
runner(x, f = sum)
runner(x, f = mean)
~~~

When k is a positive integer and no time index is supplied, it represents a number of elements.

~~~r
runner(
  x = 1:15,
  k = 4,
  f = mean
)
~~~

A fixed-size window is appropriate when observations are equally spaced and the last k observations has a meaningful interpretation.

## Time-indexed windows

For irregular data, row counts and elapsed time are different quantities. A seven-row window is not necessarily a seven-day window. Supplying idx allows k and lag to be interpreted relative to the index, including date-based intervals.

~~~r
dates <- as.Date(c(
  "2026-01-01",
  "2026-01-02",
  "2026-01-05",
  "2026-01-09",
  "2026-01-10"
))

values <- c(10, 12, 8, 15, 14)

runner(
  x = values,
  idx = dates,
  k = "5 days",
  f = mean
)
~~~

This distinction matters in finance, healthcare, sensors, and operational data, where missing weekends, outages, and asynchronous observations make row-based windows potentially misleading.

## Lags and information availability

A rolling feature used for prediction should contain only information available at prediction time. Positive lag values move the window backward relative to the evaluation point.

~~~r
runner(
  x = values,
  idx = dates,
  k = "5 days",
  lag = "1 day",
  f = mean
)
~~~

The statistical meaning is more important than the syntax. If a target is observed at time t, a feature intended for prospective prediction should generally be constructed only from data available before t.

## Evaluating only at selected points

The at argument allows calculations only at selected index values. This is useful for forecast origins, reporting dates, or intervention times.

~~~r
runner(
  x = values,
  idx = dates,
  k = "7 days",
  at = as.Date(c("2026-01-05", "2026-01-10")),
  f = mean
)
~~~

## Rolling regressions

Because runner accepts an arbitrary function, each window can contain a model fit.

~~~r
set.seed(42)

df <- data.frame(
  date = seq.Date(as.Date("2026-01-01"), by = "day", length.out = 60),
  x = rnorm(60)
)
df$y <- 2 + 1.5 * df$x + rnorm(60, sd = 0.7)

rolling_slope <- runner(
  x = df,
  idx = df$date,
  k = "20 days",
  f = function(window) {
    fit <- lm(y ~ x, data = window)
    unname(coef(fit)[["x"]])
  }
)
~~~

A rolling coefficient is not automatically evidence that a structural parameter changed. Its variation also reflects finite-sample noise, leverage, changing predictor distributions, and model misspecification.

## Grouped data

For grouped analyses, the grouping operation should happen before the rolling calculation so that the values and index remain aligned within each group.

~~~r
library(dplyr)
library(runner)

result <- df_grouped |>
  group_by(group) |>
  mutate(
    rolling_mean = runner(
      x = value,
      idx = date,
      k = "14 days",
      f = mean
    )
  )
~~~

This is clearer than relying on implicit column lookup inside a nested model function.

## Incomplete windows

For some applications, shorter startup windows are acceptable. For others, a statistic is meaningful only after a complete window has accumulated. The na_pad argument can be used when incomplete windows should return missing values.

~~~r
runner(
  x = 1:10,
  k = 5,
  f = mean,
  na_pad = TRUE
)
~~~

A five-observation volatility statistic based on one observation is not merely less precise. It is a different statistic.

## Window choice is part of the model

Short windows adapt quickly but have high sampling variability. Long windows are more stable but can average across genuine regime changes. There is no universally optimal choice. The window should follow the physical process, sampling rate, decision horizon, and validation design.

When rolling features are used in machine learning, window parameters must be chosen inside the training procedure. Selecting them using future validation outcomes leaks information just as surely as tuning a model hyperparameter on the test set.

## Conclusion

The useful distinction is

$$
\text{window width}
\quad
\text{alignment}
\quad
\text{evaluation index}.
$$

The runner API makes these choices explicit. The difficult part is not calling the function; it is deciding what each window means scientifically and ensuring that the computation respects the information available at the time the model or decision will be used.

## References

- Kałędkowski, D. runner: Running Operations for Vectors. CRAN package documentation.
- Hyndman, R. J., & Athanasopoulos, G. Forecasting: Principles and Practice.
- Shumway, R. H., & Stoffer, D. S. Time Series Analysis and Its Applications.
