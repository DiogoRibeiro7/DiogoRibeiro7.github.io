---
permalink: '/time-series/forecasting_baselines_that_are_hard_to_beat/'
title: 'Forecasting Baselines That Are Hard to Beat'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Model Evaluation
- Statistics
author_profile: false
seo_title: 'Forecasting Baselines That Are Hard to Beat'
seo_description: 'The naive and seasonal naive forecasts are not straw men. Why a baseline is the only thing that makes an accuracy number mean anything.'
excerpt: >-
  An RMSE of 4.2 means nothing on its own. Without a baseline you cannot tell
  whether a model is skilful or merely arithmetic.
summary: >-
  Why every forecasting project should start with naive, seasonal naive, drift
  and mean baselines: what each assumes, why the seasonal naive is genuinely
  difficult to beat on strongly seasonal data, and how scaled error measures
  use a baseline to make accuracy comparable across series.
keywords:
  - naive forecast
  - seasonal naive
  - forecast baseline
  - MASE
  - forecast skill
classes: wide
date: '2026-08-06'
header:
  image: /assets/images/data_science_7.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_7.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_7.jpg
  twitter_image: /assets/images/data_science_7.jpg
---
An RMSE of 4.2 tells you nothing. Not "a little", not "roughly how good the model is" — nothing at all, until you know what an unskilled forecast would have scored on the same data.

This is the single most common gap in forecasting work. A model is built, an error metric is reported, the number sounds plausible, and nobody establishes whether arithmetic that takes one line to write would have done as well.

## The Four Baselines

Each baseline encodes a different minimal assumption about how the world works.

**Naive.** The forecast for every future period is the last observed value:

$$
\hat{y}_{T+h} = y_T .
$$

This assumes the series is a random walk — that the best guess about tomorrow is today. For financial prices this is not a straw man but close to the truth, and beating it consistently is difficult enough that failing to do so is the expected outcome.

**Seasonal naive.** The forecast is the value from the same point in the previous seasonal cycle:

$$
\hat{y}_{T+h} = y_{T+h-m},
$$

for seasonal period $m$. Last January predicts this January. On strongly seasonal data this baseline is genuinely hard to beat, and many elaborate models fail to.

**Drift.** The naive forecast extended along the average historical slope:

$$
\hat{y}_{T+h} = y_T + h \cdot \frac{y_T - y_1}{T - 1},
$$

equivalent to drawing a line through the first and last observations. This captures trend without fitting anything.

**Mean.** The forecast is the average of all history, $\hat{y}_{T+h} = \bar{y}$. Appropriate when the series is stationary with no trend or season, and a useful contrast: if the mean wins, your series has less structure than you assumed.

```python
import numpy as np

def naive(y, h):
    return np.repeat(y[-1], h)

def seasonal_naive(y, h, m):
    # take the last m observations and tile them forward
    last_season = y[-m:]
    return np.array([last_season[i % m] for i in range(h)])

def drift(y, h):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)

def mean_forecast(y, h):
    return np.repeat(y.mean(), h)

# monthly series: upward trend plus a strong annual cycle
rng = np.random.default_rng(0)
t = np.arange(72)
y = 100 + 0.6 * t + 12 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, t.size)
train, test, h, m = y[:60], y[60:], 12, 12

for name, fc in [("naive", naive(train, h)),
                 ("seasonal naive", seasonal_naive(train, h, m)),
                 ("drift", drift(train, h)),
                 ("mean", mean_forecast(train, h))]:
    mae = np.abs(test - fc).mean()
    rmse = np.sqrt(((test - fc) ** 2).mean())
    print(f"{name:16} MAE={mae:7.2f}  RMSE={rmse:7.2f}")
```

On this series — trend plus a strong annual cycle — the seasonal naive wins clearly with MAE 7.61, ahead of drift at 11.00, naive at 13.10 and the mean at 21.81. The ordering is itself diagnostic. Seasonality carries more of the variance than trend does, so capturing the cycle buys more than capturing the slope; and the mean finishing last confirms the series has real structure rather than fluctuating around a fixed level.

The bar is now set. Any model worth deploying has to beat 7.61, and a model scoring 9 is worse than one line of code.

## Why the Seasonal Naive Is Hard to Beat

The seasonal naive uses exactly one observation per forecast, and it still wins often enough to embarrass more elaborate approaches. Three reasons explain this.

It has no parameters, so it cannot overfit. Every estimated parameter is an opportunity to fit noise, and on short series that cost frequently exceeds the benefit.

It adapts immediately to level shifts. A model fitted on three years of history averages across all of them; the seasonal naive uses only last year, so a permanent change in level propagates through after one cycle rather than being diluted.

And when seasonality dominates the variance — as it does for electricity, retail and tourism — most of what there is to predict is already captured by "the same period last year".

## Scaled Errors: Baselines Inside the Metric

Baselines are not only for comparison; the better accuracy measures build one in.

**MASE** divides the forecast error by the in-sample error of a naive forecast:

$$
\text{MASE} = \frac{\frac{1}{h}\sum_{t}|y_t - \hat{y}_t|}
{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t - y_{t-m}|} .
$$

The interpretation is immediate. MASE below 1 means the model beats the naive baseline; above 1 means it does not. It is scale-free, so it can be averaged across series measured in different units — the reason it is standard in forecasting competitions.

MASE also avoids the failure that makes MAPE unusable in practice. Percentage errors explode as actual values approach zero and are undefined at zero, and MAPE penalises over-forecasting and under-forecasting asymmetrically, which quietly biases model selection toward forecasts that are too low.

```python
def mase(y_train, y_test, forecast, m=1):
    naive_mae = np.abs(y_train[m:] - y_train[:-m]).mean()
    return np.abs(y_test - forecast).mean() / naive_mae

print(f"seasonal naive MASE : {mase(train, test, seasonal_naive(train, h, m), m):.3f}")
print(f"drift MASE          : {mase(train, test, drift(train, h), m):.3f}")
```

## Forecast Skill

The same idea expressed as a proportion is the **skill score**:

$$
\text{Skill} = 1 - \frac{\text{Error}_{\text{model}}}{\text{Error}_{\text{baseline}}} .
$$

A skill of 0.3 means the model reduced error by 30% against the baseline; zero means no improvement; negative means the baseline won. Reporting skill rather than raw error makes results comparable across series and forces the baseline to be named, which is the discipline that matters.

## How This Goes Wrong

Three mistakes recur.

**Choosing a baseline that is too weak.** Comparing a seasonal model against the overall mean on strongly seasonal data guarantees a flattering result that means nothing. The baseline should be the strongest simple method plausible for the data.

**Comparing on different splits.** The baseline must be evaluated on exactly the same test periods, with exactly the same forecast horizon, as the model. Comparing a one-step-ahead baseline against a twelve-step-ahead model is not a comparison.

**Ignoring cost.** A model that improves MASE from 1.00 to 0.97 while requiring a training pipeline, feature store and retraining schedule may not be worth having. The baseline runs in one line and never breaks, which has real operational value that accuracy tables do not show.

## The Habit Worth Forming

Compute the baselines first, before building anything. It takes minutes, it establishes what the data's inherent predictability is, and it occasionally ends the project early with the useful finding that the series is a random walk and no model will help.

When you report accuracy, always report the baseline beside it. A number without a reference point is not a result — and if the sophisticated model cannot beat "same as last year", that is worth knowing before it is deployed rather than after.

## References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.
- Armstrong, J. S. (2001). *Principles of Forecasting: A Handbook for Researchers and Practitioners*. Kluwer Academic.
