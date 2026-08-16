---
permalink: '/time-series/forecast_combination_why_averaging_wins/'
title: 'Forecast Combination: Why Averaging Usually Wins'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Model Evaluation
- Statistical Modeling
author_profile: false
seo_title: 'Forecast Combination and Why It Works'
seo_description: 'Combining forecasts beats picking the best one, reliably enough that it is one of the most replicated results in forecasting.'
excerpt: >-
  Choosing the best model is the obvious strategy. Averaging several is
  usually better, and the reason is not that the average is smarter but that
  it is less wrong in a specific way.
summary: >-
  Why combining forecasts reduces error, the bias-variance argument behind it,
  why simple averages are hard to beat with estimated weights, when
  combination fails, and how to choose a pool of models that are worth
  combining.
keywords:
  - forecast combination
  - forecast averaging
  - ensemble forecasting
  - combination puzzle
  - model selection
classes: wide
date: '2026-07-19'
header:
  image: /assets/images/data_science_2.jpg
  og_image: /assets/images/data_science_2.jpg
  overlay_image: /assets/images/data_science_2.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.jpg
  twitter_image: /assets/images/data_science_2.jpg
---
You have five forecasts and have to ship one number. The obvious move is to identify the best model on validation data and use it. The better move, replicated across half a century of forecasting research, is usually to average them.

## Why an Average Beats the Average Member

The mechanism is not subtle once written down. Take two unbiased forecasts with error variances $\sigma_1^2$ and $\sigma_2^2$ and correlation $\rho$. The simple average has variance

$$
\operatorname{Var}\left(\frac{e_1 + e_2}{2}\right)
= \frac{\sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2}{4}.
$$

With equal variances this simplifies to $\frac{\sigma^2(1+\rho)}{2}$. Whenever $\rho < 1$ the combination has lower error variance than either component. At $\rho = 0$ it halves.

The combination is not smarter than its members. It is less exposed to any single model's particular way of being wrong, and errors that are not perfectly correlated partially cancel.

There is a second, more important effect. Model selection is itself estimated from finite data, so the "best" model on validation is partly best by luck. Choosing it commits fully to that luck; averaging hedges against it. The gain from combination is largest exactly when you are least certain which model is best — which is most of the time.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 4000
truth = rng.normal(0, 1, n)

# three forecasts with different biases, variances and error correlation
e1 = rng.normal(0.0, 1.0, n)
e2 = rng.normal(0.2, 1.3, n)
e3 = 0.4 * e1 + rng.normal(0.0, 0.9, n)      # correlated with the first
f = np.array([truth + e1, truth + e2, truth + e3])

rmse = lambda p: np.sqrt(((p - truth) ** 2).mean())
for i, p in enumerate(f, 1):
    print(f"model {i}      RMSE = {rmse(p):.4f}")
print(f"best single   RMSE = {min(rmse(p) for p in f):.4f}")
print(f"simple mean   RMSE = {rmse(f.mean(axis=0)):.4f}")
print(f"median        RMSE = {rmse(np.median(f, axis=0)):.4f}")
print("\nerror correlations:\n", np.corrcoef([e1, e2, e3]).round(2))
```

The average beats every individual model, including the best one — and it does so without knowing in advance which that was.

## The Forecast Combination Puzzle

Theory says optimal weights exist: weight inversely by error variance, accounting for covariance. In practice, simple equal weights routinely beat estimated optimal weights. This is the **forecast combination puzzle**, and it has been observed consistently since the 1970s.

The explanation is estimation error. Optimal weights require estimating variances and covariances from a limited sample, and those estimates are noisy. The noise in the weights can easily exceed the benefit of weighting correctly, particularly with many models or short histories. Equal weighting estimates nothing and therefore contributes no estimation error at all.

The practical implication is a default: **start with the simple average and require any weighting scheme to demonstrate improvement out of sample.** Approaches that shrink estimated weights toward equality often capture most of the theoretical benefit while avoiding most of the estimation cost.

```python
half = n // 2
errs_train = (f - truth)[:, :half]

# inverse-variance weights, estimated on the first half only
inv = 1.0 / errs_train.var(axis=1)
w = inv / inv.sum()

# evaluate both schemes on the held-out second half
rmse_oos = lambda p: np.sqrt(((p[half:] - truth[half:]) ** 2).mean())
weighted = (w[:, None] * f).sum(axis=0)
equal = f.mean(axis=0)

print("estimated weights :", w.round(3))
print(f"weighted RMSE (out of sample) = {rmse_oos(weighted):.4f}")
print(f"equal    RMSE (out of sample) = {rmse_oos(equal):.4f}")
```

The puzzle shows up in the output. The inverse-variance weights are sensible — roughly 0.38, 0.23, 0.39, correctly down-weighting the worst model — and they still lose to equal weighting out of sample, 0.7202 against 0.7177. The weights were estimated from 2,000 observations, which is generous by real standards, and the estimation error still ate the theoretical gain.

## Choosing What to Combine

Since the benefit depends on error correlation, the pool matters more than the weighting.

Combining five variants of the same ARIMA achieves little — their errors are nearly identical, so $\rho$ approaches 1 and the variance reduction vanishes. Combining an exponential smoothing model, a gradient boosting model on lag features, and a seasonal naive achieves much more, because each fails in a different way.

Two practical guidelines follow. Prefer diversity of *method* over quantity of models. And do not exclude a weaker model automatically: a model with higher standalone error but uncorrelated errors can still improve the combination. What matters is marginal contribution, not individual rank.

The median is worth considering alongside the mean. It is more robust when one model occasionally produces a wild forecast, which happens more often in automated pipelines than accuracy tables suggest — a single failed fit can otherwise drag the mean badly.

## When Combination Does Not Help

Three situations limit it.

**Correlated errors.** If all models share a blind spot — none of them knows about a promotion — averaging preserves the blind spot exactly. Combination reduces variance, not shared bias.

**A genuinely dominant model.** When one method is clearly correct and the others are misspecified, averaging drags the good forecast toward the bad ones. This is rarer than practitioners assume, and it should be demonstrated rather than believed.

**Interpretability requirements.** A combined forecast is hard to explain. "Why did the number change?" has a clean answer for one model and a diffuse one for an ensemble, which matters when a planner must defend the figure.

Note also that combining *point* forecasts is not the same as combining distributions. Averaging quantiles from several models does not generally produce a calibrated predictive distribution — the linear pool of densities is typically overdispersed. Probabilistic combination is a distinct problem with its own literature.

## What to Do

Fit a small, deliberately diverse set of models. Take the simple average, and treat that as the number to beat. Compare it against the best individual model and against any weighting scheme, all on the same out-of-sample periods. Adopt weights only when they win by a margin large enough to survive the next data update.

The M4 and M5 competitions both reinforced this: the top entries were combinations, not single models, and elaborate weighting was not what separated them.

## References

- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451-468.
- Clemen, R. T. (1989). Combining forecasts: a review and annotated bibliography. *International Journal of Forecasting*, 5(4), 559-583.
- Smith, J., & Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting* (Vol. 1). Elsevier.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.
