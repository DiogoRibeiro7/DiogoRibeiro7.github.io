---
permalink: '/time-series/probabilistic_forecasting_quantiles_pinball_loss/'
title: 'Probabilistic Forecasting: Beyond the Point Estimate'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Confidence Intervals
- Model Evaluation
author_profile: false
seo_title: 'Probabilistic Forecasting and Pinball Loss'
seo_description: 'A single predicted number hides what you need for decisions. Quantile forecasts, pinball loss, and honest prediction intervals.'
excerpt: >-
  A point forecast answers the wrong question. Most decisions depend on how
  bad things could plausibly get, which is a statement about the whole
  distribution.
summary: >-
  Why point forecasts are insufficient for decision-making, how quantile
  regression and pinball loss produce and score full predictive distributions,
  why prediction intervals from most models are too narrow, and how conformal
  methods provide coverage guarantees.
keywords:
  - probabilistic forecasting
  - quantile regression
  - pinball loss
  - prediction intervals
  - conformal prediction
classes: wide
date: '2026-08-11'
why_this_exists: >-
  Many decisions depend on the upper or lower tail of future outcomes, while a
  point forecast hides the distribution that drives the decision.
evidence: >-
  Uses service-level, inventory, quantile, pinball-loss, and conformal
  forecasting concepts to connect scoring with decisions.
methodology: >-
  Starts from the decision cost, maps it to the needed quantile or interval,
  and then evaluates forecasts with a matching loss.
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_5.jpg
  og_image: /assets/images/data_science_5.jpg
  overlay_image: /assets/images/data_science_5.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_5.jpg
  twitter_image: /assets/images/data_science_5.jpg
---
"We expect 1,200 units next week." That sentence is almost never what the decision needs. Stock is set to cover demand with some service level, staffing is planned against a plausible peak, and capacity is sized for a bad day rather than an average one. Every one of those is a question about the *distribution*, and a point forecast has already thrown it away.

## What a Point Forecast Optimises

The choice of loss function silently decides what your single number means. Minimising squared error targets the conditional **mean**; minimising absolute error targets the **median**. On skewed data these differ substantially, and neither is what an inventory decision wants.

If understocking costs more than overstocking, the right quantity is not the middle of the distribution at all — it is an upper quantile chosen from the cost ratio. The classic newsvendor result makes this precise: the optimal order quantity is the $\frac{c_u}{c_u + c_o}$ quantile, where $c_u$ and $c_o$ are the unit costs of under- and over-supply. With shortage costing three times excess, you should be planning to the 75th percentile, not the mean.

A point forecast cannot express that. A quantile forecast can.

## Quantile Regression and Pinball Loss

To predict the $\tau$-th quantile directly, minimise the **pinball loss**:

$$
L_\tau(y, \hat{y}) =
\begin{cases}
\tau\,(y - \hat{y}) & \text{if } y \ge \hat{y} \\
(1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

The asymmetry is the whole mechanism. For $\tau = 0.9$, being below the actual is penalised nine times as heavily as being above it, so the minimiser sits high enough that only 10% of observations exceed it — which is the definition of the 90th percentile.

At $\tau = 0.5$ the two arms are equal and pinball loss reduces to half the absolute error, recovering the median.

```python
import numpy as np

def pinball(y, yhat, tau):
    d = y - yhat
    return np.mean(np.maximum(tau * d, (tau - 1) * d))

rng = np.random.default_rng(0)
# right-skewed demand: mean and median differ materially
y = rng.lognormal(mean=3.0, sigma=0.6, size=20000)

print(f"mean   : {y.mean():.2f}")
print(f"median : {np.median(y):.2f}")
for tau in (0.5, 0.9):
    q = np.quantile(y, tau)
    # the true quantile should minimise pinball loss at that tau
    cands = {"median": np.median(y), "mean": y.mean(), f"q{tau}": q}
    scores = {k: pinball(y, v, tau) for k, v in cands.items()}
    best = min(scores, key=scores.get)
    print(f"tau={tau}: q={q:6.2f}  lowest pinball from '{best}'  "
          + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))
```

The check that matters is the last line: at each $\tau$ the lowest pinball loss comes from the corresponding quantile, not from the mean or median. That is what makes pinball loss a *proper* scoring rule for quantiles — it is minimised by the truth, so a model trained on it has no incentive to report anything else.

## Producing a Full Distribution

Several routes lead to a predictive distribution.

**Fit one model per quantile.** Gradient boosting libraries accept a quantile objective directly, so training at $\tau \in \lbrace 0.1, 0.5, 0.9\rbrace$ gives three models describing the distribution's shape. Simple and flexible; the drawback is that independently fitted quantiles can **cross** — the predicted 90th percentile falling below the 50th — which is incoherent and needs sorting or a monotonicity constraint.

**Use a parametric model's own intervals.** ARIMA and exponential smoothing in state space form produce prediction intervals from their error variance. These are principled but depend on the model's distributional assumption, usually normality, which understates the tails of most real demand.

**Simulate.** Bootstrap the residuals and propagate them forward through the model many times, then take empirical quantiles of the simulated paths. This handles non-linear models and accumulating uncertainty naturally, and makes no normality assumption.

**Predict the parameters of a distribution.** Fit a model whose outputs are, say, the mean and dispersion of a negative binomial. This gives a coherent distribution by construction and suits count data well.

## Why Prediction Intervals Are Usually Too Narrow

An interval labelled 95% that contains the truth 80% of the time is worse than no interval, because it invites decisions taken with false confidence.

Intervals from fitted models are routinely too narrow, for a consistent reason: they account for the *irreducible noise* around the model while ignoring uncertainty in the model itself. Parameters were estimated from finite data, and the model form may be wrong. Neither source is in the standard formula.

For multi-step forecasts a second problem appears. Interval width should grow with the horizon, because errors compound — an interval for 12 steps ahead that is the same width as one step ahead is not credible. Methods that forecast each horizon independently often fail this.

**Always check empirical coverage.** Count how often the actual falls inside the nominal interval on held-out data. If a 90% interval covers 70% of the time, the interval is wrong, no matter how it was derived.

```python
def coverage(actual, lower, upper):
    inside = (actual >= lower) & (actual <= upper)
    return inside.mean()

# residual bootstrap: simple, and free of the normality assumption
train, test = y[:15000], y[15000:]
resid = train - np.median(train)
sims = np.median(train) + rng.choice(resid, size=(4000, len(test)), replace=True)
lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
print(f"nominal 90% interval, empirical coverage: {coverage(test, lo, hi):.3f}")
```

## Conformal Prediction

Conformal methods provide coverage guarantees that hold **without** distributional assumptions, given only that the data is exchangeable.

The split-conformal recipe is short. Fit on a training set, compute absolute residuals on a held-out calibration set, take the $(1-\alpha)$ quantile of those residuals, and form intervals as the point prediction plus and minus that value. The resulting interval has at least $1-\alpha$ coverage in finite samples, regardless of the model.

The caveat for time series is real: exchangeability fails when observations are dependent, which is the defining property of a time series. Adaptations exist — most notably methods that update the calibration quantile online as coverage drifts — and are the right choice when guarantees matter.

## Evaluating a Distributional Forecast

Point metrics do not apply. The standard tools instead are:

**Pinball loss averaged over quantiles**, which is the workhorse and was the metric of the M5 uncertainty competition.

**Continuous Ranked Probability Score (CRPS)**, which compares the full predicted CDF against the observation and reduces to absolute error when the forecast is a point. It rewards both calibration and sharpness together.

**Coverage and width reported as a pair.** Coverage alone is trivially maximised by an infinitely wide interval; sharpness alone by an infinitely narrow one. The goal is the narrowest interval that achieves its stated coverage.

## What to Do

Decide the quantiles from the decision, not by convention — if shortage costs three times excess, forecast the 75th percentile and say so. Report intervals alongside every point forecast, and check their empirical coverage on held-out data rather than trusting the nominal label. And when the model's own intervals prove too narrow, which is the common case, calibrate them against observed residuals rather than widening by feel.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). The M5 uncertainty competition: results, findings and conclusions. *International Journal of Forecasting*, 38(4), 1365-1385.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
