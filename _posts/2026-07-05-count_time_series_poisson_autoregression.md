---
permalink: '/time-series/count_time_series_poisson_autoregression/'
title: 'Modelling Count Time Series'
categories:
- Time Series
tags:
- Time Series
- Statistical Modeling
- Probability
- Predictive Maintenance
author_profile: false
seo_title: 'Modelling Count Time Series'
seo_description: 'Counts that depend on their own history break both Poisson regression and ARIMA. What INGARCH and related models do instead.'
excerpt: >-
  Daily incident counts are integers, non-negative, often small, and
  correlated with yesterday. ARIMA assumes none of that and Poisson regression
  assumes independence.
summary: >-
  Why continuous time series models are wrong for counts and independent count
  models are wrong for time series, how INGARCH makes the conditional mean
  depend on past counts and past means, and how to handle overdispersion and
  excess zeros.
keywords:
  - count time series
  - INGARCH
  - Poisson autoregression
  - overdispersion
  - integer-valued models
classes: wide
date: '2026-07-05'
header:
  image: /assets/images/data_science_4.jpg
  og_image: /assets/images/data_science_4.jpg
  overlay_image: /assets/images/data_science_4.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_4.jpg
  twitter_image: /assets/images/data_science_4.jpg
---
Daily counts of equipment failures, hospital admissions, or security incidents share a shape that standard time series methods handle badly. The values are integers, non-negative, frequently small, and correlated with yesterday's value.

ARIMA assumes a continuous, unbounded variable and will happily forecast −0.4 failures. Poisson regression respects the integer, non-negative structure and assumes observations are independent, which for a time series is exactly wrong.

## Why the Obvious Fixes Fail

The instinct is to transform. Taking $\log(y_t + 1)$ and fitting ARIMA is common and carries several problems: the constant is arbitrary and changes the answer, back-transforming introduces bias, and small counts remain visibly discrete after transformation.

The other instinct is to ignore the discreteness because the counts are large. That is defensible: when counts run in the hundreds, the normal approximation is reasonable and ARIMA works acceptably. The problem is specific to *small* counts — under roughly 20 per period, and acutely under 5, where the discreteness and the non-negativity both bind.

## INGARCH: Autoregression on the Conditional Mean

The cleanest framework keeps a proper count distribution and makes its *mean* depend on the past.

Let $y_t \mid \mathcal{F}_{t-1} \sim \text{Poisson}(\lambda_t)$, and model the conditional mean:

$$
\lambda_t = \omega + \sum_{i=1}^{p} \alpha_i y_{t-i} + \sum_{j=1}^{q} \beta_j \lambda_{t-j},
$$

with $\omega > 0$ and non-negative coefficients to keep $\lambda_t$ positive.

The structure mirrors GARCH, which is where the name comes from: past observations feed back through $\alpha$, and past conditional means through $\beta$, giving persistence without requiring many lags. Stationarity requires $\sum \alpha_i + \sum \beta_j < 1$, exactly as in GARCH.

Because the observation distribution is Poisson, forecasts are automatically non-negative integers with a proper predictive distribution — the thing ARIMA cannot provide for counts.

A log-linear variant models $\log \lambda_t$ instead, which removes the positivity constraints on coefficients and allows negative feedback, at the cost of a less direct interpretation.

```python
import numpy as np

def simulate_ingarch(n, omega=2.0, alpha=0.35, beta=0.4, seed=0):
    """Poisson INGARCH(1,1): counts whose conditional mean is autoregressive."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    lam = np.zeros(n)
    lam[0] = omega / (1 - alpha - beta)          # unconditional mean
    y[0] = rng.poisson(lam[0])
    for t in range(1, n):
        lam[t] = omega + alpha * y[t - 1] + beta * lam[t - 1]
        y[t] = rng.poisson(lam[t])
    return y, lam

y, lam = simulate_ingarch(2000)
print(f"mean count      : {y.mean():.2f}")
print(f"variance        : {y.var(ddof=1):.2f}")
print(f"dispersion index: {y.var(ddof=1) / y.mean():.2f}")
print(f"lag-1 autocorr  : {np.corrcoef(y[:-1], y[1:])[0, 1]:.3f}")
print(f"zeros           : {(y == 0).mean():.1%}")
```

Two properties of the output are worth noting. The lag-1 autocorrelation is substantial, which is the dependence a plain Poisson model would deny. And the dispersion index exceeds 1 even though the observation distribution is exactly Poisson — the autoregression in $\lambda_t$ induces overdispersion in the marginal counts by itself.

That second point matters practically: observing variance greater than the mean does not prove you need a negative binomial. Serial dependence produces the same symptom.

## Overdispersion and Excess Zeros

When dispersion remains after accounting for the dynamics, the observation distribution needs widening. Replacing Poisson with a **negative binomial** adds a parameter so that

$$
\operatorname{Var}(y_t \mid \mathcal{F}_{t-1}) = \lambda_t + \frac{\lambda_t^2}{\theta},
$$

recovering Poisson as $\theta \to \infty$. Ignoring genuine overdispersion does not usually bias the fitted mean much, but it makes standard errors far too small and prediction intervals far too narrow.

Excess zeros are a separate diagnosis. If more periods are zero than even an overdispersed model predicts, two structures are candidates. A **hurdle model** splits the problem into "is the count zero" and "given it is positive, how large" — appropriate when zeros arise from a distinct mechanism. A **zero-inflated model** mixes a point mass at zero with a count distribution, appropriate when some periods are structurally incapable of producing events.

The distinction is substantive rather than technical. A machine that is switched off produces structural zeros; a machine that is running and simply did not fail produces sampling zeros. Only the first justifies zero inflation.

## Alternatives Worth Knowing

**INAR models** take a different route, defining autoregression through *thinning* rather than through the mean: each of yesterday's events survives to today with probability $\alpha$, plus new arrivals. This has a natural interpretation for populations and queues, where "survival" is a real mechanism, and it keeps everything integer-valued throughout.

**GLMs with lagged covariates** are the pragmatic option: fit a Poisson or negative binomial GLM including lagged counts and rolling means as predictors. This is not a fully specified process model and it is easy to implement, easy to extend with exogenous variables, and often adequate.

**State space models for counts** put a latent continuous process behind a count observation equation, which handles missing data and multiple sources of variation cleanly at the cost of more computation.

## Practical Guidance

Check the dispersion index first, but interpret it carefully — as above, dependence inflates it independently of the observation distribution.

Plot the autocorrelation of the counts. If it is negligible, a plain GLM without dynamics is sufficient and the extra machinery is unnecessary.

Choose the evaluation metric with the discreteness in mind. RMSE on small counts is dominated by whether the model predicts 0 or 1, and the honest evaluation is probabilistic: use the ranked probability score or the log score on the predictive distribution, which is available here precisely because the model specifies one.

Finally, treat the exposure. If the periods differ in length or population at risk, include an offset — modelling counts without adjusting for exposure attributes to dynamics what is really variation in opportunity.

## References

- Ferland, R., Latour, A., & Oraichi, D. (2006). Integer-valued GARCH process. *Journal of Time Series Analysis*, 27(6), 923-942.
- Fokianos, K., Rahbek, A., & Tjøstheim, D. (2009). Poisson autoregression. *Journal of the American Statistical Association*, 104(488), 1430-1439.
- Liboschik, T., Fokianos, K., & Fried, R. (2017). tscount: an R package for analysis of count time series following generalized linear models. *Journal of Statistical Software*, 82(5), 1-51.
- Weiß, C. H. (2018). *An Introduction to Discrete-Valued Time Series*. Wiley.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
