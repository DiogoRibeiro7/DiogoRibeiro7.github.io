---
author_profile: false
categories:
- Time Series
classes: wide
title: 'A Forecast Distribution Should Be Calibrated and Sharp'
excerpt: 'Probabilistic forecasts should be judged by both calibration and concentration. A distribution can be safe by being too wide and still be operationally useless.'
keywords:
- probabilistic forecasting
- calibration
- sharpness
- proper scoring rules
- CRPS
seo_title: 'A Forecast Distribution Should Be Calibrated and Sharp'
seo_description: 'A mathematical draft on probabilistic forecast calibration, sharpness, log score, CRPS, PIT diagnostics, coverage, and proper scoring rules.'
seo_type: article
summary: 'A planned article developing probabilistic forecast evaluation from calibration and sharpness, with proper scoring rules that reward useful distributions rather than point accuracy alone.'
tags:
- Probabilistic Forecasting
- Forecast Evaluation
- Calibration
- Proper Scoring Rules
why_this_exists: 'Point metrics such as RMSE cannot evaluate whether a forecast distribution represents uncertainty correctly. Coverage alone is also insufficient because arbitrarily wide intervals can cover almost everything.'
evidence: 'Synthetic calibrated and miscalibrated forecast distributions, PIT histograms, interval coverage and width, log scores and CRPS calculations.'
methodology: 'Construct controlled forecast distributions with identical point means but different uncertainty quality, then evaluate them using calibration diagnostics and strictly proper scoring rules.'
---

<!--
Development contract
Question: What makes a probabilistic forecast useful rather than merely conservative?
Claim: Good probabilistic forecasts combine calibration with sharpness, and proper scoring rules provide principled incentives for reporting the true predictive distribution.
Counterclaim: Calibration can be conditional on information sets and aggregation levels, so global calibration diagnostics can conceal local failures.
Evidence object: Three synthetic forecasters with the same mean predictions but different dispersion, one PIT example, and exact CRPS/log-score comparisons.
Failure case: Ranking interval forecasts by coverage alone, evaluating quantiles with squared error, or interpreting nominal coverage without checking conditional calibration.
Reader payoff: Evaluate full forecast distributions rather than treating uncertainty as a decorative band around a point forecast.
Exclusions: A model zoo of probabilistic forecasting architectures.
-->

## Mathematical spine

For predictive CDF $F_t$ and observation $y_t$, define

$$
U_t=F_t(y_t).
$$

Under a correctly specified continuous predictive distribution,

$$
U_t\sim U(0,1)
$$

marginally. Explain why this does not guarantee conditional calibration.

Introduce the logarithmic score,

$$
S_{\log}(F,y)
=
-\log f(y),
$$

and CRPS,

$$
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left(
F(z)-\mathbf 1\{y\le z\}
\right)^2dz.
$$

Explain why strict propriety matters.

## Worked example

Construct three Gaussian forecasters sharing the correct mean: correct variance, overdispersed, and underdispersed. Compare empirical coverage, interval width, PIT shape, log score and CRPS.

Add a fourth forecaster that is globally calibrated but conditionally wrong in two regimes.

## Reproducibility plan

Generate PIT histograms, quantile reliability plots, interval width versus coverage and proper-score tables.

## Sources to develop

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.

Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.

Hersbach, H. (2000). Decomposition of the continuous ranked probability score.
