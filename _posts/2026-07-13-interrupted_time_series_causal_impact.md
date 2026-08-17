---
permalink: '/time-series/interrupted_time_series_causal_impact/'
title: 'Interrupted Time Series and Causal Impact'
categories:
- Time Series
tags:
- Time Series
- Correlation
- Statistical Modeling
- Experimental Design
author_profile: false
seo_title: 'Interrupted Time Series and Causal Impact'
seo_description: 'You cannot randomise a policy change. Interrupted time series and synthetic controls estimate its effect from what came before.'
excerpt: >-
  A intervention happened at a known date and you need its effect. There is no
  control group, only the series itself before and after, and the
  counterfactual has to be constructed.
summary: >-
  How to estimate the effect of an intervention when randomisation is
  impossible: segmented regression for level and slope changes, the
  assumptions the counterfactual rests on, Bayesian structural time series
  with control series, and the confounders that make a result unsafe.
keywords:
  - interrupted time series
  - causal impact
  - segmented regression
  - synthetic control
  - counterfactual
classes: wide
date: '2026-07-13'
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
---
A speed limit changed on 1 March. A new triage protocol started in January. A feature shipped on a Tuesday. In each case you need the effect of the intervention, and randomisation was never possible — the change applied to everyone at once.

What you have is the series itself, before and after. The whole problem is constructing what *would* have happened without the intervention.

## Segmented Regression

The standard approach fits a regression with terms that allow the series to change level and slope at a known date:

$$
y_t = \beta_0 + \beta_1 t + \beta_2 D_t + \beta_3 (t - T_0) D_t + \varepsilon_t,
$$

where $D_t$ is 1 after the intervention and $T_0$ is the intervention time. The coefficients separate two distinct effects:

- $\beta_1$ — the pre-intervention slope.
- $\beta_2$ — an **immediate level change** at the intervention.
- $\beta_3$ — a **change in slope** afterwards.

Distinguishing these matters substantively. A policy producing a one-off drop that then resumes the old trajectory is very different from one that changes the trajectory itself, and a model with only a level term will misattribute the second as the first.

```python
import numpy as np

rng = np.random.default_rng(0)
n, T0 = 120, 60
t = np.arange(n)
D = (t >= T0).astype(float)

# truth: pre-slope 0.5, level drop of -8 at T0, slope changes to 0.5 - 0.3
y = (50 + 0.5 * t - 8 * D - 0.3 * (t - T0) * D + rng.normal(0, 2.5, n))

X = np.column_stack([np.ones(n), t, D, (t - T0) * D])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
names = ["intercept", "pre-slope", "level change", "slope change"]
truth = [50, 0.5, -8, -0.3]
for nm, b, tr in zip(names, beta, truth):
    print(f"{nm:15}{b:8.3f}   (true {tr})")

# the counterfactual: what the pre-trend alone predicts after T0
counterfactual = beta[0] + beta[1] * t
effect = (y - counterfactual)[T0:].mean()
print(f"\nmean effect after intervention: {effect:.2f}")
```

The counterfactual here is simply the pre-intervention trend extended forward. That is the entire inferential content of the method, and it is where the assumptions live.

## What the Counterfactual Assumes

The estimate is only as good as the claim that, absent the intervention, the pre-existing trend would have continued. Several things break that.

**Concurrent events.** If anything else changed at the same time, its effect is indistinguishable from the intervention's. A speed limit introduced alongside a public safety campaign confounds the two permanently — no amount of modelling separates them.

**Regression to the mean.** Interventions are frequently triggered *by* an extreme observation. If a protocol changed because incidents spiked, some of the subsequent decline would have happened anyway. This is one of the most common ways interrupted time series overstates an effect.

**Autocorrelation.** Residuals in a time series are correlated, so ordinary standard errors are too small and significance is overstated. Newey-West standard errors, or explicitly modelling the error as ARIMA, is not optional here.

**Seasonality.** Comparing a post-intervention winter against a pre-intervention summer attributes the season to the policy. Seasonal terms must be in the model, or the comparison must span whole cycles.

**Anticipation and phase-in.** If behaviour changed before the official date because the change was announced, or the policy rolled out gradually, a sharp break at $T_0$ is the wrong specification.

## Adding a Control Series

The strongest version of this design uses a comparison series that was *not* subject to the intervention but responds to the same background conditions — a neighbouring region, an untreated product line, a similar hospital.

The counterfactual then comes from the control's behaviour rather than from extrapolating a trend, which handles concurrent shocks that affect both. This is the logic of difference-in-differences, and it rests on the **parallel trends** assumption: the two series would have moved together absent the intervention. That assumption is checkable before the intervention and untestable after, which is exactly its weakness.

**Bayesian structural time series**, as implemented in CausalImpact, formalises this. It fits a state space model to the pre-period using control series as regressors, projects it forward, and reports the difference from the observed post-period with credible intervals. Its advantages are that it handles trend and seasonality explicitly, propagates uncertainty properly, and can select among many candidate controls.

**Synthetic control** goes further, constructing a weighted combination of untreated units that best reproduces the treated unit's pre-intervention path, then using that combination as the counterfactual. It suits the case of one treated unit and many candidate controls — a single region, state, or country.

## Reading a Result Honestly

Three questions separate a credible estimate from a decorated coincidence.

Does the pre-period fit well, and over a long enough window to establish the trend? A counterfactual extrapolated from six observations is a guess.

Would the effect survive a placebo test? Running the same analysis on a fake intervention date, or on a control series that received no intervention, should produce nothing. If it produces an "effect", the method is finding structure that is not causal.

Is the effect large relative to the pre-period variability? Interrupted time series has low power for small effects, and a modest shift in a noisy series will not be distinguishable no matter how the model is specified.

Interrupted time series is a genuinely useful design when randomisation is impossible, and it is weaker than a randomised experiment in a specific way worth stating plainly: it assumes the future would have resembled the past, and that assumption is never verifiable for the period you care about.

## References

- Bernal, J. L., Cummins, S., & Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions. *International Journal of Epidemiology*, 46(1), 348-355.
- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1), 247-274.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
- Wagner, A. K., Soumerai, S. B., Zhang, F., & Ross-Degnan, D. (2002). Segmented regression analysis of interrupted time series studies in medication use research. *Journal of Clinical Pharmacy and Therapeutics*, 27(4), 299-309.
