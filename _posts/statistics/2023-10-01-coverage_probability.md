---
permalink: '/statistics/coverage_probability/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-10-01'
excerpt: Coverage probability is a property of an interval procedure under repeated sampling. Nominal coverage, conditional coverage, prediction coverage, and calibration are different ideas.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- Coverage probability
- Confidence intervals
- Prediction intervals
- Bootstrap intervals
- Calibration
- Statistical inference
seo_description: Coverage probability explained through confidence intervals, prediction intervals, conditional coverage, model misspecification, and simulation-based assessment.
seo_title: Coverage Probability in Statistical Inference
seo_type: article
tags:
- Confidence Intervals
- Statistical Modeling
title: Coverage Probability in Statistical Inference
---

Coverage probability is a property of an interval-producing procedure. If a confidence interval is written as \(C(X)\), its coverage at parameter value \(\theta\) is

$$
\operatorname{Cov}(\theta)=P_\theta\{\theta\in C(X)\}.
$$

The probability is over hypothetical repetitions of the data-generating process. The parameter is fixed within that frequentist statement; the interval is random because it changes from sample to sample.

## Nominal coverage is a target

A nominal 95% interval is constructed with the intention that

$$
P_\theta\{\theta\in C(X)\}\approx 0.95.
$$

That equality can be exact, asymptotic, conservative, or simply wrong depending on the procedure and assumptions.

Nominal coverage should therefore not be confused with actual coverage.

## Exact and approximate coverage

Some procedures have exact finite-sample coverage under their model assumptions. Others rely on asymptotic approximations.

For example, the usual normal-theory interval for a population mean with known variance has exact Gaussian-model coverage. Replacing unknown quantities by estimates often leads to approximations whose accuracy improves with sample size under regularity conditions.

Small samples, skewed distributions, boundary parameters, weak identification, or model misspecification can create substantial coverage error.

## Conservative and anti-conservative intervals

If actual coverage exceeds the nominal level, the interval is conservative. If it falls below the target, it is anti-conservative.

Coverage alone is not enough to compare intervals. An interval that always spans the entire parameter space has 100% coverage but little practical value.

Width matters too. A useful interval balances calibration and informativeness.

## Confidence intervals and prediction intervals

A confidence interval targets a parameter. A prediction interval targets a future random quantity.

For future observation \(Y_{n+1}\), a prediction procedure \(P(X)\) has coverage

$$
P\{Y_{n+1}\in P(X)\}.
$$

Prediction uncertainty usually includes both uncertainty about the conditional mean and irreducible variation of the future observation.

Calling a prediction interval a confidence interval hides that distinction.

## Marginal versus conditional coverage

An interval can have correct average coverage while performing poorly in particular regions of the predictor space.

For regression prediction, marginal coverage may satisfy

$$
P\{Y\in C(X)\}=1-\alpha,
$$

while conditional coverage

$$
P\{Y\in C(X)\mid X=x\}=1-\alpha
$$

fails for some x.

This matters whenever risk is heterogeneous. Average coverage can hide systematic undercoverage for clinically or operationally important subgroups.

## Bootstrap coverage

Bootstrap intervals are not automatically valid. Their coverage depends on whether the resampling scheme approximates the relevant sampling distribution.

Percentile, basic, studentized, and BCa intervals have different properties. Clustered, dependent, censored, or time-series data generally require a resampling design that respects that structure.

Increasing the number of bootstrap replicates reduces Monte Carlo error. It does not fix an invalid bootstrap model.

## Coverage under model misspecification

Intervals derived under the wrong variance model or likelihood can undercover badly even with large samples.

Robust or sandwich standard errors can improve some forms of variance misspecification, but they do not fix a wrong estimand, omitted confounding, dependence ignored by the design, or severe finite-sample problems.

Coverage should be assessed for the actual estimator and data-generating conditions that matter.

## Simulation as an audit tool

Coverage is often easiest to study by simulation:

1. Choose parameter values and a data-generating process.
2. Simulate many datasets.
3. Apply the complete estimation and interval procedure.
4. Record whether the true parameter lies inside each interval.
5. Estimate coverage as the fraction of successful intervals.

If \(B\) simulations are run and \(I_b\) indicates coverage in replicate b, then

$$
\widehat{\operatorname{Cov}}=\frac{1}{B}\sum_{b=1}^{B}I_b.
$$

Coverage curves over a range of parameter values are more informative than checking one convenient scenario.

## Conformal prediction and finite-sample marginal coverage

Conformal prediction provides distribution-free marginal coverage under exchangeability assumptions. That guarantee is powerful but specific.

It does not generally provide exact conditional coverage for every x, and exchangeability can fail under time dependence, drift, or clustered data.

Guarantees should always be read together with their assumptions.

## Conclusion

Coverage probability is not a confidence score attached to one interval. It is a long-run property of the procedure that generates intervals.

A serious interval analysis should ask: coverage of what quantity, under which data-generating process, marginally or conditionally, and at what cost in interval width?

## References

- Casella, G., & Berger, R. L. (2002). *Statistical Inference*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
- Wasserman, L. (2004). *All of Statistics*.
