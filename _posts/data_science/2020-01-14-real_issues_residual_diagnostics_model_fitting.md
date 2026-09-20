---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-01-14'
excerpt: Residual diagnostics should target specific model assumptions: conditional mean, variance, dependence, tail behavior, leverage, and influence. A normality test alone cannot validate or invalidate a regression model.
header:
  image: /assets/images/headers/photo-statistics-residuals.jpg
  og_image: /assets/images/headers/photo-statistics-residuals.jpg
  overlay_image: /assets/images/headers/photo-statistics-residuals.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-residuals.jpg
  twitter_image: /assets/images/headers/photo-statistics-residuals.jpg
keywords:
- residual diagnostics
- regression diagnostics
- heteroskedasticity
- influence
- normality
seo_description: Residual diagnostics explained through model assumptions, heteroskedasticity, dependence, leverage, influence, tail behavior, and predictive checks rather than a single normality test.
seo_title: 'Residual Diagnostics: Diagnose the Assumption That Matters'
seo_type: article
summary: A rigorous guide to residual diagnostics that separates mean-model misspecification, variance errors, dependence, non-normality, leverage, and influence.
tags:
- Regression
- Statistical Modeling
- Diagnostics
title: 'Residual Diagnostics: Diagnose the Assumption That Matters'
---

A residual is

$$
e_i
=
y_i-\hat y_i.
$$

That simple difference is used to diagnose several different things.

The mistake is to compress them into one question:

> Are the residuals normal?

Normality is only one possible assumption, and often not the most important one.

A useful diagnostic asks

$$
\boxed{
\text{Which property of the model would this pattern contradict?}
}
$$

## Residuals are not the true errors

In a regression model,

$$
Y_i
=
m(X_i)+\varepsilon_i,
$$

the unobserved error is $\varepsilon_i$.

The fitted residual is

$$
e_i
=
Y_i-\hat m(X_i).
$$

Residuals depend on the estimated model and are not independent copies of the errors.

In linear regression,

$$
e=(I-H)y,
$$

where $H$ is the hat matrix.

This creates different residual variances according to leverage.

Raw residuals should therefore not always be treated as identically distributed observations.

## Conditional mean misspecification

The most fundamental regression requirement is often

$$
E[\varepsilon\mid X]=0.
$$

A residual-versus-fitted plot can reveal curvature or structure suggesting

$$
E[e\mid\hat y]\ne0.
$$

Examples include:

- missing nonlinear terms;
- omitted interactions;
- wrong link function;
- missing time trend;
- unmodeled group structure.

A perfect normality test cannot repair a wrong conditional mean.

## Heteroskedasticity

If

$$
\operatorname{Var}(\varepsilon_i\mid X_i)
$$

changes with predictors, residual spread may form a funnel pattern.

This primarily affects the usual covariance estimate, not necessarily the OLS coefficient itself when the conditional mean is correct.

Responses include:

- heteroskedasticity-consistent covariance estimators;
- explicit variance models;
- weighted least squares;
- alternative response distributions.

The response should match the source of the variance change.

## Dependence

Residual autocorrelation can indicate missing temporal dynamics.

Clustered residual patterns can reveal within-group dependence.

For a time series, inspect quantities such as

$$
\operatorname{Corr}(e_t,e_{t-k}).
$$

For repeated measures, model subject-level dependence.

Normal residual histograms say nothing about whether observations are serially correlated.

## Normality and exact small-sample inference

In the classical Gaussian linear model,

$$
\varepsilon\mid X
\sim
\mathcal N(0,\sigma^2I),
$$

normality supports exact finite-sample t and F distributions.

For large samples, many coefficient estimators have approximately normal sampling distributions under much broader error distributions.

That does not mean normality never matters.

It means the consequence of non-normality depends on:

- sample size;
- leverage;
- tail behavior;
- target statistic;
- inferential method.

## Why Shapiro-Wilk is a poor gatekeeper

The Shapiro-Wilk test examines exact normality.

In small samples it may have little power against relevant departures.

In large samples it can reject tiny departures that have negligible effect on the estimator.

So this workflow is weak:

$$
\text{Shapiro p}<0.05
\Rightarrow
\text{model invalid}.
$$

The test answers a narrower question than model validity.

## Q-Q plots

A Q-Q plot is useful because the **shape** of the departure is visible.

Common patterns include:

- S-shaped tails: heavier or lighter tails;
- one curved tail: skewness;
- isolated points: potential outliers;
- broad departures: mixture or wrong error family.

The visual pattern should lead to a specific statistical question.

## Skewness and kurtosis

Sample skewness and kurtosis summarize aspects of residual shape.

They are descriptive statistics, not model diagnoses by themselves.

High kurtosis can arise from heavy tails or isolated extreme observations.

The normal distribution has ordinary kurtosis 3 and excess kurtosis 0.

Software differs in which convention it reports.

That convention should be stated before comparing a value with “3” or “0.”

## Leverage

Leverage is determined by the design matrix.

For linear regression,

$$
H
=
X(X^\top X)^{-1}X^\top.
$$

The diagonal element

$$
h_{ii}
$$

measures how unusual observation $i$ is in predictor space.

A high-leverage observation can strongly affect the fitted model even when its residual is not large.

Outlier diagnostics based only on $y$ miss this.

## Influence

Influence combines residual size and leverage.

Cook's distance is one common summary.

The purpose is not to delete every influential point.

The useful sequence is:

1. verify the data;
2. understand why the point is influential;
3. refit with and without it;
4. report sensitivity if conclusions change.

Automatic deletion creates its own selection bias.

## Standardized and studentized residuals

Because residual variance depends on leverage, standardized forms are easier to compare.

A common internally studentized residual is roughly

$$
r_i
=
\frac{
e_i
}{
\hat\sigma\sqrt{1-h_{ii}}
}.
$$

Externally studentized residuals estimate $\sigma$ with the observation omitted.

These are more appropriate than raw residuals for identifying unusually large conditional errors.

## Mixed models need conditional diagnostics

For a mixed model such as

$$
Y_{ij}
=
X_{ij}^\top\beta
+
Z_{ij}^\top b_i
+
\varepsilon_{ij},
$$

there are at least two stochastic components:

- random effects $b_i$;
- residual errors $\varepsilon_{ij}$.

A single Shapiro-Wilk test on one residual vector cannot diagnose both.

Useful diagnostics include:

- conditional residuals;
- random-effect distributions;
- residual variance by time or group;
- within-subject correlation;
- influence at the subject level.

The design unit matters.

## Predictive diagnostics

If prediction is the goal, residual fit on the training sample is not enough.

Out-of-sample residuals

$$
e_i^{test}
=
y_i-\hat y_i^{train}
$$

reveal generalization error.

Calibration, coverage of prediction intervals, and performance across subgroups may be more important than whether training residuals look Gaussian.

## Simulation-based diagnostics

For complex models, simulate replicated data from the fitted model.

If the model is adequate, simulated data should reproduce features that matter scientifically:

- variance;
- zeros;
- extreme values;
- autocorrelation;
- cluster patterns;
- event rates.

This idea appears in posterior predictive checking, parametric bootstrap diagnostics, and simulation-based residual methods.

It scales better than forcing every model into a normal-residual template.

## Conclusion

Residual diagnostics are not one test.

They are a collection of checks linked to specific assumptions:

$$
\boxed{
\text{mean}
+
\text{variance}
+
\text{dependence}
+
\text{tails}
+
\text{leverage}
+
\text{influence}
+
\text{prediction}.
}
$$

The right question is not whether residuals pass a normality threshold.

It is whether the fitted model is adequate for the inferential or predictive claim being made.

## References

- Cook, R. D., & Weisberg, S. (1982). *Residuals and Influence in Regression*. Chapman & Hall.
- Fox, J. (2015). *Applied Regression Analysis and Generalized Linear Models* (3rd ed.). SAGE.
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3/4), 591–611.
