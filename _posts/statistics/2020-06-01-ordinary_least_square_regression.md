---
permalink: '/statistics/ordinary_least_square_regression/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-06-01'
excerpt: OLS is a projection estimator with precise properties under specific assumptions. This article separates unbiasedness, consistency, Gauss-Markov efficiency, normal-theory inference, and causal interpretation.
header:
  image: /assets/images/headers/photo-statistics-f-test.jpg
  og_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-f-test.jpg
  twitter_image: /assets/images/headers/photo-statistics-f-test.jpg
keywords:
- Ordinary least squares
- Linear regression
- Gauss-Markov
- Consistency
- Heteroskedasticity
- Causal inference
seo_description: OLS regression explained through projection, exogeneity, consistency, Gauss-Markov efficiency, robust standard errors, and the distinction between association and causation.
seo_title: 'OLS Regression: What Its Properties Actually Require'
seo_type: article
summary: A precise guide to ordinary least squares that separates estimation from inference and causal interpretation.
tags:
- Regression
- Statistical Modeling
- Probability
title: 'Ordinary Least Squares: What Its Properties Actually Require'
---

Ordinary least squares is simple enough to be taught in a first statistics course and subtle enough to be misused for an entire career.

The estimator itself is only an optimization rule.

Given a response vector \(y\) and design matrix \(X\), OLS chooses

$$
\hat\beta
=
\arg\min_b
(y-Xb)^\top(y-Xb).
$$

If \(X^\top X\) is invertible,

$$
\hat\beta
=
(X^\top X)^{-1}X^\top y.
$$

Everything else — unbiasedness, consistency, standard errors, optimality, likelihood interpretation, or causal meaning — requires assumptions.

Those properties should not be bundled together.

## OLS as a projection

Write the population model as

$$
Y=X^\top\beta+\varepsilon.
$$

At the sample level, the fitted values are

$$
\hat y = Hy,
$$

where

$$
H
=
X(X^\top X)^{-1}X^\top
$$

is the projection matrix.

The residual vector is

$$
\hat\varepsilon
=
(I-H)y.
$$

Geometrically, OLS projects the observed response onto the column space of \(X\).

That geometric statement is exact and does not require normal errors.

## Unbiasedness requires conditional mean zero

A central condition is

$$
E(\varepsilon\mid X)=0.
$$

Under this assumption,

$$
E(\hat\beta\mid X)=\beta.
$$

This is the finite-sample unbiasedness result.

It is stronger than saying that the regressors are merely uncorrelated with the residuals in the observed sample. Sample residual orthogonality,

$$
X^\top\hat\varepsilon=0,
$$

is created mechanically by OLS and therefore cannot verify population exogeneity.

The assumption concerns the data-generating process.

## Consistency is not identical to unbiasedness

Consistency means

$$
\hat\beta
\xrightarrow{p}
\beta
$$

as the sample size increases.

Under standard regularity conditions, one route to consistency is

$$
\frac{1}{n}X^\top\varepsilon
\xrightarrow{p}
0
$$

together with a well-behaved limiting design matrix.

An estimator can be biased in finite samples and still be consistent.

Conversely, an estimator can be approximately unbiased in one finite sample setting without satisfying the conditions needed for consistency under repeated sampling.

The two concepts should be kept separate.

## What Gauss-Markov actually says

Suppose

$$
E(\varepsilon\mid X)=0
$$

and

$$
\operatorname{Var}(\varepsilon\mid X)
=
\sigma^2 I.
$$

Then OLS is BLUE:

$$
\boxed{
\text{Best Linear Unbiased Estimator}
}
$$

The word **linear** matters.

The theorem says that among estimators that are linear in \(y\) and unbiased, OLS has the smallest covariance matrix in the positive-semidefinite ordering.

It does **not** say that OLS has minimum variance among all possible unbiased estimators.

That stronger claim is false in general.

## Heteroskedasticity does not bias OLS by itself

Suppose instead

$$
\operatorname{Var}(\varepsilon_i\mid X)
=
\sigma_i^2.
$$

If

$$
E(\varepsilon\mid X)=0
$$

still holds, OLS coefficients remain unbiased under the classical fixed-\(X\) argument and consistent under standard asymptotic conditions.

What fails is the homoskedastic variance formula.

The conventional estimator

$$
\widehat{\operatorname{Var}}(\hat\beta)
=
\hat\sigma^2(X^\top X)^{-1}
$$

is then generally wrong.

Heteroskedasticity-robust covariance estimators address the inference problem without changing the OLS point estimate.

This distinction is important:

$$
\boxed{
\text{heteroskedasticity}
\not\Rightarrow
\text{biased OLS coefficients}
}
$$

unless it is accompanied by a failure of the conditional-mean assumption or another source of misspecification.

## Correlated errors change efficiency and inference

In time series, panel data or clustered samples,

$$
\operatorname{Cov}(\varepsilon_i,\varepsilon_j\mid X)
\neq 0
$$

may be expected.

Again, OLS coefficients can remain consistent under suitable exogeneity conditions, while naive standard errors fail.

Depending on the design, alternatives include:

- heteroskedasticity-and-autocorrelation-consistent covariance estimators;
- cluster-robust covariance estimators;
- generalized least squares;
- explicit time-series or hierarchical models.

The covariance structure is an inferential assumption, not a decorative detail.

## Normality is not required for OLS estimation

Normality enters a different part of the theory.

If

$$
\varepsilon\mid X
\sim
\mathcal N(0,\sigma^2I),
$$

then maximizing the Gaussian likelihood with respect to \(\beta\) is equivalent to minimizing the residual sum of squares.

So under the Gaussian model, OLS is also the maximum-likelihood estimator for \(\beta\).

Normality is not required for the algebraic OLS solution, finite-sample unbiasedness under conditional mean zero, or large-sample consistency.

It is primarily relevant to exact finite-sample distribution theory and likelihood-based interpretation.

## OLS coefficients are not automatically causal effects

Suppose

$$
Y=\beta_0+\beta_1X+\varepsilon.
$$

The coefficient \(\beta_1\) is causal only if the design and assumptions support that interpretation.

If an omitted variable \(Z\) affects both \(X\) and \(Y\), then

$$
E(\varepsilon\mid X)\neq 0
$$

after \(Z\) is omitted.

The regression coefficient can then mix the effect of \(X\) with systematic differences in \(Z\).

Adding more observations does not remove omitted-variable bias.

OLS is an estimator.

Causal identification comes from research design and assumptions such as randomization, conditional exchangeability, instrumental variables, discontinuities, panel structure, or another defensible identification strategy.

## Prediction and coefficient interpretation are different objectives

A linear model can predict well even when individual coefficients are unstable because predictors are highly correlated.

Conversely, a coefficient can have a clear scientific interpretation while the linear model is not the best predictive system.

For prediction, the relevant target is out-of-sample loss.

For parameter inference, the relevant target is uncertainty about \(\beta\) under a model.

For causal inference, the target is a causal estimand under an identification strategy.

Those goals can overlap.

They should not be conflated.

## What multicollinearity actually does

When predictors are nearly linearly dependent, \(X^\top X\) becomes ill-conditioned.

The variance of the OLS estimator can become large because

$$
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2(X^\top X)^{-1}
$$

under homoskedastic errors.

This does not create bias by itself.

It creates unstable coefficient estimates and makes it difficult to separate the contributions of correlated predictors.

Regularization can improve prediction by trading some bias for lower variance, but it changes the estimator and therefore the inferential problem.

## Residual diagnostics should target assumptions

A useful diagnostic workflow asks which assumption each plot or test addresses.

- Residuals versus fitted values can reveal nonlinearity or changing variance.
- Q-Q plots can expose tail behavior relevant to finite-sample normal theory.
- Residuals over time can reveal dependence.
- Leverage and influence diagnostics identify observations with unusual impact on the fitted coefficients.
- Out-of-sample validation addresses predictive generalization.

No single residual plot can establish exogeneity.

The most consequential assumption often cannot be verified from residuals alone.

## Conclusion

OLS has several different properties under several different assumption sets.

The clean summary is:

- the estimator minimizes squared residuals by definition;
- conditional mean zero gives unbiasedness;
- regularity conditions plus exogeneity give consistency;
- homoskedastic uncorrelated errors give Gauss-Markov efficiency among linear unbiased estimators;
- Gaussian errors give the familiar likelihood interpretation and exact normal-theory results;
- none of those conditions, by themselves, identify a causal effect.

That separation makes OLS easier to reason about because each conclusion is attached to the assumption that actually supports it.

## References

- Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium*.
- Markov, A. A. (1912). *Wahrscheinlichkeitsrechnung*.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- Wooldridge, J. M. (2020). *Introductory Econometrics: A Modern Approach* (7th ed.). Cengage.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
