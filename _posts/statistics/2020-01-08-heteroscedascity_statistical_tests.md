---
permalink: '/statistics/heteroscedascity_statistical_tests/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-08'
excerpt: Heteroskedasticity changes the conditional variance of regression errors. It does not by itself bias OLS coefficients when the conditional mean is correctly specified, but it changes efficiency and invalidates naive standard errors.
header:
  image: /assets/images/headers/photo-statistics-regression-errors.jpg
  og_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-regression-errors.jpg
  twitter_image: /assets/images/headers/photo-statistics-regression-errors.jpg
seo_description: Heteroskedasticity explained through conditional variance, Breusch-Pagan and White tests, robust covariance estimators, weighted least squares, and model misspecification.
seo_title: 'Heteroskedasticity: What Changes and What Does Not'
seo_type: article
summary: A precise guide to heteroskedastic regression errors, diagnostics, robust inference, and the distinction between variance misspecification and conditional-mean misspecification.
tags:
- Regression
- Economics
- Statistical Inference
title: 'Heteroskedasticity: What Changes and What Does Not'
---

Consider the linear conditional-mean model

$$
Y_i=X_i^\top\beta+\varepsilon_i,
$$

with

$$
E(\varepsilon_i\mid X_i)=0.
$$

Homoskedasticity adds the assumption

$$
\operatorname{Var}(\varepsilon_i\mid X_i)
=
\sigma^2.
$$

Heteroskedasticity means instead that

$$
\operatorname{Var}(\varepsilon_i\mid X_i)
=
\sigma_i^2
$$

depends on the observation or on predictors. The distinction matters because heteroskedasticity is primarily a **variance-model problem**. If the conditional mean is otherwise correctly specified, ordinary least squares does not suddenly become biased merely because the error variance changes.

## What happens to OLS

The OLS estimator is

$$
\hat\beta
=
(X^\top X)^{-1}X^\top Y.
$$

Under conditional mean zero,

$$
E(\hat\beta\mid X)=\beta
$$

regardless of whether the covariance matrix is $\sigma^2I$. But the actual covariance is

$$
\operatorname{Var}(\hat\beta\mid X)
=
(X^\top X)^{-1}
X^\top\Omega X
(X^\top X)^{-1},
$$

where

$$
\Omega
=
\operatorname{Var}(\varepsilon\mid X).
$$

The usual homoskedastic formula

$$
\sigma^2(X^\top X)^{-1}
$$

is therefore wrong when $\Omega$ is not proportional to the identity matrix. That is why naive t-tests and confidence intervals can be miscalibrated.

## Gauss-Markov efficiency is also lost

Under homoskedastic uncorrelated errors, OLS is BLUE: best among linear unbiased estimators. With known heteroskedastic variances, generalized or weighted least squares can use that variance structure more efficiently. So two consequences should be separated:

1. the ordinary homoskedastic variance estimator is wrong;
2. OLS is no longer generally the most efficient linear unbiased estimator.

Neither statement implies automatic coefficient bias.

## Residual plots are the first diagnostic

Formal tests are useful, but a residual-versus-fitted plot often reveals more. Patterns to look for include a funnel shape, variance increasing with the fitted mean, separate variance bands by group, curvature indicating mean-model misspecification, or isolated high-leverage observations. A changing residual spread may arise because the variance truly changes.

It may also be a symptom of a missing nonlinear term or omitted group structure. Diagnosing the mean and variance together is therefore important.

## Breusch-Pagan test

The Breusch-Pagan idea is to model whether squared OLS residuals vary systematically with predictors. After fitting the original regression, an auxiliary regression is constructed for

$$
\hat\varepsilon_i^2.
$$

A common LM form uses

$$
nR^2
$$

from that auxiliary regression and compares it asymptotically with a chi-square distribution whose degrees of freedom equal the number of variance regressors excluding the intercept. The important point is that the test targets a specified form of variance dependence. A non-rejection does not establish homoskedasticity. A rejection does not identify the correct variance model.

## White's test

White's test uses a richer auxiliary regression containing original regressors, squares, and cross-products. This makes it sensitive to broader forms of heteroskedasticity. The flexibility also means that the auxiliary regression can absorb symptoms of mean-model misspecification. A rejection should therefore prompt inspection of the whole model, not merely replacement of the standard-error formula.

## Why skewed predictors do not imply heteroskedasticity

A predictor can be highly skewed while the conditional error variance remains constant. Likewise, a normally distributed predictor can have strongly heteroskedastic errors. Heteroskedasticity is a property of

$$
\operatorname{Var}(Y\mid X),
$$

not of the marginal shape of $X$ alone. The same warning applies to measurement error: predictor measurement error can bias coefficients through an errors-in-variables mechanism, but it does not mechanically imply heteroskedasticity.

## Robust covariance estimators

If the conditional mean model is the target and heteroskedasticity is the main concern, heteroskedasticity-consistent covariance estimators are often the simplest response. The sandwich form is

$$
\widehat{\operatorname{Var}}(\hat\beta)
=
(X^\top X)^{-1}
X^\top\hat\Omega X
(X^\top X)^{-1}.
$$

Different HC estimators differ in how $\hat\Omega$ corrects for leverage and finite-sample behavior. HC0 is the original large-sample form. HC1 applies a degrees-of-freedom adjustment. HC2 and HC3 increase the correction for high-leverage observations, with HC3 often preferred in smaller samples. The coefficient estimates remain the OLS coefficients.

Only the estimated covariance changes.

## Weighted least squares

If the conditional variance is known up to a useful model,

$$
\operatorname{Var}(\varepsilon_i\mid X_i)
=
\sigma^2 v_i,
$$

then weighted least squares uses

$$
w_i\propto\frac{1}{v_i}.
$$

The estimator is

$$
\hat\beta_{WLS}
=
(X^\top W X)^{-1}
X^\top W Y.
$$

This can improve efficiency substantially. But estimated weights can be wrong. A poorly specified variance model can make WLS less attractive than OLS with robust inference.

## Transformations change the estimand

Taking

$$
\log Y
$$

can sometimes stabilize variance, particularly when variability grows approximately in proportion to the mean. But this is not a free repair. A regression for

$$
E[\log Y\mid X]
$$

is not the same model as a regression for

$$
E[Y\mid X].
$$

Back-transforming fitted values also requires care because

$$
E[\exp(Z)]
\neq
\exp(E[Z])
$$

in general. Transformations should therefore be chosen because the transformed model is scientifically and statistically sensible, not merely because a residual plot looks cleaner.

## Heteroskedasticity can be the model

For positive outcomes, counts, durations, and other non-Gaussian responses, changing conditional variance may be expected. In a Poisson model,

$$
\operatorname{Var}(Y\mid X)
=
E(Y\mid X)
$$

under the basic specification. In a Gamma model, variance often scales with the square of the mean. Trying to remove heteroskedasticity from data generated by such mechanisms can miss the point. A generalized linear model may express the mean-variance relationship directly.

## Reproducible Python example

~~~python
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

rng = np.random.default_rng(2026)

n: int = 1_000
x: np.ndarray = rng.uniform(0.0, 4.0, size=n)

error_sd: np.ndarray = 0.5 + 0.8 * x
epsilon: np.ndarray = rng.normal(
    loc=0.0,
    scale=error_sd,
)

y: np.ndarray = 1.0 + 2.0 * x + epsilon

design = sm.add_constant(x)
model = sm.OLS(y, design).fit()

bp = het_breuschpagan(
    model.resid,
    model.model.exog,
)

robust = model.get_robustcov_results(
    cov_type="HC3"
)

print("OLS parameters:", model.params)
print("naive SE:", model.bse)
print("HC3 SE:", robust.bse)
print("Breusch-Pagan p-value:", bp[1])
~~~

The point estimates remain OLS estimates. The uncertainty calculation changes.

## Conclusion

Heteroskedasticity means that the conditional variance of the regression error is not constant. Its direct consequences are that the classical homoskedastic covariance formula is wrong, OLS loses Gauss-Markov efficiency, and naive significance tests may be miscalibrated. It does not, by itself, prove that the OLS coefficients are biased.

The correct response depends on the objective: robust covariance estimation for inference, weighted least squares for a credible variance model, or a different mean-variance model when heteroskedasticity is part of the data-generating mechanism.

## References

- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287–1294.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.
