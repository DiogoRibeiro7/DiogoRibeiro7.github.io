---
permalink: '/statistics/assessing_goodnessoffit_nonparametric_data/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-03'
excerpt: The Kolmogorov-Smirnov statistic measures the largest gap between cumulative distributions, but its null distribution changes when model parameters are estimated from the same data.
header:
  image: /assets/images/headers/photo-statistics-ecdf.jpg
  og_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-ecdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-ecdf.jpg
keywords:
- Kolmogorov-Smirnov test
- goodness of fit
- empirical distribution function
- Lilliefors test
- distribution testing
seo_description: A rigorous guide to one-sample and two-sample Kolmogorov-Smirnov tests, parameter estimation, Lilliefors corrections, and when other goodness-of-fit tests are preferable.
seo_title: 'Kolmogorov-Smirnov Goodness-of-Fit: What the Test Actually Assumes'
seo_type: article
summary: A precise treatment of the Kolmogorov-Smirnov statistic, including the difference between fully specified null distributions and distributions fitted from the same sample.
tags:
- Hypothesis Testing
- Goodness of Fit
- Statistics
title: 'Kolmogorov-Smirnov Goodness-of-Fit: What the Test Actually Assumes'
---

The Kolmogorov-Smirnov test is often introduced as a distribution-free way to ask whether data follow a named distribution.

That description is incomplete.

The statistic is simple. The null distribution is simple only in the classical case where the reference distribution is fully specified before seeing the data.

The distinction matters most in the common workflow of estimating a normal mean and variance from the sample and then running an ordinary one-sample K-S test against that fitted normal distribution.

The usual K-S p-value is not valid for that procedure.

## The empirical distribution function

Given observations

$$
X_1,\ldots,X_n,
$$

the empirical cumulative distribution function is

$$
F_n(x)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1(X_i\le x).
$$

For a fully specified continuous reference distribution with CDF $F_0$, the one-sample K-S statistic is

$$
D_n
=
\sup_x
|F_n(x)-F_0(x)|.
$$

It measures the largest vertical separation between the empirical and theoretical CDFs.

The null hypothesis is

$$
H_0:
X_i\overset{\mathrm{iid}}{\sim}F_0.
$$

The word **specified** is doing important work.

## Why the classical one-sample test is distribution-free

If $F_0$ is continuous and completely specified, the probability integral transform gives

$$
U_i=F_0(X_i)
\sim
\mathrm{Uniform}(0,1)
$$

under $H_0$.

Therefore the null distribution of $D_n$ does not depend on the particular continuous $F_0$.

That is the classical distribution-free property.

It does not mean that every workflow involving a fitted distribution has the same null law.

## Estimating parameters changes the null distribution

Suppose the null model is normal but the parameters are unknown:

$$
X_i
\sim
\mathcal N(\mu,\sigma^2).
$$

If $\mu$ and $\sigma$ are estimated from the same sample and the empirical CDF is then compared with

$$
\Phi
\left(
\frac{x-\hat\mu}{\hat\sigma}
\right),
$$

the fitted CDF has been pulled toward the observations.

The discrepancy is therefore systematically smaller than it would be against a fixed distribution.

The standard Kolmogorov critical values no longer apply.

For the normal case, Lilliefors derived the corresponding corrected null distribution.

So these are different tests:

$$
\text{K-S against } \mathcal N(0,1)
$$

and

$$
\text{normality test after estimating } \mu,\sigma.
$$

They should not share the same p-value calibration.

## The two-sample K-S test

For independent samples with empirical CDFs $F_n$ and $G_m$, the two-sample statistic is

$$
D_{n,m}
=
\sup_x
|F_n(x)-G_m(x)|.
$$

The null hypothesis is

$$
H_0:F=G.
$$

This is stronger than equality of means or medians.

The test can react to differences in location, scale, skewness, tail behavior, or any other feature that changes the CDF.

A rejection therefore does not tell us how the distributions differ.

That requires plots, effect summaries, or a more targeted model.

## K-S is not a general test for “non-parametric data”

The phrase “non-parametric data” is not useful here.

Data are not parametric or non-parametric.

Models and procedures are.

The K-S statistic can compare an empirical distribution with a parametric model, or compare two empirical samples without specifying a parametric family.

The procedure is called nonparametric in the two-sample setting because the null does not impose a finite-dimensional parametric family.

That terminology should not be confused with a property of the observations themselves.

## Normality testing is a special case

If the sole question is exact normality with estimated mean and variance, Shapiro-Wilk is usually a stronger general-purpose choice than plugging fitted parameters into an uncorrected K-S test.

Anderson-Darling places additional weight in the tails.

A Lilliefors-type test modifies the K-S calibration to account for parameter estimation.

The useful question is not which test is universally best.

It is

$$
\boxed{
\text{Which deviations from the model matter for the analysis?}
}
$$

A tail-sensitive problem may call for a tail-sensitive diagnostic. A location-scale modeling problem may be better assessed through residual structure rather than a generic omnibus test.

## P-values do not identify the model

A large p-value does not establish that the proposed distribution is true.

It says that the observed discrepancy is not unusually large under the null calibration of the test.

A small sample may have little power against important alternatives.

A very large sample may reject a model because of a tiny discrepancy that has no practical consequence.

Graphical diagnostics remain valuable because they reveal where the model fails.

## Q-Q plots and ECDF differences

A Q-Q plot compares empirical order statistics with theoretical quantiles.

The shape of the departure can distinguish skewness, heavy tails, light tails, isolated outliers, mixtures, or central fit with tail failure.

Similarly, plotting

$$
F_n(x)-F_0(x)
$$

shows where the K-S maximum occurs.

The scalar $D_n$ records only the largest discrepancy.

The plot contains more information.

## Discrete distributions need separate care

The classical continuous K-S null distribution assumes a continuous reference CDF.

For discrete distributions, ties occur with positive probability and the null distribution of the statistic changes.

Using continuous critical values can be conservative or otherwise miscalibrated depending on the setting.

Exact, Monte Carlo, or discrete-specific goodness-of-fit procedures are preferable when the null distribution is discrete.

## A reproducible example

Suppose we genuinely want to test

$$
H_0:X\sim\mathcal N(0,1).
$$

Then the reference distribution is fully specified.

~~~python
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)

sample: np.ndarray = rng.normal(
    loc=0.0,
    scale=1.0,
    size=200,
)

result = stats.kstest(
    sample,
    "norm",
    args=(0.0, 1.0),
)

print(result.statistic)
print(result.pvalue)
~~~

That is a classical one-sample K-S test.

If we instead estimate the mean and standard deviation from the same sample, the ordinary K-S p-value is no longer the correct normality-test calibration.

## Simulation can calibrate fitted-model tests

When parameters are estimated and no convenient analytic correction is available, a parametric bootstrap gives a direct solution.

The logic is:

1. fit the model to the observed data;
2. compute the observed goodness-of-fit statistic;
3. simulate many samples from the fitted model;
4. refit the model separately in each simulated sample;
5. recompute the statistic;
6. compare the observed statistic with that simulated null distribution.

The crucial step is refitting on every bootstrap sample.

That reproduces the same parameter-estimation effect that occurred in the original analysis.

## Conclusion

The K-S statistic is easy to compute:

$$
D=\sup_x|F_n(x)-F(x)|.
$$

The hard part is knowing which null distribution belongs to the way $F$ was obtained.

If the reference CDF is fully specified in advance, the classical continuous one-sample K-S distribution applies.

If parameters are estimated from the same observations, it generally does not.

That distinction is more important than labeling the method “non-parametric.”

## References

- Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.
- Smirnov, N. (1948). Table for estimating the goodness of fit of empirical distributions. *Annals of Mathematical Statistics*, 19(2), 279–281.
- Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. *Journal of the American Statistical Association*, 62(318), 399–402.
- Stephens, M. A. (1974). EDF statistics for goodness of fit and some comparisons. *Journal of the American Statistical Association*, 69(347), 730–737.
