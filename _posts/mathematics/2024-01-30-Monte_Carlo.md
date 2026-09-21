---
permalink: '/mathematics/Monte_Carlo/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-01-30'
draft: false
excerpt: "Monte Carlo methods approximate expectations with random samples. MCMC is one important special case, not a synonym for Monte Carlo."
header:
  image: /assets/images/headers/photo-mathematics-voronoi.jpg
  og_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-voronoi.jpg
  twitter_image: /assets/images/headers/photo-mathematics-voronoi.jpg
keywords:
- Monte Carlo
- Importance sampling
- MCMC
- Variance reduction
- Bayesian computation
math: true
seo_description: "A rigorous introduction to Monte Carlo estimation, error rates, importance sampling, variance reduction, and the role of MCMC."
seo_title: "Monte Carlo Methods: Sampling, Error, and Variance Reduction"
seo_type: article
tags:
- Probability
- Monte Carlo
- Bayesian Statistics
title: "Monte Carlo Methods: Sampling, Error, and Variance Reduction"
---

Monte Carlo methods approximate mathematical quantities with random simulation.

The basic problem is often an expectation,

$$
\mu
=
E_f[h(X)]
=
\int h(x)f(x)\,dx.
$$

If we can generate independent draws

$$
X_1,\ldots,X_n\sim f,
$$

then

$$
\widehat\mu_n
=
\frac{1}{n}
\sum_{i=1}^n h(X_i)
$$

is the standard Monte Carlo estimator.

## Monte Carlo error

If

$$
\operatorname{Var}[h(X)]<\infty,
$$

then

$$
\operatorname{Var}(\widehat\mu_n)
=
\frac{\sigma_h^2}{n}.
$$

The standard error therefore decreases as

$$
O(n^{-1/2}).
$$

This convergence rate is slow but largely independent of the dimension of the integration space, which is one reason Monte Carlo remains useful in high-dimensional problems.

## The central limit theorem

Under standard conditions,

$$
\sqrt n(
\widehat\mu_n-\mu
)
\xrightarrow{d}
N(0,\sigma_h^2).
$$

This supports Monte Carlo confidence intervals based on the empirical variance of $h(X_i)$.

The uncertainty reported here is simulation uncertainty. It is distinct from statistical uncertainty in the model being simulated.

## Estimating pi is a toy example

Drawing uniform points in a square and counting how many fall inside a quarter-circle gives

$$
\widehat\pi
=
4
\frac{1}{n}
\sum_{i=1}^n
\mathbf 1\{X_i^2+Y_i^2\le 1\}.
$$

This demonstrates the principle but not the reason Monte Carlo is important. Its real strength is approximating integrals and distributions that are difficult analytically.

## Importance sampling

Suppose sampling from the target density $f$ is difficult, but we can sample from proposal density $g$.

Then

$$
E_f[h(X)]
=
E_g\left[
h(X)\frac{f(X)}{g(X)}
\right],
$$

provided $g$ covers the relevant support.

The weights are

$$
w(x)=\frac{f(x)}{g(x)}.
$$

A poor proposal can produce extremely variable weights and a nearly useless estimator.

Importance sampling is therefore a variance-design problem, not merely a change of distribution.

## Variance reduction

More samples are not the only way to improve Monte Carlo accuracy.

Common methods include:

- antithetic variates
- control variates
- stratified sampling
- importance sampling
- quasi-Monte Carlo

A control variate uses a correlated variable $C$ with known expectation:

$$
\widehat\mu_{cv}
=
\bar H
-
\beta(
\bar C-E[C]
).
$$

The optimal coefficient depends on covariance between $H$ and $C$.

## MCMC is a special case

When direct independent sampling is unavailable, Markov chain Monte Carlo constructs dependent samples whose stationary distribution is the target.

The estimator still has the form

$$
\widehat\mu_n
=
\frac{1}{n}
\sum_{i=1}^n h(X_i),
$$

but autocorrelation changes its variance.

An effective sample size summarizes this loss of information.

Thus

> Monte Carlo = simulation-based integration

while

> MCMC = Monte Carlo using a Markov chain to generate target-distributed draws.

They should not be used as synonyms.

## Burn-in is not magic

Discarding an initial segment can reduce initialization effects, but it does not prove convergence.

A poorly mixing chain can remain wrong after an arbitrarily chosen burn-in period.

Multiple chains, diagnostics, and geometry-aware samplers are more informative.

## Quasi-Monte Carlo

Quasi-Monte Carlo replaces random points with deterministic low-discrepancy sequences designed to fill the unit cube evenly.

For sufficiently smooth integrands, convergence can be faster than ordinary Monte Carlo.

Randomized quasi-Monte Carlo can recover uncertainty estimation while retaining improved space-filling properties.

## Nested Monte Carlo

Some problems contain expectations inside expectations.

Naively nesting simulation can be very expensive and can create bias in nonlinear outer functions.

This appears in Bayesian design, reinforcement learning, risk analysis, and probabilistic numerics.

The nesting structure should be analyzed before simply increasing sample counts.

## Reproducibility

Modern code should use explicit random-number generators rather than global random state.

~~~python
from __future__ import annotations

import numpy as np


def estimate_pi(draws: int, seed: int = 42) -> tuple[float, float]:
    """Estimate pi and its Monte Carlo standard error."""
    if draws <= 0:
        raise ValueError("draws must be positive")

    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(draws, 2))

    inside = np.sum(points**2, axis=1) <= 1.0
    p_hat = float(np.mean(inside))

    pi_hat = 4.0 * p_hat
    se = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / draws)

    return pi_hat, float(se)
~~~

A simulation result without an estimate of simulation error is incomplete.

## Conclusion

Monte Carlo methods replace difficult integration with sampling.

Their core questions are:

- what distribution is sampled?
- what estimator is used?
- how large is its variance?
- are samples independent?
- can variance be reduced more efficiently than by adding samples?

MCMC belongs inside this larger framework rather than defining it.

## References

- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*.
- Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
