---
permalink: '/mathematics/Importance_Sampling/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-11'
header:
  image: /assets/images/headers/photo-mathematics-lecture-blackboard.jpg
  og_image: /assets/images/headers/photo-mathematics-lecture-blackboard.jpg
  overlay_image: /assets/images/headers/photo-mathematics-lecture-blackboard.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-lecture-blackboard.jpg
  twitter_image: /assets/images/headers/photo-mathematics-lecture-blackboard.jpg
redirect_from:
- '/mathematics/statistics/data science/Importance_Sampling/'
seo_description: "Importance sampling estimates expectations under one distribution using samples from another, with accuracy determined by weight variance and support overlap."
seo_title: "Importance Sampling: Reweighting Monte Carlo Carefully"
seo_type: article
subtitle: "Proposal design, weight stability, and rare-event estimation"
tags:
- Monte Carlo
- Probability
- Statistics
title: "Importance Sampling: Reweighting Monte Carlo Carefully"
---

Importance sampling estimates an expectation under a target distribution using samples from a different proposal distribution.

Suppose

$$
mu
=
E_p[h(X)]
=
int h(x)p(x)\,dx.
$$

If we sample from $q$ instead of $p$, then

$$
mu
=
E_q\left[
h(X)\frac{p(X)}{q(X)}
\right],
$$

provided $q(x)>0$ wherever $h(x)p(x)\neq0$.

The importance weight is

$$
w(x)=\frac{p(x)}{q(x)}.
$$

## Why it can help

Ordinary Monte Carlo wastes samples when the important contribution to the integral lies in a rare region under $p$.

A well-chosen proposal $q$ puts more mass in that region.

For a rare-event probability

$$
P_p(X\in A),
$$

the estimator becomes

$$
\widehat p_A
=
\frac{1}{n}
\sum_{i=1}^n
\mathbf 1\{X_i\in A\}
\frac{p(X_i)}{q(X_i)},
\qquad X_i\sim q.
$$

The gain comes from seeing $A$ more often under $q$.

## Unbiasedness does not imply low variance

The basic estimator is unbiased under standard conditions.

Its variance is

$$
\frac{1}{n}
\operatorname{Var}_q\left[
h(X)\frac{p(X)}{q(X)}
\right].
$$

A poor proposal can make this variance enormous or even infinite.

Thus importance sampling does not automatically improve Monte Carlo. It improves Monte Carlo only when the proposal controls weight variability.

## Support mismatch is fatal

If

$$
p(x)>0
$$

in a region where

$$
q(x)=0,
$$

the proposal can never sample that region.

No reweighting can repair information that was never observed.

This support condition is one of the most important practical checks.

## Optimal proposal intuition

For nonnegative $h$, the zero-variance ideal is proportional to

$$
q^*(x)
\propto
h(x)p(x).
$$

Of course, its normalizing constant is the unknown expectation we are trying to compute.

The formula is therefore conceptual: a good proposal should resemble the contribution to the integral, not merely the target density itself.

## Self-normalized importance sampling

Sometimes $p$ is known only up to a normalizing constant.

Then one uses normalized weights

$$
\widetilde w_i
=
\frac{w_i}{\sum_j w_j}
$$

and estimates

$$
\widehat\mu_{SNIS}
=
\sum_i
\widetilde w_i h(X_i).
$$

This estimator is generally biased at finite $n$, although consistent under suitable conditions.

That trade-off should be stated explicitly.

## Effective sample size

A common weight diagnostic is

$$
ESS
=
\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}.
$$

If one weight dominates, ESS can be tiny even when the raw sample size is large.

ESS is a useful warning signal, not a complete proof of estimator quality.

## Heavy tails and instability

Importance sampling is especially fragile when the target has heavier tails than the proposal.

Then large values of

$$
p(x)/q(x)
$$

can occur.

A proposal that looks reasonable near the mode can still produce unstable estimates because tail coverage is inadequate.

## Adaptive importance sampling

Adaptive methods update the proposal using previous weighted samples.

This can improve efficiency, but adaptation must preserve the validity of the estimator.

The adaptation rule is part of the algorithm and should be documented and tested.

## A simple Python implementation

~~~python
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


Density = Callable[[NDArray[np.float64]], NDArray[np.float64]]
Function = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def importance_sampling(
    samples: NDArray[np.float64],
    target_density: Density,
    proposal_density: Density,
    function: Function,
) -> float:
    """Estimate E_p[h(X)] using draws sampled from q."""
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")

    p = target_density(samples)
    q = proposal_density(samples)

    if np.any(q <= 0):
        raise ValueError("proposal density must be positive at all sampled points")

    weights = p / q
    values = function(samples)

    return float(np.mean(weights * values))
~~~

A production implementation should additionally inspect weight diagnostics and Monte Carlo uncertainty.

## Importance sampling in machine learning

Importance weighting appears in covariate-shift correction, off-policy evaluation, rare-event simulation, Bayesian computation, and some reinforcement-learning estimators.

The same warning recurs in all of them:

> large or unstable weights can dominate the estimate.

Clipping weights reduces variance but introduces bias.

## Conclusion

Importance sampling is not simply "focus on important regions."

It is a change-of-measure identity whose success depends on proposal support and weight variance.

The practical workflow is:

$$
\text{choose }q
\rightarrow
\text{compute weights}
\rightarrow
\text{inspect stability}
\rightarrow
\text{estimate uncertainty}.
$$

## References

- Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
