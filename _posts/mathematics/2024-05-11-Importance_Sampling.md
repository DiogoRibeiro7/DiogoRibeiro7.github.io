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
seo_description: "A rigorous treatment of importance sampling, covering change of measure, variance, proposal design, self-normalized estimators, effective sample size, and rare-event simulation."
seo_title: "Importance Sampling: Change of Measure, Variance, and Proposal Design"
seo_type: article
subtitle: "Change of measure, variance control, and rare-event estimation"
tags:
- Monte Carlo
- Probability
- Statistics
title: "Importance Sampling: Change of Measure, Variance, and Proposal Design"
---

Importance sampling is one of the most useful examples of a simple probabilistic identity becoming a powerful computational method. The central idea is to estimate an expectation with respect to one probability distribution while drawing samples from another, provided that the discrepancy between the two distributions is corrected through an appropriate likelihood ratio. This sounds, at first, like a technical reformulation of ordinary Monte Carlo integration. In practice, however, the choice of sampling distribution can determine whether an estimator is computationally efficient, numerically unstable, or effectively useless. Importance sampling is therefore not merely a device for "sampling important regions more often." It is a change-of-measure method whose statistical performance depends on support, tail behaviour, and the variance of the resulting weights.

Suppose that the quantity of interest is

$$
\mu
=
E_p[h(X)]
=
\int h(x)p(x)\,dx,
$$

where $p$ is the target density and $h$ is an integrable function. If direct sampling from $p$ is straightforward, the ordinary Monte Carlo estimator is

$$
\widehat\mu_n
=
\frac{1}{n}
\sum_{i=1}^n h(X_i),
\qquad
X_i\sim p.
$$

Under the usual finite-variance assumptions, this estimator is unbiased and its standard error decreases at the familiar Monte Carlo rate $n^{-1/2}$. The difficulty arises when the region that contributes most strongly to the expectation has low probability under $p$, or when sampling directly from $p$ is difficult. A rare-event probability is the clearest example. If we want to estimate $P_p(X\in A)$ for an event $A$ that occurs once in a million draws, then ordinary Monte Carlo spends nearly all of its computational budget observing outcomes that contribute exactly zero to the indicator function. The estimator remains valid, but its efficiency can be extremely poor.

## The change-of-measure identity

Importance sampling introduces a proposal density $q$ from which sampling is easier or more strategically useful. Provided that $q(x)>0$ wherever $h(x)p(x)\neq0$, we may multiply and divide the integrand by $q(x)$:

$$
\mu
=
\int
h(x)
\frac{p(x)}{q(x)}
q(x)\,dx.
$$

The same expectation can therefore be written as

$$
\mu
=
E_q
\left[
h(X)
\frac{p(X)}{q(X)}
\right].
$$

If $X_1,\ldots,X_n$ are sampled independently from $q$, the corresponding importance-sampling estimator is

$$
\widehat\mu_{IS}
=
\frac{1}{n}
\sum_{i=1}^n
h(X_i)w(X_i),
\qquad
w(x)
=
\frac{p(x)}{q(x)}.
$$

The ratio $w(x)$ is the importance weight. Its role is to undo the deliberate distortion introduced by sampling from $q$ rather than from $p$. Observations drawn too frequently under $q$ receive relatively small weights, whereas observations that are rare under $q$ but important under $p$ receive larger weights. The proposal distribution is therefore allowed to bias the sampling mechanism, but the estimator corrects that bias through reweighting.

This identity is mathematically elementary, yet its practical implications are substantial. A good proposal distribution places more probability mass in regions where the product $|h(x)|p(x)$ is large, thereby allocating simulation effort where it contributes most strongly to the integral. A poor proposal can do the opposite. In that case, the importance weights become highly variable, a small number of observations dominate the estimate, and the variance can be far larger than that of ordinary Monte Carlo.

## Unbiasedness is not the same as efficiency

For the basic estimator above, the variance is

$$
\operatorname{Var}
\left(
\widehat\mu_{IS}
\right)
=
\frac{1}{n}
\operatorname{Var}_q
\left[
h(X)
\frac{p(X)}{q(X)}
\right].
$$

This formula contains the central practical lesson of importance sampling. The method is useful only when the proposal controls the variability of the weighted integrand. It is therefore incorrect to describe importance sampling as a procedure that inherently reduces variance. It can reduce variance dramatically, but it can also increase it dramatically, and in pathological cases the variance of the estimator may not even exist.

The ideal proposal makes this point especially clear. If $h(x)\ge0$, then the zero-variance proposal is proportional to

$$
q^*(x)
\propto
h(x)p(x).
$$

Under this hypothetical proposal, the product

$$
h(x)\frac{p(x)}{q^*(x)}
$$

is constant, so the Monte Carlo variance vanishes. The construction is not directly usable because its normalizing constant is precisely the unknown integral $\mu$ that we are trying to estimate. Nevertheless, it gives a strong design principle: an efficient proposal should resemble the contribution of the integrand to the target expectation, rather than merely resemble the target density in a vague visual sense.

This distinction becomes important in rare-event simulation. If $h(x)=\mathbf 1\{x\in A\}$, then the integral receives contributions only from the rare event region $A$. A proposal that shifts mass toward $A$ can reduce variance by orders of magnitude, provided that the reweighting remains stable. Merely choosing a proposal with the same mode as the target does little if the event of interest lies far into the tail.

## Support and tail behaviour

The most fundamental validity condition is a support condition. If there exists a region where

$$
h(x)p(x)\neq0
$$

but

$$
q(x)=0,
$$

then that region can never be sampled under the proposal. No importance weight can compensate for observations that have zero probability of being generated. The estimator is then invalid for the intended expectation.

A less obvious but equally important problem occurs when the proposal technically has the correct support but has tails that are too light. Suppose $p(x)$ remains appreciable in regions where $q(x)$ becomes extremely small. The ratio

$$
\frac{p(x)}{q(x)}
$$

can then become enormous. Such observations may be rare under $q$, but when they do occur they carry huge weights and dominate the estimate. This produces unstable Monte Carlo behavior and can yield infinite variance even though the estimator is formally unbiased.

For this reason, proposals with heavier tails than the target are often safer than proposals with lighter tails, particularly when the integrand is sensitive to extreme values. The proposal does not need to match $p$ perfectly. It does need to cover the regions that matter, and it must do so without creating a weight distribution whose upper tail is uncontrollable.

## Self-normalized importance sampling

In Bayesian computation and related problems, the target density is often known only up to a normalizing constant. Suppose

$$
p(x)
=
\frac{\widetilde p(x)}{Z},
$$

where $Z$ is unknown. The ordinary importance weights cannot be evaluated exactly because they depend on $Z$. A common alternative is self-normalized importance sampling, using unnormalized weights

$$
w_i
=
\frac{\widetilde p(X_i)}{q(X_i)}
$$

and normalized weights

$$
\widetilde w_i
=
\frac{w_i}
{\sum_{j=1}^n w_j}.
$$

The expectation is then estimated by

$$
\widehat\mu_{SNIS}
=
\sum_{i=1}^n
\widetilde w_i h(X_i).
$$

Unlike the ordinary importance-sampling estimator, the self-normalized estimator is generally biased at finite sample size. Under standard regularity conditions it is nevertheless consistent, and in practice it is indispensable when the target is available only through an unnormalized density. The distinction between these two estimators matters because discussions of importance sampling often move between them without stating that their finite-sample properties differ.

The normalization also makes clear that the relevant issue is not simply whether the raw weights are numerically large. Multiplying every weight by the same positive constant leaves the normalized estimator unchanged. What matters is their relative concentration. If one or two observations receive nearly all of the normalized weight, then the nominal sample size can be very misleading.

## Effective sample size and weight diagnostics

A widely used diagnostic is the importance-sampling effective sample size,

$$
ESS
=
\frac{
\left(\sum_{i=1}^n w_i\right)^2
}{
\sum_{i=1}^n w_i^2
}.
$$

If all weights are equal, then $ESS=n$. If one observation dominates, the effective sample size approaches one. The diagnostic is useful because it converts weight concentration into an interpretable scale, but it should not be treated as a complete measure of estimator quality. Different functions $h$ can interact with the same weights differently, and a moderate ESS does not guarantee that tail behavior is well controlled.

Several other diagnostics can be useful. The coefficient of variation of the weights, the maximum normalized weight, Pareto-tail diagnostics, repeated simulation, and direct Monte Carlo standard-error estimates can all reveal instability that is not obvious from the point estimate alone. Importance sampling should be treated as an estimator with its own uncertainty, not as a deterministic correction applied after sampling.

Weight clipping is sometimes used when a small number of weights become extreme. Replacing weights above a threshold by a bounded value can greatly reduce variance, but this reduction is obtained by introducing bias. That trade-off can be entirely reasonable in applications such as off-policy evaluation or covariate-shift correction, yet it should be described explicitly rather than presented as a harmless numerical stabilization.

## Rare-event estimation

Rare events are among the clearest settings in which importance sampling can transform an infeasible calculation into a practical one. Suppose that

$$
p_A
=
P_p(X\in A)
$$

is extremely small. Ordinary Monte Carlo estimates it with

$$
\widehat p_A
=
\frac{1}{n}
\sum_{i=1}^n
\mathbf 1\{X_i\in A\},
\qquad
X_i\sim p.
$$

The variance is

$$
\frac{p_A(1-p_A)}{n}.
$$

When $p_A$ is tiny, the relative error can remain very large unless $n$ is enormous. Importance sampling replaces $p$ by a proposal that makes $A$ much more common:

$$
\widehat p_A^{IS}
=
\frac{1}{n}
\sum_{i=1}^n
\mathbf 1\{X_i\in A\}
\frac{p(X_i)}{q(X_i)},
\qquad
X_i\sim q.
$$

The method is successful when the proposal increases visits to the rare-event region while keeping the likelihood ratio sufficiently stable. This is the basis of importance sampling in reliability analysis, insurance, queueing, credit risk, and financial tail-risk simulation. The challenge is not the reweighting formula itself; it is constructing a proposal that approximates the conditional distribution of the system given the rare event without requiring knowledge of that conditional distribution in advance.

Adaptive importance sampling, cross-entropy methods, exponential tilting, and mixture proposals are all responses to that proposal-design problem. They differ in how they learn or construct $q$, but they share the same objective: concentrate simulation effort in informative regions without allowing the importance weights to become pathological.

## Importance sampling in modern machine learning

The same mathematics appears under several names in machine learning. Under covariate shift, for example, the training and deployment feature distributions differ while the conditional response model is assumed stable:

$$
p_{\text{train}}(x)
\neq
p_{\text{test}}(x),
\qquad
p(y\mid x)
\text{ unchanged}.
$$

Risk under the test distribution can then be represented as a weighted expectation over training data using the density ratio

$$
\frac{
p_{\text{test}}(x)
}{
p_{\text{train}}(x)
}.
$$

In off-policy reinforcement learning, trajectories or actions generated under one policy are reweighted to estimate performance under another. In Bayesian computation, importance sampling estimates posterior expectations or marginal likelihoods. In sequential Monte Carlo, importance weights are repeatedly updated as particles evolve through a sequence of target distributions.

These applications differ operationally, but they share the same weakness: density ratios can become unstable when the sampling distribution and target distribution are poorly aligned. In high-dimensional spaces, even apparently modest distributional differences can produce severe weight degeneracy. This is why naive importance weighting is often unreliable when extrapolation is substantial.

## A small implementation

The following implementation illustrates the ordinary estimator. It deliberately keeps proposal generation outside the function so that the distinction between sampling and reweighting remains explicit.

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
) -> tuple[float, float]:
    """Estimate an expectation and its Monte Carlo standard error.

    The input samples must already have been drawn from the proposal density.
    """
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("samples must be a one-dimensional array with at least two values")

    target = target_density(samples)
    proposal = proposal_density(samples)

    if target.shape != samples.shape or proposal.shape != samples.shape:
        raise ValueError("density functions must return arrays matching samples")
    if np.any(target < 0) or np.any(proposal <= 0):
        raise ValueError("densities must be non-negative and proposal strictly positive")

    weighted_values = function(samples) * (target / proposal)

    estimate = float(np.mean(weighted_values))
    standard_error = float(
        np.std(weighted_values, ddof=1) / np.sqrt(samples.size)
    )

    return estimate, standard_error
~~~

The code reports a Monte Carlo standard error because a simulation estimate without an assessment of simulation uncertainty is incomplete. In a serious analysis, one would also inspect the weight distribution and verify that the proposal provides adequate support in all regions relevant to the integrand.

## Conclusion

Importance sampling is best understood as a controlled change of measure. Its mathematical identity is exact, but its computational success is conditional. A proposal distribution can make rare-event or difficult integrals dramatically easier by reallocating simulation effort, yet the same mechanism can create unstable estimators when support is inadequate or weights are excessively variable. The method therefore illustrates a general principle in computational statistics: unbiasedness alone is not enough. The geometry of the sampling distribution and the variance of the estimator determine whether the calculation is useful in finite time.

The most important practical questions are consequently not whether importance sampling is theoretically valid in the abstract, but whether the chosen proposal covers the relevant support, whether its tails are sufficiently heavy, whether the resulting weights are stable, and whether the Monte Carlo uncertainty is acceptable for the scientific decision being made. Once those questions are treated as part of the statistical design rather than as implementation details, importance sampling becomes much more than a reweighting trick. It becomes a principled framework for allocating simulation effort where information is most valuable.

## References

- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.
