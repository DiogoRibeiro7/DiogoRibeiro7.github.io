---
permalink: '/mathematics/Markov_Chain/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-17'
header:
  image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  og_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  twitter_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
keywords:
- Markov chains
- Transition matrix
- Stationary distribution
- Hidden Markov models
- Stochastic processes
redirect_from:
- '/mathematics/statistics/data science/machine learning/Markov_Chain/'
seo_description: "Markov chains explained through transition matrices, communicating classes, stationary distributions, recurrence, mixing, and Hidden Markov Models."
seo_title: "Markov Chains: Transition, Recurrence, and Stationarity"
seo_type: article
tags:
- Stochastic Processes
- Probability
title: "Markov Chains: Transition, Recurrence, and Stationarity"
---

A Markov chain is a stochastic process whose next-state distribution depends on the present state, not on the full past.

For discrete time,

$$
P(X_{t+1}=j\mid X_t=i,X_{t-1},\ldots,X_0)
=
P(X_{t+1}=j\mid X_t=i).
$$

This is a conditional-independence statement.

It does not mean the process has no history in an ordinary-language sense.

## Transition matrix

For a finite homogeneous chain,

$$
P_{ij}
=
P(X_{t+1}=j\mid X_t=i).
$$

Each row sums to one.

If the row vector $\pi_t$ contains the state probabilities at time $t$, then

$$
\pi_{t+1}
=
\pi_t P.
$$

After $n$ steps,

$$
\pi_{t+n}
=
\pi_t P^n.
$$

## Stationary distribution

A stationary distribution satisfies

$$
\pi=\pi P.
$$

Starting the chain from $\pi$ leaves its marginal distribution unchanged over time.

A stationary distribution need not imply rapid convergence to it.

## Irreducibility

A finite chain is irreducible if every state can be reached from every other state.

This rules out multiple closed communicating classes.

## Periodicity

A state has period determined by the greatest common divisor of possible return times.

An irreducible chain can have a unique stationary distribution yet oscillate periodically rather than converge from every starting state.

Aperiodicity prevents this type of cycle.

## Finite-state convergence

For a finite irreducible aperiodic chain,

$$
P^n(i,\cdot)
\to
\pi
$$

as $n\to\infty$.

This is the basis for long-run equilibrium claims.

The convergence rate depends on the spectrum and geometry of the transition matrix.

## Recurrence and transience

A recurrent state is revisited with probability one when starting there.

A transient state may never be revisited.

In infinite state spaces these distinctions are essential because irreducibility alone does not guarantee a stationary probability distribution.

## Detailed balance

A distribution $\pi$ satisfies detailed balance if

$$
\pi_iP_{ij}
=
\pi_jP_{ji}.
$$

Detailed balance implies stationarity.

It is sufficient, not necessary.

This distinction matters in MCMC, where reversible chains are common because detailed balance is easy to verify.

## Hitting times

For target set $A$, the hitting time is

$$
\tau_A
=
\inf\{t\ge0:X_t\in A\}.
$$

Expected hitting times answer questions such as time to failure, absorption, or queue overflow.

These can often be computed from linear systems rather than simulation alone.

## Absorbing chains

A state $i$ is absorbing if

$$
P_{ii}=1.
$$

Absorbing chains are useful for reliability, default, disease progression, and customer-lifecycle models.

The fundamental matrix can characterize expected visits before absorption.

## Hidden Markov models

An HMM adds an unobserved Markov state sequence $Z_t$ and observations $Y_t$ emitted conditionally on the hidden state.

A standard factorization is

$$
P(z_{1:T},y_{1:T})
=
P(z_1)
\prod_{t=2}^T
P(z_t\mid z_{t-1})
\prod_{t=1}^T
P(y_t\mid z_t).
$$

Inference tasks include filtering, smoothing, decoding, and parameter estimation.

## Time-inhomogeneous systems

Real systems often change by time of day, season, policy, or covariates.

Then

$$
P_t
$$

depends on $t$.

A single stationary transition matrix is no longer an adequate model.

The parking-lot example in the old article implicitly depended on time of day and day of week, so a homogeneous chain would be inconsistent unless those variables were included in the state.

## State design matters

The Markov property can often be restored by expanding the state.

If tomorrow depends on the last two observations, define the state to include both.

Thus “is the system Markov?” depends partly on how state is represented.

## Conclusion

A Markov model is useful when the present state captures the information needed for future evolution.

The central questions are:

- what is the state?
- is the transition law time-homogeneous?
- are there multiple communicating classes?
- does a stationary distribution exist?
- does the chain converge to it?
- how fast?

Those questions are more informative than describing Markov chains simply as memoryless predictors.

## References

- Norris, J. R. (1997). *Markov Chains*.
- Levin, D. A., Peres, Y., & Wilmer, E. L. (2009). *Markov Chains and Mixing Times*.
- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.
