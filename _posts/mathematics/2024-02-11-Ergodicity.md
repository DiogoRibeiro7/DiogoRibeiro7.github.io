---
permalink: '/mathematics/Ergodicity/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-02-11'
excerpt: "Ergodicity is a precise property of a measure-preserving dynamical system. It is not merely a temporary regime, nor is chaos sufficient for ergodicity."
header:
  image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  og_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  overlay_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  twitter_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
keywords:
- Ergodicity
- Birkhoff ergodic theorem
- Measure preserving systems
- Mixing
- Dynamical systems
seo_description: "A rigorous introduction to ergodicity, invariant sets, Birkhoff's theorem, mixing, stationary processes, and the distinction between ergodicity and chaos."
seo_title: "Ergodicity: Time Averages, Invariant Sets, and Mixing"
seo_type: article
tags:
- Dynamical Systems
- Probability
- Mathematics
title: "Ergodicity: Time Averages, Invariant Sets, and Mixing"
toc: false
---

Ergodicity is a standard mathematical property of a measure-preserving dynamical system or stationary stochastic process. It is therefore misleading to reject the phrase "ergodic process" as a misconception.

The subtlety is different: ergodicity is always defined relative to a probability measure and dynamical transformation, and empirical finite-time behavior need not look ergodic even when the theoretical system is.

## Measure-preserving dynamics

Let $(\Omega,\mathcal F,\mu)$ be a probability space and let

$$
T:\Omega\to\Omega
$$

be measure preserving, meaning

$$
\mu(T^{-1}A)=\mu(A)
$$

for measurable sets $A$.

The system is ergodic if every invariant set satisfies

$$
T^{-1}A=A
$$

up to measure zero only when

$$
\mu(A)\in\{0,1\}.
$$

Intuitively, the state space cannot be decomposed into two nontrivial invariant regions.

## Birkhoff's ergodic theorem

For integrable observable $f$, the time average

$$
\frac{1}{n}\sum_{k=0}^{n-1}f(T^k\omega)
$$

converges almost surely to a $T$-invariant function.

If the system is ergodic, that limit is constant almost surely and equals

$$
\int f\,d\mu.
$$

This is the rigorous form of the slogan that time averages equal ensemble averages.

## Finite-time equality is not required

Ergodicity is an asymptotic property.

A finite trajectory can display long transients, metastability, or large sampling fluctuations while the underlying system remains ergodic.

Conversely, approximate agreement of one time average and one ensemble average over a finite window does not prove ergodicity.

## Stationary stochastic processes

For a stationary process $(X_t)$, ergodicity means shift-invariant events have probability zero or one.

Under ergodicity, sample averages can converge to population expectations:

$$
\frac{1}{n}\sum_{t=1}^n g(X_t)
\to
E[g(X_0)].
$$

This justifies estimating some population properties from one long realization.

## IID sequences

An IID sequence is stationary and ergodic under the shift.

The law of large numbers is closely related to this setting, but ergodicity is broader than independence.

Dependent Markov chains can be ergodic too.

## Markov-chain ergodicity

For MCMC, one wants the chain to have a unique target stationary distribution and to forget its initial state under suitable irreducibility and recurrence conditions.

Terminology varies across texts, but convergence to stationarity is a stronger operational requirement than simply writing down an invariant distribution.

## Ergodicity is not the same as mixing

Mixing is stronger.

A mixing system satisfies, roughly,

$$
\mu(T^{-n}A\cap B)
\to
\mu(A)\mu(B).
$$

Mixing implies ergodicity, but ergodicity does not imply mixing.

Thus correlation decay is not part of the definition of ergodicity.

## Chaos is not ergodicity

A positive Lyapunov exponent indicates sensitivity to initial conditions.

That does not by itself prove ergodicity.

Some chaotic systems have multiple invariant components; some ergodic systems are not chaotic in the usual sense.

Lyapunov exponents, Kolmogorov-Sinai entropy, mixing, and ergodicity describe different properties.

## Observables matter empirically, not definitionally

Different observables can converge at very different rates.

That can make a system appear ergodic for one measured quantity and not for another over finite data.

But this does not mean ergodicity itself is merely an observable-specific temporary regime. It means empirical diagnosis from finite trajectories is difficult.

## Non-ergodic systems

A mixture of two invariant components is a simple example.

If a trajectory starts in one component and never enters the other, its long-run time average reflects only that component, while the full ensemble average mixes both.

This is exactly what the invariant-set definition captures.

## Economics and multiplicative growth

Claims about "ergodicity economics" often concern whether ensemble-average wealth growth matches the time growth experienced by one trajectory.

That can be a useful distinction, especially for multiplicative dynamics, but it should not redefine ergodicity loosely.

The relevant stochastic process and observable should be stated explicitly.

## Empirical assessment

No finite dataset can prove mathematical ergodicity in full generality.

Useful diagnostics can include:

- multiple initial conditions
- autocorrelation and mixing-rate estimates
- recurrence properties
- comparison of long-run summaries
- invariant-density estimation
- transition structure

These provide evidence about the model, not a binary empirical proof.

## Conclusion

Ergodicity is a mathematically precise property of a dynamical system under a measure.

The main practical lesson is not to replace "ergodic process" with "ergodic regime." It is to distinguish:

$$
\text{theoretical ergodicity}
\neq
\text{finite-time appearance of equilibration}.
$$

That distinction preserves both the mathematics and the practical caution needed in real systems.

## References

- Walters, P. (1982). *An Introduction to Ergodic Theory*.
- Petersen, K. (1983). *Ergodic Theory*.
- Meyn, S. P., & Tweedie, R. L. (2009). *Markov Chains and Stochastic Stability*.
