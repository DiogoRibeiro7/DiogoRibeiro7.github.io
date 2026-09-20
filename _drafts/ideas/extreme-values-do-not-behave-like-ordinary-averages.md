---
author_profile: false
categories:
- Statistics
classes: wide
title: 'Extreme Values Do Not Behave Like Ordinary Averages'
excerpt: 'Extreme-value theory models the tail through limits for maxima and threshold exceedances rather than by extending Gaussian intuition into regions where it is least reliable.'
keywords:
- extreme value theory
- generalized extreme value distribution
- peaks over threshold
- generalized Pareto distribution
- return levels
seo_title: 'Extreme Values Do Not Behave Like Ordinary Averages'
seo_description: 'A mathematical draft on block maxima, peaks over threshold, return levels, tail index estimation, threshold choice, and why ordinary regression intuition breaks in the extremes.'
seo_type: article
summary: 'A planned article on the asymptotic structure of extremes, with explicit derivations for GEV and GPD models and a worked comparison showing why Gaussian tail extrapolation can fail catastrophically.'
tags:
- Extreme Value Theory
- Tail Risk
- Statistical Inference
- Risk
why_this_exists: 'Tail events are often analysed by fitting ordinary distributions to the centre and extrapolating. Extreme-value theory asks a different asymptotic question and therefore produces a different model class.'
evidence: 'Classical EVT results, Pickands-Balkema-de Haan threshold theory, return-level calculations, and synthetic heavy-tail examples.'
methodology: 'Develop block-maxima and peaks-over-threshold approaches from limiting arguments, then compare tail extrapolation under Gaussian, GEV and GPD models on controlled examples.'
---

<!--
Development contract
Question: Why should maxima and threshold exceedances be modelled differently from ordinary observations?
Claim: The limiting distributions governing extremes are structurally different from central-limit approximations, so tail inference should be built around maxima or exceedances rather than ordinary mean-variance summaries.
Counterclaim: EVT is not automatically superior for every rare-event problem. Threshold and block choices discard information and can increase variance when data are limited.
Evidence object: Exact return-level calculations under light and heavy tails, one threshold-stability plot, and one synthetic example where central modelling underestimates rare-event risk.
Failure case: Choosing a threshold after inspecting the desired answer, treating a return level as a guaranteed recurrence time, or estimating far beyond the information content of the tail sample.
Reader payoff: Know when to use GEV versus GPD methods and what diagnostics must accompany a tail-risk claim.
Exclusions: A catalogue of every tail-index estimator or a finance-only treatment of extreme risk.
-->

## Mathematical spine

For block maxima, develop the limiting statement

$$
\frac{M_n-b_n}{a_n}
\Rightarrow
G_\xi,
$$

where $M_n=\max(X_1,\ldots,X_n)$. Explain how the shape parameter $\xi$ separates bounded, exponential-type and heavy-tailed regimes.

For peaks over threshold, model

$$
Y=X-u\mid X>u
$$

with the generalized Pareto family,

$$
P(Y\le y)
=
1-
\left(
1+\xi\frac{y}{\beta}
\right)^{-1/\xi}.
$$

The worked example should compare a Gaussian model and a heavy-tailed model that agree near the centre but imply orders-of-magnitude differences for very high quantiles.

## Questions to resolve

How should block size and threshold be selected? What information is lost by annual maxima? How should clustering of extremes be handled? What does a 100-year return level mean probabilistically, and how uncertain is it?

## Reproducibility plan

Create synthetic light-tail and heavy-tail samples with similar central variance. Reproduce return-level curves, threshold-stability diagnostics and empirical exceedance probabilities in the reproducibility repository.

## Sources to develop

Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*.

Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*.

Pickands, J. (1975). Statistical inference using extreme order statistics.

Balkema, A. A., & de Haan, L. (1974). Residual life time at great age.
