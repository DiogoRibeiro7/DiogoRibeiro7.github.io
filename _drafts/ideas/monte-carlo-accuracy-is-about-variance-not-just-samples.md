---
author_profile: false
categories:
- Mathematics
classes: wide
title: 'Monte Carlo Accuracy Is About Variance, Not Just Samples'
excerpt: 'The familiar square-root convergence rate does not say that all Monte Carlo estimators are equally useful. Variance reduction can change computational cost by orders of magnitude without changing the target expectation.'
keywords:
- Monte Carlo
- variance reduction
- importance sampling
- control variates
- antithetic variates
seo_title: 'Monte Carlo Accuracy Is About Variance, Not Just Samples'
seo_description: 'A mathematical draft on Monte Carlo error, importance sampling, control variates, antithetic sampling, stratification, and rare-event simulation.'
seo_type: article
summary: 'A planned article deriving Monte Carlo error from the central limit theorem and showing how estimator design matters more than brute-force sample growth in expensive or rare-event problems.'
tags:
- Monte Carlo
- Simulation
- Variance Reduction
- Numerical Methods
why_this_exists: 'Monte Carlo is often taught as draw many samples and average. That hides the real design problem: constructing an unbiased or controlled-bias estimator with manageable variance.'
evidence: 'Exact variance calculations, synthetic integration problems, and a rare-event example comparing naive and importance-sampling estimators.'
methodology: 'Start from the Monte Carlo estimator and CLT, derive cost scaling, then show analytically and numerically how variance-reduction methods change estimator efficiency.'
---

<!--
Development contract
Question: Why can two unbiased Monte Carlo estimators of the same quantity have radically different computational value?
Claim: Monte Carlo error depends on estimator variance as well as sample size, so good simulation design spends computation where it reduces variance most.
Counterclaim: Sophisticated variance-reduction schemes can be fragile under poor proposal choices or high-dimensional mismatch.
Evidence object: One ordinary integral, one exact control-variate calculation, and one rare-event probability where naive Monte Carlo is practically useless.
Failure case: Reporting only the number of simulations, ignoring effective sample size, or using importance weights with infinite or extreme variance.
Reader payoff: Recognise when brute-force simulation is wasteful and choose a variance-reduction strategy appropriate to the structure of the problem.
Exclusions: A general MCMC survey.
-->

## Mathematical spine

For

$$
\mu=\mathbb E[f(X)],
$$

use

$$
\hat\mu_N
=
\frac1N\sum_{i=1}^N f(X_i),
\qquad
\operatorname{Var}(\hat\mu_N)
=
\frac{\sigma_f^2}{N}.
$$

The computational consequence should be explicit: reducing standard error by a factor of ten through sample size alone requires about one hundred times as many draws.

For a control variate $Z$ with known mean, derive

$$
\hat\mu_c
=
\bar f-c(\bar Z-\mathbb E[Z]),
$$

with

$$
c^\star
=
\frac{\operatorname{Cov}(f,Z)}
{\operatorname{Var}(Z)}.
$$

Then derive importance sampling,

$$
\mu
=
\mathbb E_q
\left[
f(X)\frac{p(X)}{q(X)}
\right],
$$

and treat proposal quality as a variance problem.

## Worked examples

Use one control-variate problem with exact variance reduction. Add a rare Gaussian tail probability such as $P(Z>6)$ to show why naive indicator simulation is wasteful, then construct a shifted proposal.

## Reproducibility plan

Compare empirical RMSE against theoretical variance for naive Monte Carlo, control variates and importance sampling over repeated experiments.

## Sources to develop

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.

Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*.

Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*.
