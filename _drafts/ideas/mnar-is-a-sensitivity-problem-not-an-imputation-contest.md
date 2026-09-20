---
author_profile: false
categories:
- Statistics
classes: wide
title: 'MNAR Is a Sensitivity Problem, Not an Imputation Contest'
excerpt: 'When missingness depends on unobserved values, the missing-data mechanism is not identified from observed data alone. Better prediction of missing values does not remove that sensitivity.'
keywords:
- missing not at random
- MNAR
- pattern mixture models
- selection models
- tipping point analysis
seo_title: 'MNAR Is a Sensitivity Problem, Not an Imputation Contest'
seo_description: 'A structured draft on non-ignorable missingness, selection models, pattern-mixture models, delta adjustments, tipping-point analysis, and partial identification.'
seo_type: article
summary: 'A planned article showing why MNAR assumptions cannot generally be validated from observed data and should therefore be handled through explicit sensitivity parameters rather than claims of uniquely correct imputation.'
tags:
- Missing Data
- Sensitivity Analysis
- Partial Identification
- Statistical Inference
why_this_exists: 'Missing-data discussions often focus on which imputation algorithm predicts missing values best. Under MNAR, the central scientific problem is instead that the unseen distribution depends on assumptions not identified by the observed sample.'
evidence: 'Exact binary and Gaussian examples, selection and pattern-mixture parameterisations, delta-adjustment sensitivity curves and tipping-point calculations.'
methodology: 'Build two full-data distributions with identical observed-data likelihoods but different missing-value distributions, then show how explicit sensitivity parameters change the target estimand.'
---

<!--
Development contract
Question: What can observed data identify when missingness depends on values that were not observed?
Claim: Under MNAR, the observed-data distribution generally does not identify the missing-data mechanism, so inference requires unverifiable assumptions that should be exposed through sensitivity analysis.
Counterclaim: External information, refreshment samples, validation subsamples or strong scientific structure can identify parts of the missingness process.
Evidence object: One non-identification construction, one selection-model parameterisation, one pattern-mixture delta adjustment and one tipping-point plot.
Failure case: Comparing imputation RMSE on artificially masked MAR data and calling the winner valid for MNAR, or treating the missingness mechanism as empirically testable from observed data alone.
Reader payoff: Replace algorithm shopping with a transparent sensitivity analysis tied to the estimand.
Exclusions: Repeating the existing draft on imputation accuracy versus inferential validity.
-->

## Mathematical spine

Let $Y$ be the outcome and $R$ indicate observation. Under MNAR,

$$
P(R=1\mid Y,X)
$$

depends on the unobserved value of $Y$ after conditioning on observed $X$.

Develop a selection model,

$$
\operatorname{logit}
P(R=1\mid Y,X)
=
\alpha+\beta^\top X+\gamma Y,
$$

where $\gamma$ is generally not identified from observed data alone.

Contrast this with a pattern-mixture sensitivity model,

$$
\mathbb E[Y\mid R=0,X]
=
\mathbb E[Y\mid R=1,X]+\Delta.
$$

Treat $\Delta$ as a sensitivity parameter and identify the tipping point at which the substantive conclusion changes.

## Worked example

Construct two full-data processes with the same observed-data distribution and missingness pattern but different means among missing cases.

## Reproducibility plan

Reproduce the non-identification construction and sensitivity curve exactly, including a table marking assumptions as identified, partially identified or sensitivity-driven.

## Sources to develop

Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*.

Daniels, M. J., & Hogan, J. W. (2008). *Missing Data in Longitudinal Studies*.

Molenberghs, G., & Kenward, M. G. (2007). *Missing Data in Clinical Studies*.
