---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-01-13'
excerpt: Statistical test selection should begin with the estimand, study design, sampling structure, and error criterion. Flowcharts fail when they reduce those decisions to data type and normality.
header:
  image: /assets/images/headers/photo-data-science-dashboard.jpg
  og_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-dashboard.jpg
  twitter_image: /assets/images/headers/photo-data-science-dashboard.jpg
keywords:
- statistical test selection
- estimands
- study design
- hypothesis testing
- robust inference
permalink: '/data-science/rethinking_statistical_test_selection_why_diagrams_failing_us/'
redirect_from:
- '/data science/statistics/rethinking_statistical_test_selection_why_diagrams_failing_us/'
- '/data science/rethinking_statistical_test_selection_why_diagrams_failing_us/'
seo_description: A principled approach to statistical test selection based on estimands, design, dependence, sampling, and error control rather than mechanical normality flowcharts.
seo_title: 'Choosing Statistical Tests: Start with the Estimand'
seo_type: article
summary: A replacement for statistical-test flowcharts that organizes test choice around the scientific quantity being estimated, the study design, and the assumptions required for valid inference.
tags:
- Statistical Modeling
- Data Science
- Hypothesis Testing
title: 'Choosing Statistical Tests: Start with the Estimand'
---

Statistical test-selection diagrams fail for a simple reason.

They usually begin with questions such as:

- Is the outcome continuous?
- Is the sample normal?
- Is $n>30$?
- Are there two groups or three?

Those questions can matter.

They are not the first questions.

A defensible analysis starts with

$$
\boxed{
\text{scientific question}
\rightarrow
\text{estimand}
\rightarrow
\text{design}
\rightarrow
\text{sampling structure}
\rightarrow
\text{estimator/test}.
}
$$

The named test comes last.

## Start with the estimand

Suppose there are two groups.

Possible targets include:

- mean difference;
- median difference;
- risk difference;
- risk ratio;
- odds ratio;
- hazard ratio;
- stochastic ordering;
- quantile difference;
- full-distribution equality.

These are different estimands.

No normality test can decide which one the scientific question requires.

## Means

If the target is

$$
\Delta_\mu
=
E[Y\mid G=1]
-
E[Y\mid G=0],
$$

use a method for means.

For independent groups, Welch's t procedure is often a sensible default for an unadjusted mean contrast because it does not require equal variances.

That does not make it universally optimal.

Clustered observations, repeated measures, extreme tails, small samples, survey weights, or covariate adjustment can require another approach.

The right method preserves the mean estimand.

## Medians and quantiles

If the target is a median difference,

$$
Q_{0.5}(Y\mid G=1)
-
Q_{0.5}(Y\mid G=0),
$$

quantile regression gives a direct model for that quantity.

The Mann-Whitney test does not generally test equality of medians.

Its null is more naturally expressed through equality of rank distributions or pairwise ordering probabilities.

Under additional equal-shape assumptions, it can acquire a location interpretation.

Those assumptions should be stated rather than silently imported.

## Binary outcomes

For binary outcomes, meaningful targets include

$$
p_1-p_0,
$$

$$
\frac{p_1}{p_0},
$$

or

$$
\frac{
p_1/(1-p_1)
}{
p_0/(1-p_0)
}.
$$

These are the risk difference, risk ratio, and odds ratio.

Logistic regression targets log odds.

A log-binomial or modified Poisson approach can target risk ratios.

A linear probability model targets risk differences directly, with suitable robust inference.

Choosing logistic regression simply because the outcome is binary does not decide which effect measure is scientifically preferred.

## Counts and rates

A count $Y$ observed over exposure time $T$ may be modeled through

$$
Y
\sim
\operatorname{Poisson}(\mu),
$$

with

$$
\log\mu
=
X^\top\beta
+
\log T.
$$

The term

$$
\log T
$$

acts as an offset for exposure.

Overdispersion may motivate negative-binomial or quasi-likelihood methods.

Again, the model follows the data-generating structure.

## Dependence comes before distribution shape

Ten thousand observations from ten subjects are not equivalent to ten thousand independent subjects.

Dependence can arise from:

- repeated measures;
- families;
- hospitals;
- schools;
- spatial neighborhoods;
- time series;
- matched designs.

Ignoring dependence can make standard errors badly wrong.

No amount of marginal normality checking repairs that.

## Transformations are modeling choices

A log transformation is not inherently bad.

Neither is it automatically good.

If

$$
\log Y
=
X^\top\beta+\varepsilon
$$

is scientifically meaningful, then the transformation defines the scale of the estimand.

The problem is not “transforming data.”

The problem is transforming without asking what quantity the transformed model estimates.

A transformation should be justified through the model and interpretation.

## There is no universal $n>30$ rule

The Central Limit Theorem is asymptotic.

Approximation quality depends on:

- skewness;
- tail weight;
- dependence;
- statistic;
- sample balance;
- leverage.

For some distributions, $n=20$ is enough for a useful approximation.

For others, $n=10{,}000$ may still leave problematic tail behavior.

Sample-size rules cannot replace diagnostics.

## Parametric and nonparametric are not quality rankings

“Nonparametric” does not mean robust, modern, or assumption-free.

“Parametric” does not mean fragile or obsolete.

A parametric model can be highly robust for one target.

A rank test can answer the wrong question perfectly.

The meaningful distinction is what assumptions connect the data to the estimand.

## Permutation tests

A permutation test is valid when the permutation scheme represents the null exchangeability structure.

For a two-group randomized experiment, labels may be permutable under the randomization design.

For paired data, unrestricted permutation across all observations is wrong.

The design determines the allowed permutations.

Permutation is not a magic assumption-free wrapper around any statistic.

## Bootstrap

Bootstrap methods approximate the sampling distribution by resampling from an empirical estimate of the data-generating process.

The resampling unit must match the dependence structure.

Examples include:

- ordinary bootstrap for independent observations;
- cluster bootstrap for clustered data;
- block bootstrap for time series.

Resampling individual rows from a dependent dataset can destroy the very structure the inference depends on.

## Multiple predictors and interactions

A named two-sample test becomes insufficient when the scientific question includes:

- confounding adjustment;
- interactions;
- nonlinear predictors;
- multiple groups;
- repeated measures.

Regression modeling is often useful because it makes the estimand conditional on explicit covariates.

But adding a regression formula does not automatically create causal interpretation.

Design and identification remain separate.

## A practical framework

Before choosing a test, write down:

1. **Population:** Who or what is the target?
2. **Outcome:** What random quantity is observed?
3. **Exposure or groups:** How were they assigned?
4. **Estimand:** Mean, median, risk, odds, rate, quantile, survival?
5. **Dependence:** Independent, paired, clustered, longitudinal?
6. **Censoring or missingness:** How can observations disappear?
7. **Model:** What conditional structure is plausible?
8. **Error criterion:** Confidence interval, FWER, FDR, prediction loss?
9. **Sensitivity:** Which assumptions are not empirically testable?

Only then choose the named procedure.

## Conclusion

The problem with test-selection diagrams is not that the tests listed in them are old.

It is that the diagrams usually hide the estimand and design.

A better rule is:

$$
\boxed{
\text{Do not choose a test from the shape of the spreadsheet.}
}
$$

Choose an estimand from the scientific question, then choose a method whose assumptions identify and estimate that quantity under the actual study design.

## References

- Lehmann, E. L., & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3rd ed.). Springer.
- Greenland, S., Senn, S. J., Rothman, K. J., et al. (2016). Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations. *European Journal of Epidemiology*, 31, 337–350.
- Lumley, T., Diehr, P., Emerson, S., & Chen, L. (2002). The importance of the normality assumption in large public health data sets. *Annual Review of Public Health*, 23, 151–169.
