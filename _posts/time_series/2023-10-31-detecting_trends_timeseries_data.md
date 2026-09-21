---
author_profile: false
categories:
- Time Series
classes: wide
date: '2023-10-31'
excerpt: "The Mann-Kendall test detects monotone association with time, but serial dependence, seasonality, ties, change points, and irregular sampling must be handled explicitly."
header:
  image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  og_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  overlay_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  twitter_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
keywords:
- Mann-Kendall test
- Trend detection
- Sen slope
- Autocorrelation
- Seasonal Mann-Kendall
- Time series
permalink: '/time-series/detecting_trends_timeseries_data/'
redirect_from:
- '/time-series analysis/detecting_trends_timeseries_data/'
- '/time series/detecting_trends_timeseries_data/'
seo_description: "A rigorous guide to the Mann-Kendall trend test, including serial dependence, seasonality, ties, Sen slopes, change points, and irregular sampling."
seo_title: "Mann-Kendall Trend Test: Assumptions and Pitfalls"
seo_type: article
tags:
- Time Series
- Climate and Environment
- Statistics
title: "Mann-Kendall Trend Test: Assumptions and Pitfalls"
---

The Mann-Kendall test is a rank-based test for monotone association between an ordered index and observations. It is popular because it does not require Gaussian marginal distributions, but "non-parametric" does not mean assumption-free.

The most important practical issue is serial dependence. Positive autocorrelation can make the usual variance formula too small, inflating false-positive rates.

## The statistic

For observations $x_1,\ldots,x_n$, define

$$
S=\sum_{i<j}\operatorname{sign}(x_j-x_i).
$$

Large positive S means later observations tend to exceed earlier observations. Large negative S means the opposite.

With no ties,

$$
\operatorname{Var}(S)=\frac{n(n-1)(2n+5)}{18}.
$$

Ties require a correction because equal observations contribute zero to S but alter its null variance.

## What the null hypothesis means

The common large-sample test is derived under an independence structure in which the observations have no systematic monotone ordering over time.

It should not be summarized as "the data are randomly ordered" without qualification. Time-series dependence can exist under no deterministic trend and still invalidate the ordinary variance calculation.

## Autocorrelation is not a minor detail

Environmental, hydrological, economic, and sensor series often have persistence.

If successive observations are positively correlated, the effective information content is smaller than n independent observations. Applying the textbook Mann-Kendall variance can then produce overly small p-values.

Possible approaches include block bootstrap methods, modified variance estimators, prewhitening strategies, or explicit time-series models. Each has assumptions and trade-offs; prewhitening can also distort the trend signal if done mechanically.

## Seasonality

A monotone long-term trend can coexist with strong seasonal structure. Pooling all months or seasons can confound the rank comparison.

Seasonal Mann-Kendall procedures compare observations within the same season before combining evidence across seasons. This is appropriate when the seasonal pattern is stable enough for those strata to be meaningful.

## Irregular spacing

The ordinary Mann-Kendall statistic uses order, not elapsed time. Two observations one day apart and two observations five years apart are both just ordered pairs.

That may be acceptable if the scientific question is monotone ordering, but it can be inadequate when the rate of change per unit time matters.

Regression on actual time, generalized additive models, or state-space models may be more appropriate for irregularly spaced data.

## Trend magnitude: use an effect estimate

A p-value does not tell us how large the trend is.

Sen's slope estimates a typical pairwise slope:

$$
\hat\beta_{Sen}=\operatorname{median}_{i<j}\frac{x_j-x_i}{t_j-t_i}.
$$

Reporting a slope and uncertainty is usually more informative than reporting only whether the Mann-Kendall test rejected.

## Monotone trend is not the only temporal structure

A series can have a strong U-shape, step change, structural break, or oscillation while having little net monotone trend.

Conversely, a significant Mann-Kendall result does not imply a linear trend.

Before testing, plot the series and consider whether the scientific hypothesis is actually monotonicity.

## Change points

If the data contain a level shift, the Mann-Kendall test may detect a trend even though the process is better described as a structural break.

Change-point models, segmented regression, or intervention analysis may then be more interpretable.

## Multiple testing

Trend studies often test many stations, variables, months, or regions. Running one test per series creates a multiple-testing problem.

False discovery rate or family-wise error control may be needed depending on the inferential goal and dependence structure.

## A useful workflow

1. Plot the series and sampling times.
2. Define whether the target is monotonicity, slope, or structural change.
3. Examine seasonality and serial dependence.
4. Use an appropriate variance or resampling scheme.
5. Report a magnitude estimate such as Sen's slope.
6. Adjust for multiplicity when many series are tested.
7. Check whether a more explicit time-series model answers the question better.

## Conclusion

The Mann-Kendall test is useful when the scientific question concerns monotone ordering and the dependence structure is handled correctly.

Its main advantage is robustness to marginal distribution shape. Its main danger is the belief that rank-based inference removes the need to model time.

## References

- Mann, H. B. (1945). Nonparametric Tests Against Trend.
- Kendall, M. G. (1975). *Rank Correlation Methods*.
- Sen, P. K. (1968). Estimates of the Regression Coefficient Based on Kendall's Tau.
- Yue, S., Pilon, P., Phinney, B., & Cavadias, G. (2002). The influence of autocorrelation on the ability to detect trend in hydrological series.
