---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-04-01'
excerpt: The Friedman test compares within-block ranks across repeated conditions. It is useful for blocked or repeated-measures designs when a rank-based estimand is appropriate, not simply whenever normality fails.
header:
  image: /assets/images/headers/photo-data-science-openalex.jpg
  og_image: /assets/images/headers/photo-data-science-openalex.jpg
  overlay_image: /assets/images/headers/photo-data-science-openalex.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-openalex.jpg
  twitter_image: /assets/images/headers/photo-data-science-openalex.jpg
keywords:
- Friedman test
- repeated measures
- blocked designs
- rank tests
- Kendall's W
permalink: '/data-science/friedman_test/'
redirect_from:
- '/data analysis/friedman_test/'
- '/data science/friedman_test/'
seo_description: The Friedman test explained as a blocked rank test, including its null hypothesis, tie correction, effect size, exact alternatives, and appropriate post-hoc comparisons.
seo_title: 'Friedman Test: A Rank Test for Blocked Repeated Measures'
seo_type: article
summary: A rigorous guide to the Friedman test that explains within-block ranking, what the null means, when the chi-square approximation is valid, and how to perform follow-up comparisons.
tags:
- Nonparametric Methods
- Hypothesis Testing
- Repeated Measures
title: 'Friedman Test: A Rank Test for Blocked Repeated Measures'
---

The Friedman test is commonly called the nonparametric alternative to repeated-measures ANOVA. That description is useful but incomplete. The procedures do not estimate exactly the same object. Repeated-measures ANOVA is mean-based. Friedman discards the within-block numerical distances and analyzes ranks. The correct question is therefore not

> Did the normality test fail?

It is

> Is a within-block rank comparison a scientifically appropriate summary of these repeated conditions?

## Design

Suppose $n$ blocks or subjects are each observed under $k$ conditions. Let

$$
Y_{ij}
$$

be the response for subject $i$ under condition $j$. The blocks must be independent of one another. Within a block, observations are deliberately related because the same subject or matched unit receives every condition. A complete-block design requires all $k$ conditions within each analyzed block.

## Within-block ranks

For each block $i$, replace

$$
Y_{i1},\ldots,Y_{ik}
$$

with ranks

$$
R_{i1},\ldots,R_{ik}.
$$

The smallest value receives rank 1, the largest rank $k$, with average ranks for ties. Let

$$
R_{\cdot j}
=
\sum_{i=1}^{n}
R_{ij}
$$

be the rank sum for condition $j$. If conditions are exchangeable under the null, the expected rank sum is the same for every condition.

## Test statistic

Without ties, the Friedman statistic is

$$
Q
=
\frac{12}
{nk(k+1)}
\sum_{j=1}^{k}
R_{\cdot j}^2
-
3n(k+1).
$$

For sufficiently large $n$,

$$
Q
\approx
\chi^2_{k-1}
$$

under the null. For small samples, exact or permutation calibration is preferable. Ties require a correction because they reduce rank variability. Software should handle that explicitly.

## What the null hypothesis means

The Friedman test is often described as testing equal medians. That is too narrow. The randomization-style null is that condition labels are exchangeable within blocks. Under stronger location-shift assumptions, a median or location interpretation may be reasonable. In general, Friedman detects systematic within-block rank differences. It does not isolate which distributional feature changed.

## What blocking accomplishes

Subjects can have very different overall response levels. Ranking within each subject removes those baseline level differences. Suppose one assessor scores every algorithm high and another scores every algorithm low. Friedman is interested in their **relative ordering** of algorithms, not the absolute scale difference between assessors. This is the reason the blocking structure matters.

## Missing cells are a design problem

If a subject is missing one condition, the simple Friedman design is incomplete. Deleting only the missing cell is not valid because ranking requires the full within-block set. Options include:

- complete-case blocks;
- models for incomplete repeated measures;
- mixed-effects approaches;
- multiple imputation under a defensible missing-data model.

The right response depends on why the value is missing.

## Worked example

~~~python
from __future__ import annotations

import numpy as np
from scipy import stats

scores: np.ndarray = np.array(
    [
        [7, 8, 5],
        [6, 9, 6],
        [8, 8, 4],
        [5, 7, 5],
        [7, 9, 6],
        [6, 8, 3],
        [8, 9, 5],
        [7, 7, 4],
    ],
    dtype=float,
)

result = stats.friedmanchisquare(
    *scores.T
)

ranks = np.apply_along_axis(
    stats.rankdata,
    1,
    scores,
)

mean_ranks = ranks.mean(axis=0)

n, k = scores.shape

kendalls_w: float = float(
    result.statistic
    / (n * (k - 1))
)

print(result)
print(mean_ranks)
print(f"Kendall's W = {kendalls_w:.3f}")
~~~

The omnibus test says whether systematic rank differences exist. The mean ranks show direction.

## Kendall's W

For complete rankings without complications, a common effect-size form is

$$
W
=
\frac{Q}
{n(k-1)}.
$$

Values range from 0 to 1. In a rating-by-assessor setting, it can be interpreted as concordance in the rank structure. In a repeated-treatment experiment, it is better described cautiously as a standardized Friedman effect-size measure rather than automatically as “agreement among raters.” Context determines the interpretation.

## Post-hoc comparisons

A significant Friedman test does not identify which conditions differ. Possible follow-ups include:

- pairwise Wilcoxon signed-rank tests;
- Nemenyi-type rank comparisons;
- model-based repeated-measures contrasts.

Multiplicity must be controlled across the chosen family of pairwise tests. Holm adjustment is a common FWER-controlling choice. The post-hoc method should match the estimand and data structure. There is no requirement that one particular named post-hoc test must follow Friedman.

## Wilcoxon signed-rank caveat

The paired Wilcoxon signed-rank test assumes more structure than a simple sign comparison. Its usual location-shift interpretation relies on symmetry of paired differences. If that assumption is not plausible, a sign test or permutation procedure may better match the target. Calling every pairwise rank method assumption-free recreates the problem that led to misuse of Friedman in the first place.

## Repeated-measures ANOVA versus Friedman

Repeated-measures ANOVA uses numerical distances and models means. Friedman uses ranks within blocks. If the scientific target is a mean difference, a linear mixed model or repeated-measures mean model may remain preferable even with some non-normality. If the response is genuinely ordinal or the rank estimand is primary, Friedman can be natural.

The decision should follow the estimand, not a Shapiro-Wilk p-value.

## More complex repeated designs

Friedman handles one repeated factor in a complete-block structure. It does not naturally handle:

- multiple repeated factors;
- interactions;
- time-varying covariates;
- unequal observation schedules;
- incomplete blocks.

Mixed models, generalized estimating equations, ordinal mixed models, or rank-based factorial methods may be more appropriate depending on the target. The limitation is structural, not evidence that the Friedman test is obsolete.

## Reporting

A useful report includes:

- number of blocks $n$;
- number of conditions $k$;
- Friedman statistic;
- degrees of freedom;
- p-value;
- mean or median ranks as appropriate;
- effect size such as $W$;
- tie handling;
- post-hoc method and multiplicity adjustment.

The block structure should be explicit.

## Conclusion

The Friedman test is a rank-based procedure for complete blocked or repeated-measures designs. Its strength is that it removes between-block level differences and compares relative ordering within each block. Its limitation is the same feature:

$$
\boxed{
\text{rank information replaces metric information}.
}
$$

Use it when that is the quantity you want to analyze, not merely because raw observations are non-normal.

## References

- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
