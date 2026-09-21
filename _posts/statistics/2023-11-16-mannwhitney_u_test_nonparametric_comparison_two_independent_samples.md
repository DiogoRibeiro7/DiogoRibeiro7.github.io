---
permalink: '/statistics/mannwhitney_u_test_nonparametric_comparison_two_independent_samples/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-11-16'
excerpt: "The Mann-Whitney U test compares pairwise ordering between two independent distributions. It is not automatically a test of medians or a fallback whenever normality fails."
header:
  image: /assets/images/headers/photo-statistics-regression-errors.jpg
  og_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-regression-errors.jpg
  twitter_image: /assets/images/headers/photo-statistics-regression-errors.jpg
keywords:
- Mann-Whitney U test
- Wilcoxon rank-sum
- Probability of superiority
- Rank-biserial correlation
- Nonparametric statistics
redirect_from:
- '/non-parametric tests/mannwhitney_u_test_nonparametric_comparison_two_independent_samples/'
seo_description: "A rigorous guide to the Mann-Whitney U test, its pairwise-ordering estimand, assumptions, ties, effect sizes, and relationship to the two-sample t-test."
seo_title: "Mann-Whitney U Test: What It Actually Tests"
seo_type: article
tags:
- Hypothesis Testing
- Nonparametric Methods
- Data Analysis
title: "Mann-Whitney U Test: What It Actually Tests"
---

The Mann-Whitney U test, also called the Wilcoxon rank-sum test, is frequently taught as the non-parametric replacement for the independent-samples t-test. That description causes two common errors: using it automatically when a normality test rejects, and interpreting rejection as evidence that two medians differ.

The test is better understood through pairwise ordering.

## Pairwise interpretation

Let X be an observation from group 1 and Y an independent observation from group 2. A natural effect is

$$
\theta
=
P(X>Y)+\frac{1}{2}P(X=Y).
$$

If the two distributions are identical, theta is 0.5.

The U statistic estimates this pairwise ordering probability up to scaling. For samples of sizes $n_1$ and $n_2$,

$$
\hat\theta = \frac{U}{n_1n_2}
$$

for the appropriate orientation of U.

This gives the test a useful effect-size interpretation beyond a p-value.

## Rank construction

Pool the observations, assign ranks, and let R_1 be the rank sum for group 1. One common definition is

$$
U_1
=
R_1-\frac{n_1(n_1+1)}{2}.
$$

Equivalent formulas may report $n_1n_2-U_1$, so software orientation should be checked before interpreting direction.

## What the null hypothesis is

Under the classical distributional null,

$$
F_X=F_Y.
$$

The test is sensitive to differences in location, spread, shape, or other features that alter pairwise ordering.

Only under additional assumptions, such as distributions with the same shape differing by a location shift, can the result be interpreted cleanly as a location or median comparison.

Thus the statement "Mann-Whitney compares medians" is not generally correct.

## It is not a normality fallback

The two-sample t-test does not require the raw observations themselves to be exactly normal in moderate or large samples, and Welch's t-test handles unequal variances without assuming homoscedasticity.

Choosing Mann-Whitney solely because Shapiro-Wilk rejected normality is poor practice. The choice should depend on the estimand.

If the scientific question concerns a difference in means, Welch's t-test may remain the correct procedure even with skewness. If the question concerns stochastic ordering or rank-based location, Mann-Whitney may be more suitable.

## Independence remains essential

Rank-based does not mean dependence-free. Observations must be independent across experimental units under the usual test.

Matched pairs require a paired method such as the Wilcoxon signed-rank test or a model for paired outcomes. Clustered observations require cluster-aware inference.

## Ties

Ties are common for ordinal scales, rounded measurements, and count data. They affect the null variance and exact distribution.

Modern software usually applies tie corrections for asymptotic inference, but exact p-values may not be available in the same form when ties are present.

With heavily discrete outcomes, methods tailored to the scale of measurement may be preferable.

## Large-sample approximation

Without ties, under the null,

$$
E(U)=\frac{n_1n_2}{2}
$$

and

$$
\operatorname{Var}(U)
=
\frac{n_1n_2(n_1+n_2+1)}{12}.
$$

A normal approximation can then be used, with tie and sometimes continuity corrections as appropriate.

There is no universal sample-size threshold such as 20 that suddenly makes the approximation valid. Accuracy depends on both group sizes and the discreteness of the data.

## Effect sizes

Alongside $\hat\theta$, rank-biserial correlation can summarize direction and magnitude:

$$
r_{rb}=2\hat\theta-1.
$$

Values near zero indicate little pairwise dominance. Positive or negative values indicate direction according to the chosen group ordering.

Confidence intervals for an effect size are usually more informative than reporting only a significance decision.

## Example in Python

~~~python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import mannwhitneyu


def mann_whitney_summary(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> tuple[float, float, float]:
    """Return U, two-sided p-value, and probability-of-superiority estimate."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both groups must be one-dimensional")
    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain observations")

    result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    superiority = float(result.statistic / (a.size * b.size))
    return float(result.statistic), float(result.pvalue), superiority
~~~

The probability-of-superiority estimate should be interpreted with the same orientation used by the software's U statistic.

## Relation to other tests

The Wilcoxon signed-rank test is for paired differences and makes assumptions about the distribution of those differences.

The Kruskal-Wallis test extends rank-based comparison to more than two independent groups, but it is not a general replacement for one-way ANOVA when the target is a mean difference.

Permutation tests offer another route when an exchangeability null is scientifically appropriate and can target statistics such as mean differences directly.

## Conclusion

The Mann-Whitney U test is a test about relative ordering of two independent distributions. It becomes a location or median test only under additional structure.

Choose it because the rank-based estimand answers the scientific question, not because the data failed a normality test.

## References

- Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether One of Two Random Variables Is Stochastically Larger Than the Other.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods.
- Fay, M. P., & Proschan, M. A. (2010). Wilcoxon-Mann-Whitney or t-test? On assumptions for hypothesis tests and multiple interpretations of decision rules.
