---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.webp
  og_image: /assets/images/data_science_8.avif
  overlay_image: /assets/images/data_science_9.webp
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.webp
  twitter_image: /assets/images/data_science_8.avif
keywords:
- Repeated measures anova
- Non-parametric test
- Friedman test
- Ordinal data
permalink: '/data-science/friedman_test/'
redirect_from:
- '/data analysis/friedman_test/'
- '/data science/friedman_test/'
seo_description: The Friedman test as a non-parametric alternative to repeated measures ANOVA, and its use with ordinal data or non-normal distributions.
seo_title: 'Friedman Test: Non-Parametric Repeated Measures'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Nonparametric Methods
- Hypothesis Testing
- Regression
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
---

In data analysis, we often encounter situations where we need to compare three or more related groups. When the assumptions of normality or homogeneity of variances are not met, using parametric methods such as repeated measures ANOVA may not be appropriate. In such cases, the **Friedman test** offers a robust **non-parametric alternative**.

The Friedman test is particularly useful for analyzing **ordinal data** or **non-normal distributions** in repeated measures designs, where the same subjects are measured under different conditions or across different time points. This article explains the test, how it works, and when to use it.

## When and How to Use the Friedman Test

The Friedman test is ideal for scenarios where:

1. **Data is ordinal**: The values can be ranked, but the distance between the ranks is not necessarily equal.
2. **Data is not normally distributed**: The test is robust to violations of normality, making it suitable for skewed or non-normal data.
3. **Repeated measurements on the same subjects**: When the same subjects are exposed to multiple conditions or measured at different time points.
4. **Small sample sizes**: Because it is non-parametric, the Friedman test can handle smaller sample sizes better than parametric alternatives.

### Assumptions of the Friedman Test

Despite being non-parametric, the Friedman test has its own set of assumptions:

- **Repeated measures**: The data must be from the same subjects, measured under different conditions.
- **Ordinal or continuous data**: The test can handle both ordinal and continuous data as long as ranks can be assigned.
- **Independence within groups**: While the measurements are related within subjects, the observations should be independent across subjects.

Two clarifications are worth making, because both are commonly misstated. "Non-parametric" does not mean assumption-free — it means no assumption about the *shape* of the distribution. The design assumptions above still bind, and a violated independence assumption invalidates the test just as thoroughly as it would a parametric one.

The test also does not require equal variances or symmetry, but it does assume that the blocks (subjects) are exchangeable and that there is no subject-by-treatment interaction. If a treatment helps some subjects and harms others in roughly equal measure, the rank sums can come out even and the test will report nothing while a real, heterogeneous effect exists.

## How the Friedman Test Works

The test ranks the data **within each subject** across the treatments, then compares the rank sums. This within-subject ranking is the crucial design feature: it removes any between-subject differences in overall level, so a consistently high-scoring participant contributes no more to the result than a consistently low-scoring one.

If treatments are equivalent, each should collect roughly the same total rank. Systematic differences produce systematically different rank sums.

With $n$ subjects and $k$ treatments, let $R_j$ be the sum of ranks for treatment $j$. The statistic is

$$
\chi_F^2 = \frac{12}{n k (k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1),
$$

which under the null follows approximately a chi-square distribution with $k-1$ degrees of freedom. The approximation is reasonable when $n$ is moderate; for small $n$ and $k$, exact distributions or a permutation test are preferable, since the asymptotic version is conservative there.

Ties within a subject receive average ranks, and a tie correction should be applied when they are common — otherwise the statistic is deflated and the test loses power.

## A Worked Example

Suppose eight assessors each rate three algorithms on a 1-10 usability scale. The same assessor rates all three, so the measurements are related.

```python
import numpy as np
from scipy import stats

# rows = assessors, columns = algorithms A, B, C
scores = np.array([
    [7, 8, 5], [6, 9, 6], [8, 8, 4], [5, 7, 5],
    [7, 9, 6], [6, 8, 3], [8, 9, 5], [7, 7, 4],
])

stat, p = stats.friedmanchisquare(*scores.T)
print(f"Friedman chi-square = {stat:.3f}, p = {p:.5f}")

# rank within each assessor, then sum per algorithm
ranks = np.apply_along_axis(stats.rankdata, 1, scores)
print("mean rank per algorithm:", ranks.mean(axis=0).round(2))

# Kendall's W: the same information expressed as agreement, 0 to 1
n, k = scores.shape
W = stat / (n * (k - 1))
print(f"Kendall's W = {W:.3f}")
```

A significant result says only that the algorithms are not interchangeable. It does not say which differ, and reporting the omnibus p-value alone is the most common way this test is under-used.

Kendall's $W$ is worth computing alongside it. It rescales the same statistic onto $[0, 1]$ as a measure of agreement among the assessors, giving an effect size where the p-value gives only a decision.

## Following Up a Significant Result

Once the omnibus test rejects, pairwise comparisons identify where the differences lie — and those comparisons need adjusting for multiplicity, since three treatments give three pairs and five give ten.

The Nemenyi test is the standard post-hoc for Friedman, comparing mean rank differences against a critical distance derived from the studentised range. It requires no distributional assumption beyond those already made. The Conover test is more powerful but should be used only after a significant omnibus result, since it borrows the overall rank variance.

A simpler and often adequate route is pairwise Wilcoxon signed-rank tests with a Holm or Benjamini-Hochberg correction. Whichever you choose, decide before looking at the data, because selecting the post-hoc that yields significance is exactly the practice these corrections exist to prevent.

## How It Compares to the Alternatives

Repeated measures ANOVA is the parametric counterpart. When its assumptions hold it is more powerful, so the Friedman test costs you something — the asymptotic relative efficiency is about 0.95 for three treatments under normality, rising as $k$ grows. That is a modest price for robustness.

The choice is therefore not automatic. If the data are genuinely continuous and roughly normal, and sphericity holds or can be corrected for, repeated measures ANOVA is the better tool. If the data are ordinal, badly skewed, or contain outliers that would dominate a mean, the Friedman test is more trustworthy.

For two related samples the Friedman test reduces to a form equivalent to the sign test, and the Wilcoxon signed-rank test is the more powerful choice there. For independent rather than related groups, the Kruskal-Wallis test is the correct analogue — applying Friedman to unrelated groups is a design error, not a robustness choice.

## Reporting

A complete report gives the statistic, degrees of freedom, sample size and p-value, together with mean ranks per condition, an effect size such as Kendall's $W$, and the post-hoc procedure with its adjusted p-values. Mean ranks matter because they show the direction of the effect, which the chi-square statistic alone conceals.

Stating the design explicitly is equally important: readers need to know the measurements were repeated on the same subjects to judge whether the test was appropriate at all.

## References

- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Siegel, S., & Castellan, N. J. (1988). *Nonparametric Statistics for the Behavioral Sciences* (2nd ed.). McGraw-Hill.
