---
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-03-01'
excerpt: The Chi-Square Test is a powerful tool for analyzing relationships in categorical data. Learn its principles and practical applications.
header:
  image: /assets/images/data_science_9.webp
  og_image: /assets/images/data_science_9.webp
  overlay_image: /assets/images/data_science_9.webp
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.webp
  twitter_image: /assets/images/data_science_9.webp
keywords:
- Chi-square test
- Categorical data
- Goodness-of-fit
- Independence test
seo_description: Discover how to use the Chi-Square Test to analyze categorical data, including tests for independence and goodness-of-fit.
seo_title: Chi-Square Test for Categorical Data
seo_type: article
summary: An exploration of the Chi-Square Test, focusing on its use in testing the association between categorical variables and examining goodness-of-fit in statistical analysis.
tags:
- Data Analysis
- Hypothesis Testing
title: 'Chi-Square Test: Testing Categorical Data'
---

The chi-square test is the standard tool for asking whether counts of categorical outcomes differ from what some hypothesis predicts. It applies wherever data arrives as frequencies in categories rather than measurements on a scale — survey responses, defect classes, disease status, click-through outcomes.

## What the Chi-Square Test Measures

Every version of the test compares observed counts against counts expected under a null hypothesis, and combines the discrepancies into one number:

$$
\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i},
$$

where $O_i$ is the observed count in cell $i$ and $E_i$ the expected count. Squaring makes over- and under-shoots count equally; dividing by $E_i$ scales each discrepancy by how large it should have been, so a shortfall of 10 in a cell expecting 20 weighs far more than the same shortfall in a cell expecting 2000.

Large values indicate the observed pattern is unlikely under the null. Under the null the statistic follows approximately a chi-square distribution, whose degrees of freedom depend on which version you are running.

Note that the test operates on **counts, never on percentages**. Feeding it proportions destroys the sample size information the statistic depends on, and is one of the most common ways to get a badly wrong answer while producing a plausible-looking number.

## The Two Main Tests

### Goodness of Fit

This compares one categorical variable against a hypothesised distribution. Are the six faces of a die equally likely? Do observed blood types match population frequencies?

Expected counts are $E_i = n p_i$ under hypothesised proportions $p_i$, with $k - 1$ degrees of freedom for $k$ categories. If any parameters were estimated from the same data, subtract one degree of freedom for each.

### Test of Independence

This asks whether two categorical variables are associated, using a contingency table. Expected counts come from the marginals:

$$
E_{ij} = \frac{R_i \times C_j}{N},
$$

where $R_i$ is the row total, $C_j$ the column total and $N$ the grand total. Degrees of freedom are $(r-1)(c-1)$.

A closely related third form, the test of homogeneity, uses identical arithmetic but a different design: independence samples one population and classifies it two ways, while homogeneity samples several populations and compares one classification across them. The computation does not change; the interpretation does.

## A Worked Example

```python
import numpy as np
from scipy import stats

# rows = treatment arm, columns = outcome (improved, unchanged, worse)
observed = np.array([[45, 30, 25],
                     [30, 35, 35]])

chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"chi-square = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
print("expected counts:\n", expected.round(1))
print("minimum expected count:", expected.min().round(1))

# Cramer's V: effect size, 0 to 1
n = observed.sum()
V = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
print(f"Cramer's V = {V:.3f}")

# which cells drive the result?
resid = (observed - expected) / np.sqrt(expected)
print("standardised residuals:\n", resid.round(2))
```

Two habits make this far more informative than the p-value alone. **Cramér's V** converts the statistic into an effect size between 0 and 1, which matters because chi-square grows with sample size — with 100,000 observations a trivial association becomes highly significant. **Standardised residuals** show which cells depart from expectation and in which direction; values beyond roughly ±2 mark the cells actually driving the result. A significant test with no explanation of where the discrepancy lies is an incomplete analysis.

## Assumptions and When They Fail

The test rests on conditions that are easy to overlook:

- **Independent observations.** Each subject contributes to exactly one cell. Repeated measurements on the same people violate this outright and call for McNemar's test (2×2) or Cochran's Q instead.
- **Sufficient expected counts.** The chi-square approximation degrades when expected counts are small. The usual rule is that all expected counts should exceed 5, or at minimum that no more than 20% fall below 5 and none below 1. When violated, Fisher's exact test gives an exact answer for small tables.
- **Counts, not proportions or means.** As above.
- **Mutually exclusive, exhaustive categories.** Every observation belongs to one and only one cell.

For 2×2 tables, Yates's continuity correction is sometimes applied to compensate for approximating a discrete distribution with a continuous one. It is conservative and generally unnecessary once expected counts are adequate; SciPy applies it by default for 2×2, which is worth knowing since it changes the result.

## What a Significant Result Does Not Tell You

A significant chi-square says the variables are associated. It does not say the association is strong, does not say which categories are responsible, and above all does not establish causation.

It also gives no direction. Unlike a correlation coefficient, chi-square has no sign, because the categories may have no natural ordering. If your categories *are* ordered — none, mild, moderate, severe — the standard test discards that information and a test for trend, such as the Cochran-Armitage test, will be more powerful.

Simpson's paradox applies here with full force. An association in a pooled table can vanish or reverse within every subgroup, so check whether an obvious stratifying variable changes the picture before treating the aggregate as the finding.

## Choosing Between the Alternatives

| Situation | Test |
|---|---|
| One variable vs a hypothesised distribution | Chi-square goodness of fit |
| Two variables, independent observations, adequate counts | Chi-square test of independence |
| Small expected counts | Fisher's exact test |
| Paired binary data (before/after on the same subjects) | McNemar's test |
| Three or more paired binary conditions | Cochran's Q |
| Ordered categories with a trend hypothesis | Cochran-Armitage trend test |
| Very large sparse tables | Likelihood-ratio (G) test |

The G-test is worth noting as a near-equivalent: it uses the same degrees of freedom and gives similar answers, but decomposes additively, which makes it easier to work with in hierarchical models.

## Reporting

Report the statistic, degrees of freedom, sample size and p-value together — $\chi^2(2, N = 200) = 6.28, p = .043$ — plus an effect size and the cells driving the effect. Stating $N$ is not a formality: it is what lets a reader judge whether a significant result reflects a meaningful association or merely a large sample.

## References

- Agresti, A. (2018). *An Introduction to Categorical Data Analysis* (3rd ed.). Wiley.
- Pearson, K. (1900). On the criterion that a given system of deviations from the probable... *Philosophical Magazine*, 50(302), 157-175.
- Cochran, W. G. (1954). Some methods for strengthening the common chi-square tests. *Biometrics*, 10(4), 417-451.
- Sharpe, D. (2015). Chi-square test is statistically significant: now what? *Practical Assessment, Research & Evaluation*, 20(8).
