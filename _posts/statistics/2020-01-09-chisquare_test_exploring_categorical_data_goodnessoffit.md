---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-09'
excerpt: Pearson's chi-square statistic compares observed counts with counts expected under a null model. The approximation depends on the sampling design and expected counts, and a significant result does not identify which cells drive the association.
header:
  image: /assets/images/headers/photo-statistics-categorical.jpg
  og_image: /assets/images/headers/photo-statistics-categorical.jpg
  overlay_image: /assets/images/headers/photo-statistics-categorical.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-categorical.jpg
  twitter_image: /assets/images/headers/photo-statistics-categorical.jpg
keywords:
- chi-square test
- contingency tables
- categorical data
- goodness of fit
- Cramer's V
seo_description: Pearson chi-square goodness-of-fit and independence tests explained through expected counts, sampling models, residuals, effect sizes, and exact alternatives.
seo_title: 'Chi-Square Tests: Expected Counts, Association, and Effect Size'
seo_type: article
summary: A rigorous guide to Pearson chi-square tests that derives the statistic, distinguishes goodness-of-fit from independence, and explains expected-count conditions, residual diagnostics, effect size, and exact tests.
tags:
- Categorical Data
- Hypothesis Testing
- Statistics
title: 'Chi-Square Tests: Expected Counts, Association, and Effect Size'
---

Pearson's chi-square statistic is one formula used in several related categorical-data tests.

The common structure is

$$
X^2
=
\sum
\frac{
(O-E)^2
}{
E
},
$$

where $O$ is an observed count and $E$ is the count expected under a specified null model.

The important part is not the formula.

It is how the expected counts were obtained.

## Goodness-of-fit

Suppose one categorical variable has $k$ categories.

Let

$$
O_1,\ldots,O_k
$$

be the observed counts and let the null model specify category probabilities

$$
p_1,\ldots,p_k,
\qquad
\sum_i p_i=1.
$$

For total sample size $n$, the expected counts are

$$
E_i=np_i.
$$

Pearson's statistic is

$$
X^2
=
\sum_{i=1}^{k}
\frac{
(O_i-E_i)^2
}{
E_i
}.
$$

If the null probabilities are fully specified and the usual asymptotic conditions hold,

$$
X^2
\xrightarrow{d}
\chi^2_{k-1}.
$$

If parameters used to compute $p_i$ are estimated from the same data, the degrees of freedom must be reduced accordingly and the ordinary reference distribution may require more care.

## Independence in a contingency table

Now suppose two categorical variables form an $r\times c$ table.

Let

$$
O_{ij}
$$

be the observed count in row $i$, column $j$.

Under independence,

$$
P(A=i,B=j)
=
P(A=i)P(B=j).
$$

The fitted expected count is

$$
E_{ij}
=
\frac{
(\text{row }i\text{ total})
(\text{column }j\text{ total})
}{
n
}.
$$

The statistic is

$$
X^2
=
\sum_{i=1}^{r}
\sum_{j=1}^{c}
\frac{
(O_{ij}-E_{ij})^2
}{
E_{ij}
}.
$$

Under the usual asymptotic conditions,

$$
X^2
\xrightarrow{d}
\chi^2_{(r-1)(c-1)}.
$$

This tests association.

It does not estimate a causal effect.

## Why the same formula appears twice

The goodness-of-fit and independence tests are both comparisons between observed counts and counts implied by a null model.

In the goodness-of-fit problem, the expected probabilities are specified externally or by a fitted model.

In the independence problem, expected counts are fitted under the factorization constraint

$$
p_{ij}=p_{i+}p_{+j}.
$$

The chi-square statistic measures discrepancy from the corresponding constrained model.

## The sampling design matters

The same numerical contingency table can arise from different designs.

For example:

- multinomial sampling with fixed total $n$;
- independent multinomial samples with fixed row totals;
- product-binomial sampling in a case-control or cohort design;
- Poisson sampling of cell counts.

The large-sample Pearson statistic can look similar across these formulations.

But the meaning of parameters and appropriate effect measures can differ.

A two-by-two table from a cohort naturally supports risk ratios and risk differences.

A case-control sample generally does not estimate population risks directly from the sampled row totals.

The table alone does not encode the design.

## Expected counts and the chi-square approximation

The chi-square reference distribution is asymptotic.

Small expected counts can make it inaccurate.

The common rule

> every expected cell count must be at least 5

is a heuristic, not a theorem.

What matters is the entire table structure, sparsity, dimension, and how extreme the expected counts are.

When counts are sparse, options include:

- exact conditional tests;
- Monte Carlo calibration;
- likelihood-based models;
- category aggregation when scientifically defensible.

Combining categories solely to satisfy a rule can change the estimand and discard meaningful information.

## Fisher's exact test

For a two-by-two table with fixed margins, Fisher's exact test conditions on the row and column totals.

Under the null, the cell count follows a hypergeometric distribution.

This gives exact finite-sample calibration under that conditional sampling model.

“Exact” does not mean universally superior.

The test conditions on margins and can be conservative depending on the inferential target.

It is one tool for sparse two-by-two data, not a generic replacement for every chi-square test.

## A significant chi-square statistic says only that the model does not fit

If an independence test rejects, we know that the observed table is incompatible with independence at the chosen level.

We do not yet know:

- which cells drive the discrepancy;
- the direction of association;
- the magnitude of association;
- whether the difference is practically important.

The omnibus p-value should therefore be followed by residuals and effect sizes.

## Pearson residuals

A simple residual is

$$
r_{ij}
=
\frac{
O_{ij}-E_{ij}
}{
\sqrt{E_{ij}}
}.
$$

Large absolute residuals identify cells contributing strongly to the Pearson statistic.

Because cell residuals are not independent and their variance is affected by fitted margins, adjusted standardized residuals are often more useful for diagnostic interpretation.

The point is to move from

$$
X^2
$$

to the structure of the departure.

## Effect size

For an $r\times c$ table, Cramér's $V$ is

$$
V
=
\sqrt{
\frac{
X^2
}{
n\min(r-1,c-1)
}
}.
$$

It ranges from 0 to 1.

It summarizes association strength but does not reveal direction.

For a two-by-two table, odds ratios, risk ratios, and risk differences can be more interpretable depending on the sampling design and scientific question.

A p-value and an effect size answer different questions.

## The chi-square test is not “non-parametric” in the sense of assumption-free

The statistic does not require normally distributed observations.

That does not make it assumption-free.

The analysis still depends on:

- independent sampling units or an appropriate dependence model;
- a correctly specified null structure;
- adequate asymptotic approximation or an exact alternative;
- correct classification of observations into categories.

Clustered survey data, repeated measures, or matched pairs violate the ordinary independence formulation.

For those designs, a standard Pearson chi-square test can have the wrong variance.

## Paired binary data need McNemar's test

Suppose the same subjects are measured before and after an intervention.

The two responses are paired.

A standard two-by-two independence test treats the counts as if the observations came from independent groups.

That is incorrect.

McNemar's test focuses on discordant pairs and is designed for paired binary data.

Study design comes before table format.

## Reproducible Python example

~~~python
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency

table: np.ndarray = np.array(
    [
        [42, 18, 10],
        [26, 24, 20],
    ],
    dtype=int,
)

result = chi2_contingency(
    table,
    correction=False,
)

chi2: float = float(result.statistic)
p_value: float = float(result.pvalue)
expected: np.ndarray = result.expected_freq

n: int = int(table.sum())
rows, cols = table.shape

cramers_v: float = float(
    np.sqrt(
        chi2
        / (
            n
            * min(rows - 1, cols - 1)
        )
    )
)

pearson_residuals: np.ndarray = (
    table - expected
) / np.sqrt(expected)

print(f"X^2 = {chi2:.3f}")
print(f"p = {p_value:.4f}")
print(f"Cramer's V = {cramers_v:.3f}")
print(pearson_residuals)
~~~

The omnibus statistic, effect size, and cell diagnostics should be interpreted together.

## Conclusion

A chi-square test is a model-discrepancy test for counts.

The core logic is

$$
\boxed{
\text{sampling design}
\rightarrow
\text{null model}
\rightarrow
\text{expected counts}
\rightarrow
X^2
\rightarrow
\text{effect size and diagnostics}.
}
$$

The formula is easy.

The statistical work lies in defining the correct expected counts and the correct sampling model.

## References

- Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine*, 50, 157–175.
- Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
- Bishop, Y. M. M., Fienberg, S. E., & Holland, P. W. (1975). *Discrete Multivariate Analysis*. MIT Press.
