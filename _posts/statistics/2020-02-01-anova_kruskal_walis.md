---
permalink: '/statistics/anova_kruskal_walis/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-02-01'
excerpt: ANOVA and Kruskal-Wallis do not answer the same question under different assumptions. The choice should follow the estimand, design, variance structure, and shape of the group distributions.
header:
  image: /assets/images/headers/photo-statistics-clt-binomial.jpg
  og_image: /assets/images/headers/photo-statistics-clt-binomial.jpg
  overlay_image: /assets/images/headers/photo-statistics-clt-binomial.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-clt-binomial.jpg
  twitter_image: /assets/images/headers/photo-statistics-clt-binomial.jpg
keywords:
- Kruskal-Wallis
- ANOVA
- Welch ANOVA
- rank tests
- hypothesis testing
redirect_from:
- '/statistics/data analysis/hypothesis testing/anova_kruskal_walis/'
seo_description: ANOVA, Welch ANOVA, and Kruskal-Wallis answer different questions. This article explains the estimands, assumptions, and consequences of choosing among them.
seo_title: 'ANOVA, Welch ANOVA, and Kruskal-Wallis'
seo_type: article
summary: A rigorous comparison of mean-based ANOVA, Welch ANOVA, and the rank-based Kruskal-Wallis procedure, with emphasis on what each null hypothesis means.
tags:
- Hypothesis Testing
- Nonparametric Methods
- Statistics
title: 'ANOVA, Welch ANOVA, and Kruskal-Wallis: They Are Not Interchangeable'
---

The usual decision tree is familiar:

$$
\text{normal data} \Rightarrow \text{ANOVA},
\qquad
\text{non-normal data} \Rightarrow \text{Kruskal-Wallis}.
$$

A second version says:

$$
\text{equal variances} \Rightarrow \text{ANOVA},
\qquad
\text{unequal variances} \Rightarrow \text{Kruskal-Wallis}.
$$

Both rules are too crude.

ANOVA and Kruskal-Wallis do not generally test the same estimand under different assumptions. If the scientific question is about **means**, replacing ANOVA by a rank test because a normality test rejected can change the question being asked.

The useful sequence is

$$
\boxed{
\text{estimand}
\rightarrow
\text{study design}
\rightarrow
\text{sampling assumptions}
\rightarrow
\text{test}
}
$$

not the other way around.

## Classical one-way ANOVA

Suppose there are $g$ independent groups,

$$
Y_{ij}=\mu_i+\varepsilon_{ij},
$$

where $i=1,\ldots,g$ indexes groups and $j=1,\ldots,n_i$ indexes observations.

The classical one-way ANOVA null hypothesis is

$$
H_0:
\mu_1=\mu_2=\cdots=\mu_g.
$$

The total variation is decomposed into between-group and within-group components. With

$$
N=\sum_{i=1}^{g}n_i,
$$

the test statistic is

$$
F
=
\frac{MS_{\mathrm{between}}}
{MS_{\mathrm{within}}}.
$$

Under the classical Gaussian homoscedastic model,

$$
\varepsilon_{ij}
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma^2),
$$

the null distribution is an $F$ distribution with $g-1$ and $N-g$ degrees of freedom.

The null hypothesis is about means.

That point should remain visible throughout the analysis.

## Normality is not a binary gatekeeper

The classical finite-sample derivation uses normal errors, but practical robustness depends on sample size, imbalance, tail behavior and outliers.

A rejection from Shapiro-Wilk does not imply that the mean is no longer the target or that ANOVA must be abandoned.

Likewise, failure to reject normality does not prove that the Gaussian model is correct.

The relevant diagnostic object is the within-group error structure or model residuals, not a pooled histogram of all observations.

When the scientific target is a difference in means, robustness checks should preserve that estimand.

## Unequal variances point to Welch ANOVA

If the group means remain the target but the variances differ,

$$
\operatorname{Var}(Y_{ij})=\sigma_i^2,
$$

the natural alternative is **Welch's ANOVA**, not automatically Kruskal-Wallis.

Welch's procedure modifies the weighting and degrees of freedom so that inference on group means remains useful under heteroscedasticity.

Conceptually, this is important:

$$
\boxed{
\text{heteroscedastic means problem}
\Rightarrow
\text{use a method for heteroscedastic means}
}
$$

rather than changing to a rank estimand without noticing.

## What Kruskal-Wallis actually tests

The Kruskal-Wallis statistic is based on pooled ranks.

Let $R_{ij}$ denote the rank of observation $Y_{ij}$ among all $N$ observations and let $\bar R_i$ be the mean rank in group $i$. Ignoring the tie correction for notation, the statistic is

$$
H
=
\frac{12}{N(N+1)}
\sum_{i=1}^{g}
n_i
\left(
\bar R_i-\frac{N+1}{2}
\right)^2.
$$

Under the null hypothesis that the group distributions are the same, and under the usual regularity conditions, $H$ is approximately chi-square with $g-1$ degrees of freedom.

The general null is therefore about equality of distributions,

$$
H_0:
F_1=F_2=\cdots=F_g.
$$

If all group distributions have the same shape and differ only by a location shift, the procedure can be interpreted as a test of location. Under stronger symmetry assumptions, that is sometimes summarized informally as a comparison of medians.

Without those assumptions, "Kruskal-Wallis compares medians" is not generally correct.

## Unequal spreads can make Kruskal-Wallis reject

Suppose two groups have the same center but very different spread.

Their rank distributions can differ even though their means or medians are equal.

Kruskal-Wallis can therefore reject because the distributions differ in scale or shape.

This is why the statement

> Use Kruskal-Wallis when variances are unequal

is particularly dangerous.

Unequal variances are not a nuisance that the rank test simply ignores. They can be part of what drives the rank differences.

## Outliers do not automatically define the estimand

Rank procedures are less sensitive to the numerical magnitude of extreme values because only order is retained.

That can be useful.

But the correct response to outliers depends on why they exist.

An outlier caused by a recording error should be corrected or removed for a documented reason. A genuine extreme observation may be scientifically important. A heavy-tailed population may call for a robust location estimator, transformed model, generalized linear model, bootstrap procedure or explicit heavy-tailed likelihood.

Switching to ranks solely because an observation looks inconvenient is not a statistical principle.

## A practical decision framework

### If the target is the group mean

Use a mean-based method.

- Classical ANOVA is appropriate under the standard homoscedastic model.
- Welch ANOVA is preferable when variances differ materially.
- With difficult tails or small samples, consider bootstrap or permutation procedures designed around the mean, or a model whose error distribution better reflects the data.

### If the target is a location shift or stochastic ordering

A rank-based procedure may be appropriate.

Kruskal-Wallis is useful when the scientific question is naturally expressed through relative ranks and when its interpretation matches the shapes of the group distributions.

### If distributions differ in several ways

Then a single location test may be inadequate.

Plot the distributions and consider models that allow differences in scale, shape or other features explicitly.

## Post-hoc comparisons must match the global test

A significant omnibus result does not identify which groups differ.

After classical ANOVA, Tukey's HSD is a common familywise-error-controlled method when its assumptions fit.

After Welch ANOVA, a heteroscedastic post-hoc procedure such as Games-Howell is more coherent than ordinary Tukey HSD.

After Kruskal-Wallis, pairwise rank-based comparisons can be used with multiplicity correction, but their interpretation remains rank-based.

The global and follow-up analyses should answer the same kind of question.

## Example

Suppose three treatment groups have sample means

$$
10.2,\quad 11.1,\quad 13.8,
$$

but the third group has much larger variance.

If the research question is

> Are the population means equal?

then unequal variance does not make the question disappear.

Welch ANOVA preserves the mean comparison.

Replacing the analysis with Kruskal-Wallis changes the object from equality of means to equality of rank distributions. If the third group has the same center but a much wider distribution, the rank test may still react.

That may be scientifically interesting.

It is simply a different claim.

## Conclusion

ANOVA and Kruskal-Wallis are not parametric and nonparametric versions of one identical question.

Classical ANOVA is a model-based test of group means under homoscedastic assumptions. Welch ANOVA keeps the mean estimand while relaxing equality of variances. Kruskal-Wallis is a rank-based test whose general null concerns the equality of group distributions.

The right choice therefore begins with the scientific target:

$$
\boxed{
\text{Do I care about means, locations, ranks, or the full distributions?}
}
$$

Once that is clear, the statistical method becomes much easier to defend.

## References

- Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621. https://doi.org/10.1080/01621459.1952.10483441
- Welch, B. L. (1951). On the comparison of several mean values: An alternative approach. *Biometrika*, 38(3/4), 330–336.
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
