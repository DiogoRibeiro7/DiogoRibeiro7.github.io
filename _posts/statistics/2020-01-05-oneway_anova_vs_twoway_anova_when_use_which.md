---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-05'
excerpt: One-way and two-way ANOVA are linear models with categorical predictors. Two-way ANOVA adds a second factor and, crucially, an interaction term whose interpretation changes the meaning of main effects.
header:
  image: /assets/images/headers/photo-statistics-anova.jpg
  og_image: /assets/images/headers/photo-statistics-anova.jpg
  overlay_image: /assets/images/headers/photo-statistics-anova.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-anova.jpg
  twitter_image: /assets/images/headers/photo-statistics-anova.jpg
keywords:
- one-way ANOVA
- two-way ANOVA
- factorial ANOVA
- interaction effects
- linear models
permalink: '/statistics/oneway_anova_vs_twoway_anova_when_use_which/'
seo_description: One-way and two-way ANOVA explained as linear models, with emphasis on factor coding, interactions, unbalanced designs, and what main effects mean when interactions are present.
seo_title: 'One-Way vs Two-Way ANOVA: The Linear-Model View'
seo_type: article
summary: A rigorous comparison of one-way and factorial ANOVA that shows how interactions change interpretation and why nonparametric substitutes are not simple drop-in replacements.
tags:
- ANOVA
- Experimental Design
- Linear Models
title: 'One-Way vs Two-Way ANOVA: The Linear-Model View'
---

ANOVA is often taught as a separate branch of statistics. It is easier to understand as a linear model with categorical predictors. A one-way ANOVA has one factor. A two-way ANOVA has two factors and, when scientifically relevant, an interaction between them. The difficult part is not the F statistic. It is deciding what mean structure the model should represent and how to interpret that structure when factors interact.

## One-way ANOVA as a linear model

Suppose factor $A$ has $a$ levels. A cell-means representation is

$$
Y_{ij}
=
\mu_i+\varepsilon_{ij},
$$

where $\mu_i$ is the mean in group $i$. The usual null hypothesis is

$$
H_0:
\mu_1=\mu_2=\cdots=\mu_a.
$$

With treatment coding, the same model can be written as

$$
Y
=
X\beta+\varepsilon.
$$

The familiar ANOVA decomposition and the regression formulation are two descriptions of the same least-squares geometry.

## The F statistic

Let

$$
SS_A
$$

denote variation explained by group membership and

$$
SS_E
$$

the residual variation. The one-way F statistic is

$$
F
=
\frac{
SS_A/(a-1)
}{
SS_E/(N-a)
}.
$$

Under the classical homoskedastic Gaussian model, this has an F distribution under the null. The test says only that not all means are equal. It does not identify which groups differ.

## Two-way ANOVA

Now suppose there are two factors:

- factor $A$ with $a$ levels;
- factor $B$ with $b$ levels.

The factorial model is

$$
Y_{ijk}
=
\mu
+
\alpha_i
+
\beta_j
+
(\alpha\beta)_{ij}
+
\varepsilon_{ijk}.
$$

The interaction term

$$
(\alpha\beta)_{ij}
$$

allows the effect of one factor to depend on the level of the other. Without that term, the model imposes additivity.

## What an interaction means

Suppose $A$ is treatment and $B$ is sex. If

$$
(\alpha\beta)_{ij}=0
$$

for all cells, the treatment contrast is the same across sex levels on the model's additive scale. With interaction, the treatment contrast can differ. For a simple two-by-two case,

$$
\text{interaction}
=
(\mu_{11}-\mu_{21})
-
(\mu_{12}-\mu_{22}).
$$

This is a difference of differences. That quantity is often more scientifically interesting than either marginal main effect.

## Main effects become conditional when interaction is present

A common mistake is to interpret a significant main effect as though it summarized the factor uniformly across the other factor. If interaction is substantial, there may be no single effect that deserves that interpretation. For example, treatment can help in one subgroup and harm in another. Averaging over the second factor can then produce a main effect near zero even though the subgroup effects are large.

When interaction matters, report simple effects or cell means rather than forcing the discussion back to marginal main effects.

## Parallel lines and interaction plots

Interaction plots are useful, but the common rule

> parallel lines mean no interaction, crossing lines mean interaction

is too literal. Nonparallel lines indicate interaction on the plotted scale. They do not need to cross. And visual parallelism is not a statistical test. The size and uncertainty of the interaction contrast should be reported.

## Assumptions belong to the errors, not the raw outcome pooled across groups

The classical model assumes

$$
E(\varepsilon\mid X)=0
$$

and, for the standard F tests,

$$
\operatorname{Var}(\varepsilon\mid X)
=
\sigma^2.
$$

Normality concerns the error distribution within the model, not whether all observed outcomes pooled together form a normal histogram. A strongly multimodal pooled distribution can be exactly what we expect when group means differ. Testing the pooled response for normality therefore answers the wrong question.

## Unequal variances

If the target remains a comparison of means but variances differ, heteroskedastic mean-comparison methods should be considered. For one factor, Welch ANOVA is often appropriate. For factorial designs, robust covariance estimators or heteroskedastic linear-model approaches can preserve the mean structure without changing the estimand to ranks.

Kruskal-Wallis and Friedman tests are not universal substitutes for ANOVA assumptions. They answer different questions and correspond to different designs.

## Friedman is not a nonparametric two-way ANOVA

The Friedman test is designed for blocked or repeated-measures layouts where each block receives multiple treatments. It is not the generic replacement for an ordinary two-factor between-subjects ANOVA. If both factors are between-subjects and interaction is scientifically important, there is no single rank test that reproduces the full two-way ANOVA problem without additional assumptions or modeling choices.

This is exactly why study design should come before choosing a named test.

## Balanced and unbalanced designs

In a balanced factorial experiment, each cell has the same sample size. Then main effects and interactions are orthogonal under standard coding, and sums of squares are straightforward. In observational or incomplete data, cell sizes often differ. Now the definition of a “main effect” depends on how marginal means are weighted. Different software may report Type I, II, or III sums of squares, which answer different hypotheses in unbalanced designs.

The solution is not to memorize one preferred type. It is to write down the estimable contrast that matches the scientific question.

## Coding affects coefficients, not fitted cell means

Treatment coding, sum coding, Helmert coding, and other parameterizations produce different coefficient tables. But if they span the same model space, fitted cell means are unchanged. This is another reason to interpret estimated marginal means and explicit contrasts rather than reading individual dummy-variable coefficients as if they were invariant scientific effects.

## A reproducible Python example

~~~python
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

rng = np.random.default_rng(2026)

rows: list[dict[str, object]] = []

for treatment in ["A", "B"]:
    for exercise in ["low", "high"]:
        mean = 10.0

if treatment == "B":
            mean += 2.0

if exercise == "high":
            mean += 1.0

if (
            treatment == "B"
            and exercise == "high"
        ):
            mean += 3.0

values = rng.normal(
            loc=mean,
            scale=2.0,
            size=40,
        )

for value in values:
            rows.append(
                {
                    "y": float(value),
                    "treatment": treatment,
                    "exercise": exercise,
                }
            )

data = pd.DataFrame(rows)

model = smf.ols(
    "y ~ C(treatment) * C(exercise)",
    data=data,
).fit()

print(anova_lm(model, typ=2))
print(model.params)
~~~

The interaction is built deliberately into the data-generating process. A model without the interaction term would impose the wrong mean structure.

## Post-hoc comparisons

After a significant one-way omnibus test, pairwise comparisons may be appropriate. But the family of comparisons must be defined. Tukey's HSD controls family-wise error for all pairwise comparisons under its assumptions. In factorial models, a more useful follow-up is often a set of prespecified simple contrasts, such as treatment differences within each exercise level.

Those contrasts should be adjusted for multiplicity when the inferential family requires it.

## Conclusion

One-way and two-way ANOVA are not different species of method. They are linear models with different categorical mean structures. The key progression is

$$
\boxed{
\text{one factor}
\rightarrow
\text{two factors}
\rightarrow
\text{interaction}
}
$$

The interaction term is the central addition because it tests whether the effect of one factor depends on the other. Assumptions, heteroskedasticity, repeated measures, and unbalanced designs should be handled within the model that matches the study design rather than by mechanically switching to an unrelated named test.

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters* (2nd ed.). Wiley.
- Maxwell, S. E., Delaney, H. D., & Kelley, K. (2018). *Designing Experiments and Analyzing Data* (3rd ed.). Routledge.
