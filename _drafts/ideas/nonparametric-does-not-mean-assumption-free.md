---
author_profile: false
categories:
- Statistics
classes: wide
excerpt: Rank-based and distribution-free procedures are not assumption-free. Changing from a parametric test to a nonparametric alternative can also change the estimand, the null hypothesis, and the scientific question being answered.
header:
  image: /assets/images/headers/photo-statistics-ecdf.jpg
  og_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-ecdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-ecdf.jpg
keywords:
- nonparametric statistics
- Mann-Whitney
- Wilcoxon signed-rank
- Kruskal-Wallis
- estimands
- hypothesis testing
- rank tests
seo_description: Why nonparametric tests are not assumption-free, how rank tests can change the scientific question, and why Mann-Whitney is not generally a test of medians.
seo_title: Nonparametric Does Not Mean Assumption-Free
seo_type: article
summary: A careful look at what rank-based tests actually assume and test, why distribution-free is not the same as assumption-free, and why the estimand must be chosen before the test.
tags:
- Nonparametric Statistics
- Hypothesis Testing
- Statistical Inference
- Estimands
title: 'Nonparametric Does Not Mean Assumption-Free'
---

A common statistical workflow goes like this:

1. test the data for normality;
2. if normality is rejected, replace the t-test with Mann–Whitney, the paired t-test with Wilcoxon signed-rank, or ANOVA with Kruskal–Wallis;
3. interpret the new p-value as answering the same scientific question with fewer assumptions.

That workflow is attractive because it feels conservative.

It is also often wrong.

The central problem is not that rank-based tests are bad. They are extremely useful.

The problem is that moving from a parametric procedure to a nonparametric one can change

- the null hypothesis;
- the estimand;
- the interpretation of an effect;
- the assumptions required for a location-shift interpretation;
- the target population comparison.

So the important distinction is:

$$
\boxed{
\text{distribution-free}
\neq
\text{assumption-free}.
}
$$

And, just as importantly,

$$
\boxed{
\text{changing the test can change the question}.
}
$$

## Start with the estimand, not the test name

Suppose we have two independent populations with random variables

$$
X\sim F,
\qquad
Y\sim G.
$$

There are many legitimate ways to compare them.

We might care about a difference in means,

$$
\Delta_\mu
=
E[Y]-E[X],
$$

a difference in medians,

$$
\Delta_{0.5}
=
Q_Y(0.5)-Q_X(0.5),
$$

a probability of superiority,

$$
\theta_P
=
P(Y>X)
+
\frac12P(Y=X),
$$

or a full-distribution hypothesis,

$$
H_0:F=G.
$$

These are not interchangeable.

Two distributions can have the same mean and different medians.

They can have the same median and different spread.

They can have the same mean and median but different tails.

They can have

$$
P(Y>X)\neq\frac12
$$

even when the medians are equal.

So before choosing a procedure, we need to decide which object is scientifically meaningful.

## Mann–Whitney does not generally test equality of medians

For independent samples, the Mann–Whitney statistic can be interpreted through pairwise comparisons.

For continuous distributions, one natural parameter is

$$
\theta_P
=
P(Y>X).
$$

Under identical distributions,

$$
\theta_P=\frac12.
$$

The Mann–Whitney test is therefore naturally connected to stochastic ordering and rank dominance, not automatically to the difference in medians.

The familiar median interpretation becomes valid only under additional structure, such as a common-shape location-shift model.

Suppose

$$
G(y)=F(y-\delta).
$$

Then the two populations differ only by a horizontal shift $\delta$.

In that special case,

- mean shift,
- median shift,
- quantile shift,
- rank ordering

all move coherently with the same underlying location parameter.

But once shape or spread changes, that equivalence disappears.

## A counterexample with equal medians

Let

$$
X\sim N(0,1)
$$

and

$$
Y=E-\log 2,
\qquad
E\sim\operatorname{Exp}(1).
$$

The exponential distribution has median $\log 2$, so

$$
\operatorname{median}(Y)=0.
$$

The standard normal also has median zero:

$$
\operatorname{median}(X)=0.
$$

Therefore

$$
\boxed{
\Delta_{0.5}=0.
}
$$

Yet

$$
P(Y>X)
\approx 0.5569.
$$

So a random draw from $Y$ is more likely to exceed a random draw from $X$, even though the population medians are identical.

That is not a contradiction.

The two distributions have different shapes.

The example shows why the statement

> “Mann–Whitney tests whether the two medians are equal”

is false without extra assumptions.

A significant rank-sum test can reflect differences in

- location;
- spread;
- skewness;
- tails;
- or some combination of them.

If the scientific question is specifically about a median difference, then the median itself should appear explicitly in the estimand and inferential procedure.

## The null hypothesis matters

There are several different hypotheses that are easily conflated:

### Equality of distributions

$$
H_0:F=G.
$$

### Equal means

$$
H_0:E[X]=E[Y].
$$

### Equal medians

$$
H_0:Q_X(0.5)=Q_Y(0.5).
$$

### Probability of superiority equal to one half

$$
H_0:P(Y>X)+\frac12P(Y=X)=\frac12.
$$

These hypotheses can disagree.

A test calibrated for one hypothesis should not be interpreted as though it tested another.

This is the first major lesson:

$$
\boxed{
\text{a p-value inherits the meaning of its null hypothesis.}
}
$$

It does not inherit the analyst's preferred verbal interpretation after the fact.

## “Nonparametric” does not mean “no model”

The term nonparametric is often interpreted too literally.

Usually it means that we are not specifying a finite-dimensional parametric family such as

$$
X\sim N(\mu,\sigma^2).
$$

But rank-based inference still relies on structure.

Depending on the test and interpretation, we may need

- independent observations;
- identically distributed observations within groups;
- exchangeability under the null;
- continuous measurements or a tie-handling rule;
- symmetric paired differences;
- a common-shape location-shift structure;
- comparable measurement scales;
- an appropriate sampling design.

Those are assumptions.

They are simply different assumptions from normality.

## Distribution-free refers to the reference distribution

A more precise way to understand classical rank tests is through the phrase **distribution-free under the null**.

Under appropriate exchangeability conditions, the null distribution of a rank statistic may not depend on the unknown continuous distribution $F$.

That is extremely useful.

But it says something about the **sampling distribution of the statistic under a specified null**.

It does not say

$$
\text{the scientific interpretation is free of assumptions}.
$$

Nor does it say

$$
\text{the observations may have any dependence structure}.
$$

If the design breaks exchangeability, the usual reference distribution may no longer be valid.

So “distribution-free” should not be expanded into “works under anything.”

## Independence does not disappear when we rank the data

Suppose observations come from clusters, repeated measurements, cross-validation folds, households, schools, hospitals, or time series.

Ranks do not remove dependence.

If the procedure assumes independent sampling units but we feed it correlated observations, the nominal null distribution can be wrong.

This point matters especially in machine learning, where people often compare fold-level performance scores with a Wilcoxon test as if the folds were independent replicates.

They are not necessarily independent because training sets overlap.

A nonparametric test cannot rescue a misidentified sampling unit.

The correct question is still:

$$
\boxed{
\text{what is the independent experimental unit?}
}
$$

## Paired tests operate on differences

For paired data, let

$$
D_i=Y_i-X_i.
$$

The paired t-test focuses on

$$
E[D].
$$

The sign test focuses on the sign of $D$ and is naturally connected to the median of the difference distribution.

The Wilcoxon signed-rank test uses both signs and the ranks of absolute differences.

Those are different pieces of information.

So again, these procedures are not interchangeable merely because all three accept paired observations.

## Wilcoxon signed-rank needs symmetry for a location interpretation

A common description says that Wilcoxon signed-rank is simply a nonparametric test of whether the median paired difference is zero.

That description is too loose.

The signed-rank procedure exploits the ranking of

$$
|D_i|
$$

while retaining the sign of $D_i$.

For the conventional location interpretation, the difference distribution is assumed to be symmetric around a location parameter.

Under symmetry about zero,

$$
D\overset{d}{=}-D.
$$

If the distribution of paired differences is strongly asymmetric, a zero median does not imply the signed-rank null in the usual location-shift sense.

The signed-rank statistic is sensitive to the interaction between sign and magnitude.

That is precisely what gives it more power than a sign test under symmetric location shifts.

But it also means that asymmetry matters.

## Median zero and signed-rank center are different objects

To see this, take a shifted exponential difference

$$
D=E-\log 2,
\qquad
E\sim\operatorname{Exp}(1).
$$

Its median is exactly zero:

$$
\operatorname{median}(D)=0.
$$

But the distribution is strongly right-skewed, not symmetric.

The center targeted by signed-rank methods is related to the Hodges–Lehmann pseudomedian, which for a one-sample location problem can be expressed as the median of pairwise averages

$$
\frac{D_i+D_j}{2}.
$$

For this shifted exponential population, the corresponding population pseudomedian is approximately

$$
0.146.
$$

So

$$
\boxed{
\text{median}(D)=0
\quad\text{but}\quad
\text{pseudomedian}(D)>0.
}
$$

This is exactly the kind of case where saying “Wilcoxon tests the median” hides the real assumptions.

If the scientific estimand is literally the median paired difference, the sign test or direct median inference may be conceptually closer, although efficiency and discreteness still need consideration.

## The sign test makes a different trade-off

The sign test throws away the magnitude of each nonzero difference and keeps only

$$
I(D_i>0).
$$

That loss of information can reduce power.

But the test also requires less structure for a median interpretation.

For continuous $D$, the hypothesis

$$
P(D>0)=\frac12
$$

corresponds to a median of zero.

So there is a genuine trade-off:

$$
\boxed{
\text{use more information with more structure}
\quad\text{or}\quad
\text{use less information with weaker structure}.
}
$$

That is a much better way to think about the sign test versus signed-rank test than “weak test versus strong test.”

## Kruskal–Wallis has the same interpretive trap

For $k$ independent groups, the Kruskal–Wallis test ranks all observations jointly and asks whether the groups have the same rank distribution under the null.

It is often introduced as the nonparametric version of one-way ANOVA.

That analogy is useful operationally but dangerous scientifically.

ANOVA typically focuses on mean structure:

$$
H_0:
\mu_1=\cdots=\mu_k.
$$

Kruskal–Wallis is naturally tied to equality of distributions or rank location.

Only under additional common-shape assumptions can it be interpreted cleanly as a test of equal medians or a common location parameter.

If groups differ mainly in spread or skewness, Kruskal–Wallis may reject even when medians are equal.

So replacing ANOVA with Kruskal–Wallis after a normality test can silently change the scientific hypothesis.

## The same warning applies to effect sizes

A p-value is not enough.

If we use a rank-based test, the effect size should match the estimand.

For Mann–Whitney, useful quantities include

$$
P(Y>X)+\frac12P(Y=X),
$$

rank-biserial correlation, or another explicitly defined stochastic-order parameter.

If a location-shift model is scientifically reasonable, a Hodges–Lehmann shift estimate may be useful.

But it should be reported as a shift estimand under that model, not automatically relabeled as a difference in medians.

For paired data, the pseudomedian of differences and the median of differences should not be conflated under asymmetry.

The effect estimate should tell the reader exactly what is being compared.

## Why the normality-pretest workflow is conceptually weak

Consider this common rule:

$$
\text{if Shapiro–Wilk }p<0.05,
\text{ use Mann–Whitney.}
$$

There are several problems.

First, exact normality may be irrelevant to the robustness of the mean-based procedure.

Second, a rank test does not necessarily target the mean.

Third, a nonsignificant normality test does not establish normality.

Fourth, the final analysis becomes data-dependent in a way that is rarely reflected in the reported uncertainty.

Fifth, the choice of estimand is being made by a diagnostic p-value rather than by the scientific question.

A better workflow is:

$$
\boxed{
\text{estimand}
\rightarrow
\text{sampling design}
\rightarrow
\text{plausible assumptions}
\rightarrow
\text{estimator/test}
\rightarrow
\text{robustness analysis}.
}
$$

Normality diagnostics may appear in that workflow, but they should not define the scientific target.

## Robust methods are often better comparators than rank substitution

If the target is a mean difference, non-normality does not force us to abandon the mean.

Depending on the design, we may consider

- Welch's t-test;
- heteroskedasticity-robust standard errors;
- permutation tests built around the desired statistic;
- bootstrap confidence intervals;
- trimmed-mean procedures;
- robust regression;
- generalized linear models;
- transformations with an explicit interpretation;
- model-based likelihood methods.

The choice depends on the estimand and failure mode.

For example, if unequal variances are the problem, Welch's test may preserve the mean-difference estimand while addressing heteroskedasticity more directly than switching to Mann–Whitney.

This is a recurring principle:

$$
\boxed{
\text{fix the assumption problem without changing the question unless you intend to change the question.}
}
$$

## Permutation tests make the design assumptions visible

Permutation reasoning is useful because it forces us to ask what can legitimately be permuted under the null.

If treatment labels are exchangeable under random assignment, a randomization test can be exact for a statistic tied directly to the design.

If observations are paired, permutations must respect the pairs.

If data are clustered, the cluster may be the unit of exchangeability.

If dependence is temporal, unrestricted permutations are usually invalid.

The word “nonparametric” does not solve any of these design questions.

The permutation group itself encodes assumptions.

## A practical decision table

| Scientific target | Candidate method | Key assumptions / interpretation |
| --- | --- | --- |
| Difference in means, independent groups | Welch t / robust mean inference | independent units; finite-moment and large-sample conditions as appropriate |
| Equality of full distributions | rank/permutation procedures | exchangeability or sampling assumptions; interpretation is distributional |
| Probability that one random draw exceeds another | Mann–Whitney effect parameter | independent samples; tie convention; not automatically a median effect |
| Common location shift | Mann–Whitney + Hodges–Lehmann | common-shape/location-shift structure |
| Mean paired difference | paired t / robust paired mean inference | independent pairs; assumptions on differences for finite-sample exactness |
| Median paired difference | sign-based or direct median inference | independent pairs; continuity/tie handling |
| Symmetric paired location shift | Wilcoxon signed-rank | independent pairs; symmetry for standard location interpretation |
| Equality across several distributions | Kruskal–Wallis | independent groups; median interpretation needs common-shape structure |

The table is not a recipe.

It is a reminder that every method is attached to a target and an assumption set.

## Simulation should vary shape, not just location

A good comparison of parametric and rank-based procedures should not simulate only normal location shifts.

That design bakes in the location-shift interpretation.

A stronger simulation grid would vary

- location;
- variance;
- skewness;
- tail weight;
- sample-size imbalance;
- ties and discreteness;
- dependence;
- contamination;
- clustering.

Then evaluate

- Type I error for the **actual null being tested**;
- power for specified alternatives;
- bias of the chosen effect estimator;
- confidence-interval coverage;
- whether the procedure still targets the scientific estimand.

The last item is often omitted.

But it is the most important one.

## “Robust” and “nonparametric” are not synonyms

A procedure can be nonparametric and still be sensitive to

- dependence;
- unequal shapes;
- ties;
- zero inflation;
- informative sampling;
- asymmetry;
- the choice of scoring or ranking rule.

A parametric or semiparametric method can sometimes be more robust for the target estimand if it models the relevant failure mode directly.

So the useful question is not

> Which method has fewer assumptions?

It is

> Which assumptions are being made, and which of them matter for the estimand I care about?

That question is harder.

It is also statistically honest.

## The broader lesson

Statistics does not offer an assumption-free escape hatch.

Every inferential procedure gets its meaning from a combination of

$$
\boxed{
\text{estimand}
+
\text{sampling design}
+
\text{assumptions}
+
\text{procedure}.
}
$$

Rank transformations can remove sensitivity to some parametric distributional forms.

They do not remove the need to define what is being estimated or tested.

And they do not guarantee that a named “nonparametric alternative” answers the same question as the parametric procedure it replaces.

That is why the right starting point is not

> “Is my data normal?”

It is

$$
\boxed{
\text{What quantity am I trying to learn about the population?}
}
$$

Only after that question is fixed should we choose the test.

## References

- Wilcoxon F. Individual Comparisons by Ranking Methods. *Biometrics Bulletin*. 1945;1(6):80–83.
- Mann HB, Whitney DR. On a Test of Whether One of Two Random Variables Is Stochastically Larger than the Other. *Annals of Mathematical Statistics*. 1947;18(1):50–60.
- Kruskal WH, Wallis WA. Use of Ranks in One-Criterion Variance Analysis. *Journal of the American Statistical Association*. 1952;47(260):583–621.
- Lehmann EL. *Nonparametrics: Statistical Methods Based on Ranks*. Holden-Day; 1975.
- Hollander M, Wolfe DA, Chicken E. *Nonparametric Statistical Methods*. 3rd ed. Wiley; 2014.
- Fay MP, Proschan MA. Wilcoxon-Mann-Whitney or t-test? On assumptions for hypothesis tests and multiple interpretations of decision rules. *Statistical Surveys*. 2010;4:1–39. doi:10.1214/09-SS051.
- Hart A. Mann-Whitney test is not just a test of medians: differences in spread can be important. *BMJ*. 2001;323:391–393.
