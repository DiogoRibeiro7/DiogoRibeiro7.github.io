---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-02-02'
excerpt: A null hypothesis is a model restriction, not a statement to be accepted when a p-value is large. Statistical tests measure compatibility through a chosen statistic whose sensitivity depends on the alternative.
header:
  image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  og_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  twitter_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
keywords:
- null hypothesis
- p-values
- test statistics
- compatibility
- statistical power
seo_description: A rigorous explanation of null hypotheses, p-values, test statistics, power, omnibus versus directional alternatives, and why non-rejection is not evidence of equivalence.
seo_title: 'The Null Hypothesis: Compatibility, Power, and Test Sensitivity'
seo_type: article
summary: Statistical testing explained through model restrictions and test statistics, with emphasis on what p-values measure, why tests detect different departures, and why equivalence requires its own hypothesis.
tags:
- Hypothesis Testing
- Statistical Inference
- Statistics
title: 'The Null Hypothesis: Compatibility, Power, and Test Sensitivity'
---

A null hypothesis is not “nothing is happening.”

It is a mathematical restriction on the data-generating model.

Examples include

$$
H_0:\mu_1-\mu_0=0,
$$

$$
H_0:\beta=0,
$$

or

$$
H_0:F=F_0.
$$

A statistical test asks whether the observed data are sufficiently incompatible with that restriction according to a chosen test statistic.

The phrase **chosen test statistic** matters.

Different statistics detect different kinds of departure.

## A p-value is conditional on the null model

Let $T(X)$ be a statistic where larger values are more incompatible with $H_0$.

The p-value is

$$
p
=
P_{H_0}
\left[
T(X^\ast)
\ge
T(x_{\mathrm{obs}})
\right],
$$

with modifications for two-sided or discrete tests.

The probability is computed under the null model.

It is not

$$
P(H_0\mid X).
$$

Frequentist testing does not assign a posterior probability to the null hypothesis without adding a prior and a Bayesian model.

## Non-rejection is not acceptance

If

$$
p>0.05,
$$

the data did not cross the chosen rejection threshold.

That can occur because:

- the null is approximately true;
- the alternative effect is small;
- the sample is too small;
- the measurement is noisy;
- the test is insensitive to the actual departure;
- the design contains little information.

Therefore

$$
\text{fail to reject }H_0
\not\Rightarrow
\text{prove }H_0.
$$

## Equivalence is a different hypothesis

Suppose effects smaller than $\Delta$ are practically negligible.

An equivalence question is not

$$
H_0:\theta=0.
$$

A common equivalence formulation is

$$
H_0:
|\theta|
\ge
\Delta
$$

against

$$
H_1:
|\theta|
<
\Delta.
$$

Now rejection supports the claim that the effect lies within the equivalence margin.

This reverses the usual logic.

A non-significant ordinary superiority test is not an equivalence test.

## Test statistics encode sensitivity

Consider testing whether a distribution equals a reference $F_0$.

The Kolmogorov-Smirnov statistic uses

$$
D
=
\sup_x
|F_n(x)-F_0(x)|.
$$

It is especially sensitive to the largest CDF discrepancy.

Anderson-Darling applies greater tail weight.

Shapiro-Wilk uses ordered observations and is designed specifically for normality.

These procedures can produce different p-values from the same sample because they weight departures differently.

That is not a contradiction.

They are asking different discrepancy questions.

## Omnibus tests trade specificity for breadth

An omnibus test can detect many kinds of departure.

The price is that rejection often does not identify the mechanism.

For example, a chi-square test of independence can reject because several cells differ from expectation.

The scalar statistic does not say which association pattern matters.

A residual analysis or targeted contrast is needed afterward.

Targeted tests can have greater power when the alternative is specified in advance.

## Power depends on the alternative

For parameter value $\theta$, define

$$
\pi(\theta)
=
P_\theta(\text{reject }H_0).
$$

This is the power function.

There is no single number called “the power” unless a particular alternative or distribution over alternatives is specified.

A normality test can have high power against heavy tails and lower power against a mild mixture.

A mean test can have almost no power against a pure variance change.

Sensitivity follows the statistic.

## Large samples detect small departures

Suppose the null is not exactly true but differs only slightly from reality.

As $n$ increases, many consistent tests will eventually reject.

This can produce

$$
p\ll0.001
$$

for a discrepancy too small to matter scientifically.

Statistical significance therefore does not measure practical importance.

Effect size and uncertainty remain necessary.

## Small samples hide meaningful departures

The reverse problem occurs in small samples.

A large clinically relevant effect can remain non-significant because the estimate is imprecise.

This is why interpretation should include the confidence interval.

An interval such as

$$
[-0.2,1.8]
$$

contains zero but also contains effects that may be scientifically important.

The p-value alone hides that.

## Model assumptions sit underneath the test

A p-value is valid only under the assumptions used to derive its null distribution.

Those may include:

- independence;
- randomization;
- distributional form;
- variance assumptions;
- censoring assumptions;
- asymptotic approximation;
- correct model specification.

A precisely computed p-value from the wrong null distribution is still wrong.

## Multiple testing changes the reference problem

If a researcher tries many analyses and reports only the smallest p-value, the nominal null calibration no longer describes the selection procedure.

The relevant probability becomes conditional on the entire search process.

Multiplicity corrections, selective-inference methods, preregistration, or held-out confirmation can address different parts of this problem.

The unit of inference is the procedure, not one number extracted after the fact.

## Bayesian inference answers another question

Bayesian inference assigns a prior

$$
p(\theta)
$$

and updates it through the likelihood:

$$
p(\theta\mid y)
\propto
p(y\mid\theta)p(\theta).
$$

Now posterior probabilities such as

$$
P(\theta>0\mid y)
$$

are meaningful inside the Bayesian model.

That is not a repaired p-value.

It is a different inferential framework with different assumptions.

## Confidence intervals and tests are linked

For many standard procedures, a two-sided level-$\alpha$ test rejects

$$
H_0:\theta=\theta_0
$$

exactly when the corresponding

$$
100(1-\alpha)\%
$$

confidence interval excludes $\theta_0$.

The interval is usually more informative because it displays a range of parameter values compatible with the procedure.

It still should not be interpreted as a posterior probability interval.

## A practical reporting standard

Instead of writing only

> $p=0.03$, statistically significant,

report:

1. the estimand;
2. the effect estimate;
3. the uncertainty interval;
4. the test and statistic;
5. the p-value if relevant;
6. assumptions and design;
7. practical or scientific scale.

This makes it much harder for the threshold to replace the scientific argument.

## Conclusion

A null hypothesis is a restriction on a model.

A test measures discrepancy from that restriction through a chosen statistic.

The correct chain is

$$
\boxed{
\text{null model}
\rightarrow
\text{test statistic}
\rightarrow
\text{null distribution}
\rightarrow
\text{p-value}
\rightarrow
\text{effect and uncertainty}.
}
$$

Non-rejection is not proof of the null.

Rejection is not proof of importance.

The value of a test comes from knowing what departure it is capable of detecting and what scientific claim that departure represents.

## References

- Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society A*, 231, 289–337.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 70(2), 129–133.
- Greenland, S., Senn, S. J., Rothman, K. J., et al. (2016). Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations. *European Journal of Epidemiology*, 31, 337–350.
