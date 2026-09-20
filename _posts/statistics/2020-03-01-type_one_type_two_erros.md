---
permalink: '/statistics/type_one_type_two_erros/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-03-01'
excerpt: Type I and Type II errors are properties of decision rules under specified parameter values. Their trade-off depends on the significance level, sample size, effect size, and test design.
header:
  image: /assets/images/headers/photo-statistics-dice-coins.jpg
  og_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-dice-coins.jpg
  twitter_image: /assets/images/headers/photo-statistics-dice-coins.jpg
keywords:
- Type ii error
- False positive
- False negative
- Hypothesis testing
- Type i error
- Statistical power
seo_description: Type I and Type II errors explained through test size, power functions, effect size, and study design rather than fixed 5% rules.
seo_title: Type I and Type II Errors, Size, and Power
seo_type: article
summary: A mathematical guide to Type I and Type II errors that distinguishes nominal significance levels from actual test size and shows how power changes with effect size and sample size.
tags:
- Hypothesis Testing
- Model Evaluation
title: 'Type I and Type II Errors: Size, Power, and Study Design'
---

Type I and Type II errors are often taught with a two-by-two table.

That table is useful, but it hides an important fact: the error probabilities belong to a **decision rule under specified parameter values**. They are not universal constants attached to a scientific question.

Suppose a test concerns a parameter $\theta$ and divides the parameter space into

$$
H_0:\theta\in\Theta_0
$$

and

$$
H_1:\theta\in\Theta_1.
$$

Let $R$ denote the rejection region. The probability of rejecting at a particular parameter value is the **power function**

$$
\pi(\theta)
=
P_\theta(X\in R).
$$

This one function contains both kinds of error.

For $\theta\in\Theta_0$,

$$
\pi(\theta)
$$

is the probability of a Type I error.

For $\theta\in\Theta_1$,

$$
1-\pi(\theta)
$$

is the probability of a Type II error.

That is more precise than saying that every test has one fixed false-positive rate and one fixed false-negative rate.

## Type I error and the size of a test

A Type I error occurs when the procedure rejects $H_0$ even though the data-generating parameter lies in the null space.

A level-$\alpha$ test is designed so that

$$
\sup_{\theta\in\Theta_0}
P_\theta(\text{reject }H_0)
\le \alpha.
$$

The quantity

$$
\sup_{\theta\in\Theta_0}
P_\theta(\text{reject }H_0)
$$

is the **size** of the test.

For some continuous tests, the size is exactly $\alpha$.

For discrete tests, conservative tests, or composite null hypotheses, the actual rejection probability can be smaller than the nominal level at some or all null parameter values.

So the common sentence

> $\alpha=0.05$ means there is exactly a 5% chance of a false positive

is too broad.

A better statement is:

> The procedure is calibrated so that its Type I error probability is controlled at, or approximately at, 5% under the null assumptions.

That calibration is conditional on the model being correct.

## Type II error is indexed by the alternative

A Type II error occurs when the procedure fails to reject $H_0$ at a parameter value in the alternative.

Its probability is

$$
\beta(\theta)
=
P_\theta(\text{fail to reject }H_0),
\qquad
\theta\in\Theta_1.
$$

Therefore,

$$
\beta(\theta)
=
1-\pi(\theta).
$$

There is usually no single value called "the Type II error rate" unless a particular alternative has been specified.

A test can have very low power against a tiny effect and very high power against a large effect.

That dependence on effect size is central to study design.

## A concrete normal-mean example

Suppose

$$
X_1,\ldots,X_n
\overset{\mathrm{iid}}{\sim}
\mathcal N(\mu,\sigma^2),
$$

with known $\sigma$, and consider the one-sided test

$$
H_0:\mu=0
$$

against

$$
H_1:\mu>0.
$$

Reject when

$$
Z
=
\frac{\bar X}{\sigma/\sqrt n}
>
z_{1-\alpha}.
$$

Under $\mu=0$,

$$
Z\sim\mathcal N(0,1),
$$

so the Type I error probability is exactly

$$
P_0(Z>z_{1-\alpha})
=
\alpha.
$$

Under an alternative $\mu=\mu_1>0$,

$$
Z
\sim
\mathcal N
\left(
\frac{\mu_1\sqrt n}{\sigma},
1
\right).
$$

The power is therefore

$$
\pi(\mu_1)
=
1-
\Phi
\left(
z_{1-\alpha}
-
\frac{\mu_1\sqrt n}{\sigma}
\right).
$$

This equation shows the design trade-offs directly.

Power rises when:

- the effect $\mu_1$ is larger;
- the sample size $n$ is larger;
- the measurement noise $\sigma$ is smaller;
- the significance level $\alpha$ is larger.

## Lowering alpha does not create a universal inverse relationship

For a fixed sample size, effect size, model, and test statistic, lowering $\alpha$ makes the rejection region harder to enter. Power therefore decreases and Type II error increases.

But it is misleading to say that Type I and Type II errors are intrinsically inversely related.

Sample size can reduce Type II error while leaving the nominal Type I level unchanged.

Better measurement can increase power without changing $\alpha$.

A more efficient design can do the same.

The trade-off exists **conditional on the rest of the design**.

## Power should be tied to a scientifically meaningful effect

A statement such as

> the study has 80% power

is incomplete.

Power against what effect?

A defensible design specifies a minimum effect of scientific or practical interest, say $\Delta$, and targets

$$
P_{\theta=\Delta}(\text{reject }H_0)
\ge 1-\beta.
$$

The choice of $\Delta$ should come from the scientific or operational problem, not from whichever effect size makes the desired sample size convenient.

This matters because a very large study can have high power to detect effects that are too small to matter.

## False positive is useful shorthand, but not always literal classification

The language of false positives and false negatives is intuitive:

| Decision | Null state | Alternative state |
|---|---|---|
| Reject $H_0$ | Type I error | Correct rejection |
| Fail to reject $H_0$ | Correct non-rejection | Type II error |

But hypothesis testing is not always a binary diagnostic classifier.

A composite null can contain many parameter values. The actual Type I error probability may vary across them.

Likewise, the alternative is often a continuum, so the Type II error probability is a function rather than one number.

The table is a summary of the decision logic, not the whole statistical model.

## Failing to reject is not accepting the null

If a test produces

$$
p>0.05,
$$

the correct conclusion is usually that the data did not provide sufficient evidence to reject the null at the chosen level.

It does not follow that the null has been shown to be true.

The study may simply have little power for scientifically relevant alternatives.

This is why non-significant results should be accompanied by effect estimates and uncertainty intervals.

An equivalence or non-inferiority question requires a test designed for that question, not a failed conventional significance test.

## Choosing alpha is a decision problem

The conventional value

$$
\alpha=0.05
$$

is not a law of nature.

The acceptable Type I error rate depends on the consequences of the decision and on the regulatory or scientific context.

In a confirmatory clinical trial, false evidence of efficacy can have substantial costs.

In an early screening stage, a larger false-positive rate may be tolerable if the purpose is to avoid discarding promising candidates too early.

The statistical design should make those consequences explicit.

## Multiple testing changes the error problem

If many hypotheses are tested, controlling each test at

$$
\alpha=0.05
$$

does not generally control the probability of one or more false rejections across the whole family.

The relevant error criterion may instead be:

- family-wise error rate;
- false discovery rate;
- per-comparison error rate;
- or a domain-specific decision loss.

Type I error control therefore has to be defined at the level of the inferential procedure being used.

## Report estimates, not only decisions

A hypothesis-test decision compresses a continuous data set into one bit:

$$
\text{reject}
\quad\text{or}\quad
\text{do not reject}.
$$

That is rarely enough.

A useful report includes:

- the estimated effect;
- a confidence or credible interval;
- the test statistic and p-value when relevant;
- the effect size used in the power calculation;
- the planned Type I error criterion;
- and the sample-size assumptions.

The error rates belong to the procedure.

The scientific interpretation belongs to the estimated effect and its uncertainty.

## Conclusion

Type I and Type II errors are not merely two boxes in a table.

They are features of a decision rule over a parameter space.

The central object is the power function

$$
\pi(\theta)
=
P_\theta(\text{reject }H_0).
$$

Under the null, it describes Type I error.

Under the alternative, it determines Type II error through

$$
\beta(\theta)=1-\pi(\theta).
$$

This makes the main lesson clear: hypothesis testing is a design problem. The significance level, target effect, sample size, measurement quality, multiplicity structure, and consequences of each decision should be specified together.

## References

- Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society A*, 231, 289–337.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 70(2), 129–133.
