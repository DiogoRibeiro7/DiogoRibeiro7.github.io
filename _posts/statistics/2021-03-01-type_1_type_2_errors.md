---
permalink: '/statistics/type_1_type_2_errors/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2021-03-01'
excerpt: Type I and Type II errors are properties of a decision rule, but applied work also needs effect size, multiplicity control, selective inference, and the costs of the decisions that follow.
header:
  image: /assets/images/headers/photo-statistics-regression-errors.jpg
  og_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-regression-errors.jpg
  twitter_image: /assets/images/headers/photo-statistics-regression-errors.jpg
keywords:
- Type I error
- Type II error
- statistical power
- multiple testing
- decision theory
seo_description: A practical extension of Type I and Type II error theory covering power functions, minimum relevant effects, multiplicity, selective inference, decision costs, and why model-classification errors are not the same thing.
seo_title: 'Beyond Type I and Type II Errors: Decisions, Power, and Multiplicity'
seo_type: article
summary: A companion to the introductory Type I/II article, focusing on how error rates behave in real research workflows with multiple tests, effect-size thresholds, sequential analysis, and operational decisions.
tags:
- Hypothesis Testing
- Statistical Inference
- Decision Theory
title: 'Beyond Type I and Type II Errors: Decisions, Power, and Multiplicity'
---

The basic Type I and Type II error table is useful. It is not enough for real analysis. A hypothesis test is embedded in a larger workflow:

$$
\boxed{
\text{scientific question}
\rightarrow
\text{estimand}
\rightarrow
\text{test}
\rightarrow
\text{selection}
\rightarrow
\text{decision}.
}
$$

Error rates defined for one prespecified test can change when we search across outcomes, models, subgroups, and stopping times. This article focuses on those practical extensions.

## Error probabilities belong to a procedure

Let $R(X)$ be the rule that decides whether to reject a null hypothesis. For parameter value $\theta$, define

$$
\pi(\theta)
=
P_\theta(
R(X)=1
).
$$

This is the power function. When $\theta$ lies in the null parameter space,

$$
\pi(\theta)
$$

is a Type I rejection probability. When $\theta$ lies in the alternative,

$$
1-\pi(\theta)
$$

is the Type II error probability. There is usually no single Type II error rate without specifying the alternative value.

## Nominal alpha and actual size

A test advertised at

$$
\alpha=0.05
$$

is designed so that

$$
\sup_{\theta\in\Theta_0}
P_\theta(
\text{reject}
)
\le
0.05
$$

or approximately so under the intended assumptions. The actual rejection probability can be lower for conservative discrete tests or vary across a composite null. Therefore nominal $\alpha$ is a design level, not a universal empirical false-positive fraction.

## Power depends on effect size

Suppose a two-sided test concerns

$$
H_0:\theta=0.
$$

Power is not one number. It is a function

$$
\pi(\theta).
$$

A study can have:

- low power for $\theta=0.1$;
- moderate power for $\theta=0.5$;
- high power for $\theta=1.0$.

Writing only

> power = 80%

is incomplete unless the alternative effect is specified.

## Minimum effect of interest

Study design should be tied to an effect that matters scientifically. Let

$$
\Delta
$$

be the smallest effect worth detecting. Then a sensible power target is

$$
P_{\theta=\Delta}
(
\text{reject }H_0
)
\ge
0.80
$$

or another prespecified level. Choosing $\Delta$ after the sample-size calculation reverses the logic. The effect should come from the scientific or operational problem.

## Statistical significance is not decision significance

With enough observations, an arbitrarily small nonzero effect can become statistically detectable. That does not increase the Type I error rate. It increases the ability to reject a false null. The problem is interpretation, not false-positive calibration. A large sample can produce

$$
p<10^{-6}
$$

for an effect too small to matter. This is why the previous version's statement that large samples “increase Type I error by detecting irrelevant effects” was wrong.

## Multiplicity changes the error target

Suppose $m$ true null hypotheses are tested independently at level $\alpha$. The probability of at least one false rejection is

$$
1-(1-\alpha)^m.
$$

At

$$
m=20,
\qquad
\alpha=0.05,
$$

this is about 0.64. The individual tests remain level 0.05. The family-level procedure does not. This is why multiple testing requires its own error criterion.

## Family-wise error rate

Let $V$ be the number of false rejections. Family-wise error is

$$
FWER
=
P(V\ge1).
$$

Bonferroni and Holm procedures control this criterion. This can be appropriate when even one false claim in the family is costly.

## False discovery rate

When many exploratory hypotheses are tested, a less stringent target may be the expected false-discovery proportion:

$$
FDR
=
E
\left[
\frac{
V
}{
\max(R,1)
}
\right],
$$

where $R$ is the total number of rejections. Benjamini-Hochberg controls FDR under its dependence conditions. FWER and FDR solve different problems.

## Optional stopping

Suppose a researcher checks a conventional p-value after every new group of observations and stops the first time

$$
p<0.05.
$$

The overall probability of eventually crossing 0.05 under the null can exceed 5%. The problem is not that p-values “stop working.” The procedure has changed. Valid sequential analysis requires a design built for repeated looks, such as:

- group-sequential boundaries;
- alpha-spending functions;
- always-valid p-values or confidence sequences.

The stopping rule is part of the inferential procedure.

## Model and subgroup search

If ten regression specifications and twenty subgroups are examined and only the most significant result is reported, the selected coefficient no longer has the same sampling behavior as a coefficient from one prespecified model. This creates selection bias. The reported effect tends to be exaggerated because the analysis selected an extreme estimate.

This is one form of the winner's curse. Preregistration, held-out confirmation, selective-inference methods, or explicit multiplicity adjustments address different versions of this problem.

## Type S and Type M errors

When power is low, conditional on achieving significance, the estimated effect can be badly distorted. Gelman and Carlin distinguish:

- **Type S error:** the estimated effect has the wrong sign;
- **Type M error:** the magnitude is strongly exaggerated.

These are not replacements for Type I and Type II errors. They describe another practical consequence of noisy, selected estimates.

## Equivalence and non-inferiority

A failed superiority test does not demonstrate no meaningful effect. Suppose effects with absolute magnitude below

$$
\Delta
$$

are negligible. An equivalence analysis can test

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

The hypothesis is deliberately reversed. To support absence of a practically important difference, design the study around equivalence from the beginning.

## Error rates and decision costs

Statistical error rates do not tell us the cost of an error. Suppose a false positive costs

$$
c_I
$$

and a false negative costs

$$
c_{II}.
$$

A simplified expected loss is

$$
L
=
c_I
P(\text{Type I})
+
c_{II}
P(\text{Type II}).
$$

Different domains imply different costs. A screening stage may tolerate more false positives. A confirmatory licensing decision may not. The conventional 0.05 threshold is not a universal solution to those costs.

## Classification errors are related but not identical

In binary classification,

$$
FPR
=
P(
\hat Y=1
\mid
Y=0
)
$$

and

$$
FNR
=
P(
\hat Y=0
\mid
Y=1
).
$$

These resemble Type I and Type II errors structurally. But overfitting is **not** itself a Type I error, and underfitting is **not** itself a Type II error. Overfitting describes a failure of generalization. A false positive is a case-level classification error. The earlier version of this article incorrectly equated those concepts.

## Cross-validation does not control hypothesis-test alpha

Cross-validation estimates predictive performance when used correctly. It does not automatically reduce Type I and Type II error probabilities of a scientific hypothesis test. If model selection and inference use the same resampling results without accounting for selection, inferential calibration can still fail. Prediction error and hypothesis-test error are different targets.

## Randomization addresses bias, not every test error directly

Randomization can create exchangeability between treatment groups and protect against confounding in expectation. Blinding can reduce measurement and behavioral biases. These design improvements can increase validity and sometimes power. But saying they “reduce Type I and Type II errors” without specifying the test is too broad. Design affects the data-generating structure.

Test error rates are then derived under that design.

## Sample size

Increasing sample size generally increases power for a fixed nonzero alternative while keeping the nominal Type I error level fixed when the test is correctly calibrated. It does **not** inherently increase Type I error. What changes is the ability to detect smaller departures from the null. Therefore:

$$
\boxed{
\text{large }n
\rightarrow
\text{high sensitivity},
}
$$

not

$$
\text{large }n
\rightarrow
\text{more false positives}.
$$

Practical significance must be handled with effect-size criteria, not by deliberately keeping the study small.

## Reporting

A useful confirmatory report states:

- primary estimand;
- null and alternative hypotheses;
- significance criterion;
- multiplicity strategy;
- effect estimate;
- uncertainty interval;
- prespecified minimum relevant effect;
- power assumptions;
- whether analyses were selected after seeing the data.

The decision threshold should be visible rather than implicit.

## Conclusion

Type I and Type II errors are only the first layer of statistical decision-making. Applied inference also has to manage

$$
\boxed{
\text{effect size}
+
\text{power}
+
\text{multiplicity}
+
\text{stopping}
+
\text{selection}
+
\text{decision cost}.
}
$$

A single $\alpha=0.05$ threshold cannot solve all six.

## References

- Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society A*, 231, 289–337.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S and Type M errors. *Perspectives on Psychological Science*, 9(6), 641–651.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
