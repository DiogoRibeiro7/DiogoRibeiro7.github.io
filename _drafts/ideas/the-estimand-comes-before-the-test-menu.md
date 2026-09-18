---
author_profile: false
categories:
- Statistics
classes: wide
title: 'The Estimand Comes Before the Test Menu'
excerpt: The same four paired observations can imply a higher mean, a lower median change, and improvement for most units. Decide which question matters before choosing a test.
header:
  image: /assets/images/headers/photo-statistics-ecdf.jpg
  og_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-ecdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-ecdf.jpg
keywords:
- estimands
- paired data
- repeated measures
- experimental design
- statistical inference
seo_title: 'The Estimand Comes Before the Test Menu'
seo_description: 'Worked paired-data and repeated-measure examples show why population, outcome, weighting, and comparison must be defined before choosing a statistical test.'
seo_type: article
summary: 'A concrete analysis of how means, medians, probabilities of improvement, and observation weights answer different questions even on the same dataset.'
tags:
- Estimands
- Experimental Design
- Statistical Inference
- Repeated Measures
why_this_exists: 'Test-selection guides often leave the scientific target implicit. Two small counterexamples make the consequences of that omission calculable.'
evidence: 'Original four-workload paired example, a permutation preserving both marginals, and an unequal-session-count example; all values are constructed.'
methodology: 'Compute competing summaries exactly, identify which require joint observations, and derive the difference between equal-person and equal-observation weighting.'
reviewed_at: 2026-09-18
---

<!--
Development contract
Question: Which decisions must precede the choice of a statistical test?
Claim: The target population, outcome, contrast, weighting, and observation design determine the question a test can answer.
Counterclaim: A standard test menu can be useful once those decisions are fixed.
Evidence object: Exact paired-data and repeated-measure counterexamples with executable arithmetic.
Failure case: Naming a causal or individual-effect estimand does not make it identifiable from the available data.
Reader payoff: Write a complete target statement and check whether the design supplies the required information.
Exclusions: A catalogue of tests, a full missing-data treatment, and a general causal-identification tutorial.
-->

Suppose a new implementation is faster on three out of four workloads, but its average runtime is worse. Has it improved performance?

The answer depends on whether we care about total computing time, the change for a typical workload, the fraction of workloads helped, or the upper tail of the runtime distribution. A significance test cannot make that choice for us.

Many test-selection guides begin with the number of groups and the type of variable. Those facts matter, but they do not specify the quantity we want to learn. The earlier decision is the **estimand**: the population quantity that would answer the substantive question if we knew it exactly.

An **estimator** is a rule for learning that quantity from data. An **estimate** is the value produced by that rule. A **test** assesses a stated hypothesis under a reference distribution. Confusing these objects makes it easy to obtain a precise answer without settling what the answer means.

## One table, several legitimate answers

Consider these constructed runtimes, in seconds, for four workloads measured under two implementations. Treat each row as a genuine pair. For the arithmetic below, every workload receives equal weight.

| Workload | Old implementation | New implementation | New minus old |
| --- | --- | --- | --- |
| A | 10 | 9 | -1 |
| B | 20 | 80 | +60 |
| C | 30 | 29 | -1 |
| D | 40 | 39 | -1 |

Negative changes mean faster execution. Several statements are simultaneously true:

| Summary | Value | What it describes |
| --- | --- | --- |
| Difference in means | +14.25 seconds | Change in average runtime |
| Difference in marginal medians | +9 seconds | Change between the two median runtimes |
| Median of paired changes | -1 second | Middle of the within-workload change distribution |
| Fraction of pairs improved | 75% | Share of these workloads that became faster |
| Cross-sample probability of a faster new runtime | 43.75% | Comparison of independently selected workload runtimes |

These are descriptive calculations on four invented pairs, not significance claims or evidence about a real system. Sample medians use the usual midpoint convention for even sample sizes.

The difference in means follows directly:

$$
\bar Y-\bar X=39.25-25=14.25.
$$

The median paired change is $-1$, while the new marginal median is $34$ and the old marginal median is $25$. Thus

$$
\operatorname{median}(Y-X)
\ne
\operatorname{median}(Y)-\operatorname{median}(X).
$$

Expectation is linear; the median is not. A test or interval for one of these median quantities does not automatically answer a question about the other.

For an operator paying for total computing time across this fixed workload mix, the mean increase is directly relevant. For a developer asking how often a workload benefits, the 75 percent improvement fraction answers a different useful question. Neither makes the 60-second regression disappear.

This is why replacing one summary with another should be a scientific decision. Choosing whichever gives the most favorable conclusion changes the claim after seeing the data.

## “A random unit improves” requires a joint distribution

Write $X$ for the old runtime and $Y$ for the new runtime of the **same** workload. The probability of a within-workload improvement is

$$
\theta_{\mathrm{paired}}=P(Y<X).
$$

Now independently draw a workload for the new implementation and another for the old implementation. If their runtimes are $Y'$ and $X'$, the comparison is

$$
\theta_{\mathrm{independent}}=P(Y'<X').
$$

The second quantity depends only on the two marginal distributions. The first depends on how the outcomes are paired.

In the table, the new value 9 is faster than all four old values; 80 is faster than none; 29 is faster than two; and 39 is faster than one. Among the 16 equally weighted cross-comparisons, seven favor the new implementation:

$$
\widehat\theta_{\mathrm{independent}}=\frac7{16}=0.4375.
$$

This does not contradict the paired improvement fraction of $3/4$. Independent comparisons mix implementation differences with differences between workloads. There are no ties in this example; an application with ties must specify whether to count them separately, as half a win, or otherwise.

A direct identification check is to keep both columns' values but reassign the new runtimes to workloads:

| Workload | Old runtime | Reassigned new runtime |
| --- | --- | --- |
| A | 10 | 80 |
| B | 20 | 9 |
| C | 30 | 39 |
| D | 40 | 29 |

Every marginal summary stays the same, including the mean difference and the 43.75 percent independent-comparison probability. Only two of the four pairs now improve.

These are two possible joint datasets with identical marginals and different within-unit improvement rates. Consequently, observing only the marginals cannot generally identify the improvement rate for the same units.

The arithmetic is reproducible with standard-library Python:

```python
from statistics import mean, median

old = [10, 20, 30, 40]
new = [9, 80, 29, 39]
reassigned = [80, 9, 39, 29]

def summaries(x, y):
    differences = [b - a for a, b in zip(x, y)]
    return {
        "mean difference": mean(differences),
        "difference of medians": median(y) - median(x),
        "median difference": median(differences),
        "paired improvement": mean(b < a for a, b in zip(x, y)),
        "independent comparison": mean(b < a for a in x for b in y),
    }

print("original", summaries(old, new))
print("reassigned", summaries(old, reassigned))
```

The first dictionary gives the five values in the summary table. The second preserves all of them except the paired improvement fraction, which becomes 0.5. In this particular reassignment even the median paired change happens to remain $-1$; preservation of that median is not a general property of reassigning pairs.

## Pairing changes information, but not every target

There is an important qualification. The mean difference satisfies

$$
E[Y-X]=E[Y]-E[X]
$$

whenever the expectations exist. Genuine pairing and independent sampling can therefore support estimation of the same difference in marginal means, provided they refer to the same target populations and conditions.

What changes is the available information and the sampling variance. For $n$ independent pairs,

$$
\operatorname{Var}(\bar Y-\bar X)
=\frac{\sigma_Y^2+\sigma_X^2-2\operatorname{Cov}(X,Y)}n.
$$

Ignoring useful positive within-pair covariance can discard precision. Inventing pairs by sorting unrelated observations does not create the design needed to use that covariance. The [paired-versus-independent design article](/statistics/paired_vs_independent_samples_hypothesis_testing/) discusses that distinction in more detail.

Thus “choose the estimand first” does not mean that every change of test changes the estimand. Some procedures target the same quantity with different assumptions or efficiency. Others target different quantities. We need to establish which situation applies.

## Repeated measurements introduce a weighting decision

Suppose participants contribute unequal numbers of sessions. Let $m_i$ be the number of sessions for participant $i$, and let $\bar Y_i$ be that participant's mean outcome.

Two common summaries are

$$
\widehat\mu_{\mathrm{person}}
=\frac1N\sum_{i=1}^N\bar Y_i
$$

and

$$
\widehat\mu_{\mathrm{session}}
=\frac{\sum_i m_i\bar Y_i}{\sum_i m_i}.
$$

The first gives every person the same weight. The second gives every observed session the same weight. Under appropriate sampling assumptions, they estimate an average for a randomly selected person and an average for a randomly selected session, respectively. Those are different populations of units.

Consider two participants. One always takes 10 seconds to complete a task; the other always takes 100 seconds. Before a process change, the faster participant contributes 90 sessions and the slower participant 10. After the change, their counts reverse.

| Period | Fast participant's sessions | Slow participant's sessions | Session-weighted mean | Equal-person mean |
| --- | --- | --- | --- | --- |
| Before | 90 | 10 | 19 seconds | 55 seconds |
| After | 10 | 90 | 91 seconds | 55 seconds |

No participant became slower. The observed session mix changed. The 72-second difference is a valid description of the pooled sessions, but it is not a within-person slowdown.

```python
from statistics import mean

def session_mean(counts, participant_means):
    return sum(n * y for n, y in zip(counts, participant_means)) / sum(counts)

participant_means = [10, 100]
for counts in ([90, 10], [10, 90]):
    print("sessions", session_mean(counts, participant_means),
          "people", mean(participant_means))
```

A standard-error correction for clustering addresses dependence. It does not, by itself, turn the session-weighted estimand into the person-weighted one. Both the estimator's weights and its uncertainty calculation need to match the intended target.

Equal-person weighting also has limits. If participants are observed preferentially during their best or worst periods, averaging their observed sessions may fail to estimate their average over the intended time window. We must specify the observation process as well as the unit weights.

## Time and missing outcomes belong in the question

“Did the intervention improve the trajectory?” still leaves several estimands open: the outcome at week 12, the average over weeks 1–12, the slope over that interval, or the probability of crossing a threshold before week 12.

These quantities can disagree. An early improvement followed by convergence can change the average trajectory without changing the final outcome. An average slope can hide a sharp initial jump. A fitted model does not choose which feature is scientifically relevant.

We must also say what a missing outcome means for the target. An implementation that times out has not produced an unobserved ordinary runtime in quite the same sense as a measurement accidentally lost by the logger. Excluding timeouts estimates performance among completed jobs. Assigning a fixed penalty defines a different outcome. Modelling completion time with censoring introduces assumptions about what the timeout threshold reveals.

Clinical research formalizes a related discipline in the ICH estimand framework: define the treatment conditions, population, outcome, handling of intercurrent events, and population-level summary so the analysis aligns with the intended question. That framework is useful context, although the examples here concern computational and behavioral measurements. [ICH E9(R1), *Estimands and Sensitivity Analysis in Clinical Trials*](https://www.fda.gov/media/148473/download).

## A causal estimand needs an identification argument

Writing a precise target is necessary, but it cannot supply information the design lacks.

For a causal question, we might define potential outcomes $Y(1)$ and $Y(0)$ and target

$$
\tau=E[Y(1)-Y(0)].
$$

Random assignment, consistency, an appropriate treatment definition, and assumptions about interference can support identification of an average causal effect. Observational data require an additional argument about treatment assignment and confounding. The distinction between defining a causal quantity and identifying it from observed data is developed in [Hernán and Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook).

Even in a randomized parallel-group experiment, the probability

$$
P\{Y(1)<Y(0)\}
$$

usually requires more information or assumptions than the average effect. Each person supplies only one of the two potential outcomes; randomization identifies marginal distributions, not their unobserved coupling. The reassigned-workload example shows the algebraic problem in miniature.

A before-and-after comparison has a further limitation: elapsed time, learning, changing workloads, and other interventions can explain the observed change. Genuine repeated measurements establish pairing. They do not automatically establish a causal comparison.

## Write the target in a form that can be challenged

Before opening the test menu, fill in a short specification:

| Decision | Example specification |
| --- | --- |
| Population and unit | Workloads drawn from a defined production mix, with weights fixed before evaluation |
| Conditions | Old and new implementations under the same documented hardware and load protocol |
| Outcome and horizon | Runtime per workload, with an explicit timeout rule |
| Summary and contrast | Weighted mean of new minus old runtime |
| Observation design | Both implementations measured on each workload, with execution order randomized |
| Dependence | Repeated runs nested within workloads; workloads are the resampling units |
| Practical decision | Require evidence of a worthwhile average gain and an acceptable regression rate |

This specification still needs numerical thresholds and defensible assumptions. Its value is that disagreements now have a location. Someone may prefer equal-workload weighting to production weighting, or a tail constraint to a mean criterion, and the resulting change in the question is visible.

Only then choose an estimator, its uncertainty calculation, and any hypothesis test. A paired mean analysis, a sign-based analysis, and a distributional comparison would be candidates for different parts of this example, not interchangeable buttons for the same question.

The strongest objection is practical: standard tests already work well for many routine problems. They do, when the target and design are understood. A menu is a useful shortcut after those decisions. It cannot supply them.

For the four workloads, an honest result would report both the average regression and the fraction improved, then connect the action to a declared objective. The difficult choice is deciding which consequences matter. That choice should be visible before a p-value is calculated.
