---
permalink: '/science-communication/read_the_starting_risk_before_the_percentage/'
title: 'Read the Starting Risk Before the Percentage'
date: '2026-06-18'
categories:
- Science Communication
tags:
- Risk Communication
- Health News
- Scientific Literacy
- Statistics
author_profile: false
classes: wide
seo_title: 'What a 50 Percent Risk Reduction Actually Tells You'
seo_description: 'Two worked examples show why the same relative risk reduction can mean ten fewer events or one fewer event per thousand people over the same period.'
seo_type: article
excerpt: >-
  A headline promising a 50% risk reduction leaves the starting risk unstated.
  Two hypothetical examples show how the same percentage can describe very
  different absolute changes.
summary: >-
  Natural frequencies and a shared-scale figure compare two populations with
  the same relative reduction. The article then separates units, time windows,
  uncertainty, and causal evidence when interpreting a health headline. Worked
  odds-ratio, sampling-error, and population-mixture examples show how a
  numerically correct comparison can still answer the wrong question.
keywords:
- relative risk reduction
- absolute risk reduction
- health headlines
- percentage points
- natural frequencies
why_this_exists: >-
  Readers need a repeatable way to reconstruct the quantities hidden behind
  a large percentage. This article works through two full examples and shows
  what remains unknown even after the arithmetic is correct.
evidence: >-
  Original hypothetical five-year risk tables, an original shared-scale figure,
  odds and uncertainty calculations, a population-mixture counterexample,
  and primary explanations from the National Cancer Institute and FDA.
methodology: >-
  Express both comparison groups per 1,000 people over five years, calculate
  relative and absolute changes, contrast risks with odds, and distinguish
  assumed probabilities from observed sample proportions. Standardise an
  invented mixture to demonstrate how differing baseline risks affect comparisons.
reviewed_at: '2026-09-18'
header:
  image: /assets/images/headers/photo-statistics-dice-coins.jpg
  og_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-dice-coins.jpg
  twitter_image: /assets/images/headers/photo-statistics-dice-coins.jpg
---

<!--
Development contract
Question: How much information is missing from a headline reporting a 50% reduction in risk?
Claim: The starting risk, population, outcome, and time window are needed to express the absolute change.
Counterclaim: Relative effects can be useful comparisons when reported alongside those details.
Evidence object: Hypothetical risk and odds tables, a standardisation example, a sampling-error calculation, and an original figure.
Failure case: Correct risk arithmetic does not establish causality, certainty, or transportability to another population.
Reader payoff: Translate a risk headline into events per a common denominator over a defined period.
Exclusions: Personal treatment choices, a real intervention's effectiveness, and a full survival-analysis tutorial.
-->

“The risk fell by 50%” sounds informative. It tells us that one risk is half another, if the percentage really is a relative reduction. It does not tell us what either risk was.

Halving a risk of 2% gives 1%. Halving a risk of 0.2% gives 0.1%. Both changes can support the same headline, while describing different numbers of events in the same-sized population.

The National Cancer Institute distinguishes absolute risk over a stated period from relative risk, which compares risks between groups. Keeping those quantities separate is the first step in reading the headline. [NCI explanation of risk measures](https://www.cancer.gov/about-cancer/screening/patient-screening-overview-pdq).

The following examples use invented risks. They do not describe a medicine, screening programme, or clinical trial.

## Give both groups the same denominator

Suppose a hypothetical intervention changes the five-year risk of a specified event. For Population A, the comparison risk is 20 events per 1,000 people, and the intervention risk is 10 per 1,000.

That is a change from 2% to 1%.

The **relative reduction** is the decrease divided by the starting risk:

$$
\frac{20-10}{20}=0.50=50\%.
$$

The **absolute reduction** is the difference between the two risks:

$$
\frac{20}{1000}-\frac{10}{1000}
=\frac{10}{1000}=0.01.
$$

That is 10 fewer events per 1,000 people over five years, or a reduction of **one percentage point**.

“50% lower” and “one percentage point lower” are compatible descriptions. They use different denominators. The first divides by the starting risk; the second expresses the difference on the original probability scale.

## The same headline can describe a smaller absolute change

Now consider Population B. Its hypothetical five-year risk changes from 2 events per 1,000 people to 1 event per 1,000.

The relative reduction is again 50%. The absolute reduction is one event per 1,000, or 0.1 percentage points.

| Five-year quantity | Population A | Population B |
| --- | ---: | ---: |
| Comparison risk, per 1,000 people | 20 | 2 |
| Intervention risk, per 1,000 people | 10 | 1 |
| Relative reduction | 50% | 50% |
| Absolute reduction, per 1,000 people | 10 | 1 |
| Absolute reduction, percentage points | 1.0 | 0.1 |

The table treats each listed risk as an assumed probability, expressed as an expected count in 1,000 people. It is not a table of observed trial counts. There is consequently no empirical sample size from which to estimate confidence intervals.

![Two panels use the same vertical scale. Population A falls from 20 to 10 events per 1,000 people, and Population B falls from 2 to 1. Both represent a 50 percent relative reduction.](/assets/images/figures/science_relative_absolute_risk.png){: width="1465" height="697" loading="lazy"}

*Original hypothetical example. Both panels concern the same five-year time window and use the same scale. Neither population represents a measured clinical group.*

The plot would tell a different visual story if each panel were stretched independently to fill its available height. A shared scale helps preserve the comparison in absolute event frequencies.

## Check whether the headline says risk or odds

A risk is a probability, $p$. The corresponding odds are $p/(1-p)$: the probability of the event divided by the probability of no event. These quantities are related, but they are not interchangeable.

For a risk of 20%, the odds are $0.20/0.80=0.25$, or one to four. For a risk of 10%, they are $0.10/0.90\approx0.111$, or one to nine. The risk ratio is $0.10/0.20=0.5$, while the odds ratio is approximately $0.111/0.25=0.444$.

| Comparison risk | Intervention risk | Risk ratio | Odds ratio |
| --- | --- | ---: | ---: |
| 0.2% | 0.1% | 0.500 | 0.499 |
| 20% | 10% | 0.500 | 0.444 |
| 50% | 25% | 0.500 | 0.333 |

Every row halves the risk. The odds ratio departs further from one as the underlying risks become larger. When both event probabilities are small, odds and risks are numerically close, which explains why an approximation can sometimes be reasonable. The approximation should not silently become a general identity.

If a report gives an odds ratio $\mathrm{OR}$ and a suitable comparison risk $p_0$, the corresponding risk implied by that odds relationship is

$$
p_1=\frac{\mathrm{OR}\,p_0}{1-p_0+\mathrm{OR}\,p_0}.
$$

For an odds ratio of 0.5 and a comparison risk of 50%, this gives an intervention risk of one-third, not 25%. A claim of “50% lower odds” cannot simply be rewritten as “50% lower risk.” Applying the conversion to an adjusted estimate from a real study also requires care about which population and baseline risk it describes.

The first reading task is therefore literal: identify the reported measure before interpreting its percentage. Risk ratios, odds ratios, and measures based on event rates over time are different statistical objects. A headline that drops the name of the measure can remove information needed to reconstruct the result.

## A smaller absolute change is not automatically unimportant

One fewer event per 1,000 people can matter. Its importance depends on the event, the intervention's burdens and harms, and the people affected.

The arithmetic does not decide those values. Avoiding a mild inconvenience and avoiding a severe outcome cannot be compared just by counting them as interchangeable events.

Likewise, identical relative effects are not inherently misleading. Relative measures can be useful when their definitions and context are clear. The problem arises when one is presented as though it supplied the missing absolute scale.

The FDA's guide to communicating risks and benefits discusses the different ways a benefit can be expressed, including relative changes, absolute changes, and numbers needed to treat. The choice of presentation affects what information is visible to the reader. [FDA communication guide](https://www.fda.gov/media/81597/download).

## Keep the clock attached to the number

Our table describes five-year risks. A one-year risk and a lifetime risk would answer different questions even if they used the same percentage.

If one group is described over five years and the other over one year, placing their percentages side by side does not produce the comparison above. We would first need a valid way to account for the different observation periods.

It is also unsafe to assume that a five-year probability is always five times a one-year probability. As an illustration, if a constant annual conditional event risk were 2%, with no competing events, the chance of at least one event over five years would be

$$
1-(1-0.02)^5\approx9.61\%,
$$

rather than exactly 10%. Real risks can change with time, and competing events can complicate the calculation further. The equation is a separate teaching example, not an assumption about either population in the table.

Keeping the time window in the sentence prevents a small typographical omission from becoming a large interpretive error.

## A number needed to treat still needs context

If the assumed difference represented a causal intervention effect, its reciprocal could be expressed as a number needed to treat over the stated period.

For Population A, $1/0.01=100$. For Population B, $1/0.001=1000$.

Those numbers summarise how many people would need the intervention, on average, to avert one additional event over five years under the assumptions. They do not identify which person benefits or promise that exactly one event will be prevented in every group of that size.

They also inherit the original comparison's limitations. If the risk difference is uncertain, its reciprocal is uncertain. If the study did not establish a causal effect, taking a reciprocal does not establish one either.

## Correct arithmetic does not establish the cause

So far, the examples have assigned risks by assumption. A real report must estimate them from evidence.

Suppose people who choose a particular behaviour have fewer events than people who do not. Calculating a risk ratio can describe that association accurately. It does not, by itself, establish what would happen if the second group adopted the behaviour. Other differences between the groups may contribute to their outcomes.

That is a separate question from whether the headline used percentages correctly. A report can be numerically well presented and still make an unsupported causal claim. Conversely, a well-designed study can be described by a headline that leaves out essential scale.

The same separation applies to uncertainty. A ratio based on a small number of events may be imprecise. The point estimate is not the complete evidence; readers also need an appropriate uncertainty interval and information about the study design.

## The same proportions can have very different uncertainty

Our opening table used assumed probabilities. To discuss sampling error, we need a different example containing actual hypothetical *observed counts* and explicit sample sizes.

Imagine two independent studies, each with a comparison group and an intervention group:

| Hypothetical study | Comparison events / people | Intervention events / people | Observed relative reduction |
| --- | ---: | ---: | ---: |
| Small | 2 / 100 | 1 / 100 | 50% |
| Large | 200 / 10,000 | 100 / 10,000 | 50% |

The point estimates are identical: risks of 2% and 1%, an absolute reduction of one percentage point, and a relative reduction of 50%. Their precision is not identical. In the small study, changing the outcome of one person substantially alters the result.

Under an independent binomial sampling model, the estimated standard error of the difference between the two sample risks is

$$
\operatorname{SE}(\widehat p_0-\widehat p_1)
=\sqrt{\frac{\widehat p_0(1-\widehat p_0)}{n_0}
+\frac{\widehat p_1(1-\widehat p_1)}{n_1}}.
$$

For the large study, it is about 0.001718 on the probability scale, or 0.172 percentage points. A simple large-sample 95% interval is therefore approximately $1.00\pm1.96(0.172)$ percentage points: about 0.66 to 1.34 percentage points.

That calculation assumes complete binary outcomes, independent groups, and the stated sampling model. It is not suitable without adjustment for clustering, repeated outcomes, or some forms of incomplete follow-up. The elementary normal approximation is also unreliable for the small study's sparse event counts; quoting the same style of interval there would give a misleading impression of adequate methodology.

The example shows why “per 1,000” in a communication graphic must not be mistaken for “1,000 people actually studied.” A denominator used to express a probability supplies scale. The real sample size and study design supply information about precision.

Statistical precision does not eliminate bias either. A very large observational study can estimate an association precisely while failing to identify the intervention effect of interest. More observations and a better causal comparison solve different problems.

## A change in the population mix can reverse the comparison

Even a clear denominator and time window do not guarantee comparable groups. Consider two hypothetical baseline-risk strata, labelled lower risk and higher risk. Assign the following five-year event probabilities:

| Stratum | Comparison risk | Intervention risk |
| --- | ---: | ---: |
| Lower risk | 2% | 1% |
| Higher risk | 20% | 10% |

The risk is halved within each stratum. Now suppose the comparison group has 900 lower-risk and 100 higher-risk people, while the intervention group has 100 lower-risk and 900 higher-risk people.

The expected event counts are

$$
\text{Comparison: }900(0.02)+100(0.20)=38,
$$

$$
\text{Intervention: }100(0.01)+900(0.10)=91.
$$

The pooled comparison is 91 versus 38 events per 1,000: a crude risk ratio of about 2.39. An aggregate headline could report higher risk in the intervention group even though the assigned risks are lower within each stratum.

There is no arithmetic contradiction. The groups put different weights on the lower-risk and higher-risk populations. The crude ratio combines a within-stratum comparison with a composition difference.

To compare the assigned risks under a common mixture, give each stratum equal weight in both groups. The standardised comparison risk is $(2\%+20\%)/2=11\%$, and the standardised intervention risk is $(1\%+10\%)/2=5.5\%$. The relative reduction returns to 50%, with an absolute difference of 5.5 percentage points for that specified mixture.

This is a constructed illustration of a reversal caused by aggregation. It does not show that every unfavourable result should be adjusted until it becomes favourable. The common weights define a target population, and choosing relevant strata requires substantive knowledge. Real data may contain other differences, measurement errors, or unmeasured confounding that this two-stratum calculation cannot remove.

The reader's useful question is: are the groups comparable for the claim being made? If the original result is adjusted, ask what it was adjusted for and which population the estimate represents. If it is unadjusted, ask whether its denominator combines people with very different starting risks.

## Specify the event before judging its importance

An outcome label can hide as much as a percentage. A study may count the first occurrence of a particular event, the number of recurrent events, or a combined endpoint that includes several different outcomes. Those definitions can produce different numerators from the same people's experiences.

Suppose a hypothetical combined endpoint includes both a mild event and a severe event. A reduction in the combined endpoint does not establish an equal reduction in each component. If the mild event is much more common, changes in it can dominate the overall number. The component results and their uncertainty matter for interpreting what changed.

Likewise, “ten fewer events” and “ten fewer people experiencing any event” need not mean the same thing if events can recur. Our opening examples assumed a binary outcome per person over five years. Applying their probability arithmetic to a recurrent-event count would change the estimand: the quantity the analysis is trying to describe.

This is why a complete sentence should name the population, intervention or exposure, comparison, outcome definition, time window, and numerical effect. An uncertainty interval and an explanation of the design then show how strongly the evidence supports that sentence. The percentage is one part of the result, not its replacement.

## Reconstruct the sentence before sharing it

A useful version of the headline would say:

> In the hypothetical Population A, the five-year risk of the specified event changes from 20 to 10 per 1,000 people: 10 fewer events per 1,000, equivalent to a 50% relative reduction.

For a real result, add who was studied, what the event was, the uncertainty, and whether the design supports a causal interpretation.

Those details are not decoration. They determine whether two headlines describe comparable quantities and whether the result answers the reader's question.

## Reproduce the comparison

```python
for name, comparison, intervention in [("A", 20, 10), ("B", 2, 1)]:
    difference = comparison - intervention
    relative_reduction = difference / comparison
    percentage_points = 100 * difference / 1000
    print(f"{name}: {relative_reduction:.0%} relative reduction; "
          f"{difference}/1000 absolute reduction; "
          f"{percentage_points:.1f} percentage points")
```

The odds, sampling-error, and mixture calculations are also reproducible:

```python
from math import sqrt

for p0, p1 in [(0.002, 0.001), (0.20, 0.10), (0.50, 0.25)]:
    risk_ratio = p1 / p0
    odds_ratio = (p1 / (1 - p1)) / (p0 / (1 - p0))
    print(f"risk ratio={risk_ratio:.3f}; odds ratio={odds_ratio:.3f}")
se = sqrt(0.02 * 0.98 / 10000 + 0.01 * 0.99 / 10000)
print("Large-study interval, percentage points:",
      100 * (0.01 - 1.96 * se), 100 * (0.01 + 1.96 * se))
comparison = 900 * 0.02 + 100 * 0.20
intervention = 100 * 0.01 + 900 * 0.10
print("Crude expected counts:", comparison, intervention)
print("Equal-mixture risks:", (0.02 + 0.20) / 2, (0.01 + 0.10) / 2)
```

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces the original comparison table and shared-scale plot. For the related problem of concentration without quantity, see [Natural Origin Does Not Establish Safety](/science-communication/natural_origin_does_not_establish_safety/).

*Archive note: dated 18 June 2026 for this collection; written and source-checked on 18 September 2026.*
