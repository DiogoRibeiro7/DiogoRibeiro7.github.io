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
  uncertainty, and causal evidence when interpreting a health headline.
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
  and primary explanations from the National Cancer Institute and FDA.
methodology: >-
  Express both comparison groups per 1,000 people over five years, calculate
  relative and absolute changes, and keep the numerical comparison separate
  from uncertainty and claims about intervention effects.
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
Evidence object: Two hypothetical populations, natural-frequency tables, exact arithmetic, and an original figure.
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

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces the table and shared-scale plot. For the related problem of concentration without quantity, see [Natural Origin Does Not Establish Safety](/science-communication/natural_origin_does_not_establish_safety/).

*Archive note: dated 18 June 2026 for this collection; written and source-checked on 18 September 2026.*
