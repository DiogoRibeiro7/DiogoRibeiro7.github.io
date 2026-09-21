---
permalink: '/mathematics/new_illiteracy_that-s_crippling_our_decisionmaking/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2023-09-26'
excerpt: Quantitative literacy is less about performing arithmetic quickly than about reasoning with ratios, uncertainty, variation, denominators, and evidence.
header:
  image: /assets/images/headers/photo-mathematics-mobius-strip.jpg
  og_image: /assets/images/headers/photo-mathematics-mobius-strip.jpg
  overlay_image: /assets/images/headers/photo-mathematics-mobius-strip.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-mobius-strip.jpg
  twitter_image: /assets/images/headers/photo-mathematics-mobius-strip.jpg
keywords:
- Numeracy
- Statistical literacy
- Quantitative reasoning
- Risk communication
- Base rates
- Percentages
- Uncertainty
- Data literacy
seo_description: A rigorous introduction to quantitative literacy: ratios, denominators, base rates, uncertainty, graphs, and the reasoning needed to interpret numerical claims.
seo_title: 'Quantitative Literacy: Reading Numbers Without Being Misled'
seo_type: article
tags:
- Mathematics
- Statistical Literacy
title: Quantitative Literacy: Reading Numbers Without Being Misled
---

![Inumeracy - Quantitative Literacy](/assets/images/inumeracy.jpg){: width="1400" height="1867" loading="lazy"}

Quantitative literacy is sometimes framed as the ability to calculate quickly or remember school mathematics. That definition is too narrow. In everyday decisions, the difficult part is rarely computing a percentage by hand. It is understanding what the denominator is, whether two quantities are comparable, how uncertainty changes a conclusion, whether a graph uses an appropriate scale, or whether an observed association supports the claim being made.

For that reason, innumeracy should not be treated as a cultural insult or as a claim that most people are incapable of mathematics. It is more useful to think of quantitative literacy as a collection of reasoning skills that can be taught, practiced, and audited.

## Ratios need denominators

A statement such as "risk increased by 50%" is incomplete without the baseline risk. If an event probability rises from 2% to 3%, the relative increase is

$$
\frac{0.03-0.02}{0.02}=0.5,
$$

or 50%, while the absolute increase is one percentage point.

Neither representation is inherently wrong. They answer different questions. Relative effects describe proportional change. Absolute effects describe the change in expected frequency. Good quantitative communication often requires both.

Denominators also matter for rates. Ten events in a population of 100 and ten events in a population of 10,000 are not comparable counts. The relevant quantities are rates such as

$$
r=\frac{\text{events}}{\text{population at risk}}.
$$

Even then, rates may need adjustment for exposure time, age, case mix, or other structural differences before comparisons are meaningful.

## Percentages and percentage points are not the same

If a rate moves from 20% to 25%, it rises by five percentage points but by 25% relative to its starting value:

$$
\frac{25-20}{20}=0.25.
$$

Confusing these two quantities can make small changes look dramatic or large changes look modest. Whenever a percentage change is reported, the original level should be recoverable from the presentation.

## Base rates change interpretation

Suppose a screening procedure has 95% sensitivity and 95% specificity. Those numbers do not tell us the probability that a person with a positive result actually has the condition.

If prevalence is 1%, then among 10,000 people we expect roughly 100 true cases. About 95 of those will test positive. Among the 9,900 people without the condition, about 495 will also test positive. The positive predictive value is therefore approximately

$$
\frac{95}{95+495}\approx 0.161.
$$

A test can have high sensitivity and specificity while most positive results are false positives in a low-prevalence population. This is a base-rate effect, not a paradox.

## Expected values are not guarantees

An expected value is an average over a probability distribution. If a random payoff X has outcomes x_i with probabilities p_i, then

$$
E[X]=\sum_i p_i x_i.
$$

The expected value need not be an outcome that will ever occur. It also says nothing by itself about variability, tail risk, or whether losses are acceptable.

Two decisions can have the same expectation and very different risk profiles. Quantitative reasoning therefore requires looking beyond averages.

## Variation is information

A mean without a measure of spread can be misleading. Consider two processes with the same average output but different standard deviations. If operational failure occurs beyond a threshold, the more variable process may be much riskier even though the means are identical.

Sampling variation matters too. An estimate based on finite data is uncertain. For many estimators, uncertainty decreases roughly with the square root of sample size:

$$
\operatorname{SE}\propto \frac{1}{\sqrt n}.
$$

Doubling the sample does not halve the standard error. Roughly four times as many independent observations are required for that.

## Correlation is not an intervention

Two variables can move together because one affects the other, because the causal direction is reversed, because both respond to a third variable, because of selection, or because of chance.

A regression coefficient or correlation is therefore not automatically a causal effect. Causal interpretation requires a design or assumptions that justify the counterfactual comparison.

This distinction is part of quantitative literacy because many numerical claims become misleading precisely when descriptive evidence is narrated as causal evidence.

## Graphs can be numerically correct and still misleading

Visualizations influence interpretation through scale, aggregation, truncation, and choice of baseline. A bar chart with a truncated vertical axis can exaggerate small differences. Aggregating time series over different intervals can hide volatility. A map based on counts can mostly reflect population size rather than risk.

Useful questions include:

- What is the denominator?
- Is the axis linear or logarithmic?
- Does the axis start at a meaningful baseline?
- Are values totals, rates, percentages, or changes?
- Has aggregation hidden important variation?
- Are uncertainty intervals shown when they matter?

## Simpson's paradox and aggregation

An association observed in aggregated data can reverse after conditioning on a relevant variable. This is commonly called Simpson's paradox.

The lesson is not that aggregated statistics are useless. It is that the level of aggregation is part of the model. A conclusion about individuals, regions, hospitals, or time periods must be supported by data and assumptions at the corresponding level.

## Numerical precision is not evidential strength

A model can return 0.873421 and still be poorly identified. Software can produce a p-value with many decimal places for a scientifically weak comparison. Decimal precision should not be confused with uncertainty reduction.

A good report rounds numbers to the precision justified by measurement and sampling error, then states the uncertainty explicitly.

## Quantitative literacy is partly about asking better questions

Many numerical mistakes can be caught with a short checklist:

1. What exactly is being counted or estimated?
2. What is the denominator?
3. What population and time period does the number describe?
4. Is the quantity absolute or relative?
5. What uncertainty surrounds the estimate?
6. Is the comparison like-for-like?
7. Is the claim descriptive, predictive, or causal?
8. What assumptions connect the number to the conclusion?

These questions are more valuable than memorizing a catalogue of formulas because they transfer across healthcare, economics, engineering, business, and public statistics.

## Conclusion

Numeracy is not a competition between "numbers people" and everyone else. It is the ability to reason carefully when quantities are used as evidence. The essential skills are proportional reasoning, attention to denominators, understanding of uncertainty, distinction between association and causation, and the ability to interrogate how a number was produced.

Modern life contains more numerical claims than any individual can verify from first principles. The practical goal is therefore not omniscience. It is to recognize the structure of a quantitative argument well enough to ask where it can fail.

## References

- Gigerenzer, G. (2002). *Calculated Risks*. Simon & Schuster.
- Huff, D. (1954). *How to Lie with Statistics*. W. W. Norton.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Paulos, J. A. (1988). *Innumeracy: Mathematical Illiteracy and Its Consequences*. Hill and Wang.
- Spiegelhalter, D. (2019). *The Art of Statistics*. Basic Books.
