---
permalink: '/time-series/forecast_value_added_analysis/'
title: 'Forecast Value Added: Is Your Process Helping?'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Model Evaluation
- Business Intelligence
author_profile: false
seo_title: 'Forecast Value Added Analysis'
seo_description: 'Every step in a forecasting process is assumed to add accuracy. FVA measures whether it does, and often the answer is no.'
excerpt: >-
  Forecasting processes accumulate steps: a statistical model, a planner
  override, a consensus meeting. Each is assumed to improve the number. FVA is
  how you find out.
summary: >-
  What forecast value added measures, how to construct the comparison against
  a naive baseline and against each preceding process step, why judgmental
  overrides frequently destroy accuracy, and how to run the analysis without
  it becoming an audit of individuals.
keywords:
  - forecast value added
  - FVA
  - judgmental adjustment
  - forecasting process
  - demand planning
classes: wide
date: '2026-07-11'
header:
  image: /assets/images/data_science_7.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_7.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_7.jpg
  twitter_image: /assets/images/data_science_7.jpg
---
A demand planning process typically has several stages. A statistical model produces a baseline. A planner reviews and adjusts it. Sales adds market intelligence. A consensus meeting settles the final number. Each stage exists because someone believes it improves accuracy.

Forecast Value Added asks whether it does, by measuring accuracy after each stage against accuracy before it. The results are frequently uncomfortable.

## The Comparison

FVA is a difference in accuracy attributable to a process step:

$$
\text{FVA} = \text{Accuracy}_{\text{after step}} - \text{Accuracy}_{\text{before step}} .
$$

Two comparisons matter, and both are needed.

**Against the naive forecast.** Does the statistical model beat "same as last period" — or the seasonal naive where seasonality exists? If not, the entire modelling apparatus is subtracting value, and the honest response is to ship the naive forecast.

**Against the preceding step.** Does the planner's override improve on the statistical baseline? Does the consensus number improve on the planner's? Each stage is judged against the thing it modified, not against the naive.

The output is conventionally a stairstep table:

| Process step | MAPE | FVA vs naive | FVA vs previous |
|---|---|---|---|
| Naive forecast | 28.4% | — | — |
| Statistical model | 22.1% | +6.3 | +6.3 |
| Planner override | 23.6% | +4.8 | −1.5 |
| Consensus forecast | 22.9% | +5.5 | +0.7 |

This shape — the model helps, the override hurts, the meeting partially repairs the damage — is common enough in published FVA studies to be the expected result rather than a surprising one.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 500
actual = 100 + rng.normal(0, 12, n)

naive = np.roll(actual, 1); naive[0] = actual[0]
statistical = actual + rng.normal(0, 8, n)                  # genuinely useful
# planner override: small real signal, larger added noise
override = statistical + rng.normal(1.5, 7, n)
consensus = 0.6 * statistical + 0.4 * override

def mape(f):
    return np.mean(np.abs((actual - f) / actual)) * 100

steps = [("naive", naive), ("statistical", statistical),
         ("planner override", override), ("consensus", consensus)]

base = mape(naive)
prev = base
print(f"{'step':20}{'MAPE':>8}{'vs naive':>10}{'vs prev':>9}")
for name, f in steps:
    m = mape(f)
    print(f"{name:20}{m:7.2f}%{base - m:+10.2f}{prev - m:+9.2f}")
    prev = m
```

The negative column is the one that changes behaviour. A step with consistently negative FVA is consuming analyst time to make the forecast worse, and that is a finding a process owner can act on.

## Why Judgmental Overrides Often Hurt

The empirical literature is fairly consistent: manual adjustments to statistical forecasts have mixed value, and a large share destroy accuracy.

Several mechanisms explain it. Planners adjust too often, treating every forecast as needing intervention, when most need none. Small adjustments in particular tend to be noise — the evidence suggests large adjustments, made for a specific known reason, are more likely to help than routine tinkering.

Adjustments are also asymmetric. Upward revisions are more common and less accurate than downward ones, which is consistent with optimism and with incentives that penalise stockouts more visibly than excess inventory.

And planners often adjust for information the model already contains. Reacting to a seasonal peak that the seasonal component has already captured double-counts it.

The useful distinction is between information the model *cannot* have — a competitor's announced exit, a confirmed one-off order, a plant closure — and pattern-reading that the model does better. Overrides restricted to the first category tend to add value; overrides applied routinely do not.

## Running the Analysis Without Blame

FVA is a process diagnostic, and it fails the moment it becomes a performance review. If planners believe the analysis will be used against them, they stop recording their reasoning, and the data needed to improve the process disappears.

Some practices that keep it useful. Report at the level of the process step, aggregated across planners, rather than naming individuals. Require a recorded reason for each override, which both improves the adjustments and makes the analysis interpretable. Frame a negative FVA as "this step is not where our effort pays off" rather than as an error. And act on the finding by removing or narrowing the step, which is the point — an FVA report that changes nothing is wasted work.

## Practical Considerations

**Use a scaled or absolute metric, not MAPE, where series differ in scale.** MAPE breaks near zero and penalises over- and under-forecasting asymmetrically, which biases the comparison. MASE is the safer default for cross-series aggregation.

**Segment the results.** A process step can add value on high-volume, stable items and destroy it on intermittent ones. A single average conceals that, and the actionable conclusion is usually "keep overrides for the top 200 SKUs, automate the rest".

**Use enough history.** Forecast accuracy is noisy, and a difference measured over a handful of periods will not replicate. Several months of forecast-actual pairs at minimum, and treat small differences as inconclusive rather than as findings.

**Compare like with like.** All steps must be measured on the same items, the same periods, and the same forecast horizon — an override made at a shorter lead time has an unfair advantage over the statistical forecast it modified.

The underlying question FVA poses is worth asking of any analytical process, not only forecasting: which of these steps would we notice if we stopped doing it?

## References

- Gilliland, M. (2010). *The Business Forecasting Deal: Exposing Myths, Eliminating Bad Practices, Providing Practical Solutions*. Wiley.
- Fildes, R., Goodwin, P., Lawrence, M., & Nikolopoulos, K. (2009). Effective forecasting and judgmental adjustments. *International Journal of Forecasting*, 25(1), 3-23.
- Syntetos, A. A., Nikolopoulos, K., Boylan, J. E., Fildes, R., & Goodwin, P. (2009). The effects of integrating management judgement into intermittent demand forecasts. *International Journal of Production Economics*, 118(1), 72-81.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
