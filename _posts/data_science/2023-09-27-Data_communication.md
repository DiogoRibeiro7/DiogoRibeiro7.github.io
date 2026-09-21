---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-09-27'
excerpt: Good data communication preserves the structure of the evidence: the estimand, denominator, uncertainty, assumptions, and distinction between description, prediction, and causation.
header:
  image: /assets/images/headers/photo-data-science-neural-network.jpg
  og_image: /assets/images/headers/photo-data-science-neural-network.jpg
  overlay_image: /assets/images/headers/photo-data-science-neural-network.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-neural-network.jpg
  twitter_image: /assets/images/headers/photo-data-science-neural-network.jpg
keywords:
- Data communication
- Statistical communication
- Data visualization
- Uncertainty
- Scientific communication
- Decision support
permalink: '/data-science/Data_communication/'
redirect_from:
- '/data science/Data_communication/'
seo_description: How to communicate data without losing statistical meaning, including context, uncertainty, causal language, visualization, and decision relevance.
seo_title: 'Data Communication: Preserve the Evidence'
seo_type: article
tags:
- Research Methodology
- Data Analysis
title: Data Communication: Preserve the Evidence
---

![Communication - Data Communication](/assets/images/communication.jpg){: width="2048" height="1366" loading="lazy"}

Data communication is often described as storytelling. The metaphor is useful up to a point: audiences need structure, context, and a reason to care. But evidence is not fiction, and a statistical result should not be reshaped merely to produce a cleaner narrative. The primary obligation is to preserve what the analysis can and cannot support.

A strong data presentation therefore begins before the slide deck. It begins with a clear estimand, a documented data-generating process, and an analysis whose limitations are understood.

## Start with the question

Before deciding how to visualize or narrate a result, state the question precisely. Is the analysis describing what happened, predicting what will happen, estimating the effect of an intervention, or choosing an action under uncertainty?

Those targets require different language. A descriptive increase is not automatically a forecast. A predictive association is not automatically causal. A causal estimate is not automatically a decision recommendation.

Communication becomes unreliable when these layers are blended.

## Report the estimand, not just the metric

Consider a statement such as "conversion improved by 10%." It is unclear whether that means ten percentage points, a 10% relative increase, an adjusted model coefficient, or a posterior mean.

A useful result states the estimand directly. For example:

> The conversion rate increased from 20% to 22%, an absolute increase of 2 percentage points and a relative increase of 10%.

This gives the audience enough information to reconstruct the claim.

## Uncertainty belongs next to the estimate

Point estimates create false precision when displayed without uncertainty. If an estimated effect is \(\hat\theta\), a report should usually include an interval or distribution that reflects the relevant uncertainty.

For a confidence interval

$$
C(X)=[L(X),U(X)],
$$

the interval procedure has a repeated-sampling interpretation. It should not be translated into a probability statement about a fixed parameter unless a Bayesian model is being used.

Uncertainty can also come from measurement error, model specification, missing data, external validity, or forecast distribution shift. A narrow standard error does not eliminate those sources.

## Context is not the same as a causal explanation

Suppose sales rose during a promotion. Showing both series together can provide useful context, but temporal coincidence alone does not establish that the promotion caused the increase.

Words such as *caused*, *driven by*, and *due to* should be reserved for analyses with a causal design or explicit identification assumptions. Otherwise, use language such as *coincided with*, *was associated with*, or *was higher during*.

This is one of the most important disciplines in data communication because causal language can change a descriptive chart into an unsupported recommendation.

## Visualization should expose structure

A good chart reduces cognitive load without hiding relevant variation. The graphical form should follow the question.

- use distributions, not only averages, when heterogeneity matters
- show raw points when sample size permits
- use rates rather than counts when exposure differs
- use logarithmic scales when multiplicative structure is important
- show uncertainty intervals when estimates are noisy
- avoid dual axes unless the mapping is essential and clearly explained
- avoid three-dimensional decoration that does not encode data

Axis choices should be defensible. A truncated axis can be appropriate for a line chart when small changes are the object of study, but a truncated bar chart can badly distort visual comparisons because bar length encodes magnitude from the baseline.

## Tables and charts answer different questions

Charts are good for patterns. Tables are good for exact lookup. A report often needs both.

If the audience must compare trends across time, a chart is usually better. If the audience must retrieve a regulatory threshold or an exact estimate with confidence limits, a table may be better.

Do not force every result into one visual grammar.

## The denominator should be visible

Counts can mislead when population size or exposure varies. Ten failures in 100 machine-hours and ten failures in 10,000 machine-hours are different phenomena.

Whenever a percentage, rate, or ratio is shown, the denominator should either be displayed or be obvious from the context. This is especially important for subgroup comparisons because apparently large differences can be driven by small sample sizes.

## Distinguish signal from exploratory search

Exploratory analysis is valuable, but post-hoc discoveries should not be presented as if they were prespecified confirmatory tests.

If dozens of metrics, segments, and time windows were inspected before one striking pattern was selected, the communication should say so. Selective presentation hides the search process and makes chance patterns look inevitable.

Reproducible notebooks, preregistered analyses where appropriate, and a record of the tested hypotheses make the provenance of a result easier to audit.

## Communicate model performance in deployment terms

Machine-learning results are often reported through one aggregate metric. That is rarely enough.

For classifiers, communicate discrimination, calibration, threshold behavior, prevalence, and relevant error costs. For forecasts, show performance by horizon and include predictive uncertainty. For ranking systems, report performance where the decision is actually made, not only across the full sample.

Validation should mimic deployment. A random train-test split is not persuasive evidence for future-time performance if the system will be used prospectively.

## Recommendations require an explicit decision rule

Analysis and action should be connected through a decision framework. If the recommendation is to intervene when a predicted risk exceeds a threshold, explain why that threshold is appropriate.

Under a simple two-action loss model, the decision can be written as

$$
a^*(x)=\arg\min_a E[L(a,Y)\mid X=x].
$$

This makes clear that a probability estimate is not itself a decision. Costs, capacity, constraints, and objectives determine the action.

## A useful communication structure

A concise technical report can follow this order:

1. **Question** — what quantity or decision is being studied?
2. **Data** — what population, period, and measurement process generated the observations?
3. **Method** — what assumptions connect the data to the estimand?
4. **Result** — what is the estimate and its uncertainty?
5. **Limitations** — what could invalidate or limit the conclusion?
6. **Decision implication** — what action follows, under which loss or constraint assumptions?

This structure is more robust than forcing every analysis into a dramatic beginning-middle-end narrative.

## Conclusion

Good data communication does not make uncertainty disappear. It makes uncertainty intelligible. It does not turn every association into a story about causes. It separates observation from interpretation and interpretation from action.

The aim is not to make numbers sound persuasive. It is to make the evidential chain visible enough that another person can understand, question, and use it correctly.

## References

- Cleveland, W. S. (1985). *The Elements of Graphing Data*. Wadsworth.
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.
- Gelman, A., Hill, J., & Vehtari, A. (2020). *Regression and Other Stories*. Cambridge University Press.
- Spiegelhalter, D. (2019). *The Art of Statistics*. Basic Books.
