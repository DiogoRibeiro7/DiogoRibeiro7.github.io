---
permalink: '/mathematics/AI_fairness/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-15'
header:
  image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  og_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  overlay_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-polyhedra.jpg
  twitter_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
redirect_from:
- '/mathematics/statistics/data science/machine learning/ethics research/AI_fairness/'
seo_description: "AI fairness metrics encode different normative goals and can conflict. Fairness analysis therefore requires explicit decision context, measurement assumptions, and governance."
seo_title: "AI Fairness: Metrics, Trade-offs, and Limits"
seo_type: article
subtitle: "Why no single fairness metric settles the problem"
tags:
- Ethics
- Machine Learning
title: "AI Fairness: Metrics, Trade-offs, and Limits"
---

Fairness in machine learning is not one mathematical property.

Different metrics formalize different goals, and those goals can conflict.

The correct starting point is therefore the decision system, not a checklist of fairness scores.

## Demographic parity

Demographic parity requires

$$
P(\widehat Y=1\mid A=a)
$$

to be equal across protected groups $A$.

This can be useful when equal allocation rates are the policy objective.

It does not require equal error rates or equal predictive meaning across groups.

## Equal opportunity

Equal opportunity requires equal true-positive rates:

$$
P(\widehat Y=1\mid Y=1,A=a).
$$

This focuses on access among people who truly satisfy the positive condition.

It does not constrain false-positive rates.

## Equalized odds

Equalized odds requires both true-positive and false-positive rates to match across groups:

$$
\widehat Y
\perp
A
\mid
Y.
$$

This is a stronger criterion.

## Calibration

Group calibration means that among people assigned risk $r$, outcome frequency is approximately $r$ within each group:

$$
P(Y=1\mid \widehat p=r,A=a)
\approx r.
$$

Calibration concerns the meaning of predicted probabilities.

It is different from equalized error rates.

## Incompatibility

When base rates differ across groups, calibration and equalized odds generally cannot both hold for a non-perfect predictor.

That is not a software limitation.

It is a mathematical conflict between fairness definitions.

Therefore a toolkit cannot decide which fairness objective is ethically appropriate.

## Labels may already encode unfairness

A fairness audit often starts too late.

If the target label reflects unequal access, policing, diagnosis, investigation, or historical decisions, then optimizing prediction of that label can reproduce those mechanisms.

The pipeline is

$$
\text{population}
\rightarrow
\text{measurement}
\rightarrow
\text{label}
\rightarrow
\text{model}
\rightarrow
\text{decision}.
$$

Every stage can create disparity.

## Fairness through unawareness is weak

Removing a protected attribute does not remove its information.

Postcode, school, occupation, language, and many other variables can proxy group membership.

Protected attributes may also be necessary to audit disparities.

## Threshold adjustment

Group-specific thresholds can equalize selected error rates.

That is a policy choice with legal and ethical implications, not merely a technical optimization trick.

The consequences should be evaluated in the actual decision context.

## Individual fairness

The principle “similar individuals should be treated similarly” depends on a similarity metric.

Defining that metric is itself normative.

If the distance function is unjustified, the fairness guarantee is empty.

## Counterfactual fairness

Counterfactual fairness asks whether a prediction would change under a hypothetical intervention on a protected attribute in a causal model.

This requires a causal graph and structural assumptions.

It is not obtained by simply flipping one column in the dataset while holding all correlated variables fixed.

## Long-term effects

A decision rule can alter future data.

Lending, hiring, recommendations, and policing change behavior and opportunity.

Static group metrics can therefore miss feedback loops.

Fairness analysis may need to model downstream dynamics.

## Tools are implementation aids

Libraries such as Fairlearn and AI Fairness 360 can compute metrics and implement mitigation algorithms.

They do not determine the correct fairness definition.

The scientific and governance work comes first.

## A defensible workflow

1. Define the decision and affected population.
2. Audit measurement and labels.
3. Identify relevant groups and harms.
4. Choose fairness criteria tied to those harms.
5. Quantify uncertainty in group metrics.
6. Test calibration and performance by group.
7. Evaluate threshold and intervention consequences.
8. Monitor after deployment.
9. Provide contestability and human accountability.

## Conclusion

Fairness metrics are useful because they make value choices explicit.

They are dangerous when treated as interchangeable numerical badges.

The central question is not

> Is this model fair?

but

> Fair according to which criterion, for which people, under which decision process, and with what consequences?

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent Trade-Offs in the Fair Determination of Risk Scores.
- Barocas, S., Hardt, M., & Narayanan, A. *Fairness and Machine Learning*.
