---
permalink: '/statistics/gini_coefficiente/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2024-05-19'
excerpt: "In binary credit scoring, the model Gini is a linear transformation of ROC AUC. It measures ranking discrimination, not calibration, default rate, or economic value."
header:
  image: /assets/images/headers/photo-statistics-student-t.jpg
  og_image: /assets/images/headers/photo-statistics-student-t.jpg
  overlay_image: /assets/images/headers/photo-statistics-student-t.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-student-t.jpg
  twitter_image: /assets/images/headers/photo-statistics-student-t.jpg
keywords:
- Gini coefficient credit scoring
- ROC AUC
- Credit risk discrimination
- Default rate
- Model calibration
seo_description: "Credit-score Gini explained through its relationship with ROC AUC, ranking discrimination, default prevalence, calibration, and validation."
seo_title: "Credit-Score Gini: Ranking Is Not Calibration"
seo_type: article
tags:
- Finance
- Model Evaluation
- Risk Management
title: "Credit-Score Gini: Ranking Is Not Calibration"
---

In binary credit scoring, the commonly reported model Gini is

$$
G
=
2\,AUC-1.
$$

It measures ranking discrimination.

It does not measure calibration, default rate, profitability, fairness, or stability.

## ROC AUC interpretation

For a score oriented so higher values indicate higher risk,

$$
AUC
=
P(S_D>S_N)
+
\frac12P(S_D=S_N),
$$

where $S_D$ is a score for a randomly selected defaulter and $S_N$ for a randomly selected non-defaulter.

Thus Gini is a rescaling of pairwise ranking probability.

AUC $=0.5$ gives

$$
G=0.
$$

Perfect ranking gives

$$
G=1.
$$

A reversed ranking can produce negative Gini.

## Default rate is a separate quantity

Portfolio default rate is

$$
DR
=
\frac{\text{defaults}}
{\text{exposures}}
$$

under a specified default definition and observation horizon.

A model can have the same Gini in two portfolios with very different default rates.

Discrimination and prevalence are distinct.

## Calibration

A well-calibrated probability model satisfies approximately

$$
P(Y=1\mid \widehat p\approx p)
\approx p.
$$

High Gini does not imply good calibration.

A model can rank accounts well while systematically overestimating or underestimating PD.

Credit-risk validation therefore needs both discrimination and calibration.

## Sample dependence

Estimated Gini depends on the evaluation population.

Changes in product mix, underwriting policy, macroeconomic conditions, and observation horizon can change the estimate.

Comparing Gini across datasets requires comparable target definitions and sampling frames.

## Confidence intervals

A reported Gini is an estimate.

Bootstrap or asymptotic methods can quantify uncertainty.

Small validation samples can make differences between two scorecards statistically uninformative.

## Ties

Credit scores are often discrete.

Ties should receive half credit in the pairwise AUC interpretation.

Implementation details matter when many accounts share the same score.

## Gini for continuous targets is different

Some competitions and regression tasks use a normalized Gini based on Lorenz-style ordering of continuous outcomes.

That is not the same object as binary credit-score Gini.

The shared name causes confusion.

The article should state which definition is being used.

## No universal 'good Gini' threshold

Rules such as “40-60% is typical” depend on portfolio, product, horizon, data quality, and score purpose.

There is no universal cutoff at which a model becomes acceptable.

A weaker Gini can be operationally preferable if calibration, stability, cost, and governance are better.

## Economic value

Ranking quality matters because decisions such as pricing, approval, and limit assignment depend on ordering risk.

But economic value also depends on exposure, loss given default, margin, capital, and decision thresholds.

Gini alone cannot determine the optimal lending policy.

## A simple Python implementation

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


def binary_gini(
    y_true: NDArray[np.int_],
    score: NDArray[np.float64],
) -> float:
    """Return binary-model Gini = 2*AUC - 1."""
    if y_true.ndim != 1 or score.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if y_true.size != score.size:
        raise ValueError("inputs must have equal length")

    auc = roc_auc_score(y_true, score)
    return float(2.0 * auc - 1.0)
~~~

The score orientation must be documented.

## Conclusion

Binary credit-score Gini is a discrimination metric.

It answers:

> How well does the score rank defaulters relative to non-defaulters?

It does not answer:

- Are predicted probabilities calibrated?
- Is the portfolio default rate acceptable?
- Is the model stable?
- Is the lending policy profitable?

Those require separate analyses.

## References

- Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve.
- Thomas, L. C., Crook, J. N., & Edelman, D. B. (2017). *Credit Scoring and Its Applications*.
