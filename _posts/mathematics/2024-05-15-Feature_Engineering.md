---
permalink: '/mathematics/Feature_Engineering/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-15'
header:
  image: /assets/images/headers/photo-mathematics-voronoi.jpg
  og_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-voronoi.jpg
  twitter_image: /assets/images/headers/photo-mathematics-voronoi.jpg
redirect_from:
- '/mathematics/statistics/data science/machine learning/Feature_Engineering/'
seo_description: "Feature engineering is the construction of model inputs from available information, with leakage control, time alignment, and reproducibility more important than automated feature count."
seo_title: "Feature Engineering Without Leakage"
seo_type: article
subtitle: "Representation, information availability, and validation"
tags:
- Feature Engineering
- Machine Learning
- Data Science
title: "Feature Engineering Without Leakage"
---

Feature engineering transforms available information into model inputs.

The main goal is not to create as many variables as possible.

It is to represent useful structure without leaking information from the future, target, or validation data.

## Start from information availability

For prediction at time $t$, every feature must be computable using information available no later than $t$.

A rolling mean defined with future observations is invalid for prospective prediction.

A customer feature built from transactions after churn is leakage.

Point-in-time correctness is therefore part of feature design.

## Transformations encode assumptions

Common transformations include:

- log transforms
- splines
- interactions
- ratios
- lags
- rolling statistics
- counts
- embeddings
- categorical encodings

Each changes the model class.

A ratio such as

$$
X_1/X_2
$$

is not automatically meaningful just because it improves validation score.

It should correspond to a plausible scale or mechanism.

## Scaling

Standardization is useful for algorithms sensitive to feature scale.

It is not required for every model.

Tree-based methods, for example, are generally invariant to monotone rescaling of individual features.

Scaling should be fitted on training data only.

## Target encoding

Target encoding can be powerful and dangerous.

If category means are computed using the same observation being encoded, the target leaks into the feature.

Out-of-fold or leave-one-out procedures are required.

Smoothing is often needed for rare categories.

## Time-based features

Calendar variables can encode seasonality:

- hour
- weekday
- month
- holidays

Cyclic features may use

$$
\sin(2\pi t/T),
\qquad
\cos(2\pi t/T).
$$

This avoids artificial discontinuities such as December being numerically far from January.

## Aggregations

Entity-level models often need historical aggregates.

Examples include:

- purchase count
- mean transaction value
- recency
- failure count
- rolling variance

The aggregation window and cutoff timestamp must be explicit.

## Feature selection

Feature selection is different from feature engineering.

Selection asks which constructed variables to retain.

Methods include regularization, stability selection, mutual information, domain constraints, and model-specific screening.

Selection must occur inside cross-validation.

Selecting features on the full dataset before validation leaks test information.

## Dimensionality reduction is not ordinary feature selection

PCA creates new linear combinations of variables.

t-SNE is primarily a visualization method and is not a sensible generic feature-selection stage.

Using t-SNE coordinates as production model features is usually difficult to justify.

## Automated feature engineering

Tools can generate transformations and relational aggregates automatically.

That does not remove the need for semantic validation.

Automated systems can easily create:

- duplicated information
- target leakage
- post-outcome variables
- unstable high-cardinality features
- expensive transformations

Automation increases search space and therefore increases the need for disciplined validation.

## Feature stores

A feature store can help standardize definitions, version transformations, and maintain training-serving consistency.

It cannot fix a logically invalid feature.

Feature governance should record:

- source
- transformation
- timestamp semantics
- owner
- version
- expected range
- freshness
- leakage constraints

## Interactions and nonlinear terms

Linear models can represent rich nonlinear structure when the basis is engineered explicitly.

For example,

$$
Y
=
\beta_0
+
\beta_1X
+
\beta_2X^2
+
\beta_3Z
+
\beta_4XZ
+
\varepsilon.
$$

This can be preferable to replacing an interpretable model with a black box when the relevant structure is known.

## Validation

Every learned feature transformation belongs inside the resampling loop.

That includes:

- imputation
- scaling
- PCA
- target encoding
- vocabulary construction
- feature selection
- learned embeddings

A pipeline object is often the safest implementation pattern.

## Conclusion

Feature engineering is best understood as representation design under information constraints.

The most important question is not

> Can this variable improve validation score?

It is

> Could this variable exist at prediction time, and does its construction remain valid outside this sample?

## References

- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*.
- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*.
