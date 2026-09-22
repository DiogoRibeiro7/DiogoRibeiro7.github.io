---
permalink: '/mathematics/Bhattacharyya_Distance/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-19'
excerpt: "Bhattacharyya distance measures overlap between probability distributions. It should be treated separately from predictive loss functions such as squared error and cross-entropy."
header:
  image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  og_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  overlay_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
  twitter_image: /assets/images/headers/photo-mathematics-geometry-symmetry.jpg
keywords:
- Bhattacharyya coefficient
- Bhattacharyya distance
- Hellinger distance
- Distribution overlap
redirect_from:
- '/mathematics/statistics/data science/machine learning/Bhattacharyya_Distance/'
seo_description: "Bhattacharyya coefficient and distance explained, including Gaussian closed forms, relation to Hellinger distance, and distinction from machine-learning loss functions."
seo_title: "Bhattacharyya Distance: Measuring Distribution Overlap"
seo_type: article
tags:
- Probability
- Information Theory
title: "Bhattacharyya Distance: Measuring Distribution Overlap"
---

The Bhattacharyya coefficient measures overlap between two probability distributions.

For discrete distributions,

$$
BC(P,Q)
=
\sum_x
\sqrt{p(x)q(x)}.
$$

For densities,

$$
BC(P,Q)
=
\int
\sqrt{p(x)q(x)}
\,dx.
$$

The Bhattacharyya distance is

$$
D_B(P,Q)
=
-\log BC(P,Q).
$$

## Basic properties

The coefficient satisfies

$$
0\le BC(P,Q)\le1.
$$

If $P=Q$, then

$$
BC(P,Q)=1
$$

and

$$
D_B(P,Q)=0.
$$

If the distributions have disjoint support, then

$$
BC(P,Q)=0
$$

and the distance is infinite.

The distance is symmetric:

$$
D_B(P,Q)=D_B(Q,P).
$$

It is not generally a metric because the triangle inequality need not hold.

## Relation to Hellinger distance

The squared Hellinger distance is

$$
H^2(P,Q)
=
1-BC(P,Q)
$$

under one common convention.

Thus Bhattacharyya and Hellinger measures are monotone transformations of the same overlap coefficient.

## Gaussian case

For one-dimensional normal distributions

$$
P=N(\mu_1,\sigma_1^2),
\qquad
Q=N(\mu_2,\sigma_2^2),
$$

the Bhattacharyya distance is

$$
D_B
=
\frac14
\log\left[
\frac14
\left(
\frac{\sigma_1^2}{\sigma_2^2}
+
\frac{\sigma_2^2}{\sigma_1^2}
+
2
\right)
\right]
+
\frac14
\frac{(\mu_1-\mu_2)^2}
{\sigma_1^2+\sigma_2^2}.
$$

This separates location and scale differences.

## Classification connection

The Bhattacharyya coefficient appears in upper bounds on Bayes classification error.

Greater overlap between class-conditional distributions makes classification intrinsically harder.

This is a theoretical connection.

It does not imply that Bhattacharyya distance is itself a training loss for every classifier.

## Histogram comparison

In computer vision, normalized histograms can be compared with Bhattacharyya-type measures.

The result depends on binning.

Changing histogram resolution can change the estimated overlap substantially.

## Bhattacharyya distance versus KL divergence

Bhattacharyya distance is symmetric.

KL divergence is directional.

KL is based on log density ratios.

Bhattacharyya is based on square-root overlap.

The two should not be treated as interchangeable.

## Distribution distance versus loss function

The original article mixed Bhattacharyya distance with a catalogue of regression and classification losses.

These are different objects.

A loss function such as

$$
(y-\hat y)^2
$$

scores a prediction against an outcome.

Bhattacharyya distance compares two distributions.

The fact that both can appear in machine learning does not make them conceptually one family.

## Estimation error

In practice, $P$ and $Q$ are often estimated from data.

The resulting Bhattacharyya distance inherits sampling uncertainty and density-estimation error.

Plug-in estimates in high dimensions can be unstable.

Bootstrap or model-based uncertainty assessment may be appropriate.

## Conclusion

Bhattacharyya distance is best understood as an overlap measure between probability distributions.

Its strongest interpretations concern:

- distribution similarity
- class separability
- histogram overlap
- parametric distribution comparison

It should not be bundled indiscriminately with predictive loss functions.

## References

- Bhattacharyya, A. (1943). On a Measure of Divergence Between Two Statistical Populations.
- Kailath, T. (1967). The Divergence and Bhattacharyya Distance Measures in Signal Selection.
