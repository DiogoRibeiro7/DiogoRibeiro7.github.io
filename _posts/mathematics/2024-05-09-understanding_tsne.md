---
permalink: '/mathematics/understanding_tsne/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-09'
header:
  image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  og_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  overlay_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  twitter_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
redirect_from:
- '/mathematics/statistics/machine learning/understanding_tsne/'
seo_description: "How t-SNE constructs local probability neighborhoods, minimizes KL divergence, and why its plots must not be read as literal global geometry or cluster evidence."
seo_title: "Understanding t-SNE Without Overinterpreting the Map"
seo_type: article
subtitle: "Local neighborhoods, KL divergence, and visualization limits"
tags:
- Dimensionality Reduction
- Data Visualization
- Machine Learning
title: "Understanding t-SNE Without Overinterpreting the Map"
---

t-SNE is a visualization method, not a general-purpose clustering algorithm and not a faithful low-dimensional reconstruction of all high-dimensional geometry.

Its strength is local neighborhood preservation.

## High-dimensional similarities

For each point $x_i$, t-SNE defines conditional neighbor probabilities

$$
p_{j\mid i}
=
\frac{
\exp(-\|x_i-x_j\|^2/2\sigma_i^2)
}{
\sum_{k\ne i}
\exp(-\|x_i-x_k\|^2/2\sigma_i^2)
}.
$$

The bandwidth $\sigma_i$ is chosen so that the local distribution has a specified perplexity.

Perplexity is therefore related to an effective neighborhood size, not to a literal number of clusters.

## Symmetrization

t-SNE forms joint high-dimensional similarities

$$
p_{ij}
=
\frac{p_{j\mid i}+p_{i\mid j}}{2n}.
$$

In the low-dimensional embedding, similarities are defined with a heavy-tailed Student distribution:

$$
q_{ij}
=
\frac{
(1+\|y_i-y_j\|^2)^{-1}
}{
\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}
}.
$$

The heavy tail helps reduce the crowding problem.

## Objective

t-SNE minimizes

$$
KL(P\|Q)
=
\sum_{i\ne j}
p_{ij}
\log\frac{p_{ij}}{q_{ij}}.
$$

Because the divergence is asymmetric, failing to place high-probability neighbors near one another is penalized strongly.

Putting unrelated points moderately close can be less costly.

This is one reason local structure is emphasized more strongly than global geometry.

## What distances mean

Nearby points in a stable t-SNE embedding often indicate local similarity.

Distances between far-apart clusters should not be interpreted quantitatively.

Likewise, cluster area and apparent density can be distorted by the optimization.

A large empty gap is not evidence for a correspondingly large high-dimensional separation.

## t-SNE can create visual clusters

Even continuous or weakly structured data can produce separated islands under some parameter choices.

Therefore:

> a t-SNE cluster is not, by itself, statistical evidence that a population cluster exists.

Any cluster hypothesis should be checked in the original or otherwise scientifically meaningful representation.

## Perplexity

Low perplexity emphasizes very local neighborhoods.

Higher perplexity pools information over broader neighborhoods.

Good practice is to examine several plausible values and ask which qualitative structures persist.

A single attractive plot chosen after many parameter experiments is weak evidence.

## Initialization and randomness

Different initializations can lead to different layouts with similar objective values.

Recent implementations often support PCA initialization, which can improve stability, but reproducibility still requires fixing the random state and recording software parameters.

## Early exaggeration

t-SNE temporarily magnifies attractive forces early in optimization.

This can help establish separation between local groups before the embedding settles.

It is an optimization device, not a scientific statement about cluster strength.

## PCA before t-SNE

For very high-dimensional noisy data, reducing to a moderate number of principal components before t-SNE can improve computation and remove near-zero-variance directions.

But PCA itself changes the representation.

The number of retained components should therefore be part of the documented pipeline.

## Do not cluster the t-SNE map by default

Running k-means directly on two-dimensional t-SNE coordinates is often difficult to justify.

The embedding was optimized for visualization, not for preserving all distances needed by k-means.

If clustering is the goal, fit the clustering model in the original or a validated representation and use t-SNE only for display.

## A careful Python example

~~~python
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


X, labels = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

embedding = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=42,
).fit_transform(X_scaled)
~~~

The labels are useful for coloring the final plot because they were not used by t-SNE itself.

## Validation strategy

If the visualization suggests a scientific hypothesis, test that hypothesis outside the t-SNE map.

Examples include:

- classification performance under cross-validation
- cluster stability in the original feature space
- known biological markers
- external metadata not used during embedding

This separates visualization from confirmation.

## Conclusion

t-SNE is valuable because it makes local neighborhoods visible.

Its most common misuse is reading the two-dimensional map as though it preserved the original geometry globally.

The safe interpretation is narrower:

$$
\text{local neighborhood evidence}
\neq
\text{global metric truth}.
$$

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE.
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to Use t-SNE Effectively.
