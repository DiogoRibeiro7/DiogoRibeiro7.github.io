---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-02-08'
excerpt: "Clustering does not discover a unique hidden partition of data. It produces groups relative to a representation, similarity measure, algorithm, and scale."
header:
  image: /assets/images/headers/photo-wind-turbines.jpg
  og_image: /assets/images/headers/photo-wind-turbines.jpg
  overlay_image: /assets/images/headers/photo-wind-turbines.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-wind-turbines.jpg
  twitter_image: /assets/images/headers/photo-wind-turbines.jpg
keywords:
- Clustering
- K-means
- DBSCAN
- Hierarchical clustering
- Cluster stability
- Similarity metrics
permalink: '/data-science/realworld_data_distributions/'
redirect_from:
- '/data science/realworld_data_distributions/'
seo_description: "A rigorous guide to clustering as a model of similarity, including K-means, hierarchical clustering, DBSCAN, validation, stability, and representation dependence."
seo_title: "Clustering Is a Model of Similarity"
seo_type: article
tags:
- Clustering
- Unsupervised Learning
- Data Science
title: "Clustering Is a Model of Similarity"
---

Clustering is often described as discovering the natural groups hidden in data. That wording is too strong.

A clustering result depends on at least four choices:

$$
\text{representation}
+
\text{distance}
+
\text{algorithm}
+
\text{scale}.
$$

Change any of them and the partition may change.

The correct question is therefore not whether an algorithm recovered the true clusters, unless a generative model or external labels define what truth means. The question is whether the grouping is stable, interpretable, and useful for the scientific or operational task.

## K-means

K-means solves

$$
\min_{C_1,\ldots,C_k}
\sum_{j=1}^k
\sum_{x_i\in C_j}
\|x_i-\mu_j\|_2^2.
$$

It favors compact Euclidean clusters around centroids.

The objective is well defined, but Lloyd's algorithm only finds a local optimum. Initialization matters, which is why multiple restarts or k-means++ initialization are common.

K-means is not appropriate merely because a dataset is numeric. Feature scaling and geometry determine the meaning of Euclidean distance.

## Hierarchical clustering

Agglomerative hierarchical clustering begins with singleton observations and repeatedly merges clusters.

The result depends on the linkage criterion.

Single linkage uses minimum pairwise distance and can recover elongated structures, but is vulnerable to chaining.

Complete linkage emphasizes cluster diameter.

Ward linkage approximately minimizes increases in within-cluster squared error and is naturally tied to Euclidean geometry.

A dendrogram is a hierarchy induced by the algorithm, not proof that the population itself has a hierarchical ontology.

## DBSCAN

DBSCAN defines clusters through local density connectivity.

Its main parameters are neighborhood radius $\varepsilon$ and minimum point count.

This allows non-convex clusters and explicit noise labels, but the result is scale dependent.

If density varies strongly across the dataset, one global $\varepsilon$ can merge dense regions or fragment sparse ones.

A point labeled noise is therefore not intrinsically anomalous. It is noise under the selected metric and density scale.

## Representation comes first

Raw features, standardized variables, PCA scores, learned embeddings, and domain-specific distances can produce different clusterings.

Suppose one variable is measured in euros and another in fractions. Without scaling, the euro variable can dominate Euclidean distance.

Standardization is not automatically correct either. A physically meaningful large-scale variable may deserve more weight.

The metric encodes scientific assumptions.

## High-dimensional data

As dimension grows, distances can concentrate and neighborhoods become unstable.

This does not imply spectral clustering or deep learning automatically solve high-dimensional clustering.

Useful strategies include feature selection, dimension reduction, sparse models, domain-specific metrics, or representation learning validated against the downstream goal.

## How many clusters?

There is no universal estimator of the true cluster count.

Elbow plots, silhouette scores, gap statistics, likelihood criteria, and stability measures encode different objectives.

A plateau or bend in a diagnostic curve is evidence about that criterion, not proof of an ontological number of populations.

## Internal validation is circular in a useful but limited way

Silhouette score evaluates separation using the same distance geometry that defined the clustering.

That can help compare algorithms within one representation.

It cannot tell whether the clusters correspond to meaningful external structure.

External validation, downstream utility, or domain interpretation is needed for that.

## Stability

A useful clustering should not change arbitrarily after small perturbations to the sample.

Bootstrap or subsampling procedures can estimate stability of assignments or co-clustering.

But stability is not truth. A consistently biased representation can produce highly stable but unhelpful groups.

## Clustering for decisions

Customer segmentation illustrates the distinction.

A clustering may produce neat customer groups, but a marketing intervention should be evaluated by incremental response, not by how visually separated the clusters look.

Clustering can generate hypotheses or simplify heterogeneity. It does not establish that different actions are causally optimal for different groups.

## Anomaly detection

Distance from a cluster or DBSCAN noise status can be used as an anomaly score, but only relative to the fitted representation.

Rare legitimate cases can look anomalous. Fraudulent cases can form a dense cluster.

Anomaly detection requires a definition of operational abnormality, not merely geometric isolation.

## A practical workflow

1. Define what similarity should mean.
2. Build a simple representation.
3. Inspect scaling and distance sensitivity.
4. Compare multiple clustering assumptions.
5. Assess stability under resampling.
6. Interpret clusters using variables not used to manufacture the separation when possible.
7. Validate usefulness on an external task.

## Conclusion

Clustering is not a machine for revealing hidden categories.

It is a family of methods for imposing useful summaries on similarity structure.

The strongest analysis makes the construction explicit enough that another analyst can understand why the clusters exist and how they would change under different reasonable choices.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Hennig, C. (2015). What Are the True Clusters?
- von Luxburg, U. (2010). Clustering Stability: An Overview.
