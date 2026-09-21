---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-02-09'
excerpt: "Spectral clustering converts a similarity graph into an eigenvector embedding and then clusters that embedding. The graph construction is the model."
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
keywords:
- Spectral clustering
- Graph Laplacian
- Normalized cut
- Eigenvectors
- Similarity graph
permalink: '/data-science/spectral_clustering/'
redirect_from:
- '/data science/spectral_clustering/'
seo_description: "A rigorous introduction to spectral clustering through similarity graphs, graph Laplacians, normalized cuts, eigenvector embeddings, and parameter sensitivity."
seo_title: "Spectral Clustering: The Graph Is the Model"
seo_type: article
tags:
- Clustering
- Graph Theory
- Machine Learning
title: "Spectral Clustering: The Graph Is the Model"
toc: false
---

Spectral clustering is often described as clustering after dimensionality reduction. That description misses the main idea.

The method first replaces the dataset with a weighted graph. Eigenvectors of a graph Laplacian then reveal partitions that are difficult to express as Euclidean centroid clusters.

The crucial modeling decision is therefore the graph.

## Similarity graph

For observations $x_i$, define weights

$$
w_{ij}\ge0.
$$

A Gaussian kernel is common:

$$
w_{ij}
=
\exp\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right).
$$

But one may instead use a $k$-nearest-neighbor graph, an $\varepsilon$-graph, or domain-specific similarities.

Different graph constructions can produce different clusterings even before eigenvectors are computed.

## Graph Laplacian

Let $W$ be the weight matrix and

$$
D_{ii}=\sum_j w_{ij}.
$$

The unnormalized Laplacian is

$$
L=D-W.
$$

Two normalized forms are

$$
L_{sym}
=
I-D^{-1/2}WD^{-1/2}
$$

and

$$
L_{rw}
=
I-D^{-1}W.
$$

These matrices correspond to related but distinct spectral objectives.

## Connected components

For the unnormalized Laplacian, the multiplicity of eigenvalue zero equals the number of connected components of the graph.

This gives an idealized intuition: if the graph had exactly $k$ disconnected components, the first $k$ eigenvectors would identify them perfectly.

Real graphs are usually only approximately separated.

## Relaxing graph cuts

Partitioning a graph by minimizing combinatorial cut objectives is difficult.

Spectral methods replace discrete indicator variables with continuous eigenvectors.

Normalized-cut formulations account for cluster volume, avoiding some trivial partitions that isolate very small sets.

The eigenvectors therefore arise from a relaxation of a graph-partitioning problem, not from generic dimensionality reduction.

## Spectral embedding

For $k$ clusters, construct a matrix from the relevant $k$ eigenvectors.

Rows of this matrix represent graph nodes in an embedding where graph connectivity is easier to separate.

K-means is often applied to those rows.

The final k-means stage does not mean spectral clustering assumes spherical clusters in the original feature space.

## Kernel scale

The Gaussian bandwidth $\sigma$ can completely change the graph.

If it is too small, the graph fragments.

If it is too large, most points become strongly connected and cluster structure disappears.

Local scaling can help when densities vary, but introduces another modeling choice.

## Number of neighbors

A $k$-nearest-neighbor graph also has a scale parameter.

Too few neighbors create disconnected components driven by sampling noise.

Too many neighbors add shortcuts across genuine manifold structure.

Graph connectivity diagnostics should therefore be inspected before interpreting clusters.

## Eigenvalue gaps

An eigengap can suggest a useful low-dimensional spectral structure.

It is not a guaranteed estimator of the true number of clusters.

Finite samples, weak separation, heterogeneous density, and graph-construction choices can all blur the spectrum.

## Scalability

A dense $n\times n$ similarity matrix requires $O(n^2)$ storage.

Large-scale spectral clustering therefore relies on sparse nearest-neighbor graphs, approximate neighbor search, Nyström approximations, landmark methods, or iterative sparse eigensolvers.

Scalability is primarily a graph and linear-algebra problem.

## Out-of-sample points

Standard spectral clustering is transductive: the embedding is computed for the observed graph.

Assigning new points requires an extension rule, refitting, Nyström-style approximation, or a learned mapping.

This matters in production systems where observations arrive continuously.

## Validation

Internal cluster metrics should be computed with care because the clustering was designed around a particular graph geometry.

Stability under resampling and sensitivity to graph parameters are often more informative than one silhouette score in the original feature space.

## Conclusion

Spectral clustering is best understood as

$$
\text{data}
\rightarrow
\text{similarity graph}
\rightarrow
\text{Laplacian}
\rightarrow
\text{eigenvectors}
\rightarrow
\text{partition}.
$$

The eigenvectors are powerful, but the graph determines what similarity means.

A spectral clustering result is therefore only as defensible as its graph construction.

## References

- von Luxburg, U. (2007). A Tutorial on Spectral Clustering.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On Spectral Clustering: Analysis and an Algorithm.
- Shi, J., & Malik, J. (2000). Normalized Cuts and Image Segmentation.
