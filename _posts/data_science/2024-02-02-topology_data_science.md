---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-02-02'
excerpt: "Topological data analysis summarizes multiscale shape through constructions such as persistent homology and Mapper, but topology does not make high-dimensional geometry or preprocessing irrelevant."
header:
  image: /assets/images/headers/photo-solar-panels.jpg
  og_image: /assets/images/headers/photo-solar-panels.jpg
  overlay_image: /assets/images/headers/photo-solar-panels.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-solar-panels.jpg
  twitter_image: /assets/images/headers/photo-solar-panels.jpg
keywords:
- Topological data analysis
- Persistent homology
- Mapper
- Persistence diagram
- Simplicial complexes
permalink: '/data-science/topology_data_science/'
redirect_from:
- '/data science/topology_data_science/'
seo_description: "A rigorous introduction to topological data analysis, persistent homology, persistence diagrams, Mapper, stability, and the dependence on metric and filtration choices."
seo_title: "Topological Data Analysis: Shape Across Scales"
seo_type: article
tags:
- Topological Data Analysis
- Mathematics
- Data Science
title: "Topological Data Analysis: Shape Across Scales"
---

Topological data analysis studies qualitative shape in data across multiple scales.

The key idea is not that topology is immune to geometry. TDA begins from a metric, similarity, graph, or filtration. Those choices determine the topological summaries that follow.

## From points to complexes

Suppose observations are points $x_1,\ldots,x_n$ in a metric space.

For scale $\varepsilon$, a Vietoris-Rips complex connects points whose pairwise distances are sufficiently small and fills higher-dimensional simplices whenever all required edges are present.

As $\varepsilon$ increases, the complex changes.

Connected components merge, loops appear and disappear, and higher-dimensional cavities can emerge.

## Homology

Homology groups summarize holes by dimension.

- $H_0$: connected components
- $H_1$: loops
- $H_2$: voids

Their ranks are Betti numbers.

For one fixed scale, these quantities can be extremely sensitive to the chosen threshold. Persistent homology avoids committing to one threshold.

## Persistence

A topological feature has a birth scale $b$ and death scale $d$.

Its lifetime is

$$
\ell=d-b.
$$

Persistence diagrams represent features as points $(b,d)$.

Long-lived features are often treated as more structurally important than short-lived features, but “short-lived = noise” is not a theorem about the data-generating process. Small real structures can be short, and sampling artifacts can sometimes persist.

Interpretation remains domain-dependent.

## Stability

One strength of persistent homology is stability: small perturbations of the underlying metric data can lead to bounded changes in persistence diagrams under suitable conditions.

This is a precise mathematical robustness statement.

It does not mean TDA is invariant to arbitrary feature scaling, metric choice, or preprocessing.

## Metric choice matters

Euclidean distance after standardizing variables and cosine distance on embeddings define different neighborhood structures.

Since the filtration is built from those neighborhoods, they can produce different topology.

Feature units and distance definitions therefore remain part of the model.

## High dimension

It is misleading to say topology automatically solves the curse of dimensionality.

Distance concentration can still damage the neighborhood graph used to construct the filtration.

Dimensionality reduction, representation learning, or domain-specific metrics may be necessary before TDA becomes meaningful.

## Mapper

Mapper produces a graph-like summary.

A typical workflow is:

1. choose a filter function
2. cover its range with overlapping intervals
3. cluster observations inside each interval
4. create one node per local cluster
5. connect nodes that share observations

The result depends on the filter, cover resolution, overlap, clustering method, and metric.

Mapper is therefore an exploratory construction rather than a uniquely determined topological truth.

## Statistical inference

Persistence diagrams are estimates from finite samples.

Inference can use bootstrap ideas, persistence landscapes, silhouettes, kernels, or other vectorized representations.

But dependence, sampling design, and multiple comparisons remain relevant.

## A small example

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ripser import ripser


def noisy_circle(
    n: int = 300,
    noise_sd: float = 0.08,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate points near a unit circle."""
    if n <= 0 or noise_sd < 0:
        raise ValueError("n must be positive and noise_sd non-negative")

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)

    points = np.column_stack((np.cos(theta), np.sin(theta)))
    points += rng.normal(0.0, noise_sd, size=points.shape)
    return points


points = noisy_circle()
diagrams = ripser(points)["dgms"]
~~~

A persistent $H_1$ feature is expected because a circle has one loop.

That example is useful because the topology is known in advance. Real datasets rarely provide such clean ground truth.

## Conclusion

TDA is valuable when shape across scales is scientifically meaningful.

Its central workflow is

$$
\text{metric data}
\rightarrow
\text{filtration}
\rightarrow
\text{homology}
\rightarrow
\text{persistence summary}.
$$

The topological summary is robust in specific mathematical senses, but it still depends on representation, metric, sampling, and scale construction.

## References

- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology*.
- Carlsson, G. (2009). Topology and Data.
- Ghrist, R. (2008). Barcodes: The Persistent Topology of Data.
