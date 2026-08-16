---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-02-02'
excerpt: Dive into Topological Data Analysis (TDA) and discover how its methods, such
  as persistent homology and the mapper algorithm, help uncover hidden insights in
  high-dimensional and complex datasets.
header:
  image: /assets/images/data_science_8.avif
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_8.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.avif
  twitter_image: /assets/images/data_science_1.jpg
keywords:
- Topological data analysis
- Persistent homology
- Mapper algorithm
- Data science
- Computational topology
- High-dimensional data
- Anomaly detection
- Network analysis
- Interdisciplinary data science
- Mathematical foundations
permalink: '/data-science/topology_data_science/'
redirect_from:
- '/data science/topology_data_science/'
seo_description: Topological Data Analysis (TDA) in data science, from persistent homology to the mapper algorithm, revealing structure in complex datasets.
seo_title: Topological Data Analysis (TDA) in Data Science
seo_type: article
subtitle: Exploring Topological Data Analysis and Its Impact on Uncovering Hidden
  Insights in Complex Data Sets
tags:
- Data Science
- Machine Learning
- Data Analysis
- Data Engineering
- Graph Theory
- Anomaly Detection
title: Convergence of Topology and Data Science
toc: false
toc_label: The Complexity of Real-World Data Distributions
---

The relationship between topology, a branch of mathematics concerned with the properties of space that are preserved under continuous deformation, and data science is less exotic than it first appears. Topology studies shape without reference to distance or coordinates, and a great deal of high-dimensional data has shape that survives the choice of coordinates while its distances do not.

## Why Shape Survives When Distance Does Not

Topological properties are those unchanged by stretching, bending or twisting — but not tearing or gluing. A coffee cup and a doughnut are topologically identical because each has exactly one hole. The precise diameter of that hole is geometric information; the fact that there is one is topological.

This distinction matters in practice for two reasons.

High-dimensional distances become uninformative. As dimension grows, the ratio between the nearest and farthest neighbour of a point tends toward one, so "close" and "far" stop distinguishing anything. Methods that depend on absolute distances degrade; methods that depend on which points are connected to which degrade more slowly.

Coordinates are usually arbitrary. Gene expression levels, sensor readings and survey responses are measured in units chosen for convenience. Any conclusion that changes when you rescale a feature was a statement about your units, not your data. Topological summaries are invariant to that kind of transformation.

## Topological Data Analysis (TDA)

TDA turns a point cloud into a shape and then measures that shape. The central method is **persistent homology**.

The construction is simple to describe. Place a ball of radius $\varepsilon$ around every data point and connect points whose balls overlap, producing a simplicial complex — a graph with higher-dimensional faces filled in. Then grow $\varepsilon$ from zero upward and watch what happens.

At $\varepsilon = 0$ every point is its own component. As $\varepsilon$ grows, components merge, loops form and later fill in, voids open and close. Each feature has a **birth** radius where it appears and a **death** radius where it disappears, and its **persistence** is the difference.

The key idea is that features persisting across a wide range of scales are structural, while those appearing and vanishing quickly are noise. This is what makes the method robust: rather than choosing a scale and hoping it is right, persistent homology reports every scale at once and lets persistence rank the findings.

Results are displayed as a **persistence diagram** — each feature a point plotted at (birth, death), with long-lived features sitting far from the diagonal — or as a **barcode**, where each feature is a horizontal bar.

The features are counted by dimension. $H_0$ counts connected components, $H_1$ counts loops, $H_2$ counts enclosed voids. A dataset sampled from a circle shows one persistent $H_1$ feature; one sampled from a sphere shows a persistent $H_2$ feature and no $H_1$.

```python
import numpy as np
from ripser import ripser
from persim import plot_diagrams

rng = np.random.default_rng(0)
theta = rng.uniform(0, 2 * np.pi, 300)
noisy_circle = np.column_stack([np.cos(theta), np.sin(theta)]) \
    + rng.normal(0, 0.08, (300, 2))

result = ripser(noisy_circle)
dgms = result["dgms"]          # dgms[0] = H0, dgms[1] = H1

h1 = dgms[1]
lifetimes = h1[:, 1] - h1[:, 0]
print(f"H1 features found      : {len(h1)}")
print(f"longest-lived lifetime : {lifetimes.max():.3f}")
print(f"next longest           : {np.sort(lifetimes)[-2]:.3f}")

plot_diagrams(dgms, show=True)
```

One $H_1$ feature dominates and the rest are near-zero: the loop is real, the others are sampling noise. Notably, k-means on this data would report clusters that do not exist, because a circle has no clusters — it has a hole.

## The Mapper Algorithm

The second workhorse of TDA is **Mapper**, which builds a compressed graph summarising a dataset's shape.

It works in three steps. A filter function projects the data to a low-dimensional space — density, a principal component, or an outcome variable. The range of that filter is covered by overlapping intervals. Within each interval the original points are clustered, each cluster becomes a node, and nodes sharing points are joined by an edge.

The result is a graph whose flares, loops and branches correspond to structure in the original space. Mapper's best-known application identified a subgroup of breast cancer patients with distinctive survival, visible as a flare in the graph that conventional clustering had not separated.

Mapper is genuinely sensitive to its parameters: the filter, the number of intervals, the overlap fraction and the clustering algorithm all change the output. It is an exploratory instrument, and a flare that appears under one parameter setting and vanishes under neighbouring ones should not be trusted.

## Applications in Data Science

The methods earn their place where shape is the signal.

In biology, persistent homology detects structure in gene expression and protein folding data where the relevant relationships are not linear. In time series, embedding a signal into a delay-coordinate space turns periodicity into a loop, so an $H_1$ feature detects cyclic behaviour without assuming a period — useful for physiological signals and machine vibration.

For machine learning, persistence diagrams can be vectorised into persistence images or landscapes and used as features alongside conventional ones. This is where most practical value lies, since topological features capture properties that summary statistics miss entirely.

In anomaly detection, points that appear only as short-lived components, or that create a topological feature no other point supports, are anomalous in a sense distinct from being far from the mean.

## Limitations Worth Stating

TDA is not a replacement for statistics, and three constraints bound its usefulness.

Computation scales badly. Building the full complex is exponential in dimension in the worst case, and although Vietoris-Rips with sparse approximations and libraries like Ripser have made moderate datasets tractable, this is not a method to reach for on millions of points without care.

Interpretation is genuinely hard. Knowing there is a persistent one-dimensional hole in a 40-dimensional dataset is a fact about the data whose *meaning* still requires domain knowledge to establish. The method tells you the shape, not what the shape means.

Statistical inference is less developed than in classical statistics. Bootstrap confidence bands for persistence diagrams exist, but deciding whether an observed feature is significantly persistent is not as routine as computing a p-value, and the temptation to read a long bar as a discovery should be resisted without one.

## Significance of the Relationship

Topology contributes something specific to data science: a way of describing structure that does not depend on coordinates, scales gracefully as dimensions grow, and is stable under small perturbations of the data. Where a dataset's important property is that it loops, branches or encloses a void, no amount of correlation analysis will find it, because those are not properties any pairwise summary can express.

That is a narrow contribution rather than a general-purpose one, and it is most valuable as a complement to conventional methods rather than a substitute for them.

## References

- Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255-308.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
- Nicolau, M., Levine, A. J., & Carlsson, G. (2011). Topology based data analysis identifies a subgroup of breast cancers with a unique mutational profile and excellent survival. *PNAS*, 108(17), 7265-7270.
- Ghrist, R. (2008). Barcodes: the persistent topology of data. *Bulletin of the American Mathematical Society*, 45(1), 61-75.
- Chazal, F., & Michel, B. (2021). An introduction to topological data analysis. *Frontiers in Artificial Intelligence*, 4, 667963.
