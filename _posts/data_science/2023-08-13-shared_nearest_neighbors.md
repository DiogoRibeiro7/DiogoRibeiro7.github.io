---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-08-13'
excerpt: SNN is a distance metric that enhances traditional methods like k Nearest
  Neighbors, especially in high-dimensional, variable-density datasets.
header:
  image: /assets/images/headers/photo-wind-turbines.jpg
  og_image: /assets/images/headers/photo-wind-turbines.jpg
  overlay_image: /assets/images/headers/photo-wind-turbines.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-wind-turbines.jpg
  twitter_image: /assets/images/headers/photo-wind-turbines.jpg
keywords:
- Shared nearest neighbors
- Snn
- Outlier detection
- Clustering algorithms
- K-nearest neighbors
- High dimensionality
- Distance metrics
- Machine learning
- Python
permalink: '/data-science/shared_nearest_neighbors/'
redirect_from:
- '/data science/shared_nearest_neighbors/'
seo_description: An exploration of Shared Nearest Neighbors (SNN) as a distance metric,
  and its application in outlier detection, clustering, and density-based algorithms.
seo_title: Shared Nearest Neighbors in Outlier Detection
seo_type: article
summary: Shared Nearest Neighbors (SNN) is a distance metric designed to enhance outlier
  detection, clustering, and predictive modeling in datasets with high dimensionality
  and varying density. This article explores how SNN mitigates the weaknesses of traditional
  metrics like Euclidean and Manhattan, providing robust performance in complex data
  scenarios.
tags:
- Machine Learning
- Anomaly Detection
- Clustering
- Data Science
- Mathematical Modeling
- Python
title: Exploring Shared Nearest Neighbors (SNN) for Outlier Detection
---

Shared Nearest Neighbors (SNN) is a neighborhood-overlap similarity construction used in clustering and anomaly detection. Its motivation is not that Euclidean or Manhattan distance simply “fails” in every high-dimensional problem, but that absolute distances can become difficult to interpret when neighborhoods are unstable, irrelevant dimensions dominate, or local density varies strongly. SNN replaces the raw magnitude of distance with overlap among local neighbor sets, so two observations are considered similar when they are embedded in much the same local neighborhood. The method still depends on the original nearest-neighbor graph, which means that feature scaling, representation, the base metric, and the choice of $k$ remain central modeling decisions.

## The Importance of Distance Metrics in Machine Learning

Machine learning algorithms that rely on distance calculations are ubiquitous in data science. Distance metrics are a foundational concept in several key tasks, including:

1. **Predictive Modeling:** Distance metrics are integral in models like k Nearest Neighbors (kNN). For classification tasks, kNN uses distance metrics to identify the k most similar records to the instance being classified. The majority class of these neighbors is then assigned as the prediction. For regression, the predicted value is typically the average of the target values of the k nearest neighbors.

2. **Clustering:** Distance metrics play a central role in clustering algorithms. Whether using centroid-based clustering (e.g., k-means), density-based clustering (e.g., DBSCAN), or hierarchical clustering, distance measures are used to group similar instances into clusters. The choice of metric significantly impacts the performance and accuracy of the clustering process.

3. **Outlier Detection:** Outlier detection algorithms frequently rely on distance calculations. Methods like k Nearest Neighbors outlier detection, Local Outlier Factor (LOF), and Local Outlier Probabilities (LoOP) utilize distance metrics to quantify how far a record is from its nearest neighbors. Records that are significantly farther from their neighbors than average are flagged as outliers.

### Why Standard Distance Metrics Fail in High-Dimensional Data

The **curse of dimensionality** is one of the most challenging problems in machine learning, especially when working with distance-based algorithms. As the number of dimensions increases, the distance between all points tends to converge, making it difficult to distinguish between truly similar and dissimilar records. This phenomenon manifests in several ways:

- **Equidistant Points:** As the number of dimensions grows, all points tend to become equidistant from one another, which weakens the discriminative power of distance metrics like Euclidean and Manhattan.
- **Irrelevant Features:** High-dimensional data often contains irrelevant or redundant features, which distort distance calculations by introducing noise. Distance metrics fail to account for the varying relevance of features in different contexts.
- **Dense vs. Sparse Regions:** Real-world data often has regions with variable density. For example, financial transactions may contain a large number of low-value sales and a few high-value transactions. In such datasets, standard distance metrics may misclassify instances from sparse regions as outliers, even if they are not unusual within their context.

### Addressing the Curse of Dimensionality with SNN

Formally, the shared-neighbor count is a **similarity**, not a distance. If $N_k(i)$ denotes the $k$ nearest neighbors of observation $i$, then a basic similarity is

$$
s(i,j)=|N_k(i)cap N_k(j)|.
$$

Larger values mean more local structure is shared. Any clustering algorithm that expects distances must therefore convert this similarity into a dissimilarity explicitly. This distinction is not cosmetic: the previous implementation passed shared-neighbor similarities directly to DBSCAN as a precomputed distance matrix, reversing the intended geometry.

## Origins of Shared Nearest Neighbors in Clustering

The shared-neighbor idea predates DBSCAN and is closely associated with the Jarvis-Patrick clustering framework. Later SNN density methods reused the idea to reduce sensitivity to raw-distance scale. DBSCAN itself defines neighborhoods through a global radius $\varepsilon$, so a single radius can be problematic when the data contain clusters with sharply different local scales. SNN changes the neighborhood representation, but it does not remove parameter sensitivity or guarantee recovery of clusters with arbitrary density variation.

### A Quick Overview of DBSCAN

To understand how SNN improves DBSCAN, let's briefly review how DBSCAN works. DBSCAN clusters data based on the following principles:

1. **Core Points:** A point is considered a core point if it has a sufficient number of neighbors within a specified distance (`eps`). These core points form the backbone of clusters.
2. **Directly Density-Reachable Points:** Points that are within `eps` distance from a core point are considered to be part of the same cluster.
3. **Noise Points:** Points that do not belong to any cluster are labeled as noise, or outliers.

DBSCAN is effective at clustering datasets where clusters have roughly uniform densities. However, when different clusters have different densities, DBSCAN's global `eps` threshold fails to distinguish between dense and sparse clusters.

### Enhancing DBSCAN with Shared Nearest Neighbors

Shared Nearest Neighbors (SNN) addresses the density problem in DBSCAN by refining how distances are measured. Instead of relying on the direct Euclidean or Manhattan distances between points, SNN considers the number of shared neighbors between two points. In essence, two points are considered similar if they share many of the same nearest neighbors, even if their raw distance is large.

SNN-based density clustering can reduce sensitivity to some local-density differences, but it introduces its own parameters and does not guarantee correct recovery of clusters with arbitrary density variation. By shifting the focus from absolute distances to shared neighborhoods, SNN adapts more effectively to the local structure of the data. Here’s a general process of how SNN works for clustering:

1. **Calculate Pairwise Distances:** Start by calculating the pairwise distances between all points using a traditional metric like Euclidean or Manhattan.
2. **Determine Nearest Neighbors:** For each point, find the k nearest neighbors based on the pairwise distances.
3. **Calculate Shared Nearest Neighbors:** For each pair of points, count how many of their k nearest neighbors are shared. The more neighbors they share, the closer they are considered to be.
4. **Cluster Formation (DBSCAN-like):** Once the shared-neighbor distances are calculated, clustering proceeds similarly to DBSCAN. Core points are identified, clusters are expanded, and noise points are labeled as outliers.

### The Jarvis-Patrick Clustering Algorithm

The concept of shared nearest neighbors is not entirely new. In fact, the earliest iteration of this idea can be traced back to the **Jarvis-Patrick Clustering Algorithm** (1973). This algorithm grouped points based on the number of shared neighbors, but it did not gain widespread popularity due to its computational cost. SNN builds on this foundational idea but introduces a more robust approach for modern applications, such as outlier detection and high-dimensional clustering. While DBSCAN works well for uniform-density data, SNN is one candidate for data where shared local neighborhoods carry useful structure. Whether it improves clustering must be established empirically against simpler metrics and density methods.

## Shared Nearest Neighbors in Outlier Detection

In anomaly detection, the relevant question is whether an observation has weak support from the local structure represented by the chosen neighborhood graph. kNN distance, LOF, and SNN-based scores quantify that support differently. None has a universal claim to robustness, so an SNN anomaly score should be defined mathematically and validated against the type of anomalies the application actually cares about.

### Challenges in Outlier Detection

The primary challenge in outlier detection lies in accurately quantifying how "different" a point is from the rest of the data. Traditional methods, such as kNN outlier detection, compute the average or maximum distance between a point and its k nearest neighbors. However, this can be problematic in two key scenarios:

1. **High-Dimensional Data:** As mentioned earlier, distance metrics become less reliable as the number of dimensions increases. In high-dimensional spaces, distances between points tend to converge, making it difficult to distinguish between normal and anomalous points.

2. **Varying Densities:** In datasets with regions of varying density, traditional outlier detection methods often struggle. For example, in a financial dataset, the density of low-value transactions may be much higher than the density of high-value transactions. A traditional kNN outlier detector might incorrectly classify high-value transactions as outliers because their distances to neighbors are greater than those in the low-value region.

### How SNN Improves Outlier Detection

Shared Nearest Neighbors (SNN) offers a solution to these challenges by focusing on the local structure of the data rather than on absolute distances. Two points are considered similar not just because they are close in terms of raw distance, but because they share a similar neighborhood. In other words, the number of shared nearest neighbors between two points is a better indicator of their similarity than their direct distance.

#### SNN in kNN Outlier Detection

In the k Nearest Neighbors (kNN) outlier detection algorithm, each point's outlier score is typically the average distance to its k nearest neighbors. If a point is much farther from its neighbors than average, it is flagged as an outlier. SNN modifies this approach by using shared neighbors as the distance metric. Instead of calculating the raw distance between a point and its neighbors, SNN computes the number of neighbors that the point shares with each of its k nearest neighbors. A point that shares few or no neighbors with its nearest neighbors is considered an outlier, even if its raw distance is not extreme.

This approach is especially useful in datasets with varying densities, where traditional distance-based methods tend to fail. In such datasets, SNN's focus on shared neighbors provides a more reliable indication of whether a point is truly an outlier. For example, in a dataset of financial transactions, a high-value transaction may be distant from its neighbors in terms of raw dollar amount but may share many neighbors with them based on other features (e.g., the time of the transaction, the type of account involved). SNN would correctly classify this point as normal, while a traditional kNN outlier detector might misclassify it as an outlier.

### SNN and DBSCAN for Outlier Detection

SNN can also be integrated with density-based clustering algorithms like DBSCAN for outlier detection. DBSCAN naturally handles outliers (referred to as "noise points") by leaving them unclustered. However, DBSCAN's effectiveness depends on the choice of distance metric, which is typically Euclidean. By using SNN distances instead of Euclidean distances, DBSCAN becomes more robust to high-dimensional data and varying densities. In SNN-DBSCAN, the outliers are points that have few or no shared neighbors with other points, and thus, are left unclustered.

This approach is particularly powerful in datasets where outliers are not just distant from the rest of the data but also reside in regions of low density. By focusing on shared neighbors, SNN-DBSCAN can identify outliers that might be missed by other methods.

## Advantages of SNN Over Traditional Distance Metrics

SNN offers several key advantages over traditional distance metrics like Euclidean and Manhattan distances, particularly in the context of outlier detection and clustering:

1. **Rank-based local structure:** SNN can sometimes preserve useful neighborhood structure when absolute distances concentrate, but it still inherits errors from the underlying nearest-neighbor search. By replacing raw distance with neighborhood overlap, SNN can be useful when local rank structure is more stable than absolute distances. It does not remove the curse of dimensionality: the original k-nearest-neighbor graph still depends on a base metric and feature representation.

2. **Adaptability to Varying Densities:** Traditional distance-based methods struggle with datasets that have regions of varying density. SNN, by contrast, adapts to local densities by focusing on shared neighborhoods. This makes it particularly well-suited to real-world datasets with uneven distributions.

3. **Alternative anomaly signal:** low shared-neighbor support can be an anomaly score, but there is no general guarantee that it is more robust than LOF, kNN distance, isolation methods, or density estimators because they have few or no shared neighbors with other points. This makes SNN especially useful in applications like fraud detection and network security.

4. **Compatibility with graph-based methods:** shared-neighbor similarities can be used in clustering or anomaly methods that accept a similarity graph or a properly constructed dissimilarity matrix. This flexibility makes SNN a valuable tool for a wide range of machine learning tasks.

### Limitations of SNN

Despite its advantages, SNN is not without its limitations:

1. **Computational Complexity:** Calculating shared nearest neighbors is more computationally expensive than calculating raw distances. This makes SNN less suitable for very large datasets unless optimizations like approximate nearest neighbor search are employed.

2. **Parameter Sensitivity:** SNN relies on the choice of the parameter k (the number of nearest neighbors to consider). If k is set too low, the metric may not capture meaningful relationships between points. If k is set too high, the metric may lose its discriminative power, especially in datasets with a large number of irrelevant features.

3. **Mixed-Type Data:** SNN is primarily designed for numeric data. When working with datasets that contain categorical, date, or text features, SNN may not perform as well without appropriate preprocessing or feature engineering.

## Practical Implementation of SNN

In practice, implementing SNN involves two key steps:

1. **Calculate Pairwise Distances:** Use a traditional distance metric (e.g., Euclidean or Manhattan) to calculate the pairwise distances between points.

2. **Determine Shared Nearest Neighbors:** For each pair of points, count how many of their nearest neighbors are shared. This shared-neighbor count is used as the distance metric in subsequent calculations.

We will include detailed Python implementations of SNN-based outlier detection and SNN-enhanced DBSCAN in the appendix. These implementations can be used as a starting point for applying SNN to your own datasets.

## Appendix: Python Code for SNN Outlier Detection

Below is the Python code for implementing SNN-based outlier detection, along with an SNN-enhanced version of DBSCAN for clustering and outlier detection.

### A safer shared-neighbor implementation

The original implementation returned **larger values for more similar pairs** and then passed that matrix to DBSCAN with metric="precomputed". DBSCAN interprets smaller values as closer distances, so the geometry was reversed. Worse, pairs with no mutual-neighbor relation were left at zero and therefore treated as identical. A correct implementation should distinguish similarity from distance explicitly.

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64] def shared_neighbor_distance(
    x: FloatArray,
    *,
    k: int,
) -> FloatArray:
    if x.ndim != 2:
        raise ValueError(
            "x must be a two-dimensional array."
        ) if not 1 <= k < x.shape[0]:
        raise ValueError(
            "k must lie between 1 and n_samples - 1."
        ) nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
    ).fit(x) indices: IntArray = nn.kneighbors(
        return_distance=False
    )[:, 1:] neighbor_sets = [
        set(row.tolist())
        for row in indices
    ] n: int = x.shape[0] distance = np.full(
        (n, n),
        fill_value=float(k),
        dtype=float,
    ) np.fill_diagonal(
        distance,
        0.0,
    ) for i in range(n):
        for j in range(i + 1, n):
            shared = len(
                neighbor_sets[i]
                & neighbor_sets[j]
            )

            # Larger overlap means smaller dissimilarity.
            d_ij: float = float(
                k - shared
            )

distance[i, j] = d_ij
            distance[j, i] = d_ij return distance distance = shared_neighbor_distance(
    x,
    k=20,
) labels = DBSCAN(
    eps=8.0,
    min_samples=5,
    metric="precomputed",
).fit_predict(distance)
~~~ This is still only a teaching implementation. Computing all pairwise shared-neighbor values requires

$$
O(n^2)
$$

storage and work after the neighbor search. For large datasets, construct a sparse kNN graph and compute overlaps only for candidate edges.

## SNN anomaly scores need a definition

An SNN anomaly score is not unique. Possible choices include:

- average dissimilarity to the strongest shared neighbors;
- number of edges above a shared-neighbor threshold;
- density of the SNN graph around a point;
- cluster/noise labels from an SNN density algorithm.

Each produces a different ranking. An article should therefore not refer to "the SNN outlier score" without defining the score mathematically.

## Parameter selection

SNN has at least two scales:

1. $k$, which defines the original neighborhood;
2. a shared-neighbor threshold or density parameter used afterward.

These should be tuned using stability, held-out labels when available, or domain-scale arguments. Choosing $k=100$ and DBSCAN eps=975, as in the previous code, had no meaningful relationship to a shared-neighbor count bounded by $k$ and exposed the similarity/distance error directly.

## Feature scaling still matters

SNN begins by constructing a nearest-neighbor graph. If that graph uses Euclidean distance, variables with larger numerical scale can dominate. Standardization, domain-specific metrics, embeddings, or learned representations may therefore be required before SNN. Shared neighbors cannot repair a meaningless initial metric.

## High dimensionality

Distance concentration can make nearest-neighbor identities unstable in high dimensions. SNN sometimes improves robustness by using rank overlap, but it does not eliminate this instability. Useful checks include:

- neighbor-set stability under resampling;
- sensitivity to $k$;
- comparison across metrics;
- dimensionality reduction fitted inside the validation process.

## Conclusion

SNN should be understood as a graph-based similarity construction:

$$
s(i,j)
=
|N_k(i)\cap N_k(j)|.
$$

A clustering or anomaly method must then define how that similarity becomes a decision. The main lesson is

$$
\boxed{
\text{nearest-neighbor graph}
\rightarrow
\text{shared-neighbor similarity}
\rightarrow
\text{explicit dissimilarity or density rule}.
}
$$

Skipping the middle distinction, as the previous implementation did, reverses the geometry of the algorithm.

## References

- Jarvis, R. A., & Patrick, E. A. (1973). Clustering using a similarity measure based on shared near neighbors. *IEEE Transactions on Computers*, C-22(11), 1025–1034.
- Ertöz, L., Steinbach, M., & Kumar, V. (2003). Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data. *Proceedings of SIAM SDM*.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD*, 226–231.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *Proceedings of SIGMOD*, 93-104.
