---
permalink: '/time-series/dynamic_time_warping_time_series_clustering/'
title: 'Dynamic Time Warping and Time Series Clustering'
categories:
- Time Series
tags:
- Time Series
- Clustering
- Unsupervised Learning
- Python
author_profile: false
seo_title: 'Dynamic Time Warping and Clustering'
seo_description: 'Euclidean distance fails when two series have the same shape at different speeds. How DTW aligns them, and what that means for clustering.'
excerpt: >-
  Two series can trace an identical shape while one runs slightly ahead of the
  other. Point-by-point distance calls them dissimilar; dynamic time warping
  does not.
summary: >-
  How dynamic time warping aligns series that differ in timing rather than
  shape, why it is not a metric and what that costs, the constraints that make
  it tractable, and how to cluster series sensibly once you have a distance
  that reflects shape.
keywords:
  - dynamic time warping
  - DTW
  - time series clustering
  - shape-based distance
  - k-medoids
classes: wide
date: '2026-07-01'
header:
  image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_1.jpg
  twitter_image: /assets/images/data_science_1.jpg
---
Two machines run the same production cycle. One runs slightly faster. Their sensor traces have identical shape — same peaks, same troughs, same order — but at every instant the values differ, because one is a little ahead. Euclidean distance reports them as dissimilar. Any clustering built on that distance separates them.

Dynamic time warping exists to fix exactly this: comparing series by shape rather than by simultaneity.

## What Warping Means

Euclidean distance pairs observation $i$ in one series with observation $i$ in the other, and only ever that pairing. DTW instead searches over all monotonic alignments — mappings that may stretch or compress the time axis — and returns the cost of the cheapest one.

Formally, build the cost matrix $D$ where $D_{ij}$ accumulates the distance between prefix $i$ of the first series and prefix $j$ of the second:

$$
D_{ij} = d(x_i, y_j) + \min\left(D_{i-1,j},\ D_{i,j-1},\ D_{i-1,j-1}\right).
$$

Each step takes the cheapest way of arriving: consume a point from the first series, from the second, or from both. The final entry $D_{nm}$ is the DTW distance, and tracing the minimising path backwards recovers the alignment itself.

Three constraints define a valid path. It starts at $(1,1)$ and ends at $(n,m)$ — **boundary**. It never moves backwards in either series — **monotonicity**. And it advances by at most one index at a time — **continuity**. Together these guarantee every observation is matched at least once and the ordering is preserved.

```python
import numpy as np

def dtw(x, y, window=None):
    """DTW distance with an optional Sakoe-Chiba band."""
    n, m = len(x), len(y)
    w = max(window or max(n, m), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])

t = np.linspace(0, 2 * np.pi, 60)
a = np.sin(t)
b = np.sin(t * 1.15)          # same shape, running slightly faster
c = np.sin(t) * 0.2           # same timing, much smaller amplitude

euclid = lambda p, q: np.sqrt(((p - q) ** 2).sum())
print(f"{'':12}{'euclidean':>11}{'DTW':>9}")
print(f"{'a vs b':12}{euclid(a, b):11.3f}{dtw(a, b):9.3f}")
print(f"{'a vs c':12}{euclid(a, c):11.3f}{dtw(a, c):9.3f}")
```

The comparison is the point. Series `b` differs from `a` only in speed and `c` differs only in amplitude. Euclidean distance treats the timing difference as substantial; DTW recognises the shapes as nearly identical while still keeping the genuinely different amplitude apart.

## The Constraints That Make It Usable

Unconstrained DTW has two problems.

It is $O(nm)$ in time and memory, which becomes prohibitive on long series and on the all-pairs distance matrix that clustering requires.

More subtly, unconstrained warping is *too* permissive: a single point in one series can be matched to a long stretch of the other, producing pathological alignments that map a brief spike onto an entire plateau.

The **Sakoe-Chiba band** restricts the path to within $w$ steps of the diagonal, which both bounds the cost to $O(nw)$ and prevents degenerate alignments. The window size encodes a real assumption — how much timing distortion you believe exists — and small windows often perform *better* than large ones, not merely faster.

For large-scale work, lower-bounding techniques such as LB_Keogh compute a cheap bound first and skip the full computation whenever the bound already exceeds the current best. This is what makes nearest-neighbour search over large collections tractable.

## Why DTW Is Not a Metric

DTW violates the triangle inequality. It is possible to have $d(x,z) > d(x,y) + d(y,z)$.

That is not a technicality. Many algorithms assume a metric space: k-means relies on centroids being meaningful, and indexing structures such as ball trees and k-d trees require the triangle inequality to prune correctly. Using DTW with them is unsound, even though the code runs.

The consequences for clustering are specific. **k-medoids** works, because it uses actual series as cluster centres and needs only a distance matrix. **Hierarchical clustering** works for the same reason. **k-means** does not work directly, because averaging series under DTW is not an ordinary mean — the correct analogue is DBA (DTW Barycentre Averaging), an iterative procedure that computes a consensus series under warping.

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

rng = np.random.default_rng(0)
# three groups: same shapes at different speeds and phases
series, truth = [], []
for g, (freq, amp) in enumerate([(1.0, 1.0), (1.0, 0.3), (2.0, 1.0)]):
    for _ in range(8):
        speed = rng.uniform(0.9, 1.1)
        phase = rng.uniform(0, 0.4)
        s = amp * np.sin(freq * speed * t + phase) + rng.normal(0, 0.05, t.size)
        series.append(s); truth.append(g)
series, truth = np.array(series), np.array(truth)

n = len(series)
Dm = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        Dm[i, j] = Dm[j, i] = dtw(series[i], series[j], window=8)

labels = fcluster(linkage(squareform(Dm), method="average"), 3, criterion="maxclust")
purity = sum(np.bincount(truth[labels == k]).max() for k in np.unique(labels)) / n
print(f"cluster purity against the true groups: {purity:.2f}")
```

Cluster purity is reported rather than accuracy because cluster labels are arbitrary — the algorithm has no obligation to number groups the way you did.

## When Not to Use It

DTW is the right tool when timing distortion is genuine and irrelevant to your question — gesture recognition, speech, repeated production cycles, gait analysis.

It is the wrong tool when timing *is* the signal. If two patients' symptoms follow the same trajectory but one deteriorates twice as fast, warping the difference away discards the clinically important part. If you are aligning series that must stay on a shared calendar, warping is meaningless: January is January.

DTW is also insensitive to amplitude scaling unless you normalise deliberately. Z-normalising each series first compares pure shape; leaving them raw keeps magnitude in play. Neither is right in general, and the choice should be explicit.

A final caution from the classification literature: one-nearest-neighbour with DTW is a famously strong baseline that many elaborate methods fail to beat. Before reaching for something more sophisticated, run it.

## References

- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- Keogh, E., & Ratanamahatana, C. A. (2005). Exact indexing of dynamic time warping. *Knowledge and Information Systems*, 7(3), 358-386.
- Petitjean, F., Ketterlin, A., & Gançarski, P. (2011). A global averaging method for dynamic time warping, with applications to clustering. *Pattern Recognition*, 44(3), 678-693.
- Aghabozorgi, S., Shirkhorshidi, A. S., & Wah, T. Y. (2015). Time-series clustering: a decade review. *Information Systems*, 53, 16-38.
- Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off. *Data Mining and Knowledge Discovery*, 31(3), 606-660.
