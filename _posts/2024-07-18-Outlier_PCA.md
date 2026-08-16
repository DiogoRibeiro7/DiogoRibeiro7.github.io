---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-07-18'
header:
  image: /assets/images/data_science_4.jpg
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_4.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_4.jpg
  twitter_image: /assets/images/data_science_1.jpg
permalink: '/data-science/Outlier_PCA/'
redirect_from:
- '/data science/machine learning/Outlier_PCA/'
- '/data science/Outlier_PCA/'
seo_description: How Principal Component Analysis detects outliers, the difference between anomalies and novelties, and the methods available.
seo_title: Detecting Outliers with PCA
seo_type: article
tags:
- Dimensionality Reduction
- Anomaly Detection
title: Detecting Outliers Using Principal Component Analysis (PCA)
---

Principal Component Analysis (PCA) is best known as a dimensionality reduction technique, but the same machinery detects outliers. The idea is direct: PCA learns the subspace the bulk of the data occupies, and points that do not fit that subspace stand out.

## Understanding Outlier Detection

An outlier is an observation that deviates markedly from the rest of the data. That definition is deliberately vague, because whether a point is an error, a rare event, or the most interesting observation in the dataset is a question the statistics cannot answer.

What the method can do is quantify *how* unusual a point is, and PCA offers a particular and useful notion of unusual.

### Anomalies vs. Novelties

The distinction is about what the training data contains. **Anomaly detection** assumes the training set is contaminated — anomalies are already present, and the task is to identify them. **Novelty detection** assumes the training set is clean and asks whether a *new* point belongs.

The difference is practical. In anomaly detection the outliers influence the fitted model, dragging the principal components toward themselves and partially masking their own deviation. In novelty detection the subspace is fitted on known-good data and new points are scored against it, which is the cleaner setup when it is available.

## Two Distances, Two Kinds of Outlier

PCA gives two genuinely different scores, and conflating them is the most common mistake in applying it.

**Reconstruction error** measures distance *away from* the subspace. Project a point onto the first $k$ components, map it back, and measure how far it landed from the original:

$$
e_i = \left\lVert x_i - \hat{x}_i \right\rVert^2, \qquad
\hat{x}_i = \mu + W_k W_k^\top (x_i - \mu),
$$

where $W_k$ holds the first $k$ loading vectors. A large error means the point violates the correlation structure that holds for everything else — its features are individually plausible but jointly impossible.

**Mahalanobis distance** measures distance *within* the subspace. Using the retained components and their variances:

$$
d_i^2 = \sum_{j=1}^{k} \frac{z_{ij}^2}{\lambda_j},
$$

where $z_{ij}$ is the score of point $i$ on component $j$ and $\lambda_j$ that component's variance. This finds points that follow the normal correlation pattern but are extreme along it — a genuinely large-but-consistent observation.

These catch different things. A person 2.0 m tall weighing 110 kg has a large Mahalanobis distance and small reconstruction error: big, but proportioned normally. A person 1.5 m tall weighing 110 kg has the reverse: neither value is extreme alone, but the combination breaks the height-weight relationship. Monitoring only one of these misses half the outliers.

In process monitoring these are the classic $T^2$ and $Q$ (or SPE) statistics, and control charts are conventionally maintained for both.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(data)     # scaling is mandatory here

pca = PCA(n_components=0.90)                 # keep 90% of variance
Z = pca.fit_transform(X)
X_hat = pca.inverse_transform(Z)

spe = ((X - X_hat) ** 2).sum(axis=1)                       # Q / reconstruction
t2 = (Z ** 2 / pca.explained_variance_).sum(axis=1)        # Hotelling's T-squared

# thresholds from the observed distribution, not from a normal assumption
spe_lim, t2_lim = np.percentile(spe, 99), np.percentile(t2, 99)
flagged = (spe > spe_lim) | (t2 > t2_lim)

print(f"components retained : {pca.n_components_}")
print(f"flagged by SPE only : {((spe > spe_lim) & (t2 <= t2_lim)).sum()}")
print(f"flagged by T2 only  : {((t2 > t2_lim) & (spe <= spe_lim)).sum()}")
print(f"flagged by both     : {((spe > spe_lim) & (t2 > t2_lim)).sum()}")
```

The counts of "SPE only" versus "T² only" are worth printing every time — if they are both non-trivial, the two statistics are genuinely doing different work on your data.

## Choices That Change the Answer

**Scaling is not optional.** PCA maximises variance, so an unscaled feature measured in larger units dominates the components regardless of its importance. Standardise unless the features share meaningful units and their relative variances are themselves the signal.

**How many components to keep** decides what counts as an outlier. Keep too many and the subspace absorbs the anomalies, shrinking their reconstruction error to nothing. Keep too few and ordinary variation looks anomalous. Retaining enough components for 90-95% of variance is a reasonable default, but the right number depends on where the noise floor sits.

**PCA assumes linearity.** It finds a linear subspace, so data lying on a curved manifold will show large reconstruction error everywhere, and the method will report that most of the dataset is anomalous. Kernel PCA or an autoencoder handles curvature; the autoencoder is the direct non-linear generalisation of exactly this reconstruction-error idea.

**Outliers corrupt the fit.** In the anomaly-detection setting the extreme points influence the covariance matrix that defines the subspace, masking themselves. Robust PCA, or fitting on a trimmed subset and scoring everything against it, mitigates this.

## Setting a Threshold

Distributional thresholds derived from the assumption of multivariate normality exist for both statistics, but production data is rarely multivariate normal and those limits tend to be badly calibrated.

An empirical percentile of the training distribution is more defensible, with the caveat that it guarantees you flag that percentage of points whether or not anything is wrong. If the practical question is "how many cases can we investigate per day", setting the threshold from that capacity is more honest than pretending it came from theory.

Where labels exist for even a small sample, use them: a precision-recall curve over the score answers whether the detector is useful far better than any distributional argument.

## Categorical Data

PCA operates on covariances and therefore expects continuous input. Applying it to one-hot encoded categories technically runs but yields components driven largely by category frequency.

Better options exist: Multiple Correspondence Analysis is the categorical analogue of PCA, and for mixed data Factor Analysis of Mixed Data handles both types coherently. For purely categorical outlier detection, frequency-based methods such as the Frequent Patterns Outlier Factor address the problem directly rather than forcing it into a continuous frame.

## Where PCA Fits Among the Alternatives

PCA-based detection is fast, interpretable — the loadings say *which* variables drive an anomaly — and well suited to correlated numeric data, which is why it is standard in industrial process monitoring.

It is a poor fit when relationships are strongly non-linear, when features are mostly categorical, or when anomalies are defined by local density rather than global structure. Isolation Forest handles high-dimensional numeric data without a linearity assumption; Local Outlier Factor finds points anomalous relative to their neighbourhood rather than to the whole dataset; autoencoders extend the reconstruction idea to non-linear manifolds.

The most useful property PCA retains over all of these is explanation. When a point is flagged, the reconstruction residual decomposes across the original variables, so you can say which measurements are inconsistent — and in an operational setting, that is usually the difference between an alert someone acts on and one they ignore.

## References

- Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065).
- Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341-349.
- Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.
- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), 1-37.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *Proceedings of SIGMOD*, 93-104.
