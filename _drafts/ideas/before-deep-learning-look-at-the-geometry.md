---
author_profile: false
categories:
- Machine Learning
classes: wide
excerpt: Longitudinal clustering is often treated as an algorithm-selection problem. The harder question is what representation preserves the temporal structure that actually separates groups.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- longitudinal clustering
- time series clustering
- functional data
- representation learning
- Gaussian mixture models
- temporal features
seo_description: Why longitudinal clustering should start from the geometry of temporal variation rather than the choice of clustering algorithm.
seo_title: Before Deep Learning, Look at the Geometry
seo_type: article
summary: A representation-first view of longitudinal clustering, finite-sample feature estimation, functional data, and when complex sequence models are actually justified.
tags:
- Machine Learning
- Time Series
- Clustering
title: 'Before Deep Learning, Look at the Geometry'
---

Longitudinal clustering problems are often introduced as though the main decision were which algorithm to use.

Usually it is not.

The harder question is what object should be clustered in the first place.

A subject observed through time is not naturally a row in a rectangular table. It is a trajectory. Depending on the scientific problem, the meaningful variation may lie in level, trend, persistence, periodicity, volatility, change points, recovery time, event timing, or some combination of these.

If the representation is wrong, a sophisticated clustering algorithm can only organize the wrong geometry more efficiently.

That is the central argument of this article.

## Clustering is geometry plus an objective

A clustering algorithm does not act on the scientific object directly. It acts on a representation of that object together with some notion of similarity or probability.

Write the observed trajectory for subject $i$ as

$$
X_i=(X_i(t_1),\ldots,X_i(t_T)).
$$

Before clustering, we implicitly or explicitly choose a map

$$
\phi:X_i\mapsto z_i,
$$

where $z_i$ may be the raw trajectory, a vector of handcrafted temporal features, coefficients in a basis expansion, latent-state summaries, or a learned embedding.

We then cluster the $z_i$ values under some geometry or probabilistic model.

So the actual modelling pipeline is

$$
\boxed{
\text{trajectory}
\rightarrow
\text{representation}
\rightarrow
\text{geometry/model}
\rightarrow
\text{clusters}
}
$$

The representation determines which differences are visible and which are suppressed.

A Euclidean distance on raw trajectories emphasizes pointwise discrepancies. A distance between spectral summaries emphasizes periodic structure. A Gaussian mixture on estimated autocorrelations emphasizes dependence features. A functional principal-component representation emphasizes dominant modes of trajectory variation.

There is no representation-free clustering problem.

## Start with the scientific distinction

Suppose two latent groups differ primarily in temporal persistence rather than mean level.

Then clustering raw observations can fail even if the groups are genuinely distinct. The information may live in a summary such as an autocorrelation, spectral statistic, transition rate, dwell time, or another feature that reflects temporal dependence.

The modelling sequence should therefore be

$$
\text{scientific distinction}
\rightarrow
\text{temporal representation}
\rightarrow
\text{clustering method}.
$$

Not the other way around.

This sounds obvious when written down, but many workflows begin with the reverse sequence: choose k-means, hierarchical clustering, a Gaussian mixture, an autoencoder, or a transformer, and only later ask what temporal distinction the representation preserves.

That is backwards.

## A useful thought experiment

Consider two stationary processes with the same marginal mean and variance but different autocorrelation structures.

For instance, imagine two zero-mean AR(1) processes,

$$
X_t^{(g)}=\rho_g X_{t-1}^{(g)}+\varepsilon_t,
\qquad g\in\{1,2\},
$$

with

$$
\rho_1\neq\rho_2,
$$

but with innovation variances chosen so that both groups have the same marginal variance.

If ordering is ignored and observations are treated as exchangeable samples, much of the class information disappears. The one-dimensional marginal distributions can be nearly indistinguishable even though the temporal dynamics differ.

A persistence feature such as lag-one autocorrelation,

$$
\widehat\rho_1
=
\frac{\sum_{t=2}^{T}(X_t-\bar X)(X_{t-1}-\bar X)}
{\sum_{t=1}^{T}(X_t-\bar X)^2},
$$

can recover part of the distinction because it represents the mechanism by which the trajectories differ.

The point is not that autocorrelation is universally sufficient. It is that the representation should encode the structure the science says matters.

## Population geometry is not observed geometry

Even when a temporal feature is theoretically discriminative, we usually do not observe its population value.

Let

$$
z_i^*=\phi(P_i)
$$

be the ideal population-level representation associated with subject $i$'s data-generating process, and let

$$
\widehat z_i=\phi_T(X_i)
$$

be the feature estimated from a finite trajectory of length $T$.

Then

$$
\widehat z_i=z_i^*+\eta_i,
$$

where $\eta_i$ is feature-estimation error.

The clustering algorithm therefore sees a noisy cloud of estimated features rather than the ideal population geometry.

This gives a useful decomposition of failure:

$$
\boxed{
\text{weak population separation}
+
\text{representation error}
+
\text{clustering-estimation error}
}
$$

These mechanisms are different and suggest different remedies.

If population feature separation is weak, no clustering method can manufacture information that is not there.

If feature estimation is noisy because trajectories are short, the remedy may be longer follow-up, shrinkage, pooling, or a different representation.

If the representation is informative but the fitted mixture is unstable, then the problem lies in the clustering layer.

Without this decomposition, it is very easy to blame the wrong component.

## Short trajectories can destroy a good population feature

Suppose the latent groups differ cleanly in a population feature,

$$
Z^*\mid G=1\sim F_1,
\qquad
Z^*\mid G=2\sim F_2,
$$

with clear separation between $F_1$ and $F_2$.

In practice we observe

$$
\widehat Z=Z^*+\eta_T,
$$

where the variance of $\eta_T$ decreases as trajectory length grows.

Then the observed class distributions are effectively convolutions,

$$
F_1 * H_T,
\qquad
F_2 * H_T,
$$

where $H_T$ is the estimation-error distribution.

Even if the population distributions are well separated, short trajectories can blur them enough to make the clustering problem genuinely difficult.

This is a basic but important point: **more subjects and longer trajectories solve different problems**.

Increasing the number of subjects improves estimation of the mixture or cluster structure. Increasing trajectory length improves estimation of subject-level temporal features.

Those two sample-size dimensions should not be collapsed into one generic notion of “more data.”

## Functional data provide another representation

When trajectories are dense enough to be treated as functions, a natural alternative is to represent each subject by a smooth function or basis expansion,

$$
X_i(t)
\approx
\sum_{k=1}^{K} c_{ik}\psi_k(t).
$$

The coefficients $c_{ik}$ then define a low-dimensional geometry for clustering.

Functional principal components go one step further by choosing basis directions that explain dominant modes of variation across curves.

This can work very well when the relevant distinctions are smooth shape differences.

But functional representations also encode assumptions. Smoothing can suppress abrupt change points. A low-rank basis can erase local events. Registration can remove timing variation that may itself be scientifically meaningful.

So “functional data analysis” is not a neutral preprocessing step. It is another choice of geometry.

## Irregular and sparse observation change the problem

Longitudinal data are often irregularly sampled.

Two subjects may have different numbers of observations, different visit times, and long gaps in different parts of the trajectory.

In that setting, computing the same naive summary for every subject can create incomparable feature noise.

For subject $i$, we may observe

$$
\{(t_{ij},X_i(t_{ij}))\}_{j=1}^{n_i},
$$

with $n_i$ varying substantially between individuals.

Then the precision of $\widehat z_i$ is itself subject-specific.

That suggests several possibilities:

- model the trajectories jointly rather than estimating every feature independently;
- use functional methods designed for sparse observation;
- carry feature-estimation uncertainty into the clustering layer;
- restrict clustering to features that are estimable at the available observation density;
- perform sensitivity analyses across alternative representations.

What should not happen is pretending that irregularly measured trajectories form a clean rectangular matrix merely because software prefers one.

## Distances also encode scientific assumptions

Feature-based clustering is only one route.

Another is to define a distance directly between trajectories,

$$
d(X_i,X_j),
$$

and use hierarchical clustering, k-medoids, spectral clustering, or another distance-based method.

But the distance is itself a representation choice.

Euclidean distance treats time points as aligned and penalizes vertical discrepancies. Dynamic time warping permits local temporal deformation. Correlation-based distances emphasize shape after removing level and scale. Model-based distances compare fitted stochastic mechanisms.

Each answers a different scientific question.

For example, if two patients have the same physiological response but one responds two hours later, should they be considered similar?

There is no algorithmic answer. The answer depends on whether timing is nuisance variation or part of the phenotype being clustered.

## Baselines should be difficult to beat for the right reason

Useful baselines include:

- hand-designed temporal summaries with Gaussian mixtures;
- standardized trajectory distances with hierarchical clustering;
- spline or functional principal-component scores with k-means or mixtures;
- simple hidden-state or state-space representations when discrete regimes have a scientific interpretation.

These are not straw men.

They answer an important question:

$$
\boxed{
\text{How much of the problem is already solved by choosing the right representation?}
}
$$

If a simple model performs well after representation is fixed, that is useful information.

It means the main difficulty may have been representation rather than clustering capacity.

## Cluster number is not an afterthought

Another common mistake is to treat the number of clusters $K$ as a tuning parameter detached from the scientific problem.

Different values of $K$ can describe different levels of heterogeneity rather than competing estimates of one uniquely correct partition.

A two-cluster solution may separate broad temporal phenotypes, while a four-cluster solution may split each phenotype by severity or timing.

Internal metrics such as silhouette width or likelihood criteria can help, but they do not make the scientific interpretation disappear.

A useful analysis should therefore ask whether the recovered groups are:

1. stable under resampling or perturbation;
2. reproducible across reasonable representations;
3. interpretable in terms of the temporal mechanisms of interest;
4. robust to plausible choices of $K$.

The goal is not to discover the metaphysically true number of clusters. It is to determine whether a useful and stable structure is supported by the data.

## Deep learning is sometimes justified

Learned sequence representations can be appropriate when the data are large, multivariate, irregular, strongly nonlinear, or contain long-range dependencies that simpler features fail to capture.

A learned encoder can be written abstractly as

$$
z_i=f_\theta(X_i),
$$

where the representation itself is fitted from data.

That flexibility can be valuable, but it changes the statistical problem.

Now there are at least four sources of uncertainty:

$$
\text{population separation}
+
\text{finite-trajectory noise}
+
\text{representation-learning error}
+
\text{clustering error}.
$$

If the learned embedding is unstable across random initialization, sample perturbation, or architecture choices, apparent cluster structure may be a property of the representation learner rather than the underlying trajectories.

So sequence data alone are not an argument for a recurrent network or transformer.

The relevant comparison is

$$
\boxed{
\text{Does the flexible representation capture stable information that the simpler one genuinely misses?}
}
$$

That is a testable question.

## A stronger evaluation design

For unsupervised longitudinal problems, there is rarely one perfect metric.

I would separate evaluation into several layers.

### 1. Representation diagnostics

Check whether the chosen representation preserves the temporal distinctions that motivated the analysis.

### 2. Cluster stability

Repeat the analysis under bootstrap samples, perturbed initializations, and reasonable hyperparameter changes.

### 3. Representation sensitivity

Compare several defensible geometries rather than only several algorithms on one geometry.

### 4. External validation

When external variables exist, ask whether clusters differ on outcomes not used to construct them.

### 5. Synthetic falsification

Use simulated data where the population-generating mechanism is known. This can isolate whether failure comes from short trajectories, feature estimation, or the clustering method itself.

That final step is especially useful because real longitudinal data rarely reveal the true partition.

## The practical workflow

For a new longitudinal clustering problem, I would usually begin by asking:

1. What scientific distinction should separate the groups?
2. Is timing itself informative, or should trajectories be aligned?
3. Which temporal summaries, basis coefficients, state representations, or distances encode that distinction?
4. How noisy are those representations at the available trajectory length and sampling pattern?
5. What cluster structure is visible under a transparent baseline?
6. How stable is that structure under resampling and alternative reasonable representations?
7. Where exactly does a more flexible learned representation improve on that baseline?

The core principle is simple:

$$
\boxed{
\text{Before increasing clustering complexity, make sure the geometry is the one you intend to study.}
}
$$

## References

- Aghabozorgi S, Shirkhorshidi AS, Wah TY. *Time-series clustering – A decade review*. Information Systems. 2015;53:16–38. DOI: [10.1016/j.is.2015.04.007](https://doi.org/10.1016/j.is.2015.04.007).
- Liao TW. *Clustering of time series data—a survey*. Pattern Recognition. 2005;38(11):1857–1874. DOI: [10.1016/j.patcog.2005.01.025](https://doi.org/10.1016/j.patcog.2005.01.025).
- Wang X, Smith K, Hyndman R. *Characteristic-Based Clustering for Time Series Data*. Data Mining and Knowledge Discovery. 2006;13:335–364. DOI: [10.1007/s10618-005-0039-x](https://doi.org/10.1007/s10618-005-0039-x).
- James GM, Sugar CA. *Clustering for Sparsely Sampled Functional Data*. Journal of the American Statistical Association. 2003;98(462):397–408. DOI: [10.1198/016214503000189](https://doi.org/10.1198/016214503000189).
- Jacques J, Preda C. *Functional data clustering: a survey*. Advances in Data Analysis and Classification. 2014;8:231–255. DOI: [10.1007/s11634-013-0158-y](https://doi.org/10.1007/s11634-013-0158-y).
