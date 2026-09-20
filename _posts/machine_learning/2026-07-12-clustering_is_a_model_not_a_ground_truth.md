---
permalink: '/machine-learning/clustering_is_a_model_not_a_ground_truth/'
title: 'Clustering Is a Model of Similarity, Not a Discovery of Ground Truth'
categories:
- Machine Learning
- Statistics
tags:
- Unsupervised Learning
- Clustering
- Cluster Validation
- Statistical Learning
- Model Selection
author_profile: false
seo_title: 'Why Clustering Does Not Reveal a Unique Ground Truth'
seo_description: 'Clustering does not recover a unique partition from data alone. The representation, distance, objective, number of groups and stability criterion all define what a cluster means.'
excerpt: >-
  A clustering algorithm always answers a question, but the question is partly
  specified by us. Changing scale, distance, representation or objective can change
  the partition without changing the observations. Treating that partition as a
  discovered ground truth confuses a modelling choice with an empirical fact.
summary: >-
  A mathematical examination of why clustering is an ill-posed unsupervised
  learning problem. A four-point example shows that a simple change of scale can
  reverse the optimal K-means partition. The article then separates geometry,
  objective functions, cluster number, stability and external validation, and
  develops a practical protocol for interpreting clusters as model-dependent
  scientific objects rather than labels waiting to be recovered.
keywords:
- unsupervised learning
- clustering
- cluster validation
- k-means
- clustering stability
- similarity metrics
- clusterability
classes: wide
date: '2026-07-12'
why_this_exists: >-
  Clustering results are often described as though the data contained one natural
  partition and the algorithm merely revealed it. That interpretation is stronger
  than what unsupervised learning usually supports. The article makes the hidden
  choices explicit and shows, with a minimal calculation, how representation alone
  can change the optimum.
evidence: >-
  A worked Euclidean example, the objective functions behind several major
  clustering families, Kleinberg's clustering impossibility result, work on
  clustering stability, and methodological literature on the context-dependent
  meaning of a cluster.
methodology: >-
  Treat a clustering as the output of a modelling pipeline rather than a property
  of the raw sample. Vary representation, metric and objective while holding the
  observations fixed, then distinguish internal fit, perturbation stability,
  external agreement and scientific usefulness.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
---

When a clustering algorithm returns three groups, it is tempting to say that the data contain three clusters.

That statement is usually too strong.

The observations certainly contain structure. They may contain separated modes, elongated clouds, density ridges, manifolds, connected components, gradients, subpopulations, repeated trajectories, or no useful grouping at all. A clustering algorithm converts some of that structure into a partition. The partition, however, depends on choices that are not supplied by the observations alone.

A representation has to be chosen. A notion of similarity has to be defined. Some methods require the number of groups in advance, while others require a scale or density threshold. An objective function then decides which differences matter.

For that reason, the output of clustering is better written as

$$
\Pi
=
\mathcal{A}
\left(
X;
\phi,
d,
L,
\lambda
\right),
$$

where

- $X$ is the observed data,
- $\phi$ is the representation or feature map,
- $d$ is a distance or similarity rule,
- $L$ is the criterion being optimized,
- $\lambda$ collects tuning choices,
- and $\Pi$ is the resulting partition.

The notation is useful because it removes an ambiguity from the phrase "the clusters in the data." The data are only one argument of the procedure.

Change the other arguments and the partition can change as well.

## A Four-Point Example Is Enough

Consider four observations in two dimensions,

$$
A=(0,0),\qquad
B=(0,1),\qquad
C=(4,0),\qquad
D=(4,1).
$$

Suppose we fit K-means with $k=2$ using ordinary squared Euclidean distance.

There are two visually natural partitions worth comparing.

The first separates the two vertical pairs,

$$
\Pi_V
=
\{\{A,B\},\{C,D\}\}.
$$

For the left pair, the centroid is $(0,0.5)$. The two squared distances to that centroid are both $0.25$. The right pair contributes the same amount. The total within-cluster sum of squares is therefore

$$
J(\Pi_V)=1.
$$

Now consider the horizontal partition,

$$
\Pi_H
=
\{\{A,C\},\{B,D\}\}.
$$

Each centroid lies halfway between $x=0$ and $x=4$, so the total distortion is

$$
J(\Pi_H)=16.
$$

K-means strongly prefers the vertical partition.

Now change no observations in any substantive sense. Merely express the second coordinate in units ten times larger:

$$
\phi(x_1,x_2)
=
(x_1,10x_2).
$$

The transformed observations become

$$
A'=(0,0),\qquad
B'=(0,10),\qquad
C'=(4,0),\qquad
D'=(4,10).
$$

Under the same K-means objective, the vertical partition now has distortion

$$
J(\Pi_V')=100,
$$

while the horizontal partition still has

$$
J(\Pi_H')=16.
$$

The optimal partition reverses.

Nothing about the identity of the four observations changed. No new information arrived. We changed the geometry by changing the coordinate scale.

This is not a defect specific to K-means. It is the direct consequence of asking an algorithm to define groups through a geometry. If one coordinate counts metres and another counts millimetres, Euclidean distance treats their numerical scales as scientific information unless we intervene.

Standardisation does not remove the modelling decision. It replaces one geometry with another.

## Representation Comes Before Clustering

Let the original observation be $x\in\mathbb{R}^p$, and suppose a preprocessing step maps it to

$$
z=\phi(x).
$$

Clustering takes place in the geometry of $z$, not necessarily in the geometry of the original object.

That feature map might include standardisation, logarithms, principal components, embeddings, selected variables, learned representations, time-series summaries, graph construction, or domain-specific distances.

Even a linear transformation can alter pairwise Euclidean distances. If

$$
z=Ax,
$$

then

$$
\|z_i-z_j\|_2^2
=
(x_i-x_j)^T A^T A (x_i-x_j).
$$

Unless $A^TA$ is proportional to the identity, the transformation weights directions differently.

So a statement such as

> these observations are close

is incomplete until "close" has been defined.

In supervised learning, labels can help us evaluate whether a representation preserves information relevant to prediction. In unsupervised learning there may be no external target to perform that correction. The representation therefore carries more inferential weight than it is often given.

## Different Algorithms Formalise Different Meanings of a Cluster

K-means defines clusters through squared distance to centroids. Its objective is

$$
\min_{z_1,\ldots,z_n,\mu_1,\ldots,\mu_k}
\sum_{i=1}^{n}
\left\|
x_i-\mu_{z_i}
\right\|_2^2.
$$

A compact, approximately spherical group is therefore cheap. An elongated or curved group can be expensive even when a human observer regards it as one coherent structure.

A Gaussian mixture model asks a different question. It represents the density as

$$
p(x)
=
\sum_{g=1}^{G}
\pi_g
\mathcal{N}(x\mid\mu_g,\Sigma_g),
$$

and estimates parameters that make the observations probable under the mixture. Its groups are components of a probabilistic model. Allowing a full covariance matrix permits anisotropic clusters that K-means represents poorly.

Single-linkage hierarchical clustering uses yet another notion. Two groups can merge because one pair of observations is sufficiently close. This makes connectivity central and can recover shapes that centroid methods cannot, but it also produces the well-known chaining behaviour.

Density-based methods such as DBSCAN define clusters through dense connected regions separated by areas of lower density. Spectral methods transform the problem into graph geometry and look for partitions that are natural under a chosen affinity matrix and graph objective.

These methods can disagree even when each is implemented perfectly.

The disagreement is not necessarily evidence that all but one algorithm failed. They may be solving different mathematical problems.

## The Number of Clusters Is Usually Not an Observed Quantity

For K-means, the value of $k$ is supplied by the analyst.

That fact is easy to obscure after a plot has been produced. Once four colours appear on a figure, the four groups look as though they were properties of the sample. Yet the algorithm was instructed to produce four groups.

Internal selection criteria do not eliminate the issue. The silhouette coefficient, gap statistic, information criteria, stability scores and related procedures formalise additional preferences about separation, compactness, likelihood or reproducibility.

They can be useful, but they answer model-selection questions such as

$$
\text{Which candidate partition scores best under this criterion?}
$$

That is different from proving

$$
\text{The population contains exactly }k\text{ objectively real groups.}
$$

A continuous population can often be partitioned at several resolutions. A biological population may admit a coarse taxonomy for one purpose and a finer taxonomy for another. Customer behaviour can vary continuously while still supporting operational segments. Disease phenotypes may overlap rather than form disjoint classes.

The useful number of groups can therefore depend on the scientific or operational question.

## An Algorithm Will Usually Produce Clusters Even When There Are None

This is one of the most important asymmetries in unsupervised learning.

A classifier cannot be evaluated without some concept of a target. A clustering algorithm, by contrast, can return an apparently clean partition for data generated from a single continuous distribution.

If K-means is asked for

$$
k=5,
$$

it will seek five centroids.

The existence of five output labels is not evidence that five populations generated the observations.

This is why clusterability deserves attention before interpretation. One should ask whether the data contain structure strong enough to support the kind of grouping the method is designed to detect.

For some applications, multimodality matters. For others, separation matters. For still others, stable connectivity, low-density boundaries or repeated trajectory shapes are the relevant structure.

There is no universal definition of "clusterable" because there is no universal definition of a cluster.

## A Formal Impossibility Result Explains Part of the Difficulty

Kleinberg's impossibility theorem makes the problem precise.

Consider a clustering function defined from pairwise distances. Three properties can sound individually reasonable:

1. **Scale invariance:** multiplying every distance by the same positive constant should not change the clustering.
2. **Richness:** every possible partition should be obtainable for some distance function.
3. **Consistency:** if distances within clusters shrink and distances between clusters expand, the clustering should not change.

Kleinberg showed that no clustering function can satisfy all three simultaneously.

The result does not say that clustering is impossible. It says that there is no universal procedure satisfying this particular set of natural requirements at once.

That matters because it turns a vague methodological discomfort into a mathematical statement. Some trade-off has to enter the definition of a clustering method.

An algorithm therefore encodes a position about which properties should be preserved and which may be sacrificed.

## Stability Helps, but Stability Is Not Truth

A common response to ambiguity is to prefer a clustering that is stable under resampling or perturbation.

The idea is reasonable. If a small change in the sample completely reorganises the partition, then strong scientific interpretation is difficult to defend.

One may generate bootstrap samples, perturb the measurements, vary initialisation, or slightly alter tuning parameters. For observations that can be aligned across runs, a useful object is the co-clustering probability

$$
P_{ij}
=
\Pr
\left(
z_i=z_j
\right).
$$

If $P_{ij}$ is close to one, observations $i$ and $j$ are repeatedly assigned together. Values near one half indicate ambiguity.

The matrix

$$
P=(P_{ij})
$$

contains considerably more information than one final label vector.

However, stability is not sufficient for truth.

A procedure can stably recover a partition implied by an arbitrary representation. A very coarse partition may be stable because it ignores meaningful smaller-scale structure. Some algorithms become more stable simply because their regularisation or constraints make them less responsive to data variation.

The theoretical literature on clustering stability makes the same general point: stability has to be interpreted relative to the data-generating process and the clustering method. It is evidence about reproducibility of a solution, not a certificate that the solution is the unique scientific description of the population.

## Internal Validation Can Become Circular

Suppose an algorithm is built around Euclidean compactness and we evaluate it with a metric that rewards Euclidean compactness.

A high score tells us that the algorithm produced what the criterion prefers.

That can be useful for model selection, but it is not independent evidence that the clusters are scientifically meaningful.

The circularity becomes clearer if we write

$$
\widehat\Pi
=
\arg\min_{\Pi} L(\Pi;X)
$$

and then assess the result mainly with another quantity that is strongly aligned with $L$.

We have learned that the optimiser found a good solution under related geometry. We have not yet learned whether the grouping corresponds to a mechanism, population, process or decision problem outside that geometry.

Internal metrics are therefore best treated as diagnostics, not as ontological tests.

## External Labels Do Not Automatically Solve the Problem

If known labels exist, it is common to compare the clustering with them using adjusted Rand index, mutual information, Fowlkes-Mallows score or a related agreement measure.

This can be informative, but only if the external labels represent the grouping concept we care about.

A clustering of patients based on treatment response need not reproduce diagnostic categories. A clustering of customers based on purchasing dynamics need not reproduce demographic labels. A grouping of time series by shape need not match business units.

In such cases, poor agreement with an external label can coexist with a useful clustering, because the two partitions encode different questions.

Conversely, high agreement can be misleading if the representation already contains variables that are near-proxies for the labels.

External validation is strongest when the external variable has an explicit scientific role that was not used to manufacture the grouping.

## The Scientific Object Is the Cluster Concept

Christian Hennig has argued that discussions of "true clusters" need to make the underlying cluster concept explicit. This is a productive way to formulate the problem.

Instead of asking

> Which clustering algorithm finds the real groups?

ask

> What properties should observations share for us to call them members of the same group?

Possible answers include:

- similar location in a scientifically justified metric,
- membership in the same high-density region,
- common parameters in a generative mixture,
- similar temporal dynamics,
- common graph connectivity,
- similar response to an intervention,
- or equivalent downstream decisions.

Once that concept is explicit, algorithm choice becomes more disciplined because the mathematical criterion can be evaluated against the intended meaning.

Without that step, changing algorithms can amount to changing the scientific question without acknowledging it.

## A More Defensible Clustering Workflow

For serious exploratory work, I would separate the analysis into several layers.

### 1. Define the object being grouped

Are the objects patients, trajectories, distributions, locations, documents, graphs, repeated measurements or parameter vectors?

The answer determines which transformations preserve scientifically meaningful information.

### 2. Define similarity before choosing the algorithm

Write down why two objects should count as similar.

For longitudinal data, Euclidean distance on concatenated measurements may be inappropriate if phase shifts, irregular sampling or temporal derivatives matter. For compositional data, ordinary Euclidean geometry ignores the simplex structure. For text embeddings, cosine similarity may express the intended geometry better than raw Euclidean distance.

### 3. Compare plausible representations

If the result disappears after a modest and scientifically defensible change of scaling or representation, interpretation should be correspondingly cautious.

A cluster that exists only under one arbitrary preprocessing pipeline is evidence about that pipeline before it is evidence about the population.

### 4. Compare algorithm families, not only initialisations

Running K-means one hundred times tests optimisation sensitivity within the K-means model.

It does not test whether centroid clustering was the right model.

A stronger analysis compares methods built on different cluster concepts: centroid, mixture, density, connectivity, graph or trajectory-based models, depending on the application.

### 5. Quantify assignment uncertainty

Hard labels such as

$$
z_i\in\{1,\ldots,k\}
$$

discard ambiguity.

When possible, retain posterior membership probabilities, co-clustering probabilities, bootstrap assignment frequencies, or sensitivity to tuning parameters.

An observation with membership probabilities

$$
(0.49,0.48,0.03)
$$

should not be communicated in the same way as one with

$$
(0.99,0.005,0.005).
$$

### 6. Look for external consequences

The strongest cluster interpretations usually connect the grouping to something outside the fitting criterion.

Do the groups differ in future outcomes, physical mechanisms, treatment response, process parameters, replicated measurements or another pre-specified quantity?

This does not turn an exploratory analysis into a causal one, but it tests whether the partition captures structure that survives outside its own construction.

### 7. Report alternatives that remain plausible

If two different clusterings fit the data similarly well and support different interpretations, that ambiguity is part of the result.

Suppressing it creates a false precision that unsupervised learning has not earned.

## Why This Matters for Semi-Supervised Learning

The same issue reappears in semi-supervised learning.

Unlabelled observations describe the geometry or density of the feature distribution. They become useful for classification only when that geometry is related to the label mechanism.

The cluster assumption is one common example: points in the same high-density region are assumed to tend to share labels, while decision boundaries are expected to pass through lower-density regions.

That assumption can be powerful.

It can also be wrong.

If the unsupervised geometry groups observations according to nuisance variation rather than label-relevant structure, then using more unlabelled data can reinforce the wrong representation. The previous article on negative transfer under covariate shift demonstrated one version of that failure.

This gives a useful sequence of questions:

$$
\text{What geometry does the unlabelled sample reveal?}
$$

then

$$
\text{Why should that geometry be related to the target labels?}
$$

and only then

$$
\text{Which semi-supervised algorithm should use it?}
$$

Semi-supervised learning does not eliminate the assumptions of unsupervised learning. It adds information that can sometimes identify which structures matter.

## Conclusion

Clustering is valuable precisely because labelled answers are not required in advance. That same freedom is what makes interpretation difficult.

A partition is not extracted from data independently of modelling choices. It is produced by a representation, a similarity concept, an objective function and a set of tuning assumptions.

The four-point example makes the issue visible in its simplest form:

$$
\boxed{
\text{change the geometry}
\quad\Longrightarrow\quad
\text{change the optimal clustering}
}
$$

That does not make clustering arbitrary.

It means that the scientific work begins before the algorithm is fitted and continues after the labels are returned.

The relevant question is rarely whether an algorithm has discovered *the* clusters.

A better question is whether the grouping is stable, interpretable and useful under a cluster concept that can be defended independently of the colours on the final plot.

## References

- Hennig, C. (2015). What are the true clusters? *Pattern Recognition Letters*, 64, 53–62. https://doi.org/10.1016/j.patrec.2015.04.009
- Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666. https://doi.org/10.1016/j.patrec.2009.09.011
- Kleinberg, J. M. (2002). An impossibility theorem for clustering. *Advances in Neural Information Processing Systems*, 15.
- von Luxburg, U. (2010). Clustering stability: An overview. *Foundations and Trends in Machine Learning*, 2(3), 235–274. https://doi.org/10.1561/2200000008
