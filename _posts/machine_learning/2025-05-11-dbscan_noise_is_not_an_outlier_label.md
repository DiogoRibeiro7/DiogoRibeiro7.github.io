---
permalink: '/machine-learning/dbscan_noise_is_not_an_outlier_label/'
title: 'DBSCAN Noise Is Not an Outlier Label'
categories:
- Machine Learning
- Statistics
tags:
- Unsupervised Learning
- DBSCAN
- Density-Based Clustering
- Outlier Detection
- Cluster Validation
- Statistical Learning
author_profile: false
seo_title: 'Why DBSCAN Noise Is Scale-Dependent, Not Ground Truth'
seo_description: 'DBSCAN labels points as noise when they fail a density condition at a chosen scale. The result depends on epsilon, min_samples, dimension, metric and feature scaling.'
excerpt: >-
  DBSCAN is often used as though its noise label were an anomaly detector. It is
  not. A point is marked as noise because it fails a density-connectivity rule at a
  chosen geometric scale. Change epsilon, min_samples, the metric or the feature
  scaling and the same observation can move from noise to border point to core
  point without the data changing.
summary: >-
  A mathematical analysis of DBSCAN's core, border and noise labels. The article
  derives the approximate local density threshold implied by epsilon and
  min_samples, shows its epsilon^{-d} dependence, explains feature-scaling and
  dimensional effects, and separates density-based clustering noise from
  statistical anomaly detection.
keywords:
- DBSCAN noise
- DBSCAN epsilon
- min_samples
- density-based clustering
- outlier detection
- unsupervised learning
- local density
classes: wide
date: '2025-05-11'
why_this_exists: >-
  DBSCAN's noise points are frequently described as outliers or anomalies. That
  language is stronger than the algorithm supports. Noise is defined relative to a
  neighbourhood radius and minimum local count. The same point can change status
  under another scientifically defensible scale.
evidence: >-
  The original DBSCAN density-connectivity definitions, local count asymptotics,
  exact geometric scaling arguments and density-based clustering literature.
methodology: >-
  Rewrite the core-point rule as an approximate local density threshold, inspect
  how that threshold changes with epsilon, dimension and metric scaling, and
  distinguish parameter sensitivity from substantive anomaly evidence.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/constellation.jpg
  og_image: /assets/images/headers/constellation.jpg
  overlay_image: /assets/images/headers/constellation.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/constellation.jpg
  twitter_image: /assets/images/headers/constellation.jpg
---

DBSCAN has one feature that makes it especially attractive in exploratory analysis.

It can return observations with cluster label

$$
-1.
$$

Many libraries call those observations noise.

It is tempting to translate that immediately into

> these points are outliers.

That is not what DBSCAN proves.

A DBSCAN noise point is an observation that does not satisfy the algorithm's density-connectivity rules under the chosen metric, neighbourhood radius and minimum local count.

Change those choices and the same observation can become:

- a core point,
- a border point,
- or a noise point.

The data did not change.

The scale of the density question did.

## The Two Parameters Define a Local Density Requirement

Let

$$
X_1,\ldots,X_n
$$

be observations in a metric space with distance $d$.

For a point $x$, define its epsilon neighbourhood

$$
N_\varepsilon(x)
=
\left\{
x_j:
d(x,x_j)\leq\varepsilon
\right\}.
$$

A point is a DBSCAN core point when

$$
|N_\varepsilon(x)|
\geq m,
$$

where $m$ denotes the minimum number of observations required by the implementation.

Different libraries differ slightly in whether the point itself is counted.

That convention matters for exact counts but not for the argument here.

The essential rule is

$$
\boxed{
\text{core point}
\Longleftrightarrow
\text{enough observations fall inside an epsilon ball}.
}
$$

This is a local density criterion.

It is also explicitly scale-dependent.

## From Neighbour Counts to Density

Suppose the data are drawn from a smooth density

$$
f(x)
$$

in

$$
\mathbb R^d.
$$

For small $\varepsilon$, the probability mass inside an epsilon ball around $x$ is approximately

$$
P
\left(
X\in B_\varepsilon(x)
\right)
\approx
f(x)
V_d
\varepsilon^d,
$$

where $V_d$ is the volume of the unit ball in dimension $d$.

Therefore the expected number of sample observations inside the neighbourhood is approximately

$$
E
\left[
|N_\varepsilon(x)|
\right]
\approx
n
f(x)
V_d
\varepsilon^d.
$$

Ignoring the self-count convention, the core condition

$$
|N_\varepsilon(x)|
\geq m
$$

corresponds roughly to

$$
n
f(x)
V_d
\varepsilon^d
\gtrsim
m.
$$

Solving for density gives

$$
\boxed{
f(x)
\gtrsim
\frac{
m
}{
nV_d\varepsilon^d
}.
}
$$

This is the most useful way to understand the parameter pair.

DBSCAN is implicitly asking whether local density exceeds a threshold determined by

$$
m,
\quad
n,
\quad
d,
\quad
\varepsilon,
$$

and the metric geometry.

## epsilon Is a Density Threshold in Disguise

Define the approximate threshold

$$
\tau(\varepsilon,m)
=
\frac{
m
}{
nV_d\varepsilon^d
}.
$$

A smaller epsilon means a smaller neighbourhood volume.

Therefore a much higher density is required to collect the same number of observations.

The dependence is

$$
\tau
\propto
\varepsilon^{-d}.
$$

That exponent is important.

In one dimension,

$$
\tau
\propto
\frac1\varepsilon.
$$

In two dimensions,

$$
\tau
\propto
\frac1{\varepsilon^2}.
$$

In ten dimensions,

$$
\tau
\propto
\frac1{\varepsilon^{10}}.
$$

A modest change in epsilon can therefore represent a very large change in the effective density threshold.

## Doubling epsilon Has a Dimension-Dependent Effect

Suppose epsilon doubles.

Then

$$
\tau(2\varepsilon,m)
=
\frac{
m
}{
nV_d(2\varepsilon)^d
}
=
2^{-d}
\tau(\varepsilon,m).
$$

So the required density falls by a factor

$$
2^d.
$$

For $d=2$, the threshold falls by four.

For $d=5$, it falls by 32.

For $d=10$, it falls by 1024.

This is one reason high-dimensional DBSCAN can be extremely sensitive to neighbourhood scale.

The parameter is not a minor tuning knob.

It defines what counts as locally dense.

## min_samples Changes the Threshold Linearly

Holding everything else fixed,

$$
\tau
\propto
m.
$$

Doubling the minimum local count doubles the approximate density required for a core point.

So increasing min_samples does not simply make DBSCAN "more conservative."

It changes the density level at which the cluster structure is defined.

The pair

$$
(\varepsilon,m)
$$

should therefore be interpreted jointly.

A value of epsilon has no substantive meaning without the required count.

## Core, Border and Noise Are Relational Labels

DBSCAN does not classify every non-core point as noise.

A point can be a border point.

Roughly:

- a core point has enough neighbours,
- a border point does not have enough neighbours itself but lies inside the epsilon neighbourhood of a core point,
- a noise point is neither core nor density-reachable from the cluster expansion.

So the label of one point depends on nearby points.

An observation can fail the local count criterion and still be attached to a cluster because a neighbouring core point connects it.

Noise is therefore not a purely pointwise anomaly score.

It is a graph-like relational outcome.

## One Extra Observation Can Change Another Point's Status

Suppose a point has exactly

$$
m-1
$$

observations in its epsilon neighbourhood.

It is not a core point.

Add one new observation inside the neighbourhood.

Now the count becomes

$$
m.
$$

The point becomes core.

That change can trigger further density reachability and merge previously separate regions.

The new observation may itself be ordinary.

Yet it changes the status of other observations.

This is another reason the noise label should not be interpreted as an intrinsic property of an individual point.

## Sample Size Changes the Same Density Rule

The approximate threshold is

$$
\tau
=
\frac{
m
}{
nV_d\varepsilon^d
}.
$$

If epsilon and min_samples are kept fixed while sample size increases, then

$$
\tau
$$

falls like

$$
1/n.
$$

So a denser sample from the same population can turn previously non-core regions into core regions.

The population did not become more clustered.

The finite sample became denser.

This is not a bug.

DBSCAN was originally designed as a sample-based density-connectivity procedure.

But it means that parameter values are not automatically portable across sample sizes.

## A Uniform Distribution Can Produce Noise

Consider

$$
X\sim\operatorname{Unif}(0,1).
$$

There is no anomalous subpopulation.

Every interior location has the same population density.

Yet with finite data and sufficiently small epsilon, many observations will fail the minimum-neighbour condition.

They can be labelled noise.

That happens because local sample counts fluctuate.

Near the boundaries at zero and one, the effect is even stronger because the epsilon ball is truncated by the support.

So DBSCAN noise can arise from:

- finite-sample variation,
- boundary geometry,
- sparse sampling,
- or parameter choice,

without any anomalous generating mechanism.

## Boundary Points Are Structurally Disadvantaged

For an interior point in one dimension, an epsilon neighbourhood has length approximately

$$
2\varepsilon.
$$

Near the boundary at zero, the available interval may be only

$$
\varepsilon.
$$

So under a uniform population, expected neighbour count near the edge is roughly half that of an interior point.

The point is not statistically unusual relative to the support.

It is geometrically close to a boundary.

DBSCAN can therefore produce more noise near support boundaries.

That should not be confused with anomaly evidence.

## A Point Can Move From Noise to Core Without Moving

Suppose all observations are held fixed.

Run DBSCAN with epsilon

$$
\varepsilon_1.
$$

A particular observation is noise.

Now increase epsilon to

$$
\varepsilon_2>\varepsilon_1.
$$

Its neighbourhood can only grow:

$$
N_{\varepsilon_1}(x)
\subseteq
N_{\varepsilon_2}(x).
$$

Therefore the neighbour count is non-decreasing.

The point may become border or core.

Its coordinates are identical.

The new label reflects a new density scale.

The observation did not become less anomalous in any absolute sense.

## Feature Scaling Changes epsilon Geometry

Suppose

$$
x=(x_1,x_2)
$$

and Euclidean distance is used.

The epsilon ball is

$$
(x_1-\mu_1)^2
+
(x_2-\mu_2)^2
\leq
\varepsilon^2.
$$

Now rescale the second coordinate:

$$
x_2^\star
=
c x_2.
$$

The same numeric epsilon corresponds in original coordinates to

$$
(x_1-\mu_1)^2
+
c^2
(x_2-\mu_2)^2
\leq
\varepsilon^2.
$$

This is an ellipse in the original coordinate system.

So standardization, unit conversion and weighting all change which observations count as neighbours.

DBSCAN's clusters and noise labels are conditional on that geometry.

## Unit Conversion Can Break an Unadjusted Analysis

Imagine one coordinate measured in metres.

Convert it to millimetres:

$$
x^\star=1000x.
$$

If epsilon is not transformed accordingly, pairwise distances change by orders of magnitude in that coordinate.

A point can move from dense to isolated numerically without any physical change.

This is an extreme example, but it exposes the principle.

A density threshold in metric space inherits the units of the metric.

Parameter values are not meaningful independently of units.

## Standardization Is a Model Choice, Not a Repair Button

A common recommendation is:

> standardize all variables before DBSCAN.

Sometimes that is appropriate.

Sometimes it is not.

Standardization replaces the original metric with one that weights coordinates by inverse sample standard deviation.

For diagonal scaling,

$$
z_j
=
\frac{x_j-\bar x_j}{s_j},
$$

Euclidean distance becomes

$$
d_z(x,x')^2
=
\sum_j
\frac{
(x_j-x'_j)^2
}{
s_j^2
}.
$$

Variables with larger empirical variance receive less weight.

That may be scientifically sensible.

It may also suppress a physically meaningful scale.

There is no universal reason the standard-deviation metric is the correct notion of neighbourhood.

## The Distance Metric Changes the Shape of Density

Under Euclidean distance, epsilon neighbourhoods are balls.

Under Manhattan distance, they are cross-polytopes.

Under Chebyshev distance, they are axis-aligned cubes.

Under Mahalanobis distance, they become ellipsoids adapted to covariance.

The local volume

$$
V_d\varepsilon^d
$$

changes with the metric.

So does the set of neighbours.

Therefore density is not a single scalar concept independent of geometry.

DBSCAN estimates connectivity under a chosen metric notion of local volume.

## "Arbitrary Shape" Does Not Mean "No Geometry Assumptions"

DBSCAN is often praised because it can recover non-spherical clusters.

That is true.

It does not imply that the algorithm is geometry-free.

Density connectivity is still defined through epsilon neighbourhoods.

The metric determines those neighbourhoods.

A curved cluster can be recovered if adjacent dense neighbourhoods create a chain.

But change the metric or epsilon and the chain can break.

"Arbitrary shape" means the cluster need not be representable by one centroid or covariance ellipse.

It does not mean every shape is equally recoverable.

## Density Reachability Can Create Chaining

Suppose dense local neighbourhoods form a sequence

$$
A_1,A_2,\ldots,A_r
$$

where each overlaps the next.

DBSCAN can connect them into one cluster through density reachability.

The endpoints can be far apart.

This is a feature when the population forms a curved or elongated dense set.

It can be a problem when a narrow bridge joins two otherwise separate dense regions.

A small amount of connecting mass can merge them.

So density connectivity is a topological modelling choice.

## A Thin Bridge Can Merge Two Populations

Imagine two dense clouds separated by a sparse corridor.

If the corridor contains enough points for a chain of core neighbourhoods, DBSCAN returns one cluster.

Reduce epsilon or raise min_samples and the bridge may break.

Now there are two clusters.

Neither answer is automatically wrong.

They correspond to different density levels.

The substantive question is which density level matches the phenomenon being studied.

## Variable-Density Data Are Difficult for One Global epsilon

Suppose one true population is very dense and another is much more diffuse.

A small epsilon may recover the dense cluster while labelling much of the diffuse cluster as noise.

A large epsilon may recover the diffuse cluster but merge the dense cluster with nearby structure.

This is a known limitation of DBSCAN.

The difficulty is not merely parameter tuning.

One global epsilon defines one global density scale.

The data may contain structure at several scales.

## HDBSCAN Changes the Question

Hierarchical DBSCAN-style methods address variable density by considering a hierarchy across scales rather than committing to one epsilon.

That can be useful.

It does not eliminate modelling choices.

Now the analysis involves notions such as cluster persistence or stability across density levels.

The question becomes:

> which density-connected structures persist over a meaningful range of scales?

That is richer than one DBSCAN run.

It is still not a model-free definition of "real clusters."

## Noise Is Not the Same as Low Probability

A statistical outlier is often understood as an observation with low probability under a reference model.

DBSCAN noise is different.

A point can lie in a low-density but perfectly legitimate region of a continuous population.

A point can also be unusual under a global model yet belong to a locally dense group and therefore be clustered by DBSCAN.

These notions can disagree.

For example, a small remote group can be globally rare but internally dense.

DBSCAN may treat it as a cluster rather than noise.

That may be exactly what we want.

It also shows why noise is not synonymous with anomaly.

## A Remote Dense Group Is the Clearest Counterexample

Suppose 98 percent of the data come from one large population.

The remaining 2 percent come from a compact, remote component.

If those 2 percent are dense enough relative to epsilon and min_samples, DBSCAN will form a separate cluster.

An anomaly detector based on rarity might call them unusual.

DBSCAN does not.

Conversely, isolated points around the edge of the majority distribution may be labelled noise even if they are plausible tail observations.

The two tasks ask different questions:

$$
\text{density-connected grouping}
$$

versus

$$
\text{anomaly relative to a reference distribution}.
$$

## DBSCAN Noise Has No Natural Ranking

The standard output assigns a common label

$$
-1
$$

to all noise points.

That does not rank them by anomaly severity.

One noise point may be just outside a cluster boundary.

Another may be extremely isolated.

The categorical label discards that distinction.

If anomaly detection is the goal, a continuous score is often more informative.

Examples include:

- local outlier factor,
- nearest-neighbour distance,
- reachability distance,
- density-ratio methods,
- isolation scores,
- model-based tail probability,
- or task-specific residuals.

Each has its own assumptions.

## The k-Distance Plot Is Also a Heuristic

A common DBSCAN workflow plots the distance to the $k$th nearest neighbour and looks for a bend.

This is often used to select epsilon.

The logic resembles the K-means elbow method.

The sorted distance curve can reveal a transition between dense and sparse regions.

But the location of an apparent knee depends on:

- sample size,
- dimension,
- metric,
- scaling,
- $k$,
- mixture proportions,
- and plotting choices.

The knee can be useful.

It is not a theorem that one epsilon is uniquely correct.

## The Chosen k-Distance Must Match min_samples

If the minimum local count is

$$
m,
$$

the relevant neighbour-distance diagnostic should be connected to that count.

Otherwise the plot and clustering criterion refer to different local scales.

Even then, the heuristic does not remove the modelling decision.

It visualizes the empirical distribution of neighbourhood radii required to collect a given number of points.

That is useful information about density heterogeneity.

It does not define anomaly truth.

## High Dimension Makes Neighbourhood Counts Harder to Interpret

The epsilon-ball volume contains

$$
\varepsilon^d.
$$

This is one manifestation of the curse of dimensionality.

In high dimensions, points become sparse.

Distances concentrate.

A neighbourhood radius large enough to capture $m$ points can span a substantial fraction of the data geometry.

Small changes in scaling can have large consequences.

DBSCAN can still be useful in a well-designed lower-dimensional representation.

But the representation then becomes part of the clustering model.

## Dimensionality Reduction Before DBSCAN Composes Assumptions

Suppose we first map

$$
x
\mapsto
z=\phi(x),
$$

then run DBSCAN on $z$.

The result is

$$
\widehat\Pi
=
\operatorname{DBSCAN}
\left(
\phi(X);
\varepsilon,m,d
\right).
$$

Now the noise label depends on:

- the feature map $\phi$,
- the metric $d$,
- epsilon,
- min_samples.

A two-dimensional UMAP followed by DBSCAN is therefore not "DBSCAN on the original data."

It is density clustering in the geometry created by UMAP.

That can be useful.

It should be described accurately.

## Density in an Embedding Is Not Original-Space Density

This matters particularly for t-SNE and UMAP.

Both transformations can distort visual density.

If DBSCAN is then applied to the two-dimensional embedding, its local counts refer to embedding density.

A dense island in the visualization need not correspond to a high-density region in the original representation.

So the pipeline

$$
X
\rightarrow
\text{nonlinear 2D embedding}
\rightarrow
\text{DBSCAN}
$$

can produce clean groups whose density interpretation belongs primarily to the embedding.

Validation should return to the original problem.

## Sample Duplication Can Change Core Status

Suppose observations are duplicated because of resampling, repeated records or data integration.

Neighbour counts increase.

A region can cross the min_samples threshold without any new geometric support.

This is an important practical issue when the dataset contains duplicates.

DBSCAN treats duplicate observations as density.

Sometimes duplicates represent genuine repeated mass.

Sometimes they are data-engineering artifacts.

The algorithm cannot distinguish them.

## Measurement Precision Can Create Artificial Density

Suppose a continuous variable is rounded heavily.

Many observations pile up at identical or nearby values.

DBSCAN sees a dense region.

The density may reflect measurement precision rather than a latent population.

This is analogous to digit heaping in statistics.

A cluster can be real in the observed data and artificial relative to the underlying process.

Preprocessing and measurement design matter.

## Noise Fraction Is Not an Error Rate

Suppose DBSCAN labels

$$
12\%
$$

of observations as noise.

That does not mean:

- 12 percent of the data are bad,
- 12 percent are anomalies,
- 12 percent were mismeasured,
- or 12 percent do not belong to the population.

It means that 12 percent were not assigned to a density-connected cluster under that parameterization.

The correct interpretation is conditional:

> under metric $d$, epsilon $\varepsilon$ and min_samples $m$, these points were not density-reachable from the identified core regions.

That sentence is less dramatic.

It is also what the algorithm actually established.

## Stability Across epsilon Is More Informative Than One Noise Label

Instead of treating one DBSCAN run as definitive, vary epsilon across a scientifically defensible range.

For each observation $i$, define

$$
r_i
=
\frac{
1
}{
R
}
\sum_{r=1}^{R}
\mathbb 1
\left\{
i
\text{ is labelled noise under setting }r
\right\}.
$$

This is a noise-frequency diagnostic across parameter settings.

It is not a probability that the observation is anomalous.

But it distinguishes:

- points labelled noise only under very strict scales,
- from points labelled noise across almost every plausible scale.

That sensitivity information is often more useful than one binary label.

## Cluster Persistence Across Scale Is Also Useful

Similarly, examine whether groups persist across nearby epsilon values.

A cluster that exists only at one narrow parameter setting is different from one that remains coherent across a broad density range.

This idea motivates hierarchical density methods.

Again, persistence is evidence of scale robustness.

It is not proof of a latent generating class.

## External Validation Still Matters

Suppose DBSCAN identifies a small group and several noise points.

If the group or noise status correlates with an independent future outcome, equipment failure, biological marker or process condition, that can make the segmentation useful.

The external variable supplies information not present in the density criterion.

Without such evidence, the grouping remains a geometric description.

That is not a criticism.

Exploratory geometry is valuable.

The interpretation should match the evidence.

## Outlier Claims Need a Reference Distribution

If the substantive claim is

> this observation is anomalous,

define what normal means.

Possible references include:

- a parametric population model,
- an empirical baseline period,
- a conditional model given covariates,
- a local density comparison,
- a process-control distribution,
- or a predictive residual model.

DBSCAN provides one density-connectivity reference.

It may be appropriate.

But calling every noise point an outlier skips the step of defining anomaly relative to the scientific problem.

## A Better Reporting Template

For a DBSCAN analysis, report at least:

- the input representation,
- feature scaling,
- distance metric,
- epsilon,
- min_samples,
- sample size,
- number of clusters,
- noise fraction,
- and sensitivity to nearby parameter values.

If noise points are interpreted substantively, also report:

- why the chosen scale is scientifically meaningful,
- whether the status persists across defensible alternatives,
- and any external evidence that the points are genuinely unusual.

This makes the analysis reproducible and keeps the interpretation proportional to the method.

## A Better Way to State the Result

Instead of:

> DBSCAN detected 74 outliers.

Prefer:

> Under the selected DBSCAN density scale, 74 observations were not density-reachable from any core cluster.

If additional evidence supports anomaly status, say so separately.

For example:

> Of those 74 density-isolated observations, 61 also exceeded the pre-specified process-control threshold.

Now the anomaly interpretation is supported by an independent criterion.

## The Density Threshold Is the Main Result

The approximate relation

$$
f(x)
\gtrsim
\frac{
m
}{
nV_d\varepsilon^d
}
$$

captures most of what practitioners need to remember.

It says:

- epsilon sets the neighbourhood scale,
- dimension changes how neighbourhood volume grows,
- min_samples sets the required count,
- sample size changes expected counts,
- and the metric defines the geometry of the ball.

Noise is therefore always relative to this density model.

There is no parameter-free DBSCAN noise label.

## Conclusion

DBSCAN is valuable because it does not force every observation into a cluster.

That makes its noise label operationally useful.

But useful does not mean intrinsic.

A DBSCAN noise point is not automatically an outlier, anomaly or bad observation.

It is a point that fails a density-connectivity condition at a chosen scale.

The effective density threshold is approximately

$$
\boxed{
\tau
\approx
\frac{
m
}{
nV_d\varepsilon^d
}.
}
$$

Change epsilon.

Change min_samples.

Change dimension, scaling or metric.

The threshold changes.

So can the noise set.

The right interpretation is therefore

$$
\boxed{
\text{DBSCAN noise is a scale-dependent geometric label, not ground truth}.
}
$$

If anomaly detection is the real goal, use the density result as evidence.

Do not let one algorithmic label carry more meaning than the model supplied.

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, 226–231.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *Advances in Knowledge Discovery and Data Mining*, 160–172. https://doi.org/10.1007/978-3-642-37456-2_14
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), 19. https://doi.org/10.1145/3068335
