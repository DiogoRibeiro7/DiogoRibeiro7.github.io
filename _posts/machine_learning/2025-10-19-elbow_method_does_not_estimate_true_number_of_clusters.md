---
permalink: '/machine-learning/elbow_method_does_not_estimate_true_number_of_clusters/'
title: 'The Elbow Method Does Not Estimate the True Number of Clusters'
categories:
- Machine Learning
- Statistics
tags:
- Unsupervised Learning
- Clustering
- K-Means
- Elbow Method
- Model Selection
- Cluster Validation
author_profile: false
seo_title: 'Why the Elbow Method Does Not Reveal the True Number of Clusters'
seo_description: 'K-means inertia must fall as k increases. For a continuous uniform population, optimal distortion decreases smoothly like 1/k^2, so an apparent elbow is a judgement about diminishing returns rather than proof of latent groups.'
excerpt: >-
  The elbow method is useful as a heuristic for balancing fit against complexity,
  but it is often interpreted too strongly. Within-cluster sum of squares decreases
  mechanically with k, and even a single continuous uniform population has an
  optimal K-means distortion curve proportional to 1/k^2.
summary: >-
  A mathematical critique of the elbow method for selecting the number of clusters.
  The article derives the population K-means distortion for a uniform distribution,
  proves the objective is non-increasing in k, explains why apparent elbows depend
  on plotting scale and judgement, and separates compression trade-offs from claims
  about latent population structure.
keywords:
- elbow method
- k-means inertia
- number of clusters
- unsupervised learning
- clustering validation
- quantization error
- model selection
classes: wide
date: '2025-10-19'
why_this_exists: >-
  The elbow method is frequently described as a way to find the optimal or true
  number of clusters. That interpretation confuses a curve of approximation error
  with a population parameter. A simple continuous distribution already produces
  the characteristic diminishing-return shape without containing discrete groups.
evidence: >-
  Population K-means quantization for the uniform distribution, the monotonicity of
  within-cluster sum of squares as k increases, and methodological literature on
  cluster-number selection and vector quantization.
methodology: >-
  Derive the optimal K-means distortion for a uniform population, distinguish
  approximation error from latent class structure, and examine how curvature,
  scaling, finite-sample noise and candidate range affect what analysts perceive as
  an elbow.
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

The elbow method is one of the most familiar heuristics in clustering.

Fit K-means for several values of

$$
k,
$$

plot the within-cluster sum of squares, and look for the point where the curve bends.

The usual interpretation is:

> the elbow reveals the right number of clusters.

Sometimes it is useful.

That interpretation is still too strong.

The K-means objective must decrease as

$$
k
$$

increases.

That is true even when the population contains no discrete groups at all.

A continuous distribution can produce a smooth curve with diminishing returns simply because more centroids approximate the same population more accurately.

The elbow method therefore answers a compression question before it answers a clustering question.

## The K-Means Objective

Let

$$
X=\{x_1,\ldots,x_n\}
$$

be observations in Euclidean space.

For a fixed number of clusters

$$
k,
$$

K-means solves

$$
W_k
=
\min_{C_1,\ldots,C_k}
\sum_{g=1}^{k}
\sum_{x_i\in C_g}
\|x_i-\mu_g\|_2^2,
$$

where

$$
\mu_g
=
\frac{1}{|C_g|}
\sum_{x_i\in C_g}
x_i
$$

is the centroid of cluster

$$
C_g.
$$

Depending on software, the reported quantity may be called:

- inertia,
- within-cluster sum of squares,
- distortion,
- residual sum of squares,
- or quantization error.

The elbow plot is simply

$$
k
\mapsto
W_k.
$$

The first thing to notice is that this curve has a built-in direction.

## The Objective Cannot Increase When k Increases

Suppose the optimal solution with

$$
k
$$

clusters has objective

$$
W_k.
$$

Now allow

$$
k+1
$$

clusters.

One valid solution is to keep the original

$$
k
$$

clusters and split one cluster into two identical copies, or assign the additional centroid without changing any observation's effective representation.

Therefore the feasible set for

$$
k+1
$$

contains a solution no worse than the optimal

$$
k
$$

solution.

Hence

$$
\boxed{
W_{k+1}\leq W_k.
}
$$

This monotonicity is structural.

A decreasing curve is not evidence that more clusters are more real.

It is evidence that a richer approximation cannot fit worse.

That distinction is the starting point.

## Why the Curve Usually Flattens

The second important fact is that the marginal gain often shrinks.

Define

$$
\Delta_k
=
W_k-W_{k+1}.
$$

The elbow method looks for a point where

$$
\Delta_k
$$

becomes small relative to earlier improvements.

That can happen because genuine groups are being resolved.

It can also happen because approximation error exhibits diminishing returns.

The latter occurs even for completely continuous populations.

## A Uniform Distribution With No Clusters

Consider

$$
X\sim\operatorname{Unif}(0,1).
$$

There is one flat density,

$$
f(x)=1,
\qquad
0\leq x\leq1.
$$

There are no mixture components.

There is no density valley.

There is no special boundary separating latent subpopulations.

Now ask K-means to use

$$
k
$$

centroids.

At the population level, the optimal solution divides the interval into

$$
k
$$

equal cells of width

$$
\frac{1}{k}.
$$

The centroid of each cell lies at its midpoint.

Consider one cell of width

$$
h=\frac{1}{k}.
$$

Relative to its midpoint, let

$$
U\sim\operatorname{Unif}\left(-\frac h2,\frac h2\right).
$$

Its mean squared quantization error is

$$
E[U^2]
=
\frac{h^2}{12}.
$$

Substituting

$$
h=\frac1k,
$$

we obtain

$$
E[U^2]
=
\frac{1}{12k^2}.
$$

Because every cell contributes the same population-weighted distortion, the total optimal population distortion is

$$
\boxed{
D_k
=
\frac{1}{12k^2}.
}
$$

This is the K-means distortion curve for a population with no latent cluster structure.

It already has diminishing returns.

## The Improvements Shrink Automatically

For the uniform population,

$$
D_k
=
\frac{1}{12k^2}.
$$

The improvement from

$$
k
$$

to

$$
k+1
$$

is

$$
\Delta_k
=
D_k-D_{k+1}.
$$

Thus

$$
\Delta_k
=
\frac1{12}
\left(
\frac1{k^2}
-
\frac1{(k+1)^2}
\right).
$$

Simplifying,

$$
\Delta_k
=
\frac{2k+1}{
12k^2(k+1)^2
}.
$$

As

$$
k\to\infty,
$$

this behaves like

$$
\Delta_k
\sim
\frac{1}{6k^3}.
$$

The marginal improvement therefore decays rapidly.

A curve that falls quickly and then flattens is exactly what we should expect from ordinary approximation.

No latent groups are needed.

## Numerical Values Make the Point Clear

For the uniform population,

| $k$ | $D_k = 1/(12k^2)$ |
| ---: | ---: |
| 1 | 0.08333 |
| 2 | 0.02083 |
| 3 | 0.00926 |
| 4 | 0.00521 |
| 5 | 0.00333 |
| 6 | 0.00231 |
| 8 | 0.00130 |
| 10 | 0.00083 |

The first split gives a large improvement.

The second gives another substantial improvement.

Later gains become smaller.

If these numbers came from a real K-means analysis, it would be easy to point to

$$
k=2,
$$

$$
k=3,
$$

or perhaps

$$
k=4
$$

and describe an elbow.

But the generating distribution is exactly one continuous uniform population.

The curve contains no hidden information about a true discrete

$$
k.
$$

## The Elbow Is Often in the Eye of the Analyst

The phrase "look for the elbow" sounds operational.

In practice, the bend can be ambiguous.

Suppose the sequence is

$$
W_1,W_2,\ldots,W_{10}.
$$

One analyst may focus on the large drop from one to two.

Another may notice that improvements remain substantial until three.

A third may choose four because the curve looks nearly flat afterwards.

Without a formal criterion, the elbow is partly visual judgement.

That is not automatically a problem.

Heuristics can be useful.

The problem is describing a subjective approximation trade-off as though it were an estimator with a unique population target.

## Plotting Scale Can Move the Apparent Elbow

Take the uniform distortion

$$
D_k=\frac1{12k^2}.
$$

On the ordinary scale, the curve drops sharply and flattens.

On a logarithmic vertical scale,

$$
\log D_k
=
-\log12-2\log k.
$$

The same data now follow a much smoother relation.

On a log-log plot,

$$
\log D_k
$$

against

$$
\log k
$$

is exactly linear with slope

$$
-2.
$$

The visual elbow largely disappears.

This is an important warning.

A structural property of the population should not depend strongly on whether the analyst plotted the vertical axis linearly or logarithmically.

A visual elbow can.

## The Candidate Range Also Matters

Suppose we only evaluate

$$
k=1,\ldots,5.
$$

The curve may look as though the meaningful bend occurs near

$$
k=3.
$$

Now extend the analysis to

$$
k=20.
$$

The same first few points are present, but the visual context changes.

The horizontal scale stretches.

Later flattening becomes more visible.

The apparent location of the elbow can shift because human perception is relative to the displayed range.

Again, this is fine for an exploratory heuristic.

It is weaker evidence than the phrase "optimal number of clusters" often suggests.

## Finite Samples Add Noise

At the population level, the uniform example gives a smooth deterministic curve,

$$
D_k=\frac1{12k^2}.
$$

With a finite sample, we observe

$$
\widehat W_k.
$$

This quantity varies because of:

- sampling variation,
- K-means initialization,
- optimization quality,
- outliers,
- and preprocessing.

The second differences

$$
\widehat W_{k-1}
-
2\widehat W_k
+
\widehat W_{k+1}
$$

can fluctuate substantially.

An apparent kink may therefore reflect sampling noise or local optimization rather than a meaningful structural transition.

Repeating K-means with several initializations helps with optimization noise.

It does not remove sampling uncertainty.

## K-Means Always Benefits From More Centroids

K-means is also a vector quantizer.

Its centroids approximate the support of the distribution.

This interpretation is useful because it separates two different purposes.

### Purpose 1: compression

Use

$$
k
$$

representative prototypes to summarize the data.

Then choosing

$$
k
$$

is a rate-distortion or complexity problem.

### Purpose 2: latent group discovery

Interpret each Voronoi cell as a distinct population.

Then choosing

$$
k
$$

is a scientific classification claim.

The same K-means optimization can be used for both.

The meaning of the centroids is different.

An elbow can be perfectly useful for the first purpose while providing weak evidence for the second.

## Quantization Explains Why Continuous Data Produce "Clusters"

For any set of centroids

$$
\mu_1,\ldots,\mu_k,
$$

K-means induces Voronoi cells

$$
V_g
=
\left\{
x:
\|x-\mu_g\|
\leq
\|x-\mu_h\|
\text{ for all }h
\right\}.
$$

Every point belongs to one cell.

So even a smooth continuous density is partitioned into discrete regions.

These regions are algorithmic.

They need not correspond to generating subpopulations.

The uniform interval makes this transparent:

$$
[0,1]
$$

is partitioned into

$$
k
$$

equal bins.

The bins are useful approximations.

Calling them

$$
k
$$

natural populations would add an interpretation that the distribution itself does not supply.

## Genuine Mixtures Can Also Lack a Clear Elbow

The converse problem matters too.

Suppose data truly come from a mixture with

$$
G
$$

components.

There is no guarantee that the K-means distortion curve will show an obvious elbow at

$$
k=G.
$$

If components overlap heavily, splitting one component further may reduce squared error almost as much as separating another.

If component variances differ strongly, K-means may allocate multiple centroids to one diffuse component and one centroid to several tight nearby components.

If mixture components are elongated, curved or unequal in mass, Euclidean centroid distortion may not align with the generative classes.

Thus,

$$
\text{true mixture component count}
\not\Rightarrow
\text{obvious K-means elbow at the same }k.
$$

The heuristic can fail in both directions.

## Unequal Variance Is a Simple Failure Mode

Imagine two Gaussian components.

One is very tight:

$$
X\mid Z=1
\sim
\mathcal N(\mu_1,\sigma_1^2I),
$$

with small

$$
\sigma_1.
$$

The other is diffuse:

$$
X\mid Z=2
\sim
\mathcal N(\mu_2,\sigma_2^2I),
$$

with

$$
\sigma_2\gg\sigma_1.
$$

With

$$
k=2,
$$

K-means may place one centroid in each component.

With

$$
k=3,
$$

the best squared-error reduction may come from splitting the diffuse component into two regions.

That improvement can be substantial even though the generative mixture still has only two components.

K-means counts prototypes.

A mixture model counts components.

Those are not always the same object.

## Cluster Number Depends on the Cluster Concept

This is a broader issue.

For centroid clustering, the relevant object is a set of prototypes.

For density clustering, groups may correspond to connected high-density regions.

For mixture models, groups may correspond to latent components.

For hierarchical clustering, one dataset can support several resolutions simultaneously.

For graph clustering, groups depend on connectivity and cut objectives.

There is therefore no universal quantity called

$$
\text{the number of clusters}
$$

independent of the clustering concept.

The elbow method is specifically tied to the K-means distortion objective.

Its answer should be interpreted within that model.

## Curvature Can Be Formalized, But the Meaning Does Not Change

Several methods try to automate elbow detection.

For a discrete curve

$$
W_k,
$$

one can examine second differences,

$$
\Delta^2 W_k
=
W_{k-1}
-
2W_k
+
W_{k+1}.
$$

Or one can compute distance from the line joining the first and last candidate points.

Or estimate piecewise-linear breakpoints.

These methods can make the selection reproducible.

They do not transform the elbow into a test for latent groups.

They formalize a bend in a distortion curve.

The statistical meaning remains

$$
\text{a change in marginal approximation gain}.
$$

Whether that change corresponds to population structure is a separate question.

## Scaling Features Can Change the Elbow

Suppose

$$
x=(x_1,x_2)
$$

and we rescale the second coordinate,

$$
x_2^\star=cx_2.
$$

The K-means objective becomes

$$
\sum_i
\left[
(x_{i1}-\mu_{z_i,1})^2
+
c^2
(x_{i2}-\mu_{z_i,2})^2
\right].
$$

The distortion curve

$$
W_k
$$

changes.

So can its curvature.

So can the selected elbow.

This is another reason why the elbow is not an intrinsic property of the raw observations.

It is conditional on representation and geometry.

A valid analysis should report the preprocessing under which the elbow was obtained.

## Outliers Can Create Spurious Improvements

Suppose most observations form one compact population and a few extreme points lie far away.

For small

$$
k,
$$

those outliers contribute heavily to squared distance.

Adding a centroid may sharply reduce the objective simply by assigning one prototype to the extreme region.

The elbow plot can then suggest a new group.

Whether that group is a meaningful population or merely a handful of unusual observations is a substantive question.

K-means itself does not distinguish them.

Squared error rewards whichever allocation reduces distance most.

## The Elbow Is a Cost-Benefit Curve

A more accurate interpretation is:

$$
\boxed{
W_k
\text{ describes the cost of representing the data with }k\text{ centroids}.
}
$$

Choosing

$$
k
$$

then becomes a trade-off between:

- approximation error,
- model complexity,
- interpretability,
- operational cost,
- and perhaps downstream usefulness.

This is not a weakness.

It is often exactly the decision we need.

For example, if a logistics operation can only support five service tiers, then selecting

$$
k=5
$$

may be sensible regardless of whether an abstract criterion prefers seven.

The practical optimum and the population structure are different questions.

## Add an Explicit Complexity Penalty

If the real problem is balancing fit and complexity, write that objective directly.

For example,

$$
J(k)
=
W_k+\lambda k.
$$

Then choose

$$
\hat k
=
\arg\min_k J(k).
$$

The parameter

$$
\lambda
$$

states how much additional complexity costs.

This is conceptually cleaner than pretending the curve itself contains a uniquely correct elbow.

Other model-selection frameworks use penalties motivated by likelihood or coding length.

The exact criterion depends on the model.

The important point is that complexity preferences should be explicit.

## The Gap Statistic Changes the Question

The gap statistic compares observed clustering distortion with distortion under a reference distribution.

Its basic form is

$$
\operatorname{Gap}(k)
=
E^\star[\log W_k^\star]
-
\log W_k.
$$

Now the question is not merely

> How much did distortion fall?

It is

> Is the observed distortion unusually small relative to a specified null distribution?

That is a stronger inferential framework.

It still depends on the reference distribution.

If the null is poorly chosen, the comparison can be misleading.

But at least the no-cluster baseline is explicit.

The elbow method does not provide one.

## Silhouette Asks a Different Question

Silhouette compares within-cluster cohesion with separation from the nearest competing cluster.

For observation

$$
i,
$$

$$
s(i)
=
\frac{b(i)-a(i)}
{\max\{a(i),b(i)\}}.
$$

It is therefore sensitive to a different aspect of geometry than raw K-means distortion.

In another article, I showed that even a single uniform distribution split into two halves has population average silhouette

$$
2-\frac54\log3
\approx0.627.
$$

So silhouette is not a proof of latent groups either.

The lesson is not that all validation metrics are useless.

It is that each one answers a specific question.

## Stability Adds Reproducibility, Not Truth

Suppose

$$
k=3
$$

produces highly stable assignments across bootstrap samples.

That is evidence that the three-way partition is reproducible under the chosen representation and algorithm.

It does not prove that exactly three populations generated the data.

A continuous distribution can produce stable quantization cells.

Stability tells us about sensitivity.

The elbow tells us about approximation gain.

Silhouette tells us about cohesion and separation.

These are different pieces of evidence.

None should silently absorb the meaning of the others.

## External Validation Can Change the Decision

Suppose several values of

$$
k
$$

have similar distortion.

The elbow is ambiguous.

But cluster assignments at

$$
k=4
$$

strongly predict a future operational outcome that was not used in fitting.

Then

$$
k=4
$$

may be a useful choice.

Conversely, a visually perfect elbow at

$$
k=3
$$

may produce groups with no reproducibility or external relevance.

This is why model selection should reflect the actual purpose of the clustering.

The distortion curve is one input.

It is not the whole decision.

## A Better Workflow

For K-means analyses where

$$
k
$$

matters substantively, I would use several steps.

### 1. Plot the distortion curve

The elbow plot is still useful.

It shows the rate of approximation improvement.

### 2. Show marginal gains

Report

$$
\Delta_k
=
W_k-W_{k+1}.
$$

This makes the diminishing-return structure explicit.

### 3. Normalize when appropriate

A relative improvement,

$$
R_k
=
\frac{W_k-W_{k+1}}{W_k},
$$

can be easier to compare across scales.

### 4. Examine multiple visual scales

Look at linear and logarithmic axes.

If the elbow disappears under a harmless plotting transformation, treat the visual argument cautiously.

### 5. Compare with a null model

Use a reference distribution appropriate to the application.

### 6. Evaluate stability

Check whether the proposed partition survives resampling and reasonable preprocessing changes.

### 7. Examine external meaning

Where possible, test whether the groups differ on independent variables or future outcomes.

### 8. State the purpose of k

Is

$$
k
$$

being chosen for compression, interpretation, operations, or a claim about latent populations?

Different purposes justify different criteria.

## Do Not Call Every Selected k "Optimal"

The word

> optimal

needs an objective.

Optimal under what criterion?

For K-means distortion alone,

$$
k=n
$$

is optimal because every observation can receive its own centroid and

$$
W_n=0.
$$

We reject that solution because complexity matters.

Therefore any practically selected

$$
k<n
$$

already reflects a trade-off beyond distortion.

Calling the elbow-derived value "the optimal number of clusters" hides that trade-off.

A more precise phrase is:

> the selected number of clusters under the chosen model-selection criterion.

That is less dramatic.

It is also more accurate.

## The Uniform Example Is Enough to Break the Strong Interpretation

The key counterexample needs no high-dimensional geometry.

No noise variables.

No model misspecification.

No optimization pathology.

Only

$$
X\sim\operatorname{Unif}(0,1).
$$

For this population,

$$
D_k=\frac1{12k^2}.
$$

The distortion curve decreases rapidly at first and then flattens.

That shape is the raw material from which elbows are visually identified.

Yet the population contains no discrete latent grouping.

Therefore,

$$
\boxed{
\text{an elbow-like distortion curve does not imply a true cluster count}.
}
$$

That logical implication fails in the simplest possible setting.

## Conclusion

The elbow method is useful when interpreted modestly.

It summarizes diminishing returns in K-means approximation error.

It can help choose a parsimonious number of centroids.

It can support an operational trade-off.

It can suggest candidate values of

$$
k
$$

for further analysis.

What it does not do is estimate a universal population quantity called the true number of clusters.

K-means distortion decreases mechanically as model capacity increases.

For a single continuous uniform population,

$$
\boxed{
D_k=\frac1{12k^2},
}
$$

so the characteristic flattening appears even when no latent groups exist.

The right interpretation is therefore:

$$
\boxed{
\text{the elbow is a diminishing-return heuristic, not a proof of cluster existence}.
}
$$

If the chosen

$$
k
$$

will carry scientific meaning, the distortion curve should be followed by stronger evidence: null comparisons, stability, external validation and a clear definition of what a cluster is meant to represent.

Use the elbow to narrow the search.

Do not ask it to decide what the population is.

## References

- Gersho, A., & Gray, R. M. (1992). *Vector Quantization and Signal Compression*. Kluwer Academic Publishers.
- Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666. https://doi.org/10.1016/j.patrec.2009.09.011
- Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika*, 18, 267–276. https://doi.org/10.1007/BF02289263
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B*, 63(2), 411–423. https://doi.org/10.1111/1467-9868.00293
