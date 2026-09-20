---
permalink: '/machine-learning/tsne_umap_plots_are_not_evidence_of_clusters/'
title: 'A t-SNE or UMAP Plot Is Not Evidence That Clusters Exist'
categories:
- Machine Learning
- Statistics
tags:
- Unsupervised Learning
- t-SNE
- UMAP
- Dimensionality Reduction
- Clustering
- Data Visualization
author_profile: false
seo_title: 'Why t-SNE and UMAP Plots Do Not Prove Clusters Exist'
seo_description: 't-SNE and UMAP optimize low-dimensional neighbourhood representations, not tests of cluster existence. Apparent islands, gaps, sizes and distances in 2D can depend strongly on embedding assumptions and parameters.'
excerpt: >-
  A two-dimensional embedding is a model of selected relationships in the original
  data, not a neutral photograph of high-dimensional geometry. t-SNE and UMAP can
  produce informative visualizations, but separated islands in the plot do not by
  themselves establish latent populations, density gaps or a unique clustering.
summary: >-
  A mathematical explanation of why nonlinear dimensionality-reduction plots
  should not be used as stand-alone evidence for clusters. The article derives the
  neighbourhood objectives behind t-SNE and UMAP, explains which aspects of
  geometry they preserve and distort, discusses parameter sensitivity and
  stochasticity, and proposes a validation workflow that separates visualization
  from clustering evidence.
keywords:
- t-SNE clusters
- UMAP clusters
- dimensionality reduction
- unsupervised learning
- clustering validation
- embedding visualization
- perplexity
- nearest neighbour graph
classes: wide
date: '2025-12-14'
why_this_exists: >-
  Low-dimensional embeddings are often interpreted visually as though they were
  direct observations of the high-dimensional population. This encourages strong
  claims from gaps, islands, apparent densities and distances that are partly
  consequences of the embedding objective itself. The article makes those
  objectives explicit and separates useful visualization from evidence of
  clustering.
evidence: >-
  The original t-SNE formulation, UMAP's graph-based manifold construction,
  methodological work on interpreting t-SNE, and the mathematical fact that severe
  dimensionality reduction cannot preserve all high-dimensional relationships.
methodology: >-
  Write the optimization objectives explicitly, identify which pairwise relations
  receive the largest penalties, then examine what can and cannot be inferred from
  the resulting two-dimensional coordinates. Validation recommendations are tied
  back to the original representation rather than the visual embedding alone.
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

A familiar exploratory workflow is:

1. take a high-dimensional dataset,
2. run t-SNE or UMAP,
3. obtain a two-dimensional scatter plot,
4. see several separated islands,
5. conclude that the data contain several clusters.

The first three steps are legitimate.

The fifth does not follow from the fourth.

A nonlinear embedding is not a photograph of the original geometry. It is the output of an optimization problem designed to preserve some relationships while sacrificing others.

That is exactly why these methods are useful.

It is also why their plots are easy to over-interpret.

The correct question is not

> Do I see blobs?

It is

> Which properties of the original data does this embedding objective preserve strongly enough for the visual pattern to support my claim?

For t-SNE and UMAP, that answer is primarily about neighbourhood structure.

It is not a general guarantee about latent groups.

## Dimensionality Reduction Must Distort Something

Suppose the original observations are

$$
x_1,\ldots,x_n\in\mathbb R^p
$$

with large $p$, and the visualization places them at

$$
y_1,\ldots,y_n\in\mathbb R^2.
$$

If the original geometry is genuinely high-dimensional, there is no reason to expect a two-dimensional map to preserve every pairwise distance,

$$
\|x_i-x_j\|
\approx
\|y_i-y_j\|
$$

for all pairs.

There are

$$
\frac{n(n-1)}{2}
$$

pairwise relationships competing for representation in only two coordinates per observation.

A dimensionality-reduction method therefore has to decide which relationships matter most.

PCA makes one choice.

Multidimensional scaling makes another.

t-SNE and UMAP make others.

The map is shaped by those priorities.

## What t-SNE Actually Optimizes

t-SNE begins by converting distances in the original space into neighbour probabilities.

For each observation $x_i$, define

$$
p_{j\mid i}
=
\frac{
\exp\left(
-\|x_i-x_j\|^2/(2\sigma_i^2)
\right)
}{
\sum_{k\neq i}
\exp\left(
-\|x_i-x_k\|^2/(2\sigma_i^2)
\right)
}.
$$

The scale

$$
\sigma_i
$$

is not fixed globally.

It is chosen separately for each observation so that the conditional distribution has a target perplexity.

The conditional probabilities are then symmetrized,

$$
p_{ij}
=
\frac{
p_{j\mid i}+p_{i\mid j}
}{
2n
}.
$$

In the low-dimensional embedding, similarities are defined using a heavy-tailed Student distribution,

$$
q_{ij}
=
\frac{
\left(
1+\|y_i-y_j\|^2
\right)^{-1}
}{
\sum_{k\neq l}
\left(
1+\|y_k-y_l\|^2
\right)^{-1}
}.
$$

The embedding minimizes

$$
\operatorname{KL}(P\|Q)
=
\sum_{i\neq j}
p_{ij}
\log
\frac{p_{ij}}{q_{ij}}.
$$

That objective tells us what t-SNE considers expensive.

If two observations have large high-dimensional similarity

$$
p_{ij},
$$

placing them too far apart so that

$$
q_{ij}
$$

is small creates a substantial penalty.

The method therefore works hard to preserve important local neighbours.

That is not the same as preserving all distances.

## The Asymmetry of the KL Divergence Matters

The t-SNE objective is

$$
\operatorname{KL}(P\|Q),
$$

not a symmetric distance between the two similarity matrices.

If

$$
p_{ij}
$$

is large and

$$
q_{ij}
$$

is very small, the contribution

$$
p_{ij}\log(p_{ij}/q_{ij})
$$

can be large.

A high-dimensional neighbour that is placed far away is expensive.

But if

$$
p_{ij}
$$

is tiny, the contribution remains relatively small even when the low-dimensional embedding gives the pair more proximity than the original geometry would suggest.

This is one reason t-SNE is excellent at preserving local neighbourhoods while global geometry is more difficult to interpret.

The method does not optimize

$$
\sum_{i<j}
\left(
\|x_i-x_j\|
-
\|y_i-y_j\|
\right)^2.
$$

If that were the objective, pairwise distances would have a very different status.

## Perplexity Defines a Scale of Neighbourhood

The parameter called perplexity is often treated as a technical tuning detail.

It is more important than that.

For the conditional similarities around observation $i$, perplexity is defined from entropy,

$$
\operatorname{Perp}(P_i)
=
2^{H(P_i)},
$$

where

$$
H(P_i)
=
-\sum_j
p_{j\mid i}\log_2 p_{j\mid i}.
$$

Informally, perplexity controls the effective neighbourhood size that t-SNE tries to represent.

A small perplexity emphasizes very local structure.

A larger perplexity asks the embedding to account for broader neighbourhood relationships.

There is therefore no parameter-free t-SNE picture waiting to be revealed.

Different perplexities ask different geometric questions.

Martin Wattenberg, Fernanda Viégas and Ian Johnson demonstrated this clearly with simple examples: the same data can produce visibly different t-SNE structures under different perplexities and optimization settings.

A convincing plot under one value is not enough.

## Adaptive Local Scaling Changes the Meaning of Density

Because every observation has its own bandwidth

$$
\sigma_i,
$$

t-SNE adapts to local density.

That is useful for neighbourhood preservation.

It also means that visual density in the embedding should not be read naively as original-space density.

Consider two genuine high-dimensional groups.

One is tightly concentrated.

The other is much more diffuse.

The adaptive bandwidths can normalize some of this local density difference when converting distances to probabilities.

As a result, their apparent sizes or densities in a two-dimensional t-SNE plot can be much more similar than they were in the original space.

The visual statement

> these two islands have similar spread

does not imply

> these two populations have similar variance in the original representation.

Cluster area is not a conserved quantity.

## Distances Between t-SNE Islands Are Not Ordinary Distances

Suppose a t-SNE visualization contains three islands,

$$
A,\quad B,\quad C.
$$

On the plot, perhaps

$$
d(A,B)
<
d(A,C).
$$

It is tempting to conclude that cluster $A$ is more similar to $B$ than to $C$.

That can be unjustified.

t-SNE spends most of its optimization effort preserving high-probability neighbour relationships.

Pairs that are already far apart in the original space have very small

$$
p_{ij}.
$$

Their precise relative placement in the embedding receives much less constraint.

The map needs to put non-neighbours somewhere.

The visible gap between remote islands is therefore partly layout.

It should not automatically be interpreted as a calibrated inter-group distance.

## A Gap in Two Dimensions Is Not Necessarily a Density Gap

This distinction is especially important for clustering.

A visual gap in an embedding is a region of the two-dimensional plane that contains no embedded points.

A density gap in the original data is a statement about

$$
P_X(x)
$$

in the original or scientifically meaningful representation.

They are different objects.

The transformation

$$
\phi:
\mathbb R^p
\rightarrow
\mathbb R^2
$$

is nonlinear and many-to-one at the level of geometry.

A region that is continuous in the original space can become stretched in the embedding.

A curved manifold can unfold.

Local neighbourhood constraints can create visible spacing between regions whose original-space separation is not naturally interpreted as a population boundary.

Therefore,

$$
\boxed{
\text{empty space in the embedding}
\not\Rightarrow
\text{low density in the original space}.
}
$$

That implication needs separate evidence.

## Continuous Structure Can Look Discrete

Suppose observations lie along a continuous nonlinear trajectory.

There may be no point at which the population changes abruptly from one latent type to another.

Yet if local neighbourhood relationships vary along the trajectory, a nonlinear embedding can arrange different sections into visually distinct regions.

Sampling density can strengthen the effect.

If one section of the trajectory is sparsely sampled, the neighbour graph contains fewer connections across that region.

The resulting two-dimensional embedding can show what looks like a gap.

That gap may reflect:

- a real low-density transition,
- uneven sampling,
- a change in local curvature,
- preprocessing,
- neighbour-graph construction,
- or embedding optimization.

The picture alone cannot distinguish them.

## The Reverse Problem Also Happens

Embeddings can also hide separation.

Suppose two groups are distinct in a feature direction that receives little weight in the neighbourhood construction.

If other variables dominate distance, the groups can overlap strongly in the embedding even though the scientifically relevant feature separates them.

So the visual logic fails in both directions:

$$
\text{separate in 2D}
\not\Rightarrow
\text{separate in the original problem},
$$

and

$$
\text{overlap in 2D}
\not\Rightarrow
\text{indistinguishable in the original problem}.
$$

The embedding is a representation.

It is not the population.

## Random Initialization and Optimization Matter

t-SNE is typically solved numerically.

The objective is non-convex.

Different initializations and stochastic optimization paths can produce different arrangements.

Modern implementations are often much more stable than the early folklore suggests, especially when initialization and settings are controlled.

Still, a serious exploratory workflow should distinguish

$$
\text{stable neighbourhood structure}
$$

from

$$
\text{one visually attractive optimization result}.
$$

Repeated runs are cheap compared with the cost of building a scientific interpretation on an accidental layout.

## UMAP Uses a Different Objective

UMAP is not t-SNE with another name.

Its construction begins from local neighbourhood relationships and builds a weighted graph intended to represent a fuzzy topological structure.

For each observation, local distances are rescaled using neighbourhood-dependent quantities. The high-dimensional data become a weighted nearest-neighbour graph.

The low-dimensional embedding then seeks a graph with similar fuzzy connectivity.

A simplified form of the low-dimensional connection strength between embedded points is

$$
v_{ij}
=
\frac{1}{
1+a\|y_i-y_j\|^{2b}
},
$$

with parameters chosen according to the requested embedding behaviour.

The optimization uses a cross-entropy-like objective comparing high-dimensional and low-dimensional membership strengths.

The precise construction differs from t-SNE.

The interpretive warning is similar:

UMAP optimizes a representation of selected neighbourhood and topological relationships.

It does not perform a statistical test that latent classes exist.

## The UMAP Neighbour Parameter Changes the Question

The parameter

$$
n_{\text{neighbors}}
$$

controls the neighbourhood scale used to construct the graph.

Small values emphasize local relationships.

Larger values incorporate broader structure.

Again, this is not merely an implementation knob.

It changes which aspects of the original geometry the algorithm attempts to preserve.

A dataset can look fragmented at one neighbourhood scale and connected at another.

Neither image is automatically false.

They represent different resolutions of the same data.

The mistake is treating one chosen resolution as a unique empirical truth.

## min_dist Changes the Visual Compactness

UMAP also has a parameter commonly called

$$
\text{min\_dist}.
$$

This influences how tightly neighbouring observations may pack in the low-dimensional embedding.

Smaller values permit tighter local concentrations.

Larger values tend to spread local neighbourhoods more broadly.

That means visual compactness is partly controlled by the algorithm.

If a scientific argument depends on the statement

> these observations form a very tight group,

then a parameter that directly controls visual tightness deserves attention.

The relevant question is whether the tightness exists in the original representation, not only in the displayed embedding.

## Beautiful Islands Are Psychologically Persuasive

Humans are extremely good at seeing groups in space.

Give points different colours and spatial separation, and the grouping becomes almost impossible not to perceive.

This creates a communication problem.

A two-dimensional embedding can convert an abstract high-dimensional modelling choice into a picture that looks observationally direct.

Once the plot exists, caveats about perplexity, neighbour graphs or cross entropy feel secondary.

They are not.

The visualization is the end of a modelling pipeline:

$$
X
\rightarrow
\text{preprocessing}
\rightarrow
\text{metric}
\rightarrow
\text{neighbourhood graph}
\rightarrow
\text{embedding objective}
\rightarrow
Y_{\text{2D}}.
$$

Every arrow can affect the final islands.

The eye sees the last object.

The analysis must remember the entire chain.

## Colouring by Known Labels Is Not Independent Validation

Another common plot applies t-SNE or UMAP and then colours points by known class labels.

If colours separate, the representation looks strongly discriminative.

This can be informative.

But the interpretation depends on how the representation was obtained.

If the high-dimensional features came from a supervised model trained using those same labels, then the labels already influenced the geometry before the embedding was computed.

The plot is not independent confirmation.

It is a visualization of a representation trained to separate those classes.

That can still be useful for diagnosis.

It should not be described as independent evidence that the classes are naturally separated in the raw data.

## Clustering the 2D Plot Is Usually a Different Problem

A particularly risky workflow is

$$
X
\rightarrow
\text{t-SNE or 2D UMAP}
\rightarrow
\text{K-means}
\rightarrow
\text{clusters}.
$$

After dimensionality reduction to two coordinates, the clustering algorithm sees only the geometry created by the embedding.

Any information discarded by the embedding is unavailable.

Any distortion introduced by the embedding becomes part of the clustering problem.

The resulting partition is therefore

$$
\widehat\Pi
=
\mathcal A\{\phi(X)\},
$$

not

$$
\mathcal A(X).
$$

That may be intentional.

If the goal is literally to group points by their visualization geometry, it is coherent.

But if the goal is to infer clusters in the original data, the two-dimensional embedding is generally too aggressive a bottleneck to treat casually.

## Dimensionality Reduction Before Clustering Can Still Be Sensible

The previous warning should not be converted into another absolute rule.

Dimensionality reduction before clustering can be very useful.

High-dimensional data often contain noise, redundant variables and unstable distances.

A lower-dimensional representation can improve clustering.

The important distinction is between

$$
\text{dimension reduction for estimation}
$$

and

$$
\text{dimension reduction for plotting}.
$$

A representation with, say, 20 or 50 dimensions may preserve useful structure for clustering while reducing noise.

Collapsing the same data to two dimensions for visualization imposes a much stronger information constraint.

The appropriate dimension should be chosen according to the modelling objective, not because two dimensions fit on a screen.

## PCA Has Different Failure Modes

PCA is sometimes presented as the safe alternative because it is linear.

It is not automatically safe.

PCA preserves directions of large variance.

It does not preserve class relevance or cluster structure by definition.

Suppose cluster separation occurs along a low-variance direction while unrelated noise dominates the first principal components.

Projection onto the leading components can erase the separation.

Conversely, large variance unrelated to latent groups can dominate the plot and create apparent gradients.

The general lesson is broader than t-SNE and UMAP:

$$
\boxed{
\text{a projection preserves what its objective asks it to preserve}.
}
$$

Interpretation should start from that objective.

## A Useful Diagnostic: Trustworthiness

If a low-dimensional embedding is meant to preserve local neighbours, one can measure how well it does so.

Trustworthiness penalizes observations that become neighbours in the embedding even though they were not close in the original space.

At neighbourhood size $k$, it has the general form

$$
T(k)
=
1
-
\frac{
2
}{
nk(2n-3k-1)
}
\sum_i
\sum_{j\in U_k(i)}
\left(
r(i,j)-k
\right),
$$

where

- $U_k(i)$ contains low-dimensional neighbours of $i$ that were not among its original $k$ nearest neighbours,
- $r(i,j)$ is the original-space rank of observation $j$ relative to $i$.

Values close to one indicate good local neighbourhood preservation.

This is a useful embedding diagnostic.

It still does not test for clusters.

It tests whether the map preserves a particular kind of local structure.

Again, metric and claim should match.

## Continuity Measures the Other Direction

A complementary question is whether original-space neighbours remain neighbours in the embedding.

Measures often called continuity penalize original neighbours that become separated after projection.

Trustworthiness and continuity therefore examine different projection errors:

$$
\text{false neighbours introduced}
$$

versus

$$
\text{true neighbours lost}.
$$

A visualization can look clean while performing poorly on one of these quantities.

Computing them is a useful antidote to judging embeddings only by aesthetics.

## Preserve the Original Neighbourhood Graph

If the scientific claim depends on local structure, inspect it before plotting.

For example, construct a $k$-nearest-neighbour graph in the original representation.

Then ask:

- Are the visual islands weakly connected in that original graph?
- How many edges cross the apparent gaps?
- Are those edges concentrated in a small transition region?
- Does the graph remain similar across plausible values of $k$?
- Are apparent islands stable under bootstrap resampling?
- Do graph communities persist without the two-dimensional embedding?

This moves the evidence closer to the geometry that generated the visualization.

A plot can suggest the question.

The original graph should help answer it.

## Test Cluster Claims in the Original Representation

Suppose the embedding suggests four groups.

If the claim is that the original data contain four separated populations, return to the original or scientifically justified representation and test that claim there.

Possible checks include:

- within- versus between-group distances,
- density valleys,
- mixture-model fit,
- graph connectivity,
- resampling stability,
- cluster assignment uncertainty,
- external outcomes,
- predictive differences,
- and sensitivity to scaling or feature choice.

If the four-group interpretation disappears outside the two-dimensional embedding, that is important evidence.

The visualization should not overrule it.

## Parameter Sweeps Should Be Standard

For t-SNE, vary at least the neighbourhood scale represented by perplexity.

For UMAP, vary

$$
n_{\text{neighbors}}
$$

and

$$
\text{min\_dist}.
$$

For both, consider repeated initializations where relevant.

The goal is not to search until a preferred picture appears.

It is the opposite.

We want to know which visual conclusions survive reasonable changes in the embedding specification.

Suppose one parameter choice gives five islands, another gives three, and another gives a continuous arc.

The honest result is not the prettiest plot.

The honest result is that the apparent discrete structure is parameter-sensitive.

## Never Tune the Plot to the Story

There is an especially dangerous feedback loop in exploratory work:

$$
\text{choose parameters}
\rightarrow
\text{view plot}
\rightarrow
\text{adjust parameters}
\rightarrow
\text{preferred story emerges}.
$$

Because embedding parameters are flexible and visual judgment is subjective, this process can create substantial researcher degrees of freedom.

If the final plot is used only as an exploratory aid, that may be acceptable.

If it becomes evidence for a scientific claim, the search process should be documented.

Otherwise, the final figure hides how many alternative geometries were considered and rejected.

## A Strong Workflow Separates Discovery and Confirmation

One useful design is:

### Exploratory stage

Use t-SNE or UMAP to generate hypotheses.

Look for:

- possible subpopulations,
- gradients,
- rare regions,
- local outliers,
- transitions,
- and potential batch effects.

### Confirmatory stage

Return to the original representation or a representation justified independently of the plot.

Test whether the hypothesized structure survives:

- alternative embeddings,
- resampling,
- explicit clustering,
- null models,
- external variables,
- or new data.

This gives visualization an important role without asking it to carry inferential weight it was not designed for.

## Batch Effects Can Produce Excellent-Looking Clusters

Suppose samples come from two laboratories.

The biological process is identical, but the instruments differ slightly.

If the measurement shift is large relative to within-laboratory variation, a neighbour-based embedding may separate the laboratories cleanly.

The resulting islands are real in the sense that the measurements differ.

They are not necessarily biological subtypes.

This is another reason why visual separation is not enough.

A cluster can be statistically strong and scientifically irrelevant.

Metadata should be inspected alongside the embedding.

Sometimes the first variable to colour by is not the target of interest.

It is batch, site, acquisition date, device, preprocessing version or missingness pattern.

## Density Is Particularly Dangerous to Read Visually

Suppose one t-SNE island contains points packed tightly together and another appears diffuse.

It is natural to infer that the first high-dimensional group has greater density.

That inference is unsafe because t-SNE uses adaptive local bandwidths.

UMAP also performs local distance normalization during graph construction.

Visual density therefore mixes properties of the original data with properties of the embedding transformation.

If density is scientifically important, estimate density in a representation where it has a defined meaning.

Do not read it from point packing in a nonlinear 2D plot.

## Area Has No Simple Population Interpretation

The same applies to apparent cluster area.

If one island occupies four times as much screen area as another, that does not imply

$$
\operatorname{Var}(X\mid C_1)
=
4\operatorname{Var}(X\mid C_2),
$$

or any similarly simple relation.

The axes themselves usually have no direct scientific units.

Rotation, reflection and global rescaling of the embedding often change nothing about the objective.

Local stretching and compression are part of the solution.

The plot is topological and relational before it is metric.

Treating area as a quantitative population attribute is usually unjustified.

## The Axes Are Usually Not Features

For PCA,

$$
\text{PC}_1
$$

and

$$
\text{PC}_2
$$

are explicit linear combinations of the original variables.

One can inspect loadings.

For t-SNE and UMAP, the two embedding coordinates typically do not have comparable feature interpretations.

The statement

> observation A has a larger UMAP-1 value than observation B

usually has no standalone scientific meaning.

Only relative configuration matters.

This is another clue that the plot should not be treated as a conventional measurement space.

## Do Not Compute Ordinary Inferential Statistics on the Plot Without a Model

Once the embedding exists, it can be tempting to compute:

- centroids,
- Euclidean distances between cluster centres,
- regression slopes on UMAP-1,
- confidence regions in the 2D coordinates,
- or tests comparing mean t-SNE coordinates.

These numbers are mathematically computable.

Their scientific interpretation is another question.

The coordinates are generated estimates from the entire dataset.

They depend on tuning parameters and other observations.

Their sampling behaviour is not the same as directly observed variables.

Using them in ordinary inferential procedures requires an explicit argument about what quantity is being estimated.

A picture coordinate is not automatically a measured covariate.

## Clustering After UMAP Needs a Different Standard Than Plotting After UMAP

UMAP is sometimes used not only for visualization but as a representation step before clustering.

That can be reasonable, especially when the embedding dimension is larger than two.

But the validation should be designed around the whole pipeline,

$$
X
\rightarrow
\phi_{\text{UMAP}}(X)
\rightarrow
\mathcal A
\rightarrow
\widehat\Pi.
$$

Ask:

- Is the partition stable across UMAP seeds?
- Is it stable across nearby values of $n_{\text{neighbors}}$?
- Does increasing embedding dimension materially change the clusters?
- Are cluster relations visible in the original neighbour graph?
- Does the solution replicate on new data?
- Does it carry external information?

The fact that UMAP often produces visually compelling groups should not lower the validation standard.

It should raise awareness of how persuasive the output can be.

## A Projection Is a Model

It is helpful to write dimensionality reduction in the same language as any other statistical model.

Let

$$
Y
=
\phi_{\lambda}(X),
$$

where

- $X$ is the original data,
- $\phi$ is the embedding method,
- $\lambda$ contains perplexity, neighbour size, minimum distance, metric, initialization and other choices,
- and $Y$ is the displayed representation.

Then every visual claim is conditional on

$$
(\phi,\lambda).
$$

This is exactly analogous to clustering.

In a previous article I argued that a cluster partition is not discovered independently of representation, metric and objective.

The same logic applies one step earlier.

The representation itself is model-dependent.

If clustering is performed on top of an embedding, the assumptions compose.

## What a t-SNE or UMAP Plot Can Support

None of this makes nonlinear embeddings useless.

They are extremely useful exploratory tools.

A carefully interpreted plot can support statements such as:

- some observations have similar local neighbourhoods,
- a subgroup appears locally isolated under this representation,
- an embedding reveals a possible transition,
- labelled classes are locally mixed or separated under the chosen features,
- certain observations may be outliers relative to their neighbours,
- or a candidate structure deserves further investigation.

These are valuable findings.

They are simply narrower than

> the data contain five true clusters.

Precision of language matters because it determines what needs to be validated next.

## A Practical Reporting Template

For any published t-SNE or UMAP figure, I would report at least:

- the input representation,
- preprocessing and scaling,
- the distance metric,
- the embedding algorithm and version,
- perplexity for t-SNE,
- $n_{\text{neighbors}}$ and $\text{min\_dist}$ for UMAP,
- initialization where relevant,
- random seed,
- whether multiple parameter settings were examined,
- and whether the visual conclusion was stable across them.

If the figure motivates cluster claims, I would additionally report validation outside the two-dimensional map.

That documentation turns the embedding from decorative evidence into a reproducible analysis.

## The Important Distinction

A low-dimensional embedding answers a representation problem.

Clustering answers a grouping problem.

They can be connected.

They are not identical.

The logical chain

$$
\text{2D islands}
\Rightarrow
\text{clusters}
\Rightarrow
\text{latent populations}
$$

contains two unsupported jumps unless additional evidence is supplied.

A safer chain is

$$
\text{2D islands}
\Rightarrow
\text{candidate structure}
\Rightarrow
\text{validation in the original problem}
\Rightarrow
\text{qualified interpretation}.
$$

That is slower.

It is also much harder to fool ourselves with.

## Conclusion

t-SNE and UMAP are valuable because they deliberately distort high-dimensional geometry in useful ways.

t-SNE places strong emphasis on preserving probabilistic neighbourhood relationships.

UMAP builds and embeds a locally normalized neighbour graph.

Neither method is designed as a test of whether latent clusters exist.

Therefore,

$$
\boxed{
\text{separated islands in a t-SNE or UMAP plot}
\not\Rightarrow
\text{distinct populations in the original data}.
}
$$

The plot may reveal genuine structure.

It may reveal a useful scale of neighbourhoods.

It may reveal a batch effect.

It may exaggerate a continuous transition.

It may depend strongly on perplexity, neighbour count, minimum distance, metric, preprocessing or initialization.

The correct response to a compelling embedding is not distrust.

It is curiosity with a validation plan.

Use the picture to generate the question.

Do not use the picture as the entire answer.

## References

- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426. https://doi.org/10.48550/arXiv.1802.03426
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to use t-SNE effectively. *Distill*. https://doi.org/10.23915/distill.00002
