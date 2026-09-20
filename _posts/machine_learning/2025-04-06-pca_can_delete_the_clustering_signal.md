---
permalink: '/machine-learning/pca_can_delete_the_clustering_signal/'
title: 'PCA Can Delete the Clustering Signal'
categories:
- Machine Learning
- Statistics
tags:
- Unsupervised Learning
- PCA
- Clustering
- Dimensionality Reduction
- Representation Learning
- Variance
author_profile: false
seo_title: 'Why PCA Can Destroy Cluster Structure'
seo_description: 'PCA preserves variance, not cluster separation. A two-cluster Gaussian example retains about 86 percent of total variance in the first principal component while discarding all information about cluster identity.'
excerpt: >-
  Principal component analysis is often used before clustering to remove noise and
  reduce dimension. That can help, but PCA optimises variance rather than cluster
  separation. A simple Gaussian construction shows that the first principal
  component can retain most of the total variance while containing zero information
  about the true cluster label.
summary: >-
  A mathematical counterexample to the routine use of PCA before clustering. The
  article derives a two-cluster Gaussian model in which a high-variance nuisance
  coordinate becomes the first principal component while the low-variance
  coordinate contains essentially perfect cluster separation. Projection onto the
  leading component retains about 86 percent of variance and reduces the best
  possible cluster-classification accuracy from essentially 100 percent to 50
  percent.
keywords:
- PCA clustering
- principal component analysis
- unsupervised learning
- dimensionality reduction
- cluster separation
- variance explained
- low variance signal
classes: wide
date: '2025-04-06'
why_this_exists: >-
  PCA is frequently inserted before clustering as a generic preprocessing step,
  often justified by variance explained. That criterion is not aligned with cluster
  recovery. The article gives an exact population example where preserving most
  variance destroys all information about the latent grouping.
evidence: >-
  Population covariance analysis for a balanced Gaussian mixture, the Bayes
  classification error before and after projection, and standard PCA geometry.
methodology: >-
  Construct one high-variance nuisance direction and one lower-variance
  cluster-separating direction, derive the principal components analytically, and
  compare retained variance with retained information about the latent class.
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

Principal component analysis is often placed before clustering almost automatically.

The reasoning is familiar:

1. high-dimensional distances are noisy,
2. PCA removes redundant directions,
3. retain the components explaining most of the variance,
4. cluster in the reduced space.

Sometimes this is exactly the right thing to do.

But there is a gap in the argument.

PCA preserves directions of large variance.

Clustering cares about directions that separate groups.

Those directions need not be the same.

A low-variance coordinate can contain nearly perfect cluster information.

A high-variance coordinate can contain none.

If we reduce dimension according to variance alone, PCA can remove the signal we hoped clustering would discover.

## PCA Solves a Variance Problem

Let the centered observation be

$$
X\in\mathbb R^p
$$

with covariance matrix

$$
\Sigma
=
E[XX^T].
$$

The first principal component direction is

$$
v_1
=
\arg\max_{\|v\|=1}
\operatorname{Var}(v^T X).
$$

Because

$$
\operatorname{Var}(v^T X)
=
v^T\Sigma v,
$$

the solution is the eigenvector corresponding to the largest eigenvalue of $\Sigma$.

The second component maximizes remaining variance subject to orthogonality, and so on.

Nothing in this definition refers to:

- cluster separation,
- density valleys,
- latent classes,
- future outcomes,
- or scientific relevance.

PCA asks one question:

$$
\boxed{
\text{which directions explain the most variance?}
}
$$

That is not the same as asking which directions distinguish groups.

## A Two-Cluster Population

Let the latent cluster label be

$$
Z\in\{-1,+1\}
$$

with

$$
P(Z=-1)
=
P(Z=+1)
=
\frac12.
$$

Now define a two-dimensional observation

$$
X=(X_1,X_2).
$$

The first coordinate is pure nuisance variation:

$$
X_1
\sim
N(0,\sigma_n^2),
$$

independent of $Z$.

The second coordinate carries the cluster signal:

$$
X_2
=
\mu Z+\varepsilon,
$$

where

$$
\varepsilon
\sim
N(0,\sigma_s^2)
$$

and is independent of both $X_1$ and $Z$.

Thus the two clusters differ only in the second coordinate:

$$
X_2\mid Z=+1
\sim
N(\mu,\sigma_s^2),
$$

$$
X_2\mid Z=-1
\sim
N(-\mu,\sigma_s^2).
$$

The first coordinate has exactly the same distribution in both clusters.

## The Population Covariance Is Diagonal

Because the mixture is balanced,

$$
E[X_2]
=
0.
$$

We also have

$$
E[X_1]=0.
$$

The variance of the nuisance coordinate is simply

$$
\operatorname{Var}(X_1)
=
\sigma_n^2.
$$

For the signal coordinate,

$$
\operatorname{Var}(X_2)
=
\operatorname{Var}(\mu Z+\varepsilon).
$$

Since $Z$ and $\varepsilon$ are independent,

$$
\operatorname{Var}(X_2)
=
\mu^2\operatorname{Var}(Z)
+
\sigma_s^2.
$$

Because

$$
Z\in\{-1,+1\}
$$

with equal probabilities,

$$
\operatorname{Var}(Z)=1.
$$

Therefore,

$$
\operatorname{Var}(X_2)
=
\mu^2+\sigma_s^2.
$$

The covariance between the coordinates is zero, so

$$
\boxed{
\Sigma
=
\begin{pmatrix}
\sigma_n^2 & 0\\
0 & \mu^2+\sigma_s^2
\end{pmatrix}.
}
$$

The PCA directions are therefore exactly the coordinate axes.

## When Nuisance Variance Wins

Suppose

$$
\sigma_n^2
>
\mu^2+\sigma_s^2.
$$

Then the largest eigenvalue of $\Sigma$ is

$$
\sigma_n^2.
$$

The first principal component direction is

$$
v_1
=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

The first PCA score is therefore

$$
T_1
=
v_1^T X
=
X_1.
$$

But $X_1$ is independent of the cluster label:

$$
X_1\perp Z.
$$

Hence

$$
P(Z=1\mid T_1)
=
P(Z=1)
=
\frac12.
$$

The retained principal component contains no information about cluster identity.

Formally,

$$
\boxed{
I(Z;T_1)=0.
}
$$

PCA has selected the direction with the most variance.

That direction is pure nuisance.

## The Best Possible Classifier After Projection Is Random Guessing

Because

$$
T_1=X_1
$$

has the same distribution in both clusters,

$$
T_1\mid Z=+1
\sim
N(0,\sigma_n^2),
$$

and

$$
T_1\mid Z=-1
\sim
N(0,\sigma_n^2).
$$

No classifier using only $T_1$ can distinguish the two groups.

With equal class probabilities, the Bayes error is

$$
R^\star_{\text{PCA}}
=
\frac12.
$$

So after reducing to the first principal component, the best possible classification accuracy is

$$
50\%.
$$

This is not a failure of K-means.

It is not a failure of Gaussian mixtures.

It is not a poor initialization.

The information has been removed before clustering begins.

## Before PCA, the Clusters Can Be Almost Perfectly Separated

In the original two-dimensional space, all cluster information lies in $X_2$.

The Bayes classifier is simply

$$
\hat Z
=
\operatorname{sign}(X_2).
$$

For the positive cluster, the error probability is

$$
P(X_2<0\mid Z=+1)
=
P(\mu+\varepsilon<0).
$$

Therefore,

$$
R^\star
=
\Phi
\left(
-\frac{\mu}{\sigma_s}
\right),
$$

where $\Phi$ is the standard normal distribution function.

If the separation ratio

$$
\frac{\mu}{\sigma_s}
$$

is large, the error is tiny.

So PCA can convert an essentially trivial clustering problem into an impossible one.

## A Numerical Example

Choose

$$
\sigma_n=5,
$$

$$
\mu=2,
$$

and

$$
\sigma_s=0.3.
$$

Then

$$
\operatorname{Var}(X_1)=25,
$$

while

$$
\operatorname{Var}(X_2)
=
4+0.09
=
4.09.
$$

The first principal component is therefore the nuisance coordinate.

The proportion of total variance explained by PC1 is

$$
\frac{25}{25+4.09}
\approx
0.859.
$$

So PCA retains about

$$
85.9\%
$$

of the total variance in one component.

That sounds excellent.

But the retained component contains zero cluster information.

Meanwhile, before projection,

$$
\frac{\mu}{\sigma_s}
=
\frac2{0.3}
\approx
6.67.
$$

The Bayes error is

$$
\Phi(-6.67),
$$

which is on the order of

$$
10^{-11}.
$$

So the original clusters are essentially perfectly recoverable.

After retaining the component explaining about 86 percent of the variance, the best possible accuracy becomes

$$
50\%.
$$

The contrast is extreme:

$$
\boxed{
\text{variance retained}
\approx86\%
\qquad
\text{cluster information retained}
=
0.
}
$$

## Explained Variance Is Not Explained Structure

This example is deliberately simple.

Its purpose is to separate two statements that are often conflated.

Statement one:

> the retained components explain most of the total variance.

Statement two:

> the retained components preserve the structure relevant to clustering.

The first does not imply the second.

Variance is a property of the marginal feature distribution.

Cluster separation is a property of how that distribution differs across latent groups.

They can point in different directions.

## Why High Variance Can Be Irrelevant

Large variance can arise from many sources unrelated to the grouping of interest:

- measurement scale,
- subject size,
- background intensity,
- device differences,
- seasonality,
- operating load,
- batch effects,
- random nuisance variables,
- or a continuous factor shared by all groups.

PCA has no way to know that these directions are irrelevant.

If they dominate covariance, they dominate the leading components.

This is correct PCA behaviour.

The problem is using a variance objective as a proxy for a clustering objective.

## Low-Variance Features Can Be Decisive

Suppose two populations differ by a small but highly consistent shift in one feature.

Within each population, that feature has little variation.

The total marginal variance can therefore be modest.

Yet because within-group variance is even smaller, the feature may separate groups almost perfectly.

Cluster relevance depends on the ratio

$$
\text{between-group variation}
\quad\text{relative to}\quad
\text{within-group variation},
$$

not simply on total marginal variance.

PCA does not optimize that ratio.

## Fisher's Discriminant Makes the Contrast Clear

In supervised two-class problems, Fisher's linear discriminant considers

$$
J(v)
=
\frac{
v^T S_B v
}{
v^T S_W v
},
$$

where

- $S_B$ measures between-class variation,
- $S_W$ measures within-class variation.

This explicitly rewards directions that separate classes relative to their internal spread.

PCA instead considers

$$
v^T S_T v,
$$

where $S_T$ is total covariance.

For labelled data,

$$
S_T
=
S_W+S_B.
$$

A direction can have enormous total variance because $S_W$ is large while offering little separation.

Another can have moderate total variance but a large between-to-within ratio.

PCA and discriminant analysis therefore solve different problems.

The same distinction remains important when labels are hidden.

## In Unsupervised Learning, We Do Not Know Which Variance Matters

The challenge is harder without labels.

If we knew the true clusters, we could measure which directions separate them.

But clustering is supposed to discover the groups.

This creates a circularity.

We want to reduce dimension before clustering, but the dimensions worth preserving may depend on the unknown cluster structure.

PCA resolves the problem by preserving variance.

That is a defensible generic criterion.

It is not guaranteed to preserve clusters.

## Standardization Changes the Example

One might object that sensible analysts standardize features before PCA.

That can change the result completely.

If $X_1$ and $X_2$ are each scaled to unit variance, the population correlation matrix in the example becomes the identity:

$$
R
=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}.
$$

Now the principal directions are not uniquely identified.

Any orthonormal basis is a valid PCA basis.

The cluster signal has not automatically become the first principal component.

Standardization has removed the dominance of the nuisance scale.

It has not told PCA which direction contains clustering information.

## Standardization Is Another Modelling Choice

If the original units are meaningful, standardizing may over-weight noisy low-variance measurements.

If units are arbitrary or incomparable, failing to standardize may let scale dominate.

There is no universal rule.

The important point is that preprocessing determines the covariance geometry that PCA sees.

The full pipeline is

$$
X
\xrightarrow{\text{scaling}}
X^\star
\xrightarrow{\text{PCA}}
Z
\xrightarrow{\text{clustering}}
\widehat\Pi.
$$

Cluster interpretation is conditional on the entire pipeline.

## Whitening Goes Even Further

PCA whitening rescales principal components to unit variance.

If

$$
Z_j
$$

has eigenvalue

$$
\lambda_j,
$$

whitening uses

$$
\widetilde Z_j
=
\frac{Z_j}{\sqrt{\lambda_j}}.
$$

This removes the variance dominance among retained components.

That can help distance-based clustering.

It can also amplify small-eigenvalue directions that are mostly noise.

Again, the transformation changes the metric.

Whitening is neither universally right nor universally wrong.

It encodes a choice about how directions should be weighted after projection.

## Keeping More Components Delays the Problem

In the two-dimensional example, retaining both principal components preserves all information.

So the problem appears only when dimension is reduced.

In higher-dimensional data, the same issue is unavoidable once some directions are discarded.

Suppose there are

$$
p
$$

nuisance dimensions with larger eigenvalues than one informative dimension.

If PCA retains only the first

$$
r<p+1
$$

components, the cluster signal can disappear entirely.

Retaining 90, 95 or even 99 percent of variance does not guarantee protection.

A sufficiently low-variance but decisive signal can live in the discarded tail.

## Ninety-Five Percent Variance Is Not a Statistical Guarantee

The rule

> keep enough PCs to explain 95 percent of the variance

is convenient.

It does not come from a theorem saying 95 percent of clustering structure is preserved.

The retained variance ratio is

$$
\frac{
\sum_{j=1}^{r}\lambda_j
}{
\sum_{j=1}^{p}\lambda_j
}.
$$

This is a statement about covariance eigenvalues.

It contains no term involving latent group labels.

Therefore it cannot guarantee latent-group recovery.

The threshold is a compression criterion.

It is not a clustering guarantee.

## Even 99 Percent Can Fail

Suppose a nuisance direction has variance

$$
99
$$

and the cluster-signal direction has variance

$$
1.
$$

The first principal component explains

$$
99\%
$$

of total variance.

If the cluster signal lies entirely in the second component, retaining 99 percent of variance removes all cluster information.

The number 99 sounds reassuring only because it is large.

Its relevance depends on the target structure.

## The Scree Plot Has the Same Limitation

A scree plot displays eigenvalues

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p.
$$

An elbow can suggest a low-dimensional variance structure.

It does not establish that the components below the elbow are irrelevant to clustering.

A small eigenvalue can still correspond to a highly reproducible low-noise contrast between groups.

Scree selection is about covariance approximation.

Cluster relevance is a different question.

## Reconstruction Error Can Be Excellent While Clustering Is Destroyed

PCA of rank $r$ minimizes mean squared reconstruction error among linear rank-$r$ projections.

That is,

$$
P_r X
$$

is optimal for reconstructing $X$ in squared Euclidean loss.

In the counterexample, projecting onto $X_1$ is therefore correct if reconstruction error is the objective.

The problem is not that PCA chose badly.

The problem is that reconstruction and cluster recovery are different objectives.

An excellent reconstruction can omit a low-energy feature carrying the cluster identity.

## This Is an Information Bottleneck

After projection,

$$
Z=P_rX,
$$

clustering can use only information contained in $Z$.

By the data processing inequality,

$$
I(Z;C)
\leq
I(X;C),
$$

where $C$ is any latent cluster label.

Dimension reduction cannot create information about $C$ that was not already present.

It can preserve some.

It can discard some.

In the counterexample,

$$
I(X;Z_{\text{cluster}})
>0
$$

for the latent class relation, but

$$
I(T_1;Z)=0
$$

when $T_1=X_1$.

The bottleneck has removed all label information.

## PCA Can Also Improve Clustering

The critique should not be overextended.

Suppose high-dimensional data contain many independent noise variables with small or moderate variance.

If the cluster structure lies in a low-dimensional subspace that also carries substantial variance, PCA can improve clustering by:

- reducing distance concentration,
- removing noisy directions,
- stabilizing covariance estimates,
- lowering computational cost,
- and suppressing measurement error.

PCA before clustering is often useful.

The point is that usefulness is conditional.

It should be validated rather than assumed from explained variance alone.

## A Favourable Example

Suppose

$$
X
=
\mu_Z+\varepsilon,
$$

where cluster means differ strongly along the first few eigenvectors and the remaining dimensions contain isotropic low-variance noise.

Then the between-cluster structure contributes substantially to total covariance.

The leading PCs can recover the relevant subspace.

In this setting, PCA and clustering objectives happen to align.

That alignment should be demonstrated for the problem.

It is not built into PCA.

## Mixture Covariance Shows When PCA Can Work

For a mixture with latent class $Z$, the total covariance can be decomposed as

$$
\Sigma_T
=
E
\left[
\operatorname{Cov}(X\mid Z)
\right]
+
\operatorname{Cov}
\left(
E[X\mid Z]
\right).
$$

Write

$$
\Sigma_W
=
E
\left[
\operatorname{Cov}(X\mid Z)
\right]
$$

for within-cluster covariance and

$$
\Sigma_B
=
\operatorname{Cov}
\left(
E[X\mid Z]
\right)
$$

for between-cluster covariance.

Then

$$
\boxed{
\Sigma_T
=
\Sigma_W+\Sigma_B.
}
$$

PCA diagonalizes $\Sigma_T$.

Cluster separation is driven by $\Sigma_B$ relative to $\Sigma_W$.

If large eigenvalues of $\Sigma_T$ are dominated by $\Sigma_B$, PCA can help.

If they are dominated by $\Sigma_W$, PCA can focus on nuisance variation.

This decomposition explains both success and failure.

## Batch Effects Are a Common Real-World Version

Suppose biological samples contain subtle disease subtypes.

Now add a large laboratory batch effect.

The batch effect may explain the largest share of total variance.

PCA will correctly put it in the leading components.

A clustering on those components may recover laboratories rather than biological states.

The clustering can be:

- stable,
- visually separated,
- high silhouette,
- and completely irrelevant to the scientific question.

This is not a contradiction.

It is exactly what variance-based representation predicts.

## Subject Size Can Dominate Shape

In longitudinal or functional data, one common nuisance factor is overall level or amplitude.

Suppose two scientifically relevant groups differ in trajectory shape but individuals vary much more in baseline level.

PCA on raw trajectories may devote the first component to level.

If only one or two components are retained, subtle shape differences can disappear.

Centering each subject before PCA asks a different question.

Neither representation is universally correct.

The scientific object must determine which variation is nuisance and which is signal.

## PCA After Standardization Can Promote Noise

Suppose one variable has tiny variance because it is measured very precisely and contains no useful structure.

Standardizing it to unit variance gives it the same marginal scale as every other variable.

Now PCA can allocate substantial loading to that direction.

So the advice

> always standardize before PCA

is no safer than

> never standardize.

The correct transformation depends on units, noise levels and what differences should count as meaningful.

## Measurement Error Matters

Suppose

$$
X_j
=
S_j+\eta_j,
$$

where $S_j$ is signal and $\eta_j$ is measurement error.

A variable can have high variance because

$$
\operatorname{Var}(\eta_j)
$$

is large.

PCA sees total variance:

$$
\operatorname{Var}(X_j)
=
\operatorname{Var}(S_j)
+
\operatorname{Var}(\eta_j)
$$

under independence.

Without a measurement model, it cannot distinguish the two sources.

High variance is not automatically high information.

## Sparse PCA Does Not Solve the Objective Mismatch

Sparse PCA can make components easier to interpret by encouraging many loadings to be zero.

That can be useful.

It still optimizes a variance-oriented objective.

Sparsity changes which variance representation is preferred.

It does not automatically target cluster separation.

The same caution applies to robust PCA, kernel PCA and nonlinear variants.

Changing the projection family does not remove the need to align the objective with the task.

## Autoencoders Have an Analogous Problem

An autoencoder trained to minimize reconstruction error also preserves information useful for reconstructing inputs.

If cluster-relevant structure contributes little to reconstruction loss, the bottleneck can ignore it.

A nonlinear model may retain more flexible structure than PCA.

It does not automatically know which variation matters for clustering.

The broader principle is:

$$
\boxed{
\text{representation learning preserves what its training objective rewards}.
}
$$

## Clustering in PCA Space Changes the Metric

Suppose PCA retains matrix

$$
V_r
$$

with orthonormal columns.

Projected Euclidean distance is

$$
\|V_r^T(x_i-x_j)\|_2^2.
$$

This equals

$$
(x_i-x_j)^T
V_rV_r^T
(x_i-x_j).
$$

So clustering after PCA is equivalent to clustering with a metric that ignores all directions orthogonal to the retained subspace.

The discarded directions receive exactly zero weight.

If the cluster signal lives there, it is gone by design.

## The Number of PCs Is a Clustering Hyperparameter

Once PCA is placed before clustering, the retained dimension

$$
r
$$

is not merely a preprocessing parameter.

It is part of the clustering model.

Different $r$ values define different metrics and can produce different partitions.

Therefore sensitivity to

$$
r
$$

should be analysed just like sensitivity to:

- K-means cluster count,
- DBSCAN radius,
- graph neighbour count,
- or kernel bandwidth.

Choosing $r$ only from explained variance ignores its downstream role.

## Validate the Whole Pipeline

The object being evaluated is not PCA alone.

It is

$$
X
\xrightarrow{\text{preprocessing}}
X^\star
\xrightarrow{\text{PCA}_r}
Z
\xrightarrow{\mathcal A}
\widehat\Pi.
$$

So validation should perturb the whole pipeline.

For bootstrap stability, refit PCA inside each bootstrap.

Do not fit PCA once on the complete sample and then bootstrap only the clustering stage if the goal is to include representation uncertainty.

Otherwise the stability estimate conditions on a fixed projection that was itself estimated from the data.

## Compare Clustering With and Without PCA

A simple diagnostic is often overlooked.

Run the clustering:

1. in the original standardized or scientifically chosen representation,
2. after PCA at several retained dimensions.

Compare the partitions.

If the result changes dramatically, PCA is not a neutral compression step.

That disagreement is information.

It should be reported rather than resolved by automatically trusting the lower-dimensional version.

## Use Pairwise Agreement, Not Only Visual Similarity

For partitions

$$
\Pi_r
$$

obtained at different retained dimensions, use metrics such as:

- adjusted Rand index,
- variation of information,
- normalized mutual information,
- co-assignment matrices.

This quantifies sensitivity to dimension selection.

A pretty PCA scatter plot is not enough.

## Track Cluster Separation by Component

If candidate clusters are already available from a full-space method or from external structure, inspect how their separation distributes across PCs.

For component $j$, compare:

$$
\text{between-cluster variation}
$$

with

$$
\text{within-cluster variation}.
$$

A component with modest eigenvalue can still have a large separation ratio.

This is useful diagnostically.

It should not be used circularly to claim independent confirmation of clusters derived from the same data.

## Examine Discarded Components

The usual PCA workflow focuses on what is retained.

For clustering, the discarded subspace deserves attention too.

If a proposed clustering shows systematic differences in discarded PCs, then the projection may have removed relevant structure.

Conversely, if discarded directions look like unstructured measurement noise, dimension reduction is easier to defend.

Discarded variance is not automatically discarded noise.

## Simulation Can Test the Pipeline

When the application suggests plausible covariance and cluster structures, simulation is valuable.

Construct scenarios varying:

- within-cluster covariance,
- mean separation,
- number of nuisance dimensions,
- variance ratios,
- cluster imbalance,
- and measurement error.

Then compare cluster recovery before and after PCA.

This directly tests whether the proposed preprocessing is likely to preserve the kinds of structures the scientific problem cares about.

## Null Simulations Matter Too

Use a continuous no-cluster population with the same covariance structure.

If PCA plus clustering produces stable apparent groups there as well, that tells us how much structure the pipeline can manufacture through quantization.

This connects PCA validation with the earlier stability and silhouette arguments.

A pipeline should be tested not only where clusters exist but where they do not.

## Do Not Select PCs by the Best Cluster Score Alone

Suppose we try

$$
r=2,3,\ldots,20
$$

and choose the value with the highest silhouette score.

Now the same data have been used to:

- estimate the PCA basis,
- choose the retained dimension,
- fit the clustering,
- and evaluate the clustering.

The winning score is optimistically selected.

If the result will carry substantive meaning, this selection process should be accounted for through resampling, holdout structure, external validation or at least transparent sensitivity analysis.

## Clustering Can Help Choose a Representation, But That Is a Joint Model

There is nothing wrong with jointly selecting representation and clustering.

One might deliberately optimize

$$
(r,k,\text{metric})
$$

together.

The resulting object should then be described as a joint model-selection problem.

It should not be presented as though PCA independently discovered the correct low-dimensional space and clustering then discovered the groups.

The two choices interact.

## PCA Is Particularly Dangerous When Signal Is Subtle

The most serious failures occur when cluster differences are:

- low variance,
- sparse,
- local,
- nonlinear,
- confined to a small subset of variables,
- or overwhelmed by continuous nuisance variation.

These are exactly the situations in which a generic variance criterion can overlook the relevant structure.

If the expected cluster signal is subtle, dimension reduction deserves more scrutiny, not less.

## PCA Is Particularly Helpful When Noise Is Diffuse

Conversely, PCA can be excellent when:

- signal is concentrated in a dominant low-dimensional subspace,
- noise is spread across many weak directions,
- covariance estimation is stable,
- and the clustering concept is approximately Euclidean in the retained subspace.

This is why PCA often works well in practice.

Its success is not mysterious.

The variance and cluster objectives happen to align.

## Explained Variance Should Be Reported, But Not Used as the Only Argument

The cumulative explained variance ratio is still useful.

It tells us how much covariance energy the projection retains.

Report it.

Just do not let the statement

> the first five PCs explain 95 percent of variance

silently become

> the first five PCs preserve 95 percent of clustering structure.

Those are different quantities.

The first is computed.

The second needs evidence.

## A Better Validation Contract

Before clustering on PCA scores, I would check several things.

### 1. State the reason for PCA

Is the goal:

- denoising,
- computational reduction,
- visualization,
- regularization,
- or removal of collinearity?

Different goals imply different validation.

### 2. Justify scaling

Explain why raw covariance or standardized correlation is the appropriate geometry.

### 3. Vary retained dimension

Do not rely on one explained-variance threshold.

### 4. Refit PCA during resampling

Representation uncertainty is part of pipeline uncertainty.

### 5. Compare against clustering without PCA

The projection should demonstrate value.

### 6. Inspect discarded directions

Check whether discarded PCs contain structured differences.

### 7. Use external variables where available

A useful clustering should survive outside the internal variance objective.

### 8. Simulate plausible low-variance signals

Stress-test the exact failure mode PCA is most likely to miss.

## The Counterexample in One Line

The model is

$$
X_1\sim N(0,\sigma_n^2),
$$

$$
X_2=\mu Z+\varepsilon,
$$

with

$$
Z\in\{-1,+1\},
$$

$$
\varepsilon\sim N(0,\sigma_s^2).
$$

If

$$
\sigma_n^2
>
\mu^2+\sigma_s^2,
$$

then PC1 is

$$
X_1,
$$

which satisfies

$$
X_1\perp Z.
$$

Therefore,

$$
\boxed{
I(Z;\text{PC1})=0.
}
$$

PCA can retain the largest share of variance and discard the entire cluster signal.

## Conclusion

PCA is one of the most useful tools in multivariate analysis.

It is also one of the easiest to give a job it was never designed to do.

Its objective is variance preservation.

Clustering requires preservation of whatever geometry defines the groups of interest.

Those objectives can align.

They can also be orthogonal.

In the Gaussian counterexample, the first principal component explains about

$$
85.9\%
$$

of total variance and contains exactly zero information about cluster identity.

Meanwhile, the discarded coordinate permits essentially perfect classification.

So the correct principle is

$$
\boxed{
\text{high explained variance does not imply preserved cluster structure}.
}
$$

Use PCA before clustering when there is a reason to believe the relevant structure lies in the retained variance subspace.

Then test that belief.

Do not let a large percentage on a scree plot substitute for evidence that the clusters survived the projection.

## References

- Ding, C., & He, X. (2004). K-means clustering via principal component analysis. *Proceedings of the 21st International Conference on Machine Learning*, 29.
- Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374, 20150202. https://doi.org/10.1098/rsta.2015.0202
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
