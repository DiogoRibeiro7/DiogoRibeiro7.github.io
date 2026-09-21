---
permalink: '/machine-learning/unlabelled_data_does_not_identify_the_decision_boundary/'
title: 'Unlabelled Data Does Not Identify the Decision Boundary'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Unlabelled Data
- Statistical Learning
- Cluster Assumption
- Manifold Learning
- Identifiability
author_profile: false
seo_title: 'Why Unlabelled Data Needs Assumptions in Semi-Supervised Learning'
seo_description: 'Unlabelled data identifies the marginal distribution of features, not the label mechanism. Semi-supervised learning helps only when assumptions connect feature geometry to P(Y|X).'
excerpt: >-
  Unlabelled observations can estimate where the data live, how dense different
  regions are and which points are close. They do not, by themselves, identify
  which side of a decision boundary should receive which label. Semi-supervised
  learning works only when additional assumptions connect feature geometry to the
  label mechanism.
summary: >-
  A mathematical treatment of why semi-supervised learning requires structural
  assumptions. The article separates the marginal distribution P$X$ from the
  conditional distribution P(Y|X), shows why unlabelled data alone cannot identify
  a classifier, and develops the smoothness, cluster, low-density separation,
  manifold and generative assumptions as explicit bridges between the two.
keywords:
- semi-supervised learning
- unlabelled data
- identifiability
- cluster assumption
- manifold assumption
- low-density separation
- smoothness assumption
classes: wide
date: '2026-05-24'
why_this_exists: >-
  Semi-supervised learning is often justified with the phrase that unlabelled data
  contains additional information. That statement is incomplete. Unlabelled data
  contains information about the distribution of X. A classifier needs information
  about Y given X. The useful question is therefore which assumptions allow one to
  infer something about P(Y|X) from P$X$.
evidence: >-
  Classical semi-supervised learning theory, the standard smoothness, cluster and
  manifold assumptions, and later work on conditions under which unlabelled data
  can improve learning rates.
methodology: >-
  Start from the factorisation P(X,Y)=P(Y|X)P$X$, construct distributions with the
  same feature marginal but different label mechanisms, then examine which
  additional structural assumptions make the marginal informative for
  classification.
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

Suppose we have a small labelled sample

$$
\mathcal{D}_L
=
{(x_i,y_i)}_{i=1}^{n_L}
$$

and a much larger unlabelled sample

$$
\mathcal{D}_U
=
{x_j}_{j=1}^{n_U}.
$$

The usual intuition behind semi-supervised learning is that

$$
n_U \gg n_L
$$

should help because the unlabelled observations reveal more of the structure of the data.

That statement is true only after an important qualification.

The unlabelled sample tells us about the distribution of the features,

$$
P_X.
$$

A classifier, however, needs information about the conditional distribution

$$
P(Y\mid X).
$$

Those are different objects.

Without an assumption linking them, learning more about $P_X$ does not necessarily tell us anything about the decision rule.

This is the central statistical difficulty in semi-supervised learning.

## The Factorisation That Matters

For classification, the joint distribution can be written as

$$
P(X,Y)
=
P(Y\mid X)P_X(X).
$$

The Bayes classifier depends on the posterior class probabilities. In the binary case, define

$$
\eta(x)
=
P(Y=1\mid X=x).
$$

Under equal misclassification costs, the Bayes rule is

$$
f^\star(x)
=
\mathbb{1}
\left\{
\eta(x)>\frac{1}{2}
\right\}.
$$

An infinitely large unlabelled sample can, in principle, identify the marginal distribution

$$
P_X
$$

arbitrarily well.

But the decision boundary depends on

$$
\eta(x).
$$

Knowing $P_X$ exactly does not generally identify $eta$.

That is not a technical nuisance. It is an identifiability problem.

## The Same Feature Distribution Can Support Different Classifiers

Consider a one-dimensional feature distribution

$$
X\sim \operatorname{Unif}(-2,2).
$$

Now define two possible label mechanisms.

Under the first,

$$
\eta_1(x)
=
\mathbb{1}{x>0}.
$$

The optimal boundary lies at zero.

Under the second,

$$
\eta_2(x)
=
\mathbb{1}{|x|>1}.
$$

The optimal classifier has two boundaries, at $-1$ and $1$.

The unlabelled distribution is identical in both worlds:

$$
P_X^{(1)}=P_X^{(2)}.
$$

Even if we observed ten billion unlabelled values of $X$, they could not tell us whether the true label mechanism was $eta_1$ or $eta_2$.

The information required to distinguish the two is label information.

More generally, for a fixed marginal $P_X$, there are infinitely many possible conditional distributions

$$
P(Y\mid X)
$$

and therefore infinitely many possible decision boundaries.

This gives a useful principle:

$$
\boxed{
P_X
\text{ does not identify }
P(Y\mid X)
}
$$

without additional structure.

Semi-supervised learning becomes possible when we are willing to impose such structure.

## What the Unlabelled Sample Actually Gives Us

An unlabelled sample can tell us many useful things.

It can estimate where observations concentrate. It can reveal multimodality, local density, disconnected regions, manifolds, graph structure, covariance, feature correlations, neighbourhoods and rare regions of the input space.

In other words, it tells us about the geometry of

$$
\mathcal{X}
$$

under the observed distribution.

But geometry becomes useful for classification only if it is related to labels.

That relation is introduced through assumptions.

The major semi-supervised learning assumptions are not implementation details. They are the statistical mechanism by which information about $P_X$ is allowed to constrain $P$Y\mid X$$.

## The Smoothness Assumption

A common starting point is a smoothness assumption.

Informally:

> nearby observations in a sufficiently dense region are likely to have similar outputs.

For probabilistic classification, this means that if $x$ and $x'$ are close under a meaningful geometry, then

$$
\eta(x)
\approx
\eta(x').
$$

One could express a local version as

$$
|\eta(x)-\eta(x')|
\leq
L,d(x,x')
$$

for some metric $d$ and constant $L$, at least within regions where the assumption is expected to hold.

The important part is not the Lipschitz form itself. The important part is that feature-space proximity has been connected to label similarity.

Unlabelled data can now help because it reveals which observations are neighbours and how those neighbourhoods are connected.

Without smoothness, knowing that two observations are close tells us nothing about whether their labels should agree.

### When smoothness is plausible

Smoothness can be reasonable when small changes in measured variables correspond to small changes in the underlying phenomenon.

Examples include some physical systems, carefully constructed embeddings, and measurements where nearby points genuinely represent similar states.

### When smoothness fails

It can fail sharply near thresholds.

Suppose a decision is determined by

$$
Y
=
\mathbb{1}{x>c}.
$$

Two observations can be arbitrarily close,

$$
x=c-\varepsilon,
\qquad
x'=c+\varepsilon,
$$

while having opposite labels.

Smoothness can also fail if the metric itself is poor. Euclidean proximity in a representation dominated by nuisance variables need not imply semantic similarity.

The assumption is therefore about both the label mechanism and the representation.

## The Cluster Assumption

The cluster assumption is stronger.

Roughly:

> observations in the same high-density cluster are likely to share a class.

Imagine that

$$
P_X
$$

has two well-separated modes. If almost every observation in the left mode belongs to class zero and almost every observation in the right mode belongs to class one, then a large unlabelled sample becomes highly informative.

It can estimate the two clusters accurately. A very small number of labels may then be sufficient to attach class names to them.

This is the favourable picture often used to motivate semi-supervised learning.

Suppose

$$
P_X(x)
=
\frac{1}{2}
\mathcal{N}(-3,1)
+
\frac{1}{2}
\mathcal{N}(3,1).
$$

If the label mechanism is approximately

$$
Y=
\begin{cases}
0, & X \text{ belongs to the left component},\\
1, & X \text{ belongs to the right component},
\end{cases}
$$

then estimating the density structure from unlabelled data constrains the classifier strongly.

The unlabelled observations do not provide the class names. They reveal a partition that, under the assumption, is expected to align with them.

## The Same Clusters Can Be Irrelevant to the Labels

Now keep exactly the same marginal distribution

$$
P_X(x)
=
\frac{1}{2}
\mathcal{N}(-3,1)
+
\frac{1}{2}
\mathcal{N}(3,1),
$$

but change the label mechanism.

Suppose instead that

$$
P(Y=1\mid X=x)
$$

depends on whether $x$ lies near the centre or the tail of either Gaussian component.

The two density modes still exist.

The unlabelled sample will recover them.

But those modes no longer correspond to the class structure.

A semi-supervised method that forces the classifier to respect the two clusters can now move the decision rule in the wrong direction.

The geometry has been estimated correctly.

The assumption connecting geometry to labels was wrong.

That distinction matters because practitioners often diagnose this kind of failure as an algorithm problem. Sometimes it is not. The algorithm may be faithfully exploiting a structural assumption that the application does not satisfy.

## Low-Density Separation

The cluster assumption is closely related to the low-density separation principle.

It says that a good classification boundary should avoid regions where

$$
P_X(x)
$$

is large.

In a binary problem, one might prefer a decision surface

$$
\mathcal{B}
=
{x:\eta(x)=1/2}
$$

that passes through low-density regions of the feature distribution.

This is attractive because dense regions represent many observations. Cutting directly through them assigns very similar and frequently occurring observations to different classes.

Support vector methods with semi-supervised objectives, entropy minimisation and several graph-based approaches exploit versions of this idea.

The assumption becomes useful when the class boundary is aligned with valleys in

$$
P_X.
$$

It fails when the true decision boundary cuts through a dense region.

Nothing in probability theory forbids that situation.

## A Simple Counterexample to Low-Density Separation

Let

$$
X\sim \mathcal{N}(0,1)
$$

and define

$$
Y
=
\mathbb{1}{X>0}.
$$

The Bayes boundary is

$$
x=0.
$$

But zero is also the highest-density point of the feature distribution.

Any method with a strong preference for moving boundaries toward low-density regions is being pushed away from the true classifier.

The unlabelled sample becomes more informative about the wrong objective as its size increases.

This is a useful reminder:

$$
\boxed{
\text{more unlabelled data strengthens both correct and incorrect assumptions}
}
$$

Large $n_U$ does not rescue a misspecified connection between $P_X$ and $P$Y\mid X$$.

It can make the consequences of the misspecification more stable.

## The Manifold Assumption

High-dimensional observations often occupy only a small part of the ambient space.

A data vector may live in

$$
\mathbb{R}^{1000}
$$

while the effective degrees of freedom are much smaller.

The manifold assumption says, roughly, that the feature distribution is concentrated near a lower-dimensional manifold

$$
\mathcal{M}
\subset
\mathbb{R}^p,
$$

and that the label function varies smoothly along that manifold.

Distances measured through the ambient space may then be misleading. What matters is distance along

$$
\mathcal{M}.
$$

This motivates graph-based semi-supervised methods.

Construct a graph with weights

$$
w_{ij}
=
K(x_i,x_j),
$$

where $K$ is a similarity kernel. Labels can then be propagated under the assumption that neighbouring vertices on the data graph should have similar predictions.

A common smoothness penalty has the form

$$
\sum_{i,j}
w_{ij}
\left(
f(x_i)-f(x_j)
\right)^2.
$$

In matrix form, this becomes a graph-Laplacian regulariser,

$$
f^T L f,
$$

where

$$
L=D-W.
$$

The unlabelled data define the graph.

The labels determine how class information is anchored to it.

Again, the method works because an assumption links geometry to the target.

## Manifolds Are Not Automatically Label Manifolds

The fact that data lie near a low-dimensional manifold does not imply that labels vary smoothly along it.

Imagine observations lying on a circle.

The feature geometry may be essentially one-dimensional, parameterised by an angle

$$
\theta.
$$

If the label changes once around the circle, smoothness along the manifold may be a sensible prior.

But labels could alternate rapidly with angle. They could depend on another latent variable not represented in the observed geometry. They could be noisy near particular regions.

The manifold can be real and estimated correctly while still being unhelpful for the classification problem.

This is the recurring pattern:

$$
\text{structure in }P_X

ot\Rightarrow
\text{label-relevant structure}.
$$

Semi-supervised learning requires the implication to be approximately valid.

## Generative Assumptions Create Another Bridge

A different route is to assume a generative model.

Suppose

$$
P(X,Y)
=
P(Y)P(X\mid Y;\theta).
$$

If the model family is correctly specified, unlabelled observations can help estimate parameters of the marginal mixture

$$
P_X(x)
=
\sum_y
P(Y=y)P(x\mid Y=y;\theta).
$$

Those parameters may also determine the class-conditional distributions used by the classifier.

This gives unlabelled data a route to influence classification.

For example, in a Gaussian mixture classifier, unlabelled observations can improve estimation of mixture component locations and covariances when labelled data are scarce.

But generative semi-supervised learning can be brittle under misspecification.

If the assumed class-conditional family is wrong, fitting the abundant unlabelled data can pull the estimated parameters toward a model that represents

$$
P_X
$$

well but represents

$$
P(Y\mid X)
$$

poorly.

A large unlabelled sample can therefore increase confidence in a badly specified generative model.

## Why This Is an Identifiability Problem

The core issue can be stated more formally.

Suppose two joint distributions,

$$
P_1(X,Y)
$$

and

$$
P_2(X,Y),
$$

have the same marginal feature distribution,

$$
P_{1,X}=P_{2,X},
$$

but different conditional label distributions,

$$
P_1(Y\mid X)

eq
P_2(Y\mid X).
$$

Then no procedure observing only additional unlabelled draws from $P_X$ can distinguish those two worlds.

The unlabelled likelihood is identical.

To benefit from unlabelled data, the model class must restrict which conditional distributions are compatible with the observed marginal.

Semi-supervised assumptions perform exactly that restriction.

This is why they are not optional philosophical decorations. They provide the identifying information.

## A Bayesian View Makes the Same Point

Suppose the classifier depends on parameters

$$
\theta.
$$

With labelled data only, the posterior is

$$
p(\theta\mid X_L,Y_L)
\propto
p(Y_L\mid X_L,\theta)
p(X_L\mid\theta)
p(\theta).
$$

If the discriminative model is specified only through

$$
p(Y\mid X,\theta),
$$

then unlabelled observations contribute nothing unless the model also specifies how

$$
X
$$

depends on $\theta$.

The unlabelled likelihood needs a term such as

$$
p(X_U\mid\theta).
$$

If the parameters controlling the feature distribution are unrelated to the parameters controlling the conditional labels, then learning the first does not improve inference about the second.

The same principle appears in frequentist language as well. The notation changes, but the information structure does not.

## Why More Unlabelled Data Can Hurt

If the structural assumption is correct, increasing

$$
n_U
$$

can improve estimation of the geometry or density that constrains the classifier.

If the assumption is wrong, increasing

$$
n_U
$$

can make the wrong constraint increasingly dominant.

This is one reason negative transfer is possible.

In a previous article, I constructed a paired experiment where the labelled sample and test distribution were held fixed while only the unlabelled covariates were shifted. The semi-supervised gain changed sign even though the supervised problem had not changed.

The relevant point here is deeper than covariate shift itself.

The algorithm treated the unlabelled geometry as informative about the classification boundary. Once that relation deteriorated, more information about the geometry was no longer equivalent to more information about the labels.

The article is available at:

[When Unlabelled Data Makes Semi-Supervised Learning Worse](/machine-learning/when_unlabelled_data_makes_semi_supervised_learning_worse/)

## Learning Rates Do Not Improve for Free

Theoretical work on semi-supervised learning makes the same distinction.

If unlabelled data impose no additional restrictions on the class of possible label mechanisms, then there is no general reason for the labelled-sample learning rate to improve.

Results showing that unlabelled observations can improve learning rates necessarily rely on assumptions that make

$$
P_X
$$

informative about the target.

Göpfert and co-authors studied conditions under which such rate improvements can be established. Their analysis is useful because it makes explicit that improvement from unlabelled data is a property of a problem class, not an automatic consequence of sample size.

The practical interpretation is straightforward.

Before asking whether

$$
n_U
$$

is large enough, ask whether the data-generating assumptions make

$$
n_U
$$

relevant.

## The Assumptions Are About the Problem, Not the Algorithm

It is common to classify semi-supervised methods into families:

- self-training,
- pseudo-labelling,
- graph-based methods,
- consistency regularisation,
- entropy minimisation,
- generative models,
- co-training,
- manifold regularisation.

That taxonomy is useful, but it can hide the more important distinction.

Two algorithms from different families may depend on closely related structural assumptions.

For example, a graph method and a consistency-regularised neural network may both encourage nearby observations to have similar predictions. A transductive margin method and entropy minimisation may both prefer boundaries in low-density regions.

The mathematical form changes.

The assumption can remain essentially the same.

A useful analysis should therefore document not only the algorithm but the structural claim it makes about the data.

## What Should Be Tested Before Using Semi-Supervised Learning?

The assumptions cannot usually be verified completely from the same unlabelled data used by the method. They can, however, be stress-tested.

### 1. Compare with the supervised baseline

The supervised model is not merely a weaker competitor.

It is the control condition that tells us whether the unlabelled information added value.

Report

$$
\Delta
=
R_{\text{sup}}
-
R_{\text{SSL}}
$$

or the corresponding performance difference using a metric appropriate to the task.

If the difference is small, uncertainty around it matters.

### 2. Vary the labelled fraction

Evaluate the method for several values of

$$
n_L.
$$

A method that helps only at one extremely small labelled fraction may be exploiting a fragile regime.

The shape of the gain curve matters more than one point estimate.

### 3. Perturb the unlabelled distribution

Apply controlled changes to the unlabelled pool while keeping the labelled evaluation problem fixed.

Examples include:

- covariate shifts,
- contamination,
- class-prior changes,
- additional nuisance dimensions,
- altered acquisition periods,
- subgroup over-representation.

The goal is not to simulate every possible production failure. It is to see whether the semi-supervised gain is robust to plausible violations of the assumed relationship.

### 4. Examine geometry against labels where labels are available

On the labelled subset, ask whether nearest neighbours tend to share labels, whether estimated density valleys correspond to class boundaries, or whether graph communities align at least partially with the target.

This does not prove the assumption globally, but it can falsify very poor choices.

### 5. Compare alternative representations

If semi-supervised gains disappear when the representation changes modestly, the method may be exploiting accidental geometry.

This is particularly important for learned embeddings, where the representation itself can encode strong inductive biases.

### 6. Retain negative results

Do not average away seeds or subgroups where semi-supervised learning performs worse.

A mean gain of

$$
+0.5%
$$

can conceal a method that improves half the time and causes substantial harm in the other half.

Report the distribution of gains.

## A Better Question Than "Can We Use the Unlabelled Data?"

The availability of unlabelled observations often creates pressure to use them.

That reverses the logic.

The correct starting point is not

> We have ten million unlabelled observations. Which semi-supervised method should we use?

It is

> What relationship do we believe exists between the geometry of the features and the label mechanism?

Only after answering that question does algorithm choice become meaningful.

If the answer is smoothness, a graph or consistency-based method may be appropriate.

If the answer is low-density separation, margin-based methods may be reasonable.

If a trusted class-conditional generative model exists, the unlabelled likelihood may genuinely improve parameter estimation.

If no credible connection can be articulated, the disciplined option may be to leave the unlabelled sample out of the classifier and use it for other purposes: representation diagnostics, drift monitoring, anomaly detection, data-quality analysis or active-learning candidate selection.

Unused data is not automatically wasted data.

## Conclusion

Unlabelled data can be extremely valuable, but its value in classification is conditional.

It informs

$$
P_X.
$$

The classifier depends on

$$
P(Y\mid X).
$$

The gap between those two objects is closed by assumptions.

Smoothness says nearby observations should have similar outputs. The cluster and low-density assumptions say class boundaries should respect density structure. The manifold assumption says the relevant geometry is lower-dimensional and labels vary smoothly along it. Generative approaches impose a joint model that couples the feature distribution to class membership.

Each of these assumptions can make unlabelled data useful.

Each can also fail.

The central question in semi-supervised learning is therefore not whether unlabelled data contains information.

Of course it does.

The question is whether it contains information about the particular conditional distribution we are trying to learn.

Formally,

$$
\boxed{
\text{unlabelled data helps classification only through assumptions linking }
P_X
\text{ to }
P(Y\mid X)
}
$$

Once that statement is made explicit, semi-supervised learning becomes easier to reason about.

The method is no longer a way to obtain supervision for free.

It is a way to trade labelled information for structural assumptions.

## References

- Chapelle, O., Schölkopf, B., & Zien, A. (Eds.). (2006). *Semi-Supervised Learning*. MIT Press. https://doi.org/10.7551/mitpress/9780262033589.001.0001
- Göpfert, C., Ben-David, S., Bousquet, O., Gelly, S., Tolstikhin, I., & Urner, R. (2019). When can unlabeled data improve the learning rate? *Proceedings of the Thirty-Second Conference on Learning Theory*, PMLR 99, 1500–1518.
- van Engelen, J. E., & Hoos, H. H. (2020). A survey on semi-supervised learning. *Machine Learning*, 109, 373–440. https://doi.org/10.1007/s10994-019-05855-6
- Zhu, X. (2005). *Semi-Supervised Learning Literature Survey*. Computer Sciences Technical Report 1530, University of Wisconsin-Madison.
