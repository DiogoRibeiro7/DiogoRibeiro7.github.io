---
permalink: '/mathematics/wasserstein_distance_is_geometry_not_just_another_divergence/'
title: 'Wasserstein Distance Is Geometry, Not Just Another Divergence'
date: '2026-01-15'
categories:
- Mathematics
tags:
- Optimal Transport
- Wasserstein Distance
- Probability
- Numerical Methods
- Statistical Modeling
author_profile: false
classes: wide
seo_title: 'Wasserstein Distance Is Geometry, Not Just Another Divergence'
seo_description: 'Optimal transport compares probability distributions by moving mass through the geometry of the underlying space. Wasserstein distance therefore behaves very differently from KL divergence, especially under support mismatch.'
seo_type: article
excerpt: >-
  Wasserstein distance does not compare probability densities point by point.
  It asks how much probability mass must move, how far it must travel, and what
  transport cost is required to transform one distribution into another.
summary: >-
  This article develops Wasserstein distance from Monge and Kantorovich optimal
  transport. It derives the discrete transport problem, the one-dimensional
  quantile formula, the closed form for Gaussian W2 distance, and the
  Kantorovich-Rubinstein dual representation of W1. Exact support-mismatch
  examples show why KL divergence can become infinite while Wasserstein distance
  remains finite. The article then develops displacement interpolation,
  Wasserstein barycentres, entropic regularisation and Sinkhorn iterations, and
  explains the statistical limits of empirical optimal transport in high
  dimension.
keywords:
- optimal transport
- Wasserstein distance
- Kantorovich
- Sinkhorn
- transport plan
- probability geometry
why_this_exists: >-
  Distribution distances are often presented as interchangeable ways to compare
  probability laws. They are not. Wasserstein distance incorporates the geometry
  of the sample space, which makes nearby support mismatch cheap and distant
  mismatch expensive rather than merely impossible.
evidence: >-
  Classical Monge-Kantorovich transport theory, one-dimensional optimal
  rearrangement, Gaussian W2 formulas, Kantorovich-Rubinstein duality, entropic
  optimal transport and standard statistical results on empirical Wasserstein
  convergence.
methodology: >-
  Start from finite discrete distributions and the transportation linear
  programme, pass to the Kantorovich formulation, derive Wp and the one-
  dimensional quantile representation, then compare Wasserstein and KL
  geometries on exact examples before developing interpolation, barycentres and
  computational regularisation.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/kernel_math.webp
  og_image: /assets/images/kernel_math.webp
  overlay_image: /assets/images/kernel_math.webp
  show_overlay_excerpt: false
  teaser: /assets/images/kernel_math.webp
  twitter_image: /assets/images/kernel_math.webp
---

<!--
Development contract
Question: What does Wasserstein distance measure that density-based divergences do not?
Claim: Wasserstein distance compares probability laws through the geometry of mass transport. It remains meaningful under support mismatch, induces displacement rather than mixture interpolation, and turns probability distributions into a metric space with a useful geometric structure.
Counterclaim: Wasserstein methods are not universally superior. They depend on a meaningful ground metric, can be statistically expensive in high dimensions, and entropic approximations change the geometry unless their regularisation is handled carefully.
Evidence object: Disjoint-support point masses, exact 1D quantile transport, closed-form Gaussian W2 distance, discrete transport linear programme and entropic regularisation.
Failure case: Using Euclidean transport cost when the sample-space geometry is scientifically meaningless, treating an entropically regularised objective as exact Wasserstein distance, or assuming empirical Wasserstein estimates avoid the curse of dimensionality.
Reader payoff: Understand when transport geometry is the correct notion of distributional difference and how the choice of ground metric, transport cost and regularisation changes the problem.
Exclusions: A catalogue of every optimal-transport algorithm, a GAN tutorial, and repetition of the older introductory KL-versus-Wasserstein comparison.
-->

Two probability distributions can be nearly identical in shape and still have disjoint support. Shift a narrow distribution by a tiny distance and every point moves only slightly, yet a density-ratio divergence may become undefined or infinite because one distribution assigns positive probability where the other assigns zero. Whether that should count as an enormous difference depends on the question.

Optimal transport answers a different question from information-theoretic divergence. Instead of comparing the amount of probability assigned to the same location, it asks how probability mass can be rearranged from one distribution into another and how much that rearrangement costs. A unit of mass moved one millimetre is cheaper than a unit moved one kilometre if the underlying metric says those distances are different.

That simple change introduces geometry into probability.

Wasserstein distance is therefore not merely another score to place beside Kullback-Leibler divergence, Jensen-Shannon divergence or total variation. It lives on a different idea of proximity. A probability distribution is treated as a configuration of mass over a metric space, and two distributions are close when there exists a low-cost transport plan moving one mass configuration into the other.

The resulting geometry is rich enough to define shortest paths between probability distributions, averages of distributions, gradient flows on spaces of measures and robust optimization neighbourhoods. It is also computationally demanding and statistically delicate in high dimension.

The right way to understand Wasserstein distance is to begin with the transport problem itself.

## From moving piles of mass to a linear programme

Consider two discrete probability distributions,

$$
\mu
=
\sum_{i=1}^{m}
a_i\delta_{x_i},
$$

and

$$
\nu
=
\sum_{j=1}^{n}
b_j\delta_{y_j},
$$

where

$$
a_i\ge0,
\qquad
b_j\ge0,
$$

and

$$
\sum_i a_i
=
\sum_j b_j
=
1.
$$

Suppose moving one unit of probability mass from location $x_i$ to location $y_j$ costs

$$
c_{ij}
=
c(x_i,y_j).
$$

A transport plan is a nonnegative matrix

$$
\Gamma
=
(\gamma_{ij})
$$

where

$$
\gamma_{ij}
$$

is the amount of mass moved from $x_i$ to $y_j$.

The plan must exhaust the source mass,

$$
\sum_j
\gamma_{ij}
=
a_i,
$$

and produce the destination mass,

$$
\sum_i
\gamma_{ij}
=
b_j.
$$

The optimal transport problem is

$$
\min_{\Gamma\ge0}
\sum_{i=1}^{m}
\sum_{j=1}^{n}
c_{ij}\gamma_{ij}
$$

subject to those marginal constraints.

This is a linear programme.

The important object is not only the minimum cost. The optimizer

$$
\Gamma^\star
$$

says how much mass should move between every source and destination.

Take

$$
\mu
=
\frac12\delta_0
+
\frac12\delta_2
$$

and

$$
\nu
=
\frac12\delta_1
+
\frac12\delta_3.
$$

With one-dimensional cost

$$
c(x,y)=|x-y|,
$$

the obvious optimal plan moves the mass at zero to one and the mass at two to three. The total cost is

$$
\frac12(1)
+
\frac12(1)
=
1.
$$

Crossing the assignments would move zero to three and two to one, costing

$$
\frac12(3)
+
\frac12(1)
=
2.
$$

The optimal plan is therefore shaped by the geometry of the support.

If the labels $0,1,2,3$ had no meaningful metric interpretation, the transport distance would be meaningless. Optimal transport is only as scientifically sensible as the ground cost.

This dependence on the underlying geometry is a feature, not a bug.

## Monge's map and Kantorovich's relaxation

The original transport formulation associated with Monge asks for a deterministic map

$$
T:\mathcal X\to\mathcal Y
$$

that pushes one distribution into another,

$$
T_\#\mu
=
\nu,
$$

while minimizing

$$
\int
c(x,T(x))
\,d\mu(x).
$$

The pushforward notation

$$
T_\#\mu
=
\nu
$$

means that if

$$
X\sim\mu,
$$

then

$$
T(X)\sim\nu.
$$

A deterministic map can fail to exist. One source location containing positive mass may need to split that mass across several destinations.

Kantorovich's formulation allows exactly this.

Instead of a map, choose a joint probability distribution

$$
\gamma
$$

on source-destination pairs

$$
(x,y)
$$

whose first marginal is

$$
\mu
$$

and second marginal is

$$
\nu.
$$

The collection of such couplings is written

$$
\Pi(\mu,\nu).
$$

The Kantorovich problem is

$$
\inf_{
\gamma\in\Pi(\mu,\nu)
}
\int
c(x,y)
\,d\gamma(x,y).
$$

The coupling can split mass.

This relaxation is one of the decisive ideas in modern optimal transport. It converts a difficult nonlinear map problem into a convex optimization problem over probability measures.

For suitable costs and regularity conditions, the optimal Kantorovich coupling can still be concentrated on a deterministic transport map. But the theory no longer requires such a map to exist in advance.

## Wasserstein distance is transport cost with a metric power

Let the sample space carry a metric

$$
d(x,y).
$$

For

$$
p\ge1,
$$

define the transport cost

$$
c(x,y)
=
d(x,y)^p.
$$

The $p$-Wasserstein distance is

$$
W_p(\mu,\nu)
=
\left[
\inf_{
\gamma\in\Pi(\mu,\nu)
}
\int
d(x,y)^p
\,d\gamma(x,y)
\right]^{1/p}.
$$

When both distributions have finite $p$th moments, this defines a metric on the corresponding Wasserstein space.

The word metric matters. We have

$$
W_p(\mu,\nu)\ge0,
$$

symmetry,

$$
W_p(\mu,\nu)
=
W_p(\nu,\mu),
$$

identity of indiscernibles,

$$
W_p(\mu,\nu)=0
\iff
\mu=\nu,
$$

and the triangle inequality.

The geometry of the sample space is inherited by the space of probability measures.

If the ground metric is Euclidean,

$$
d(x,y)
=
\|x-y\|_2,
$$

then $W_2$ penalizes squared Euclidean displacement before taking the square root.

If the ground metric encodes graph distance, geodesic distance, transportation cost or another scientifically meaningful geometry, the Wasserstein metric changes accordingly.

There is no universal Wasserstein distance detached from a choice of ground geometry.

## Support mismatch reveals the difference from KL divergence immediately

Take the simplest possible distributions,

$$
\mu
=
\delta_0
$$

and

$$
\nu
=
\delta_a.
$$

Every unit of mass must move from zero to $a$, so

$$
W_p(\mu,\nu)
=
|a|
$$

for every

$$
p\ge1.
$$

If

$$
a
$$

is small, the distributions are close in Wasserstein distance.

Now consider Kullback-Leibler divergence.

For

$$
a\ne0,
$$

the point masses have disjoint support. Neither distribution is absolutely continuous with respect to the other. Therefore,

$$
D_{\mathrm{KL}}(\mu\|\nu)
=
\infty
$$

and

$$
D_{\mathrm{KL}}(\nu\|\mu)
=
\infty.
$$

KL sees an impossible density ratio.

Wasserstein sees a finite displacement.

Neither answer is "more correct" without context.

If the question is information loss under one probability model used in place of another, support mismatch is catastrophic and KL's infinity is meaningful.

If the question is how far a probability distribution moved in physical space, the magnitude of $a$ matters and Wasserstein geometry is natural.

This is why transport distance became attractive in generative modelling. Two empirical distributions supported on nearby low-dimensional sets can have almost no overlap. Density-ratio divergences can saturate or become undefined, while transport cost changes continuously as the supports approach one another.

The same property is useful in distribution shift. If a sensor distribution translates by one degree, one metre or one unit of concentration, a geometric metric can express the size of that change directly.

## In one dimension, optimal transport is monotone matching

One-dimensional optimal transport has an exceptionally clean form.

Let

$$
F
$$

and

$$
G
$$

be the CDFs of

$$
\mu
$$

and

$$
\nu.
$$

Define generalized quantile functions

$$
F^{-1}(u)
=
\inf
\{
x:F(x)\ge u
\},
$$

and similarly for

$$
G^{-1}.
$$

Then

$$
W_p^p(\mu,\nu)
=
\int_0^1
\left|
F^{-1}(u)
-
G^{-1}(u)
\right|^p
du.
$$

The optimal coupling matches equal quantile levels.

The smallest source observation moves to the smallest destination observation, the next smallest to the next smallest, and so on.

This is a form of monotone rearrangement.

The result gives both intuition and computation. For empirical distributions with equal sample sizes, sort both samples and average pairwise powered distances between corresponding order statistics.

Suppose

$$
X\sim U(0,1)
$$

and

$$
Y\sim U(a,a+1).
$$

Their quantile functions are

$$
F^{-1}(u)=u,
$$

$$
G^{-1}(u)=a+u.
$$

Therefore,

$$
W_p^p
=
\int_0^1
|a|^pdu
=
|a|^p,
$$

and

$$
W_p
=
|a|.
$$

The entire distribution has translated rigidly by $a$, and Wasserstein distance returns exactly the translation magnitude.

This is what geometric faithfulness looks like.

## Gaussian distributions have a closed form under W2

For multivariate Gaussian distributions,

$$
\mu
=
N(m_1,C_1),
$$

and

$$
\nu
=
N(m_2,C_2),
$$

the squared 2-Wasserstein distance has the closed form

$$
W_2^2(\mu,\nu)
=
\|m_1-m_2\|_2^2
+
\operatorname{tr}
\left[
C_1
+
C_2
-
2
\left(
C_2^{1/2}
C_1
C_2^{1/2}
\right)^{1/2}
\right].
$$

The first term compares means geometrically.

The second compares covariance structure.

In one dimension,

$$
C_1=\sigma_1^2,
\qquad
C_2=\sigma_2^2,
$$

and the formula reduces to

$$
W_2^2
=
(\mu_1-\mu_2)^2
+
(\sigma_1-\sigma_2)^2.
$$

Consider

$$
N(0,1)
$$

and

$$
N(2,4).
$$

Then

$$
W_2^2
=
(0-2)^2
+
(1-2)^2
=
5,
$$

so

$$
W_2
=
\sqrt5
\approx
2.236.
$$

The mean shift and standard-deviation shift contribute through Euclidean geometry in the

$$
(\mu,\sigma)
$$

plane.

KL divergence behaves differently.

For univariate normals,

$$
D_{\mathrm{KL}}
\left[
N(\mu_1,\sigma_1^2)
\|
N(\mu_2,\sigma_2^2)
\right]
=
\frac12
\left[
\log
\frac{
\sigma_2^2
}{
\sigma_1^2
}
+
\frac{
\sigma_1^2
+
(\mu_1-\mu_2)^2
}{
\sigma_2^2
}
-
1
\right].
$$

For the same pair,

$$
D_{\mathrm{KL}}
\left[
N(0,1)
\|
N(2,4)
\right]
\approx
0.818,
$$

while reversing the order gives approximately

$$
2.807.
$$

The numerical values should not be compared directly to

$$
W_2
$$

because the objects have different units and meanings.

The useful contrast is structural.

Wasserstein is symmetric and geometric.

KL is asymmetric and density-relative.

## W1 has a dual form that turns transport into a function optimization problem

The primal transport problem optimizes over couplings.

For

$$
W_1,
$$

Kantorovich-Rubinstein duality gives a remarkably different representation:

$$
W_1(\mu,\nu)
=
\sup_{
\|f\|_{\mathrm{Lip}}\le1
}
\left[
E_\mu f(X)
-
E_\nu f(Y)
\right].
$$

The supremum is over all 1-Lipschitz functions,

$$
|f(x)-f(y)|
\le
d(x,y).
$$

This dual form says that

$$
W_1
$$

is the largest difference in expectation detectable by a function that cannot change faster than the underlying geometry permits.

This connects optimal transport to integral probability metrics.

The coupling formulation asks how mass moves.

The dual formulation asks which smooth-enough critic best distinguishes the distributions.

The two answers are equal.

This duality is one reason Wasserstein ideas became influential in generative modelling. Instead of solving a full transport plan in high dimension, one can try to optimize a parameterized critic subject to an approximate Lipschitz constraint.

The computational implementation can be imperfect while the mathematical dual remains exact.

The distinction should not be blurred.

## Mixture interpolation and transport interpolation are fundamentally different

Suppose two point masses are

$$
\mu_0
=
\delta_0
$$

and

$$
\mu_1
=
\delta_{10}.
$$

A conventional convex mixture is

$$
\mu_t^{\mathrm{mix}}
=
(1-t)\delta_0
+
t\delta_{10}.
$$

At

$$
t=\frac12,
$$

the distribution is

$$
\frac12\delta_0
+
\frac12\delta_{10}.
$$

Half the mass remains at the starting point and half has appeared at the destination.

Nothing occupies the space between them.

Wasserstein geometry produces a different path.

The optimal transport map sends

$$
0\mapsto10.
$$

Displacement interpolation moves the point continuously,

$$
T_t(x)
=
(1-t)x+tT(x).
$$

Therefore,

$$
\mu_t
=
(T_t)_\#\mu_0
=
\delta_{10t}.
$$

At

$$
t=\frac12,
$$

the distribution is

$$
\delta_5.
$$

The mass has moved halfway.

This is one of the most important geometric differences between Wasserstein space and ordinary linear mixture space.

A mixture interpolates probability weights.

A Wasserstein geodesic transports locations.

For distributions representing shapes, images, spatial densities or physical populations, displacement interpolation can preserve structure in a way that linear mixing does not.

For other problems, mixture interpolation may be exactly the correct operation.

Again, geometry should follow meaning.

## Wasserstein barycentres average distributions through transport

A scalar average minimizes squared Euclidean distance:

$$
\bar x
=
\arg\min_z
\sum_i
w_i
|z-x_i|^2.
$$

A Wasserstein barycentre generalizes this idea to probability distributions:

$$
\bar\mu
=
\arg\min_\mu
\sum_i
w_i
W_2^2(\mu,\mu_i),
$$

with

$$
w_i\ge0,
\qquad
\sum_i w_i=1.
$$

This defines an average in Wasserstein geometry.

For translated point masses,

$$
\mu_i
=
\delta_{x_i},
$$

the barycentre reduces to the ordinary weighted mean point mass,

$$
\bar\mu
=
\delta_{
\sum_i w_i x_i
}.
$$

For richer distributions, the barycentre can align and transport structures rather than simply overlaying them.

Consider two narrow unimodal distributions centred at different locations.

A density mixture can become bimodal.

A Wasserstein barycentre can remain unimodal at an intermediate location.

Whether that is desirable depends on what "average distribution" is intended to mean.

If two populations genuinely coexist, mixture is natural.

If one object has been translated or deformed relative to another, transport barycentres can be much more faithful.

The word average is therefore geometry-dependent.

## Wasserstein space supports gradient-flow interpretations

The space of probability measures equipped with

$$
W_2
$$

has enough geometric structure to interpret certain evolution equations as gradient flows.

One celebrated example is the heat equation,

$$
\frac{
\partial\rho
}{
\partial t
}
=
\Delta\rho.
$$

In the Jordan-Kinderlehrer-Otto formulation, diffusion can be interpreted as steepest descent of entropy in Wasserstein space.

A time-discretized step can be written schematically as

$$
\rho_{k+1}
=
\arg\min_\rho
\left[
\frac{
1
}{
2\tau
}
W_2^2(\rho,\rho_k)
+
\mathcal F(\rho)
\right],
$$

where

$$
\mathcal F
$$

is an energy functional.

The first term penalizes moving too far from the previous distribution.

The second lowers the energy.

This variational structure connects optimal transport to PDEs, diffusion, Fokker-Planck equations and statistical mechanics.

It also reveals why Wasserstein geometry is more than a distance formula.

Probability distributions form a nonlinear metric space in which dynamics can be described geometrically.

One should not overextend the analogy with finite-dimensional Riemannian geometry, but it is mathematically productive and has generated a large theory.

## Exact optimal transport can be computationally expensive

For discrete distributions with

$$
n
$$

source points and

$$
m
$$

destination points, the transport plan contains

$$
nm
$$

variables.

Large dense problems can therefore become expensive.

Entropic regularization changes the optimization problem to

$$
\min_{
\Gamma\in\Pi(a,b)
}
\left[
\langle C,\Gamma\rangle
+
\varepsilon
\sum_{ij}
\gamma_{ij}
(\log\gamma_{ij}-1)
\right].
$$

The entropy term encourages diffuse transport plans.

The regularized optimum can be computed efficiently through Sinkhorn iterations because the solution has a multiplicative scaling structure.

If

$$
K_{ij}
=
\exp
\left(
-\frac{
C_{ij}
}{
\varepsilon
}
\right),
$$

the regularized coupling can be written in the form

$$
\Gamma^\star
=
\operatorname{diag}(u)
K
\operatorname{diag}(v),
$$

with scaling vectors

$$
u
$$

and

$$
v
$$

adjusted until the marginal constraints are satisfied.

This makes large transport problems dramatically more tractable.

It also changes the problem.

As

$$
\varepsilon
$$

increases, the solution becomes smoother and more diffuse. The regularized transport cost is biased relative to exact optimal transport.

The entropy term is therefore both a computational device and a modelling perturbation.

Sinkhorn divergences correct part of the entropic self-bias by combining regularized cross-costs and self-costs, but they still define a family of geometries depending on the regularization scale.

A numerical approximation parameter has become part of the statistical object.

That should be acknowledged when results are sensitive to

$$
\varepsilon.
$$

## Statistical estimation of Wasserstein distance can be hard in high dimension

Optimal transport has attractive geometric behaviour and an important statistical limitation.

Suppose

$$
\hat\mu_n
$$

is the empirical distribution from

$$
n
$$

iid samples of a distribution

$$
\mu.
$$

The empirical Wasserstein distance

$$
W_p(\hat\mu_n,\mu)
$$

can converge slowly as dimension increases.

Under common regularity conditions, prototypical rates in sufficiently high dimension scale like

$$
n^{-1/d}
$$

rather than the dimension-free

$$
n^{-1/2}
$$

rate associated with many scalar averages.

The exact rate depends on

$$
p,
$$

dimension, moment assumptions and regularity, so

$$
n^{-1/d}
$$

should be read as the characteristic curse-of-dimensionality regime rather than a universal formula.

The practical point is clear.

Wasserstein distance can be geometrically meaningful and statistically expensive.

This becomes severe when estimating high-dimensional distributions from finite samples. Two empirical point clouds can have substantial Wasserstein distance partly because finite samples do not cover high-dimensional space densely.

Regularized transport, sliced Wasserstein distances, projection-based methods and structural assumptions can reduce computation or statistical difficulty.

A sliced Wasserstein distance projects distributions onto one-dimensional directions,

$$
\theta^\top X,
$$

computes one-dimensional Wasserstein distances there, and averages over directions.

The one-dimensional quantile formula makes those projected distances cheap.

Projection simplifies the geometry and can discard multivariate structure.

Again there is a trade-off.

## The ground metric is part of the scientific model

Suppose two categorical outcomes are encoded as integers

$$
1,2,3,4.
$$

Applying Euclidean Wasserstein distance implies that moving probability from category one to category two costs one unit, while moving from one to four costs three.

That assumption may be meaningful for ordered severity levels.

It is nonsensical for arbitrary labels such as blood type or country code.

The same issue appears in high-dimensional continuous data.

Euclidean pixel distance between images may treat a one-pixel translation as a large change despite strong perceptual similarity.

Euclidean distance between raw sensor vectors can overemphasize high-variance dimensions.

A graph metric may be more appropriate for network locations.

Geodesic distance may be required on a manifold.

A learned metric can be useful when domain similarity is not captured by raw coordinates, but then the transport result inherits the assumptions and errors of the learned representation.

Optimal transport never removes the need to define what distance means.

It amplifies that modelling choice by building the entire probability metric from it.

## Unbalanced transport relaxes conservation when total mass differs

Classical optimal transport assumes source and target contain the same total mass.

Probability distributions satisfy this automatically because both integrate to one.

Many physical applications involve measures whose total mass differs. Cell populations grow, image intensity changes, demand volumes differ, or material can be created and destroyed.

Unbalanced optimal transport relaxes exact marginal conservation and penalizes mass creation or destruction.

Schematically, one solves an objective combining transport cost with divergence penalties measuring how far the transported marginals depart from the original measures.

This distinction is scientifically important.

If total mass difference is meaningful signal, normalizing both objects to probability distributions can erase it.

If only shape matters, normalization may be appropriate.

The choice between balanced and unbalanced transport should therefore follow the process being represented.

## Wasserstein neighbourhoods also define uncertainty sets

Optimal transport connects naturally to optimization under uncertainty.

Suppose

$$
\hat P_n
$$

is the empirical distribution of observed uncertain quantities.

A Wasserstein ambiguity set is

$$
\mathcal P_\varepsilon
=
\left\{
P:
W_p(P,\hat P_n)
\le
\varepsilon
\right\}.
$$

Distributionally robust optimization can then solve

$$
\min_x
\sup_{
P\in\mathcal P_\varepsilon
}
E_P[
L(x,\xi)
].
$$

The radius

$$
\varepsilon
$$

controls how far the adversarial distribution may move from the empirical law in transport geometry.

This differs from moment-based ambiguity, which allows any distribution sharing specified means or covariances, and from divergence balls based on density ratios.

A Wasserstein ball says that probability mass may be relocated, but large relocations are expensive according to the ground metric.

This can be attractive when nearby perturbations of observed scenarios are more plausible than arbitrary reweighting.

It also inherits the same modelling choice: the ground metric defines what counts as a nearby distributional perturbation.

Robustness is therefore geometric.

## Optimal transport is useful when location matters

The simplest support-mismatch example contains the central lesson.

For

$$
\delta_0
$$

and

$$
\delta_a,
$$

KL divergence is infinite for every nonzero

$$
a.
$$

Wasserstein distance is exactly

$$
|a|.
$$

The two answers differ because they ask different questions.

KL asks whether one distribution assigns probability where the other does and how costly the density-ratio mismatch is in information terms.

Wasserstein asks how much probability mass must move through the sample space.

When the geometry of locations matters, that movement can be the scientifically relevant difference.

The one-dimensional quantile representation makes the idea exact.

The Gaussian formula shows how location and covariance geometry combine.

Displacement interpolation turns probability distributions into points connected by transport geodesics.

Barycentres define geometric averages.

Kantorovich-Rubinstein duality connects transport to Lipschitz test functions.

Entropic regularization makes large problems computationally manageable at the cost of changing the objective.

High-dimensional statistics remind us that elegant geometry does not eliminate finite-sample difficulty.

Optimal transport is therefore powerful precisely because it commits to a notion of geometry.

It should be used when that geometry means something.

## References

Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser.

Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. In *Advances in Neural Information Processing Systems 26*.

Kantorovich, L. V. (1942). On the translocation of masses. *Doklady Akademii Nauk SSSR*, 37, 227–229.

Peyré, G., & Cuturi, M. (2019). Computational optimal transport. *Foundations and Trends in Machine Learning*, 11(5-6), 355–607. https://doi.org/10.1561/2200000073

Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians*. Birkhäuser.

Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices. *The Annals of Mathematical Statistics*, 35(2), 876–879.

Villani, C. (2003). *Topics in Optimal Transportation*. American Mathematical Society.

Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
