---
permalink: '/statistics/confidence_sets_are_not_just_intervals/'
title: 'Confidence Sets Are Not Just Intervals'
categories:
- Statistics
- Statistical Computing
tags:
- Confidence Sets
- Partial Identification
- Moment Inequalities
- Test Inversion
- Projection
author_profile: false
seo_title: 'Confidence Sets Are Not Just Intervals: Test Inversion, Disconnected Sets and Projection'
seo_description: 'A confidence set is an accepted parameter set, not necessarily a single interval. Test inversion can produce disconnected regions, and projection over nuisance parameters can preserve gaps that a convex hull would wrongly fill.'
excerpt: >-
  We often report uncertainty as a lower and upper bound. That is convenient, but
  it quietly assumes the accepted parameter values form one connected interval.
  Test inversion, partial identification and nuisance-parameter projection do not
  owe us that geometry.
summary: >-
  A practical explanation of confidence sets as inverted acceptance regions. The
  article develops a reproducible disconnected-set example, explains why convex
  hulls can overstate acceptance, and distinguishes finite-grid projection from
  continuous nuisance profiling.
keywords:
  - confidence set
  - disconnected confidence interval
  - partial identification
  - test inversion
  - moment inequalities
  - projection
classes: wide
date: '2026-09-12'
why_this_exists: >-
  Statistical software often compresses set-valued inference into lower and upper
  endpoints, even when the actual inverted acceptance region is disconnected or
  only finitely represented. That presentation can erase important inferential
  structure.
evidence: >-
  The article uses a one-parameter nonlinear test-inversion example whose accepted
  parameter values split into two components, plus a two-parameter finite-grid
  projection example where profile p-values are exact maxima over represented
  nuisance rows.
methodology: >-
  Pointwise tests are inverted on explicit parameter grids. Connected components
  are defined only through adjacent accepted represented values. Projection retains
  target values for which at least one represented nuisance configuration is
  accepted, without interpolation or gap filling.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  og_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  twitter_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
---

A confidence interval is one of the most familiar objects in statistics.

We learn to report

$$
[L,U]
$$

and to interpret every value between the two endpoints as belonging to the uncertainty region.

That geometry becomes so familiar that it is easy to forget the more general object:

$$
\boxed{\text{a confidence set}.}
$$

A confidence set need not be one interval.

It can be disconnected.

It can be empty.

It can contain isolated points on a finite grid.

It can have several separated components.

And after projection from a multidimensional parameter space, the set of accepted target values can have gaps that should not be filled simply because drawing one interval looks cleaner.

The distinction is not cosmetic.

If the statistical procedure rejects a value in the middle of two accepted regions, replacing the set by its convex hull changes the inferential statement.

## Test Inversion Comes First

Suppose a pointwise test evaluates

$$
H_0:\theta=\theta_0
$$

for every candidate value \(\theta_0\).

Let

$$
\varphi(\theta_0)\in\{0,1\}
$$

be the rejection decision, where \(1\) means reject.

The inverted confidence set is

$$
\mathcal C_{1-\alpha}
=
\left\{
\theta:
\varphi(\theta)=0
\right\}.
$$

Equivalently, if the pointwise test reports a p-value \(p(\theta)\),

$$
\mathcal C_{1-\alpha}
=
\left\{
\theta:
p(\theta)>\alpha
\right\},
$$

up to whatever tie convention the procedure defines.

There is nothing in that definition requiring \(\mathcal C_{1-\alpha}\) to be connected.

The interval shape appears only in special problems where the test statistic has suitable monotonicity or convexity properties.

Those properties are common enough to shape intuition.

They are not universal.

## A Toy Example With Two Components

Consider a scalar parameter \(\theta\) and an estimator

$$
Y
=
(\theta^2-1)^2+\varepsilon,
$$

where

$$
\varepsilon\sim N(0,\sigma^2).
$$

Suppose we observe

$$
y_{obs}=0.04,
\qquad
\sigma=0.08.
$$

For a candidate \(\theta_0\), define the pointwise z-statistic

$$
Z(\theta_0)
=
\frac{y_{obs}-(\theta_0^2-1)^2}{\sigma}.
$$

The two-sided p-value is

$$
p(\theta_0)
=
2\left[1-\Phi\left(|Z(\theta_0)|\right)\right].
$$

Now invert the 5% test.

The model mean

$$
(\theta^2-1)^2
$$

is small near both

$$
\theta=-1
$$

and

$$
\theta=+1.
$$

It is much larger around

$$
\theta=0.
$$

So values near both \(-1\) and \(+1\) can be compatible with the observation even though values in between are rejected.

A short computation makes this visible.

```python
import numpy as np
from scipy.stats import norm

alpha = 0.05
y_obs = 0.04
sigma = 0.08

grid = np.linspace(-1.8, 1.8, 721)
mu = (grid**2 - 1.0) ** 2
z = (y_obs - mu) / sigma
p = 2.0 * norm.sf(np.abs(z))
accepted = p > alpha

accepted_grid = grid[accepted]
```

To recover connected components on the represented grid:

```python
components = []
start = None
previous = None

for theta, keep in zip(grid, accepted):
    if keep and start is None:
        start = theta
    if keep:
        previous = theta
    elif start is not None:
        components.append((start, previous))
        start = None
        previous = None

if start is not None:
    components.append((start, previous))

print(components)
```

The result consists of two separated regions, approximately

$$
\boxed{
\mathcal C_{0.95}
\approx
[-1.10,-0.87]
\cup
[0.87,1.10].
}
$$

The exact endpoints vary slightly with grid resolution, but the geometry does not.

There are two components.

## The Convex Hull Gives a Different Answer

The smallest ordinary interval containing the confidence set is

$$
[-1.10,1.10].
$$

That looks tidy.

It also includes values around zero.

But at \(\theta=0\), the model mean is

$$
(0^2-1)^2=1,
$$

which is far from the observed \(0.04\).

The pointwise test strongly rejects it.

So replacing

$$
[-1.10,-0.87]
\cup
[0.87,1.10]
$$

by

$$
[-1.10,1.10]
$$

changes the statement from

> these two separated parameter regions were not rejected

into

> every parameter value between the extreme endpoints belongs to the reported uncertainty region.

Those are different statements.

The convex hull is therefore not a harmless formatting choice.

It is a new set:

$$
\operatorname{hull}(\mathcal C)
\supsetneq
\mathcal C.
$$

## Why Disconnected Sets Arise

The toy model is deliberately simple, but the mechanism is general.

Disconnected confidence sets appear whenever the mapping from parameters to observables is non-monotone or many-to-one.

Suppose the data identify some function

$$
g(\theta)
$$

well, but \(g\) is not injective.

Then several separated parameter values can imply similar observable behaviour.

If

$$
g(\theta_1)
\approx
g(\theta_2)
$$

for distant \(\theta_1\) and \(\theta_2\), pointwise inversion can preserve both regions.

This is not a numerical pathology.

It is information about the identification geometry.

The set shape is telling us that the data distinguish some parameter values but not others in a globally simple ordering.

## Intervals Encode More Than Uncertainty

Reporting one interval implicitly says something geometric:

$$
\theta_1,\theta_2\in\mathcal C,
\quad
\theta_1<\theta<\theta_2
\quad\Longrightarrow\quad
\theta\in\mathcal C.
$$

That is a connectedness assumption.

Sometimes the statistical problem guarantees it.

Sometimes software silently imposes it after the fact.

Those are very different situations.

A useful discipline is therefore:

$$
\boxed{\text{derive the set first, summarize its geometry second.}}
$$

Do not decide in advance that the result must be an interval.

## Finite-Grid Inversion Is Still Set Inversion

In computational work, the continuous parameter space is often replaced by a grid

$$
\mathcal G
=
\{\theta_1,\ldots,\theta_K\}.
$$

Then the exact object computed by the software is

$$
\mathcal C_{\mathcal G}
=
\left\{
\theta_k\in\mathcal G:
p(\theta_k)>\alpha
\right\}.
$$

This is a finite represented set.

It is not automatically an approximation to one single continuous interval.

If the accepted grid points are

$$
\{-1.05,-1.00,-0.95,0.95,1.00,1.05\},
$$

then those are the accepted represented values.

A reporting layer can identify two adjacent runs,

$$
[-1.05,-0.95]_{\mathcal G}
$$

and

$$
[0.95,1.05]_{\mathcal G},
$$

but the notation should not imply that every real number inside those coarse intervals was explicitly tested.

Grid inversion answers a finite question unless extra theory or numerical refinement justifies more.

## Adjacency Is a Computational Concept

On an ordered grid

$$
\theta_1<\theta_2<\cdots<\theta_K,
$$

it is natural to call consecutive accepted grid values one component.

Suppose the decisions are

$$
0,0,1,1,1,0,0,1,1,
$$

where \(1\) means accepted.

Then there are two accepted runs.

This is useful bookkeeping.

But an important boundary remains:

if two adjacent coarse-grid points have the same decision, a coarse inversion has learned nothing about possible hidden changes between them.

The statement

$$
p(\theta_k)>\alpha,
\qquad
p(\theta_{k+1})>\alpha
$$

does not logically imply

$$
p(\theta)>\alpha
\quad\text{for every}\quad
\theta\in(\theta_k,\theta_{k+1}).
$$

That implication requires smoothness or shape information the finite grid may not provide.

## Refinement Sharpens Detected Boundaries

Suppose adjacent grid points have opposite decisions:

$$
p(a)>\alpha,
\qquad
p(b)\le\alpha.
$$

Then there is an observed decision transition between \(a\) and \(b\) on the coarse grid.

A natural numerical refinement repeatedly evaluates midpoints and narrows the bracket.

If

$$
[a_m,b_m]
$$

is the current bracket, evaluate

$$
m_m=\frac{a_m+b_m}{2}.
$$

Keep the half-bracket whose endpoints retain opposite decisions.

Continue until

$$
b_m-a_m\le\varepsilon.
$$

This can localize a detected boundary much more precisely.

It does not discover boundaries hidden inside coarse intervals whose endpoints had the same decision.

That is a different search problem.

So numerical refinement changes

$$
\text{boundary localization},
$$

not automatically

$$
\text{global set discovery}.
$$

## A Boundary Bracket Is Not an Exact Endpoint

If a refinement routine returns

$$
[a,b]
$$

with

$$
b-a\le10^{-6},
$$

it is tempting to report the midpoint as the confidence-set endpoint.

That can overstate what was established.

What the computation actually proved is closer to:

> the implemented decision changes somewhere inside a bracket this narrow, under the same Monte Carlo and test configuration.

If the pointwise p-value itself is Monte Carlo estimated, then numerical refinement does not remove Monte Carlo uncertainty.

One can have a very narrow numerical bracket around a noisy decision boundary.

Those are separate error sources:

$$
\boxed{
\text{grid/refinement error}
\neq
\text{Monte Carlo error}.
}
$$

## Floating-Point Arithmetic Adds Another Boundary

Bisection sounds infinitely refinable in real arithmetic.

Computers do not use real arithmetic.

Eventually, for adjacent representable floating-point numbers \(a<b\), the computed midpoint can equal one endpoint:

$$
\operatorname{fl}\left(\frac{a+b}{2}\right)
=a
$$

or

$$
=b.
$$

Then the algorithm cannot subdivide the bracket further.

That state is not convergence merely because the code stopped.

A careful implementation distinguishes

$$
\texttt{converged}=\texttt{TRUE}
$$

from

$$
\texttt{stalled}=\texttt{TRUE}.
$$

If the final width is still larger than the requested tolerance but no representable midpoint exists, the honest status is

$$
\boxed{
\text{stalled, not converged}.
}
$$

Set-valued inference makes these numerical semantics visible because boundaries are part of the returned scientific object.

## Now Add a Nuisance Parameter

The geometry becomes more interesting with a parameter vector

$$
\theta=(\psi,\lambda),
$$

where \(\psi\) is the target and \(\lambda\) is nuisance.

Suppose we can compute a pointwise p-value

$$
p(\psi,\lambda)
$$

for every complete parameter vector.

The joint inverted set is

$$
\mathcal C
=
\left\{
(\psi,\lambda):
p(\psi,\lambda)>\alpha
\right\}.
$$

The projected confidence set for \(\psi\) is conceptually

$$
\operatorname{proj}_{\psi}(\mathcal C)
=
\left\{
\psi:
\exists\lambda
\text{ such that }
(\psi,\lambda)\in\mathcal C
\right\}.
$$

Equivalently, if a valid profile p-value is available,

$$
p_{prof}(\psi)
=
\sup_{\lambda}p(\psi,\lambda),
$$

then

$$
\psi\in\operatorname{proj}_{\psi}(\mathcal C)
\quad\Longleftrightarrow\quad
p_{prof}(\psi)>\alpha.
$$

This is where computational claims need care.

## Finite-Grid Projection Is a Finite Maximum

Suppose the software evaluates only a finite parameter grid

$$
\mathcal G
=
\left\{
(\psi_r,\lambda_r)
\right\}_{r=1}^R.
$$

Then for a represented target value \(\psi\), the computable profile quantity is

$$
\widehat p_{prof}(\psi)
=
\max_{r:\psi_r=\psi}
p(\psi_r,\lambda_r).
$$

That is an exact maximum over represented rows.

It is not automatically

$$
\sup_{\lambda\in\Lambda}p(\psi,\lambda)
$$

over the continuous nuisance space.

The distinction matters because missing the true supremum can make the profile p-value too small.

A smaller profile p-value can incorrectly reject a target value.

So calling a finite-grid maximum a continuous profile supremum is not merely imprecise language. It can hide an anti-conservative approximation.

## A Simple Projection Example

Consider this finite joint grid:

| \(\psi\) | \(\lambda\) | p-value |
| ---: | ---: | ---: |
| -1.0 | 0.0 | 0.18 |
| -1.0 | 1.0 | 0.42 |
| -0.5 | 0.0 | 0.01 |
| -0.5 | 1.0 | 0.03 |
| 0.0 | 0.0 | 0.02 |
| 0.0 | 1.0 | 0.04 |
| 0.5 | 0.0 | 0.03 |
| 0.5 | 1.0 | 0.01 |
| 1.0 | 0.0 | 0.37 |
| 1.0 | 1.0 | 0.21 |

At level \(\alpha=0.05\), the finite-grid profile p-values are

$$
\widehat p_{prof}(-1)=0.42,
$$

$$
\widehat p_{prof}(-0.5)=0.03,
$$

$$
\widehat p_{prof}(0)=0.04,
$$

$$
\widehat p_{prof}(0.5)=0.03,
$$

and

$$
\widehat p_{prof}(1)=0.37.
$$

The projected represented set is therefore

$$
\boxed{
\{-1,1\}.
}
$$

Its convex hull is

$$
[-1,1].
$$

But the represented target values

$$
-0.5,\;0,\;0.5
$$

all have profile p-values below 0.05.

Filling the gap would erase exactly the information the projection produced.

## Projection Can Create or Preserve Disconnectedness

A joint accepted set can have complicated geometry.

Projecting it onto one coordinate can simplify that geometry, but it need not make it connected.

Imagine two separated accepted islands in \((\psi,\lambda)\)-space:

$$
\mathcal C_1
\quad\text{and}\quad
\mathcal C_2.
$$

If their \(\psi\)-ranges are separated, the projection remains disconnected:

$$
\operatorname{proj}_{\psi}(\mathcal C)
=
A\cup B,
\qquad
A\cap B=\varnothing.
$$

Again, the gap is information.

It says that no accepted nuisance configuration on the represented grid rescued those intermediate target values.

## Witnesses Are Useful

For each projected target value \(\psi\), it is often useful to retain one nuisance configuration attaining the finite-grid maximum:

$$
r^*(\psi)
\in
\arg\max_{r:\psi_r=\psi}
p_r.
$$

This row is a witness.

It answers:

> Which represented nuisance configuration made this target value look most plausible?

Witnesses are particularly useful for debugging and interpretation.

But they have their own invariance rule.

If the source grid rows are permuted, the absolute row number of the witness may change.

The scientific result should not.

The correct invariant is therefore not

$$
\text{same row index},
$$

but

$$
\boxed{
\text{witness belongs to the correct target group and attains the same profile maximum}.
}
$$

That is a good example of testing semantics rather than representation details.

## Row Order Must Not Change the Set

A finite parameter grid is a mathematical set represented as a table.

Permuting table rows should not change

- represented target values,
- profile p-values,
- acceptance decisions,
- connected components,
- projected accepted values.

Formally, if \(\pi\) is any permutation of the source rows,

$$
\operatorname{Projection}(\mathcal G)
=
\operatorname{Projection}(\pi\mathcal G)
$$

up to ordering conventions and witness row labels.

This is exactly the sort of invariant that catches implementation bugs ordinary example tests may miss.

## Scalar Transformations Need the Same Discipline

Sometimes the target is not one named coordinate but a transformation

$$
\psi=h(\theta).
$$

For example,

$$
h(\mu,\eta)=\mu-\eta.
$$

On a finite grid, several source rows may map to the same scalar target value.

Then the finite-grid profile p-value is

$$
\widehat p_{prof}(v)
=
\max_{r:h(\theta_r)=v}p_r.
$$

One subtle issue appears immediately: how are equal transformed values defined?

If the implementation groups by exact numerical equality, then

$$
0.3
$$

and

$$
0.30000000000000004
$$

may be distinct represented target values.

A tolerance-based grouping rule would be a different inferential/computational contract.

Neither should be introduced accidentally.

The grouping semantics need to be explicit because they determine the set being reported.

## A Descriptive Hull Can Still Be Useful

Sometimes users genuinely want a quick range summary.

For a finite accepted scalar set

$$
A
=
\{v_1,\ldots,v_m\},
$$

one can report

$$
\left[
\min A,
\max A
\right].
$$

That can be useful as a descriptive hull.

The important word is descriptive.

It should not be interpreted as

$$
\text{every interior value is accepted}
$$

or

$$
\text{every interior value was represented}
$$

or

$$
\text{the continuous confidence set covers the whole interval}.
$$

A good software interface can return both the exact represented accepted values and the hull while naming them differently.

The problem is not computing the hull.

The problem is silently replacing the set with it.

## Empty Sets Are Legitimate Results

Another habit inherited from interval thinking is discomfort with empty confidence sets.

But inversion can produce

$$
\mathcal C=\varnothing.
$$

On a finite grid, this simply means every represented candidate was rejected.

Possible interpretations include:

- the parameter grid missed the plausible region,
- the model is badly incompatible with the data,
- the test is too aggressive under the current approximation,
- the represented bounds are too narrow,
- the assumptions defining the model are violated.

The software should not repair emptiness by automatically returning the full range or the nearest rejected point.

An empty set is diagnostic information.

## One-Point Sets Are Legitimate Too

At the other extreme, a confidence set may contain exactly one represented value:

$$
\mathcal C_{\mathcal G}=\{\theta_k\}.
$$

There are no transition brackets to refine unless neighbouring represented points exist and disagree.

A robust implementation should handle this without pretending there must be two endpoints.

Again, the set abstraction is cleaner than the interval abstraction.

The set can contain zero, one or many components without requiring special conceptual exceptions.

## Coverage Claims Must Match the Computed Object

Suppose a joint finite-grid confidence set has the property that, if the true parameter vector is represented, it is included with probability at least \(1-\alpha\):

$$
P_{\theta_0}
\left(
\theta_0\in\mathcal C_{\mathcal G}
\right)
\ge1-\alpha,
\qquad
\theta_0\in\mathcal G.
$$

Then projection gives the represented-grid implication

$$
P_{\theta_0}
\left(
\psi_0\in
\operatorname{proj}_{\psi}(\mathcal C_{\mathcal G})
\right)
\ge1-\alpha.
$$

But this does not establish coverage for an off-grid \(\psi_0\) or for arbitrary nuisance values absent from the grid.

The difference between

$$
\text{finite represented parameter space}
$$

and

$$
\text{continuous parameter space}
$$

must remain visible in the claim.

## Why Generic Continuous Optimization Can Be Dangerous

A natural response is:

> Why not just optimize over the nuisance parameter continuously?

Sometimes that is exactly the right solution.

But it needs justification.

Suppose the profile p-value is

$$
p_{prof}(\psi)
=
\sup_{\lambda}p(\psi,\lambda).
$$

If the objective is smooth, well behaved and globally optimizable, continuous profiling may work very well.

In more complicated inferential procedures the objective can contain

- moment-selection discontinuities,
- finite-Monte-Carlo step functions,
- piecewise statistics,
- boundary effects,
- multiple local maxima.

A local optimizer returning one maximum does not certify the global supremum.

If it misses a higher nuisance configuration, the reported profile p-value is too small.

That error goes in the dangerous direction:

$$
\widehat p_{prof}(\psi)
<
p_{prof}(\psi)
$$

can make the projected set too narrow.

A finite-grid maximum is limited, but honest:

$$
\boxed{
\text{maximum over represented nuisance values}
}
$$

is a precise computational statement.

## Partial Identification Makes the Set View Natural

In point-identified problems, we often think first about an estimator

$$
\widehat\theta
$$

and then attach an interval around it.

In partially identified problems, the population object itself may be a set

$$
\Theta_I(P).
$$

The data-generating distribution does not identify one unique parameter vector even with infinite data.

Moment inequalities provide a common example. If

$$
E[m_j(W,\theta)]\le0,
\qquad
j=1,\ldots,J,
$$

then the identified set is

$$
\Theta_I
=
\left\{
\theta:
E[m_j(W,\theta)]\le0
\text{ for all }j
\right\}.
$$

This population set can itself be non-convex or disconnected depending on the moment functions and parameterization.

In that setting, forcing inference into one interval is especially unnatural.

Set-valued inference is not a complication added by software.

It reflects the object being learned.

## The Geometry Is Part of the Result

Suppose two analysts report uncertainty for the same parameter.

The first reports

$$
[-1.1,1.1].
$$

The second reports

$$
[-1.1,-0.9]
\cup
[0.9,1.1].
$$

The endpoint range is identical.

The inferential content is not.

The second result says that values near zero are inconsistent with the data under the test.

That is substantive information.

A downstream scientific decision may depend on it.

For example, perhaps negative and positive parameter values imply two qualitatively different mechanisms, while values near zero imply no meaningful effect. A disconnected set can say:

> the data support either mechanism, but not the near-zero explanation.

The convex hull erases that distinction entirely.

## What Software Should Return

For one-dimensional inversion, I would want a result object to preserve at least

- the represented grid,
- pointwise p-values or decisions,
- accepted values,
- connected components,
- the confidence level,
- test method and Monte Carlo settings,
- whether the set is empty,
- parameter bounds,
- any refined transition brackets,
- convergence and numerical-stall diagnostics.

For projection, I would add

- represented target values,
- finite-grid profile p-values,
- accepted projected values,
- witness rows or witness parameters,
- connected components where order is meaningful,
- an explicitly labelled descriptive hull if one is provided.

That may seem like more information than two endpoints.

It is exactly the information needed to avoid pretending the geometry is simpler than it is.

## Plot the Decisions, Not Just the Hull

A useful visualization is often extremely simple.

For one-dimensional inversion, plot

$$
p(\theta)
$$

against \(\theta\) with a horizontal line at \(\alpha\).

The confidence set is the region where the p-value lies above the line.

Disconnectedness becomes visually obvious.

On a finite grid, plotting accepted and rejected points directly can be even more honest than drawing filled intervals.

For projections, plot the finite-grid profile p-values

$$
\widehat p_{prof}(\psi)
$$

against represented target values.

Then the reader can see both the accepted components and the gaps.

## The Main Statistical Lesson

The phrase "confidence interval" is so common that it can quietly turn into a modelling assumption about the geometry of uncertainty.

The safer hierarchy is

$$
\boxed{
\text{pointwise test}
\rightarrow
\text{inverted confidence set}
\rightarrow
\text{geometry}
\rightarrow
\text{summary}.
}
$$

Not

$$
\boxed{
\text{decide there must be two endpoints}
\rightarrow
\text{force everything between them into the answer}.
}
$$

When the accepted parameter region is disconnected, the gaps are part of the evidence.

When the computation is performed on a finite grid, the grid is part of the inferential boundary.

When nuisance parameters are projected out, a finite maximum over represented rows is not the same thing as a continuous supremum.

And when a hull is useful for presentation, it should be labelled as a hull rather than silently substituted for the set.

The broader principle is simple:

$$
\boxed{
\text{uncertainty has geometry, and the geometry should survive the software.}
}
