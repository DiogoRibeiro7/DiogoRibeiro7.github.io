---
permalink: '/statistics/the_trouble_with_smooth_curves_in_small_simulation_studies/'
title: 'The Trouble With Smooth Curves in Small Simulation Studies'
categories:
- Statistics
- Data Science
tags:
- Monte Carlo Simulation
- Statistical Computing
- Response Curves
- Thresholds
- Reproducibility
author_profile: false
seo_title: 'The Trouble With Smooth Curves in Small Simulation Studies'
seo_description: 'A handful of Monte Carlo design points can tempt us into fitting a smooth response curve and reporting a threshold. But non-monotone sampling evidence, finite replication noise and structural changes can make that threshold illusory.'
excerpt: >-
  Simulation studies often evaluate a method on a coarse parameter grid and then
  draw a smooth curve through the results. The plot looks persuasive. The problem
  is that the smoothness may come from the plotting method rather than the
  experiment.
summary: >-
  Why observed simulation response curves should be treated as finite empirical
  objects before they are smoothed, interpolated or turned into thresholds. Uses a
  repeated-seed semi-supervised-learning example where an empirical harm crossing
  can be refined, yet local reversals remain and the boundary disappears after a
  structural change in the data-generating process.
keywords:
- simulation study
- Monte Carlo
- smoothing
- threshold estimation
- non-monotonicity
- semi-supervised learning
classes: wide
date: '2026-09-14'
why_this_exists: >-
  Coarse simulation grids encourage visual interpolation. This article separates
  what the sampled points establish from what a smoother merely suggests, and
  shows why a well-localized empirical crossing should not automatically be called
  a population breakpoint.
evidence: >-
  Motivated by repeated-seed shift experiments in which the first sampled harm
  crossing is reproducible under one structural cell but local probabilities remain
  non-monotone, while a matched falsification cell removes the clean boundary
  altogether.
methodology: >-
  Treats the response curve as a sequence of sampled Monte Carlo summaries indexed
  by a controlled design parameter. Refinement is restricted to observed
  false-to-true crossing brackets, repeated seeds are reused, and structural
  falsification is used before any claim of a universal threshold.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
---

A simulation study evaluates a method at four design points.

The result looks something like this:

$$
0.00,
\quad
0.12,
\quad
0.47,
\quad
0.81.
$$

The temptation is immediate.

Draw a smooth line through the points. Fit a spline. Fit a logistic curve. Perhaps impose monotonicity because the underlying mechanism is expected to become worse as the perturbation increases. Then solve for the point where the fitted curve crosses 0.5.

Now the study has a threshold.

The plot looks much more scientific than four dots.

But the experiment may never have established the curve that the plot suggests.

This is one of those cases where visualization can quietly turn an observed finite experiment into an unearned continuous model.

My preferred starting point is much more literal:

$$
\boxed{
\text{the simulation has observed values at the design points that were actually run.}
}
$$

Everything between those points is an additional assumption.

Sometimes that assumption is reasonable.

Sometimes it manufactures the result.

## A Response Curve Is Still an Estimator

Suppose a method is evaluated under a perturbation magnitude $r$. For replication $s$, define a response

$$
R_s(r)
=
G_s(r)-G_s(0),
$$

where $G_s(r)$ is the method's gain relative to some baseline under perturbation $r$.

A negative value means the perturbation made the method worse than its own no-perturbation version.

Across $S$ repeated datasets we might report

$$
\widehat\mu_R(r)
=
\frac{1}{S}
\sum_{s=1}^{S}R_s(r)
$$

or an empirical harm probability

$$
\widehat q(r)
=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf 1\{R_s(r)<0\}.
$$

Those quantities are Monte Carlo estimators.

At every sampled value of $r$, they contain finite-replication noise.

So when we fit a smooth function

$$
\widehat q(r)
\approx
h(r;\widehat\gamma),
$$

we are not merely making a prettier graph.

We have introduced a second statistical model on top of the first simulation experiment.

The smoother has assumptions too.

## The Dangerous Picture

Imagine we observe the following empirical harm probabilities:

| Shift magnitude | Harm probability |
| ---: | ---: |
| 0.25 | 0.34 |
| 0.34 | 0.46 |
| 0.36 | 0.56 |
| 0.50 | 0.53 |
| 0.75 | 0.61 |

A monotone sigmoid through those points will look perfectly plausible.

It may produce a crossing such as

$$
\widehat r_{0.5}=0.35.
$$

The problem is that the observed sequence itself is not perfectly monotone.

The estimate goes

$$
0.56
\rightarrow
0.53
$$

between two increasing shift magnitudes.

With only finite repeated simulations, that reversal may be sampling noise.

Or it may reflect real geometry.

What we do not know is whether we are entitled to erase it.

A monotone smoother answers that question before the data do.

## Observed Monotonicity Is Different From Assumed Monotonicity

There are at least three distinct statements that are easy to conflate.

First:

$$
\widehat q(r_1)
\le
\widehat q(r_2)
\le
\cdots
$$

on the sampled grid.

That is an empirical statement about the observed Monte Carlo summaries.

Second:

$$
q(r)
\text{ is monotone in }r.
$$

That is a population-level assumption about the underlying response function.

Third:

$$
\widetilde q(r)
\text{ is constrained to be monotone by the fitting procedure.}
$$

That is a modeling choice.

These are not equivalent.

A plot from isotonic regression can satisfy the third statement even when the first is visibly false and the second has never been established.

The smooth line then looks like evidence for monotonicity even though monotonicity was inserted by construction.

## What a First Crossing Actually Says

Suppose we use the empirical criterion

$$
\widehat q(r)\ge 0.5.
$$

If the sampled values satisfy

$$
\widehat q(0.34375)=0.46
$$

and

$$
\widehat q(0.359375)=0.56,
$$

then there is a very clean descriptive statement available:

$$
\boxed{
\text{the first sampled 50% crossing lies in }[0.34375,0.359375].
}
$$

That statement is useful.

It is also weaker than saying

$$
r^*=0.351
$$

is the true population threshold.

The bracket is about the design points we evaluated.

It does not tell us whether the unsampled population response crosses once, several times or not at all between those points.

It also does not tell us whether the population criterion is monotone.

## Refinement Without Pretending to Interpolate

There is a practical middle ground between a coarse grid and fitting a continuous response model.

If two adjacent design points have opposite decisions, refine the bracket experimentally.

Suppose the coarse grid contains

$$
r_L<r_U
$$

with

$$
\widehat q(r_L)<0.5
$$

and

$$
\widehat q(r_U)\ge0.5.
$$

Instead of fitting a curve, add new design points strictly inside

$$
[r_L,r_U].
$$

Run the same repeated-seed experiment there. Merge those results back into the evidence. Recompute the first observed crossing.

Then repeat if the bracket remains too wide.

The logic is simple:

$$
\boxed{
\text{detect an observed crossing}
\rightarrow
\text{sample more densely there}
\rightarrow
\text{re-evaluate the crossing}.
}
$$

No response values are invented between sampled points.

## Common Seeds Matter During Refinement

If possible, use the same simulation seeds when adding design points.

Suppose replication $s$ corresponds to one generated dataset. Then compare

$$
R_s(r_1),
\quad
R_s(r_2),
\quad
R_s(r_3)
$$

on the same underlying random problem.

This paired design removes a large amount of irrelevant Monte Carlo variation.

The experiment asks what happens to the same simulated problem as the perturbation changes, rather than comparing unrelated datasets at neighboring values of $r$.

That makes local response differences more interpretable.

## A Refined Bracket Can Be Real and Still Not Be Universal

Here is the more interesting part.

In one repeated-seed experiment, a semi-supervised method begins with a small positive advantage over supervised learning. As the unlabelled feature distribution is shifted, that advantage deteriorates.

With 100 fixed seeds, majority shift harm first appears between roughly

$$
0.344
\quad\text{and}\quad
0.359.
$$

A stronger operational event, majority negative transfer relative to supervised learning, appears later:

$$
\boxed{
0.4375
\le
r
\le
0.4453125.
}
$$

At the lower endpoint, negative transfer occurs in 48% of the repeated datasets.

At the upper endpoint, it reaches 50%.

That is a surprisingly tight empirical bracket.

It would be easy to call

$$
r\approx0.44
$$

"the failure threshold."

That would still be too strong.

## Change One Structural Feature

Now repeat the same experiment while changing only one structural property of the data-generating process.

The labelled fraction is unchanged.

The number of training and test observations is unchanged.

The model settings are unchanged.

The seeds are unchanged.

The perturbation direction and sampled magnitudes are unchanged.

Only the class-cluster separation becomes weaker.

Under the original geometry, the no-shift semi-supervised gain was positive enough that there was an advantage to lose.

Under the weaker separation, the mean no-shift gain is already approximately

$$
-0.00015,
$$

with negative transfer on 43% of seeds.

In other words, the method begins essentially at the failure boundary before any shift is introduced.

The nearby negative-transfer probabilities then fluctuate around 0.5 rather than tracing a clean analogue of the earlier transition.

Only at a substantially larger perturbation does negative transfer become clearly common.

The refined $r\approx0.44$ crossing did not survive the structural change.

That is not a problem with the original refinement.

It tells us what the original bracket actually meant.

## A Boundary Belongs to a Structural Cell

A simulation threshold should usually be indexed by the conditions under which it was observed.

Instead of writing

$$
r^*=0.44,
$$

write something conceptually closer to

$$
r^*(\mathcal S),
$$

where $\mathcal S$ denotes the structural design cell: sample size, signal geometry, noise level, label fraction, algorithm settings and any other fixed features of the experiment.

Then the empirical result is

$$
\widehat r^*(\mathcal S_1)\approx0.44,
$$

while another cell $\mathcal S_2$ may have no comparable crossing at all.

This notation makes the scientific question visible:

$$
\text{which features of }\mathcal S\text{ move, create or destroy the boundary?}
$$

That is more interesting than forcing a universal threshold where one does not exist.

## Smooth Curves Hide Structural Falsification

Suppose we had fitted a smooth response curve only in the first structural cell.

The final figure might show a beautiful sigmoid with a vertical line at $0.44$.

Nothing in that graph would tell the reader that changing one aspect of the geometry eliminates the phenomenon the line appears to summarize.

This is one reason I like falsification cells.

After finding a pattern, deliberately change one structural ingredient that should matter if the interpretation is correct.

Then ask whether the pattern survives.

A result that disappears can be scientifically useful because it identifies the conditions required for the phenomenon.

## Monte Carlo Noise Can Create Reversals

We should still remember that some local irregularity is simply sampling uncertainty.

If

$$
K(r)
\sim
\operatorname{Binomial}(S,q(r)),
$$

then

$$
\widehat q(r)=\frac{K(r)}{S}
$$

has Monte Carlo standard error

$$
\operatorname{SE}\{\widehat q(r)\}
=
\sqrt{\frac{q(r)(1-q(r))}{S}}.
$$

At the worst case $q=0.5$ and $S=100$,

$$
\operatorname{SE}\approx0.05.
$$

So observed values such as 0.46 and 0.53 are not dramatically separated.

Local reversals are therefore unsurprising even when the population curve is monotone.

But that is an argument for uncertainty quantification and additional replication.

It is not automatically an argument for imposing monotonicity.

## Smoothing Can Reduce Noise and Increase Bias

This is the ordinary bias-variance trade-off in a slightly disguised form.

A flexible smoother follows random Monte Carlo wiggles.

A rigid monotone smoother suppresses them.

But if the true response is genuinely non-monotone, the rigid smoother introduces structural bias.

The estimated threshold then solves

$$
\widetilde q(r)=0.5
$$

for the fitted model $\widetilde q$, not necessarily

$$
q(r)=0.5
$$

for the actual response.

The smoothness of the picture is not evidence that the smoothness assumption was correct.

## Interpolation Is a Scientific Assumption

Suppose we observe

$$
\widehat q(0.4)=0.48
$$

and

$$
\widehat q(0.5)=0.54.
$$

Linear interpolation says

$$
\widehat q(r)
\approx
0.48
+
\frac{0.54-0.48}{0.1}(r-0.4).
$$

Solving for 0.5 gives

$$
r\approx0.433.
$$

That calculation is mathematically fine.

But its interpretation depends on assuming the response changes approximately linearly between the two design points.

If that assumption is not justified, 0.433 is an interpolation artifact, not an observed threshold.

I would rather report

$$
[0.4,0.5]
$$

and run more experiments inside the interval.

## Why a Spline Can Be Worse

A spline feels more sophisticated than linear interpolation because it is smooth and flexible.

That sophistication can be misleading with sparse simulation grids.

Given only a few noisy points, spline curvature is determined partly by the basis, penalty and boundary conditions.

Different reasonable smoothing choices can imply different crossing locations.

If the scientific conclusion moves materially when we change the smoother, then the simulation did not identify the threshold very strongly.

That sensitivity itself should be reported.

## Isotonic Regression Is Different, but Still an Assumption

If the response is known from theory to be monotone, isotonic regression can be very useful.

It estimates a monotone sequence without imposing an arbitrary parametric shape.

But the phrase "known from theory" matters.

If monotonicity is merely intuitive, isotonic regression can erase precisely the pattern that should have made us question the intuition.

In exploratory simulation work, I would first show the raw sampled summaries.

Only then would I add a monotone fit, explicitly labelled as conditional on the monotonicity assumption.

## What Convergence Means in an Empirical Refinement

Suppose iterative refinement stops once the observed crossing bracket has width below $\varepsilon$:

$$
r_U-r_L\le\varepsilon.
$$

It is tempting to say the threshold estimate has converged.

A more precise statement is:

$$
\boxed{
\text{the sampled empirical crossing has been localized to the requested design resolution.}
}
$$

That does not mean Monte Carlo uncertainty vanished.

It does not mean the true population boundary is inside the bracket with 95% confidence.

It does not mean there is a unique population boundary.

It does not even mean an unsampled reversal cannot occur inside the bracket.

Numerical resolution and statistical uncertainty are different quantities.

## A Useful Separation of Questions

When looking at a simulation response curve, I find it useful to separate four questions:

1. **What was observed?**
   The means, medians, probabilities or other summaries at the sampled design points.

2. **What numerical localization was performed?**
   Additional design points inside observed crossing brackets.

3. **What structural assumptions are being imposed?**
   Smoothness, monotonicity, parametric response form or interpolation.

4. **What statistical uncertainty remains?**
   Monte Carlo variation, seed-to-seed heterogeneity and uncertainty about any population threshold.

A polished figure often collapses all four into one line.

That is exactly what I want to avoid.

## Plot the Data Before the Story

A good response-curve figure should make the observed evidence difficult to confuse with the fitted interpretation.

I would start with the sampled points themselves.

If uncertainty intervals are available, show them.

If a crossing bracket was refined experimentally, show the bracket.

If a smooth curve is added, draw it as a model-based overlay rather than as though it were the data.

And if monotonicity is imposed, say so in the caption.

The visual hierarchy should be

$$
\text{observed points}
\;>\;
\text{uncertainty}
\;>\;
\text{fitted curve}.
$$

Too many simulation figures reverse that order.

## When a Smooth Curve Is Justified

There are cases where smoothing is entirely reasonable.

If theory implies differentiability and monotonicity, the simulation grid is dense, replication error is small, and alternative plausible smoothers give essentially the same curve, interpolation can be a useful summary.

Likewise, if the response model itself is an object of interest, one can formally specify

$$
q(r;\gamma)
$$

and estimate $\gamma$ with uncertainty.

Then the smoothing model is part of the analysis rather than hidden inside the plotting library.

The important point is not "never smooth."

It is

$$
\boxed{
\text{do not smuggle a continuous response model into the analysis merely by drawing a line.}
}
$$

## What I Would Report

For a small simulation study with a possible threshold, I would preserve the finite evidence first:

- the exact design grid;
- the number of repeated seeds or replications;
- the response summary at every design point;
- Monte Carlo uncertainty for those summaries;
- whether the observed sequence is monotone;
- the first sampled crossing bracket for any criterion of interest;
- any experimental refinement points added inside that bracket;
- structural cells in which the crossing was reproduced or falsified.

Only after that would I consider a smooth threshold model.

That ordering prevents the fitted curve from becoming stronger evidence than the simulations it was fitted to.

## The Deeper Lesson

There is a broader methodological issue here.

Simulation studies often feel deterministic because the researcher controls the design.

But the outputs are still random estimates.

A grid of Monte Carlo summaries is not a known function sampled at a few points. It is a noisy empirical object generated by repeated stochastic experiments.

When we draw a smooth line through it, we make assumptions about both the underlying scientific response and the Monte Carlo noise.

Those assumptions may be reasonable.

They should not be invisible.

So when a coarse simulation suggests a threshold, my preferred reaction is not to fit a prettier curve.

It is to ask where the observed decision changes, run more experiments there, preserve local reversals, and then test whether the phenomenon survives a structural perturbation.

That workflow may produce a less elegant figure.

It produces a more honest result.

The principle is simple:

$$
\boxed{
\text{sample the boundary before you model the curve.}
}
$$
