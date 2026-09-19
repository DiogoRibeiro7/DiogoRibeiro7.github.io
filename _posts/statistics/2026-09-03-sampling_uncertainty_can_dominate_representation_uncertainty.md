---
permalink: '/statistics/sampling_uncertainty_can_dominate_representation_uncertainty/'
title: 'Sampling Uncertainty Can Dominate Representation Uncertainty'
categories:
- Statistics
- Data Science
tags:
- Longitudinal Data
- Clustering
- Bootstrap
- Uncertainty Quantification
- Statistical Computing
author_profile: false
seo_title: 'Sampling vs Representation Uncertainty in Longitudinal Clustering'
seo_description: 'A flexible trajectory representation can recover nonlinear clusters while sampling uncertainty still dominates. A controlled experiment shows how that balance changes as the signal strengthens.'
excerpt: >-
  A better representation can recover structure that a linear summary cannot see,
  but that does not mean representation choice is immediately the main source of
  uncertainty. In a weak-signal experiment, bootstrap sampling instability was
  almost four times larger than representation instability.
summary: >-
  A law-of-total-variance view of longitudinal clustering separates instability
  caused by resampling subjects from instability caused by changing the trajectory
  representation. Two matched simulation cells show a transition from a
  sampling-limited regime to one where representation disagreement becomes the
  larger average uncertainty component.
keywords:
  - longitudinal clustering
  - representation uncertainty
  - sampling uncertainty
  - bootstrap clustering
  - co-clustering stability
  - adjusted Rand index
classes: wide
date: '2026-09-03'
why_this_exists: >-
  Clustering stability is often treated as one quantity, usually measured by a
  bootstrap. That misses a second source of instability: changing how each
  longitudinal trajectory is represented before clustering. A controlled recovery
  experiment made it possible to separate the two and showed that their relative
  importance changes with signal strength.
evidence: >-
  Two completed production simulation cells with identical sample size,
  observation density and measurement noise, differing only in nonlinear class
  separation. Each cell uses 30 independent simulation seeds and 100 paired
  bootstrap replicates per seed.
methodology: >-
  Represents the same longitudinal subjects with a deliberately misspecified
  linear summary and a nonlinear spline summary, clusters both representations,
  evaluates recovery against known latent classes, computes representation-specific
  bootstrap co-clustering probabilities, and decomposes pairwise Bernoulli
  uncertainty into sampling and between-representation components.
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

A clustering result can be unstable for at least two very different reasons.

The first is familiar. Change the sample and the partition changes.

The second is easier to miss. Keep the same subjects and change only the way their trajectories are represented, and the partition changes.

Those are not the same uncertainty.

In longitudinal data the distinction matters because the raw observations are rarely clustered directly. A subject may first be represented by a fitted intercept and slope, spline coefficients, functional principal component scores, latent random effects, or some other summary of the trajectory. The clustering algorithm only sees that representation.

So a stable clustering under bootstrap resampling can still depend heavily on how the trajectory was encoded.

The reverse is also possible. Two representations may disagree, but ordinary sampling variability may still be the larger source of uncertainty.

That second case is the one I want to look at here.

## Two Sources of Instability

Let

$$
Z_{ij}^{(m,b)}
=
\mathbf 1\{i\text{ and }j\text{ are assigned to the same cluster}\},
$$

where $m$ indexes the trajectory representation and $b$ indexes a bootstrap sample.

For representation $m$, define the bootstrap co-clustering probability

$$
C_{ij}^{(m)}
=
E_b\left[Z_{ij}^{(m,b)}\mid m\right].
$$

If a pair of subjects is always placed together under one representation, then $C_{ij}^{(m)}$ is close to one. If it is always separated, the probability is close to zero. Values near one half indicate sampling instability.

Now average over representations. The law of total variance gives

$$
\operatorname{Var}_{m,b}\left(Z_{ij}^{(m,b)}\right)
=
E_m\left[
\operatorname{Var}_b\left(Z_{ij}^{(m,b)}\mid m\right)
\right]
+
\operatorname{Var}_m\left(C_{ij}^{(m)}\right).
$$

The first term is sampling uncertainty.

The second is representation uncertainty.

For a binary co-clustering indicator this can be written more concretely as

$$
\bar C_{ij}(1-\bar C_{ij})
=
\frac{1}{M}
\sum_{m=1}^{M}
C_{ij}^{(m)}\left(1-C_{ij}^{(m)}\right)
+
\operatorname{Var}_m\left(C_{ij}^{(m)}\right),
$$

where

$$
\bar C_{ij}
=
\frac{1}{M}\sum_{m=1}^{M}C_{ij}^{(m)}.
$$

This decomposition separates two qualitatively different situations.

A pair may have high sampling uncertainty because the data do not reliably determine whether the two subjects belong together, even within one fixed representation.

Or a pair may be individually stable under each representation but classified differently by different representations.

Those should not be described with the same word and left there.

## A Controlled Representation Failure

To make the distinction visible, I used a deliberately controlled nonlinear simulation.

The latent classes differ through trajectory shape. The nonlinear signal is constructed so that it has no projection onto the linear intercept-and-slope space on the observation grid. A representation based only on a fitted straight line therefore cannot see the class-defining signal in the idealized case.

A spline representation can.

This is not meant to prove that splines are universally better than linear summaries. It is a stress test. The point is to create a setting where representation choice has a known reason to matter.

Both representations are then clustered with the same type of mixture model. Bootstrap resamples are paired across representations so that differences between methods are not contaminated by giving them different subject resamples.

The experiment records two kinds of output.

First, recovery against the known latent classes, measured by the adjusted Rand index.

Second, the pairwise uncertainty decomposition above.

The full study varies signal strength, noise, sample size and observation density. Here I use two completed cells that are deliberately matched on everything except nonlinear class separation.

Both have

$$
n=150,
\qquad
T=8,
\qquad
\sigma_\varepsilon=0.15.
$$

Each cell contains 30 independent simulation seeds and 100 paired bootstrap replicates per seed.

The weak cell uses a nonlinear class separation of 0.5. The strong cell uses 3.0.

## The Weak-Signal Cell

The weak-signal result is initially surprising because the representation contrast is already obvious in recovery.

The linear representation has mean adjusted Rand index

$$
\overline{ARI}_{\text{linear}}
\approx
-0.0016,
$$

which is essentially chance-level recovery.

The spline representation reaches

$$
\overline{ARI}_{\text{spline}}
\approx
0.582.
$$

So the nonlinear representation is clearly extracting real class structure that the linear representation misses.

It would be tempting to conclude that representation uncertainty must therefore dominate.

It does not.

The average uncertainty components are

$$
\overline V_{\text{sampling}}
\approx
0.1597,
$$

and

$$
\overline V_{\text{representation}}
\approx
0.0413.
$$

Their ratio is

$$
\frac{\overline V_{\text{sampling}}}
{\overline V_{\text{representation}}}
\approx
3.87.
$$

Sampling uncertainty is almost four times larger.

Only about

$$
8.95\%
$$

of subject pairs have representation uncertainty larger than sampling uncertainty.

That is the key result.

The representation matters enough to improve recovery dramatically, but the recovered nonlinear partition is not yet sufficiently stable for disagreement between representations to become the dominant uncertainty source.

The data are still mostly sampling-limited.

## Better Recovery Does Not Imply Stable Recovery

This distinction is easy to miss if we report only the adjusted Rand index.

ARI compares a fitted partition with known truth. It tells us whether the method recovers the latent classes.

It does not tell us whether the recovered partition is stable under resampling.

A method can therefore satisfy

$$
ARI_{\text{spline}}
\gg
ARI_{\text{linear}}
$$

while still having substantial within-method bootstrap instability.

That is exactly what happens in the weak-signal cell.

The flexible representation has access to the right kind of information, but the signal is not strong enough to make every subject pair easy to classify.

This is a useful reminder for applied clustering. Choosing the right representation solves one problem. It does not automatically solve the finite-sample problem.

## Strengthening the Signal Changes the Uncertainty Budget

Now keep the sample size, observation density and measurement noise fixed, and increase only the nonlinear class separation from 0.5 to 3.0.

The spline representation becomes perfectly recovering across the 30 completed seeds:

$$
\overline{ARI}_{\text{spline}}=1.0.
$$

The linear representation remains at chance level because the additional signal still lies outside what that representation can retain.

The interesting change is not only in ARI.

Sampling uncertainty falls from

$$
0.1597
\quad\text{to}\quad
0.0665.
$$

At the same time, representation uncertainty rises from

$$
0.0413
\quad\text{to}\quad
0.0919.
$$

So the ordering reverses at the level of the averages:

$$
\boxed{
\overline V_{\text{representation}}
>
\overline V_{\text{sampling}}
}
$$

in the strong-signal cell.

The fraction of pairs for which representation uncertainty dominates also rises sharply:

$$
8.95\%
\quad\longrightarrow\quad
38.83\%.
$$

Notice that this pairwise fraction remains below one half even though the mean representation component exceeds the mean sampling component. There is no contradiction. The dominance fraction counts pairs, while the means also depend on the magnitude of the uncertainty for each pair.

A smaller set of pairs with large representation disagreement can move the average substantially.

## The Transition in One Table

The two cells can be summarized directly.

```python
import pandas as pd

results = pd.DataFrame(
    [
        {
            "signal": "weak",
            "separation": 0.5,
            "linear_ari": -0.001637,
            "spline_ari": 0.581796,
            "sampling_uncertainty": 0.159700,
            "representation_uncertainty": 0.041255,
            "representation_dominant_pairs": 0.089500,
        },
        {
            "signal": "strong",
            "separation": 3.0,
            "linear_ari": -0.001637,
            "spline_ari": 1.000000,
            "sampling_uncertainty": 0.066531,
            "representation_uncertainty": 0.091887,
            "representation_dominant_pairs": 0.388337,
        },
    ]
)

results["sampling_to_representation"] = (
    results["sampling_uncertainty"]
    / results["representation_uncertainty"]
)

print(results.to_string(index=False))
```

| Signal | Separation | Linear ARI | Spline ARI | Sampling uncertainty | Representation uncertainty | Representation-dominant pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak | 0.5 | -0.002 | 0.582 | 0.160 | 0.041 | 9.0% |
| Strong | 3.0 | -0.002 | 1.000 | 0.067 | 0.092 | 38.8% |

The qualitative transition is clearer than any single number.

In the weak regime,

$$
\text{representation helps}
\quad\text{but}\quad
\text{sampling dominates uncertainty}.
$$

In the strong regime,

$$
\text{the flexible representation is stable}
\quad\text{and}\quad
\text{method disagreement becomes more important}.
$$

That is a very different statement from saying simply that one representation is better.

## Why Representation Uncertainty Can Rise When Recovery Improves

At first glance it may seem odd that representation uncertainty rises as the signal gets stronger.

But representation uncertainty is not measuring whether the better representation is confused.

It measures disagreement between representations.

In the weak-signal regime, even the flexible representation is still moving around under bootstrap resampling. Its co-clustering probabilities often sit away from zero and one. That creates large within-representation Bernoulli variance.

As the signal strengthens, the spline representation becomes much more decisive. Within true classes its co-clustering probabilities move toward one, and between classes they move toward zero.

Sampling variance therefore falls.

The linear representation, however, still cannot see the nonlinear class signal. The two representations now disagree more systematically.

So the uncertainty budget moves from

$$
\text{within-representation instability}
$$

toward

$$
\text{between-representation disagreement}.
$$

That is exactly what the decomposition is supposed to detect.

## Stability Is Conditional on the Representation

Bootstrap stability is often reported as though it were a property of the clustering problem itself.

Strictly speaking, it is conditional on the entire analysis pipeline.

If I bootstrap a clustering method after representing every trajectory with a straight line, I am estimating stability of the straight-line pipeline.

If I bootstrap after representing every trajectory with splines, I am estimating stability of the spline pipeline.

Neither tells me whether the partition is robust to changing the representation.

This matters especially in longitudinal analysis because representation choices are substantive modelling assumptions.

An intercept-and-slope model assumes that the important between-subject differences are adequately described by level and linear trend.

A spline basis allows nonlinear shape differences.

Functional principal components emphasize directions of variation in the observed trajectories.

Random-effects models impose their own covariance and distributional structure.

Two analysts can therefore use the same subjects, the same number of clusters and the same clustering family, yet obtain different partitions because they summarized trajectories differently.

A bootstrap performed after that choice cannot reveal this by itself.

## A Stable Wrong Representation Is Still Wrong

There is another useful edge case.

Suppose one representation is completely stable under bootstrap resampling but consistently ignores the feature that defines the latent classes.

Its sampling uncertainty may be close to zero.

That sounds reassuring until another valid representation produces a completely different but also stable partition.

For a subject pair, imagine

$$
C_{ij}^{(1)}=1
\qquad\text{and}\qquad
C_{ij}^{(2)}=0.
$$

Then the within-representation sampling term is zero for both methods, but the representation term is maximal:

$$
V_{\text{sampling}}=0,
\qquad
V_{\text{representation}}=0.25.
$$

So perfect bootstrap stability is not sufficient evidence that the clustering is intrinsic to the data.

It may only mean that a particular representation makes the same decision consistently.

## What Should Be Reported?

For longitudinal clustering I would separate at least three questions.

First, if simulated truth is available, how well does each representation recover it?

Metrics such as adjusted Rand index belong here.

Second, conditional on each representation, how stable is the partition under changes in the sample?

Bootstrap co-clustering belongs here.

Third, after accounting for sampling instability, how much additional uncertainty appears when reasonable trajectory representations are changed?

That is the representation component.

These quantities should not be collapsed into one score because they answer different questions.

A method can recover truth better but remain sampling-unstable.

A method can be sampling-stable but structurally misspecified.

Two individually stable methods can disagree strongly.

A useful analysis should make those cases distinguishable.

## The Pairwise View Is More Informative Than One Global Number

The averages in this article are convenient for comparing simulation cells, but the decomposition is fundamentally pairwise.

For every pair $(i,j)$, we have

$$
V_{ij}^{\text{total}}
=
V_{ij}^{\text{sampling}}
+
V_{ij}^{\text{representation}}.
$$

That creates a richer diagnostic surface.

Some subject pairs may be easy under every representation.

Some may be unstable under bootstrap sampling regardless of representation.

Some may be individually stable but flip depending on how trajectories are summarized.

The last group is particularly interesting because it identifies where scientific conclusions depend on representation choice rather than simply on small-sample noise.

In an applied study I would want to inspect those pairs or the subjects involved, not only report a global stability index.

## What the Two Cells Do and Do Not Show

These results are deliberately narrow.

They do show a real transition under fixed sample size, observation density and measurement noise when only nonlinear class separation changes.

They show that better recovery by a flexible representation does not imply that representation uncertainty immediately dominates.

They also show that as recovery stabilizes, the relative uncertainty contribution can move toward systematic disagreement between representations.

They do not establish a universal threshold where one form of uncertainty overtakes the other.

That requires the broader grid over signal, noise, sample size and observation density.

The right conclusion from two cells is therefore not

$$
\text{representation uncertainty always dominates at strong signal}.
$$

It is the more useful statement

$$
\boxed{
\text{the dominant source of clustering uncertainty can change with the data regime.}
}
$$

That is precisely why the two components should be measured separately.

## The Practical Lesson

When a clustering analysis changes after switching from one plausible trajectory representation to another, the first reaction is often to ask which representation is correct.

Sometimes that is the right question.

But before answering it, there is another one:

> Is the disagreement larger than the instability we would already expect from sampling the subjects differently?

In the weak-signal experiment here, the answer is no. The representation choice matters for recovery, but sampling uncertainty is still almost four times larger on average.

In the strong-signal experiment, the balance changes. The spline partition becomes perfectly recovering in the completed seeds, sampling uncertainty falls, and representation disagreement becomes the larger average component.

The important object is therefore not a single stability score.

It is the decomposition

$$
\boxed{
\text{total clustering uncertainty}
=
\text{sampling uncertainty}
+
\text{representation uncertainty}.
}
$$

Once those two sources are separated, a result that initially looks contradictory becomes much easier to understand.
