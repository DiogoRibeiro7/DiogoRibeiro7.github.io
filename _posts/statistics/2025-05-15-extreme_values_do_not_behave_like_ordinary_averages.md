---
permalink: '/statistics/extreme_values_do_not_behave_like_ordinary_averages/'
title: 'Extreme Values Do Not Behave Like Ordinary Averages'
date: '2025-05-15'
categories:
- Statistics
tags:
- Extreme Value Theory
- Generalized Extreme Value
- Generalized Pareto
- Tail Risk
- Statistical Inference
author_profile: false
classes: wide
seo_title: 'Extreme Values Do Not Behave Like Ordinary Averages'
seo_description: 'Extreme-value theory models maxima and threshold exceedances through their own limiting laws. The GEV and GPD describe tail behaviour that ordinary mean-variance models can miss completely.'
seo_type: article
excerpt: >-
  Averages and maxima obey different asymptotic laws. Two distributions can have
  the same mean and variance while implying radically different rare-event
  probabilities, return levels and maximum behaviour.
summary: >-
  This article develops extreme-value theory from the distribution of maxima,
  derives the generalized extreme-value family, explains the role of the shape
  parameter, and connects block maxima to peaks over threshold through the
  generalized Pareto limit. An exact comparison between a standard normal and a
  standardized Student-t distribution shows how identical first two moments can
  coexist with tail probabilities differing by several orders of magnitude. The
  article also treats threshold selection, return levels, extremal clustering,
  and the uncertainty created by extrapolating beyond the observed record.
keywords:
- extreme value theory
- generalized extreme value distribution
- generalized Pareto distribution
- peaks over threshold
- return levels
- tail risk
why_this_exists: >-
  Statistical modelling is usually taught through averages, variances and
  central-limit arguments. Extreme events depend on a different limiting
  structure. Extrapolating a well-fitted central distribution into the far tail
  can therefore give a precise answer to the wrong asymptotic problem.
evidence: >-
  Classical extreme-value limit theory, the Fisher-Tippett-Gnedenko extremal
  types result, the Pickands-Balkema-de Haan threshold-excess theorem, exact
  Gaussian and Student-t tail calculations, and standard return-level and
  extremal-index arguments.
methodology: >-
  Begin from the exact distribution of a sample maximum and compare its scaling
  with the central-limit theorem. Derive the GEV family and its three shape
  regimes, then connect block maxima with GPD threshold exceedances. Use exact
  distributional calculations to demonstrate how moment matching fails to
  determine tail risk.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
---

<!--
Development contract
Question: Why should maxima and threshold exceedances be modelled differently from ordinary observations?
Claim: Extremes obey their own limit theory. The tail class, represented by the extreme-value shape parameter, controls maximum behaviour and cannot be inferred from mean and variance alone.
Counterclaim: EVT is asymptotic and data-hungry. In moderate samples, threshold and block choices can dominate the result, and a simpler parametric model can be preferable when its tail is scientifically justified.
Evidence object: Exact maximum distribution, GEV and GPD limits, Gaussian versus Student-t comparison with equal mean and variance, return-period calculation, and extremal-index extension for dependent sequences.
Failure case: Fitting a central distribution and extrapolating blindly, selecting thresholds to obtain a preferred tail estimate, or interpreting a return period as a deterministic recurrence schedule.
Reader payoff: Understand what EVT estimates, why GEV and GPD models are linked, what the shape parameter means, and which diagnostics are needed before trusting a far-tail extrapolation.
Exclusions: A catalogue of every tail-index estimator, a finance-only treatment, and repetition of the existing operational-risk POT simulation.
-->

Statistical intuition is built largely around averages. If independent observations have a finite variance, the sample mean becomes approximately Gaussian after centring and scaling, almost regardless of the detailed shape of the original distribution. This is one of the reasons mean and variance are so useful. Many distributions that look different at the observation level behave similarly after enough additive aggregation.

Maxima do not behave this way. The largest observation in a sample is determined by the tail, and two distributions that agree closely around their centres can produce completely different maxima. Matching means and variances does not fix the probability of an event five or ten standard deviations away. A model can fit the bulk of the data extremely well and still be unusable for design loads, flood levels, insurance losses, component stresses, heat extremes, market crashes or any other problem in which the scientific target lies in the tail rather than near the centre.

Extreme-value theory exists because the asymptotic problem is different. Instead of asking how sums behave, it asks how maxima and high threshold exceedances behave. The answer is strikingly restrictive. Under broad conditions, normalized maxima converge to one three-parameter family, the generalized extreme-value distribution, and high threshold excesses converge to the generalized Pareto family. The same shape parameter appears in both results and determines whether the underlying tail is heavy, exponentially light or bounded.

The practical value of the theory is not that it somehow observes rare events that have never happened. It provides a principled structure for extrapolating from the upper part of the data, while making the assumptions and uncertainty of that extrapolation explicit.

## Averages and maxima require different normalizations

Let

$$
X_1,\ldots,X_n
$$

be independent observations with common distribution function $F$. For the sample mean,

$$
\bar X_n
=
\frac{1}{n}
\sum_{i=1}^n X_i,
$$

the central-limit theorem says, under familiar finite-variance conditions,

$$
\frac{
\sqrt n(\bar X_n-\mu)
}{
\sigma
}
\Rightarrow
N(0,1).
$$

The mean aggregates small contributions from the entire distribution. The limiting Gaussian law is therefore controlled by first and second moments rather than by the detailed tail.

Now consider the maximum,

$$
M_n
=
\max(X_1,\ldots,X_n).
$$

Its distribution is exact:

$$
P(M_n\le x)
=
P(X_1\le x,\ldots,X_n\le x)
=
F(x)^n.
$$

Write the upper-tail probability as

$$
\bar F(x)
=
1-F(x).
$$

For a high level $x$ with small $\bar F(x)$,

$$
F(x)^n
=
\left[
1-\bar F(x)
\right]^n
\approx
\exp\left[
-n\bar F(x)
\right].
$$

A non-degenerate maximum therefore lives near levels $x_n$ for which

$$
n\bar F(x_n)
$$

is of order one. This is already enough to show why the tail matters. The maximum is located where the survival probability is approximately

$$
\frac{1}{n},
$$

not where the bulk of the data lie. As $n$ grows, the relevant region moves further into the tail.

For a normal distribution, this movement is slow, roughly of logarithmic order after appropriate transformation. For a power-law tail, the maximum grows polynomially with sample size. For a distribution with a finite upper endpoint, the maximum approaches that endpoint instead of diverging. These are fundamentally different asymptotic regimes, and they cannot be distinguished by the mean and variance alone.

## The extremal types theorem produces one family with three regimes

Suppose there exist constants

$$
a_n>0
$$

and

$$
b_n
$$

such that

$$
\frac{
M_n-b_n
}{
a_n
}
$$

converges in distribution to a non-degenerate limit. The Fisher-Tippett-Gnedenko extremal types theorem says that the limit must belong to the generalized extreme-value family. In its usual location-scale form,

$$
G(x)
=
\exp
\left\{
-
\left[
1+\xi
\left(
\frac{x-\mu}{\sigma}
\right)
\right]^{-1/\xi}
\right\},
$$

with

$$
\sigma>0
$$

and support restricted by

$$
1+\xi
\left(
\frac{x-\mu}{\sigma}
\right)>0.
$$

The limit as

$$
\xi\to0
$$

is the Gumbel form,

$$
G(x)
=
\exp
\left\{
-
\exp
\left[
-\frac{x-\mu}{\sigma}
\right]
\right\}.
$$

The parameter $\mu$ locates the extreme distribution and $\sigma$ sets its scale. The most important parameter for extrapolation is $\xi$, the extreme-value shape parameter.

When

$$
\xi>0,
$$

the distribution is of Fréchet type. The upper tail is heavy and unbounded. Pareto-type distributions fall in this class. If

$$
P(X>x)
\sim
cx^{-\alpha},
$$

then the extreme-value index is

$$
\xi
=
\frac{1}{\alpha}.
$$

When

$$
\xi=0,
$$

the distribution is of Gumbel type. The upper tail is unbounded but lighter than a power law. Gaussian and exponential distributions both lie in this domain of attraction even though their ordinary distributions are very different.

When

$$
\xi<0,
$$

the distribution is of Weibull type for maxima. The underlying distribution has a finite upper endpoint. Uniform and many bounded physical quantities are examples.

This classification is one of the deepest simplifications in extreme-value theory. The detailed form of $F$ can vary enormously, but the asymptotic behaviour of its maxima is governed by the tail class and, in particular, by $\xi$.

The same simplification also explains why a central fit can be misleading. A Gaussian model and a heavy-tailed model can agree remarkably well over the region where most observations occur while belonging to different extreme-value domains. Once extrapolation moves beyond the observed bulk, the asymptotic difference dominates.

## The first two moments do not determine the tail

Consider two distributions with mean zero and variance one. The first is standard normal,

$$
X_N
\sim
N(0,1).
$$

The second is a Student distribution with four degrees of freedom, rescaled to unit variance,

$$
X_T
=
\frac{1}{\sqrt2}
T_4.
$$

Because a Student $t$ distribution with four degrees of freedom has variance $2$, both $X_N$ and $X_T$ have

$$
\mathbb E[X]=0
$$

and

$$
\operatorname{Var}(X)=1.
$$

Near the centre, both are symmetric distributions of comparable scale. Their extreme tails are not remotely similar. The normal distribution is in the Gumbel domain with

$$
\xi=0,
$$

while the $t_4$ tail is regularly varying with exponent $4$, giving

$$
\xi=\frac14.
$$

The difference becomes visible in high quantiles.

| Upper probability | Standard normal quantile | Unit-variance $t_4$ quantile |
| ---: | ---: | ---: |
| 0.95 | 1.645 | 1.507 |
| 0.99 | 2.326 | 2.649 |
| 0.999 | 3.090 | 5.072 |
| 0.9999 | 3.719 | 9.216 |
| 0.99999 | 4.265 | 16.498 |

At the 95th percentile the two distributions are close. At the 99.999th percentile the heavy-tailed quantile is almost four times as large.

The contrast is even stronger if we fix a level rather than a probability. At

$$
x=5,
$$

the standard normal exceedance probability is approximately

$$
P(X_N>5)
\approx
2.87\times10^{-7}.
$$

For the unit-variance $t_4$ distribution,

$$
P(X_T>5)
\approx
1.06\times10^{-3}.
$$

The heavy-tailed model assigns the same event a probability about

$$
3.7\times10^3
$$

times larger.

Both models have exactly the same mean and variance. No moment calculation based only on those two quantities can determine which tail is correct.

The difference also predicts very different sample maxima. A rough location for the maximum of $n$ independent observations is the quantile

$$
F^{-1}
\left(
1-\frac{1}{n}
\right).
$$

For

$$
n=100\,000,
$$

the relevant upper probability is approximately

$$
0.99999.
$$

The normal model places the maximum near $4.27$, while the standardized $t_4$ model places it near $16.50$. A central model that cannot distinguish these two tails cannot answer a question about the maximum reliably.

This is the core reason EVT is not simply ordinary distribution fitting with more emphasis on large observations. The scientific object has changed. The relevant asymptotic information is in the tail class.

## Block maxima and threshold exceedances are two views of the same tail

The generalized extreme-value limit motivates the block-maxima method. Divide a sequence into blocks, such as years, seasons or batches, retain the maximum from each block, and fit a GEV model to those maxima. The method aligns directly with the limiting theorem and is often natural when the scientific quantity itself is an annual or batch maximum.

Its cost is information loss. If a year contains several exceptional events, only the largest survives. If the data record is short, the number of block maxima can be very small.

Peaks over threshold use more of the tail. Choose a high threshold $u$ and study the conditional excess

$$
Y
=
X-u
\mid
X>u.
$$

The Pickands-Balkema-de Haan theorem states that if the underlying distribution belongs to a maximum domain of attraction, then for sufficiently high $u$ the excess distribution is approximately generalized Pareto,

$$
P(Y\le y)
\approx
1-
\left(
1+\xi\frac{y}{\beta_u}
\right)^{-1/\xi},
$$

on the support where

$$
1+\xi\frac{y}{\beta_u}>0.
$$

When

$$
\xi=0,
$$

the limiting form is exponential,

$$
P(Y\le y)
\approx
1-
\exp
\left(
-\frac{y}{\beta_u}
\right).
$$

The important fact is that the GEV and GPD share the same shape parameter $\xi$. Block maxima and threshold exceedances are therefore not unrelated modelling tricks. They are two manifestations of the same underlying extreme-value limit theory.

If the target is a return level or a high quantile and enough raw data are available, peaks over threshold often use tail information more efficiently because all sufficiently large exceedances contribute. Block maxima can be preferable when natural blocks define the scientific quantity, when within-block observations are difficult to compare, or when only maxima have been recorded historically.

The choice should follow the data-generating process rather than fashion. The operational-risk article [Extreme Value Theory: Estimating the Tail You Have Not Seen](/statistics/extreme_value_theory_operational_risk/) develops the peaks-over-threshold approach through a full simulation. The present article focuses instead on the mathematical structure that justifies both approaches.

## Threshold selection is the central bias-variance trade-off

The generalized Pareto approximation is asymptotic. It becomes more defensible as the threshold moves further into the tail. Raising the threshold therefore reduces approximation bias. It also leaves fewer exceedances and increases statistical uncertainty.

This produces the defining practical trade-off in peaks-over-threshold analysis. A threshold that is too low includes observations that do not yet behave like the limiting tail model. A threshold that is too high produces a model based on very little data.

The GPD itself gives useful diagnostics. Suppose exceedances above $u$ follow a generalized Pareto model with shape $\xi$ and scale $\beta_u$. If we raise the threshold to

$$
v>u,
$$

the excess distribution remains GPD with the same shape parameter and updated scale

$$
\beta_v
=
\beta_u
+
\xi(v-u).
$$

The stability of $\xi$ across a range of high thresholds is therefore evidence that the tail approximation has entered a coherent regime.

For

$$
\xi<1,
$$

the mean excess exists. Under the GPD,

$$
e(u)
=
\mathbb E[X-u\mid X>u]
=
\frac{
\beta_u
}{
1-\xi
}.
$$

Using the threshold-stability relation,

$$
e(v)
=
\frac{
\beta_u+\xi(v-u)
}{
1-\xi
},
$$

which is linear in the threshold. This motivates the mean residual life plot. A roughly linear region supports, but does not prove, the use of a GPD above those thresholds.

Neither diagnostic supplies a unique mechanical threshold. In finite samples the estimates fluctuate, different diagnostics can disagree, and slowly converging distributions can retain bias far into the tail. A defensible analysis therefore reports sensitivity to a range of thresholds rather than presenting one threshold as though it were observed directly from nature.

The same logic appears in tail-index estimators such as the Hill estimator for positive $\xi$. Using more upper order statistics reduces variance and increases the risk of including non-tail observations. Using fewer order statistics reduces bias asymptotically and increases variance. The tuning parameter changes form, but the underlying problem is the same.

## Return levels are probabilities, not schedules

Extreme-value results are often communicated through return levels. Suppose annual maxima follow a GEV distribution $G$. A $T$-year return level $z_T$ is conventionally defined so that the probability of exceeding it in any one year is

$$
\frac{1}{T}.
$$

Equivalently,

$$
G(z_T)
=
1-\frac{1}{T}.
$$

For

$$
\xi\ne0,
$$

the GEV quantile formula gives

$$
z_T
=
\mu
+
\frac{\sigma}{\xi}
\left\{
\left[
-\log
\left(
1-\frac1T
\right)
\right]^{-\xi}
-
1
\right\}.
$$

For

$$
\xi=0,
$$

the limit becomes

$$
z_T
=
\mu
-
\sigma
\log
\left[
-\log
\left(
1-\frac1T
\right)
\right].
$$

The shape parameter has an increasingly large influence as $T$ grows. To see this, set

$$
\mu=0
$$

and

$$
\sigma=1.
$$

At a 100-block return period, the return levels are approximately

$$
3.01
$$

for

$$
\xi=-0.2,
$$

$$
4.60
$$

for

$$
\xi=0,
$$

and

$$
7.55
$$

for

$$
\xi=0.2.
$$

At a 1000-block return period, the same three models give approximately

$$
3.74,
\qquad
6.91,
\qquad
14.90.
$$

Small uncertainty in the shape parameter therefore becomes large uncertainty in far-tail extrapolation. This is exactly where narrow standard errors for central parameters can become misleading.

The phrase "100-year event" is also easy to misinterpret. It does not mean the event occurs once every hundred years on a deterministic schedule. If annual exceedances were independent with probability

$$
0.01
$$

per year, the probability of at least one exceedance during the next hundred years would be

$$
1-(0.99)^{100}
\approx
0.634.
$$

More generally,

$$
1-
\left(
1-\frac1T
\right)^T
\to
1-e^{-1}
\approx
0.632
$$

as $T$ grows. A return period is the reciprocal of an exceedance probability under a specified model. It is not a calendar promise.

## Dependence changes the effective number of extremes

The simplest extreme-value theory assumes independence. Environmental, financial, engineering and biological extremes often occur in clusters. Heat waves persist for several days. Market volatility creates runs of large losses. Storms can generate multiple high measurements within one event. Treating every exceedance as independent can therefore exaggerate the effective amount of tail information.

Under suitable dependence conditions, a stationary sequence can still have an extreme-value limit, but a new quantity appears: the extremal index

$$
\theta
\in
(0,1].
$$

If levels $u_n$ are chosen so that

$$
nP(X>u_n)
\to
\tau,
$$

then a common limiting form is

$$
P(M_n\le u_n)
\to
\exp(-\theta\tau).
$$

When

$$
\theta=1,
$$

the maximum behaves asymptotically like that of an independent sequence with the same marginal tail. Values

$$
\theta<1
$$

indicate clustering of extremes.

Under specific conditions, the reciprocal

$$
\frac1\theta
$$

can be interpreted in relation to an asymptotic mean cluster size, but that interpretation should not be applied mechanically in every dependence structure. The important point is that dependence changes how many effectively independent extreme episodes the record contains.

In practice this motivates declustering, cluster maxima, run-length methods or explicit time-series extreme-value models. The correct choice depends on whether the scientific target is the probability of individual exceedances, event clusters, episode maxima or cumulative impact.

Dependence also changes uncertainty. A record containing 500 threshold exceedances may contain much less than 500 independent pieces of tail information if those exceedances arrived in a small number of storms or stress episodes.

## EVT does not eliminate extrapolation risk

Extreme-value theory reduces the tail to a structured asymptotic problem, but it does not make far-tail inference easy. The rarest levels remain extrapolations beyond the observed record. The shape parameter is estimated from relatively little data, and far quantiles are highly sensitive to its uncertainty.

A complete EVT analysis therefore needs more than a maximum-likelihood estimate and one return level. Threshold or block sensitivity should be reported. Profile likelihood or bootstrap intervals should accompany remote return levels. Dependence should be assessed. Changes over time, covariate effects and non-stationarity should be considered when the physical process is evolving.

Stationarity is especially important. A 100-year return level estimated from a stationary model assumes that the distribution governing future extremes remains the same. If temperature, exposure, technology, land use, population, regulation or system capacity changes over time, the return level itself may be time-dependent. A precise stationary estimate can then answer a historical question more accurately than a future one.

There is also model risk in the asymptotic approximation. A distribution can belong to the Gumbel domain and still converge so slowly that finite-sample maxima behave poorly relative to the limiting GEV. Threshold estimates can drift over the entire observed range. Different plausible tail models can produce similar likelihoods near the data and sharply different extrapolations outside them.

This is not a reason to avoid EVT. It is a reason to report tail inference with the same seriousness given to any other inverse extrapolation problem. The further the return level lies beyond the record, the more the conclusion depends on asymptotic structure rather than direct observation.

The main practical error is to hide that dependence by fitting a familiar central distribution to all observations and reading off an extreme quantile. Such a model can appear more stable because most of the data constrain its centre tightly. That stability is irrelevant if the assumed tail class is wrong.

Extreme-value theory makes the extrapolation assumption explicit. It asks which tail domain the process belongs to, estimates the associated shape, and uses only the part of the data informative about that structure. The result is usually less comfortable than a central fit because uncertainty grows rapidly as we move deeper into the tail. That discomfort is often the correct statistical answer.

Averages and maxima do not obey the same asymptotic law. The central-limit theorem tells us why many sums become Gaussian. Extreme-value theory tells us why maxima converge to a different three-regime family and why high threshold excesses inherit the same tail shape through the generalized Pareto distribution. Mean and variance can describe the centre while saying almost nothing about the event that matters most.

When the scientific question is about the largest observation rather than the average one, the tail is not an afterthought. It is the model.

## References

Balkema, A. A., & de Haan, L. (1974). Residual life time at great age. *The Annals of Probability*, 2(5), 792–804.

Beirlant, J., Goegebeur, Y., Segers, J., & Teugels, J. (2004). *Statistics of Extremes: Theory and Applications*. Wiley.

Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.

Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.

Fisher, R. A., & Tippett, L. H. C. (1928). Limiting forms of the frequency distribution of the largest or smallest member of a sample. *Proceedings of the Cambridge Philosophical Society*, 24(2), 180–190.

Leadbetter, M. R., Lindgren, G., & Rootzén, H. (1983). *Extremes and Related Properties of Random Sequences and Processes*. Springer.

Pickands, J. (1975). Statistical inference using extreme order statistics. *The Annals of Statistics*, 3(1), 119–131.

Resnick, S. I. (1987). *Extreme Values, Regular Variation, and Point Processes*. Springer.
