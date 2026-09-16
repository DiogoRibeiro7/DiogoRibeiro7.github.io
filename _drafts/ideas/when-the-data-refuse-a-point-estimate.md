---
author_profile: false
categories:
- Statistics
classes: wide
excerpt: Sometimes the data and assumptions identify a set rather than a single parameter. Partial identification is about respecting that boundary instead of manufacturing false precision.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- partial identification
- moment inequalities
- identified sets
- econometrics
- inference
seo_description: An introduction to partial identification, moment inequalities, and why a set can be the statistically correct answer.
seo_title: When the Data Refuse a Point Estimate
seo_type: article
summary: Why some problems identify sets rather than points, and why forcing a single estimate can create precision the model has not earned.
tags:
- Statistics
- Econometrics
- Inference
title: 'When the Data Refuse a Point Estimate'
---

Statistical work often begins with an assumption that is so familiar we barely notice it: somewhere in the data there is a single parameter waiting to be estimated.

Sometimes there is not.

The data, together with assumptions we are willing to defend, may identify only a **set of plausible parameter values**. Forcing that problem into a point estimate does not create information. It creates precision that the model has not earned.

That is the central idea of partial identification.

## Identification comes before estimation

Suppose $\theta$ is the quantity of interest and the model implies moment inequalities

$$
E[m_j(W,\theta)] \ge 0,
\qquad j=1,\ldots,J.
$$

The population identified set is

$$
\Theta_I
=
\left\{
\theta\in\Theta:
E[m_j(W,\theta)]\ge 0
\text{ for all }j
\right\}.
$$

If $\Theta_I$ contains one point, the parameter is point identified. If it contains an interval, region, or several admissible values, then the model is partially identified.

This distinction is about the **population model**, not about finite-sample noise.

A parameter can be partially identified even with an arbitrarily large dataset. More data may sharpen our knowledge of $\Theta_I$, but they do not automatically collapse the set to a point.

## A simple example: interval information

Suppose the target parameter is a population mean

$$
\theta = E[Y],
$$

but $Y$ is not observed exactly. Instead we observe bounds

$$
L \le Y \le U.
$$

Taking expectations gives

$$
E[L] \le E[Y] \le E[U].
$$

Therefore

$$
\boxed{
\theta \in [E[L],E[U]].
}
$$

This is already an identified set.

Choosing the midpoint

$$
\frac{E[L]+E[U]}{2}
$$

may be convenient, but that midpoint is not identified by the observed information. It becomes a point only after adding another assumption or decision rule.

This is the basic discipline of partial identification:

> **Do not confuse a convenient representative of a set with a parameter the model uniquely determines.**

## The same idea as moment inequalities

The interval example can be written as two moment inequalities:

$$
E[\theta-L] \ge 0,
$$

and

$$
E[U-\theta] \ge 0.
$$

Together they imply

$$
\theta \ge E[L]
$$

and

$$
\theta \le E[U].
$$

So even a very simple bounds problem already has the structure

$$
E[m_j(W,\theta)]\ge0.
$$

More complicated partially identified models use the same logic with more moments, more parameters, or more complicated geometry.

## Three different uncertainties

Point estimates can hide three distinct sources of uncertainty:

1. **sampling uncertainty** — we observe a finite sample rather than the full population;
2. **model uncertainty** — conclusions depend on assumptions we choose to impose;
3. **identification uncertainty** — even at the population level, the maintained assumptions may imply a set rather than a point.

These should not be collapsed into one vague idea of “uncertainty.”

Sampling uncertainty can shrink with more data.

Identification uncertainty may remain because it is structural.

That is why the large-sample target may still be

$$
\boxed{\theta\in\Theta_I}
$$

rather than

$$
\boxed{\theta=\theta_0}.
$$

## Estimating the identified set is not the same as inference

In data, population moments are replaced by sample analogues:

$$
\widehat g_j(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}m_j(W_i,\theta).
$$

A natural plug-in estimator of the set would be

$$
\widehat\Theta_n
=
\left\{
\theta:
\widehat g_j(\theta)\ge0
\text{ for all }j
\right\}.
$$

But finite-sample noise creates a problem.

A population-compatible $\theta$ can produce a slightly negative sample moment, while an incompatible value can look acceptable by chance.

So there are really two different tasks:

$$
\boxed{
\text{estimate the identified set}
}
$$

and

$$
\boxed{
\text{construct confidence regions with valid coverage}
}
$$

They are related, but they are not the same problem.

A confidence region must account for the fact that the sample only gives noisy information about the inequalities that define the population set.

## Why the boundary is difficult

Moment-inequality procedures are most delicate near

$$
E[m_j(W,\theta)] = 0.
$$

If a moment is strongly positive, the corresponding inequality is comfortably satisfied.

If it is clearly negative, the candidate parameter value is incompatible with the model.

The difficult case is a moment close to zero.

Then sampling noise can change whether the restriction appears binding, non-binding, or violated. This is why procedures for moment inequalities often care explicitly about **which moments are binding or nearly binding**.

The geometry matters.

Two candidate parameter values can satisfy the same number of inequalities while having very different distances from the boundary.

## Why moment selection appears

Suppose there are many inequalities and only a few are close to binding for a given $\theta$.

Treating all moments as equally informative can make inference needlessly conservative, because moments that are strongly slack contribute little to whether the candidate parameter lies near the boundary of the identified set.

Generalized moment selection procedures address this by using the data to distinguish approximately binding moments from clearly slack ones while maintaining valid inference.

The idea is not to discard inconvenient inequalities. It is to avoid letting obviously non-binding restrictions dominate the critical value used for a local decision.

That distinction becomes particularly important when $J$ is not tiny or the geometry changes across the parameter space.

## Stronger assumptions shrink the set

One of the most useful features of partial identification is that it makes the role of assumptions visible.

Start with a weak assumption set $\mathcal A_1$ and obtain

$$
\Theta_I(\mathcal A_1).
$$

Add a stronger assumption set $\mathcal A_2$ and obtain

$$
\Theta_I(\mathcal A_2).
$$

Typically,

$$
\mathcal A_2 \supset \mathcal A_1
\quad\Longrightarrow\quad
\Theta_I(\mathcal A_2)
\subseteq
\Theta_I(\mathcal A_1),
$$

provided the added assumptions are genuine restrictions on the model.

This is useful because the gain in precision is no longer free or hidden. We can ask exactly which assumption made the set narrower.

That creates a natural sensitivity analysis:

$$
\text{weak assumptions}
\rightarrow
\text{wide set}
\rightarrow
\text{add justified structure}
\rightarrow
\text{narrower set}.
$$

If the set collapses only after a controversial assumption, that is scientifically important information.

## A wide set can be informative

A wide identified set is sometimes treated as disappointing.

That is often backwards.

If the available information supports only a wide region, reporting that width tells us exactly what is missing. It can reveal which assumptions carry identification and which additional measurements, instruments, or design changes would be valuable.

A narrow estimate produced by an unjustified restriction can hide all of that.

Partial identification is therefore not a weaker version of point estimation. It is a framework for separating

$$
\boxed{
\text{what the data and maintained assumptions imply}
}
$$

from

$$
\boxed{
\text{what the analyst adds to obtain more precision}.
}
$$

## The practical habit

Before selecting an estimator, ask

$$
\boxed{
\text{What does the model identify in the population?}
}
$$

If the answer is a set, keep the set.

Then ask a second question:

$$
\boxed{
\text{How should sampling uncertainty around that set be quantified?}
}
$$

Only after those two questions are separated does it make sense to discuss computation, optimization, or a preferred point summary.

Precision is valuable only after identification has earned it.

## References

- Manski CF. *Partial Identification of Probability Distributions*. Springer; 2003.
- Tamer E. Partial Identification in Econometrics. *Annual Review of Economics*. 2010;2:167–195.
- Chernozhukov V, Hong H, Tamer E. Estimation and Confidence Regions for Parameter Sets in Econometric Models. *Econometrica*. 2007;75(5):1243–1284.
- Andrews DWK, Soares G. Inference for Parameters Defined by Moment Inequalities Using Generalized Moment Selection. *Econometrica*. 2010;78(1):119–157.
- Kline B, Tamer E. Recent Developments in Partial Identification. *Annual Review of Economics*. 2023;15:125–150.
