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

That is not an estimation failure. It is a statement about what can be learned from the available information.

## A simple example

Suppose the model implies only

$$
\theta \ge a
$$

and

$$
\theta \le b.
$$

Then the information content of the model is

$$
\theta\in[a,b].
$$

Choosing the midpoint

$$
\widehat\theta=\frac{a+b}{2}
$$

may be convenient, but it is not an identified fact. It adds a decision rule that was never implied by the data.

This is the discipline of partial identification: **do not confuse a convenient representative of a set with a parameter that the model uniquely determines**.

## Three different uncertainties

Point estimates can hide three distinct problems:

1. sampling uncertainty;
2. model uncertainty;
3. identification uncertainty.

More data can reduce sampling uncertainty. More data do not automatically eliminate identification uncertainty.

Even with an arbitrarily large sample, the correct answer may remain

$$
\boxed{\theta\in\Theta_I}
$$

rather than

$$
\boxed{\theta=\theta_0}.
$$

## Why moment inequalities arise naturally

Inequalities appear whenever theory provides bounds rather than exact equalities. Examples include revealed-preference restrictions, incomplete models, missing counterfactual information, interval observations, strategic interactions, and monotonicity or sign restrictions.

The modelling question is not

> How do I get a point estimate anyway?

It is

$$
\boxed{
\text{What restrictions can I actually defend?}
}
$$

Once those restrictions are explicit, the identified set becomes part of the scientific result.

## Sample feasibility is not population identification

In data we replace expectations by sample moments,

$$
\widehat g_j(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}m_j(W_i,\theta).
$$

A naive procedure would retain every $\theta$ satisfying

$$
\widehat g_j(\theta)\ge0
\quad\text{for all }j.
$$

But finite samples introduce noise. A population-compatible value can produce a slightly negative sample moment, while an incompatible value can look acceptable by chance.

The problem therefore becomes inferential, not merely computational.

## Boundaries are where the problem becomes difficult

Moment-inequality procedures are often most delicate near

$$
E[m_j(W,\theta)] = 0.
$$

A strongly positive moment is easy to classify as non-binding. A clearly negative moment gives evidence against the candidate parameter value. Near zero, sampling variation changes which restrictions appear active.

That is why the geometry of the identified set matters. Inference depends not only on how many inequalities exist, but also on which are binding or nearly binding.

## A wide set can be informative

A wide identified set is sometimes treated as disappointing.

That is often backwards.

If the available information supports only a wide region, reporting that width tells us exactly what is missing. It can reveal which assumptions carry identification and which additional measurements or design changes would be valuable.

A narrow estimate produced by an unjustified restriction can hide all of that.

Partial identification is therefore not a weaker version of point estimation. It is a framework for separating **what the data say** from **what the analyst added**.

## The practical habit

Before selecting an estimator, ask

$$
\boxed{
\text{What does the model identify in the population?}
}
$$

If the answer is a set, keep the set.

Then build inference around that object instead of collapsing it prematurely into a number.

Precision is valuable only after identification has earned it.
