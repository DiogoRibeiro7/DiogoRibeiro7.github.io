---
permalink: '/statistics/why_exact_post_selection_confidence_intervals_can_be_enormous/'
title: 'Why Exact Post-Selection Confidence Intervals Can Be Enormous'
categories:
- Statistics
- Data Science
tags:
- Selective Inference
- Confidence Intervals
- Post Selection Inference
- Gaussian Models
- Statistical Computing
author_profile: false
seo_title: 'Why Exact Post-Selection Confidence Intervals Can Be Enormous'
seo_description: 'After selecting a signal because it looked unusually large, an exact confidence interval can become spectacularly wide. That is often the mathematically correct cost of conditioning on the search, not a software bug.'
excerpt: >-
  A very wide interval is not automatically evidence that an inferential procedure
  is broken. After selection, weak evidence can force an exact conditional interval
  to admit that the parameter is only poorly identified by the selected observation.
summary: >-
  A simple Gaussian screening example showing how a naive interval can look precise
  after selection while the exact conditional interval becomes enormous. The article
  explains truncation, weak post-selection evidence, inversion of the selective pivot,
  and why wide intervals are often the honest result.
keywords:
- selective inference
- confidence intervals
- post-selection inference
- Gaussian truncation
- winner's curse
- statistical uncertainty
classes: wide
date: '2026-09-16'
why_this_exists: >-
  Exact post-selection intervals often surprise users because they can become far
  wider than ordinary intervals precisely when a selected signal barely clears the
  selection threshold. This article explains that geometry with a self-contained
  one-dimensional example.
evidence: >-
  Motivated by stress tests of exact selective procedures in which null-data
  intervals were occasionally tens, hundreds, or thousands of standard deviations
  wide despite numerically stable inversion and correct calibration.
methodology: >-
  Conditions a Gaussian observation on passing a fixed screening threshold, derives
  the corresponding truncated-normal pivot, and numerically inverts it for several
  observed values to compare naive and selection-adjusted confidence intervals.
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

A confidence interval that runs from roughly \(-72\) to \(2.5\) for a unit-variance Gaussian mean looks ridiculous.

If I saw such an interval unexpectedly, I would check the implementation too.

But sometimes that interval is exactly what the mathematics demands.

The reason is selection.

Suppose we only report an estimate when it looks sufficiently interesting. We search, screen, rank, optimize, choose the largest signal, or keep only effects that clear a threshold. Then we construct an interval as though the reported estimate had been fixed in advance.

That interval is usually too optimistic.

Once we condition on the fact that the estimate was selected because it looked large, weak evidence can become almost uninformative.

The basic lesson is:

$$
\boxed{
\text{after selection, a huge interval can be evidence of honesty rather than failure.}
}
$$

## A Minimal Example

Let

$$
X\sim N(\mu,1).
$$

Imagine a simple reporting rule:

$$
\text{report }X\quad\text{only if}\quad X>2.
$$

This is a toy version of a much broader pattern. We inspect many possibilities, keep something only after it looks sufficiently extreme, and then want uncertainty for the selected quantity.

Suppose we observe

$$
X=2.05.
$$

If we ignore selection, the ordinary 95% Gaussian interval is

$$
2.05\pm1.96,
$$

or approximately

$$
\boxed{[0.09,4.01]}.
$$

That looks reasonably informative.

It also treats \(2.05\) as though we would have reported it regardless of whether it had crossed the threshold.

But we already know something else:

$$
X>2.
$$

The observation was selected because it landed in the upper tail.

That changes the sampling distribution relevant for inference.

## Condition on the Selection Event

Given the event

$$
X>2,
$$

the conditional distribution is a lower-truncated Gaussian.

For an observed value \(x>2\), the selective conditional CDF is

$$
F_\mu(x\mid X>2)
=
\frac{
\Phi(x-\mu)-\Phi(2-\mu)
}{
1-\Phi(2-\mu)
}.
$$

Equivalently, using Gaussian survival functions,

$$
F_\mu(x\mid X>2)
=
1-
\frac{
\overline\Phi(x-\mu)
}{
\overline\Phi(2-\mu)
}.
$$

For the true \(\mu\), this conditional pivot is uniform on \([0,1]\).

So an exact equal-tailed 95% conditional interval is obtained by solving

$$
F_{\mu_L}(x\mid X>2)=0.975
$$

and

$$
F_{\mu_U}(x\mid X>2)=0.025.
$$

For

$$
x=2.05,
$$

the solutions are approximately

$$
\boxed{
\mu_L=-71.74,
\qquad
\mu_U=2.53.
}
$$

So the exact conditional 95% interval is roughly

$$
\boxed{[-71.7,2.53]}.
$$

That is not a typo.

## Why Does the Lower Limit Explode?

The observation barely cleared the screening threshold.

Under a very negative mean, observing a value above 2 is extremely rare.

But conditional on the rare event already having happened, values just above 2 are exactly what we would expect to see.

That distinction is the whole story.

Unconditionally, \(X=2.05\) looks incompatible with a very negative mean.

Conditionally on

$$
X>2,
$$

a value such as \(2.05\) can be quite ordinary even when the selection event itself was astronomically unlikely.

Selective inference does not ask

> How surprising was it that anything crossed the threshold?

after we have conditioned on crossing it.

It asks

> Given that the threshold was crossed, how informative is the exact observed value about \(\mu\)?

When the observation is only barely above the threshold, the answer can be: not very informative at all.

## Conditioning Changes the Question

This is where selective inference often feels counterintuitive.

Before conditioning, the probability

$$
P_\mu(X>2)
$$

contains information about \(\mu\).

After conditioning on

$$
X>2,
$$

that probability is removed from the likelihood relevant to the selective pivot.

We deliberately pay that price in exchange for valid inference after the data-dependent selection rule.

The interval therefore reflects a narrower source of information:

$$
\text{where did the selected observation land inside the selection region?}
$$

If it landed barely inside, there may be little left to learn.

## The Naive Interval Uses Selection Twice

The naive interval looks precise because the same extreme observation performs two jobs.

First, it gets selected because it is large.

Second, its largeness is treated as fresh evidence that \(\mu\) is large.

Symbolically,

$$
\text{large }X
\rightarrow
\text{selection}
$$

and then again

$$
\text{large }X
\rightarrow
\text{tight positive interval}.
$$

That double use of the same fluctuation is the source of selection bias.

The selective procedure conditions on the first use and asks what remains for the second.

## The Cost Is Largest for Marginal Selections

The effect is not equally severe for every selected value.

Take several observations above the same threshold.

For \(x=2.05\):

$$
\text{selective CI}\approx[-71.7,2.53].
$$

For \(x=2.5\):

$$
\text{selective CI}\approx[-4.99,4.31].
$$

For \(x=3\):

$$
\text{selective CI}\approx[-0.93,4.93].
$$

For \(x=3.5\):

$$
\text{selective CI}\approx[0.66,5.46].
$$

Now the lower limit becomes positive.

Why?

Because once the selected observation lies comfortably inside the selection region rather than just scraping past its boundary, the conditional position carries much more information.

The problem is therefore not selection alone.

It is **weak evidence after selection**.

## A Useful Diagnostic Quantity

Define the distance above the threshold

$$
d=x-2.
$$

When

$$
d\approx0,
$$

the observation is a marginal selection.

The selective distribution is then highly skewed as a function of \(\mu\), and inversion can send one confidence limit very far away.

As \(d\) grows, the observation becomes less compatible with being a mere threshold-crossing accident.

The interval contracts accordingly.

## Search Creates the Same Geometry

The threshold example is deliberately simple, but the same logic appears after search.

Suppose we inspect candidate statistics

$$
T_1,\ldots,T_m
$$

and select

$$
\widehat j
=
\arg\max_j |T_j|.
$$

Inference for the selected effect must acknowledge the event

$$
\widehat j=j
$$

and often also the selected sign.

Conditioning on that event truncates the distribution of the selected contrast.

If the winner is only marginally larger than its competitors, the allowed truncation region can begin very close to the observed contrast.

Inverting the corresponding conditional pivot can then produce a spectacularly wide interval.

That is the multidimensional analogue of observing \(X=2.05\) after requiring \(X>2\).

## Why a Narrow Interval Can Be More Suspicious

Suppose we search over many candidate breakpoints, choose the one with the strongest apparent change, and then report an ordinary interval centered at that selected effect.

If the evidence for the break is weak, a narrow interval is not reassuring.

It may be evidence that we ignored the search.

The data were allowed to optimize the estimate first and then the uncertainty calculation pretended the location had been fixed in advance.

The exact selective interval does the opposite.

When the selection was marginal, it becomes wide because it recognizes how little independent information remains after conditioning on the search.

So in that setting:

$$
\boxed{
\text{weak selection evidence}
\Longrightarrow
\text{wide exact conditional interval}
}
$$

is not pathological.

It is the expected behavior.

## A Stress Test From Real Statistical Software Work

In one selective-inference implementation I stress-tested, null data with only 40 observations and unit variance produced the following interval-width distribution over 1000 random draws:

| Summary | Width |
| --- | ---: |
| Median | 4.7 |
| 90th percentile | 23.8 |
| Maximum | 5817.9 |

Roughly one interval in eight exceeded 20 standard deviations in width.

At first glance, that sounds like a numerical catastrophe.

But the root-finding procedure was stable, the selection event was valid, and the exact conditional pivot was doing what it was supposed to do.

The enormous intervals clustered around weak, marginal selections.

The right software change was therefore not to "fix" the intervals.

It was to document them.

## Numerical Failure and Statistical Uninformativeness Are Different

This distinction matters a lot in statistical software.

A numerical failure might look like:

- root finding does not bracket a solution;
- probabilities become `NaN`;
- monotonicity needed for inversion is violated numerically;
- the interval changes materially under harmless rescaling;
- equivalent formulations disagree.

A statistically uninformative result can look like:

$$
[-3000,3000].
$$

Those are not the same class of event.

A huge finite interval may be mathematically valid.

Software should therefore distinguish

$$
\boxed{
\text{computation failed}
\neq
\text{inference succeeded but learned almost nothing}.
}
$$

## Exact Does Not Mean Informative

The phrase "exact interval" is easy to overinterpret.

Exactness is about calibration under the stated model and conditioning event.

It does not mean the interval will be short.

It does not mean the data contain much information.

It does not mean the selected signal is scientifically convincing.

It does not mean the selection rule was efficient.

An exact method can return a uselessly wide interval because that is the correct representation of the available information.

This is no different in spirit from an ordinary confidence interval becoming wide when the sample is tiny or the noise is large.

Selection simply creates a more dramatic version of the same principle.

## The Width Is Telling You Something

A huge selective interval is not merely an inconvenience.

It is a diagnostic.

It says that the selected estimate owes a substantial part of its apparent extremeness to the selection mechanism itself.

The data do not strongly distinguish

$$
\text{a genuinely large parameter}
$$

from

$$
\text{an ordinary parameter that happened to win the search}.
$$

That is scientifically meaningful information.

## Do Not Cap the Interval for Presentation

One tempting response is to clip extreme intervals to a visually convenient range.

For example, report

$$
[-10,10]
$$

instead of

$$
[-71.7,2.53].
$$

That may make the plot readable, but it changes the inferential object.

If visual clipping is unavoidable, it should be explicit:

- preserve the actual interval numerically;
- mark the plotted boundary as truncated;
- state the true endpoint in text or a table.

Never silently convert an uninformative interval into a prettier one.

## Do Not Replace It With the Naive Interval Either

Another reaction is to say that the exact method is "too conservative" and report the unadjusted interval instead.

But in the screening example, the naive interval

$$
[0.09,4.01]
$$

looks persuasive largely because the reporting rule guaranteed that we would only examine observations above 2.

If we repeatedly use that workflow under a null or weak-signal regime, the nominal 95% interpretation no longer applies to the selected intervals.

The narrowness is purchased by ignoring the data-dependent selection step.

## Selection-Adjusted Inference Has a Different Target

This point is subtle but important.

Conditional selective inference typically targets a parameter after conditioning on the selection event.

The guarantee is therefore conditional:

$$
P_\mu\{
\mu\in C(X)
\mid
\text{selection event}
\}
=1-\alpha
$$

under the model and exact conditioning scheme.

That is not the same probability statement as an unconditional confidence interval constructed before any selection.

The interval width reflects the information left after the conditioning.

## Why Conditioning Can Feel Wasteful

Conditioning throws away information by design.

That may seem inefficient.

It is.

But the conditioning is also what makes the post-selection probability statement tractable and valid.

There are other approaches to post-selection inference, including data splitting, randomized selection, simultaneous inference, debiasing, selective likelihood methods, and Bayesian formulations.

They make different trade-offs.

The lesson is not that one should always condition as aggressively as possible.

The lesson is that when we **do** claim exact conditional validity, the resulting loss of information is real and should not be hidden.

## A Comparison With Data Splitting

Suppose instead we split the data into two independent parts.

Use the first part for selection.

Use the second part for inference.

Then the inferential sample is independent of the selection event.

The resulting interval may be easier to interpret, but it pays another price: only part of the data is used for estimation.

So there is no free lunch.

Selective conditioning spends information by conditioning.

Data splitting spends information by withholding observations from estimation.

Both are ways of avoiding the naive reuse of the same random fluctuation for selection and inference.

## Report Selection Strength Alongside the Interval

In applications, I like to accompany selective intervals with some measure of how strongly the observation cleared the selection rule.

In the threshold example that could simply be

$$
x-2.
$$

In a search problem it might be the gap between the winning statistic and the runner-up, the global search-adjusted p-value, or another diagnostic tied to the selection mechanism.

This helps explain why two selected effects of similar magnitude can have very different post-selection uncertainty.

## Software Should Warn About Interpretation, Not About Correctness

If huge intervals are expected under weak selection, emitting a warning such as

> numerical inversion may have failed

would be misleading.

A better documentation message is conceptual:

> Exact selective intervals can become extremely wide when the selected signal is marginal because conditioning on the selection event leaves little information about the target parameter.

That tells the user what happened without implying a computational error.

## Stress Tests Should Include Weak Selections

If a selective-inference implementation is tested only on strong planted signals, it can look beautifully behaved.

The difficult cases are weak signals and null data.

That is where truncation boundaries approach the observation and confidence intervals become extreme.

A useful stress-test suite should therefore include:

- null data;
- barely selected cases;
- strong selected cases;
- extreme truncation;
- sign reversals;
- equivalent parameterizations;
- numerical tail stability.

The goal is not to make every interval look reasonable.

The goal is to verify that unreasonable-looking intervals are unreasonable for the right statistical reason.

## A Tiny Reproducible Calculation

Here is the core computation in Python.

```python
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def selective_cdf(mu, x, threshold=2.0):
    log_den = norm.logsf(threshold - mu)
    log_upper = norm.logsf(x - mu)
    return 1.0 - np.exp(log_upper - log_den)


def endpoint(target, x, threshold=2.0):
    f = lambda mu: selective_cdf(mu, x, threshold) - target
    grid = np.linspace(-100.0, 100.0, 40001)

    left = grid[0]
    f_left = f(left)

    for right in grid[1:]:
        f_right = f(right)
        if f_left * f_right < 0:
            return brentq(f, left, right)
        left = right
        f_left = f_right

    return None


x = 2.05
lower = endpoint(0.975, x)
upper = endpoint(0.025, x)

print(lower, upper)
```

The survival-function formulation matters numerically because direct subtraction of two Gaussian CDF values can lose precision deep in the tail.

The resulting endpoints are approximately

```text
-71.7390  2.5305
```

The strange answer survives a numerically stable calculation.

That is the point.

## What the Interval Does Not Mean

The interval

$$
[-71.7,2.53]
$$

does not mean that \(-71\) is a plausible unconditional explanation for observing \(X=2.05\).

Unconditionally, it is fantastically implausible.

The interval means that **conditional on already knowing that the observation exceeded 2**, the exact location \(2.05\) provides very little evidence against such negative means.

That conditional statement is narrower and stranger than the unconditional intuition most of us bring to the problem.

## The Broader Lesson

Selection creates apparent evidence.

Sometimes the selected estimate remains compelling after adjusting for that fact.

Sometimes nearly all of the apparent evidence disappears.

When that happens, the confidence interval can become enormous.

The software has not necessarily failed.

The inferential procedure may be telling us something uncomfortable but correct:

$$
\boxed{
\text{we found an interesting-looking winner, but after accounting for how it was chosen, we do not know its effect very precisely.}
}

That is not a defect in uncertainty quantification.

It is exactly what uncertainty quantification is for.
