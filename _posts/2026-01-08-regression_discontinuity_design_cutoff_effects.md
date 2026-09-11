---
permalink: '/statistics/regression_discontinuity_design_cutoff_effects/'
title: 'Regression Discontinuity: Estimating an Effect From the Rule That Assigns It'
categories:
- Statistics
tags:
- Causal Inference
- Experimental Design
- Econometrics
- Statistics
author_profile: false
seo_title: 'Regression Discontinuity Design'
seo_description: 'When a threshold on a score decides who gets a treatment, the units just either side of it are as good as randomised. A simulation shows the local estimate recovering the effect, the bandwidth trading curvature bias against noise, high-order polynomials adding noise, placebo cutoffs at zero, and what manipulation of the score does.'
excerpt: >-
  Customers with a risk score of 600 or more get the credit line; those
  below do not. Comparing everyone above with everyone below gives an
  effect of 8.5. The effect at the threshold, which is the only one the
  rule can identify, is 2.0.
summary: >-
  Why a sharp cutoff makes units near it comparable and units far from
  it not, the local linear estimator and what it identifies, a simulation
  with a trend that bends differently on each side showing the naive
  comparison off by a factor of four, the bandwidth trade-off between
  curvature bias and noise, why global polynomials of high degree are the
  wrong fix, placebo cutoffs as a falsification test, what manipulation
  of the running variable does to the estimate and how the density test
  detects it, and how the precision scales with sample size.
keywords:
  - regression discontinuity
  - running variable
  - cutoff
  - local linear regression
  - bandwidth
  - McCrary density test
  - causal inference
classes: wide
date: '2026-01-08'
why_this_exists: >-
  Eligibility rules, score thresholds and rank cutoffs are everywhere in
  operations and policy, and each one is an experiment waiting to be
  analysed. This post shows how to read the effect off the cutoff, what
  can go wrong, and the checks that tell a credible estimate from a
  fragile one.
evidence: >-
  A simulated population of 5,000 units with a running variable uniform
  on minus 50 to 50, a sharp cutoff at zero, a true effect of 2 at the
  cutoff and a trend whose curvature differs on the two sides; 2,000
  replications per estimator, plus manipulation scenarios in which a
  share of units just below the cutoff move above it.
methodology: >-
  Compares the naive above-versus-below difference, local linear
  estimates at six bandwidths, global polynomials of degree one to six,
  placebo cutoffs away from the threshold, the estimate and the density
  ratio at the cutoff under manipulation, and the standard error against
  sample size.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-geometry.jpg
  og_image: /assets/images/headers/photo-geometry.jpg
  overlay_image: /assets/images/headers/photo-geometry.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-geometry.jpg
  twitter_image: /assets/images/headers/photo-geometry.jpg
---
A bank extends a credit line to customers whose internal score is at least 600. A year later the question is what the credit line did to their spending, and the first analysis compares customers who got it with customers who did not: 8.5 points more spending. Nobody believes that number, because customers above 600 are richer, older and more settled than customers below, and they would spend more with or without the line.

The rule that created the problem also contains its solution. A customer scored 599 and a customer scored 601 are, in every respect that matters, the same customer; the score has noise in it, and which side of the threshold a near-600 customer falls on is close to a coin toss. The credit line is the only thing that changed at 600. The difference in spending between customers just above and just below the cutoff is therefore the effect of the credit line, for customers at the cutoff, and the simulation this post is built on puts it at 2.0.

## Comparable at the Cutoff, Not Elsewhere

A regression discontinuity design has a running variable $X$, a cutoff $c$, and a treatment assigned by whether $X$ crosses $c$. The outcome $Y$ depends on $X$ through some smooth trend, and on the treatment. If the trend is continuous at $c$, any jump in $Y$ at $c$ is the treatment's doing, and the estimator is the difference between the limit of $Y$ from above and the limit from below.

Limits are estimated from data near the cutoff, and the standard estimator fits a straight line to the outcome on each side within a bandwidth $h$ and takes the difference of the intercepts. The assumption that earns the causal reading is continuity: nothing else changes discontinuously at $c$. That excludes other rules using the same threshold, and it excludes units choosing which side to be on, which is the manipulation case below.

What the design identifies is the effect for units at the cutoff. Customers at 800 might respond to a credit line quite differently from customers at 600, and the design says nothing about them. That is a real limitation, and it is also the reason the estimate is credible: it comes from the one place where the rule made the comparison fair.

## A Simulation

Five thousand units with a running variable uniform on $[-50, 50]$ and a cutoff at zero. The outcome has a linear trend plus a curvature that differs on the two sides, which is what real trends do, and a true effect of 2 at the cutoff.

```python
import numpy as np

rng = np.random.default_rng(0)
tau, n = 2.0, 5000

def draw(r=rng, n=n, manipulate=0.0):
    """Running variable on [-50, 50], cutoff at 0, a trend that bends differently on each side.
    manipulate: share of units just below the cutoff that push themselves just above it."""
    x = r.uniform(-50, 50, n)
    if manipulate > 0:
        near = (x < 0) & (x > -5)
        push = near & (r.random(n) < manipulate)
        x = np.where(push, r.uniform(0, 3, n), x)
    d = (x >= 0).astype(float)
    curve = np.where(x >= 0, 0.002, -0.001) * x ** 2
    y = 10 + 0.08 * x + curve + tau * d + r.normal(0, 4, n)
    if manipulate > 0:                       # the pushers are the motivated ones
        y = y + 3.0 * push
    return x, d, y

def local_linear(x, y, h, cutoff=0.0):
    """Separate lines within h of the cutoff on each side; the effect is the difference of intercepts."""
    est = []
    for side in (x >= cutoff, x < cutoff):
        m = side & (np.abs(x - cutoff) <= h)
        X = np.column_stack([np.ones(m.sum()), x[m] - cutoff])
        b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        est.append(b[0])
    return est[0] - est[1]

def global_poly(x, d, y, degree):
    cols = [np.ones(len(x)), d] + [x ** k for k in range(1, degree + 1)] + [d * x ** k for k in range(1, degree + 1)]
    b, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return b[1]

reps = 2000
naive = [y[d == 1].mean() - y[d == 0].mean() for x, d, y in (draw() for _ in range(reps))]
print(f"naive above vs below: {np.mean(naive):.2f}")
for h in (2, 5, 10, 20, 35, 50):
    est = np.array([local_linear(x, y, h) for x, d, y in (draw() for _ in range(reps))])
    print(f"bandwidth {h:>2}: mean {est.mean():.2f}, bias {est.mean()-tau:+.2f}, sd {est.std():.2f}, "
          f"rmse {np.sqrt(np.mean((est-tau)**2)):.2f}")
```

The naive comparison of everyone above the cutoff with everyone below gives 8.50, more than four times the truth, because it compares units 25 points apart on average along a trend that rises with the score. The local estimates:

| Bandwidth | Units used | Mean estimate | Bias | Standard deviation | Root mean squared error |
| --- | --- | --- | --- | --- | --- |
| 2 | 200 | 2.05 | +0.05 | 1.14 | 1.14 |
| 5 | 500 | 1.98 | -0.02 | 0.70 | 0.70 |
| 10 | 1,000 | 1.95 | -0.05 | 0.51 | 0.51 |
| 20 | 2,000 | 1.80 | -0.20 | 0.36 | 0.41 |
| 35 | 3,500 | 1.38 | -0.62 | 0.27 | 0.67 |
| 50 | 5,000 | 0.75 | -1.25 | 0.22 | 1.27 |

Two things move in opposite directions as the window widens. The standard deviation falls, because more units enter the fit. The bias grows, because a straight line is a worse description of a curved trend over a longer stretch, and the curvature differs across the cutoff so the errors do not cancel. At a bandwidth of 50, which uses every unit, the estimate is 0.75 against a truth of 2, with a tight standard deviation that would make the wrong answer look precise. The root mean squared error is smallest around a bandwidth of 20. That is the trade-off every bandwidth selector, such as the Imbens-Kalyanaraman or Calonico-Cattaneo-Titiunik rules, is built to make from the data.

![Bias, standard deviation and root mean squared error of the local linear estimate against bandwidth. Narrow windows are unbiased and noisy; wide ones are precise and biased; the error is smallest at an intermediate width.](/assets/images/figures/rdd_bandwidth_tradeoff.png){: width="1152" height="672" loading="lazy"}

## Why Not a Polynomial

The tempting fix for curvature is to use all the data with a flexible curve on each side.

```python
for deg in (1, 2, 4, 6):
    est = np.array([global_poly(x, d, y, deg) for x, d, y in (draw() for _ in range(reps))])
    print(f"degree {deg}: mean {est.mean():.2f}, sd {est.std():.2f}, rmse {np.sqrt(np.mean((est-tau)**2)):.2f}")
```

| Global polynomial, each side | Mean estimate | Standard deviation | Root mean squared error |
| --- | --- | --- | --- |
| Degree 1 | 0.75 | 0.23 | 1.27 |
| Degree 2 | 1.98 | 0.34 | 0.34 |
| Degree 4 | 2.02 | 0.57 | 0.57 |
| Degree 6 | 2.04 | 0.78 | 0.78 |

The degree-2 polynomial does well here because the simulated trend is exactly quadratic, which the analyst never knows. Beyond that, each extra degree adds noise without removing bias, and a degree-6 fit is worse than a local linear fit with a bandwidth of 20 while using five times the data. High-order global polynomials put heavy weight on observations far from the cutoff and produce estimates that swing with the degree chosen, which is Gelman and Imbens's argument for abandoning them. A local fit of low degree near the cutoff is the standard for a reason.

## Placebo Cutoffs

A discontinuity estimator applied where there is no rule should find nothing. Moving the cutoff to places where no treatment changes is the falsification test that costs nothing and catches the most.

```python
for c in (-25.0, -10.0, 10.0, 25.0):
    est = np.array([local_linear(x, y, 10, cutoff=c) for x, d, y in (draw() for _ in range(reps))])
    print(f"placebo cutoff {c:+.0f}: mean {est.mean():+.2f}, sd {est.std():.2f}")
```

| Placebo cutoff | Mean estimate | Standard deviation |
| --- | --- | --- |
| -25 | -0.02 | 0.49 |
| -10 | +0.02 | 0.51 |
| +10 | +0.00 | 0.50 |
| +25 | +0.00 | 0.51 |

All four are zero to within noise. A placebo estimate that is not zero means the outcome jumps for reasons other than the rule, and the estimate at the real cutoff inherits the same problem.

## When Units Choose Their Side

The design fails if units can control which side of the cutoff they land on, because the ones who make the effort to cross are not the ones who do not. Suppose a share of customers scored just below 600 can nudge their score above it, and those who bother are the motivated ones, who would have spent more anyway.

```python
for share in (0.0, 0.3, 0.6):
    est, dens = [], []
    for _ in range(reps // 4):
        x, d, y = draw(manipulate=share)
        est.append(local_linear(x, y, 10))
        dens.append(np.sum((x >= 0) & (x < 5)) / np.sum((x < 0) & (x > -5)))
    print(f"share {share:.0%}: estimate {np.mean(est):.2f}, density ratio above/below {np.mean(dens):.2f}")
```

| Share of near-miss units that push above the cutoff | Estimate (true 2.00) | Units just above / just below |
| --- | --- | --- |
| 0% | 1.94 | 1.01 |
| 30% | 2.93 | 1.85 |
| 60% | 3.54 | 4.05 |

With a third of the near-miss units crossing, the estimate is off by a full point; with six in ten, by more than three quarters of the truth. The column on the right is the diagnostic. In an unmanipulated population the density of the running variable is smooth through the cutoff, and there are as many units just above as just below. Manipulation piles units up on the favoured side, and the ratio of 1.85 or 4.05 is visible in a histogram before any outcome is examined. McCrary's density test formalises the comparison; in practice, a histogram of the running variable with the cutoff marked is the first plot to make, and a spike at the threshold is the reason to stop.

## How Much Data the Design Needs

Only the units near the cutoff carry information, so the design is hungrier for data than its total sample size suggests.

```python
for nn in (1000, 5000, 20000):
    est = np.array([local_linear(x, y, 10) for x, d, y in (draw(n=nn) for _ in range(500))])
    print(f"n = {nn:>6}: sd {est.std():.2f} with about {2*10/100*nn:.0f} units inside the window")
```

| Total units | Units within the bandwidth | Standard deviation of the estimate |
| --- | --- | --- |
| 1,000 | 200 | 1.11 |
| 5,000 | 1,000 | 0.55 |
| 20,000 | 4,000 | 0.26 |

The standard error falls with the square root of the units inside the window, and the window is a fifth of the data at this bandwidth. A design with a thousand customers has two hundred near the cutoff and a standard error half the size of the effect; it can see that the credit line does something, and not much more. Power calculations for a discontinuity design are done on the units within the bandwidth, and the bandwidth is set by the curvature, so the effective sample is a property of the outcome's shape as much as of the data collected.

## Where the Design Applies

Any rule that assigns something by a threshold on a continuous score is a candidate: credit lines by score, discounts by order value, scholarships by grade, inspections by risk rank, feature flags by account age, priority support by revenue, retraining by drift statistic. The requirements are that the score is measured before the assignment and cannot be precisely manipulated, that the treatment actually changes at the threshold, and that nothing else does. A fuzzy version, in which crossing the threshold raises the probability of treatment rather than guaranteeing it, is analysed like a non-compliance problem, with the jump in outcome divided by the jump in take-up.

The design cannot say what the treatment does far from the cutoff. The bank learns what a credit line does to customers scored around 600, which is exactly the population it would reach by lowering the threshold to 580, and which is the decision the estimate is for.

## What to Do

1. **Plot the outcome against the running variable** in bins, with the cutoff marked. The jump should be visible before it is estimated.
2. **Plot the density of the running variable** through the cutoff, and run the McCrary test. A pile-up on the favoured side means the estimate is not credible.
3. **Fit local linear regressions on each side** within a data-chosen bandwidth, and report the estimate at several bandwidths around it; a result that depends on the bandwidth is not a result.
4. **Do not use high-order global polynomials.** They add noise, weight far-away units, and swing with the degree.
5. **Run placebo cutoffs** away from the threshold and check that covariates fixed before assignment do not jump at it.
6. **State that the estimate is local to the cutoff**, and match it to a decision about moving the threshold rather than to a decision about the treatment in general.

## References

- Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis: an alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309-317.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: a guide to practice. *Journal of Econometrics*, 142(2), 615-635.
- Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression discontinuity estimator. *The Review of Economic Studies*, 79(3), 933-959.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: a density test. *Journal of Econometrics*, 142(2), 698-714.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in regression discontinuity designs. *Journal of Business and Economic Statistics*, 37(3), 447-456.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
