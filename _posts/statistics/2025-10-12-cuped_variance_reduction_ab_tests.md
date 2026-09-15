---
permalink: '/statistics/cuped_variance_reduction_ab_tests/'
title: 'CUPED and Regression Adjustment: Cutting A/B Test Variance With Data You Already Have'
categories:
- Statistics
tags:
- Experimental Design
- A/B Testing
- Variance Reduction
- Statistics
author_profile: false
seo_title: 'CUPED and Regression Adjustment in A/B Tests'
seo_description: 'A pre-experiment covariate correlated with the outcome removes a share of the variance equal to the squared correlation. A simulation shows CUPED, regression adjustment and stratification halving the standard error at a correlation of 0.7, keeping the false positive rate, and going wrong only when the covariate is measured after exposure.'
excerpt: >-
  The experiment needs 1,570 users per arm to detect a two percent lift.
  Each user's spend over the previous month is already in the warehouse
  and correlates 0.7 with the outcome. Using it cuts the requirement to
  801, from the same data and the same weeks of traffic.
summary: >-
  Why the noise in an A/B metric is mostly differences between users that
  existed before the experiment, how CUPED subtracts the part of the
  outcome a pre-experiment covariate predicts and why the variance drops
  by the squared correlation, a simulation comparing CUPED, regression
  adjustment and stratification across correlations, the closed-form
  sample-size saving, the check that the false positive rate is
  unchanged, the one way to break it (a covariate measured after
  exposure), and how to choose and report the covariate.
keywords:
  - CUPED
  - variance reduction
  - regression adjustment
  - A/B testing
  - pre-experiment covariate
  - stratification
  - statistical power
classes: wide
date: '2025-10-12'
why_this_exists: >-
  Most experiments run longer than they need to because the analysis
  ignores what was known about each user before the test started. This
  post measures what the standard adjustments recover under realistic
  correlations, so a team can decide whether to adopt them and how much
  traffic they save.
evidence: >-
  Simulated experiments with 2,000 users per arm, a pre-experiment
  covariate correlated 0 to 0.85 with the outcome, and true effects of
  zero and of two percent of the mean, 4,000 replications per cell;
  four estimators compared.
methodology: >-
  Measures the standard deviation of each estimator across replications
  against the 1 minus rho squared prediction, the false positive rate on
  A/A data, power for a fixed effect, users required from the closed-form
  sample-size formula, and the bias each adjusted estimator acquires when
  the covariate is itself moved by the treatment.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-lights.jpg
  og_image: /assets/images/headers/photo-lights.jpg
  overlay_image: /assets/images/headers/photo-lights.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-lights.jpg
  twitter_image: /assets/images/headers/photo-lights.jpg
---
The power calculation says 1,570 users per arm to detect a two percent lift in monthly spend, which at the current traffic is three weeks. The team ships the test and waits. In the warehouse, for every one of those users, sits the same metric measured over the month before the experiment began, and it correlates 0.7 with what they will spend during it. Nobody uses it.

Using it would have cut the requirement to 801 users per arm. Not by adding data, but by removing from each user's outcome the part that was predictable before the experiment started. A user who spent heavily last month will probably spend heavily this month whichever arm she lands in, and that predictable part is noise as far as the treatment effect is concerned. Subtracting it leaves the part the experiment can actually move, and that part has half the variance.

## What the Noise Is Made Of

The variance of an A/B estimate comes from the spread of outcomes across users within each arm. Most of that spread is not random in any interesting sense; it is persistent differences between users. Heavy users and light users, frequent visitors and occasional ones, and those differences were there before randomisation and are unaffected by it. Randomisation guarantees the arms are balanced on them in expectation, but any particular experiment is a little unbalanced, and the imbalance shows up as estimation error.

If a pre-experiment covariate $X$ predicts the outcome $Y$ with correlation $\rho$, then the residual $Y - \theta X$, with $\theta$ chosen by least squares, has variance $(1 - \rho^2)$ times the variance of $Y$. Comparing residuals between arms instead of raw outcomes estimates the same treatment effect, because the covariate is balanced across arms by randomisation and its subtraction cancels in expectation, with a standard error smaller by the factor $\sqrt{1 - \rho^2}$. That is CUPED, controlled experiments using pre-experiment data, and it is a rediscovery in the experimentation setting of what Fisher called the analysis of covariance.

## A Simulation

Two arms of 2,000 users. Each user has a covariate $X$ and an outcome $Y$ with a chosen correlation, both with standard deviation 10 around a mean of 50. Four estimators of the treatment effect are compared: the plain difference in means, CUPED with a pooled $\theta$, a regression of the outcome on treatment and the centred covariate, and a stratified estimate that compares arms within covariate quartiles and averages.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def draw(n, rho, effect=0.0, r=rng, sd=10.0, post_treatment=False):
    """Two arms of n users; X is a pre-experiment covariate correlated rho with the outcome Y."""
    t = np.repeat([0, 1], n)
    x = r.normal(50, sd, 2 * n)
    y = 50 + rho * (x - 50) + effect * t + r.normal(0, sd * np.sqrt(1 - rho ** 2), 2 * n)
    if post_treatment:                      # the "covariate" is measured after exposure and moved by it
        x = x + 0.5 * effect * t
    return t, x, y

def diff_means(t, x, y):
    a, b = y[t == 0], y[t == 1]
    return b.mean() - a.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

def cuped(t, x, y):
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)      # pooled across arms
    return diff_means(t, x, y - theta * (x - x.mean()))

def regression(t, x, y):
    X = np.column_stack([np.ones(len(t)), t, x - x.mean()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = (resid @ resid / (len(y) - 3)) * np.linalg.inv(X.T @ X)
    return beta[1], np.sqrt(cov[1, 1])

def stratified(t, x, y, k=4):
    s = np.digitize(x, np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
    est = var = 0.0
    for j in range(k):
        m = s == j
        a, b = y[m & (t == 0)], y[m & (t == 1)]
        w = m.mean()
        est += w * (b.mean() - a.mean())
        var += w ** 2 * (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return est, np.sqrt(var)

methods = {"difference in means": diff_means, "CUPED": cuped,
           "regression adjustment": regression, "stratified (quartiles)": stratified}
reps = 4000
print("standard error of the estimated effect on A/A data, 2,000 users per arm")
for rho in (0.0, 0.3, 0.5, 0.7, 0.85):
    ests = {m: [] for m in methods}
    for _ in range(reps):
        t, x, y = draw(2000, rho)
        for m, fn in methods.items():
            ests[m].append(fn(t, x, y)[0])
    print(f"rho {rho:.2f}: " + ", ".join(f"{m} {np.std(ests[m]):.3f}" for m in methods) + f"   1 - rho^2 = {1 - rho**2:.2f}")
```

**Standard error of the estimated effect** on A/A data, from the spread of estimates across 4,000 experiments.

| Correlation | Difference in means | CUPED | Regression adjustment | Stratified (quartiles) | $1 - \rho^2$ |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.320 | 0.321 | 0.321 | 0.321 | 1.00 |
| 0.30 | 0.315 | 0.301 | 0.301 | 0.304 | 0.91 |
| 0.50 | 0.312 | 0.274 | 0.274 | 0.279 | 0.75 |
| 0.70 | 0.305 | 0.221 | 0.221 | 0.234 | 0.51 |
| 0.85 | 0.316 | 0.165 | 0.165 | 0.192 | 0.28 |

At a correlation of 0.7, the CUPED and regression estimates have a variance ratio of 0.52 to the difference in means, against a predicted 0.51. At 0.85 the standard error is about half, which is the precision of four times the users. Stratification into quartiles captures most of the gain but not all of it, because a four-level categorisation of a continuous covariate loses some of the linear information; finer strata close the gap. CUPED and regression adjustment coincide here because both fit the same linear relationship; they differ only when the adjustment is estimated per arm, which is Lin's refinement and matters when the treatment changes the slope.

A covariate uncorrelated with the outcome gains nothing and costs almost nothing: the first row shows the adjusted estimators within a thousandth of the unadjusted one, the price of estimating a $\theta$ that is really zero.

![Standard error of the estimated effect against the correlation between the covariate and the outcome, for the difference in means, stratification by quartile and CUPED, with the square root of 1 minus the squared correlation as the theoretical line. CUPED follows the theory; at a correlation of 0.85 its standard error is about half the unadjusted one.](/assets/images/figures/cuped_variance_reduction.png){: width="1152" height="672" loading="lazy"}

## The False Positive Rate Does Not Move

Any variance reduction claim has to survive the A/A check: with no true effect, each method should reject at 5 percent using its own reported standard error, or the smaller confidence intervals are an illusion.

```python
hits = {m: 0 for m in methods}
for _ in range(reps):
    t, x, y = draw(2000, 0.7)
    for m, fn in methods.items():
        est, se = fn(t, x, y)
        hits[m] += abs(est / se) > 1.96
print({m: f"{hits[m]/reps:.1%}" for m in methods})
```

| Estimator | False positives at 5% |
| --- | --- |
| Difference in means | 5.0% |
| CUPED | 4.6% |
| Regression adjustment | 4.6% |
| Stratified (quartiles) | 4.5% |

All four hold their level. The adjusted standard errors are honest: smaller because the residuals are less variable, not because the formula is optimistic.

## What the Saving Buys

The same reduction can be spent on power at a fixed sample size or on a smaller sample at fixed power. With a true effect of one unit, two percent of the mean, and 2,000 users per arm:

```python
for rho in (0.0, 0.5, 0.7, 0.85):
    power = {m: 0 for m in methods}
    for _ in range(reps):
        t, x, y = draw(2000, rho, effect=1.0)
        for m, fn in methods.items():
            est, se = fn(t, x, y)
            power[m] += abs(est / se) > 1.96
    print(f"rho {rho:.2f}: " + ", ".join(f"{m} {power[m]/reps:.0%}" for m in methods))
```

| Correlation | Difference in means | CUPED | Regression adjustment | Stratified |
| --- | --- | --- | --- | --- |
| 0.00 | 89% | 89% | 89% | 89% |
| 0.50 | 88% | 96% | 96% | 95% |
| 0.70 | 88% | 99% | 99% | 98% |
| 0.85 | 88% | 100% | 100% | 100% |

Or, holding power at 80 percent, the users required follow from the usual formula with the variance scaled by $1 - \rho^2$:

$$
n = \frac{2\sigma^2 (1 - \rho^2)(z_{0.975} + z_{0.8})^2}{\delta^2}.
$$

```python
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
for rho in (0.0, 0.3, 0.5, 0.7, 0.85):
    n = 2 * 10.0 ** 2 * (1 - rho ** 2) * z ** 2 / 1.0 ** 2
    print(f"rho {rho:.2f}: {n:,.0f} users per arm ({rho**2:.0%} fewer)")
```

| Correlation | Users per arm for 80% power | Saving |
| --- | --- | --- |
| 0.00 | 1,570 | 0% |
| 0.30 | 1,428 | 9% |
| 0.50 | 1,177 | 25% |
| 0.70 | 801 | 49% |
| 0.85 | 436 | 72% |

The saving is exactly $\rho^2$. A correlation of 0.3, which is what a weak covariate such as a coarse activity level gives, saves a tenth. A correlation of 0.7, typical for the same metric measured over the preceding period, halves the experiment. That is where the method earns its keep: for metrics that are stable within users, spend, sessions, engagement, the pre-period value of the same metric is the covariate, and it is usually already computed.

## The One Way to Break It

The covariate must be unaffected by the treatment, which in practice means measured before assignment. A covariate measured during the experiment, even one that looks like a pre-treatment characteristic, may itself carry part of the effect, and adjusting for it removes that part.

```python
ests = {m: [] for m in methods}
for _ in range(reps):
    t, x, y = draw(2000, 0.7, effect=1.0, post_treatment=True)
    for m, fn in methods.items():
        ests[m].append(fn(t, x, y)[0])
print({m: f"{np.mean(ests[m]):.2f}" for m in methods})
```

| Estimator | Mean estimate (true effect 1.00) |
| --- | --- |
| Difference in means | 1.00 |
| CUPED | 0.65 |
| Regression adjustment | 0.65 |
| Stratified (quartiles) | 0.70 |

When the covariate absorbs half of the effect, the adjusted estimators report two thirds of the truth, and they do so with the tight standard errors that made them attractive. The difference in means is unbiased and does not know anything about the covariate. The rule that prevents this is simple to state and easy to enforce in a data pipeline: the covariate is computed from data with timestamps before the user's assignment time, and nothing else qualifies.

## Choosing the Covariate

The best covariate is the outcome metric itself, measured over a pre-period of similar length. Its correlation with the experiment-period value is the metric's own week-to-week stability, which for spend and engagement metrics is often 0.5 to 0.8 and for conversion of new visitors is close to zero. Several covariates can be combined by regression, or by fitting a model to predict the outcome from pre-period data and using the prediction as the single covariate; the gain is then the squared correlation of the prediction.

Users with no pre-period, new signups, have no covariate. The usual handling is to give them the pre-period mean and an indicator, which contributes no variance reduction for them and no bias. An experiment on new users alone gains nothing from CUPED, and the sample-size calculation should say so.

Estimate $\theta$ on the pooled data from both arms, not per arm and not on historical data; the pooled estimate is consistent under the null and the difference is negligible in practice. Report the adjusted estimate with its adjusted standard error, and state the covariate and the correlation achieved, so that a reader can see what was subtracted and check that it was pre-treatment.

## What to Do

1. **Compute the outcome metric over a pre-period** of the same length for every user, from data timestamped before assignment.
2. **Fit $\theta$ on the pooled data** and analyse $Y - \theta(X - \bar X)$ with an ordinary two-sample test, or equivalently regress $Y$ on treatment and the centred covariate.
3. **Size the experiment with $1 - \rho^2$** in the variance, using the pre-period correlation measured on past data.
4. **Run the A/A check** once on your own pipeline; a false positive rate above 5 percent means the implementation, not the method, is wrong.
5. **Never adjust for anything measured after assignment.** A post-treatment covariate turns the variance reduction into bias.
6. **Report the covariate and its correlation** with every result, so the precision is explainable.

## References

- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. *Proceedings of the Sixth ACM International Conference on Web Search and Data Mining*, 123-132.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1), 295-318.
- Freedman, D. A. (2008). On regression adjustments to experimental data. *Advances in Applied Mathematics*, 40(2), 180-193.
- Fisher, R. A. (1932). *Statistical Methods for Research Workers* (4th ed.). Oliver and Boyd.
- Xie, H., & Aurisset, J. (2016). Improving the sensitivity of online controlled experiments: case studies at Netflix. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 645-654.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
