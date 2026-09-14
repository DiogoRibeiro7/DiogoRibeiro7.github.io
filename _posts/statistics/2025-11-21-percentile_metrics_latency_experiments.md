---
permalink: '/statistics/percentile_metrics_latency_experiments/'
title: 'Percentile Metrics: Why p95 Latency Is Harder to Move and Harder to Measure'
categories:
- Statistics
tags:
- Model Evaluation
- Data Science
- Hypothesis Testing
- Statistics
author_profile: false
seo_title: 'Percentile Metrics and Tail Latency in Experiments'
seo_description: 'The mean, the median and the 99th percentile of a latency distribution answer different questions and move in different directions. A simulation shows a change that improves the median while making the mean worse, and why detecting a tail improvement needs a thousand times the data.'
excerpt: >-
  The change makes 97 percent of requests five percent faster and the
  remaining three percent slightly more likely to hit the slow path. The
  median improves by 4.3 percent, the mean gets 5.9 percent worse, and
  p99 rises by 19. All three numbers are correct.
summary: >-
  Why a heavy-tailed latency distribution makes the mean, the median and
  the tail percentiles answer different questions, a simulation of three
  changes that move them in different directions and sometimes opposite
  signs, the precision of each statistic against sample size and the
  closed form that explains it, how many requests are needed to detect a
  five percent improvement in each, and which confidence interval for a
  percentile actually covers.
keywords:
  - percentile metrics
  - tail latency
  - p95
  - p99
  - quantile standard error
  - order statistics
  - performance measurement
classes: wide
date: '2025-11-21'
why_this_exists: >-
  Service level objectives are written on percentiles and experiment
  dashboards report means, so the two disagree regularly and nobody can
  say which is wrong. This post shows what each statistic is sensitive
  to, how much data each needs, and which interval to trust, so that a
  latency result can be read without ambiguity.
evidence: >-
  A simulated latency distribution with a lognormal body around 68
  milliseconds and a 3 percent slow path around 900 milliseconds, two
  million requests per scenario, under three changes: a faster body, a
  faster tail, and a faster body with a more frequent slow path;
  precision measured over 200 repeats at 1,000 to 100,000 requests and
  interval coverage over 1,000 replications.
methodology: >-
  Compares the mean, median, 95th and 99th percentiles before and after
  each change; measures the relative standard error of each statistic
  against sample size and checks it against the closed form for the
  standard error of a quantile; derives the requests needed to detect a
  five percent improvement in each statistic; and compares the coverage
  of normal-approximation, order-statistic and bootstrap intervals for
  the 95th percentile.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-supercomputer.jpg
  og_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-supercomputer.jpg
  twitter_image: /assets/images/headers/photo-supercomputer.jpg
---
The service level objective is written on the 95th percentile, the experiment dashboard reports the mean, and the two have just disagreed. The change makes almost every request a little faster and occasionally pushes one onto a slow path. The median falls by 4.3 percent, the mean rises by 5.9, and the 99th percentile rises by 19. The team spends a morning deciding which number is wrong.

None of them is wrong. A latency distribution with a heavy tail is not summarised by any single number, and the mean, the median and the tail percentiles are sensitive to different parts of it. The mean is dominated by the tail, because a handful of one-second requests outweigh thousands of 60-millisecond ones. The median knows nothing about the tail at all. Choosing between them is choosing which users the metric is about.

## A Distribution With Two Parts

The simulation uses the shape that real request latency usually has: a lognormal body, and a small share of requests that take a slow path an order of magnitude worse.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def latency(n, speedup_body=0.0, speedup_tail=0.0, extra_slow=0.0, r=rng):
    """Body is lognormal; a small share of requests hit a slow path an order of magnitude worse."""
    body = r.lognormal(np.log(80) - 0.18, 0.6, n) * (1 - speedup_body)
    slow = r.random(n) < (0.03 + extra_slow)
    tail = r.lognormal(np.log(900), 0.7, n) * (1 - speedup_tail)
    return np.where(slow, tail, body)

base = latency(2_000_000)
for q in (0.5, 0.9, 0.95, 0.99):
    print(f"p{q * 100:.0f}: {np.quantile(base, q):.1f} ms")
print(f"mean: {base.mean():.1f} ms")
```

| Statistic | Value |
| --- | --- |
| Median | 68.4 ms |
| 90th percentile | 160.2 ms |
| 95th percentile | 225.5 ms |
| 99th percentile | 1,203.8 ms |
| Mean | 111.8 ms |

The mean sits at 112 milliseconds, well above the median of 68, because three percent of requests are taking around a second. Anyone who reasons about "typical" latency from the mean is reasoning about a value that most requests never come near.

## Three Changes, Three Different Verdicts

The same distribution, changed three ways: the body made five percent faster, the slow path made twenty percent faster, and the body made five percent faster at the cost of the slow path becoming one percentage point more common.

```python
scenarios = {
    "5% faster body": dict(speedup_body=0.05),
    "20% faster tail": dict(speedup_tail=0.20),
    "body 5% faster, slow path 1pp more common": dict(speedup_body=0.05, extra_slow=0.01),
}
for label, kw in scenarios.items():
    v = latency(2_000_000, **kw)
    print(f"{label}: mean {v.mean() / base.mean() - 1:+.1%}, "
          f"p50 {np.quantile(v, 0.5) / np.quantile(base, 0.5) - 1:+.1%}, "
          f"p95 {np.quantile(v, 0.95) / np.quantile(base, 0.95) - 1:+.1%}, "
          f"p99 {np.quantile(v, 0.99) / np.quantile(base, 0.99) - 1:+.1%}")
```

| Change | Mean | Median | p95 | p99 |
| --- | --- | --- | --- | --- |
| Body 5% faster | -3.4% | -5.1% | -5.0% | +0.8% |
| Tail 20% faster | -5.9% | -0.0% | -0.8% | -19.0% |
| Body 5% faster, slow path 1pp more common | +5.9% | -4.3% | +8.8% | +18.9% |

The first row is the well-behaved case: a change to the body moves the median and p95 by the full five percent and the mean by less, because the mean is anchored by a tail that did not move. The second row is its mirror: a twenty percent improvement to the tail is invisible in the median, barely visible at p95, and shows up fully at p99 and in the mean.

The third row is the one that starts arguments. Almost every request got faster, and the median says so. But one request in a hundred that used to be fast is now slow, and the mean, p95 and p99 all get worse. Whether this change is an improvement depends entirely on whether the cost is borne by users who matter, and no single statistic answers that. The reason to report the median next to a tail percentile is precisely to make this pattern visible instead of arguing about which metric is right.

## The Tail Is Also the Hardest Part to Measure

Sensitivity is only half the story. The other half is precision, and it runs in the same direction: the statistics that respond to the tail are the ones the tail makes noisy.

```python
for n in (1_000, 10_000, 100_000):
    m, q50, q95, q99 = [], [], [], []
    for _ in range(200):
        v = latency(n)
        m.append(v.mean()); q50.append(np.quantile(v, 0.5))
        q95.append(np.quantile(v, 0.95)); q99.append(np.quantile(v, 0.99))
    print(f"n = {n:,}: " + "  ".join(f"{np.std(a) / np.mean(a):.2%}" for a in (m, q50, q95, q99)))
```

**Relative standard error of each statistic**, over 200 repeats.

| Requests | Mean | Median | p95 | p99 |
| --- | --- | --- | --- | --- |
| 1,000 | 6.80% | 2.54% | 7.89% | 19.00% |
| 10,000 | 2.21% | 0.82% | 2.52% | 6.79% |
| 100,000 | 0.67% | 0.24% | 0.77% | 2.03% |

At ten thousand requests the median is known to within one percent and the 99th percentile to within seven. A five percent tail improvement measured on ten thousand requests is inside the noise.

The closed form explains it. The standard error of the $p$-th quantile is

$$
\operatorname{se}(\hat x_p) = \frac{1}{f(x_p)}\sqrt{\frac{p(1-p)}{n}},
$$

where $f$ is the density at the quantile. The numerator barely changes between the median and the tail, but the density does: observations pile up around the median and are spread thinly around p99, so the same count of observations pins down the median tightly and the tail loosely.

```python
for q in (0.5, 0.95, 0.99):
    xq = np.quantile(base, q)
    dens = np.mean((base > xq * 0.98) & (base < xq * 1.02)) / (0.04 * xq)
    se = np.sqrt(q * (1 - q) / 10_000) / dens
    print(f"p{q * 100:.0f}: density {dens:.5f}, predicted se at n = 10,000: {se:.2f} ms ({se / xq:.2%})")
```

| Quantile | Density at the quantile | Predicted relative error at n = 10,000 | Measured |
| --- | --- | --- | --- |
| Median | 0.00943 | 0.77% | 0.82% |
| p95 | 0.00039 | 2.51% | 2.52% |
| p99 | 0.00001 | 6.14% | 6.79% |

The formula matches the simulation to within a fraction of a percentage point, which is the check that the intuition about density is the right one.

![Relative standard error of the mean, median, 95th and 99th percentile against the number of requests, on log scales. The median is an order of magnitude more precise than the 99th percentile at every sample size.](/assets/images/figures/percentile_precision.png){: width="1152" height="672" loading="lazy"}

## What It Costs to Detect a Change

Combining sensitivity with precision gives the number that matters for planning an experiment: how many requests are needed to see a given improvement.

```python
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
treated = latency(2_000_000, speedup_body=0.05)
for name, q in (("mean", None), ("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
    vals = [np.mean(v) if q is None else np.quantile(v, q) for v in (latency(20_000) for _ in range(200))]
    cv = np.std(vals) / np.mean(vals) * np.sqrt(20_000)      # relative sd scaled to one observation
    stat = (lambda a: a.mean()) if q is None else (lambda a: np.quantile(a, q))
    eff = abs(stat(treated) / stat(base) - 1)
    print(f"{name}: relative effect {eff:.2%}, requests per arm {2 * (cv / eff) ** 2 * z ** 2:,.0f}")
```

| Statistic | Relative effect of a 5% faster body | Requests per arm for 80% power |
| --- | --- | --- |
| Median | 5.11% | 4,090 |
| Mean | 2.98% | 84,279 |
| p95 | 4.90% | 45,110 |
| p99 | 1.16% | 4,662,228 |

The median needs four thousand requests; the mean needs twenty times that, because it responds less to a body change and varies more; p95 needs eleven times the median's; and p99 needs nearly five million, because the change barely moves it and it is noisy besides. A team that writes its objective on p99 and then runs experiments for a day has an instrument that cannot see the changes it is making.

This is an argument for a division of labour rather than for abandoning tail metrics. The median is the sensitive instrument for detecting whether a change helped typical requests. The tail percentiles are the guardrail, watched over long windows and many requests, where the question is not "did this experiment move p99" but "is p99 drifting". Sizing an experiment on a tail percentile usually means either a very long test or accepting that the metric is there to catch disasters, not improvements.

## Which Interval Actually Covers

Percentile confidence intervals are easy to get wrong, because the normal approximation needs that density, and the density has to be estimated from the same sparse tail.

```python
def normal_ci(v, q=0.95):
    """Normal approximation with a density estimated from the sample."""
    xq = np.quantile(v, q)
    h = 1.06 * v.std() * len(v) ** -0.2
    dens = np.mean(np.abs(v - xq) < h) / (2 * h)
    se = np.sqrt(q * (1 - q) / len(v)) / max(dens, 1e-12)
    return xq - 1.96 * se, xq + 1.96 * se

def order_ci(v, q=0.95):
    """Distribution-free interval from the binomial order statistics."""
    n = len(v)
    lo = stats.binom.ppf(0.025, n, q); hi = stats.binom.ppf(0.975, n, q)
    s = np.sort(v)
    return s[int(max(lo - 1, 0))], s[int(min(hi, n - 1))]

def boot_ci(v, q=0.95, b=400, r=rng):
    idx = r.integers(0, len(v), (b, len(v)))
    qs = np.quantile(v[idx], q, axis=1)
    return np.quantile(qs, 0.025), np.quantile(qs, 0.975)

truth = np.quantile(base, 0.95)
cov = np.zeros(3)
for _ in range(1000):
    v = latency(2_000)
    for j, fn in enumerate((normal_ci, order_ci, boot_ci)):
        lo, hi = fn(v)
        cov[j] += lo <= truth <= hi
print(f"normal {cov[0] / 1000:.1%}, order statistics {cov[1] / 1000:.1%}, bootstrap {cov[2] / 1000:.1%}")
```

| Interval for p95 at n = 2,000 | Coverage of a nominal 95 percent |
| --- | --- |
| Normal approximation with estimated density | 89.8% |
| Order statistics (distribution-free) | 96.5% |
| Bootstrap | 94.7% |

The normal approximation is five points short, because the density estimate at the 95th percentile is itself unreliable and the error propagates. The order-statistic interval, which asks only how many observations fall below the quantile and inverts the binomial, needs no density and covers correctly. It is also the cheapest of the three: two sorted lookups. The bootstrap is close behind and is the one to reach for when the statistic is more complicated than a single quantile, such as a difference of percentiles between arms.

## Choosing the Metric Before the Argument

Three questions settle which statistic a given decision needs.

**Whose experience is being protected?** If the answer is "the worst-served users", the metric is a tail percentile and the analysis must be sized for it. If the answer is "typical usage", the median is both more relevant and more sensitive.

**Does total cost matter?** Capacity, compute spend and queueing all depend on the sum, so the mean is the right metric for them regardless of how unrepresentative it is of any single request.

**Is the change expected to move the body, the tail, or the share that lands in the tail?** A cache hit-rate improvement moves the share; a timeout change moves the tail; an algorithmic speedup moves the body. Predicting this before the test tells you which statistic will show it and avoids reading the absence of movement in the wrong one as a null result.

Reporting the median and a tail percentile together, always, costs nothing and makes the third row of the earlier table impossible to miss. A dashboard with only one of them will eventually ship a change that helps most users by hurting a few, or block one that helps the few because it did not move the middle.

## What to Do

1. **Report the median and a tail percentile side by side**, plus the mean when cost or capacity is at stake. A single latency number hides changes that move parts of the distribution in opposite directions.
2. **Size experiments on the statistic you will decide with.** A five percent body improvement needs four thousand requests at the median and three million at p99.
3. **Use order-statistic intervals for percentiles**, or the bootstrap for differences of percentiles; the normal approximation under-covers because the tail density cannot be estimated well.
4. **Treat tail percentiles as guardrails over long windows** rather than as experiment endpoints, unless the traffic makes them measurable.
5. **Predict which part of the distribution the change touches**, and check the statistic that should respond before concluding that nothing happened.
6. **Watch the share of requests in the slow path** as its own metric; a change to it moves the mean and the tail together while leaving the median untouched.

## References

- Dean, J., & Barroso, L. A. (2013). The tail at scale. *Communications of the ACM*, 56(2), 74-80.
- Hahn, G. J., & Meeker, W. Q. (1991). *Statistical Intervals: A Guide for Practitioners*. Wiley.
- Serfling, R. J. (1980). *Approximation Theorems of Mathematical Statistics*. Wiley.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
