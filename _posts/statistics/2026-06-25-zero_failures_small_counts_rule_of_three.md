---
permalink: '/statistics/zero_failures_small_counts_rule_of_three/'
title: 'Zero Failures in 300 Trials Proves Less Than You Think: Small Counts and Honest Intervals'
categories:
- Statistics
tags:
- Reliability
- Confidence Intervals
- Hypothesis Testing
- Statistics
author_profile: false
seo_title: 'Zero Events, Small Counts and the Rule of Three'
seo_description: 'A clean run of n trials with no failures bounds the failure rate at about 3/n, not at zero. This post gives the exact bound, the trials needed to prove a rate is below a target, why the textbook interval fails at small counts, which intervals to use instead, and the same arithmetic for events in time.'
excerpt: >-
  The release passed 300 test runs without a failure. The failure rate
  compatible with that result, at 95 percent confidence, is anything up
  to 1.2 percent. A rate of one in a thousand would have produced the
  same clean run three times out of four.
summary: >-
  What a run with zero events does and does not establish, the exact
  upper bound and the rule of three that approximates it, the number of
  failure-free trials that proves a rate is below a target and how a
  single failure changes it, the probability that a clean run hides a
  real rate, a simulation of the coverage of Wald, Wilson and exact
  intervals when few events are expected, the same bound for events in
  time, and how to size a qualification test before running it.
keywords:
  - rule of three
  - zero events
  - Clopper-Pearson
  - Wilson interval
  - small counts
  - reliability testing
  - confidence interval for a proportion
classes: wide
date: '2026-06-25'
why_this_exists: >-
  Qualification tests, canary deployments, safety audits and pilot
  studies all end with a count of bad outcomes that is usually zero or
  close to it, and the count is then read as if it were a rate. This post
  gives the arithmetic that turns a small count into what it actually
  proves, and the intervals that do not collapse when the count is tiny.
evidence: >-
  Exact binomial and Poisson calculations for zero and small counts, and
  a simulation of 10,000 samples per setting with true rates from 0.2 to
  5 percent and expected counts from 0.2 to 20 events, comparing the
  coverage of the Wald, Wilson and Clopper-Pearson intervals.
methodology: >-
  Tabulates the exact 95 percent upper bound for zero events against the
  rule of three and the Wilson and Wald bounds, derives the trials needed
  for a target rate with zero to five failures from the chi-square
  relation, computes the probability of a clean run under given true
  rates, and measures interval coverage and the share of samples with no
  events.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-calculator.jpg
  og_image: /assets/images/headers/photo-calculator.jpg
  overlay_image: /assets/images/headers/photo-calculator.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-calculator.jpg
  twitter_image: /assets/images/headers/photo-calculator.jpg
---
The release candidate ran through 300 end-to-end test executions and none failed. The report says the failure rate is zero, with a confidence interval, computed by the usual formula, of zero to zero. The candidate ships to a fleet that will execute the same path a hundred thousand times in its first week.

A failure rate of one in a thousand would have produced 300 clean runs three times out of four. A rate of one in two hundred would have produced them one time in five. What 300 clean runs actually establish, at 95 percent confidence, is that the rate is below 1.2 percent, which on a hundred thousand executions is up to 1,200 failures. The report's interval of zero to zero is not a conservative summary of a good result; it is the output of a formula applied outside the range in which it works, and it is the one situation in which that formula gives an answer that is not merely imprecise but empty.

## What Zero Proves

If the true failure probability is $p$ and $n$ trials are independent, the chance of seeing no failures is $(1-p)^n$. A rate $p$ is compatible with a clean run, at the 5 percent level, as long as that chance is at least 5 percent, and the largest such $p$ solves $(1-p)^n = 0.05$, which is $p = 1 - 0.05^{1/n}$. For any $n$ above a few dozen that is very close to $-\ln(0.05)/n \approx 3/n$, and that approximation is the rule of three: zero failures in $n$ trials bounds the rate at about $3/n$.

```python
import numpy as np
from scipy import stats

def wald(k, n, alpha=0.05):
    p = k / n; z = stats.norm.ppf(1 - alpha / 2)
    h = z * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)

def wilson(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2); p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)

def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

for n in (10, 30, 100, 300, 1000, 3000):
    print(f"n = {n:>5}: exact {clopper_pearson(0, n)[1]:.4f}, rule of three {3/n:.4f}, "
          f"Wilson {wilson(0, n)[1]:.4f}, Wald {wald(0, n)[1]:.4f}")
```

**The 95 percent upper bound on the rate after zero failures in $n$ trials.**

| Trials | Exact (Clopper-Pearson) | Rule of three, 3/n | Wilson | Wald |
| --- | --- | --- | --- | --- |
| 10 | 0.3085 | 0.3000 | 0.2775 | 0.0000 |
| 30 | 0.1157 | 0.1000 | 0.1135 | 0.0000 |
| 100 | 0.0362 | 0.0300 | 0.0370 | 0.0000 |
| 300 | 0.0122 | 0.0100 | 0.0126 | 0.0000 |
| 1,000 | 0.0037 | 0.0030 | 0.0038 | 0.0000 |
| 3,000 | 0.0012 | 0.0010 | 0.0013 | 0.0000 |

The exact bound and the rule of three agree to within a fifth at every size, and the Wilson interval, which comes from inverting a score test rather than assuming the estimate is normal, lands next to them. The Wald interval, the one in most spreadsheets and most reports, is zero wide at zero events for every $n$, because it estimates the variance from $\hat p(1 - \hat p)$ and $\hat p$ is zero. It has not concluded that the rate is zero; it has failed to produce an interval and reported the failure as certainty.

## How Many Clean Runs Prove a Rate

Turned around, the rule gives the test size that establishes a rate below a target: about $3/p$ failure-free trials, exactly $\ln(0.05)/\ln(1-p)$.

```python
for target in (0.05, 0.01, 0.001, 0.0001):
    print(f"rate below {target:.2%}: {np.ceil(np.log(0.05) / np.log(1 - target)):,.0f} clean trials")
for k in (0, 1, 2, 3, 5):
    print(f"{k} failures allowed, target 0.1%: {stats.chi2.ppf(0.95, 2 * k + 2) / (2 * 0.001):,.0f} trials")
```

| To show the rate is below | Failure-free trials needed |
| --- | --- |
| 5% | 59 |
| 1% | 299 |
| 0.1% | 2,995 |
| 0.01% | 29,956 |

Each factor of ten in the target costs a factor of ten in trials. A team that wants to claim one failure in a thousand needs three thousand clean runs, and a claim of one in ten thousand needs thirty thousand, which is why such claims are usually made from field data rather than from a test campaign. A single failure during the campaign does not end it; it raises the bill.

| Failures observed | Trials for the exact 95% bound to reach 0.1% |
| --- | --- |
| 0 | 2,996 |
| 1 | 4,744 |
| 2 | 6,296 |
| 3 | 7,754 |
| 5 | 10,513 |

These come from the chi-square relation between the Poisson count and its rate, $n = \chi^2_{0.95, 2k+2}/(2p)$, and they say that a plan allowing for one failure costs 58 percent more trials than a plan that requires none, and a plan allowing five costs three and a half times as much. The plan should be written before the campaign, with the number of failures it tolerates, because deciding after the fact that one failure "was a fluke" is the same as never having set a bound.

## The Clean Run That Hides a Real Rate

The reverse question is the one the report should have answered: if the rate were something we would care about, how likely was the clean run we saw?

```python
for p in (0.001, 0.005, 0.01, 0.02):
    print(f"true rate {p:.1%}: P(zero failures in 300 trials) = {(1 - p) ** 300:.0%}")
```

| True failure rate | Probability of 300 clean runs |
| --- | --- |
| 0.1% | 74% |
| 0.5% | 22% |
| 1.0% | 5% |
| 2.0% | 0% |

A rate of a tenth of a percent produces the clean run three times out of four, and half a percent produces it one time in five. The 300 runs rule out 2 percent and make 1 percent unlikely; they say almost nothing about the range below half a percent, which is where the difference between an acceptable release and a bad one usually lives. This table is the honest content of the report, and it fits in one line: "300 clean runs; rates up to 1.2 percent remain compatible with this result."

## Which Interval to Trust When Counts Are Small

The zero case is the extreme of a general problem: the Wald interval assumes the estimate is normally distributed, and with a handful of expected events it is not. The simulation draws 10,000 samples at each setting and checks how often each nominal 95 percent interval contains the true rate.

```python
rng = np.random.default_rng(0)
for n, p in ((100, 0.002), (500, 0.002), (500, 0.01), (2000, 0.002), (2000, 0.01), (200, 0.05)):
    cov = np.zeros(3); zeros = 0
    for _ in range(10000):
        k = rng.binomial(n, p)
        zeros += k == 0
        for j, fn in enumerate((wald, wilson, clopper_pearson)):
            lo, hi = fn(k, n)
            cov[j] += lo <= p <= hi
    print(f"n={n} p={p}: expected events {n*p:.1f}, Wald {cov[0]/10000:.1%}, Wilson {cov[1]/10000:.1%}, "
          f"exact {cov[2]/10000:.1%}, zero-event samples {zeros/10000:.0%}")
```

| Trials | True rate | Expected events | Wald | Wilson | Exact | Samples with zero events |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.2% | 0.2 | 18.2% | 98.2% | 98.2% | 82% |
| 500 | 0.2% | 1.0 | 64.8% | 91.9% | 98.0% | 35% |
| 500 | 1.0% | 5.0 | 86.9% | 96.4% | 98.0% | 1% |
| 2,000 | 0.2% | 4.0 | 90.4% | 92.9% | 96.0% | 2% |
| 2,000 | 1.0% | 20.0 | 94.9% | 94.4% | 95.7% | 0% |
| 200 | 5.0% | 10.0 | 92.7% | 96.7% | 96.7% | 0% |

With a fifth of an event expected, the Wald interval contains the truth 18 percent of the time, because 82 percent of samples have no events and the interval is then a point at zero. With one expected event it covers 65 percent; with five, 87 percent. It reaches its nominal level only around twenty expected events. The exact interval never falls below 95 percent, at the price of being wider than it needs to be. The Wilson interval sits between them and is the standard recommendation for routine use: near nominal coverage from a few events upward, and a sensible width at zero.

![Coverage of nominal 95 percent intervals against the expected number of events, at a true rate of half a percent, for Wald, Wilson and exact intervals. Wald collapses below two expected events; Wilson holds near nominal; the exact interval stays above it.](/assets/images/figures/small_count_interval_coverage.png){: width="1152" height="672" loading="lazy"}

The width of the honest interval at small counts is the other thing to internalise.

```python
for k in (0, 1, 2, 5, 10, 50):
    lo, hi = clopper_pearson(k, 1000)
    print(f"{k:>2} events in 1,000: rate {k/1000:.3f}, 95% interval [{lo:.4f}, {hi:.4f}]")
```

| Events in 1,000 trials | Point estimate | Exact 95% interval | Upper bound as a multiple of the estimate |
| --- | --- | --- | --- |
| 0 | 0.000 | 0.0000 to 0.0037 | not defined |
| 1 | 0.001 | 0.0000 to 0.0056 | 5.6 |
| 2 | 0.002 | 0.0002 to 0.0072 | 3.6 |
| 5 | 0.005 | 0.0016 to 0.0116 | 2.3 |
| 10 | 0.010 | 0.0048 to 0.0183 | 1.8 |
| 50 | 0.050 | 0.0373 to 0.0654 | 1.3 |

One event in a thousand trials is compatible with a rate five and a half times the point estimate. Two events, three and a half times. A comparison between two systems on the basis of one failure each, or one failure against none, is a comparison of two intervals that overlap almost entirely, and the arithmetic says so before any test is run.

## Events in Time

The same bound applies to a count of events in an observation period, with the Poisson distribution in place of the binomial. Zero events in $T$ hours bounds the rate at $-\ln(0.05)/T \approx 3/T$ per hour.

```python
for T in (100, 1000, 10000):
    print(f"{T:>6} hours with no event: rate below {3.0/T:.4f} per hour, {3.0/T*8760:.1f} per year")
```

| Observation with no event | 95% upper bound on the rate |
| --- | --- |
| 100 hours | 0.03 per hour, about 263 per year |
| 1,000 hours | 0.003 per hour, about 26 per year |
| 10,000 hours | 0.0003 per hour, about 2.6 per year |

A pump that ran a thousand hours without a failure has demonstrated a failure rate below roughly one a fortnight, not a failure rate of zero. A monitoring rule that fired no alerts in a hundred hours has demonstrated a false alarm rate below one every 33 hours. Hours accumulate across units, so a fleet of fifty pumps observed for two hundred hours each supplies the same bound as one pump for ten thousand, provided their failures are independent, which is the assumption to check before adding them up.

## What to Do

1. **Never report zero events as a rate of zero.** Report the count, the trials, and the exact or Wilson upper bound; the rule of three, $3/n$, is close enough for a conversation.
2. **Size qualification tests from the target**: about $3/p$ clean trials for a rate below $p$, and write down in advance how many failures the plan tolerates, using the chi-square table for the cost.
3. **Answer the reverse question** in every report: what rates would have produced this clean run with high probability?
4. **Use the Wilson interval for routine reporting** of proportions and the exact interval when a guarantee is needed; retire the Wald interval below about twenty expected events.
5. **Read small-count comparisons as interval overlaps**, not as ratios of point estimates; one failure against none establishes nothing.
6. **Pool exposure across units** for events in time, after checking that failures are independent, since the bound depends only on the total exposure.

## References

- Hanley, J. A., & Lippman-Hand, A. (1983). If nothing goes wrong, is everything all right? Interpreting zero numerators. *JAMA*, 249(13), 1743-1745.
- Jovanovic, B. D., & Levy, P. S. (1997). A look at the rule of three. *The American Statistician*, 51(2), 137-139.
- Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209-212.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101-133.
- Agresti, A., & Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119-126.
