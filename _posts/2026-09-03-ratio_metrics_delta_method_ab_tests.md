---
permalink: '/statistics/ratio_metrics_delta_method_ab_tests/'
title: 'Ratio Metrics in A/B Tests: The Session-Level Test Lies and the Delta Method Fixes It'
categories:
- Statistics
tags:
- Experimental Design
- A/B Testing
- Hypothesis Testing
- Statistics
author_profile: false
seo_title: 'Ratio Metrics and the Delta Method in A/B Tests'
seo_description: 'Conversion per session, revenue per order and clicks per view are ratios whose numerator and denominator come from the same users. Testing them as if every session were independent inflates false positives to 10 or 20 percent. The delta method with users as the unit restores 5 percent.'
excerpt: >-
  An A/A test on conversion per session, with two thousand users per arm
  and about three sessions each, comes back significant one time in ten.
  Nothing changed between the arms. The test counted sessions and the
  experiment randomised users.
summary: >-
  Why a per-session metric in a user-randomised experiment is a ratio of
  two user-level sums, why the session-level z-test understates its
  variance by the design effect of the clustering, a simulation of A/A
  tests showing the false positive rate rising with user heterogeneity and
  with sessions per user, the delta method variance that puts users back as
  the unit, a user-level bootstrap that agrees with it, the power cost of
  the honest test, and the difference between a ratio of sums and a mean
  of per-user ratios.
keywords:
  - ratio metrics
  - delta method
  - A/B testing
  - randomisation unit
  - analysis unit
  - design effect
  - cluster bootstrap
classes: wide
date: '2026-09-03'
why_this_exists: >-
  Almost every product metric is a ratio whose denominator is not the
  randomisation unit, and almost every experimentation tool will happily
  run a session-level proportion test on it. This post measures how wrong
  that is under realistic user heterogeneity and shows the two-line
  correction.
evidence: >-
  Simulated user-randomised A/A and A/B tests with 2,000 or 20,000 users
  per arm, per-user conversion propensities drawn from Beta distributions
  of varying concentration, and one to ten sessions per user; 2,000
  replications per cell.
methodology: >-
  Compares the false positive rate of the session-level proportion test,
  the delta-method test with users as the unit, and a user-level bootstrap
  under the null; checks the variance inflation against the design effect
  1 + (m - 1) rho; measures power for a five percent relative lift; and
  contrasts the ratio of sums with the mean of per-user rates when heavy
  users behave differently.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/data_science_17.jpg
  og_image: /assets/images/data_science_17.jpg
  overlay_image: /assets/images/data_science_17.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_17.jpg
  twitter_image: /assets/images/data_science_17.jpg
---
The experiment randomises users. The metric is conversion per session. The dashboard divides conversions by sessions in each arm, runs a two-proportion z-test on the two rates with the session counts as the sample sizes, and reports a p-value. It is the most common analysis in product experimentation, and when users have more than one session it is wrong in a specific, measurable way: the test believes it has as many independent observations as there are sessions, and it has as many as there are users.

Sessions from the same user are not independent. A user who converts easily converts in most of her sessions; one who never converts contributes a string of zeros. The metric's numerator and denominator are both sums over users, and the variance of their ratio depends on how users vary, not on how many sessions they generate. Run an A/A test, in which the two arms are identical by construction, and the session-level test rejects at well above its nominal 5 percent.

## Two Units

The randomisation unit is the user: assignment happens once per user and everything the user does lands in one arm. The analysis unit of the session-level test is the session. When the two differ, the test's variance formula is for a different experiment than the one that was run.

The correct variance is the variance of a ratio of sums. Write $X_i$ for user $i$'s sessions and $Y_i$ for her conversions, with $n$ users in an arm. The metric is $R = \sum Y_i / \sum X_i = \bar Y / \bar X$, and the delta method gives its variance to first order:

$$
\operatorname{Var}(R) \approx \frac{1}{n\,\mu_X^2}\left[\sigma_Y^2 - 2\frac{\mu_Y}{\mu_X}\sigma_{XY} + \frac{\mu_Y^2}{\mu_X^2}\sigma_X^2\right],
$$

where the means, variances and covariance are over users. Every quantity in it is a per-user statistic, and $n$ is the number of users. The session-level formula $p(1-p)/\sum X_i$ is what this reduces to when every user has exactly one session, or when users are identical.

## A Simulation

Each user has a propensity to convert drawn from a Beta distribution with mean 10 percent and a concentration parameter $\kappa$; a large $\kappa$ makes users alike, a small one makes them very different. Sessions per user are one plus a Poisson count, averaging three. Conversions are binomial in the user's sessions at her propensity. Three tests are run on each experiment: the session-level z-test, the delta-method test, and a bootstrap that resamples users.

```python
import numpy as np

rng = np.random.default_rng(0)

def experiment(n_users, lift=0.0, kappa=4.0, mean_sessions=3.0, r=rng):
    """Two arms of n_users; per-user propensity ~ Beta(mean 0.10 * (1 + lift), concentration kappa),
    sessions ~ 1 + Poisson. Returns per-user (sessions, conversions) for each arm."""
    out = []
    for arm_lift in (0.0, lift):
        p_mean = 0.10 * (1 + arm_lift)
        p = r.beta(p_mean * kappa, (1 - p_mean) * kappa, n_users)
        s = 1 + r.poisson(mean_sessions - 1, n_users)
        out.append((s, r.binomial(s, p)))
    return out

def naive_test(a, b):
    """Every session an independent Bernoulli trial."""
    (sa, ca), (sb, cb) = a, b
    pa, pb = ca.sum() / sa.sum(), cb.sum() / sb.sum()
    pooled = (ca.sum() + cb.sum()) / (sa.sum() + sb.sum())
    return (pb - pa) / np.sqrt(pooled * (1 - pooled) * (1 / sa.sum() + 1 / sb.sum()))

def delta_var(s, c):
    """Variance of sum(c)/sum(s) by the delta method, users as the unit."""
    n, mx, my = len(s), s.mean(), c.mean()
    vx, vy, cxy = s.var(ddof=1), c.var(ddof=1), np.cov(s, c, ddof=1)[0, 1]
    return (vy - 2 * (my / mx) * cxy + (my / mx) ** 2 * vx) / (mx ** 2 * n)

def delta_test(a, b):
    (sa, ca), (sb, cb) = a, b
    return (cb.sum() / sb.sum() - ca.sum() / sa.sum()) / np.sqrt(delta_var(sa, ca) + delta_var(sb, cb))

def bootstrap_test(a, b, r=rng, B=200):
    """Resample users, not sessions."""
    (sa, ca), (sb, cb) = a, b
    n = len(sa)
    diffs = []
    for _ in range(B):
        ia, ib = r.integers(0, n, n), r.integers(0, n, n)
        diffs.append(cb[ib].sum() / sb[ib].sum() - ca[ia].sum() / sa[ia].sum())
    return (cb.sum() / sb.sum() - ca.sum() / sa.sum()) / np.std(diffs, ddof=1)

reps = 2000
print("A/A tests declared significant at 5%")
for kappa in (1000.0, 20.0, 4.0, 1.0):
    hits = np.zeros(3)
    for _ in range(reps):
        a, b = experiment(2000, 0.0, kappa)
        hits += np.abs([naive_test(a, b), delta_test(a, b), bootstrap_test(a, b, B=200)]) > 1.96
    print(f"kappa {kappa:>5.0f}: naive {hits[0]/reps:.1%}  delta {hits[1]/reps:.1%}  bootstrap {hits[2]/reps:.1%}")
```

**False positive rates in A/A tests**, 2,000 users per arm, about three sessions each.

| User heterogeneity | Session-level test | Delta method | User bootstrap |
| --- | --- | --- | --- |
| Almost none ($\kappa = 1000$) | 4.5% | 4.5% | 4.5% |
| Mild ($\kappa = 20$) | 6.5% | 5.1% | 5.1% |
| Strong ($\kappa = 4$) | 10.3% | 4.8% | 4.7% |
| Extreme ($\kappa = 1$) | 20.2% | 4.8% | 4.5% |

When users are alike, all three tests agree and the session-level test is fine, because the only thing that made it wrong is absent. As users differ, the session-level test's false positive rate climbs to one in five, while the delta method and the user bootstrap stay at 5 percent. The strong-heterogeneity row is the realistic one: a conversion propensity spread with $\kappa = 4$ has a standard deviation of about 13 points around a 10 percent mean, which is what a mix of returning and casual users looks like.

## How Much Variance Was Missed

The inflation has a closed form. With $m$ sessions per user and an intra-user correlation $\rho$ between session outcomes, the variance of the session-level rate is larger than the independent-trials formula by the design effect $1 + (m-1)\rho$. For the Beta model, $\rho = 1/(\kappa + 1)$, which is 0.2 at $\kappa = 4$.

```python
s, c = experiment(20000, 0.0, 4.0)[0]
p, rho, m = 0.10, 1 / (4.0 + 1), s.mean()
print(f"ICC {rho:.3f}; mean sessions {m:.2f}; design effect {1 + (m - 1) * rho:.2f}")
print(f"delta variance / session-level variance: {delta_var(s, c) / (p * (1 - p) / s.sum()):.2f}")
```

The design effect is 1.40; the ratio of the delta-method variance to the session-level variance in one large sample is 1.53, the extra coming from the variation in session counts, which the design-effect formula with a fixed $m$ ignores. Either way, the session-level test is using standard errors about 20 percent too small, and a 20 percent understatement of the standard error is what turns 5 percent into 10.

The effect scales with sessions per user, because that is what determines how many "observations" the session-level test overcounts.

```python
for ms in (1.0, 2.0, 3.0, 5.0, 10.0):
    hits = np.zeros(2)
    for _ in range(reps):
        a, b = experiment(2000, 0.0, 4.0, mean_sessions=ms)
        hits += np.abs([naive_test(a, b), delta_test(a, b)]) > 1.96
    print(f"mean sessions {ms:>4.0f}: naive {hits[0]/reps:.1%}   delta {hits[1]/reps:.1%}")
```

| Mean sessions per user | Session-level test | Delta method |
| --- | --- | --- |
| 1 | 4.6% | 4.6% |
| 2 | 8.0% | 4.8% |
| 3 | 12.2% | 5.3% |
| 5 | 16.0% | 5.4% |
| 10 | 25.6% | 5.2% |

At one session per user the two tests are the same test. At ten sessions per user, which is a week of a habitual product, one A/A test in four is declared significant by the session-level analysis. That is also the rate at which real A/B tests of ineffective changes get shipped as wins.

![False positive rate of A/A tests on conversion per session against the mean number of sessions per user, for the session-level test under strong and mild user heterogeneity and for the delta method. The session-level test rises from 5 percent at one session to above 20 percent at ten; the delta method stays at 5.](/assets/images/figures/ratio_metric_false_positives.png){: width="1152" height="672" loading="lazy"}

## What the Honest Test Costs

The delta method does not lose power; it reports the power the experiment actually has. With 20,000 users per arm and a true 5 percent relative lift, from 10 to 10.5 percent conversion:

```python
hits = np.zeros(2)
for _ in range(reps):
    a, b = experiment(20000, 0.05, 4.0)
    hits += np.abs([naive_test(a, b), delta_test(a, b)]) > 1.96
print(f"power: naive {hits[0]/reps:.0%} (invalid), delta {hits[1]/reps:.0%}")
```

The session-level test finds the lift 77 percent of the time and the delta method 64 percent. The first number is not power in any useful sense, because the same test also finds lifts that do not exist 10 percent of the time; the second is the power the experiment has, and the sizing decision should be made with it. A test that needs 80 percent power on this metric needs more users than the session-level calculation says, by roughly the design effect.

## Ratio of Sums or Mean of Ratios

There is a second, quieter question hiding in "conversion per session": which average. The ratio of sums, total conversions over total sessions, weights each user by her sessions. The mean of per-user rates weights each user equally. When session counts are independent of propensity the two agree, and in the simulation above they do, 0.1004 against 0.1001 in a sample of 200,000 users. When they are not, the two are different numbers answering different questions.

```python
s, c = experiment(200000, 0.0, 4.0)[0]                    # sessions independent of propensity
print(f"independent: ratio of sums {c.sum()/s.sum():.4f}; mean of per-user rates {np.mean(c/s):.4f}")
p_user = rng.beta(0.1 * 4, 0.9 * 4, 200000)
s2 = 1 + rng.poisson(2 + 6 * (p_user < 0.05), 200000)     # low-propensity users browse more
c2 = rng.binomial(s2, p_user)
print(f"heavy users convert less: ratio of sums {c2.sum()/s2.sum():.4f}; mean of per-user rates {np.mean(c2/s2):.4f}")
```

When low-propensity users browse more, the ratio of sums is 0.056 and the mean of per-user rates is 0.100. Neither is wrong. The first is the fraction of sessions that convert, the number the business sees in its totals; the second is the typical user's conversion rate. A change that makes heavy browsers browse even more will move the first without touching the second. The choice is about the decision the experiment informs, and it has to be made before the test, because the two can move in opposite directions. Whichever is chosen, the variance is a user-level calculation: the delta method for the ratio of sums, an ordinary user-level t-test for the mean of ratios.

## The Same Problem Elsewhere

Any metric whose denominator is not the randomisation unit has this structure. Revenue per order when users are randomised. Click-through rate per impression. Average handling time per call when agents are randomised. Defect rate per part when batches are randomised. In each, the correct analysis takes the randomisation unit as the unit of variance, and the session-level, order-level or part-level proportion test overstates the evidence in proportion to how many of those units each randomised unit contributes and how much the randomised units differ.

The reverse mismatch exists too: randomising sessions and analysing users, which most platforms cannot do consistently because a user's later sessions would land in different arms. Where the randomisation unit is coarser than the analysis unit, which is the common case, the delta method or a cluster bootstrap is the fix; where it is the same, no fix is needed; where it is finer, the experiment usually has a bigger problem than its variance.

## What to Do

1. **Identify the randomisation unit** and treat it as the unit of analysis, whatever the metric's denominator is.
2. **For a ratio of sums, use the delta-method variance** with per-user sums as the inputs; it is a few lines and needs only the per-user sessions and conversions.
3. **Or bootstrap over users**, resampling whole users with their sessions attached; it agrees with the delta method and needs no formula.
4. **Size the experiment with the corrected variance.** The design effect $1 + (m-1)\rho$ is a quick first estimate of how much larger the user count must be than a session-level calculation implies.
5. **Decide between the ratio of sums and the mean of per-user rates before the test**, from the decision the result will inform, and report which one was used.
6. **Run A/A tests on your own pipeline.** A false positive rate above 5 percent in A/A data is the fastest diagnosis of a unit mismatch, and it costs nothing but compute.

## References

- Deng, A., Knoblich, U., & Lu, J. (2018). Applying the delta method in metric analytics: a practical guide with novel ideas. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 233-242.
- Deng, A., Lu, J., & Litz, J. (2017). Trustworthy analysis of online A/B tests: pitfalls, challenges and solutions. *Proceedings of the Tenth ACM International Conference on Web Search and Data Mining*, 641-649.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Bakshy, E., & Eckles, D. (2013). Uncertainty in online experiments with dependent data: an evaluation of bootstrap methods. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1303-1311.
