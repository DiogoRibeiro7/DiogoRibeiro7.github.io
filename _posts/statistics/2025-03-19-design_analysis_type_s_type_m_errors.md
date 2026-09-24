---
permalink: '/statistics/design_analysis_type_s_type_m_errors/'
title: 'Design Analysis: What a Significant Result Means in a Small Study'
categories:
- Statistics
tags:
- Hypothesis Testing
- Experimental Design
- Statistics
author_profile: false
seo_title: 'Type S and Type M Errors: Reading Significance at Low Power'
seo_description: 'At ten percent power, a significant estimate averages 3.7 times the true effect and points the wrong way once in twenty-two. A simulation matches the closed form exactly, and shows why power computed after the fact is the p-value in disguise.'
excerpt: >-
  The test had about ten percent power against the effect that was
  plausible beforehand. It came back significant at 7.7 percent. The
  truth was 2 percent, and that is not bad luck: any significant result
  from that study had to be at least three times the truth, because
  nothing smaller could have cleared the threshold.
summary: >-
  Why the significance filter guarantees exaggeration when power is low,
  the closed form for how often a significant estimate has the wrong
  sign and by how much it overstates the effect, why power computed from
  the observed effect is a relabelled p-value, what replication does to
  such results, and how designing for interval width avoids the whole
  problem.
keywords:
  - design analysis
  - type S error
  - type M error
  - exaggeration ratio
  - statistical power
  - post-hoc power
classes: wide
date: '2025-03-19'
why_this_exists: >-
  Underpowered studies do not merely fail to detect effects, they
  systematically misreport the ones they do detect, and the size of that
  distortion follows from the design alone. Knowing it before the study
  runs changes what gets built and what gets believed afterwards.
evidence: >-
  Closed-form calculations for the probability of a wrong sign and the
  expected magnitude of a significant estimate under a normal sampling
  distribution, checked against 400,000 simulated studies at each of
  seven power levels, plus replication simulations of 200,000 pairs.
methodology: >-
  Computes the sign error and exaggeration ratio conditional on
  significance by numerical integration and verifies both by simulation,
  applies them to a concrete experiment at four sample sizes, examines
  the relationship between the p-value and power computed from the
  observed effect, measures replication rates and shrinkage, and
  compares sizing for precision against sizing for power.
reviewed_at: '2026-09-13'
header:
  image: /assets/images/headers/photo-lights.jpg
  og_image: /assets/images/headers/photo-lights.jpg
  overlay_image: /assets/images/headers/photo-lights.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-lights.jpg
  twitter_image: /assets/images/headers/photo-lights.jpg
---

A team ran an experiment on four hundred users per arm and reported a 7.7 percent improvement, significant at the five percent level. The effect that the business case had assumed, and that later evidence supported, was 2 percent. The report was not fraudulent, the analysis was not wrong, and the estimate was not unlucky in any unusual way. It is what that design produces when it produces anything at all.

With four hundred users per arm the standard error was 3.2 percent, so clearing significance required an estimate of at least 6.2 percent. A true effect of 2 percent can only be declared significant by being overstated threefold. The filter, not the phenomenon, set the number that went in the report.

## The Significance Filter

Two quantities describe what the filter does, and both follow from the design alone: the chance that a significant estimate points the wrong way, and the factor by which it overstates the truth on average. They are usually called the Type S and Type M errors, for sign and magnitude.

```python
import numpy as np
from scipy import stats, integrate

RNG = np.random.default_rng(37)
ZC = stats.norm.ppf(0.975)


def power(delta, se):
    """Chance of a significant result, either sign."""
    z = delta / se
    return stats.norm.cdf(-ZC - z) + stats.norm.cdf(z - ZC)


def type_s(delta, se):
    """Chance the significant result points the wrong way."""
    z = delta / se
    wrong = stats.norm.cdf(-ZC - z)
    return wrong / power(delta, se)


def type_m(delta, se):
    """Average size of a significant estimate, relative to the truth."""
    z = delta / se
    f = lambda x: abs(x) * stats.norm.pdf(x, z, 1)
    lo = integrate.quad(f, -np.inf, -ZC)[0]
    hi = integrate.quad(f, ZC, np.inf)[0]
    return (lo + hi) / power(delta, se) / abs(z)


for target in (0.10, 0.15, 0.20, 0.35, 0.50, 0.80, 0.95):
    # Find the true effect, in standard errors, that gives this power.
    z = 0.0
    lo, hi = 0.0, 8.0
    for _ in range(60):
        z = (lo + hi) / 2
        if power(z, 1.0) < target:
            lo = z
        else:
            hi = z
    est = RNG.normal(z, 1.0, 400000)
    sig = np.abs(est) >= ZC
    print(f"{power(z, 1.0):7.0%} {z:6.2f} {type_s(z, 1.0):11.2%} {type_m(z, 1.0):12.2f}x "
          f"{np.mean(est[sig] * z < 0):21.2%} {np.mean(np.abs(est[sig])) / z:22.2f}x")
```

| Power | True effect, in standard errors | Wrong sign | Exaggeration | Simulated wrong sign | Simulated exaggeration |
| --- | --- | --- | --- | --- | --- |
| 10% | 0.65 | 4.50% | 3.71x | 4.51% | 3.71x |
| 15% | 0.91 | 1.35% | 2.70x | 1.33% | 2.70x |
| 20% | 1.11 | 0.53% | 2.26x | 0.58% | 2.26x |
| 35% | 1.57 | 0.06% | 1.67x | 0.06% | 1.67x |
| 50% | 1.96 | 0.01% | 1.41x | 0.01% | 1.41x |
| 80% | 2.80 | 0.00% | 1.12x | 0.00% | 1.12x |
| 95% | 3.60 | 0.00% | 1.03x | 0.00% | 1.03x |

Integration and simulation agree to two decimal places in every row. The table is a property of the design, not of any particular result: it says what significance is worth before a single observation is collected.

At 80 percent power the distortion is minor, twelve percent on average, and the sign is never wrong in practice. At 20 percent power a significant estimate is more than twice the truth. At 10 percent it is nearly four times, and one significant result in twenty-two has the wrong sign, which means confidently reporting a harm as a benefit. As power falls toward the five percent floor, the true effect approaches zero and the exaggeration ratio grows without bound, which is the formal way of saying that a study with no power to detect anything can only ever detect noise.

![Exaggeration of a significant estimate against the power of the study, for significance thresholds of 0.10, 0.05 and 0.01, on a log scale. All three sit near one above 60 percent power and rise steeply below 30, and at equal power the looser threshold exaggerates most.](/assets/images/figures/design_analysis_type_sm.png){: width="1152" height="672" loading="lazy"}

The figure adds one detail the table cannot show. Comparing thresholds at equal power, the looser one exaggerates more: reaching 20 percent power against a bar at 0.10 takes a smaller true effect than reaching it against a bar at 0.01, and a smaller truth means a larger ratio. Findings reported at 0.10 from small studies are the most inflated ones in the literature.

## A Concrete Experiment

The abstract version becomes clearer applied to a real design: an outcome with a standard deviation of 0.45 on a rate of a few percent, a true effect of 2 percent, and four sample sizes.

```python
TRUE = 0.02          # a 2% improvement
for n_per_arm, sd in ((400, 0.45), (2000, 0.45), (10000, 0.45), (50000, 0.45)):
    se = sd * np.sqrt(2 / n_per_arm)
    print(f"{n_per_arm:6d} per arm: se {se:.4f}, power {power(TRUE, se):5.1%}, "
          f"wrong sign {type_s(TRUE, se):6.2%}, exaggeration {type_m(TRUE, se):5.2f}x, "
          f"a significant result reads about {TRUE * type_m(TRUE, se):5.2%}")
```

| Users per arm | Standard error | Power | Wrong sign | Exaggeration | A significant result reads about |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.0318 | 9.6% | 5.00% | 3.85x | 7.69% |
| 2,000 | 0.0142 | 29.0% | 0.13% | 1.84x | 3.68% |
| 10,000 | 0.0064 | 88.2% | 0.00% | 1.07x | 2.14% |
| 50,000 | 0.0028 | 100.0% | 0.00% | 1.00x | 2.00% |

The first row is the study from the introduction, and it predicts the reported 7.7 percent almost exactly. That is the useful property of design analysis: it is computable in advance, from the sample size, the outcome's variability and an externally plausible effect. Had anyone done this arithmetic before running, the finding would have been read as "consistent with a 2 percent effect", which is what it was.

The fourth row shows the other side. A test with real power reports the truth, and its estimate needs no interpretation beyond its interval.

## Power Computed Afterwards Is the P-Value

When a result disappoints, someone usually computes power from the observed effect. That number carries no information about the design, because it is a monotone function of the p-value.

```python
se = 0.45 * np.sqrt(2 / 400)
est = RNG.normal(TRUE, se, 20000)
p = 2 * (1 - stats.norm.cdf(np.abs(est) / se))
post_hoc = power(np.abs(est), se)          # power computed from the observed effect
order = np.argsort(p)
for q in (0.001, 0.01, 0.05, 0.20, 0.50):
    i = np.argmin(np.abs(p[order] - q))
    print(f"  p = {p[order][i]:.3f}  ->  post-hoc power {post_hoc[order][i]:6.1%}")
print(f"correlation between p-value and post-hoc power: "
      f"{np.corrcoef(p, post_hoc)[0, 1]:+.3f} (rank {stats.spearmanr(p, post_hoc).statistic:+.3f})")
```

| Observed p-value | Power computed from the same data |
| --- | --- |
| 0.001 | 90.8% |
| 0.010 | 73.1% |
| 0.050 | 50.0% |
| 0.200 | 24.9% |
| 0.500 | 10.4% |

The rank correlation between the two is exactly −1.000. Every p-value maps to one post-hoc power and back again, so reporting it adds nothing, and the familiar sentence "the result was not significant but power was low, so the effect may be real" is circular: the power was low *because* the result was not significant. Design analysis avoids this by using an effect size from outside the data, from prior studies, from a business case, or from the smallest effect that would change a decision.

## What Replication Does

If a significant result from a small study is mostly filter, a repeat of the same study should disappoint. It does, predictably.

```python
for target in (0.20, 0.50, 0.80):
    z = 0.0
    lo, hi = 0.0, 8.0
    for _ in range(60):
        z = (lo + hi) / 2
        if power(z, 1.0) < target:
            lo = z
        else:
            hi = z
    first = RNG.normal(z, 1.0, 200000)
    sig = np.abs(first) >= ZC
    second = RNG.normal(z, 1.0, 200000)
    same_sign = np.sign(second[sig]) == np.sign(first[sig])
    confirms = (np.abs(second[sig]) >= ZC) & same_sign
    shrinks = np.abs(second[sig]) < np.abs(first[sig])
    print(f"power {power(z, 1.0):4.0%}: of the significant first results, "
          f"{confirms.mean():5.1%} replicate significantly with the same sign, "
          f"and {shrinks.mean():5.1%} shrink on the second run")
```

| Power of both studies | Replicate significantly with the same sign | Shrink on the second run |
| --- | --- | --- |
| 20% | 19.7% | 90.0% |
| 50% | 50.0% | 75.0% |
| 80% | 80.2% | 59.9% |

Two details are worth pulling out. The replication rate equals the power, which is obvious in hindsight and rarely stated: a field running 20 percent power studies will see one in five of its findings repeat, and will experience that as a crisis rather than as arithmetic. And nine times in ten the second estimate is smaller than the first, so the common experience of an effect "fading" on a second look is the expected behaviour of an unbiased estimator behind a filter.

Even at 80 percent power, six times in ten the repeat comes in lower. Shrinkage is not evidence that the first result was wrong.

## Designing for Precision

The way out is to stop sizing for the ability to reject a hypothesis and start sizing for the width of the interval the study will produce. Precision is a promise about the answer rather than about the verdict.

```python
for width in (0.04, 0.02, 0.01, 0.005):
    se_needed = width / (2 * ZC)
    n = 2 * 0.45 ** 2 / se_needed ** 2
    print(f"interval width {width:5.1%}: se {se_needed:.4f}, "
          f"{n:9,.0f} per arm, exaggeration at a true 2% effect "
          f"{type_m(TRUE, se_needed):5.2f}x")
```

| Target interval width | Standard error | Users per arm | Exaggeration at a true 2% effect |
| --- | --- | --- | --- |
| ±2 percentage points | 0.0102 | 3,889 | 1.41x |
| ±1 percentage point | 0.0051 | 15,558 | 1.02x |
| ±0.5 percentage points | 0.0026 | 62,232 | 1.00x |
| ±0.25 percentage points | 0.0013 | 248,927 | 1.00x |

Sizing so the interval is about half the effect you care about removes the exaggeration entirely, and it produces a result that is interpretable whether or not it clears any threshold. An interval of 1.0 to 3.0 percent, which is what 15,558 users per arm buys here, answers the business question. A significant 7.7 percent with a lower bound near zero does not, even though it looks more decisive.

When that sample size is unaffordable, the honest response is not to run the small study anyway and hope. It is either to accept an unclear answer and plan accordingly, to reduce the variance through better measurement or covariate adjustment, or to choose a decision rule that does not need a precise estimate.

## What to Do

1. Compute the exaggeration ratio and sign error during design, using an effect size from outside the data. Both take three lines and they tell you what a significant result would be worth.
2. Treat significance from a low-powered design as weak evidence that something happened and no evidence about how large it is.
3. Never compute power from the observed effect. It is the p-value wearing different units, and it cannot tell you what the study could have detected.
4. Size for interval width rather than for power when the number matters, which is most of the time in operational work.
5. Expect the second estimate to be smaller. Shrinkage on replication is the normal behaviour of the filter, not a sign that the first result was fabricated or that the effect has decayed.
6. Report the interval as the headline, not the point estimate. It carries the uncertainty that the exaggeration ratio is quantifying.

The [figure generator](https://github.com/DiogoRibeiro7/blog-reproducibility/blob/main/scripts/figures/statistics/design_analysis.py) in the [blog-reproducibility repository](https://github.com/DiogoRibeiro7/blog-reproducibility) reproduces this article's figure; run it with `--dry-run` to print the numbers behind the figure without writing an image.

## References

- Gelman, A., & Carlin, J. (2014). Beyond power calculations: assessing Type S (sign) and Type M (magnitude) errors. *Perspectives on Psychological Science*, 9(6), 641-651.
- Gelman, A., & Tuerlinckx, F. (2000). Type S error rates for classical and Bayesian single and multiple comparison procedures. *Computational Statistics*, 15(3), 373-390.
- Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power: the pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19-24.
- Button, K. S., Ioannidis, J. P. A., Mokrysz, C., Nosek, B. A., Flint, J., Robinson, E. S. J., & Munafò, M. R. (2013). Power failure: why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience*, 14(5), 365-376.
- Ioannidis, J. P. A. (2008). Why most discovered true associations are inflated. *Epidemiology*, 19(5), 640-648.
- Rothman, K. J., & Greenland, S. (2018). Planning study size based on precision rather than power. *Epidemiology*, 29(5), 599-603.
