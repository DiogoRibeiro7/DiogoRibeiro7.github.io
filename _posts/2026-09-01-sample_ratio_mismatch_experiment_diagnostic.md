---
permalink: '/statistics/sample_ratio_mismatch_experiment_diagnostic/'
title: 'Sample Ratio Mismatch: The One Diagnostic That Invalidates an Experiment'
categories:
- Statistics
tags:
- A/B Testing
- Experimental Design
- Data Quality
- Statistics
author_profile: false
seo_title: 'Sample Ratio Mismatch in A/B Tests'
seo_description: 'A 50/50 experiment that ends 49.75/50.25 has lost a thousand users to something that was not random. A simulation shows how small a mismatch a million-user test can detect, what a differential drop does to the measured lift, and why the result cannot be salvaged by adjustment.'
excerpt: >-
  The experiment assigned a million users evenly and logged 49.75 percent
  in the treatment arm. That is 2,470 users missing, a one-in-a-million
  coincidence, and it inflated the measured lift from 1.0 to 1.5 percent.
summary: >-
  Why an arm-size imbalance is a statement about the pipeline rather than
  about users, the chi-square test and how small a deviation each sample
  size can detect, a simulation of a differential drop showing the
  measured lift inflated by half while the alarm is still quiet, the
  symmetric drop that costs nothing, the common causes of mismatch, why
  no post-hoc adjustment restores the estimate, and the checks that
  belong in every experiment report.
keywords:
  - sample ratio mismatch
  - SRM
  - A/B testing
  - experiment validity
  - chi-square test
  - data quality
  - randomisation check
classes: wide
date: '2026-09-01'
why_this_exists: >-
  Sample ratio mismatch is the cheapest check in experimentation and the
  one most often skipped, because a fraction of a percent looks harmless
  next to sample sizes in the millions. This post measures what that
  fraction does to the result and shows why a failed check ends the
  analysis rather than qualifying it.
evidence: >-
  Simulated A/A experiments at 10,000 to 10,000,000 users to establish
  the null distribution of the chi-square check, and 1,000,000-user
  experiments with a true 1 percent lift in which 0 to 2 percent of
  treated users are dropped before logging, the dropped users converting
  at half the rate of the rest; 200 replications per cell.
methodology: >-
  Measures the false alarm rate of the chi-square check under a correct
  split, the probability of detecting a differential drop at each sample
  size, the measured lift and observed arm share for each drop rate, the
  same drop applied symmetrically, and the smallest detectable deviation
  from an even split by sample size.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-code.jpg
  og_image: /assets/images/headers/photo-code.jpg
  overlay_image: /assets/images/headers/photo-code.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-code.jpg
  twitter_image: /assets/images/headers/photo-code.jpg
---
The experiment ran for two weeks and assigned a million users, half to each arm. The results page shows 502,470 users in control and 497,530 in treatment, a split of 50.25 to 49.75, and a lift of 1.5 percent with a p-value of 0.001. Someone asks about the uneven counts and is told that a quarter of a percent is nothing on a million users.

A quarter of a percent is 2,470 users short of even in one arm. If assignment were a fair coin the standard deviation of an arm's size would be 500, so this split is nearly five standard deviations out, a coincidence with a probability of about one in a million. Something in the pipeline decided which users appeared in the results, and whatever it was did not consult the randomiser. In the simulation behind these numbers, that something was a one percent failure rate for slow-loading treated users, and it inflated a true lift of 1.0 percent to a measured 1.5.

## The Check

Under correct assignment the number of users in each arm is binomial, so the observed split can be compared with the intended one by a chi-square goodness-of-fit test with one degree of freedom. It takes two numbers and no assumptions about the outcome.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def chi2_p(n_a, n_b, expected=0.5):
    """Chi-square goodness of fit for the observed split against the intended one."""
    n = n_a + n_b
    e_a, e_b = n * expected, n * (1 - expected)
    stat = (n_a - e_a) ** 2 / e_a + (n_b - e_b) ** 2 / e_b
    return stats.chi2.sf(stat, 1)

for n in (10_000, 100_000, 1_000_000):
    ps = np.array([chi2_p(k, n - k) for k in rng.binomial(n, 0.5, 2000)])
    print(f"n = {n:,}: p < 0.05 in {np.mean(ps < 0.05):.1%}, p < 0.001 in {np.mean(ps < 0.001):.2%}")
```

| Users | Clean experiments with p < 0.05 | With p < 0.001 |
| --- | --- | --- |
| 10,000 | 4.5% | 0.05% |
| 100,000 | 5.4% | 0.05% |
| 1,000,000 | 5.7% | 0.10% |

The p-value is uniform when nothing is wrong, which is why the threshold matters. At 5 percent, one experiment in twenty raises a false alarm, and a team running a hundred experiments a week will stop believing the check by Wednesday. The usual threshold is 0.001: a false alarm once in a thousand experiments, and still enough power to catch the mismatches that matter.

## How Small a Mismatch Is Visible

The deviation the check can detect shrinks with the square root of the sample, which is what makes large experiments so sensitive.

```python
for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
    z = stats.norm.ppf(1 - 0.001 / 2)
    delta = z * 0.5 / np.sqrt(n)
    print(f"n = {n:>10,}: deviations above {delta:.4%} trip the alarm ({n * delta:,.0f} users)")
```

| Users | Smallest detectable deviation from 50% | Users |
| --- | --- | --- |
| 1,000 | 5.20% | 52 |
| 10,000 | 1.65% | 165 |
| 100,000 | 0.52% | 520 |
| 1,000,000 | 0.16% | 1,645 |
| 10,000,000 | 0.05% | 5,203 |

At a million users the check notices a sixth of a percent. That is the property that makes it valuable and the property that makes people distrust it: the deviations it flags look far too small to matter, and the reason they matter is not their size but their provenance. Randomisation produces deviations of a known size; a bug produces deviations of an unknown size, in an unknown direction, correlated with whatever the bug touches.

## What the Missing Users Cost

The simulation runs a million users with a true lift of one percent on a ten percent baseline. A fifth of users are on slow connections and convert at half the rate of everyone else. A share of the treated slow users never finish loading and never reach the log.

```python
def experiment(n=1_000_000, lift=0.01, drop=0.0, slow_share=0.20, slow_ratio=0.5, r=rng):
    z = r.integers(0, 2, n)
    slow = r.random(n) < slow_share
    base = np.where(slow, 0.10 * slow_ratio, 0.10 * (1 + slow_share * (1 - slow_ratio) / (1 - slow_share)))
    y = r.random(n) < base * (1 + lift * z)
    keep = np.ones(n, bool)
    if drop > 0:                        # only treated slow users are lost
        keep = ~((z == 1) & slow & (r.random(n) < min(1.0, drop / slow_share)))
    za, ya = z[keep], y[keep]
    return (ya[za == 1].mean() / ya[za == 0].mean() - 1), np.mean(za == 1), chi2_p(np.sum(za == 0), np.sum(za == 1))

for drop in (0.0, 0.002, 0.005, 0.01, 0.02):
    res = np.array([experiment(drop=drop) for _ in range(200)])
    print(f"drop {drop:.1%}: lift {res[:, 0].mean():+.2%}, treated share {res[:, 1].mean():.3%}, "
          f"median p {np.median(res[:, 2]):.1e}, alarms {np.mean(res[:, 2] < 0.001):.0%}")
```

| Treated users dropped | Measured lift (true 1.00%) | Observed treated share | Median SRM p-value | Experiments alarming at p < 0.001 |
| --- | --- | --- | --- | --- |
| 0% | +1.03% | 49.999% | 0.47 | 0% |
| 0.2% | +1.12% | 49.952% | 0.31 | 1% |
| 0.5% | +1.27% | 49.875% | 0.010 | 22% |
| 1.0% | +1.52% | 49.753% | 1 × 10⁻⁶ | 94% |
| 2.0% | +2.07% | 49.490% | 2 × 10⁻²⁴ | 100% |

Two things stand out. The bias is large relative to the effect long before it is large in absolute terms: losing one treated user in a hundred inflates a one percent lift to 1.52 percent, a 52 percent relative error, because the dropped users were disproportionately non-converters. And the alarm is well matched to the damage: it is nearly silent where the bias is a tenth of the effect, and it is certain where the bias is comparable to the effect itself. A team that investigates every p below 0.001 catches the cases that would have changed a decision.

The 0.2 percent row is the uncomfortable one. The bias is already a tenth of the effect and the check almost never fires. Nothing in the data reveals it; the defence there is not the statistical test but the engineering practice of logging assignment at the moment of randomisation rather than at the moment of exposure.

![Relative error in the measured lift and the share of experiments whose sample ratio check fires, against the share of treated users lost before logging. The check is nearly silent while the error is a tenth of the effect and fires reliably once the error approaches the effect itself.](/assets/images/figures/srm_detection_and_bias.png){: width="1152" height="672" loading="lazy"}

## Why the Result Cannot Be Rescued

The temptation after a failed check is to reweight the arms back to even, or to drop matching users from the larger arm, and carry on. Neither works, for the same reason: the missing users are missing because of something related to the treatment, and nothing in the data says what they would have done. The arms are no longer comparable in the way randomisation guaranteed, and the only honest quantity is the effect among users who survived the pipeline, which is not the effect of the treatment on users.

The contrast is a drop that applies equally to both arms.

```python
def experiment_symmetric(n=1_000_000, lift=0.01, drop=0.002, r=rng):
    z = r.integers(0, 2, n)
    slow = r.random(n) < 0.20
    y = r.random(n) < np.where(slow, 0.05, 0.1125) * (1 + lift * z)
    lose = slow & (r.random(n) < drop / 0.20)
    za, ya = z[~lose], y[~lose]
    return ya[za == 1].mean() / ya[za == 0].mean() - 1, np.mean(za == 1)
```

Dropping the same share of slow users from both arms gives a measured lift of 1.00 percent against a true 1.00 percent, and a treated share of 50.001 percent. The loss of data costs a little precision and no validity, because the filter did not know which arm a user was in. That is the distinction the check is testing for, and it is why the check is about the mechanism rather than about the number of users lost.

## Where Mismatches Come From

The causes repeat across organisations, and knowing them shortens the investigation.

**Assignment logged at exposure rather than at randomisation.** If a user is only recorded once the treatment renders, anything that stops it rendering, a slower bundle, a failed request, an ad blocker, removes treated users and not control users. This is the most common cause and the one the simulation models.

**Filters applied after assignment.** Bot detection, fraud rules, outlier removal and "engaged users only" definitions all run on behaviour, and behaviour is what the treatment changes. A filter that removes users who spent under five seconds on the page will remove more of whichever arm is faster.

**Redirects and multi-step flows.** A treatment implemented as a redirect loses users at the redirect, and the control does not redirect.

**Carry-over from a previous experiment.** Users held out or excluded by an earlier test enter the new one non-randomly, so the imbalance is inherited rather than created.

**Ratio not actually 50/50.** A ramp from 10 to 50 percent, or a targeting rule that intersects with the randomiser, gives an intended ratio that is not what the check compares against. The fix is to test against the intended ratio for the period, and to reset the analysis window when a ramp changes.

The diagnostic that separates these is to run the check on segments: by day, by platform, by browser, by country, by entry point. A mismatch confined to one platform or beginning on the day of a deployment names its own cause.

## What to Do

1. **Run the check on every experiment** with a chi-square test against the intended ratio, at a threshold of 0.001, and show the result in the report next to the effect.
2. **Treat a failed check as fatal.** Do not reweight, trim or re-randomise the analysis; fix the pipeline and rerun the experiment.
3. **Log assignment at randomisation**, not at exposure, so that users who never see the treatment are still counted in their arm.
4. **Check segments when the alarm fires**: by day, platform and entry point. A mismatch in one segment identifies the defect faster than any amount of analysis of the whole.
5. **Compare against the intended ratio**, including during ramps, and restart the analysis window whenever the allocation changes.
6. **Audit the filters** applied between assignment and analysis; each one that reads behaviour is a candidate cause, and each should be applied identically to both arms or not at all.

## References

- Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer, L., & Dmitriev, P. (2019). Diagnosing sample ratio mismatch in online controlled experiments: a taxonomy and rules of thumb for practitioners. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2156-2164.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y., & Pohlmann, N. (2013). Online controlled experiments at large scale. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1168-1176.
- Zhao, Z., Chen, M., Matheson, D., & Stone, M. (2016). Online experimentation diagnosis and troubleshooting beyond AA validation. *IEEE International Conference on Data Science and Advanced Analytics*, 498-507.
- Chen, N., Liu, M., & Xu, Y. (2019). How A/B tests could go wrong: automatic diagnosis of invalid online experiments. *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining*, 501-509.
- Crook, T., Frasca, B., Kohavi, R., & Longbotham, R. (2009). Seven pitfalls to avoid when running controlled experiments on the web. *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1105-1114.
