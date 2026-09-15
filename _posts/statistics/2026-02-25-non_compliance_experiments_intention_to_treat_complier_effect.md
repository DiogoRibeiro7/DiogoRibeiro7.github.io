---
permalink: '/statistics/non_compliance_experiments_intention_to_treat_complier_effect/'
title: "When Users Don't Take the Treatment: Intention to Treat, Per Protocol and the Complier Effect"
categories:
- Statistics
tags:
- Experimental Design
- A/B Testing
- Causal Inference
- Statistics
author_profile: false
seo_title: 'Non-Compliance in Experiments: ITT, Per Protocol and the Wald Estimator'
seo_description: 'When only some assigned users actually use a feature, comparing users by what they did instead of what they were assigned invents effects. A simulation shows as-treated and per-protocol analyses reporting effects that do not exist, and the intention-to-treat and Wald estimates that stay honest.'
excerpt: >-
  The feature is assigned to half the users and 60 percent of them turn
  it on. Comparing users who turned it on with users who did not gives an
  effect of 3.75. The true effect is 2.0, and on a feature that does
  nothing the same comparison gives 1.75.
summary: >-
  Why partial adoption breaks the naive comparison, the three kinds of
  user behind any adoption rate, a simulation contrasting intention to
  treat, as-treated, per-protocol and the Wald estimator on a feature with
  a real effect and on one with none, the cost of low compliance in
  sample size, the one-sided case where the complier effect is the effect
  on users who adopted, and how to report an experiment whose treatment
  was only partly taken.
keywords:
  - non-compliance
  - intention to treat
  - per protocol
  - complier average causal effect
  - Wald estimator
  - instrumental variables
  - A/B testing
classes: wide
date: '2026-02-25'
why_this_exists: >-
  Most product experiments assign exposure to a feature that users may or
  may not use, and the analysis that follows often compares users by what
  they did. This post shows on a simulation how far that goes wrong, what
  the honest estimates are, and how much traffic partial adoption costs.
evidence: >-
  Simulated experiments of 4,000 users assigned 50/50, with compliers,
  always-takers and never-takers whose baseline outcomes differ, a true
  effect of 2 for users who use the feature, compliance rates from 20 to
  90 percent, a feature with no effect, and one-sided non-compliance;
  3,000 replications per cell.
methodology: >-
  Compares the mean and spread of the intention-to-treat, as-treated,
  per-protocol and Wald estimators against their targets, checks which
  stay at zero when the feature does nothing, derives the sample size
  needed to detect the effect as compliance falls, and shows the Wald
  estimate equalling the effect on users who adopted when there are no
  always-takers.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-stars.jpg
  og_image: /assets/images/headers/photo-stars.jpg
  overlay_image: /assets/images/headers/photo-stars.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-stars.jpg
  twitter_image: /assets/images/headers/photo-stars.jpg
---
The experiment assigns a new planning feature to half the users. It is opt-in, and 60 percent of the assigned users turn it on. The analyst, reasonably enough, wants to know what the feature does for the people who use it, and compares the users who turned it on with the users who did not: the difference in the outcome is 3.75. The report says the feature is worth nearly four points.

The feature is worth two points, to the people who use it. The other 1.75 is who those people are. Users who turn on a planning feature are more organised than users who do not, and more organised users score higher on the outcome whether or not the feature exists. The comparison the analyst made is not an experiment. Randomisation assigned the offer; the users assigned the take-up, and they did so on the basis of exactly the characteristics the experiment was meant to balance.

## Three Kinds of User

Behind any adoption rate there are users who use the feature because they were assigned it and would not have otherwise, compliers; users who would have found and used it regardless of assignment, always-takers, which exist whenever the feature is discoverable outside the experiment; and users who would not use it even when assigned, never-takers. A fourth kind, defiers who do the opposite of their assignment, is assumed away, and for a feature offer that is a safe assumption.

Randomisation balances the three kinds across arms. It does not tell us which kind any user is: an assigned user who used the feature is a complier or an always-taker, and an unassigned user who did not is a complier or a never-taker. That is why comparing users by use mixes kinds with different baselines. Two estimators respect the randomisation. Intention to treat compares the arms as assigned, whoever used what, and estimates the effect of offering the feature. The Wald estimator divides that by the difference in take-up between arms, and estimates the effect of using the feature among the users whose use was changed by the offer, the compliers.

## A Simulation

Four thousand users, assigned 50/50. Sixty percent are compliers, 10 percent always-takers with a baseline three points higher, and 30 percent never-takers with a baseline two points lower. Using the feature adds two points. Four estimators are computed on each experiment.

```python
import numpy as np

rng = np.random.default_rng(0)
tau = 2.0          # effect of actually using the feature, for compliers

def draw(n, compliers=0.6, always=0.1, r=rng, effect=tau):
    """Assign z 50/50. Compliers use the feature iff assigned; always-takers use it regardless
    (and have higher baselines); never-takers never do (and have lower baselines)."""
    z = r.integers(0, 2, n)
    u = r.random(n)
    kind = np.where(u < compliers, "c", np.where(u < compliers + always, "a", "n"))
    d = np.where(kind == "c", z, np.where(kind == "a", 1, 0))
    base = 10 + np.where(kind == "a", 3.0, np.where(kind == "n", -2.0, 0.0))
    y = base + effect * d + r.normal(0, 5, n)
    return z, d, y

def itt(z, d, y):
    return y[z == 1].mean() - y[z == 0].mean()

def as_treated(z, d, y):
    return y[d == 1].mean() - y[d == 0].mean()

def per_protocol(z, d, y):
    """Assigned and used it, versus not assigned and did not."""
    return y[(z == 1) & (d == 1)].mean() - y[(z == 0) & (d == 0)].mean()

def wald(z, d, y):
    """ITT on the outcome over ITT on take-up: the complier average causal effect."""
    return itt(z, d, y) / (d[z == 1].mean() - d[z == 0].mean())

estimators = {"intention to treat": itt, "as treated": as_treated,
              "per protocol": per_protocol, "Wald / IV (complier effect)": wald}
reps = 3000
res = {k: [] for k in estimators}
for _ in range(reps):
    z, d, y = draw(4000)
    for k, fn in estimators.items():
        res[k].append(fn(z, d, y))
for k in estimators:
    print(f"{k:30} mean {np.mean(res[k]):.2f}  sd {np.std(res[k]):.2f}")
```

| Estimator | Mean estimate | Spread across experiments | What it targets |
| --- | --- | --- | --- |
| Intention to treat | 1.20 | 0.17 | Effect of the offer: 2.0 × 60% = 1.20 |
| As treated | 3.75 | 0.16 | Nothing: confounded by user type |
| Per protocol | 3.09 | 0.17 | Nothing: confounded by user type |
| Wald (complier effect) | 1.99 | 0.26 | Effect of use among compliers: 2.00 |

Intention to treat is exactly what it claims to be, the effect of offering the feature, diluted by the 40 percent who did not take the offer. The Wald estimate recovers the effect of use. The as-treated comparison, users who used it against users who did not, reports 3.75; the per-protocol comparison, assigned users who used it against unassigned users who did not, reports 3.09. Both are precise, both are wrong, and neither has any relationship to the truth that a reader could correct for without knowing the baselines of the three kinds of user, which is exactly what is unobservable.

The check that settles it is a feature that does nothing.

```python
res0 = {k: [] for k in estimators}
for _ in range(reps):
    z, d, y = draw(4000, effect=0.0)
    for k, fn in estimators.items():
        res0[k].append(fn(z, d, y))
print({k: f"{np.mean(res0[k]):+.2f}" for k in estimators})
```

| Estimator | Mean estimate when the true effect is zero |
| --- | --- |
| Intention to treat | -0.00 |
| As treated | +1.75 |
| Per protocol | +1.09 |
| Wald (complier effect) | -0.00 |

An inert feature shows an as-treated effect of 1.75 and a per-protocol effect of 1.09. That is the selection alone: the users who take features up score higher than those who do not, and the analyst who compares them will ship features that do nothing and believe they work. The two randomisation-respecting estimators sit at zero.

![Estimated effect against the share of assigned users who use the feature, for the four estimators, with a true effect of 2. Intention to treat falls in proportion to compliance; as-treated and per-protocol estimates sit above the truth at every rate; the Wald estimate recovers the complier effect throughout.](/assets/images/figures/noncompliance_estimators.png){: width="1152" height="672" loading="lazy"}

## What Low Compliance Costs

Intention to treat is unbiased, but it estimates a smaller quantity as compliance falls, and the noise does not fall with it. The Wald estimate scales it back up and scales the noise up with it. Either way, the experiment's ability to detect the effect of use degrades with the square of the compliance rate.

```python
from scipy import stats
z_ = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
for c in (0.9, 0.6, 0.4, 0.2):
    r_itt, r_w = [], []
    for _ in range(reps):
        z, d, y = draw(4000, compliers=c, always=0.1)
        r_itt.append(itt(z, d, y)); r_w.append(wald(z, d, y))
    n_req = 2 * 27.0 * z_ ** 2 / (tau * c) ** 2          # per arm; 27 is about the outcome variance
    print(f"compliers {c:.0%}: ITT {np.mean(r_itt):.2f} +/- {np.std(r_itt):.2f}, Wald {np.mean(r_w):.2f} +/- {np.std(r_w):.2f}, "
          f"{n_req:,.0f} users per arm for 80% power")
```

| Compliers | Intention to treat | Wald estimate | Users per arm for 80% power on the effect of use |
| --- | --- | --- | --- |
| 90% | 1.80 ± 0.16 | 2.00 ± 0.18 | 131 |
| 60% | 1.20 ± 0.17 | 2.00 ± 0.27 | 294 |
| 40% | 0.81 ± 0.17 | 2.01 ± 0.41 | 662 |
| 20% | 0.39 ± 0.17 | 1.95 ± 0.84 | 2,649 |

At 20 percent compliance the Wald estimate is still centred on the truth, but its spread is five times what it is at 90 percent, and the experiment needs twenty times the users to detect the same effect of use. The sample-size formula for the intention-to-treat contrast is the usual one with the effect multiplied by the compliance rate, so the users required grow as $1/c^2$. This is the practical argument for driving adoption before measuring effect: an experiment on a feature that a fifth of users adopt is, for the purpose of learning what the feature does, an experiment a twenty-fifth the size it appears to be.

## When There Are No Always-Takers

If the feature is only reachable through the experiment, nobody in the control arm can use it, and the compliers are exactly the users who adopted when assigned. The Wald estimate is then the effect on the users who used the feature, which is the quantity the analyst wanted in the first place.

```python
res1 = {k: [] for k in estimators}
for _ in range(reps):
    z, d, y = draw(4000, compliers=0.7, always=0.0)
    for k, fn in estimators.items():
        res1[k].append(fn(z, d, y))
print({k: f"{np.mean(res1[k]):.2f}" for k in estimators})
```

| Estimator, one-sided non-compliance | Mean estimate |
| --- | --- |
| Intention to treat | 1.40 |
| As treated | 2.92 |
| Per protocol | 2.60 |
| Wald (effect on users who adopted) | 2.00 |

The as-treated and per-protocol comparisons are still wrong, by about a point, even though every user who used the feature was assigned to it. The confounding is in who among the assigned chose to use it, and that choice is not randomised. Only the ratio of the two intention-to-treat contrasts removes it.

## Reading the Two Honest Numbers

Intention to treat and the complier effect answer different questions, and a report needs both.

The intention-to-treat effect is what shipping the feature will do to the metric across all users, given the adoption it actually gets. It is the number for the business decision about the launch as designed: 1.2 points, not 2, because 40 percent of users will not use it.

The complier effect is what the feature does for a user who uses it because of the launch. It is the number for the product decision about the feature itself, and for forecasting what a higher adoption rate would deliver, with the caution that compliers are a particular kind of user and the effect on never-takers, if they could be induced, might differ.

Neither number is the as-treated comparison, and no amount of covariate adjustment turns the as-treated comparison into one of them, because the covariates that distinguish the kinds of user are not measured. When the adoption rate is what the team wants to raise, the experiment to run is one that randomises the nudge and measures both take-up and outcome, which is the encouragement design that this whole analysis assumes.

## What to Do

1. **Analyse by assignment first.** The intention-to-treat contrast is the effect of the launch and is unbiased regardless of who adopted.
2. **Report the compliance rate** in both arms; the difference between them is the denominator of the complier effect and the diagnostic of whether the experiment can learn anything about use.
3. **Estimate the effect of use with the Wald ratio**, intention to treat on the outcome divided by intention to treat on take-up, with a delta-method or two-stage least-squares standard error.
4. **Never compare users by what they did.** As-treated and per-protocol comparisons report effects for inert features and cannot be corrected.
5. **Size the experiment for the compliance rate**: the users required for a given effect of use grow with the inverse square of adoption.
6. **Raise adoption before measuring effect** when compliance is low; an experiment that few users take up is far smaller than it looks.

## References

- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*, 91(434), 444-455.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Bloom, H. S. (1984). Accounting for no-shows in experimental evaluation designs. *Evaluation Review*, 8(2), 225-246.
- Hernán, M. A., & Robins, J. M. (2017). Per-protocol analyses of pragmatic trials. *New England Journal of Medicine*, 377(14), 1391-1398.
- Gerber, A. S., & Green, D. P. (2012). *Field Experiments: Design, Analysis, and Interpretation*. W. W. Norton.
- Deng, A., Lu, J., & Chen, S. (2016). Continuous monitoring of A/B tests without pain: optional stopping in Bayesian testing. *IEEE International Conference on Data Science and Advanced Analytics*, 243-252.
