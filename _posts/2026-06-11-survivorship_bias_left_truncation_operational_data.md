---
permalink: '/statistics/survivorship_bias_left_truncation_operational_data/'
title: 'Survivorship Bias in Operational Data: When the Failures Are Missing from the Table'
categories:
- Statistics
tags:
- Survival Analysis
- Reliability
- Data Quality
- Statistics
author_profile: false
seo_title: 'Survivorship Bias and Left Truncation in Operational Data'
seo_description: 'Analysing the machines, customers or accounts that are still there overstates lifetimes, hides the harshest conditions and inflates failure rates in the wrong direction. A simulated fleet shows the biases and the left-truncation correction that removes them.'
excerpt: >-
  A reliability study follows every machine in service today for two years
  and estimates a median life of 7.7 years. The true median is 3.9. The
  machines that would have pulled the estimate down failed before the study
  started, and the ones that remain are the survivors.
summary: >-
  Why a snapshot of units in service is a biased sample of the units ever
  installed, the three distinct distortions it produces (inflated
  lifetimes, a changed mix of conditions and misleading failure rates), a
  simulated fleet with a rising hazard and two environments that puts
  numbers on each, the left-truncation adjustment to the Kaplan-Meier
  estimator that recovers the truth from the same data, and how to
  recognise the problem in customer, credit and equipment datasets.
keywords:
  - survivorship bias
  - left truncation
  - delayed entry
  - Kaplan-Meier
  - reliability
  - length-biased sampling
  - selection bias
classes: wide
date: '2026-06-11'
why_this_exists: >-
  Operational datasets are almost always extracted at a point in time from
  the units that exist at that moment, and the units that failed, churned or
  defaulted earlier are absent. This post measures what that does to
  lifetime estimates, group comparisons and rates on a simulated fleet, and
  shows that the correction needs no new data, only the entry times.
evidence: >-
  A simulated fleet of 20,000 units installed uniformly over twelve years
  with Weibull lifetimes with a rising hazard, in two environments with
  characteristic lives of six and four years; a study cohort of the units
  in service at year twelve followed for two years.
methodology: >-
  Compares the snapshot's age and environment mix with the fleet's, estimates
  median lifetime by Kaplan-Meier with and without delayed entry against the
  reference from every unit ever installed, repeats the comparison by
  environment, and contrasts two-year failure probabilities and failure
  rates for survivors against new units.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-waves.jpg
  og_image: /assets/images/headers/photo-waves.jpg
  overlay_image: /assets/images/headers/photo-waves.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-waves.jpg
  twitter_image: /assets/images/headers/photo-waves.jpg
---
The reliability team is asked how long the pumps last. They take the asset register as of this year, which lists every pump currently in service with its installation date, follow the fleet for two years, record the failures, and fit a survival curve. The median life comes out at 7.7 years. The manufacturer's figure was closer to four, and the team writes a note explaining that field conditions must be milder than the test bench.

Field conditions are not milder. The true median in the simulation this post is built on is 3.9 years. The register the team started from lists the pumps that are still running, and a pump that failed in its second year is not in it. The sample was drawn by survival, and every estimate made on it inherits the selection.

## Three Distortions from One Selection

Sampling the units that exist at a snapshot does three different things to the data, and they are worth separating because they call for different corrections.

**The lifetimes are inflated.** A unit is in service at the snapshot only if its lifetime exceeds its current age. Long-lived units are over-represented, and the more so the older the fleet. This is length-biased sampling, the same mechanism that makes the average bus wait feel longer than half the headway.

**The mix has changed.** Anything that shortens life is under-represented among survivors. If harsh sites kill pumps faster, the surviving fleet has fewer pumps at harsh sites than were installed there, and any comparison of conditions on the survivors is made on a sample that has already thrown out most of the evidence against the harsh ones.

**The clock is wrong.** A study that starts following survivors at the snapshot observes them from their current age onward, not from installation. Treating the snapshot as time zero measures something ("remaining life of the current fleet") that is real but not the lifetime; treating installation as time zero but including each unit in the risk set from age zero, when it was only observed from its current age, is the error that produces 7.7 years.

## A Simulated Fleet

Twenty thousand units are installed at uniform times over twelve years. Lifetimes follow a Weibull distribution with shape 1.5, so the hazard rises with age, and a characteristic life of six years at normal sites and four at harsh sites; 40 percent of installations are harsh. At year twelve the analyst extracts the units in service and follows them for two years.

```python
import numpy as np

rng = np.random.default_rng(0)
shape = 1.5                                 # Weibull shape: hazard rises with age
scale = {"normal": 6.0, "harsh": 4.0}       # characteristic life in years by environment
n_units, years, followup = 20000, 12.0, 2.0

env = rng.choice(["normal", "harsh"], n_units, p=[0.6, 0.4])
install = rng.uniform(0, years, n_units)
life = np.array([rng.weibull(shape) * scale[e] for e in env])
fail_time = install + life

true_median = {e: scale[e] * np.log(2) ** (1 / shape) for e in scale}
print("true median lifetime:", {e: f"{m:.2f}" for e, m in true_median.items()})

in_service = fail_time > years
age_now = years - install
print(f"installed {n_units}, in service at year 12: {in_service.sum()} ({in_service.mean():.0%})")
print(f"harsh share: installed {np.mean(env == 'harsh'):.0%}, in service {np.mean(env[in_service] == 'harsh'):.0%}")
print(f"mean age in service {age_now[in_service].mean():.2f} y; survivors' eventual mean life "
      f"{life[in_service].mean():.2f} y; fleet-wide mean life {life.mean():.2f} y")
```

Of 20,000 units installed, 7,644 (38 percent) are in service at year twelve. Harsh sites account for 40 percent of installations and 32 percent of survivors. The survivors' mean age is 3.4 years, and the lifetimes they will eventually reach average 6.9 years against 4.7 for the fleet as a whole: the snapshot has selected units that live half as long again as a typical one, before a single failure has been observed.

The fleet takes a dozen lines of NumPy because one Weibull family and one covariate are all the argument needs. When the question involves competing failure modes, cure fractions, recurrent events or multi-state processes, [gen-surv](/packages/gensurvpy/) generates survival data with a known truth for exactly this kind of check, and the left-truncation comparison below works unchanged on its output.

## Estimating the Lifetime Three Ways

The study cohort is the 7,644 survivors. Each enters at its current age and exits at failure or at the end of the two-year window, whichever is first; 2,940 fail in the window. The Kaplan-Meier estimator below takes an optional entry age, and the only thing the entry age changes is the risk set: a unit is at risk of failing at age $t$ only if it was under observation at $t$, that is, if it entered before $t$ and had not yet exited.

```python
def km(times, events, entry=None):
    """Kaplan-Meier with optional delayed entry (left truncation)."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    entry = np.zeros_like(times) if entry is None else entry[order]
    grid = np.unique(times[events == 1])
    surv, s = [], 1.0
    for t in grid:
        at_risk = np.sum((entry < t) & (times >= t))
        d = np.sum((times == t) & (events == 1))
        s *= 1 - d / at_risk
        surv.append(s)
    return grid, np.array(surv)

def median_from(grid, surv):
    below = np.where(surv <= 0.5)[0]
    return grid[below[0]] if len(below) else np.inf

idx = np.where(in_service)[0]
entry_age = age_now[idx]
exit_age = np.minimum(life[idx], entry_age + followup)
event = (life[idx] <= entry_age + followup).astype(int)

g, s = km(exit_age - entry_age, event)            # clock starts at the snapshot
print("clock at study entry:          ", median_from(g, s))
g, s = km(exit_age, event)                        # age scale, every survivor in the risk set from age 0
print("age scale, ignoring entry:     ", round(median_from(g, s), 2))
g, s = km(exit_age, event, entry_age)             # age scale, risk set restricted to units under observation
print("age scale with left truncation:", round(median_from(g, s), 2))
g_full, s_full = km(life, np.ones(n_units, int))  # reference: every unit ever installed
print("reference, all units:          ", round(median_from(g_full, s_full), 2))
```

| Estimate | Median lifetime |
| --- | --- |
| Clock starts at the snapshot | undefined (fewer than half fail in two years) |
| Age scale, survivors counted from age zero | 7.74 years |
| Age scale with left truncation | 3.83 years |
| Reference: every unit ever installed | 3.95 years |

The first row is not wrong, only about something else: it estimates the remaining life of the current fleet, which nobody asked for. The second row is the 7.7 that went into the report. It puts every survivor into the risk set from age zero, so the estimator sees 7,644 units "surviving" their first three years with almost no failures, because the failures in those years happened to units that are not in the table. The third row uses the same 7,644 units and the same 2,940 failures and changes only the risk set; it recovers the reference to within a tenth of a year.

![Survival curves against age for the simulated fleet: the reference from every unit ever installed, the curve from survivors counted from age zero, and the left-truncated curve from the same survivors. Counting survivors from age zero roughly doubles the apparent median; the left-truncated estimate lies on the reference.](/assets/images/figures/survivorship_left_truncation.png){: width="1152" height="672" loading="lazy"}

## The Comparison Between Sites

The mix distortion shows up when the environments are compared. Both the true medians and the bias differ by site.

```python
print(f"{'':8}{'true':>7}{'ignoring entry':>16}{'left-truncated':>16}")
for e in ("normal", "harsh"):
    m = env[idx] == e
    g1, s1 = km(exit_age[m], event[m])
    g2, s2 = km(exit_age[m], event[m], entry_age[m])
    print(f"{e:8}{true_median[e]:>7.2f}{median_from(g1, s1):>16.2f}{median_from(g2, s2):>16.2f}")
```

| Site | True median | Survivors from age zero | Left-truncated |
| --- | --- | --- | --- |
| Normal | 4.70 | 8.85 | 4.61 |
| Harsh | 3.13 | 5.60 | 3.00 |

Without the correction both medians are inflated, and the harsh sites are inflated proportionally more, because selection by survival removes a larger share of the short-lived harsh units. The gap between sites, 1.6 years in truth, reads as 3.3 years on the naive curves; a decision about whether harsh sites justify a different maintenance regime would be made on a difference twice its real size, from a fleet in which the harsh units that mattered most are absent. The left-truncated estimates put both medians within a tenth of a year of the truth.

## Failure Rates That Point the Wrong Way

Rates are where survivorship bias becomes counter-intuitive. The survivors are the long-lived units, so one expects them to fail less. They are also older, and with a rising hazard, age wins.

```python
p_new = {e: 1 - np.exp(-(followup / scale[e]) ** shape) for e in scale}
print(f"two-year failure probability: survivors {event.mean():.1%}, new unit "
      f"{0.6*p_new['normal'] + 0.4*p_new['harsh']:.1%}")
for e in ("normal", "harsh"):
    m = env[idx] == e
    print(f"{e}: survivors {event[m].sum() / (exit_age[m] - entry_age[m]).sum():.3f} failures per unit-year")

fresh = rng.choice(["normal", "harsh"], 20000, p=[0.6, 0.4])
fresh_life = np.array([rng.weibull(shape) * scale[e] for e in fresh])
for e in ("normal", "harsh"):
    m = fresh == e
    t = np.minimum(fresh_life[m], followup)
    print(f"{e}: from installation {np.sum(fresh_life[m] <= followup) / t.sum():.3f} failures per unit-year")
```

| Quantity | Survivors at the snapshot | New units from installation |
| --- | --- | --- |
| Probability of failing within two years | 38.5% | 22.4% |
| Failures per unit-year, normal sites | 0.202 | 0.098 |
| Failures per unit-year, harsh sites | 0.334 | 0.167 |

The surviving fleet fails at twice the rate of new units. A third of the survivors are older than four years, and at those ages the hazard is well above its early values. A team that uses the two-year study to forecast failures among next year's installations will plan for twice the failures that arrive; a team that uses it to argue the fleet is deteriorating will be right about the fleet and wrong about the product. Neither number is the lifetime distribution, and the left-truncated curve, which is, gives the rate at any age directly.

## Where the Same Selection Hides

The fleet is a convenient example because installation dates are recorded and failure is unambiguous, but the selection is the same whenever a dataset is extracted from the units that exist at extraction time.

**Customer tenure.** A churn model trained on current customers, with tenure as a feature, sees the customers who have not churned. The long-tenured ones are survivors twice over, and the model learns that tenure protects against churn more strongly than it does. Left truncation applies with entry at the date the customer record enters the observation window.

**Credit and insurance.** Loan books extracted at a date contain the loans that have not defaulted or paid off. Default rates by age of loan, estimated by treating the book as a cohort from origination, understate early defaults exactly as the pump study overstates early survival.

**Funds, firms and products.** Performance histories of currently listed funds omit the ones that closed, which is the classic form of the bias and the reason survivorship-free databases exist. Company datasets built from the current register omit failed firms, and an analysis of what predicts growth is an analysis of what predicted growth among firms that did not die.

**Sensors and components.** Any fleet of parts with a rising hazard, followed from a maintenance system's go-live date rather than from installation, has the same age-mix and truncation structure as the pumps.

The check is always the same question: could a unit have been excluded from this dataset by the outcome I am studying? If a pump that failed early, a customer who left early or a loan that defaulted early would not be in the table, the answer is yes.

## What to Do

1. **Record the entry time** of every unit into observation, separately from its origin time. The snapshot date, the go-live date or the extraction date is the entry; installation, signup or origination is the origin.
2. **Analyse on the origin time scale with delayed entry.** Kaplan-Meier, Cox regression and parametric survival models all take an entry time; the only change is that a unit joins the risk set at its entry age rather than at zero.
3. **Do not read remaining-life estimates as lifetime estimates.** A study clocked from the snapshot answers a different question; say which one it is answering.
4. **Compare the mix** of the snapshot with the mix of everything ever created, on any variable that could shorten life. A shortfall is the bias made visible.
5. **Forecast new units from an estimate on the origin scale**, never from the rates observed among survivors, whose age distribution the new units do not share.
6. **Go back for the failures.** If the register of retired, churned or defaulted units exists anywhere, adding it turns the survivors into a full cohort and removes the problem at the source.

## References

- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481.
- Tsai, W.-Y., Jewell, N. P., & Wang, M.-C. (1987). A note on the product-limit estimator under right censoring and left truncation. *Biometrika*, 74(4), 883-886.
- Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data* (2nd ed.). Springer.
- Lagakos, S. W., Barraj, L. M., & De Gruttola, V. (1988). Nonparametric analysis of truncated survival data, with application to AIDS. *Biometrika*, 75(3), 515-523.
- Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship bias in performance studies. *The Review of Financial Studies*, 5(4), 553-580.
- Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). Survivor bias and mutual fund performance. *The Review of Financial Studies*, 9(4), 1097-1120.
