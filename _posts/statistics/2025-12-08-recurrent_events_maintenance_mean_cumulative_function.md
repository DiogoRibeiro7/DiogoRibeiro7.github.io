---
permalink: '/statistics/recurrent_events_maintenance_mean_cumulative_function/'
title: 'Recurrent Failures: Why Time to First Failure Throws Away Two Thirds of the Data'
categories:
- Statistics
tags:
- Survival Analysis
- Reliability
- Predictive Maintenance
- Statistics
author_profile: false
seo_title: 'Recurrent Events and the Mean Cumulative Function'
seo_description: 'Machines fail more than once, yet reliability analyses routinely keep only the first failure. A simulated fleet shows what that discards, how the mean cumulative function counts repeated failures with a proper risk set, why a past failure predicts the next one, and which models fit repeated events.'
excerpt: >-
  Four hundred machines, 880 failures over three years. The time-to-first-failure
  analysis uses 299 of them, reports a rate a quarter too low, and cannot
  say that machines which failed last year fail twice as often this year.
summary: >-
  What a recurrent-event process is and why a first-failure analysis
  discards most of it, a simulated fleet with staggered observation, two
  environments and machine-level frailty, the mean cumulative function as
  the recurrent-event analogue of a survival curve and why the naive
  average count goes wrong under censoring, the rate ratio between
  environments from all events, overdispersion and repeat offenders under
  frailty, the predictive value of a past failure, and the models that
  handle repeated events.
keywords:
  - recurrent events
  - mean cumulative function
  - Nelson-Aalen
  - Andersen-Gill
  - frailty
  - reliability
  - predictive maintenance
classes: wide
date: '2025-12-08'
why_this_exists: >-
  Maintenance records are full of machines with several failures each,
  and most analyses reach for a survival curve that stops at the first.
  This post shows on a simulated fleet how much that loses, how to count
  repeated failures correctly, and which questions only the full record
  can answer.
evidence: >-
  A simulated fleet of 400 machines observed for one to three years each,
  with Poisson failure processes at a base rate of 0.8 per year, doubled
  at harsh sites and multiplied by a gamma-distributed machine frailty of
  variance 0.5; 880 failures in 800 machine-years.
methodology: >-
  Compares the first-failure rate with the all-events rate and the true
  fleet average, estimates the mean cumulative function with risk sets
  against the naive average count, computes the harsh-to-normal rate
  ratio from all events and from first failures, measures overdispersion
  and the share of failures from the top decile of machines with and
  without frailty, and tests whether a year-one failure predicts year-two
  failures.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-terrain.jpg
  og_image: /assets/images/headers/photo-terrain.jpg
  overlay_image: /assets/images/headers/photo-terrain.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-terrain.jpg
  twitter_image: /assets/images/headers/photo-terrain.jpg
---
The maintenance system holds every failure of every pump for three years: 880 events across 400 machines. The reliability report fits a survival curve to time until first failure, because that is what the survival library does by default, and concludes that pumps fail at 0.80 per machine-year. The true rate is 1.09. The report used 299 of the 880 events, and the 581 it discarded are not a random subset. They are the second, third and fourth failures of the machines that fail most, which is to say the events the maintenance budget is actually spent on.

A pump that has failed and been repaired is still a pump, still in service and still at risk. Treating the first failure as the end of the story removes it from the analysis at exactly the moment it becomes most informative. The methods for repeated events are not exotic; they are the survival methods with the risk set defined correctly, and they answer questions a first-failure analysis cannot ask.

## A Simulated Fleet

Four hundred machines, 60 percent at normal sites and 40 percent at harsh ones, where the failure rate is doubled. Each machine also carries a frailty, a multiplier on its rate drawn from a gamma distribution with mean one and variance 0.5, so that some machines are chronic offenders for reasons the site does not explain. Failures follow a Poisson process at the machine's rate, repairs are instantaneous, and machines are observed for between one and three years, because they were installed at different times.

```python
import numpy as np

rng = np.random.default_rng(0)
n_machines, base_rate, harsh_ratio, frailty_var, followup = 400, 0.8, 2.0, 0.5, 3.0

env = rng.choice(["normal", "harsh"], n_machines, p=[0.6, 0.4])
frailty = rng.gamma(1 / frailty_var, frailty_var, n_machines)        # mean 1, variance 0.5
rate = base_rate * np.where(env == "harsh", harsh_ratio, 1.0) * frailty
observed = rng.uniform(1.0, followup, n_machines)                     # staggered installation

events = []                                                           # (machine, age at failure)
for i in range(n_machines):
    t = 0.0
    while True:
        t += rng.exponential(1 / rate[i])
        if t > observed[i]:
            break
        events.append((i, t))
events = np.array(events)
machine_years = observed.sum()
print(f"{len(events)} failures in {machine_years:.0f} machine-years; "
      f"{np.mean([np.sum(events[:, 0] == i) == 0 for i in range(n_machines)]):.0%} never failed")
```

The fleet accumulates 800 machine-years and 880 failures; a quarter of the machines never fail during their observation. The true fleet-average rate, known here because the simulation set it, is 1.09 failures per machine-year.

## What the First Failure Leaves Out

The first-failure analysis keeps, for each machine, the age at its first failure or the age at which observation ended, and one indicator.

```python
first = np.full(n_machines, np.inf)
for i, t in events:
    first[int(i)] = min(first[int(i)], t)
fail_first = np.isfinite(first)
t_first = np.where(fail_first, first, observed)
print(f"events used: {fail_first.sum()} of {len(events)}")
print(f"first-failure rate {fail_first.sum() / t_first.sum():.2f}/yr; all-events rate {len(events) / machine_years:.2f}/yr; "
      f"true {rate.mean():.2f}/yr")
```

| Quantity | Value |
| --- | --- |
| Failures in the record | 880 |
| Failures used by the first-failure analysis | 299 (34%) |
| Rate from first failures only | 0.80 per machine-year |
| Rate from all failures | 1.10 per machine-year |
| True fleet average | 1.09 per machine-year |

The first-failure rate is low for a reason beyond the discarded events. Under frailty, the machines that fail early are disproportionately the high-rate ones, and after their first failure they leave the first-failure risk set. What remains is a progressively more robust set of machines, and the hazard estimated from it drifts down over age. The all-events rate keeps every machine at risk for as long as it is observed, which is what the fleet actually experiences.

## The Mean Cumulative Function

The recurrent-event counterpart of a survival curve is the mean cumulative function: the expected number of failures a machine has accumulated by age $t$. Its estimator is the Nelson-Aalen sum, an increment of $1/n(a)$ at every failure age $a$, where $n(a)$ is the number of machines still under observation at that age. Machines contribute to the risk set for as long as they are observed, however many times they have failed.

```python
def mcf(event_times, obs, grid):
    """Expected cumulative failures per machine by age: sum of 1 / (machines observed at that age)."""
    out = []
    for g in grid:
        inc = sum(1.0 / np.sum(obs >= a) for a in event_times[event_times <= g])
        out.append(inc)
    return np.array(out)

grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
m_all = mcf(events[:, 1], observed, grid)
naive = np.array([np.mean([np.sum((events[:, 0] == i) & (events[:, 1] <= g)) for i in range(n_machines)]) for g in grid])
print("MCF:  ", m_all.round(2))
print("true: ", (rate.mean() * grid).round(2))
print("naive:", naive.round(2))
```

| Age (years) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- |
| Mean cumulative function | 0.56 | 1.11 | 1.59 | 2.21 | 2.73 | 3.29 |
| True (rate × age) | 0.54 | 1.09 | 1.63 | 2.18 | 2.72 | 3.26 |
| Naive average count per machine | 0.56 | 1.11 | 1.52 | 1.90 | 2.10 | 2.20 |

The mean cumulative function tracks the truth at every age. The naive average, the mean over all 400 machines of failures observed by that age, agrees for the first year, while every machine is still observed, and then flattens, because machines whose observation ended after 1.5 or 2 years contribute zeros to ages they never reached. By three years the naive curve reports 2.2 failures per machine against a true 3.3, an error that would make a three-year maintenance budget a third too small. The risk set is the whole difference between the two estimators, and it is the same idea that makes Kaplan-Meier work for a single event.

![Mean cumulative failures per machine against age for the simulated fleet, estimated with risk sets for all machines and for harsh and normal sites, with the naive average count for comparison. The naive curve flattens after two years as observation ends for many machines; the proper estimates climb at the true rates.](/assets/images/figures/recurrent_events_mcf.png){: width="1152" height="672" loading="lazy"}

## Comparing Environments

With all events, the rate at each kind of site is failures divided by machine-years observed, and the ratio between sites is a direct estimate of the environment's effect.

```python
for e in ("normal", "harsh"):
    m = env == e
    ev = events[np.isin(events[:, 0], np.where(m)[0])]
    print(f"{e}: all-events rate {len(ev)/observed[m].sum():.2f}/yr, first-failure rate "
          f"{fail_first[m].sum()/t_first[m].sum():.2f}/yr, true {rate[m].mean():.2f}/yr")
```

| Site | All-events rate | First-failure rate | True mean rate |
| --- | --- | --- | --- |
| Normal | 0.75 per year | 0.60 per year | 0.77 per year |
| Harsh | 1.55 per year | 1.18 per year | 1.50 per year |
| Ratio harsh / normal | 2.06 | 1.97 | 2.00 |

Both analyses get the ratio about right here, because the frailty affects both sites alike and cancels in the ratio, but only the all-events rates are the right magnitude. A budget built on the first-failure rates would be a quarter short at both kinds of site. In a regression with several covariates, the all-events analysis is the Andersen-Gill model, a Cox model in which each machine re-enters the risk set after every failure, and its coefficients are log rate ratios exactly like these.

## Repeat Offenders

Frailty is the reliability engineer's name for the fact that machines of the same type at the same site do not fail at the same rate. Its signature is overdispersion, a variance of failure counts larger than a Poisson process would give, and its practical consequence is that failures concentrate on a minority of machines.

```python
counts = np.array([np.sum(events[:, 0] == i) for i in range(n_machines)])
print(f"counts: mean {counts.mean():.2f}, variance {counts.var():.2f}")
top = np.argsort(-counts)[: n_machines // 10]
print(f"top 10% of machines: {counts[top].sum() / counts.sum():.0%} of failures")
pois = np.random.default_rng(1).poisson(base_rate * np.where(env == "harsh", harsh_ratio, 1.0) * observed)
print(f"without frailty: {np.sort(pois)[::-1][: n_machines // 10].sum() / pois.sum():.0%}")
```

The counts have mean 2.2 and variance 5.7, more than twice what a Poisson process would give. The top tenth of machines accounts for 34 percent of failures; in a fleet with the same site mix and observation times but no frailty, the top tenth would account for 29 percent, the share that pure chance plus the site effect produces. The excess is the chronic offenders, and identifying them is a maintenance decision worth money, which the first-failure analysis cannot support because it does not know how many times anything failed.

The most useful consequence is that the past predicts the future within a machine.

```python
had_year1 = np.array([np.sum((events[:, 0] == i) & (events[:, 1] <= 1.0)) > 0 for i in range(n_machines)])
obs2 = observed >= 2.0
for label, m in (("failed in year 1", had_year1 & obs2), ("no failure in year 1", ~had_year1 & obs2)):
    c2 = [np.sum((events[:, 0] == i) & (events[:, 1] > 1.0) & (events[:, 1] <= 2.0)) for i in np.where(m)[0]]
    print(f"{label}: {np.mean(c2):.2f} failures in year 2 per machine (n = {m.sum()})")
```

| Machines observed at least two years | Failures in year two, per machine |
| --- | --- |
| Failed at least once in year one (107 machines) | 1.46 |
| No failure in year one (92 machines) | 0.68 |

Machines that failed in their first year fail more than twice as often in their second. Under a pure Poisson process with no frailty, the year-one history would carry no information at all, since the rate would be the same for every machine at a site. The gap is a direct measurement of how much of the fleet's variation is between machines rather than within them, and it is the empirical justification for prioritising inspections by failure history.

## Which Model

The choice depends on the question, and the questions are different from the single-event case.

**How many failures will the fleet produce?** The mean cumulative function, with confidence bands from the Nelson-Aalen variance, or a Poisson regression of counts on covariates with machine-years as the exposure. Overdispersion calls for a negative binomial or a quasi-Poisson variance; the gamma frailty that generated this fleet makes the counts exactly negative binomial.

**What changes the rate?** The Andersen-Gill model, which treats each failure as an event with the machine back at risk immediately after, with robust standard errors clustered by machine to absorb the frailty. The Prentice-Williams-Peterson model stratifies by the number of previous failures when the rate itself changes after each event, as it does when repairs are imperfect.

**Which machines are the offenders?** A shared frailty model, a Cox or Poisson model with a random effect per machine, gives each machine a posterior rate multiplier that ranks it against its peers after accounting for its site and exposure time. The ranking is what the maintenance planner wants, and the year-one versus year-two table is its crude version.

**When does the process change?** If repairs restore the machine to new, the times between failures are independent and a renewal model applies; if they restore it to the state just before failure, the Poisson process applies; if they degrade it, the rate rises with each event. The mean cumulative function's shape distinguishes the cases: linear for a stationary rate, convex for a deteriorating fleet, concave for one being improved by its repairs.

## What to Do

1. **Keep every failure.** Structure the record as one row per failure with machine, age at the event and the observation window, not one row per machine.
2. **Estimate the mean cumulative function** with risk sets, and read the fleet's expected failures per machine at any age from it. Do not average raw counts across machines with different observation lengths.
3. **Compare groups by rate ratio** from all events, with machine-years as the exposure, and fit Andersen-Gill or Poisson regression when there are several covariates.
4. **Check for overdispersion** by comparing the variance of counts with the mean, and use a negative binomial or robust standard errors when it is present.
5. **Rank machines by frailty** and use failure history as the first inspection criterion; a year-one failure roughly doubles the year-two rate.
6. **Reserve first-failure analysis for the questions it answers**: warranty exposure, infant mortality and the time to the first intervention, none of which is the fleet's failure burden.

## References

- Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for counting processes: a large sample study. *The Annals of Statistics*, 10(4), 1100-1120.
- Nelson, W. B. (2003). *Recurrent Events Data Analysis for Product Repairs, Disease Recurrences, and Other Applications*. SIAM.
- Cook, R. J., & Lawless, J. F. (2007). *The Statistical Analysis of Recurrent Events*. Springer.
- Lawless, J. F., & Nadeau, C. (1995). Some simple robust methods for the analysis of recurrent events. *Technometrics*, 37(2), 158-168.
- Prentice, R. L., Williams, B. J., & Peterson, A. V. (1981). On the regression analysis of multivariate failure time data. *Biometrika*, 68(2), 373-379.
- Amorim, L. D. A. F., & Cai, J. (2015). Modelling recurrent events: a tutorial for analysis in epidemiology. *International Journal of Epidemiology*, 44(1), 324-333.
