---
permalink: '/statistics/novelty_primacy_effects_experiment_duration/'
title: 'Novelty and Primacy: When the Effect You Measure Depends on How Long You Looked'
categories:
- Statistics
tags:
- A/B Testing
- Experimental Design
- Time Series
- Statistics
author_profile: false
seo_title: 'Novelty and Primacy Effects in A/B Tests'
seo_description: 'A treatment whose effect fades as users get used to it reports 9 percent after three days and 3.9 percent after four weeks, against a long-run truth of 1 percent. A simulation shows why the day-by-day curve hides the decay and how grouping by user tenure recovers it.'
excerpt: >-
  The test reports a 9 percent lift after three days, 5.7 after two weeks
  and 3.9 after four. None of those is the answer. The effect on a user
  who has lived with the change for a month is 1 percent, and no amount
  of staring at the daily chart reveals it.
summary: >-
  Why an effect that depends on how long a user has been exposed makes the
  length of the test part of the result, a simulation of novelty and
  primacy patterns with users entering continuously, why the calendar-day
  curve is flatter and later than the true decay, how grouping by user
  tenure or following a single cohort recovers it, what each stopping
  rule reports against the long-run truth, and how to decide when a test
  has run long enough.
keywords:
  - novelty effect
  - primacy effect
  - A/B testing
  - experiment duration
  - time-varying treatment effect
  - cohort analysis
  - user tenure
classes: wide
date: '2025-10-07'
why_this_exists: >-
  Teams debate test duration as though it were only a question of
  statistical power, when the deeper problem is that the quantity being
  estimated changes with the window. This post separates the two and
  shows the analysis that reports a stable effect rather than a longer
  average of an unstable one.
evidence: >-
  A simulated experiment with 4,000 users entering each day over 28 days,
  each returning on half the days after entry, an outcome of 20 units per
  active user with a standard deviation of 10, and treatment effects that
  are constant, decay from 10 to 1 percent, or rise from minus 4 to 5
  percent with exposure; headline figures averaged over 20 runs.
methodology: >-
  Compares the effect estimated by grouping users by tenure, by calendar
  day and cumulatively over the elapsed test, against the known
  tenure-dependent truth; repeats the tenure estimate on a single
  first-day cohort; and reports what each stopping day would have
  concluded.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-stars.jpg
  og_image: /assets/images/headers/photo-stars.jpg
  overlay_image: /assets/images/headers/photo-stars.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-stars.jpg
  twitter_image: /assets/images/headers/photo-stars.jpg
---
The redesign went live to half the users on a Monday. By Wednesday the dashboard showed a 9 percent lift and the team started drafting the launch note. Two weeks in it was 5.7 percent, which someone attributed to the sample settling down. At four weeks it was 3.9 percent and the test was called a success.

The effect of the redesign on a user who has lived with it for a month is 1 percent. Every number the team saw was correct as a description of its own window and wrong as an answer to the question they were asking. The effect was never a single quantity; it was a curve, high on a user's first day with the change and decaying as the novelty wore off, and averaging a decaying curve over a window tells you about the window.

## The Effect Is a Curve, Not a Number

A treatment effect that depends on how long a user has been exposed comes in two familiar shapes. **Novelty**: the change is new and interesting, people click it because it is different, and the excess fades. **Primacy**: the change disrupts a learned habit, people are worse off while they relearn, and the benefit arrives later. Both make the measured effect a function of the test's length, and they push in opposite directions, so the same two-week convention that overstates one understates the other.

The complication that makes this hard to see is that users do not all start together. New users enter the experiment every day, so on any given calendar day the population is a mixture: some on their first exposed day, some on their twentieth. As the test runs, the mixture shifts toward longer tenures. The daily chart is a moving average over a moving mixture, which is why it neither shows the decay clearly nor settles at the long-run value.

## A Simulation

Four thousand users enter each day for 28 days and remain, active on about half the days after they join. The outcome is a per-user-day quantity with a mean of 20 and a standard deviation of 10. Three effect patterns are simulated: constant, novelty decaying from 10 percent to 1, and primacy rising from minus 4 percent to 5.

```python
import numpy as np

rng = np.random.default_rng(0)
NEW_PER_DAY, RETURN_P, BASE, SD = 4000, 0.5, 20.0, 10.0

def tenure_effect(k, kind):
    """Relative lift for a user on their k-th day of exposure (k = 0 is the first day)."""
    if kind == "constant":
        return 0.03
    if kind == "novelty":                       # starts high, decays to a modest long-run effect
        return 0.01 + 0.09 * np.exp(-k / 4.0)
    if kind == "primacy":                       # starts negative, users learn, settles positive
        return 0.05 - 0.09 * np.exp(-k / 6.0)

def run(days, kind, r=rng):
    """Users enter continuously and stay. Returns lift by calendar day, by tenure, and cumulative."""
    day_t = np.zeros(days); day_c = np.zeros(days)
    day_nt = np.zeros(days); day_nc = np.zeros(days)
    ten_t = np.zeros(days); ten_c = np.zeros(days)
    ten_nt = np.zeros(days); ten_nc = np.zeros(days)
    for entry in range(days):                    # the cohort first exposed on this day
        n = NEW_PER_DAY
        z = r.integers(0, 2, n)                  # arm, fixed for the user's life
        for d in range(entry, days):
            k = d - entry
            active = np.ones(n, bool) if k == 0 else (r.random(n) < RETURN_P)
            za = z[active]
            y = BASE * (1 + tenure_effect(k, kind) * za) + r.normal(0, SD, active.sum())
            day_t[d] += y[za == 1].sum(); day_nt[d] += (za == 1).sum()
            day_c[d] += y[za == 0].sum(); day_nc[d] += (za == 0).sum()
            ten_t[k] += y[za == 1].sum(); ten_nt[k] += (za == 1).sum()
            ten_c[k] += y[za == 0].sum(); ten_nc[k] += (za == 0).sum()
    by_day = (day_t / day_nt) / (day_c / day_nc) - 1
    by_tenure = (ten_t / ten_nt) / (ten_c / ten_nc) - 1
    cum = (np.cumsum(day_t) / np.cumsum(day_nt)) / (np.cumsum(day_c) / np.cumsum(day_nc)) - 1
    return by_day, by_tenure, cum

by_day = np.mean([run(28, "novelty")[0] for _ in range(20)], axis=0)
by_ten = np.mean([run(28, "novelty")[1] for _ in range(20)], axis=0)
for d in (0, 3, 7, 13, 20, 27):
    print(f"day {d + 1:>2}: by tenure {by_ten[d]:>6.1%}, by calendar {by_day[d]:>6.1%}, "
          f"true at that tenure {tenure_effect(d, 'novelty'):>6.1%}")
```

**Grouping by tenure recovers the truth; grouping by calendar day does not.** For the novelty pattern:

| Day | Estimated, by tenure | Estimated, by calendar day | True effect at that tenure |
| --- | --- | --- | --- |
| 1 | 9.9% | 9.9% | 10.0% |
| 4 | 5.1% | 7.9% | 5.3% |
| 8 | 2.7% | 5.9% | 2.6% |
| 14 | 1.6% | 4.3% | 1.3% |
| 21 | 1.0% | 3.3% | 1.1% |
| 28 | 1.1% | 2.7% | 1.0% |

The tenure column tracks the true decay closely at every point. The calendar column agrees on day 1, when everyone is on their first exposed day and the two groupings coincide, and then separates: at two weeks it reports 4.3 percent where the current effect on a two-week user is 1.3. The calendar curve is flatter because each day averages fresh users at 10 percent with older users at 1, and it is later because the share of older users only grows slowly.

![Estimated effect by day for the novelty pattern: the true tenure curve, the estimate grouped by tenure, and the estimate grouped by calendar day. The tenure estimate tracks the truth; the calendar curve is flatter and lags behind it.](/assets/images/figures/novelty_tenure_vs_calendar.png){: width="1152" height="672" loading="lazy"}

## What Each Stopping Rule Reports

The number a team actually quotes is the cumulative estimate over everything collected so far, which is the quantity that keeps drifting.

```python
res = {k: np.mean([run(28, k)[2] for _ in range(20)], axis=0) for k in ("constant", "novelty", "primacy")}
for label, d in (("3 days", 2), ("7 days", 6), ("14 days", 13), ("21 days", 20), ("28 days", 27)):
    print(f"{label:>10}" + "".join(f"{res[k][d]:>10.1%}" for k in ("constant", "novelty", "primacy")))
```

| Test stops after | Constant effect | Novelty pattern | Primacy pattern |
| --- | --- | --- | --- |
| 3 days | 2.8% | 9.2% | -3.3% |
| 7 days | 2.8% | 7.5% | -2.0% |
| 14 days | 3.0% | 5.7% | -0.5% |
| 21 days | 3.0% | 4.6% | 0.6% |
| 28 days | 3.0% | 3.9% | 1.4% |
| Long-run truth | 3.0% | 1.0% | 5.0% |

The constant column is the reassuring one: when the effect does not depend on tenure, the test length affects only the precision, and every stopping rule gives the same answer. The other two columns are the warning. The novelty treatment is reported at nine times its long-run value after three days and is still four times it after four weeks. The primacy treatment looks harmful for the first two weeks and is still reported at a quarter of its true benefit at four weeks; a team with a two-week convention and a "do no harm" rule kills it.

Neither estimate comes within 20 percent of the long-run effect inside 28 days. Running longer helps, but only slowly, because the cumulative average keeps carrying every early day at full weight. That is the argument for changing the analysis rather than extending the calendar.

## Two Analyses That Answer the Question

**Group by tenure.** Every observation already carries the user's days-since-first-exposure; aggregating on that instead of on the date gives the curve directly, as the first table shows. The estimate at the largest tenure available is the best available reading of the long-run effect, and its shape says whether the effect has settled.

**Follow one cohort.** Restricting to users who entered on the first day and tracking them over the test does the same thing without any aggregation subtlety.

```python
def first_cohort(days, kind, n=200_000, r=rng):
    z = r.integers(0, 2, n)
    out = []
    for k in range(days):
        active = np.ones(n, bool) if k == 0 else (r.random(n) < RETURN_P)
        za = z[active]
        y = BASE * (1 + tenure_effect(k, kind) * za) + r.normal(0, SD, active.sum())
        out.append(y[za == 1].mean() / y[za == 0].mean() - 1)
    return np.array(out)

fc = first_cohort(15, "novelty")
for k in range(0, 15, 2):
    print(f"day {k + 1:>2}: single cohort {fc[k]:>6.1%}, true {tenure_effect(k, 'novelty'):>6.1%}")
```

| Day | Single-cohort estimate | True effect |
| --- | --- | --- |
| 1 | 9.8% | 10.0% |
| 3 | 6.8% | 6.5% |
| 5 | 4.1% | 4.3% |
| 7 | 3.2% | 3.0% |
| 9 | 2.1% | 2.2% |
| 11 | 1.7% | 1.7% |
| 13 | 1.7% | 1.4% |
| 15 | 0.7% | 1.3% |

The cohort approach costs sample size, since it uses only the users who arrived first, and it buys a clean reading of how the effect evolves. In practice both are worth computing: the tenure aggregation for precision and the first cohort as a check that nothing about the first day's users was unusual.

## Telling Novelty From a Real Effect

The decay itself is not proof of novelty. Three other things produce a falling daily curve, and each has a different remedy.

**Mix shift.** If the kind of user entering the experiment changes over the test, a marketing push, a seasonal wave, a ramp in exposure, the daily estimate moves because the population moved. Grouping by tenure does not fix this; comparing the composition of the arms over time does.

**A genuinely time-varying effect.** A feature whose value depends on the day of the week, or on a promotion running in week one, has an effect that changes for reasons that have nothing to do with user tenure. The tenure curve would be flat while the calendar curve moves, which is the opposite pattern to novelty and is diagnostic.

**Survivorship among returners.** If the treatment changes who comes back, the users observed at high tenure differ between arms, and the tenure curve is comparing different populations. The check is whether return rates match across arms at each tenure; if they do not, the experiment has an attrition problem and the tenure analysis inherits it.

A useful discipline is to write down, before the test, what the effect is expected to do over time and why. A change to a habitual flow should be expected to show primacy; a visually striking change should be expected to show novelty; a backend latency improvement should show neither. A curve that contradicts the prediction is worth understanding before the result is used.

## How Long Is Long Enough

The right stopping rule is not a fixed number of weeks but a statement about the tenure curve: run until the effect at the largest observed tenure has been flat for long enough to believe it, and report that value as the long-run effect with the curve beside it. In the simulation the novelty effect is within half a point of its long-run value by tenure day 14, so a test that ran four weeks could have read the settled value from the tenure curve rather than quoting a cumulative average that was still falling.

Where that is impractical, the honest fallback is to report the effect as a range with its window attached: "5.7 percent measured over two weeks, still falling, long-run value not yet identified." That is less satisfying than a single number and considerably more useful than one that will not survive the launch.

## What to Do

1. **Record days since first exposure** on every observation, and make tenure a standard dimension of experiment analysis alongside date.
2. **Plot the effect by tenure, not by calendar day.** The calendar curve is a moving average over a shifting mixture and hides both novelty and primacy.
3. **Read the long-run effect from the largest available tenure**, not from the cumulative average, which carries every early day at full weight.
4. **Predict the time shape before the test.** A habit-disrupting change should be expected to show primacy; a striking one, novelty; a silent backend change, neither.
5. **Check the arms' composition and return rates by tenure** before trusting the curve, since mix shift and differential attrition produce similar-looking decay.
6. **Report the window with the number** whenever the effect has not settled, and say which direction it is still moving.

## References

- Kohavi, R., Deng, A., Frasca, B., Longbotham, R., Walker, T., & Xu, Y. (2012). Trustworthy online controlled experiments: five puzzling outcomes explained. *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 786-794.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Hohnhold, H., O'Brien, D., & Tang, D. (2015). Focusing on the long-term: it's good for users and business. *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1849-1858.
- Chen, N., Liu, M., & Xu, Y. (2019). How A/B tests could go wrong: automatic diagnosis of invalid online experiments. *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining*, 501-509.
- Dmitriev, P., Gupta, S., Kim, D. W., & Vaz, G. (2017). A dirty dozen: twelve common metric interpretation pitfalls in online controlled experiments. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1427-1436.
- Deng, A., Hill, D., Frasca, B., & Crook, T. (2016). Diluted treatment effect estimation for trigger analysis in online controlled experiments. *Proceedings of the Ninth ACM International Conference on Web Search and Data Mining*, 349-358.
