---
permalink: '/statistics/staggered_difference_in_differences_two_way_fixed_effects/'
title: 'Staggered Rollouts and Difference-in-Differences: When Two-Way Fixed Effects Get It Wrong'
categories:
- Statistics
tags:
- Causal Inference
- Econometrics
- Panel Data
- Statistics
author_profile: false
seo_title: 'Staggered Difference-in-Differences and Two-Way Fixed Effects'
seo_description: 'When units adopt a change at different times and the effect varies, the standard two-way fixed effects regression averages comparisons that use already-treated units as controls. A simulation shows it reporting half the true effect, and the group-time estimator that gets it right.'
excerpt: >-
  A feature rolls out to regions in three waves. The panel regression with
  region and month fixed effects says the effect is 0.6. The true average
  effect on the treated regions is 1.3, and every one of the estimator's
  ingredients is correct except the comparison it makes for the last wave.
summary: >-
  What difference-in-differences assumes and what a staggered rollout adds
  to it, why the two-way fixed effects coefficient is a weighted average of
  two-by-two comparisons that includes already-treated units as controls,
  a simulated rollout in three waves under four patterns of treatment
  effect showing when the regression is right and how far it goes wrong,
  the group-time estimator that only compares against not-yet-treated
  units, an event study that recovers the dynamic effect, and how to
  report a rollout.
keywords:
  - difference-in-differences
  - staggered adoption
  - two-way fixed effects
  - heterogeneous treatment effects
  - event study
  - group-time average treatment effect
  - panel data
classes: wide
date: '2026-09-07'
why_this_exists: >-
  Most product, policy and process changes reach units in waves rather than
  all at once, and the default analysis is a panel regression with unit
  and time fixed effects. This post shows on a simulated rollout that the
  default is right only when the effect is the same for everyone at all
  times, measures its error when it is not, and gives the estimator that
  replaces it.
evidence: >-
  A simulated panel of 60 units over 20 periods, with unit and period
  effects, 15 units adopting in each of three waves and 15 never adopting,
  under four treatment-effect patterns: constant, growing with exposure,
  larger for early adopters, and growing then fading; 500 replications
  each.
methodology: >-
  Compares the two-way fixed effects coefficient and a group-time
  estimator against the true average effect on the treated in each
  scenario, decomposes one noise-free panel into the two-by-two comparisons
  the regression averages, and aggregates group-time effects by exposure
  into an event study checked against the true dynamic effect.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/blocks.jpg
  og_image: /assets/images/headers/blocks.jpg
  overlay_image: /assets/images/headers/blocks.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/blocks.jpg
  twitter_image: /assets/images/headers/blocks.jpg
---
The new routing logic went to fifteen regions in month five, fifteen more in month ten, fifteen more in month fifteen, and fifteen regions are still waiting. The analyst has twenty months of a per-region outcome, fits a regression with region fixed effects, month fixed effects and a dummy for "routing logic active", and reads off the coefficient: 0.6. The team had hoped for one.

The true average effect on the treated regions in the simulation behind that number is 1.27. The regression is not misspecified in the usual sense: the region and month effects are real and it estimates them fine. The problem is what the treatment dummy compares. With units adopting at different times, the two-way fixed effects estimator is a weighted average of every two-by-two difference-in-differences the panel can form, and some of those comparisons use regions that already have the routing logic as the control group for regions that are just getting it. When the effect grows with time on the new logic, the early adopters' outcomes are still rising during the late adopters' post-period, and the difference-in-differences subtracts that rise from the late adopters' effect.

## Two Periods, Two Groups

With one treated group and one control group, before and after, difference-in-differences is a subtraction: the change in the treated group minus the change in the control group. The parallel-trends assumption says the control group's change is what the treated group's change would have been without treatment, and the estimate is the average effect on the treated. Fixed effects for unit and period implement exactly this subtraction, and with two groups and two periods the regression coefficient is the two-by-two difference.

A staggered rollout has several treated groups with different adoption dates. The regression still has one treatment dummy, and Goodman-Bacon showed what its coefficient is: a weighted average of all the two-by-two comparisons between pairs of groups, over the windows in which one is treated and the other is not. Some of those pairs are a treated cohort against the never-treated units, which are clean. Some are a late cohort against an early cohort in the window after the early cohort adopted, in which the early cohort's outcome includes its own treatment effect. If that effect is constant, it differences out. If it changes over time, it does not, and the late cohort's estimated effect is biased by the early cohort's dynamics.

## A Simulated Rollout

Sixty units over twenty periods, with a unit effect, a rising period effect and unit noise. Fifteen units adopt at period 5, fifteen at 10, fifteen at 15, and fifteen never. Four patterns of treatment effect are simulated, and for each the two-way fixed effects estimator is compared with a group-time estimator that only ever compares a cohort against units that have not yet adopted.

```python
import numpy as np

rng = np.random.default_rng(0)
n_units, n_periods = 60, 20
cohorts = {5: 15, 10: 15, 15: 15, None: 15}      # adoption period -> units (None = never treated)

def simulate(effect_fn, r=rng, sigma=1.0):
    g_of = np.concatenate([[g] * n for g, n in cohorts.items()]).astype(object)
    unit_fe = r.normal(0, 1, n_units)
    time_fe = np.linspace(0, 2, n_periods) + r.normal(0, 0.2, n_periods)
    Y = np.empty((n_units, n_periods)); D = np.zeros((n_units, n_periods)); tau = np.zeros((n_units, n_periods))
    for i in range(n_units):
        for t in range(n_periods):
            g = g_of[i]
            treated = g is not None and t >= g
            D[i, t] = treated
            tau[i, t] = effect_fn(t - g, g) if treated else 0.0
            Y[i, t] = unit_fe[i] + time_fe[t] + tau[i, t] + r.normal(0, sigma)
    return Y, D, tau, g_of

def twfe(Y, D):
    """Y on D with unit and period fixed effects, by the within transformation."""
    Yd = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    Dd = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
    return (Dd * Yd).sum() / (Dd ** 2).sum()

def group_time(Y, g_of, control="never"):
    """ATT(g, t) for each cohort g and post period t against never-treated or not-yet-treated
    units, with the period before adoption as the base; averaged over treated cells."""
    atts, weights = [], []
    for g in [g for g in set(g_of) if g is not None]:
        tr = np.array([x == g for x in g_of])
        for t in range(g, n_periods):
            co = np.array([x is None for x in g_of]) if control == "never" \
                 else np.array([x is None or (x > t) for x in g_of])
            att = (Y[tr, t] - Y[tr, g - 1]).mean() - (Y[co, t] - Y[co, g - 1]).mean()
            atts.append(att); weights.append(tr.sum())
    return np.average(atts, weights=weights)

def true_att(tau, D):
    return tau[D == 1].mean()

scenarios = {
    "constant effect 1.0":             lambda k, g: 1.0,
    "effect grows with exposure":      lambda k, g: 0.2 * (k + 1),
    "early adopters gain more":        lambda k, g: {5: 2.0, 10: 1.0, 15: 0.5}[g],
    "grows with exposure, fades late": lambda k, g: 0.3 * (k + 1) if k < 8 else 0.3 * 8 - 0.6 * (k - 7),
}
reps = 500
for name, fn in scenarios.items():
    est = np.array([(true_att(tau, D), twfe(Y, D), group_time(Y, g_of, "notyet"))
                    for Y, D, tau, g_of in (simulate(fn) for _ in range(reps))])
    print(f"{name:34} true {est[:, 0].mean():.2f}  TWFE {est[:, 1].mean():.2f}  group-time {est[:, 2].mean():.2f}")
```

| Treatment effect | True average effect on the treated | Two-way fixed effects | Group-time estimator |
| --- | --- | --- | --- |
| Constant, 1.0 for everyone | 1.00 | 0.98 | 0.99 |
| Grows with exposure: 0.2 per period | 1.27 | 0.60 | 1.26 |
| Early adopters gain more: 2.0, 1.0, 0.5 by wave | 1.42 | 1.15 | 1.41 |
| Grows for eight periods, then fades | 0.97 | 1.24 | 0.97 |

The first row is the case the regression was designed for, and it is fine. In the second, the effect that grows with exposure, the regression reports less than half the true average. In the third, where waves differ in how much they gain, it is a fifth low. In the fourth, where the effect fades, it is a quarter high: the direction of the bias depends on the shape of the dynamics, not just their presence, so there is no rule of thumb for which way the regression errs. The group-time estimator is within a hundredth of the truth in every row, from the same data.

The single-draw version of the second scenario is the story in the opening.

```python
Y, D, tau, g_of = simulate(scenarios["effect grows with exposure"], np.random.default_rng(1))
print(f"true ATT {true_att(tau, D):.2f}; TWFE {twfe(Y, D):.2f}; "
      f"group-time, never-treated controls {group_time(Y, g_of, 'never'):.2f}; not-yet-treated {group_time(Y, g_of, 'notyet'):.2f}")
```

True effect 1.27, regression 0.39, group-time estimator 1.14 with never-treated controls and 1.13 with not-yet-treated controls, the last two differing from the truth only by the noise of one panel.

## What the Regression Averages

The mechanism is visible in a noise-free panel of the growing-effect scenario, by computing two of the two-by-two comparisons the regression combines.

```python
Y0, D0, tau0, g0 = simulate(scenarios["effect grows with exposure"], np.random.default_rng(2), sigma=0.0)

def did_2x2(Y, g_of, treated_g, control_g, pre, post):
    tr = np.array([x == treated_g for x in g_of]); co = np.array([x == control_g for x in g_of])
    return (Y[tr, post].mean() - Y[tr, pre].mean()) - (Y[co, post].mean() - Y[co, pre].mean())

print(f"{did_2x2(Y0, g0, 10, None, 9, 14):.2f}")   # cohort 10 vs never treated, before and five periods after
print(f"{did_2x2(Y0, g0, 15, 5, 14, 19):.2f}")     # cohort 15 vs cohort 5, which adopted ten periods earlier
```

Cohort 10 against the never-treated units, from period 9 to period 14, gives 1.00, which is the true effect of five periods of exposure. Cohort 15 against cohort 5, from period 14 to period 19, gives 0.00. Cohort 15's outcome rose by its treatment effect of 1.0 plus the common trend; cohort 5's outcome rose by the common trend plus another 1.0, because its own effect was still growing from ten periods of exposure to fifteen. The subtraction cancels the late cohort's entire effect. The regression's coefficient is a weighted average of comparisons like these, and the weights, which depend on cohort sizes and the timing of adoption, put substantial weight on the contaminated ones whenever the middle of the panel is where the treatment variance is.

The word for the second comparison is "forbidden", and the reason it is forbidden is not that early adopters are a bad control group in general. It is that their outcome after adoption contains the quantity being estimated.

## The Estimator That Compares Correctly

Callaway and Sant'Anna's group-time approach builds the estimate from its clean pieces. For each adoption cohort $g$ and each post-adoption period $t$, the effect $ATT(g, t)$ is the change in the cohort's outcome from the period before adoption to $t$, minus the same change among units that are not yet treated at $t$ (or never treated). Each of those is a two-by-two comparison with a valid control group. The pieces are then averaged: over all treated cells for an overall effect, or by exposure $t - g$ for an event study.

```python
ks = range(0, 10)
ev = np.zeros(len(ks))
for _ in range(200):
    Y, D, tau, g_of = simulate(scenarios["effect grows with exposure"])
    for k in ks:
        vals = []
        for g in (5, 10, 15):
            t = g + k
            if t >= n_periods: continue
            tr = np.array([x == g for x in g_of]); co = np.array([x is None or (x > t) for x in g_of])
            vals.append((Y[tr, t] - Y[tr, g - 1]).mean() - (Y[co, t] - Y[co, g - 1]).mean())
        ev[k] += np.mean(vals) / 200
print("true     :", " ".join(f"{0.2*(k+1):.2f}" for k in ks))
print("estimated:", " ".join(f"{v:.2f}" for v in ev))
```

| Periods since adoption | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True effect | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 | 1.20 | 1.40 | 1.60 | 1.80 | 2.00 |
| Group-time estimate | 0.19 | 0.39 | 0.59 | 0.79 | 0.97 | 1.18 | 1.40 | 1.60 | 1.79 | 1.99 |

The event study recovers the dynamic effect to within two hundredths at every exposure. It also answers the question the team was asking better than any single number: the routing logic is worth 0.2 in its first month and 2.0 after ten, and a report that says "0.6" or "1.27" hides the shape that matters for deciding whether to keep rolling it out.

![Treatment effect by periods since adoption for a staggered rollout in which the effect grows with exposure. The group-time estimates lie on the true dynamic effect at every exposure; the two-way fixed effects coefficient, a horizontal line at about 0.6, sits below half the true average effect on the treated.](/assets/images/figures/staggered_did_event_study.png){: width="1152" height="672" loading="lazy"}

## Which Control Group

Two versions of the group-time estimator were run: one using only never-treated units as controls, one using every unit not yet treated at $t$. Both are valid under parallel trends, and both gave 1.26 to 1.27 in the growing-effect scenario. The not-yet-treated version uses more data and is more precise, especially in the later periods when most units have adopted, and it is the default in most implementations. The never-treated version is the one to prefer when the never-treated units are a well-defined group with a reason to be untreated that does not involve the outcome, and the not-yet-treated units might be timing their adoption in response to it.

When there are no never-treated units at all, the last cohort serves as the control for everyone else until its own adoption, and the effects for that last cohort in its post-period cannot be estimated. That is a limit of the data, not of the method, and it is better stated than papered over by a regression that will produce a number regardless.

## Reading the Assumptions

The group-time estimator fixes the comparison problem; it does not relax parallel trends, and it adds a no-anticipation assumption: units do not change their outcome before adoption because they know it is coming. Both are checkable in the same framework. Parallel trends is tested by estimating placebo effects at negative exposures, before adoption, which should be zero; the event study makes this a plot of pre-adoption coefficients around zero. Anticipation shows up as a placebo effect in the periods just before adoption, and the remedy is to move the base period earlier.

Heterogeneity across cohorts is not an assumption violation; it is the reason the method exists. If the waves were chosen because the early regions were expected to benefit most, the third scenario is the relevant one, and the group-time estimator handles it because it never pools cohorts before estimating their effects.

## What to Do

1. **Do not fit a single treatment dummy with two-way fixed effects to a staggered rollout** unless you have a reason to believe the effect is the same for every unit at every exposure. Check that belief against an event study before relying on it.
2. **Estimate group-time effects**: for each adoption cohort and post period, a difference-in-differences against never-treated or not-yet-treated units from the period before adoption.
3. **Aggregate by exposure** and report the event study, with pre-adoption placebo coefficients, as the main result. Add an overall average if a single number is needed, and say what it averages over.
4. **Choose the control group deliberately**: not-yet-treated for precision, never-treated when adoption timing might respond to the outcome.
5. **Check parallel trends and anticipation** on the pre-adoption coefficients rather than asserting them.
6. **Use an implementation that does this for you** when the panel is large: the `did` package in R and `csdid` in Stata implement Callaway and Sant'Anna, and `differences` in Python covers the same ground. Stacked regression and the imputation estimator of Borusyak, Jaravel and Spiess are alternatives with the same logic.

## References

- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964-2996.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs: robust and efficient estimation. *Review of Economic Studies*, 91(6), 3253-3285.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics*, 235(2), 2218-2244.
