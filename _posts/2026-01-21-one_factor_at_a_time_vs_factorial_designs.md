---
permalink: '/statistics/one_factor_at_a_time_vs_factorial_designs/'
title: 'One Factor at a Time Is Not an Experiment: Factorial Designs and Interactions'
categories:
- Statistics
tags:
- Experimental Design
- Design of Experiments
- Statistical Modeling
- Statistics
author_profile: false
seo_title: 'Factorial Designs vs One Factor at a Time'
seo_description: 'Changing one factor at a time from a baseline cannot see interactions, wastes runs and can end at the wrong setting. A simulation shows a factorial design of the same size finding the optimum ten times more often, and why.'
excerpt: >-
  The team tests each of four process settings on its own, sees two of them
  make things worse, keeps the baseline, and never learns that the two
  "harmful" changes together would have lifted the response by eight points.
summary: >-
  Why one-factor-at-a-time experiments cannot measure interactions and can be
  trapped by them, what a two-level factorial design measures instead and why
  every run informs every effect, a simulation of a four-factor process with
  one strong interaction comparing effect estimates and the share of
  experiments that find the best setting under equal budgets, what a half
  fraction gives up through aliasing, and how to plan a screening
  experiment.
keywords:
  - design of experiments
  - factorial design
  - one factor at a time
  - interaction effects
  - fractional factorial
  - aliasing
  - screening experiments
classes: wide
date: '2026-01-21'
why_this_exists: >-
  Most process tuning in engineering and operations is done one change at a
  time because it feels careful. This post shows, with a small simulated
  process, that the habit cannot see the effects that matter most and
  measures the rest less precisely than a factorial design of the same
  size, and gives the design that replaces it.
evidence: >-
  A simulated four-factor process with a strong two-factor interaction and
  noise comparable to the smaller effects, run 4,000 times per design
  under equal run budgets: one factor at a time from a baseline, a
  sixteen-run full factorial, an eight-run half fraction and replicated
  factorials.
methodology: >-
  Compares the bias and standard error of the main-effect estimates, the
  share of experiments ending at the true best setting, the power to detect
  the interaction, and the alias structure of the half fraction; standard
  errors are checked against the closed form for two-level designs.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/data_science_13.jpg
  og_image: /assets/images/data_science_13.jpg
  overlay_image: /assets/images/data_science_13.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_13.jpg
  twitter_image: /assets/images/data_science_13.jpg
---
A production line has four settings the engineers believe matter: a temperature, a feed rate, a catalyst grade and a mixing time. The current recipe is the baseline. To improve it they do the careful thing. Holding everything else at baseline, they raise the temperature and run three batches. Yield drops by a point. They put the temperature back, raise the feed rate, run three more batches. Yield drops by two. They try the catalyst: nothing. The mixing time: up by one. Fifteen batches later they keep the longer mixing time and file a report saying the other three settings are already where they should be.

The process they were tuning has an eight-point improvement available. Raising the temperature *and* the feed rate together lifts the yield from 8.5 to 16.5. Either change alone, at the baseline value of the other, makes things worse, which is exactly what the fifteen batches measured, honestly and precisely. One factor at a time cannot see an interaction, and when the interaction is the largest effect in the system, it walks the experiment into the wrong corner and leaves it there.

## What One Factor at a Time Measures

Write the process in coded units, with each factor at $-1$ for its baseline and $+1$ for its changed value. The simulated line is

$$
y = 10 + 2A + 1.5B + 0C + 0.5D + 2.5\,AB + \varepsilon, \qquad \varepsilon \sim N(0, 2^2),
$$

so temperature ($A$) and feed rate ($B$) both help, the catalyst ($C$) does nothing, the mixing time ($D$) helps a little, and $A$ and $B$ reinforce each other strongly. The *main effect* of a factor, in the language of experimental design, is the change in the response when it goes from $-1$ to $+1$, averaged over the settings of the other factors. For $A$ that is $2 \times 2 = 4$ points; for $B$, 3; for $D$, 1; and the $AB$ interaction effect is $2 \times 2.5 = 5$.

One factor at a time does not measure the main effect. It measures the effect of $A$ at $B = -1$, which is $2\beta_A + 2\beta_{AB} \cdot (-1) = 4 - 5 = -1$. The sign is wrong, not because of noise but because the quantity being measured is a different one. The same happens to $B$: at $A = -1$ its effect is $3 - 5 = -2$. No amount of replication fixes this, and the table below shows the estimates converging on the wrong numbers as the noise averages out.

## A Simulation

The design comparison holds the budget fixed. One factor at a time uses five settings, the baseline and four single changes, replicated three times each for fifteen runs. The full $2^4$ factorial uses all sixteen corners of the cube once. Both designs are analysed the way they are meant to be: differences of means for the first, least squares on main effects and two-factor interactions for the second.

```python
import itertools
import numpy as np

rng = np.random.default_rng(0)
sigma = 2.0                                                   # run-to-run noise
beta = {"A": 2.0, "B": 1.5, "C": 0.0, "D": 0.5, "AB": 2.5}    # coded (-1, +1) coefficients

def response(x, r=rng, noise=True):
    A, B, C, D = x.T
    mean = 10 + beta["A"]*A + beta["B"]*B + beta["C"]*C + beta["D"]*D + beta["AB"]*A*B
    return mean + (r.normal(0, sigma, len(x)) if noise else 0)

corners = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
base = -np.ones(4)

def ofat(runs_per_setting, r=rng):
    """Baseline plus one change per factor; keep a change only if it helped."""
    settings = [base.copy()]
    for col in range(4):
        x = base.copy(); x[col] = 1; settings.append(x)
    y = np.array([response(np.repeat(s[None], runs_per_setting, 0), r).mean() for s in settings])
    effects = y[1:] - y[0]
    chosen = base.copy(); chosen[effects > 0] = 1
    return effects, chosen

def factorial(design, r=rng):
    """Least squares on main effects and two-factor interactions; pick the best predicted corner."""
    pairs = list(itertools.combinations(range(4), 2))
    if len(design) == 8:                  # half fraction: BC=AD, BD=AC, CD=AB, so fit one of each pair
        pairs = [(0, 1), (0, 2), (0, 3)]
    cols = lambda d: np.column_stack([np.ones(len(d))] + [d[:, i] for i in range(4)]
                                     + [d[:, i]*d[:, j] for i, j in pairs])
    coef, *_ = np.linalg.lstsq(cols(design), response(design, r), rcond=None)
    names = ["I", "A", "B", "C", "D"] + ["ABCD"[i] + "ABCD"[j] for i, j in pairs]
    return dict(zip(names, 2*coef)), corners[(cols(corners) @ coef).argmax()]

full = corners.copy()                                # 16 runs
half = corners[np.prod(corners, axis=1) == 1]        # 2^(4-1) with I = ABCD, 8 runs
reps = 4000

ofat_eff = np.array([ofat(3)[0] for _ in range(reps)])
full_eff = [factorial(full)[0] for _ in range(reps)]
print("effect  true   OFAT mean  OFAT sd   2^4 mean  2^4 sd")
for k, f in enumerate("ABCD"):
    fe = np.array([e[f] for e in full_eff])
    print(f"{f:6}{2*beta[f]:>6.1f}{ofat_eff[:, k].mean():>11.2f}{ofat_eff[:, k].std():>9.2f}{fe.mean():>10.2f}{fe.std():>8.2f}")
ab = np.array([e["AB"] for e in full_eff])
print(f"AB    {2*beta['AB']:>6.1f}{'n/a':>11}{'':>9}{ab.mean():>10.2f}{ab.std():>8.2f}")
```

**Effect estimates at equal budgets.** Fifteen runs one at a time against sixteen runs in a factorial.

| Effect | True main effect | One at a time, mean | One at a time, sd | Factorial, mean | Factorial, sd |
| --- | --- | --- | --- | --- | --- |
| A | 4.0 | -1.0 | 1.6 | 4.0 | 1.0 |
| B | 3.0 | -2.0 | 1.6 | 3.0 | 1.0 |
| C | 0.0 | 0.0 | 1.6 | 0.0 | 1.0 |
| D | 1.0 | 1.0 | 1.6 | 1.0 | 1.0 |
| AB | 5.0 | not estimable | | 5.0 | 1.0 |

Two things are wrong with the left half of the table and only one of them is the interaction. The estimates of $A$ and $B$ carry the wrong sign, because one at a time measures each factor at the baseline of the other. But look at the standard deviations too: 1.6 against 1.0, from almost the same number of runs. The factorial estimate of every effect uses all sixteen runs, eight at $+1$ against eight at $-1$, while the one-at-a-time estimate of each effect uses six, three against three. Fisher called this hidden replication: in a factorial design every run contributes to every effect estimate. With noise $\sigma$ and $N$ runs, the standard error of a two-level factorial effect is $2\sigma/\sqrt{N}$, which is 1.0 here; one factor at a time with $N/5$ runs per setting gives $\sigma\sqrt{2/(N/5)}$, which is 1.58. To match the factorial's precision on the main effects alone, one factor at a time needs two and a half times the runs, and it still cannot estimate the interaction at any budget.

**Finding the best setting.** The point of the experiment is the recipe it ends with. Each simulated experiment finishes by choosing a setting: one at a time keeps every change that improved the response, the factorial fits its model and picks the corner with the highest prediction.

```python
def share_best(fn, arg):
    hits = 0
    for _ in range(reps):
        chosen = fn(arg)[1]
        hits += chosen[0] == 1 and chosen[1] == 1 and chosen[3] == 1    # A+, B+, D+
    return hits / reps

for label, fn, arg in [("OFAT, 15 runs", ofat, 3), ("OFAT, 40 runs", ofat, 8), ("OFAT, 100 runs", ofat, 20),
                       ("half fraction, 8 runs", factorial, half), ("full factorial, 16 runs", factorial, full),
                       ("full factorial x2, 32 runs", factorial, np.vstack([full, full]))]:
    print(f"{label:28}{share_best(fn, arg):.0%}")
```

| Design | Runs | Ends at the best setting |
| --- | --- | --- |
| One factor at a time | 15 | 7% |
| One factor at a time | 40 | 2% |
| One factor at a time | 100 | 0% |
| Half fraction $2^{4-1}$ | 8 | 69% |
| Full factorial $2^4$ | 16 | 70% |
| Full factorial, replicated | 32 | 77% |

The one-at-a-time rows get worse with more runs. That is the signature of bias rather than noise: with fifteen runs the estimates are noisy enough that the experiment occasionally stumbles into raising $A$ or $B$ by accident, and with a hundred runs it measures the wrong quantities precisely and never does. The half fraction, with eight runs, finds the optimum ten times more often than one at a time with a hundred.

The factorial designs miss the optimum about three times in ten, and every one of those misses is on $D$. Across three thousand further experiments, the sixteen-run factorial and the eight-run half fraction both set $A$ and $B$ correctly every time and $D$ correctly about 70 percent of the time. The choice of $D$ at the best corner rides on its main effect of one point plus its three interactions with the other factors, each estimated with a standard error of one, so the prediction that decides it has about even odds of the wrong sign when the effect is this small. The design finds the large effects and the interaction at once, and leaves a small effect to be resolved by replication, which is the right order to learn things in.

![Share of simulated experiments that end at the best setting against the number of runs, for a four-factor process with one strong interaction. One factor at a time is trapped by the interaction at every budget; the factorial designs find the optimum two thirds to three quarters of the time from eight to thirty-two runs.](/assets/images/figures/factorial_vs_ofat_optimum.png){: width="1152" height="672" loading="lazy"}

## Seeing the Interaction

The factorial design estimates the $AB$ interaction as the difference between the effect of $A$ at $B = +1$ and its effect at $B = -1$, halved, and every run contributes to it. At sixteen runs the estimate is $5.0 \pm 1.0$ and its $t$ statistic exceeds 2 in every one of the four thousand simulations. An interaction of this size is not subtle; it is invisible to one-factor-at-a-time experimentation by construction, not by lack of power.

```python
for n_rep in (1, 2, 4):
    d = np.vstack([full] * n_rep)
    ests = np.array([factorial(d)[0]["AB"] for _ in range(reps)])
    se = 2 * sigma / np.sqrt(len(d))
    print(f"{len(d):>3} runs: AB = {ests.mean():.2f} +/- {ests.std():.2f} (se {se:.2f}), detected {np.mean(np.abs(ests)/se > 2):.0%}")
```

| Runs | AB estimate | Standard error | Detected ($\lvert t \rvert > 2$) |
| --- | --- | --- | --- |
| 16 | 5.0 ± 1.0 | 1.00 | 100% |
| 32 | 5.0 ± 0.7 | 0.71 | 100% |
| 64 | 5.0 ± 0.5 | 0.50 | 100% |

The simulated standard deviations match $2\sigma/\sqrt{N}$ to two decimals, which is the check that the least-squares analysis is doing what the design theory says it should.

## What a Half Fraction Gives Up

Eight runs cannot estimate sixteen quantities. The half fraction $2^{4-1}$ keeps the eight corners for which $ABCD = +1$, and the price is that some effects become indistinguishable: $AB$ and $CD$ have identical columns in the design, as do $AC$ and $BD$, and $AD$ and $BC$. The design estimates their sums. Each main effect is aliased with a three-factor interaction, which is usually harmless.

```python
print("AB*CD over the 8 runs:", np.unique(half[:, 0]*half[:, 1]*half[:, 2]*half[:, 3]))
half_ab = np.array([factorial(half)[0]["AB"] for _ in range(reps)])
print(f"half fraction AB estimate: {half_ab.mean():.2f} +/- {half_ab.std():.2f}  (true AB + CD = 5.0 + 0)")
```

The product of the $A$, $B$, $C$ and $D$ columns is $+1$ on every run, so $AB$ and $CD$ coincide, and the estimate labelled $AB$ is $5.0 \pm 1.4$, which is $AB + CD$. Here $CD$ is zero and the fraction tells the truth; in general the experimenter does not know that, and the eight-run design has bought a strong hint rather than a measurement. The standard practice is to follow up: if the aliased pair matters, run the other half of the design, which separates them, and the sixteen runs together are the full factorial.

This is a resolution IV design: main effects are clear of two-factor interactions, two-factor interactions are confounded with each other. With more factors the fractions get deeper and the aliasing tighter, and a resolution III design, such as a Plackett-Burman screen of eleven factors in twelve runs, confounds main effects with two-factor interactions. That is a reasonable trade when the aim is to find which of many factors matter at all, and a bad one when interactions are the reason for the experiment.

## Why the Habit Persists

One factor at a time feels like control. Each comparison has a clear baseline and changes one thing, so each result has an obvious reading. The factorial design changes several factors in every run, and the individual runs mean nothing on their own, which is uncomfortable until the analysis is understood: the effects are averages over patterns of runs, and the patterns are chosen so that the averages separate cleanly.

There are situations in which sequential single changes are right. When a run is expensive and the response surface is smooth and the aim is to climb it, a sequence of small steps along an estimated gradient, response surface methodology, is the standard approach; even then the gradient at each step is estimated from a small factorial design, not from single changes. When factors physically cannot be changed together, or when a change is irreversible, the design has to respect that. And when the goal is to confirm one specific change against the current recipe, a two-arm comparison is the design, and the question of interactions does not arise because nothing else is moving.

Outside those cases, the factorial design is not a more sophisticated option. It is the experiment, and the one-at-a-time sequence is a set of measurements that happen to be taken at one corner of it.

## What to Do

1. **List the factors and their two levels** before running anything, including the ones believed not to matter. Four factors at two levels is sixteen runs; six is sixty-four, or sixteen in a quarter fraction.
2. **Run a two-level factorial or a fraction of it**, in random order, rather than a sequence of single changes. Randomisation is what protects the effects from drifts and batch effects that arrive over the course of the experiment.
3. **Analyse by least squares on coded factors** and read the main effects and two-factor interactions together. A large interaction means the main effects cannot be interpreted separately.
4. **Size the design from $2\sigma/\sqrt{N}$**, with $\sigma$ from past process variation, so that the smallest effect worth acting on is at least two standard errors.
5. **Know the alias structure** of any fraction before running it, and plan the fold-over that resolves the aliases that turn out to matter.
6. **Confirm the chosen setting** with a few replicated runs at the recommended corner; the model's best prediction is an estimate, and the confirmation run is cheap.

## References

- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd.
- Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley.
- Czitrom, V. (1999). One-factor-at-a-time versus designed experiments. *The American Statistician*, 53(2), 126-131.
- Plackett, R. L., & Burman, J. P. (1946). The design of optimum multifactorial experiments. *Biometrika*, 33(4), 305-325.
- Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of optimum conditions. *Journal of the Royal Statistical Society: Series B*, 13(1), 1-45.
