---
permalink: '/machine-learning/bandits_vs_ab_tests_cost_of_learning/'
title: 'Bandits or A/B Tests: What Adaptive Allocation Buys and What It Costs'
categories:
- Machine Learning
tags:
- Experimental Design
- A/B Testing
- Reinforcement Learning
- Decision Making
author_profile: false
seo_title: 'Multi-Armed Bandits vs A/B Tests'
seo_description: 'Thompson sampling loses a fraction of the conversions a fixed split loses and finds the winner just as often. It also overstates the effect, starves the losing arm of data and breaks the usual significance test. A simulation puts numbers on each side of the trade.'
excerpt: >-
  Over 20,000 users, an even split between a 10 percent and a 13 percent
  variant gives up 301 conversions to learn which is better. Thompson
  sampling gives up 19, picks the same winner, and reports its advantage
  as 3.9 points instead of 3.0.
summary: >-
  What a bandit optimises and what an A/B test optimises, a simulation of
  Thompson sampling against fixed and test-then-exploit splits over a fixed
  horizon with clear and small differences, the conversions each gives up
  and how often each identifies the winner, why estimates from adaptively
  collected data are biased and the naive test invalid, what happens when
  the arms change during the run, and a rule for choosing between the two.
keywords:
  - multi-armed bandit
  - Thompson sampling
  - A/B testing
  - regret
  - explore-exploit
  - adaptive allocation
  - experimentation
classes: wide
date: '2025-11-12'
why_this_exists: >-
  Bandits are sold as A/B tests that do not waste traffic, and they are
  usually described only by their regret. This post measures both sides of
  the trade on the same simulated experiments so that a team can decide
  which tool matches the decision in front of it.
evidence: >-
  Simulated experiments of 20,000 users with two or three variants at
  conversion rates from 10 to 13 percent, comparing Thompson sampling with
  an even split for the whole horizon and with an even split for the first
  quarter followed by exploitation; 300 replications per scenario, plus
  A/A runs and a scenario in which a variant improves mid-run.
methodology: >-
  Records conversions lost relative to always showing the best variant,
  the probability the best variant is chosen at the end, the share of
  traffic it receives, the estimated difference between the best and
  worst variants against the truth, the size of the smallest arm, the
  false positive rate of the naive z-test on identical arms, and the
  response to a mid-run change.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-dice.jpg
  og_image: /assets/images/headers/photo-dice.jpg
  overlay_image: /assets/images/headers/photo-dice.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-dice.jpg
  twitter_image: /assets/images/headers/photo-dice.jpg
---
The pitch for a bandit is simple: an A/B test sends half the traffic to the worse variant for the whole test, and a bandit stops doing that as soon as it can tell. Over 20,000 users, with variants converting at 10 and 13 percent, the even split gives up 301 conversions to find out which is which. Thompson sampling gives up 19. Both pick the right winner every time.

That is a real advantage and it is the whole of the case for bandits. The case against them is what the same simulation shows on the next line: the bandit reports the winner's advantage as 3.9 points when it is 3.0, sends only 726 of the 20,000 users to the losing arm, and on two identical variants declares a significant difference 14 percent of the time with a test that is supposed to do so 5 percent of the time. The two tools optimise different things, and the choice between them is a choice about which thing the team needs.

## Two Objectives

An A/B test fixes the allocation in advance so that, at the end, the difference between arms can be estimated without bias and tested with known error rates. It buys inference with traffic: the losing arm gets half the users, and every one of them is a conversion not made, in exchange for an estimate whose precision does not depend on which arm is winning.

A bandit reallocates traffic as it learns, toward whichever arm looks best, with just enough exploration of the others to notice if it is wrong. It buys conversions with inference: the losing arm gets few users, so the estimate of how much worse it is becomes imprecise and, because the allocation depended on the data, biased. Thompson sampling is the standard algorithm: keep a Beta posterior on each arm's rate, draw one sample from each posterior, and show the arm with the largest draw. Arms that might be best get shown in proportion to the probability that they are.

Regret, the conversions lost relative to always showing the best arm, is the bandit's objective. The probability of correctly identifying the best arm, and the accuracy of the estimate of its advantage, are the experiment's objectives. The simulation measures all three for each method.

## A Simulation

Each run is 20,000 users. The A/B policy splits them evenly across arms for a share of the horizon, then sends the rest to the arm with the best observed rate; with the share at 100 percent it is a pure test, with the share at 25 percent it is a test followed by exploitation. Thompson sampling runs for the whole horizon.

```python
import numpy as np

rng = np.random.default_rng(0)

def run_ab(p, horizon, r=rng, test_share=1.0):
    """Even split for test_share of the horizon, then everyone to the arm with the best observed rate."""
    k = len(p)
    n = np.zeros(k); s = np.zeros(k)
    n_test = int(horizon * test_share)
    arms = np.tile(np.arange(k), n_test // k + 1)[:n_test]
    conv = r.random(n_test) < p[arms]
    for a in range(k):
        n[a] = np.sum(arms == a); s[a] = np.sum(conv[arms == a])
    best = np.argmax(s / n)
    rest = horizon - n_test
    conv_rest = np.sum(r.random(rest) < p[best])
    total = s.sum() + conv_rest
    return total, best, n, s

def run_thompson(p, horizon, r=rng):
    k = len(p)
    n = np.zeros(k); s = np.zeros(k)
    for _ in range(horizon):
        a = np.argmax(r.beta(1 + s, 1 + n - s))      # one draw per arm from its Beta posterior
        n[a] += 1
        s[a] += r.random() < p[a]
    return s.sum(), np.argmax(s / np.maximum(n, 1)), n, s

def compare(p, horizon, reps=300, test_share=1.0):
    p = np.asarray(p)
    oracle = horizon * p.max()
    out = {"A/B test": [], "Thompson sampling": []}
    for _ in range(reps):
        for name, fn in (("A/B test", lambda: run_ab(p, horizon, test_share=test_share)),
                         ("Thompson sampling", lambda: run_thompson(p, horizon))):
            total, best, n, s = fn()
            est = s / np.maximum(n, 1)
            out[name].append((oracle - total, best == p.argmax(), n[p.argmax()] / horizon,
                              est[p.argmax()] - est[p.argmin()], n.min()))
    res = {}
    for name, rows in out.items():
        rows = np.array(rows, dtype=float)
        res[name] = dict(regret=rows[:, 0].mean(), identified=rows[:, 1].mean(), share_best=rows[:, 2].mean(),
                         est_diff=rows[:, 3].mean(), est_sd=rows[:, 3].std(), min_n=rows[:, 4].mean())
    return res

def report(title, p, horizon, **kw):
    print(f"\n{title}: rates {list(p)}, {horizon:,} users")
    res = compare(p, horizon, **kw)
    for name, d in res.items():
        print(f"  {name:18} lost {d['regret']:>6.1f}  picks best {d['identified']:.0%}  traffic to best {d['share_best']:.0%}  "
              f"gap {d['est_diff']:.4f} +/- {d['est_sd']:.4f} (true {max(p) - min(p):.4f})  smallest arm {d['min_n']:,.0f}")

report("Two arms, clear winner", [0.10, 0.13], 20000)
report("Two arms, small difference", [0.10, 0.105], 20000)
report("Three arms, one winner", [0.10, 0.10, 0.12], 20000)
report("A/B test on the first quarter, then exploit", [0.10, 0.13], 20000, test_share=0.25)
report("A/B test on the first quarter, small difference", [0.10, 0.105], 20000, test_share=0.25)
```

**Two arms, a clear winner** at 10 and 13 percent.

| Policy | Conversions lost | Picks the best | Traffic to the best | Estimated gap (true 0.030) | Smallest arm |
| --- | --- | --- | --- | --- | --- |
| A/B test, whole horizon | 301 | 100% | 50% | 0.0295 ± 0.0047 | 10,000 |
| Thompson sampling | 19 | 100% | 96% | 0.0390 ± 0.0168 | 726 |

Both policies identify the winner every time. The bandit loses a fifteenth of the conversions, because after a few hundred users the posteriors have separated and it sends 96 percent of traffic to the better arm. The price is on the right of the table: the bandit's estimate of the gap is 3.9 points against a true 3.0, with a spread three and a half times the A/B test's, and the losing arm has 726 users behind it instead of 10,000. If the question was "which variant", the bandit answered it cheaply. If the question was "how much better", it gave the wrong number.

**Two arms, a small difference** at 10 and 10.5 percent.

| Policy | Conversions lost | Picks the best | Traffic to the best | Estimated gap (true 0.005) | Smallest arm |
| --- | --- | --- | --- | --- | --- |
| A/B test, whole horizon | 51 | 86% | 50% | 0.0050 ± 0.0046 | 10,000 |
| Thompson sampling | 34 | 84% | 67% | 0.0065 ± 0.0123 | 4,563 |

When the arms are close, there is less to gain by exploiting and less to lose by exploring, and the two policies converge: similar regret, similar identification, and the bandit cannot tell the arms apart well enough to move much traffic. The estimation problem remains. A half-point true difference is reported as 0.65 with a spread of 1.2 points, which is to say the bandit's data cannot measure the effect at all, while the A/B test's estimate has a standard error of under half a point.

**Three arms, one winner** at 10, 10 and 12 percent.

| Policy | Conversions lost | Picks the best | Traffic to the best | Estimated gap (true 0.020) | Smallest arm |
| --- | --- | --- | --- | --- | --- |
| A/B test, whole horizon | 266 | 100% | 33% | 0.0197 ± 0.0054 | 6,666 |
| Thompson sampling | 55 | 100% | 86% | 0.0267 ± 0.0141 | 835 |

More arms favour the bandit, because the fixed split wastes traffic on every loser at once, while the bandit drops all of them. The estimate is again a third too high.

![Cumulative conversions lost against the number of users for Thompson sampling, an even split for the first quarter followed by exploitation, and an even split for the whole test, with variants at 10 and 13 percent. The fixed split loses at a constant rate; Thompson sampling stops losing after the first few thousand users.](/assets/images/figures/bandit_cumulative_regret.png){: width="1152" height="672" loading="lazy"}

## The Test-Then-Exploit Middle

Most teams do not run an A/B test for the whole life of a decision. They test for a while, pick, and ship. That policy is the honest comparator for a bandit, and the simulation runs it with a quarter of the horizon spent testing.

| Scenario | Policy | Conversions lost | Picks the best | Estimated gap | Smallest arm |
| --- | --- | --- | --- | --- | --- |
| Clear winner (0.030) | Test a quarter, then exploit | 77 | 100% | 0.0300 ± 0.0086 | 2,500 |
| Clear winner (0.030) | Thompson sampling | 23 | 100% | 0.0375 ± 0.0171 | 785 |
| Small difference (0.005) | Test a quarter, then exploit | 35 | 68% | 0.0047 ± 0.0088 | 2,500 |
| Small difference (0.005) | Thompson sampling | 33 | 89% | 0.0079 ± 0.0104 | 4,306 |

Testing for a quarter and exploiting recovers most of the bandit's advantage when the winner is clear, at 77 lost conversions against 23, and keeps an unbiased estimate with 2,500 users on each arm. When the difference is small the picture reverses in an instructive way: the fixed quarter is too short to tell the arms apart, so the test picks the wrong one a third of the time, whereas the bandit, which never stops looking, ends up right 89 percent of the time and has by then put more users on the losing arm than the test did. A bandit is a test that does not have to decide when to stop; a test-then-exploit policy has to choose the test length before it knows the effect size, and gets it wrong in one direction or the other.

## Why the Bandit's Estimate Is Wrong

The bias is not a defect of Thompson sampling; it belongs to any procedure that allocates on the basis of the data. An arm that happens to run hot early is shown more, so its estimate is based on many users and is close to its true rate; an arm that runs cold early is shown less and its estimate stays where the unlucky start left it. The difference between them is systematically overstated, and the sampling distribution of the naive estimate is not the one the z-test assumes.

```python
p = np.array([0.10, 0.10])
diffs_ab, diffs_ts, sig_ab, sig_ts = [], [], 0, 0
for _ in range(400):
    _, _, n, s = run_ab(p, 20000)
    est = s / n; se = np.sqrt((est * (1 - est) / n).sum())
    diffs_ab.append(est[1] - est[0]); sig_ab += abs((est[1] - est[0]) / se) > 1.96
    _, _, n, s = run_thompson(p, 20000)
    est = s / np.maximum(n, 1); se = np.sqrt((est * (1 - est) / np.maximum(n, 1)).sum())
    diffs_ts.append(abs(est[1] - est[0])); sig_ts += abs((est[1] - est[0]) / se) > 1.96
print(f"A/B: mean |gap| {np.mean(np.abs(diffs_ab)):.4f}, 'significant' {sig_ab/400:.0%}")
print(f"Thompson: mean |gap| {np.mean(diffs_ts):.4f}, 'significant' {sig_ts/400:.0%}")
```

| Two identical arms, 20,000 users | Mean absolute estimated gap | Naive z-test "significant" |
| --- | --- | --- |
| A/B test | 0.0034 | 5% |
| Thompson sampling | 0.0077 | 14% |

On arms that are the same, the bandit's data show a gap more than twice the size the A/B test's data show, and the standard test rejects nearly three times as often as it should. A team that runs a bandit and then reports a p-value from the final counts has a 14 percent false positive rate and does not know it. Valid inference from adaptively collected data exists, through inverse-probability weighting of each observation by the probability the arm was shown at the time, or through always-valid sequential methods, but it has to be built in, and the estimate it yields is far less precise than an A/B test's because the losing arm has so little data.

## When the World Moves

A bandit's confidence is built on its history, and history can mislead. If a variant starts worse and becomes better, because it needs a cache to warm or a model to retrain or because the audience changes, the bandit has already concluded against it.

```python
def run_ts_shift(horizon, r=rng):
    n = np.zeros(2); s = np.zeros(2); total = 0
    for i in range(horizon):
        p = np.array([0.10, 0.08 if i < 5000 else 0.13])   # arm B improves after 5,000 users
        a = np.argmax(r.beta(1 + s, 1 + n - s))
        c = r.random() < p[a]
        n[a] += 1; s[a] += c; total += c
    return total, n

res = [run_ts_shift(20000) for _ in range(200)]
oracle = 5000 * 0.10 + 15000 * 0.13
print(f"Thompson: lost {np.mean([oracle - t for t, _ in res]):.0f}; traffic to B overall {np.mean([n[1] for _, n in res]) / 20000:.0%}")
print(f"Even split: lost {oracle - (0.5 * 5000 * 0.18 + 0.5 * 15000 * 0.23):.0f}")
```

Arm B converts at 8 percent for the first 5,000 users and 13 percent afterwards, against a constant 10 percent. Thompson sampling, having learned that B is worse, sends it little traffic and takes a long time to notice the change: over the whole run B receives 52 percent of users where an oracle would have given it 75 percent, and 184 conversions are lost. The even split loses 275, more in total, but it never stops watching, and the analyst reading its data sees the change the week it happens. A bandit built for this situation discounts old observations or uses a sliding window, at the cost of never fully converging; a bandit not built for it is confidently wrong.

## Choosing

The rule that falls out of the tables is about the decision, not the algorithm.

Use a bandit when the goal is to earn during the experiment and the decision does not need a number: many arms, short-lived content, personalised recommendations, headline choices, anything where "show the best one" is the whole of the requirement and the effect size will never be quoted. The regret saving is largest exactly there: many arms, clear differences, long horizons.

Use an A/B test when the decision needs an estimate: a change that will be rolled out permanently, reported to stakeholders as a lift, weighed against a cost, or used to build a model of what works. The traffic spent on the losing arm is the price of a number that can be trusted, and test-then-exploit recovers most of the bandit's saving once the test is long enough to decide.

Use neither as a substitute for the other. A bandit with a p-value stapled to it has the bias without the precision, and an A/B test that runs forever because nobody will call it has the regret without the learning.

## What to Do

1. **Write down whether the decision needs an effect size.** If yes, run an A/B test; if the requirement is only to show the best arm, a bandit is the right tool.
2. **For an A/B test, fix the length in advance** from the smallest effect worth detecting, then exploit; the quarter-horizon test in the simulation lost a quarter of what the full-horizon test lost.
3. **Never report a naive estimate or p-value from bandit data.** If inference is needed from adaptive allocation, use inverse-probability weighting or an always-valid method, and expect wide intervals.
4. **Keep a floor on exploration** in any bandit that will run for long, so a variant that improves can be rediscovered, and log the allocation probabilities with every observation.
5. **Prefer bandits when arms are many and differences large**, which is where the regret saving is biggest and identification is not in doubt.
6. **Treat the winner's estimated advantage from a bandit as an upper bound**, not a measurement; in the simulation it ran a third above the truth.

## References

- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4), 285-294.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1-96.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.
- Nie, X., Tian, X., Taylor, J., & Zou, J. (2018). Why adaptively collected data have negative bias and how to correct for it. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics*, 1261-1269.
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence intervals for policy evaluation in adaptive experiments. *Proceedings of the National Academy of Sciences*, 118(15), e2014602118.
- Scott, S. L. (2015). Multi-armed bandit experiments in the online service economy. *Applied Stochastic Models in Business and Industry*, 31(1), 37-45.
