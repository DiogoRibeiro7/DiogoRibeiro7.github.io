---
permalink: '/machine-learning/proxy_metrics_goodhart_optimisation_pressure/'
title: 'Proxy Metrics Under Optimisation: Why a Correlation of 0.6 Is Not a Substitute for the Goal'
categories:
- Machine Learning
tags:
- Model Evaluation
- Decision Making
- Metrics
- Machine Learning
author_profile: false
seo_title: 'Proxy Metrics, Goodhart and Optimisation Pressure'
seo_description: 'A proxy that correlates 0.63 with the goal looks like a safe target. A simulation shows that optimising it delivers half the gain it reports, that the correlation collapses among the selected items, and that when the manipulable part costs the goal, hard optimisation makes things worse.'
excerpt: >-
  The proxy correlates 0.63 with the metric that matters, so the team
  optimises it and reports a gain of 3.8. The goal metric moved by 1.9.
  Change one thing, make the manipulable part mildly costly, and the
  same optimisation moves the goal by nothing at all.
summary: >-
  Why a proxy is the sum of the thing you want and the part that is
  easier to move, a simulation in which selecting hard on the proxy
  delivers a known fraction of the reported gain, how that fraction
  follows the variance decomposition rather than the observed
  correlation, why the proxy-goal correlation collapses inside the
  selected set, what happens as optimisation pressure grows with the
  candidate pool, the case where optimisation actively harms the goal,
  how small a holdout detects it, and why averaging several proxies
  helps.
keywords:
  - proxy metrics
  - Goodhart's law
  - surrogate outcomes
  - optimisation pressure
  - metric design
  - holdout
  - model evaluation
classes: wide
date: '2026-09-12'
why_this_exists: >-
  Every team optimises something other than what it cares about, because
  the thing it cares about is slow, noisy or unmeasurable. The usual
  justification is a correlation computed before any optimisation began.
  This post shows what that correlation is worth once the optimiser
  starts pushing, and what to measure instead.
evidence: >-
  Simulated candidate pools of 100 to 200,000 items in which the proxy
  is the sum of a quality component and a manipulable component and the
  goal metric depends on quality, with the manipulable part contributing
  nothing or actively costing the goal; selection of the top 50 to 1
  percent by proxy, 200 replications per cell.
methodology: >-
  Measures the observed proxy-goal correlation before selection, the
  proxy gain and goal gain from selecting on the proxy, the share of the
  reported gain that is real against the variance decomposition, the
  correlation inside the selected set, the effect of enlarging the
  candidate pool at fixed output, the goal gain when the manipulable
  component carries a cost, the detection power of a goal-metric
  holdout, and the effect of averaging independent proxies.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-punchcards.jpg
  og_image: /assets/images/headers/photo-punchcards.jpg
  overlay_image: /assets/images/headers/photo-punchcards.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-punchcards.jpg
  twitter_image: /assets/images/headers/photo-punchcards.jpg
---
The metric the business cares about arrives too late and too noisily to optimise: renewal, satisfaction, long-term retention. So the team picks a proxy that arrives immediately, checks that it correlates 0.63 with the goal across the existing catalogue, and starts ranking by it. Six months later the proxy is up by 3.8 units and the goal is up by 1.9.

Nothing was faked and no bug was introduced. The correlation of 0.63 was real, and it was computed on a population nobody was optimising. Once the optimiser starts choosing items by the proxy, it stops sampling that population and starts sampling the top of it, and the top of a proxy is enriched in whatever is easiest to raise, which is not the same as what the goal rewards.

## A Proxy Is a Sum

Write the proxy as the thing you want plus the part that moves it without moving the goal:

$$
P = q + m, \qquad Y = q - c\,m + \varepsilon .
$$

Here $q$ is quality, which the goal rewards; $m$ is the manipulable component, clickbait in a headline, padding in an answer, urgency in a subject line, anything that raises the proxy on its own; and $c$ says whether that component is merely useless to the goal ($c = 0$) or actively costly ($c > 0$). The observed correlation between $P$ and $Y$ depends on how much variance $m$ contributes, and it can look reassuring while $m$ is doing most of the work at the top of the distribution.

```python
import numpy as np

rng = np.random.default_rng(0)

def pool(n, var_m=1.0, cost=0.0, r=rng):
    """Proxy P = q + m; goal Y = q - cost * m. The manipulable part m raises the proxy and
    is worthless (cost 0) or harmful (cost > 0) to the goal."""
    q = r.normal(0, 1, n)
    m = r.normal(0, np.sqrt(var_m), n)
    y = q - cost * m + r.normal(0, 0.5, n)
    p = q + m
    return q, m, y, p

for var_m, cost in ((0.25, 0.0), (1.0, 0.0), (4.0, 0.0), (1.0, 0.5), (1.0, 1.0)):
    q, m, y, p = pool(200000, var_m, cost)
    print(f"var(m) {var_m:.2f}, cost {cost:.1f}: corr(proxy, goal) = {np.corrcoef(p, y)[0, 1]:+.2f}")
```

| Manipulable variance | Cost to the goal | Observed correlation |
| --- | --- | --- |
| 0.25 | 0 | +0.80 |
| 1.00 | 0 | +0.63 |
| 4.00 | 0 | +0.40 |
| 1.00 | 0.5 | +0.29 |
| 1.00 | 1.0 | -0.00 |

The last row is the one to hold on to. A proxy can be uncorrelated with the goal in the population and still be the thing everyone optimises, because the correlation that gets checked is usually computed on a sample where $q$ dominates, or long before $m$ had a reason to grow.

## What Optimising It Delivers

Selecting the best items by the proxy raises the proxy by a known amount. How much of that is real depends on how the proxy's variance splits.

```python
for var_m in (0.25, 1.0, 4.0):
    for frac in (0.5, 0.1, 0.01):
        gp, gy = [], []
        for _ in range(200):
            q, m, y, p = pool(20000, var_m)
            sel = np.argsort(p)[-int(len(p) * frac):]
            gp.append(p[sel].mean() - p.mean()); gy.append(y[sel].mean() - y.mean())
        print(f"top {frac:.0%}, var(m) {var_m}: proxy +{np.mean(gp):.2f}, goal +{np.mean(gy):.2f}, "
              f"real share {np.mean(gy)/np.mean(gp):.0%}")
```

| Selection | Manipulable variance | Proxy gain | Goal gain | Share of the reported gain that is real |
| --- | --- | --- | --- | --- |
| Top 50% | 0.25 | +0.89 | +0.71 | 80% |
| Top 10% | 0.25 | +1.96 | +1.57 | 80% |
| Top 1% | 0.25 | +2.98 | +2.38 | 80% |
| Top 50% | 1.00 | +1.13 | +0.56 | 50% |
| Top 10% | 1.00 | +2.48 | +1.24 | 50% |
| Top 1% | 1.00 | +3.77 | +1.88 | 50% |
| Top 50% | 4.00 | +1.78 | +0.36 | 20% |
| Top 10% | 4.00 | +3.93 | +0.79 | 20% |
| Top 1% | 4.00 | +5.96 | +1.19 | 20% |

The real share is constant within each block and equals $\operatorname{var}(q)/[\operatorname{var}(q) + \operatorname{var}(m)]$: 80 percent at $\operatorname{var}(m) = 0.25$, 50 percent at 1, 20 percent at 4. That is the quantity that governs a proxy under optimisation, and it is not the correlation. At $\operatorname{var}(m) = 1$ the correlation is a respectable 0.63 while only half the reported gain is real; at $\operatorname{var}(m) = 4$ the correlation is 0.40 and four fifths of every reported gain is an artefact of selection.

For comparison, selecting on the goal metric itself gains 0.89, 1.96 and 2.98 at the three selection levels. The proxy at $\operatorname{var}(m) = 1$ delivers 0.56, 1.24 and 1.88: about 63 percent of what direct optimisation would have achieved, which is the correlation making its one honest appearance.

## The Correlation Does Not Survive Selection

If the proxy-goal correlation is being monitored to confirm that the proxy is still a good target, it will look worse and worse, and the decline is a property of selection rather than evidence of a change in the world.

```python
for frac in (1.0, 0.5, 0.1, 0.01):
    cs = []
    for _ in range(200):
        q, m, y, p = pool(20000, 1.0)
        sel = np.argsort(p)[-int(len(p) * frac):] if frac < 1 else np.arange(len(p))
        cs.append(np.corrcoef(p[sel], y[sel])[0, 1])
    print(f"top {frac:.0%}: corr among the selected {np.mean(cs):+.2f}")
```

| Set examined | Proxy-goal correlation |
| --- | --- |
| Whole population | +0.63 |
| Top 50% by proxy | +0.44 |
| Top 10% by proxy | +0.32 |
| Top 1% by proxy | +0.25 |

Restricting to high-proxy items removes most of the variation in the proxy and leaves the two components trading off against each other within the set, which is the same mechanism that makes a selected sample look uncorrelated in any restricted range. A team that recomputes the correlation on shipped items and concludes the proxy has "degraded" is measuring the selection it performed.

## More Search Is More Pressure

The share that is real is fixed by the variance split, but the absolute damage grows with how hard the optimiser searches. Holding the output fixed at ten items and enlarging the pool they are chosen from is the cleanest way to see it.

```python
for n in (100, 1000, 10000, 100000):
    gp, g0, g1 = [], [], []
    for _ in range(200):
        q, m, y, p = pool(n, 1.0); sel = np.argsort(p)[-10:]
        gp.append(p[sel].mean() - p.mean()); g0.append(y[sel].mean() - y.mean())
        q, m, y, p = pool(n, 1.0, cost=1.0); sel = np.argsort(p)[-10:]
        g1.append(y[sel].mean() - y.mean())
    print(f"{n:>7,} candidates: proxy +{np.mean(gp):.2f}, goal +{np.mean(g0):.2f} (cost 0), {np.mean(g1):+.2f} (cost 1)")
```

| Candidates searched | Proxy gain | Goal gain, harmless manipulation | Goal gain, costly manipulation |
| --- | --- | --- | --- |
| 100 | +2.44 | +1.19 | +0.05 |
| 1,000 | +3.73 | +1.87 | -0.00 |
| 10,000 | +4.75 | +2.39 | -0.06 |
| 100,000 | +5.58 | +2.79 | +0.03 |

A thousand-fold larger search more than doubles the reported gain and delivers a little over double the real one when the manipulable part is harmless. When it carries a cost equal to its benefit, the entire gain from a hundred-thousand-candidate search is zero: the optimiser has become very good at finding items whose proxy advantage is exactly the thing the goal penalises. This is the mechanism behind the observation that a metric stops working when it becomes a target, and it sharpens it: what breaks the metric is not the target itself but the amount of search applied to it.

## When Optimisation Goes Backwards

The cost parameter decides whether hard optimisation is merely inefficient or actively harmful.

```python
for frac in (1.0, 0.5, 0.25, 0.1, 0.01):
    row = []
    for cost in (0.0, 0.5, 1.0, 2.0):
        gy = []
        for _ in range(200):
            q, m, y, p = pool(20000, 1.0, cost)
            sel = np.argsort(p)[-int(len(p) * frac):] if frac < 1 else np.arange(len(p))
            gy.append(y[sel].mean() - y.mean())
        row.append(np.mean(gy))
    print(f"top {frac:.0%}: " + "  ".join(f"{v:+.2f}" for v in row))
```

**Goal gain by selection intensity and by how costly the manipulable component is.**

| Selection | Cost 0 | Cost 0.5 | Cost 1.0 | Cost 2.0 |
| --- | --- | --- | --- | --- |
| No selection | 0.00 | 0.00 | 0.00 | 0.00 |
| Top 50% | +0.56 | +0.28 | -0.00 | -0.56 |
| Top 25% | +0.90 | +0.45 | +0.00 | -0.90 |
| Top 10% | +1.24 | +0.62 | +0.00 | -1.24 |
| Top 1% | +1.89 | +0.95 | -0.01 | -1.88 |

With the components of equal variance, a cost of 1 is the break-even point: optimising the proxy as hard as you like moves the goal by exactly nothing, while the proxy reports a gain of 3.77. Above it, every increment of optimisation makes the product worse while the dashboard improves. The team is not being deceived by anyone; it is reading a number that was never measuring the thing it names.

![Goal gain against selection intensity for four levels of cost carried by the manipulable component. At zero cost, harder selection raises the goal at half the rate the proxy reports; at cost one it is flat; above it, the goal falls as the proxy rises.](/assets/images/figures/proxy_goodhart_selection.png){: width="1152" height="672" loading="lazy"}

## A Small Holdout Settles It

None of this is detectable from the proxy, and all of it is detectable from a sample measured on the goal. Because the effect of hard optimisation on the goal is large when it is harmful, the holdout can be tiny.

```python
for n_hold in (50, 200, 800, 3200):
    detect = 0
    for _ in range(400):
        q, m, y, p = pool(20000, 1.0, cost=2.0)
        sel = np.argsort(p)[-200:]
        base = rng.choice(len(y), n_hold, replace=False)
        take = rng.choice(sel, min(n_hold, len(sel)), replace=False)
        diff = y[take].mean() - y[base].mean()
        se = np.sqrt(y[take].var(ddof=1) / len(take) + y[base].var(ddof=1) / len(base))
        detect += (diff / se) < -1.96
    print(f"{n_hold} per group: goal significantly worse in {detect / 400:.0%} of runs")
```

| Holdout size per group | Runs detecting that the goal metric is significantly worse |
| --- | --- |
| 50 | 98% |
| 200 | 100% |
| 800 | 100% |
| 3,200 | 100% |

Fifty items measured on the goal metric, against fifty random ones, catch a harmful optimisation virtually every time. The cost of that check is a rounding error against the cost of running the optimisation, and the reason it is so often skipped is not expense but that the proxy was chosen precisely because measuring the goal was inconvenient.

## Making the Proxy Harder to Exploit

Two structural defences follow directly from the decomposition. The first is to combine proxies whose manipulable components differ, so that raising one does not raise the others.

```python
for k in (1, 2, 4):
    gp, gy = [], []
    for _ in range(200):
        q = rng.normal(0, 1, 20000); y = q + rng.normal(0, 0.5, 20000)
        ms = [rng.normal(0, 1, 20000) for _ in range(k)]
        ps = np.mean([q + m for m in ms], axis=0)
        sel = np.argsort(ps)[-200:]
        gp.append(ps[sel].mean() - ps.mean()); gy.append(y[sel].mean() - y.mean())
    print(f"{k} proxies averaged: proxy +{np.mean(gp):.2f}, goal +{np.mean(gy):.2f}")
```

| Proxies averaged | Proxy gain | Goal gain |
| --- | --- | --- |
| 1 | +3.77 | +1.88 |
| 2 | +3.26 | +2.18 |
| 4 | +2.98 | +2.39 |

Averaging four proxies with independent manipulable parts raises the realised goal gain from 1.88 to 2.39, most of the way to the 2.98 that direct optimisation of the goal would have produced, because the average of independent manipulable components has a quarter of the variance while the shared quality component survives intact. The second defence is to cap or penalise the proxy rather than maximise it: a target that says "at least this much" stops the search before it reaches the region where only $m$ is left to give.

## What to Do

1. **Write the proxy as quality plus the manipulable part**, and estimate the share of its variance that is quality. That share, not the correlation, is the fraction of any reported gain you should expect to be real.
2. **Measure the goal on a holdout** whenever optimisation pressure changes, and size it from the harm you would want to catch; fifty items per group detects a serious regression here.
3. **Expect the proxy-goal correlation to fall among selected items** and do not read the fall as evidence about the world.
4. **Treat search effort as a dial on the risk.** The same proxy is safe under light selection and dangerous under heavy selection; a larger candidate pool raises pressure without any change in the metric.
5. **Combine proxies with different failure modes**, or cap rather than maximise, when the goal cannot be measured often enough to govern the optimiser directly.
6. **Re-derive the proxy when the system changes.** A proxy validated before optimisation says nothing about the regime that optimisation creates.

## References

- Goodhart, C. A. E. (1984). Problems of monetary management: the UK experience. In *Monetary Theory and Practice*. Palgrave Macmillan.
- Manheim, D., & Garrabrant, S. (2018). Categorizing variants of Goodhart's law. *arXiv preprint*, arXiv:1803.04585.
- Prentice, R. L. (1989). Surrogate endpoints in clinical trials: definition and operational criteria. *Statistics in Medicine*, 8(4), 431-440.
- Fleming, T. R., & DeMets, D. L. (1996). Surrogate end points in clinical trials: are we being misled? *Annals of Internal Medicine*, 125(7), 605-613.
- Dmitriev, P., Gupta, S., Kim, D. W., & Vaz, G. (2017). A dirty dozen: twelve common metric interpretation pitfalls in online controlled experiments. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1427-1436.
- Deng, A., & Shi, X. (2016). Data-driven metric development for online controlled experiments: seven lessons learned. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 77-86.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *Proceedings of the 40th International Conference on Machine Learning*, 10835-10866.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
