---
permalink: '/time-series/intermittent_demand_forecasting_croston/'
title: 'Intermittent Demand Forecasting: Croston''s Method and Its Successors'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Supply Chain
- Python
author_profile: false
seo_title: 'Intermittent Demand Forecasting with Croston'
seo_description: 'Why standard forecasters fail on sparse, lumpy demand, and how Croston, SBA and TSB handle series that are mostly zero.'
excerpt: >-
  Spare parts and slow-moving stock produce series that are mostly zeros.
  Standard forecasters quietly fail on them; Croston's method and its
  successors are built for exactly this shape of data.
summary: >-
  A practical guide to forecasting intermittent demand: why point-forecast
  metrics mislead on sparse series, how Croston decomposes demand into size
  and interval, the bias that motivated the Syntetos-Boylan approximation, and
  when TSB is the better choice.
keywords:
  - intermittent demand
  - Croston's method
  - spare parts forecasting
  - SBA
  - TSB
classes: wide
date: '2026-08-05'
why_this_exists: >-
  Intermittent demand is often treated with ordinary forecasting tools even
  when the zero-heavy data shape makes those tools operationally misleading.
evidence: >-
  Uses a spare-parts inventory scenario and the Croston/SBA/TSB method family
  to show why demand size and demand timing must be separated.
methodology: >-
  Compares model assumptions, metric behavior, and inventory implications for
  sparse demand rather than listing forecasting methods generically.
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_1.jpg
  twitter_image: /assets/images/data_science_1.jpg
---
Most forecasting material assumes a series that moves smoothly: sales that rise and fall, temperatures that cycle, traffic that peaks predictably. Spare parts inventory looks nothing like that. A part might sell three units in March, nothing until August, then one unit, then nothing for four months.

This is **intermittent demand**, and it breaks the standard toolkit in a way that is easy to miss because the tools keep producing numbers.

## Why Standard Methods Fail Here

Consider a series that is zero on 80% of days. Fit any conventional model minimising squared error and you will get something close to a flat line at the mean — perhaps 0.4 units per day.

That forecast is optimal by RMSE and useless operationally. You cannot stock 0.4 units. Worse, the metric actively rewards the wrong behaviour: because most periods are zero, predicting near-zero everywhere scores well while conveying nothing about *when* demand arrives or *how large* it is when it does.

The deeper problem is that a single number is being asked to describe two separate processes. How often does demand occur? How big is it when it does? A mean conflates them, and two products with the same average demand — one selling 1 unit every week, another selling 52 units once a year — require completely different inventory policies.

## Croston's Decomposition

Croston's insight in 1972 was to stop forecasting demand directly and instead forecast its two components separately.

Let $z_t$ be the size of demand when it occurs, and $p_t$ the number of periods between occurrences. Both are updated by exponential smoothing, but **only in periods where demand actually occurs**:

$$
\hat{z}_t = \alpha y_t + (1 - \alpha)\hat{z}_{t-1}, \qquad
\hat{p}_t = \alpha q_t + (1 - \alpha)\hat{p}_{t-1},
$$

where $q_t$ is the observed interval since the last non-zero period. The forecast of demand per period is their ratio:

$$
\hat{y}_t = \frac{\hat{z}_t}{\hat{p}_t}.
$$

Updating only on non-zero periods is the essential mechanism. Standard exponential smoothing decays the estimate toward zero during every quiet stretch, so a long gap erases what the method knew about demand size. Croston's estimate of size is unaffected by how long it has been waiting.

```python
import numpy as np

def croston(y, alpha=0.1):
    """Classic Croston. Returns the per-period demand rate forecast."""
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y)
    if nz.size == 0:
        return 0.0
    z = y[nz[0]]                       # demand size estimate
    p = float(nz[0] + 1)               # interval estimate
    last = nz[0]
    for t in nz[1:]:
        interval = t - last
        z += alpha * (y[t] - z)        # update size only when demand occurs
        p += alpha * (interval - p)    # update interval likewise
        last = t
    return z / p

demand = np.array([0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 1])
print(f"non-zero periods : {np.count_nonzero(demand)} of {len(demand)}")
print(f"naive mean       : {demand.mean():.3f} per period")
print(f"Croston rate     : {croston(demand):.3f} per period")
```

On this series the naive mean gives 0.55 units per period while Croston gives 0.78 — noticeably higher. Part of that gap is the method weighting recent activity, and part of it is a genuine upward bias that took two decades to be identified, which is the subject of the next section.

The more important difference is structural rather than numerical. Croston carries the size and interval estimates separately, and those are what an inventory policy actually consumes: the mean alone cannot tell you whether to expect one unit weekly or fifty units annually.

## The Bias Nobody Noticed for Twenty Years

Croston's method is biased. Syntetos and Boylan showed in 2001 that the expected value of the ratio $\hat{z}/\hat{p}$ is not the expected demand rate, because the expectation of a ratio is not the ratio of expectations. The method systematically over-forecasts, and the effect grows as demand becomes more intermittent — exactly the regime it was designed for.

The **Syntetos-Boylan Approximation (SBA)** applies a correction factor:

$$
\hat{y}_t^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{\hat{z}_t}{\hat{p}_t}.
$$

The adjustment is small — with $\alpha = 0.1$ it shrinks the forecast by 5% — but it is systematic, and over a large parts catalogue a persistent 5% over-forecast is a real inventory cost.

```python
def sba(y, alpha=0.1):
    """Syntetos-Boylan approximation: Croston with the bias correction."""
    return (1 - alpha / 2) * croston(y, alpha)

for a in (0.05, 0.1, 0.2):
    print(f"alpha={a}: croston={croston(demand, a):.4f}  sba={sba(demand, a):.4f}")
```

## When Demand Stops Entirely

Both Croston and SBA share a blind spot: because they update only when demand occurs, a part that stops selling forever keeps its last forecast indefinitely. The method has no mechanism for concluding that demand has ceased — obsolescence is invisible to it.

The **Teunter-Syntetos-Babai (TSB)** method fixes this by replacing the interval estimate with a *probability* of demand occurring, updated every period including the zeros:

$$
\hat{d}_t = \beta \mathbb{1}[y_t > 0] + (1 - \beta)\hat{d}_{t-1}, \qquad
\hat{y}_t = \hat{d}_t \hat{z}_t .
$$

Because $\hat{d}_t$ decays during quiet periods, a discontinued part's forecast falls toward zero on its own. For catalogues where obsolescence is common — which is most spare parts operations — this matters more than the bias correction.

## Classifying Demand Before Choosing

Syntetos, Boylan and Croston proposed a classification scheme that is still the practical starting point. It uses two quantities: the average inter-demand interval $p$, and the squared coefficient of variation of demand sizes $\text{CV}^2$.

| | $\text{CV}^2 < 0.49$ | $\text{CV}^2 \ge 0.49$ |
|---|---|---|
| **$p < 1.32$** | Smooth | Erratic |
| **$p \ge 1.32$** | Intermittent | Lumpy |

Smooth series can use conventional methods. Intermittent series suit Croston or SBA. Erratic and lumpy series are the hardest, and for these the honest answer is often that point forecasting is the wrong frame entirely.

```python
def classify(y):
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y)
    p = len(y) / len(nz)                          # average interval
    sizes = y[nz]
    cv2 = (sizes.std(ddof=1) / sizes.mean()) ** 2 if len(sizes) > 1 else 0.0
    label = ("smooth" if p < 1.32 and cv2 < 0.49 else
             "erratic" if p < 1.32 else
             "intermittent" if cv2 < 0.49 else "lumpy")
    return p, cv2, label

p, cv2, label = classify(demand)
print(f"interval p = {p:.2f}, CV^2 = {cv2:.2f} -> {label}")
```

## Measure the Right Thing

The metric problem is as important as the method. RMSE and MAE reward forecasts near zero on sparse series, and MAPE is undefined the moment an actual value is zero — which here is most of them.

Better options exist. **MASE** scales error against a naive baseline, so it stays interpretable when the series is mostly zeros. Even so, the more useful question is usually not point accuracy at all but whether the *inventory decision* was right, which depends on the distribution of demand over the replenishment lead time rather than on a single period's forecast.

This is why intermittent demand work tends to end up probabilistic. What stock control needs is $P(\text{demand over lead time} > s)$, and that is a statement about a distribution. Bootstrapping demand over the lead time, or fitting a compound distribution such as a Poisson arrival process with a size distribution attached, answers the operational question directly in a way no point forecast can.

## Practical Guidance

Start by classifying the catalogue, because a single method across thousands of heterogeneous parts is rarely right. Use SBA over classic Croston as the default, since the bias correction costs nothing. Switch to TSB where obsolescence is a live concern. And judge results by service level and holding cost rather than by RMSE, because those are the quantities the forecast exists to inform.

Above all, resist the pull of a smooth model that produces plausible-looking numbers. A flat forecast of 0.4 units on a part that sells four units twice a year is not a mild approximation; it describes a demand pattern that never happens.

## References

- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289-303.
- Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303-314.
- Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503.
- Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606-615.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
