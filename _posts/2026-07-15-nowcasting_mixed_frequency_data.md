---
permalink: '/time-series/nowcasting_mixed_frequency_data/'
title: 'Nowcasting with Mixed-Frequency Data'
categories:
- Time Series
tags:
- Time Series
- Economics
- Forecasting
- Statistical Modeling
author_profile: false
seo_title: 'Nowcasting with Mixed-Frequency Data'
seo_description: 'GDP is quarterly and published late. Daily indicators are available now. Mixed-frequency methods combine them without throwing information away.'
excerpt: >-
  The quantity you care about arrives quarterly and two months late. Related
  indicators arrive daily. Nowcasting is the problem of estimating the present
  from what has already been published.
summary: >-
  How to combine data sampled at different frequencies: why aggregating to the
  lowest frequency discards most of the information, how MIDAS regressions
  weight high-frequency observations parametrically, and how state space
  models handle ragged-edge data naturally.
keywords:
  - nowcasting
  - mixed frequency
  - MIDAS
  - ragged edge
  - high-frequency indicators
classes: wide
date: '2026-07-15'
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_9.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_9.jpg
---
GDP for the current quarter is published weeks after the quarter ends. Meanwhile, electricity consumption, card transactions, freight movements and job postings are all available daily. Nowcasting is the problem of estimating the low-frequency quantity you care about, now, from the high-frequency data that has already arrived.

The name is deliberate: this is not forecasting the future so much as estimating a present that has not yet been measured.

## Two Structural Difficulties

**Mixed frequencies.** The target is quarterly; the indicators are monthly or daily. The naive fix — aggregating everything to quarterly — throws away most of the information, and specifically the most recent part of it.

**The ragged edge.** Different series are published with different lags. At any moment you have complete daily data up to yesterday, monthly data up to last month, and the previous quarter's GDP. The end of the dataset is jagged rather than square, and most estimation routines expect a rectangle.

Any usable method has to handle both.

## Bridge Equations

The simplest approach aggregates high-frequency indicators to the target frequency and regresses:

$$
y_t^{Q} = \alpha + \beta \bar{x}_t^{Q} + \varepsilon_t .
$$

This is easy and it discards timing information. Aggregating three months into one quarterly average treats a strong January and a weak March identically to the reverse, and for nowcasting that distinction is exactly what matters.

Bridge equations remain useful as a baseline and because they are transparent, which counts for something when the output feeds a policy discussion.

## MIDAS

Mixed Data Sampling regression keeps the high-frequency observations distinct and weights them with a parsimonious function:

$$
y_t = \alpha + \beta \sum_{j=0}^{J} w_j(\theta)\, x_{t - j/m} + \varepsilon_t .
$$

The insight is in $w_j(\theta)$. Estimating a free coefficient per high-frequency lag would require estimating dozens of parameters from a handful of quarterly observations. Instead the weights are constrained to a smooth parametric family — commonly the exponential Almon or a beta density — controlled by two or three parameters.

The estimated weight profile is interpretable in its own right: it shows how far back the indicator matters and whether recent observations dominate, which is often the substantive question.

```python
import numpy as np

def beta_weights(J, t1, t2):
    """Normalised beta weights over J high-frequency lags."""
    x = (np.arange(1, J + 1) - 0.5) / J
    w = x ** (t1 - 1) * (1 - x) ** (t2 - 1)
    return w / w.sum()

for name, (a, b) in {"recent-heavy": (1.0, 5.0),
                     "flat":         (1.0, 1.0),
                     "hump":         (2.0, 3.0)}.items():
    w = beta_weights(12, a, b)
    print(f"{name:14} first 4 lags {np.round(w[:4], 3)}  "
          f"share in first quarter of lags: {w[:3].sum():.2f}")
```

The three profiles express different beliefs about how information decays. A recent-heavy profile concentrates almost all weight on the most recent observations; a flat one is equivalent to simple averaging, which is the bridge equation. MIDAS lets the data choose between them rather than imposing the flat case by default.

## State Space Models and Dynamic Factors

The most flexible approach treats the problem as one of missing data.

Put the high- and low-frequency series in a single state space model where the low-frequency observation is simply *unobserved* in the intervening periods. The Kalman filter then handles the ragged edge natively: run the prediction step where nothing is observed, run the update wherever something is. No aggregation, no alignment, no imputation.

For many indicators, a **dynamic factor model** extracts a small number of common factors driving them all, and the target is related to those factors. This addresses the practical reality that nowcasting inputs number in the dozens or hundreds and are heavily correlated with each other.

This combination — a dynamic factor model in state space form, updated by the Kalman filter — is what central banks generally use, and its defining practical feature is that the nowcast updates continuously as each new release arrives.

## News, Not Levels

The most useful output of a nowcasting system is not the number itself but the **decomposition of its revision**.

When the nowcast moves from 1.8% to 2.1%, the informative question is which release caused the change. Formally, the revision decomposes into contributions from the "news" in each release — the difference between what was published and what the model expected to be published.

An indicator that comes in exactly as expected contributes nothing, however important it is in general. A surprise in a minor indicator can move the nowcast substantially. This framing is what makes a nowcast a monitoring tool rather than a periodic number, and it also identifies which releases are worth watching.

## Practical Cautions

**Use vintage data for evaluation.** Macroeconomic series are revised, sometimes heavily. Backtesting against final revised values overstates accuracy badly, because the model is being given numbers that were not available at the time. Real-time vintage databases exist for exactly this reason.

**Publication lags must be respected in the backtest.** Simulating a nowcast for a past date requires reconstructing what had actually been published by then, not simply truncating the dataset.

**More indicators is not automatically better.** Highly correlated inputs add estimation error without adding information, which is the argument for factor extraction over throwing everything into a regression.

**Check against a simple benchmark.** A constant, or a simple autoregression on the target, is the baseline. Nowcasting systems are elaborate, and it is worth confirming that the elaboration is buying accuracy rather than complexity.

## References

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch: mixed data sampling regression models. Working paper, UNC and UCLA.
- Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the real-time data flow. In *Handbook of Economic Forecasting* (Vol. 2A). Elsevier.
- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: the real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.
- Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business cycles based on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4), 427-443.
- Andreou, E., Ghysels, E., & Kourtellos, A. (2013). Should macroeconomic forecasters use daily financial data? *Journal of Business & Economic Statistics*, 31(2), 240-251.
