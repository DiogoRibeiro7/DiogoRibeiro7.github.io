---
permalink: '/time-series/temporal_hierarchies_cross_temporal_reconciliation/'
title: 'Temporal Hierarchies: Reconciling Across Time Granularities'
categories:
- Time Series
tags:
- Time Series
- Forecasting
- Statistical Modeling
- Supply Chain
author_profile: false
seo_title: 'Temporal Hierarchies in Forecasting'
seo_description: 'Monthly forecasts rarely sum to the annual one. Temporal aggregation is a hierarchy too, and reconciling it improves accuracy.'
excerpt: >-
  A hierarchy does not have to be geographic. Aggregating a series over time
  produces the same coherence problem, and the same machinery solves it.
summary: >-
  How temporal aggregation creates a forecasting hierarchy, why forecasts made
  at different granularities disagree, how reconciling across them improves
  accuracy at every level, and why aggregation attenuates noise while
  preserving signal.
keywords:
  - temporal hierarchies
  - temporal aggregation
  - cross-temporal reconciliation
  - THieF
  - multi-granularity forecasting
classes: wide
date: '2026-07-09'
header:
  image: /assets/images/data_science_6.jpg
  og_image: /assets/images/data_science_6.jpg
  overlay_image: /assets/images/data_science_6.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_6.jpg
  twitter_image: /assets/images/data_science_6.jpg
---
Hierarchical forecasting usually means a structure of products or regions. But aggregating a single series **over time** produces a hierarchy too: weekly totals sum to monthly, monthly to quarterly, quarterly to annual. Forecasts made at each granularity will not agree, and the same reconciliation machinery applies.

## The Structure

Take a monthly series. Aggregating pairs of months gives a bi-monthly series, aggregating four gives quarterly, twelve gives annual. Each level is a legitimate view of the same underlying process, and each supports its own forecast.

The relationship is exactly the summing structure of a hierarchy: annual is the sum of quarters, each quarter the sum of months. What varies is that levels here are indexed by *aggregation period* rather than by category.

The name for the combined structure is a **temporal hierarchy**, and the important claim about it is not merely that reconciliation enforces consistency, but that different levels contain genuinely different information.

## Why Levels Disagree Informatively

Temporal aggregation changes the signal-to-noise ratio, and it does so asymmetrically.

Noise is largely independent across periods, so summing $k$ periods grows the noise standard deviation by roughly $\sqrt{k}$. Signal — trend, and any component persisting across periods — grows by $k$. The ratio therefore improves by about $\sqrt{k}$ with aggregation.

This has a concrete consequence. **Trend is easier to see in aggregated data**, because the noise partially cancels. **Seasonality within the aggregation period disappears**, because summing a full cycle removes it. Annual totals show trend cleanly and say nothing about which month is busy; monthly data shows seasonality clearly and buries the trend in noise.

Neither level is more correct. They are complementary, which is the argument for using both rather than choosing.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 240                                        # 20 years of monthly data
t = np.arange(n)
signal = 100 + 0.4 * t + 20 * np.sin(2 * np.pi * t / 12)
y = signal + rng.normal(0, 15, n)

def aggregate(x, k):
    m = len(x) // k
    return x[:m * k].reshape(m, k).sum(axis=1)

print(f"{'level':12}{'periods':>9}{'trend/noise':>13}")
for k, name in [(1, "monthly"), (3, "quarterly"), (12, "annual")]:
    agg = aggregate(y, k)
    # slope estimated by OLS against its own index, relative to residual sd
    idx = np.arange(len(agg))
    slope = np.polyfit(idx, agg, 1)[0]
    resid_sd = (agg - np.polyval(np.polyfit(idx, agg, 1), idx)).std(ddof=2)
    print(f"{name:12}{len(agg):>9}{abs(slope) / resid_sd:>13.3f}")
```

The trend-to-noise ratio rises sharply with aggregation. The annual series has only 20 observations, and the trend in it is far more visible than in the 240 monthly ones — which is why an annual forecast can be more reliable about direction even with a twelfth of the data.

## Reconciling Across Time

Producing forecasts at every aggregation level and reconciling them uses the same projection as the cross-sectional case:

$$
\tilde{y} = S(S^\top W^{-1}S)^{-1}S^\top W^{-1}\hat{y},
$$

with $S$ now encoding temporal rather than categorical summation.

The reported benefit is consistent across studies: reconciled forecasts beat the base forecasts at *every* level, including the level each was optimised for. The monthly forecast improves because the annual forecast contributes information about trend that the monthly data alone estimates poorly, and the annual forecast improves because monthly data pins down the recent level more precisely.

Reconciliation also handles a mundane but common organisational problem. Finance forecasts annually, operations forecast monthly, and the two numbers differ. Temporal reconciliation gives a principled way to align them rather than scaling one to match the other.

## Aggregation Also Changes the Model

A less obvious effect: the appropriate model changes with the aggregation level. Temporal aggregation of an ARIMA process yields a different ARIMA process, and aggregation tends to attenuate autoregressive structure — heavily aggregated series often look close to white noise around a trend.

This is a reason not to force one model family across all levels. Fitting each level independently with automatic model selection, then reconciling, generally works better than imposing a single specification.

For intermittent demand the effect is particularly useful. A daily series that is 90% zeros becomes, at monthly aggregation, a series of modest counts with far fewer zeros — a much easier forecasting problem. Forecasting at the aggregated level and disaggregating is a standard technique for exactly this reason, and it is the temporal-hierarchy idea applied deliberately.

## Practical Notes

**Choose levels that divide evenly.** Weekly data does not aggregate cleanly into months, since months contain 4 or 5 weeks. Use levels with integer ratios — for monthly data, 1, 2, 3, 4, 6 and 12 all divide the year.

**Watch the sample size at the top.** Twenty years of monthly data gives 20 annual observations, which is a thin basis for any model. The top level contributes information about trend but should not be over-trusted.

**Aggregation and the forecast horizon interact.** If decisions are made monthly, a reconciled monthly forecast is what you need; the annual level is an input, not the output. Do not let the improved appearance of the aggregate series tempt you into planning at a granularity nobody acts on.

**Combine with cross-sectional hierarchies carefully.** Reconciling across products *and* time simultaneously — cross-temporal reconciliation — is possible and substantially more complex, since the two structures interact. It is worth attempting only when both hierarchies genuinely matter for decisions.

## References

- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., & Petropoulos, F. (2017). Forecasting with temporal hierarchies. *European Journal of Operational Research*, 262(1), 60-74.
- Kourentzes, N., Petropoulos, F., & Trapero, J. R. (2014). Improving forecasting by estimating time series structural components across multiple frequencies. *International Journal of Forecasting*, 30(2), 291-302.
- Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., & Assimakopoulos, V. (2011). An aggregate-disaggregate intermittent demand approach (ADIDA). *Journal of the Operational Research Society*, 62(3), 544-554.
- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804-819.
