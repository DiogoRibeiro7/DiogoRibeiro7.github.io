---
permalink: '/time-series/global_vs_local_models_time_series/'
title: 'Global vs Local Models in Time Series Forecasting'
categories:
- Time Series
tags:
- Forecasting
- Time Series
- Machine Learning
- Model Evaluation
author_profile: false
seo_title: 'Global vs Local Models for Forecasting'
seo_description: 'One model per series, or one model across all of them? What the M-competitions changed about forecasting many related series.'
excerpt: >-
  The traditional approach fits one model per series. Modern practice often
  fits a single model across thousands of them, and usually wins.
summary: >-
  The shift from local to global forecasting models: why cross-learning across
  related series works, what it demands in terms of scaling and feature
  design, when a local model is still the right choice, and what the M4 and M5
  competitions actually demonstrated.
keywords:
  - global forecasting models
  - cross-learning
  - M4 competition
  - M5 competition
  - many related series
classes: wide
date: '2026-08-13'
why_this_exists: >-
  Forecasting many related series is no longer only a one-model-per-series
  problem, and the local-versus-global choice changes the whole workflow.
evidence: >-
  Uses SKU/store examples and lessons from large forecasting competitions to
  explain cross-learning and scaling requirements.
methodology: >-
  Compares local and global models by sample size, feature design,
  heterogeneity, evaluation, and operational maintenance.
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_6.jpg
  og_image: /assets/images/data_science_6.jpg
  overlay_image: /assets/images/data_science_6.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_6.jpg
  twitter_image: /assets/images/data_science_6.jpg
---
Classical forecasting fits one model per series. Each SKU, sensor or store gets its own ARIMA, with its own parameters estimated from its own history. This is the **local** approach, and it was the default for fifty years.

The **global** approach fits a single model across all series at once, learning shared patterns and distinguishing series through features rather than through separate parameter sets. It sounds worse — surely a bespoke model beats a generic one — and it usually wins.

## Why Cross-Learning Works

A local model for a single retail SKU has perhaps 104 weekly observations and needs to estimate several parameters from them. That is a small sample, and the resulting estimates are noisy.

A global model trained on 10,000 similar SKUs has 1,040,000 observations. It cannot tailor itself to any one series as precisely, but it can learn patterns that no individual series demonstrates clearly: what a promotion does to demand, how a product ramps after launch, what December looks like. Each series contributes evidence about the shared structure.

The trade-off is bias against variance. The local model is unbiased for its own series and high-variance because of the small sample. The global model imposes structure that may not fit every series exactly, but estimates that structure far more precisely. Where series are genuinely related, the second wins comfortably.

This is also the only practical option at scale. Fitting, tuning and monitoring 100,000 individual models is an operational burden that a single global model avoids entirely.

## What the Competitions Showed

The M4 competition (2018, 100,000 series) was won by a hybrid combining exponential smoothing with a globally trained recurrent network. Pure machine learning entries that treated each series independently performed poorly; the successful approaches shared parameters across series.

The M5 competition (2020, Walmart hierarchical sales) was dominated by gradient boosting trained globally across all series, with series identity supplied as features. Notably, the winning approaches were not exotic architectures — they were LightGBM with careful feature engineering and thoughtful validation.

The lesson is narrower than "machine learning wins". It is that **sharing information across related series beats isolating them**, and that the method used to do the sharing matters less than the decision to share.

## Making a Global Model Work

Three requirements do most of the work.

**Scale the series.** Series in a global model differ by orders of magnitude — one store sells thousands of units, another sells tens. Without normalisation the model devotes its capacity to the largest series and ignores the rest. Dividing each series by its own mean, or by a rolling level estimate, puts them on a comparable footing. Forecasts are rescaled back afterwards.

**Give the model a way to tell series apart.** Since one parameter set serves all series, differences must come through features: static attributes (category, region, price tier), and dynamic ones (recent level, volatility, trend). Categorical series identifiers can be included, though on very large catalogues embeddings or aggregated attributes generalise better than raw IDs.

**Validate across series and time.** A random split leaks in two directions at once — across time, and across series if the same series appears in both sets. Hold out a time period for all series, and additionally hold out entire series to test generalisation to items the model has never seen.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_series, T = 200, 120

# many related series: shared seasonality, different levels and noise
frames = []
for i in range(n_series):
    level = rng.uniform(20, 2000)
    t = np.arange(T)
    y = level * (1 + 0.25 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, level * 0.05, T)
    frames.append(pd.DataFrame({"series": i, "t": t, "y": y}))
df = pd.concat(frames, ignore_index=True)

# scaling is what makes one model serve series of very different magnitude
scale = df.groupby("series")["y"].transform("mean")
df["y_scaled"] = df["y"] / scale

print("raw range of series means  :",
      f"{df.groupby('series')['y'].mean().min():.0f} to "
      f"{df.groupby('series')['y'].mean().max():.0f}")
print("scaled range of series means:",
      f"{df.groupby('series')['y_scaled'].mean().min():.2f} to "
      f"{df.groupby('series')['y_scaled'].mean().max():.2f}")
```

Before scaling, the series means span two orders of magnitude and a squared-error objective would be dominated almost entirely by the largest. After scaling they are directly comparable, and every series contributes to the fit.

## When Local Is Still Right

Global models are not universally better, and three situations favour local ones.

**Few series.** With five series and long histories, there is little to cross-learn from and no operational burden to avoid. Fit them individually.

**Heterogeneous series.** Global models assume the series share structure. Forecasting electricity demand, website traffic and cattle prices with one model pools things that have nothing in common, and the imposed structure becomes pure bias.

**Interpretability requirements.** A local ARIMA states its parameters plainly. A global gradient boosting model over hundreds of features does not, and where a regulator or a planner needs to understand *why* the forecast moved, that matters.

A middle path is often best: cluster series by behaviour, then fit one global model per cluster. This keeps most of the cross-learning benefit while avoiding the pooling of genuinely unrelated series.

## Practical Cautions

**Cold-start is a real advantage.** A global model can forecast a brand-new series with no history at all, using only its static attributes — something no local model can do. For retailers launching products continuously, this alone can justify the approach.

**Watch which series dominate.** Even after scaling, series with more observations or higher variance exert more influence. If a handful of large series drive the loss, the model may forecast the long tail badly while reporting good aggregate error.

**Report accuracy by segment.** A single overall MASE hides the case where the model is excellent on the top 100 series and worse than a seasonal naive on the remaining 9,900. Break results down by volume decile, and compare each segment against its own baseline.

The underlying shift is worth stating plainly: forecasting many related series is a different problem from forecasting one series well, and the methods that suit it borrow more from supervised learning than from classical time series analysis.

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364.
- Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: locality and globality. *International Journal of Forecasting*, 37(4), 1632-1653.
- Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting*, 36(1), 75-85.
- Januschowski, T., et al. (2020). Criteria for classifying forecasting methods. *International Journal of Forecasting*, 36(1), 167-177.
