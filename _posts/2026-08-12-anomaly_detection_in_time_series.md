---
permalink: '/time-series/anomaly_detection_in_time_series/'
title: 'Anomaly Detection in Time Series'
categories:
- Time Series
tags:
- Anomaly Detection
- Time Series
- Model Monitoring
- Signal Processing
author_profile: false
seo_title: 'Anomaly Detection in Time Series'
seo_description: 'A point that is normal in isolation can be anomalous in context. Point, contextual and collective anomalies, and how to catch each.'
excerpt: >-
  Outlier detection asks whether a value is unusual. Time series anomaly
  detection asks whether it is unusual *now*, which is a different and harder
  question.
summary: >-
  The three kinds of time series anomaly and the methods that catch them:
  residual-based detection after removing trend and seasonality, why
  point-wise precision and recall are the wrong metrics for range-based
  events, and how to avoid a detector that alerts constantly and is therefore
  ignored.
keywords:
  - time series anomaly detection
  - contextual anomaly
  - collective anomaly
  - residual analysis
  - range-based evaluation
classes: wide
date: '2026-08-12'
why_this_exists: >-
  Time-series anomalies are contextual and range-based; generic point outlier
  detection misses what operators actually need to investigate.
evidence: >-
  Uses temperature, server traffic, residual, contextual, and collective
  anomaly examples.
methodology: >-
  Classifies anomalies by failure mode first, then maps each class to suitable
  detection and evaluation strategies.
reviewed_at: '2026-08-16'
header:
  image: /assets/images/data_science_9.webp
  og_image: /assets/images/data_science_9.webp
  overlay_image: /assets/images/data_science_9.webp
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.webp
  twitter_image: /assets/images/data_science_9.webp
---
A temperature of 28°C is unremarkable in July and alarming in January. A server handling 400 requests per second is normal at midday and suspicious at 4am. Generic outlier detection asks whether a value is unusual; time series anomaly detection asks whether it is unusual *given when it occurred*, which is a harder question.

## Three Kinds of Anomaly

The distinction matters because different methods catch different kinds.

**Point anomalies** are single values far from anything plausible — a sensor reporting 10,000 when it normally reports 20. These are the easy case, catchable by a fixed threshold.

**Contextual anomalies** are values that are normal in isolation but wrong for their context. The January 28°C reading is the canonical example: within the annual range, impossible for the season. Detecting these requires modelling the context first.

**Collective anomalies** are sequences where no individual point is unusual but the pattern is. A heart rate holding perfectly flat at 70 bpm is anomalous precisely because real signals vary; every reading is normal, the absence of variation is not. Level shifts, flatlines and frequency changes all fall here.

Most production incidents are contextual or collective. Most naive detectors only catch point anomalies, which is why they miss the events that matter.

## Residual-Based Detection

The most reliable general approach is to model what is expected and inspect what is left over. Decompose the series, remove trend and seasonality, and detect anomalies in the residual.

This converts contextual anomalies into point anomalies. Once seasonality is removed, the January 28°C reading becomes a large positive residual, which a simple threshold catches.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

rng = np.random.default_rng(0)
n = 730
idx = pd.date_range("2023-01-01", periods=n, freq="D")
t = np.arange(n)

# strong weekly cycle: weekdays busy, weekends quiet
y = 100 + 25 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)

# contextual anomaly: a weekday-level value landing on a quiet day
quiet_day = int(np.argmin(y[:60]))            # a trough early in the series
y[quiet_day] = 128.0
series = pd.Series(y, index=idx)

res = STL(series, period=7, robust=True).fit()
resid = res.resid

# robust threshold: median absolute deviation, not standard deviation
med = resid.median()
mad = (resid - med).abs().median()
score = 0.6745 * (resid - med) / mad          # modified z-score
flagged = score.abs() > 3.5

print(f"raw value on the quiet day : {series.iloc[quiet_day]:.1f}")
print(f"overall range of the series: "
      f"{series.min():.1f} to {series.max():.1f}")
print(f"modified z of its residual : {score.iloc[quiet_day]:.1f}")
print(f"total points flagged       : {flagged.sum()} of {len(series)}")
```

The injected value of 128 sits comfortably inside the series' overall range of 68 to 134, so no threshold on the raw values could catch it. Against its own decomposed context it scores a modified z of 30.

Two details make this work. The threshold uses the **median absolute deviation** rather than the standard deviation, because the anomalies themselves inflate a standard deviation and mask their own detection. And STL is fitted with `robust=True` so a large spike does not distort the seasonal estimate for every subsequent cycle.

Note also what else the run reports: 34 points flagged out of 730, when only one was injected. The other 33 are ordinary noise crossing a fixed threshold. On a real series that is roughly one alert a fortnight for nothing, which is precisely the volume problem discussed below — and it is a property of the threshold, not a failure of the decomposition.

## Other Approaches Worth Knowing

**Forecast-based.** Predict the next value, flag when the actual falls outside the prediction interval. Natural for online detection and gives a probabilistic score directly. Its weakness is that a good model may *learn* a recurring anomaly and stop flagging it.

**Distance and density based.** Embed subsequences as vectors — a sliding window of length $w$ becomes a point in $\mathbb{R}^w$ — then apply k-nearest-neighbours or Local Outlier Factor. This catches collective anomalies naturally, since unusual *shapes* become distant points. The matrix profile is an efficient modern implementation of this idea.

**Change-point detection.** Where the concern is a persistent shift in level or variance rather than a transient spike, change-point methods are the correct framing. A level shift is not an outlier; it is a new regime.

**Reconstruction based.** Train an autoencoder on normal behaviour and flag windows it reconstructs poorly. Flexible for multivariate series, and demanding in data and tuning.

## Evaluating a Detector Properly

This is where most time series anomaly detection work goes wrong.

Anomalies in time series are usually **ranges**, not points. An outage lasts twenty minutes. If a detector flags one minute inside that window, has it succeeded? Point-wise precision and recall say it got 1 of 20 and scores terribly, when operationally it did exactly what was needed — it raised the alarm.

The opposite failure is equally common: a widely used "point-adjust" convention marks the entire range correct if any single point within it is flagged. This inflates scores so severely that a random detector can appear excellent, and several published results have been shown to rest on it.

Range-aware measures that credit detection, overlap and timeliness separately are the better answer. Whatever you use, **report detection latency**: an anomaly found six hours late may be worthless regardless of precision.

The base rate deserves equal attention. Anomalies are rare by definition, so accuracy is meaningless — a detector that never fires scores 99.9% on a series with 0.1% anomalies. Precision at a fixed alert budget ("of the 20 alerts we can investigate daily, how many were real") maps far better onto how the system is actually used.

## The Failure Mode That Matters Most

A detector that fires constantly is functionally identical to no detector, because the alerts get ignored. Alert fatigue is the dominant practical failure, not detection accuracy.

Several habits help. Set thresholds from the alert volume the team can absorb rather than from a statistical convention. Require persistence — several consecutive anomalous points before alerting — since isolated spikes are usually noise. Suppress alerts during known events such as deployments and holidays, which are anomalous but expected. And keep a feedback path so confirmed false positives adjust the threshold rather than being silently tolerated.

## A Practical Order of Work

Establish what "normal" means before trying to detect departures from it: decompose the series, look at the residuals, and check whether they behave like noise. If structure remains in the residuals, the model is incomplete and every anomaly score derived from it will be unreliable.

Then start with the simplest detector that addresses the anomaly type you actually care about, measure it against a labelled sample rather than by eye, and tune the threshold against alert capacity. Sophistication in the detector rarely compensates for a poor model of normal behaviour.

## References

- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: a survey. *ACM Computing Surveys*, 41(3), 1-58.
- Blázquez-García, A., Conde, A., Mori, U., & Lozano, J. A. (2021). A review on outlier/anomaly detection in time series data. *ACM Computing Surveys*, 54(3), 1-33.
- Kim, S., Choi, K., Choi, H.-S., Lee, B., & Yoon, S. (2022). Towards a rigorous evaluation of time-series anomaly detection. *Proceedings of AAAI*, 36(7), 7194-7201.
- Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., & Gottschlich, J. (2018). Precision and recall for time series. *Advances in Neural Information Processing Systems*, 31.
- Yeh, C.-C. M., et al. (2016). Matrix profile I: all pairs similarity joins for time series. *Proceedings of ICDM*, 1317-1322.
