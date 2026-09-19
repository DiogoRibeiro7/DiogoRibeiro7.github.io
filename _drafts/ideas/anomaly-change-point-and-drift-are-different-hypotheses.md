---
author_profile: false
categories:
- Time Series
classes: wide
title: 'Anomaly, Change Point, and Drift Are Different Hypotheses'
excerpt: A point anomaly, structural break, gradual distribution shift, and performance degradation can produce similar alerts while requiring different models and interventions.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- anomaly detection
- change point
- concept drift
- monitoring
- time series
seo_title: 'Anomaly, Change Point, and Drift Are Different Hypotheses'
seo_description: 'Why production monitoring should distinguish isolated anomalies, structural breaks, gradual drift, and performance degradation instead of collapsing them into one alert.'
seo_type: article
summary: 'A formal distinction among anomalies, change points, drift, and model degradation, with identical-looking alert patterns that imply different operational responses.'
tags:
- Change Point Detection
- Monitoring
- Anomaly Detection
why_this_exists: 'Monitoring systems often use drift as a generic label for any unusual signal. The resulting alerts are hard to interpret because they collapse different statistical hypotheses.'
evidence: 'Synthetic mean-shift, point-outlier, variance-change, and gradual-drift examples together with standard change-point and drift literature.'
methodology: 'Define each hypothesis at the data-generating level, then compare detectors and operational decisions under controlled synthetic examples.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: What exactly changed when a monitoring system raises an alert?
Claim: Point anomalies, structural breaks, gradual drift, and model degradation are different hypotheses and should not share one generic detector or response.
Counterclaim: One generic alert layer can still be operationally useful if it is treated as triage rather than diagnosis.
Evidence object: Four synthetic processes that produce superficially similar monitoring spikes but require different interpretations.
Failure case: The taxonomy is not exhaustive; periodicity changes, dependence changes, and label shift add further cases.
Reader payoff: Turn an alert into a question about the data-generating process before deciding whether to retrain or intervene.
Exclusions: Ranking commercial monitoring products.
-->

A monitoring dashboard goes red.

That event is often described as drift.

The label is too broad.

A single outlier, a permanent level shift, a slow change in the input distribution, a variance increase, a sensor fault and a genuine decline in predictive performance can all trigger an alert. They are not the same statistical event.

If the hypothesis is unclear, the operational response becomes guesswork.

## A point anomaly is local

A point anomaly concerns one or a small number of observations that are surprising relative to the local model.

For example,

$$
y_t = mu + epsilon_t
$$

for most (t), with one observation

$$
y_	au = mu + 8sigma.
$$

The process before and after (	au) may be unchanged.

If the outlier is due to sensor corruption, retraining the model would be the wrong response.

The relevant action may be data validation, robust estimation, or simply no action.

## A change point is structural

A change point asserts that the data-generating process itself changes at an unknown time.

A simple mean-shift model is

$$
y_t =
egin{cases}
mu_1 + epsilon_t, & t le 	au,\
mu_2 + epsilon_t, & t > 	au.
end{cases}
$$

Now the important quantity is (	au), not the extremeness of one observation.

The same visible jump can therefore represent either one bad reading or a permanent regime change.

The distinction cannot be made from amplitude alone.

Persistence is evidence.

## Drift can be gradual

Gradual drift is different again.

Suppose

$$
mu_t = mu_0 + eta t.
$$

There may be no single break point.

A detector designed for abrupt change can either trigger late or produce repeated weak alarms.

The operational question also differs.

A gradual seasonal shift may be expected and harmless. A gradual measurement bias may be dangerous. A population shift may matter only if the model's conditional relationship becomes wrong.

Calling all of these drift does not help.

## Variance change is not mean change

A system can keep the same mean while becoming much less stable.

For example,

$$
y_t sim N(0,1)
$$

before a change and

$$
y_t sim N(0,9)
$$

afterwards.

A mean-based detector can miss the event entirely.

Yet for a quality-control or risk system the variance increase may be more important than a mean shift.

Monitoring therefore needs to make explicit which feature of the distribution is under surveillance.

A one-number drift score hides that question.

## Data drift is not performance degradation

A predictive model can remain accurate under substantial covariate shift if the conditional relationship (P(Ymid X)) remains stable.

Conversely, performance can deteriorate even when marginal feature distributions look unchanged.

This means that an input-distribution alarm is not a diagnosis of model failure.

At best it is evidence that the operating environment changed.

Whether that change matters depends on the model and target.

## The same chart can support different hypotheses

Imagine a rolling statistic that suddenly exceeds a threshold.

That picture could arise because:

- one extreme observation entered the window;
- a permanent mean shift occurred;
- the variance increased;
- the sampling process changed;
- a seasonal component moved phase;
- a missingness mechanism changed.

The chart is not the hypothesis.

A monitoring system should therefore attach alerts to explicit candidate explanations rather than to generic colour states.

## Detection and intervention should be separate

An alert asks whether something deserves investigation.

An intervention asks what action should follow.

Those are different decisions.

For a point outlier, the action may be to quarantine one record.

For a structural break, the action may be to refit or recalibrate.

For gradual population drift, the action may be to collect labels and monitor performance.

For a broken sensor, the action may be hardware maintenance.

A system that routes all alerts to retraining is treating retraining as a ritual rather than a decision.

## Use negative controls

Monitoring pipelines benefit from synthetic negative controls.

A detector should be tested on:

- isolated outliers with no structural change;
- variance changes with stable mean;
- gradual drift;
- abrupt mean shifts;
- seasonal changes;
- random missing bursts.

If the detector labels all of them identically, the alert semantics are weak.

The question is not whether the detector is sensitive.

It is whether the alert supports the operational interpretation attached to it.

## Conclusion

Anomaly, change point and drift are different hypotheses about what generated the data.

A point anomaly is local. A change point is structural. Drift can be gradual. Variance can change without the mean changing. Performance degradation is a property of the model-task system, not merely of the input distribution.

The first job of monitoring is therefore not to produce a red light.

It is to preserve the distinction among the hypotheses that could have produced it.

## References

- Basseville M, Nikiforov IV. *Detection of Abrupt Changes: Theory and Application*. Prentice Hall.
- Truong C, Oudre L, Vayatis N. Selective review of offline change point detection methods. *Signal Processing*. 2020.
- Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A survey on concept drift adaptation. *ACM Computing Surveys*. 2014.
- Chandola V, Banerjee A, Kumar V. Anomaly detection: a survey. *ACM Computing Surveys*. 2009.
