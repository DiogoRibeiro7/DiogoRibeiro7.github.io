---
permalink: '/statistics/survival_analysis/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2024-05-10'
header:
  image: /assets/images/headers/photo-statistics-survival.jpg
  og_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-survival.jpg
  twitter_image: /assets/images/headers/photo-statistics-survival.jpg
seo_description: "A rigorous introduction to survival analysis covering censoring, truncation, hazards, Kaplan-Meier, Cox models, competing risks, validation, and business event-time applications."
seo_title: "Survival Analysis: Time, Censoring, and Hazard"
seo_type: article
tags:
- Survival Analysis
- Statistics
- Time-to-Event
title: "Survival Analysis: Time, Censoring, and Hazard"
---

Survival analysis models the time until an event while accounting for incomplete observation.

The event can be death, machine failure, churn, recovery, default, employee departure, or any other well-defined transition.

The defining statistical feature is not the subject matter. It is censoring and event time.

## Event time and censoring

Let $T$ denote event time.

For right censoring, we observe

$$
Y=\min(T,C)
$$

and indicator

$$
\Delta=\mathbf 1\{T\le C\},
$$

where $C$ is censoring time.

If $\Delta=0$, we know only that

$$
T>C.
$$

Treating censored observations as though the event occurred at $C$ biases the analysis.

## Independent censoring

Standard methods require assumptions about censoring.

Roughly, conditional on modeled information, censoring should not reveal additional information about the unobserved event time.

If high-risk customers disappear from tracking systems for reasons related to impending churn, ordinary censoring assumptions can fail.

## Survival and hazard

The survival function is

$$
S(t)=P(T>t).
$$

For continuous event times, the hazard is

$$
h(t)
=
\lim_{\Delta t\to0}
\frac{
P(t\le T<t+\Delta t\mid T\ge t)
}{\Delta t}.
$$

The hazard is an instantaneous rate conditional on surviving to time $t$.

It is not a probability and can exceed one.

## Kaplan-Meier estimator

At ordered event times $t_j$, let $d_j$ be the number of events and $n_j$ the number at risk just before $t_j$.

The Kaplan-Meier estimator is

$$
\widehat S(t)
=
\prod_{t_j\le t}
\left(1-\frac{d_j}{n_j}\right).
$$

It is appropriate for estimating survival under independent censoring.

Comparing curves visually is useful, but differences should be reported with uncertainty and a prespecified estimand.

## Cox proportional hazards model

The Cox model is

$$
h(t\mid x)
=
h_0(t)
\exp(x^\top\beta).
$$

The coefficient $e^{\beta_j}$ is a hazard ratio under the model.

A hazard ratio is not a risk ratio, probability ratio, or ratio of median survival times.

## Proportional hazards

The Cox model assumes covariate hazard ratios are constant over time unless time-varying effects are modeled.

Violations can be investigated through Schoenfeld residuals, time interactions, or graphical diagnostics.

Failure of proportional hazards does not imply that survival analysis itself has failed.

Alternatives include:

- time-varying coefficients
- accelerated failure-time models
- restricted mean survival time
- flexible parametric survival models

## Time-dependent covariates

A covariate whose value changes over time is not the same thing as a covariate whose effect changes over time.

A time-dependent covariate is written $x(t)$.

A time-varying coefficient is written $\beta(t)$.

Confusing those concepts leads to incorrect model specification.

## Left truncation

Some individuals enter observation only after being event-free for a period.

For example, a customer dataset extracted today may include only customers who survived long enough to appear in the system.

Risk sets must account for delayed entry.

This is truncation, not censoring.

## Competing risks

If different event types prevent one another, ordinary Kaplan-Meier treatment of competing events as censoring can overestimate cause-specific event probability.

For event type $k$, the cumulative incidence function

$$
F_k(t)
=
P(T\le t,J=k)
$$

is often the relevant quantity.

Cause-specific hazards and subdistribution hazards answer different questions.

## Recurrent events

Customer purchases, hospital admissions, faults, and service calls can recur.

Reducing recurrent events to time-to-first-event discards information.

Counting-process, gap-time, frailty, or multi-state models may be more appropriate.

## Business applications need causal caution

If higher monthly bills are associated with higher churn hazard, it does not follow that lowering prices will reduce churn by the estimated hazard ratio.

Pricing, service quality, customer selection, and tenure may all confound the association.

Survival regression describes event-time association unless a causal identification strategy justifies more.

## Prediction

Survival models can be evaluated using:

- time-dependent discrimination
- Brier scores
- calibration at clinically or operationally relevant horizons
- prediction error curves

The concordance index alone is insufficient because it measures ranking and not probability calibration.

Validation should respect time and entity structure.

## Example in Python

~~~python
from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter


def fit_kaplan_meier(
    durations: pd.Series,
    events: pd.Series,
) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier curve with basic input checks."""
    if len(durations) != len(events):
        raise ValueError("durations and events must have equal length")
    if (durations < 0).any():
        raise ValueError("durations must be non-negative")

    model = KaplanMeierFitter()
    model.fit(durations=durations, event_observed=events)
    return model
~~~

## Conclusion

Survival analysis is not simply binary prediction with a time column.

Its core objects are event time, censoring, risk sets, survival probabilities, and hazards.

Once those are represented correctly, the framework extends naturally to business, engineering, medicine, and reliability.

## References

- Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations.
- Cox, D. R. (1972). Regression Models and Life-Tables.
- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). *Statistical Models Based on Counting Processes*.
- Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*.
