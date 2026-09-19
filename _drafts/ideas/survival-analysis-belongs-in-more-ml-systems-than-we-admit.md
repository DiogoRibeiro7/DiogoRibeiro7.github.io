---
author_profile: false
categories:
- Statistics
classes: wide
title: 'Survival Analysis Belongs in More ML Systems Than We Admit'
excerpt: Churn, equipment failure, disease progression, default, and remaining useful life are all time-to-event problems when censoring and timing matter. Binary classification often throws that structure away.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- survival analysis
- censoring
- churn
- predictive maintenance
- hazard
- time to event
seo_title: 'Survival Analysis Belongs in More ML Systems Than We Admit'
seo_description: 'Why churn, failure, progression, and default are often survival problems rather than binary classification problems.'
seo_type: article
summary: 'When event timing and censoring matter, survival analysis preserves information that fixed-horizon classification discards.'
tags:
- Survival Analysis
- Time to Event
- Machine Learning
why_this_exists: 'Many production ML tasks are framed as binary classification because classifiers are familiar, even when the natural estimand is time to event.'
evidence: 'Censoring examples, fixed-horizon label construction, hazard and survival functions, and competing-risk discussion.'
methodology: 'Start from the event-time estimand, show the information lost by binary labels, and compare fixed-horizon classification with survival modelling.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: When is a binary classification target hiding a time-to-event problem?
Claim: If timing and censoring matter, survival analysis preserves information and aligns better with the decision.
Counterclaim: Fixed-horizon classification is entirely reasonable when the decision is genuinely defined at one horizon and censoring is negligible.
Evidence object: Churn, equipment failure, and disease-progression examples.
Failure case: Survival models can be unnecessary overhead when the event window is fixed and labels are complete.
Reader payoff: Recognise censoring and time structure before constructing binary labels.
Exclusions: A catalogue of survival algorithms.
-->

A customer has not churned.

A machine has not failed.

A patient has not progressed.

Those statements are often converted into negative labels.

That can be wrong.

The event may simply not have happened yet.

This is the central reason survival analysis belongs in more machine-learning systems than we usually admit.

## Time carries information

Suppose two customers have not churned.

One has been observed for three days.

The other has remained active for three years.

A binary label treats them identically.

A survival model does not.

The observation time is part of the evidence.

## Censoring is not a negative outcome

Let (T) be the event time and (C) the censoring time.

We observe

$$
Y = min(T,C)
$$

and

$$
delta = I(T le C).
$$

If (delta=0), the event time is not known.

We only know that

$$
T > C.
$$

Calling the case negative discards that information and can bias the model.

## Fixed-horizon classification answers one specific question

A classifier for “failure within 30 days” estimates something like

$$
P(T le 30 mid X).
$$

That can be useful.

But it says nothing directly about day 31 or day 180.

A survival model estimates a function over time.

The survival function is

$$
S(tmid X)=P(T>tmid X).
$$

The hazard describes instantaneous event risk conditional on survival so far.

These quantities support multiple horizons from one model.

## Churn is naturally time-to-event

Subscription churn is usually not just “will churn / will not churn”.

Businesses care about when churn is likely because retention value and intervention timing depend on it.

A customer likely to churn tomorrow and one likely to churn in a year should not receive identical treatment.

Time-to-event modelling expresses that distinction directly.

## Predictive maintenance has the same structure

Equipment that has not failed by the end of the study is censored.

Treating it as a permanent non-failure can teach the model the wrong lesson.

Survival methods also align naturally with remaining-useful-life and hazard-based maintenance decisions.

This connects prediction to intervention timing.

## Disease progression is another obvious case

Clinical studies routinely use survival methods because follow-up differs across patients.

Machine-learning systems sometimes discard that structure by constructing binary labels such as progression within one year.

That may be appropriate for one clinical decision.

It should not be the default merely because binary classifiers are convenient.

## Competing events complicate simple labels

A customer can churn for different reasons.

A patient can die before experiencing another endpoint.

A machine can be replaced before it fails.

These are competing risks.

A binary target that ignores them can estimate a quantity that does not correspond to the real process.

Survival analysis has explicit tools for this structure.

## Time-varying covariates matter

Risk changes.

Usage declines.

Sensor vibration rises.

Treatment changes.

A baseline classifier may freeze features at one time and miss this evolution.

Survival frameworks can incorporate time-varying covariates when the data support it.

Again, the point is not that every system needs maximum complexity.

It is that the modelling language can represent the actual decision process.

## Evaluation should respect censoring

Standard accuracy metrics are not automatically valid under censoring.

Survival models use quantities such as concordance, time-dependent AUC, Brier scores, calibration at specific horizons, and likelihood-based criteria.

The metric should match the estimand.

A high C-index does not guarantee good calibration.

A good binary AUC at one horizon does not imply useful risk estimates across time.

## Classification can still be the right choice

If the operational decision is genuinely:

> intervene if failure probability within the next seven days is high,

then a seven-day classifier may be exactly the right tool.

The point is not to replace every classifier with a Cox model.

It is to notice when the label construction has discarded useful time information.

## The target should come from the decision

Ask:

- Does event timing matter?
- Are some observations censored?
- Do follow-up times differ?
- Are there competing events?
- Does the intervention depend on horizon?

If the answer is yes, the task is at least partly a survival problem.

## Conclusion

Many ML problems are time-to-event problems wearing binary labels.

Churn, equipment failure, disease progression, and default all contain information about when the event occurs and how long an observation remains event-free.

Survival analysis preserves that structure.

Binary classification is not wrong.

It is a special case created by choosing a horizon and discarding the rest.

## References

- Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *JASA*. 1958.
- Cox DR. Regression models and life-tables. *JRSS B*. 1972.
- Harrell FE. *Regression Modeling Strategies*. Springer.
- Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *JASA*. 1999.
