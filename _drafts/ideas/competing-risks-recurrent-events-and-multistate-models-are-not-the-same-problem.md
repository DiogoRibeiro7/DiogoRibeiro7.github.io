---
author_profile: false
categories:
- Statistics
classes: wide
title: 'Competing Risks, Recurrent Events, and Multi-State Models Are Not the Same Problem'
excerpt: 'Time-to-event analysis changes when an event can recur, when several event types compete, or when subjects move through intermediate states before the final outcome.'
keywords:
- survival analysis
- competing risks
- recurrent events
- multi-state models
- joint models
seo_title: 'Competing Risks, Recurrent Events, and Multi-State Models'
seo_description: 'A structured draft on competing risks, recurrent events, multi-state models, and joint longitudinal-survival analysis.'
seo_type: article
summary: 'A planned article showing how different event-history data structures imply different estimands, risk sets and likelihoods, and why ordinary Kaplan-Meier reasoning can fail outside the single terminal-event setting.'
tags:
- Survival Analysis
- Competing Risks
- Recurrent Events
- Multi-State Models
why_this_exists: 'Many applied survival analyses use one event indicator and one clock even when the data-generating process contains competing causes, repeated events or intermediate transitions.'
evidence: 'Exact cumulative-incidence calculations, recurrent-event simulations, multi-state transition probabilities and standard event-history methodology.'
methodology: 'Begin from counting-process notation and risk sets, then separate competing-risk, recurrent-event, multi-state and joint longitudinal-event estimands through worked examples.'
---

<!--
Development contract
Question: How should survival analysis change when the event process is more complex than one terminal event?
Claim: Competing risks, recurrent events and multi-state processes require different probability objects and risk sets, so forcing them into a single-event survival model changes the scientific question.
Counterclaim: Simpler models can still be adequate when the estimand is deliberately narrow and competing processes are irrelevant to the decision.
Evidence object: One competing-risk example where Kaplan-Meier overstates cumulative incidence, one recurrent-event process, and one three-state illness-death model.
Failure case: Treating competing events as independent censoring, collapsing repeated events into time to first event without acknowledging the estimand change, or interpreting cause-specific hazards as probabilities.
Reader payoff: Match the event-history model to the scientific process before choosing software or a hazard model.
Exclusions: Repeating introductory Kaplan-Meier and Cox material already covered elsewhere.
-->

## Mathematical spine

For competing risks with cause-specific hazards $\lambda_k(t)$, define

$$
S(t)
=
\exp\left[
-\int_0^t
\sum_k\lambda_k(u)\,du
\right]
$$

and

$$
F_k(t)
=
\int_0^t
S(u^-)\lambda_k(u)\,du.
$$

Use this to show why censoring competing events and applying $1-\hat S_k(t)$ generally overstates the actual probability of cause $k$.

For recurrent events, introduce a counting process $N_i(t)$ and distinguish calendar-time, gap-time, marginal-rate and frailty formulations. For multi-state models, define transition intensities $\lambda_{rs}(t)$ and transition probabilities $P_{rs}(s,t)$.

The final section should connect longitudinal biomarkers and event times through shared-parameter joint models rather than treating the longitudinal trajectory as an error-free time-varying covariate.

## Reproducibility plan

Simulate one two-cause process, one recurrent-event process and one illness-death model. Plot cumulative incidence and transition probabilities.

## Sources to develop

Aalen, O. O., Borgan, Ø., & Gjessing, H. K. (2008). *Survival and Event History Analysis*.

Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). *Statistical Models Based on Counting Processes*.

Putter, H., Fiocco, M., & Geskus, R. B. (2007). Competing risks and multi-state models.

Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event Data*.
