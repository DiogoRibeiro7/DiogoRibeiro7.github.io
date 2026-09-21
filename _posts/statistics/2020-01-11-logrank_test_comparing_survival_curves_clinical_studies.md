---
permalink: '/statistics/logrank_test_comparing_survival_curves_clinical_studies/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-11'
excerpt: The log-rank test compares event incidence over risk sets through time. Proportional hazards makes it especially powerful, but it is not the validity assumption usually claimed in textbook summaries.
header:
  image: /assets/images/headers/photo-statistics-survival.jpg
  og_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-survival.jpg
  twitter_image: /assets/images/headers/photo-statistics-survival.jpg
keywords:
- log-rank test
- survival analysis
- censoring
- Kaplan-Meier
- proportional hazards
seo_description: A rigorous explanation of the log-rank test, its risk-set construction, censoring assumptions, proportional-hazards interpretation, and alternatives for crossing survival curves.
seo_title: 'Log-Rank Test: What It Tests and When It Loses Power'
seo_type: article
summary: A mathematical guide to the log-rank test that distinguishes validity assumptions from the proportional-hazards condition under which the test is especially efficient.
tags:
- Survival Analysis
- Hypothesis Testing
- Clinical Statistics
title: 'Log-Rank Test: What It Tests and When It Loses Power'
---

The log-rank test is often described as a nonparametric test of whether two survival curves are equal. That is broadly correct. The usual explanation then adds that the log-rank test assumes proportional hazards. That statement needs qualification. Proportional hazards is the setting in which the ordinary log-rank weighting is especially natural and powerful. Non-proportional hazards, such as crossing survival curves, can severely reduce power and make a single hazard-ratio summary misleading.

But proportional hazards is not the same kind of validity condition as independent censoring.

## Survival and hazard functions

Let $T$ be an event time. The survival function is

$$
S(t)
=
P(T>t).
$$

For a continuous event-time distribution, the hazard is

$$
h(t)
=
\lim_{\Delta t\downarrow0}
\frac{
P(t\le T<t+\Delta t\mid T\ge t)
}{
\Delta t
}.
$$

The log-rank test compares groups by repeatedly contrasting the observed number of events with the number expected under a common event-rate structure among individuals currently at risk.

## Risk sets and expected events

Consider two groups. At ordered event time $t_j$, let

$$
Y_{1j},\quad Y_{0j}
$$

be the numbers at risk, and let

$$
d_{1j},\quad d_{0j}
$$

be the numbers of observed events. Define

$$
Y_j=Y_{1j}+Y_{0j},
\qquad
d_j=d_{1j}+d_{0j}.
$$

Under the null of equal event hazards at that event time, the expected number of group-1 events is

$$
E_{1j}
=
d_j\frac{Y_{1j}}{Y_j}.
$$

The log-rank numerator accumulates

$$
U
=
\sum_j
(d_{1j}-E_{1j}).
$$

A variance estimate $V$ is constructed from the same risk sets, giving

$$
Z
=
\frac{U}{\sqrt V},
$$

which is asymptotically standard normal under the null. For more than two groups, the vector form leads to a chi-square statistic.

## What the null hypothesis means

A common formulation is equality of survival distributions:

$$
H_0:
S_1(t)=S_0(t)
\quad
\text{for all }t.
$$

For continuous distributions, equality of survival functions is equivalent to equality of hazard functions over the relevant time range. The test is omnibus in the sense that systematic differences in event incidence can accumulate over time. But its weighting is not equally sensitive to every possible alternative.

## Why proportional hazards matters

Suppose

$$
h_1(t)
=
h_0(t)\exp(\beta).
$$

The hazard ratio

$$
\exp(\beta)
$$

is then constant over time. Against alternatives close to this proportional-hazards form, the standard log-rank test has strong efficiency properties. This is why proportional hazards appears so often in discussions of the test. Now consider crossing hazards:

- treatment is beneficial early;
- harmful later;
- the two effects partly cancel in the accumulated log-rank score.

The log-rank test can then have low power even when the survival curves differ substantially. That is a sensitivity problem, not proof that the p-value is automatically invalid.

## Independent censoring is more fundamental

Right censoring removes an individual from future risk sets. For standard survival inference, censoring must be independent of the future event process in the appropriate conditional sense. Informally, among individuals with the same relevant history, those censored at a given time should not have systematically different future event prospects from those remaining under observation.

If high-risk patients preferentially drop out for reasons not captured in the analysis, the observed risk sets can become unrepresentative. That can bias both Kaplan-Meier estimates and group comparisons. This is a deeper problem than non-proportional hazards.

## Administrative censoring is usually benign

If a trial stops on a fixed calendar date, some participants are censored simply because they entered later. That is administrative censoring. When study entry and the administrative end date are appropriately handled, this mechanism is often plausibly non-informative. Loss to follow-up is more difficult because the reason for leaving may be related to health status.

The censoring mechanism should therefore be described, not merely labeled.

## A significant test does not estimate the size of the effect

A log-rank p-value answers a global comparison question. It does not tell us the absolute survival difference at 1 year, the restricted mean survival time difference, the median survival difference, the hazard ratio, or whether the difference is clinically important. Kaplan-Meier curves and effect estimates should accompany the test. A survival-analysis report that gives only a p-value throws away most of the information.

## Cox regression is not a remedy for non-proportional hazards

A common recommendation is to use Cox regression when proportional hazards fails. That is backwards. The standard Cox proportional-hazards model assumes

$$
h(t\mid X)
=
h_0(t)\exp(X^\top\beta),
$$

which imposes time-constant hazard ratios for time-fixed coefficients. If hazards are non-proportional, an ordinary Cox model with a single coefficient can be misleading. Possible responses include time-varying coefficients, stratified Cox models for nuisance factors, piecewise effects, flexible parametric survival models, restricted mean survival time comparisons, or weighted log-rank tests chosen for the scientific alternative.

The method should reflect the time pattern of the effect.

## Weighted log-rank tests

A general weighted statistic has the form

$$
U_w
=
\sum_j
w(t_j)
(d_{1j}-E_{1j}).
$$

The ordinary log-rank test uses approximately equal weight across event times in its score construction. Other choices emphasize early or late events. Fleming-Harrington weights can be written using the pooled Kaplan-Meier estimate $\hat S(t)$:

$$
w(t)
=
\hat S(t)^p
\{1-\hat S(t)\}^q.
$$

Different $(p,q)$ values target different time regions. Choosing a weight after inspecting the curves, however, creates a multiplicity and selection problem. The weighting strategy should ideally be prespecified when confirmatory inference is intended.

## Restricted mean survival time

When hazards are non-proportional, a direct time-scale estimand can be more interpretable. For horizon $\tau$, the restricted mean survival time is

$$
\operatorname{RMST}(\tau)
=
\int_0^\tau S(t)\,dt.
$$

The difference

$$
\Delta_{\mathrm{RMST}}(\tau)
=
\operatorname{RMST}_1(\tau)
-
\operatorname{RMST}_0(\tau)
$$

has units of time. It answers how much additional event-free time is associated with one group compared with the other by time $\tau$. This remains meaningful when survival curves cross, provided the horizon is scientifically justified.

## Interpreting a non-significant log-rank test

If

$$
p>0.05,
$$

the conclusion is not that the survival curves are equal. It is that the observed log-rank statistic did not provide sufficient evidence against the null at that threshold. Low event counts, heavy censoring, crossing hazards, or a genuinely small difference can all produce a non-significant result. Effect estimates and confidence intervals are necessary to distinguish these possibilities.

## Reproducible Python example

~~~python
from __future__ import annotations

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

time_a = [3, 5, 6, 8, 10, 12, 14, 16]
event_a = [1, 1, 0, 1, 1, 0, 1, 0]

time_b = [2, 4, 7, 7, 9, 11, 13, 15]
event_b = [1, 1, 1, 0, 1, 1, 0, 1]

test = logrank_test(
    time_a,
    time_b,
    event_observed_A=event_a,
    event_observed_B=event_b,
)

print(test.test_statistic)
print(test.p_value)

km_a = KaplanMeierFitter().fit(
    time_a,
    event_observed=event_a,
    label="A",
)

km_b = KaplanMeierFitter().fit(
    time_b,
    event_observed=event_b,
    label="B",
)
~~~

The test should be interpreted together with the fitted survival curves and a relevant effect estimate.

## Conclusion

The log-rank test compares groups through observed-minus-expected event counts over successive risk sets. Its essential requirements concern valid group comparison and censoring. Proportional hazards is the alternative under which the standard test is particularly well matched and powerful, not a simplistic switch that makes the test valid or invalid.

When hazards cross or effects change over time, the analysis should move beyond a single log-rank p-value and report time-specific or time-integrated effects that match the scientific question.

## References

- Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. *Cancer Chemotherapy Reports*, 50(3), 163–170.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–220.
- Fleming, T. R., & Harrington, D. P. (1991). *Counting Processes and Survival Analysis*. Wiley.
- Royston, P., & Parmar, M. K. B. (2011). The use of restricted mean survival time to estimate the treatment effect in randomized clinical trials when the proportional hazards assumption is in doubt. *Statistics in Medicine*, 30(19), 2409–2421.
