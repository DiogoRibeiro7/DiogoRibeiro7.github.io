---
permalink: '/statistics/survival_analysis/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2024-05-10'
header:
  image: /assets/images/headers/photo-statistics-survival-analysis.jpg
  og_image: /assets/images/headers/photo-statistics-survival-analysis.jpg
  overlay_image: /assets/images/headers/photo-statistics-survival-analysis.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-survival-analysis.jpg
  twitter_image: /assets/images/headers/photo-statistics-survival-analysis.jpg
seo_description: "A rigorous introduction to survival analysis through censoring, risk sets, Kaplan-Meier estimation, Cox regression, competing risks, recurrent events, prediction, and causal interpretation."
seo_title: "Survival Analysis: Censoring, Hazards, and Time-to-Event Models"
seo_type: article
subtitle: "Censoring, hazards, regression, and prediction over time"
tags:
- Survival Analysis
- Statistics
- Time-to-Event
- Regression
title: "Survival Analysis: Censoring, Hazards, and Time-to-Event Models"
---

Survival analysis is the statistical study of **time until an event occurs**, but that familiar description understates what makes the subject distinctive. The central difficulty is not simply that the response variable is measured in units of time. It is that event times are frequently observed only partially: a patient may still be alive when a study ends, a customer may still be subscribed when the data are extracted, a machine may still be operating when monitoring stops, or an employee may leave the observation system for reasons unrelated to the event of interest. These observations cannot be discarded, because they tell us that the event time exceeds a known duration, but neither can they be treated as if the event had occurred at the last observation time. Survival analysis developed precisely to preserve that partial information while keeping the timing of events explicit.

This distinction gives survival methods a structure that ordinary regression on a binary outcome cannot reproduce. If a customer has not churned by twelve months, a binary model that simply records “no churn” throws away the fact that the customer has remained active for twelve months and ignores the possibility of churn later. Likewise, converting every observation into an event indicator by a fixed cutoff makes the result depend on the arbitrary horizon chosen by the analyst. Time-to-event methods instead describe how the event process evolves through time, how the composition of the population at risk changes, and how covariates are associated with that evolution. The resulting framework applies to medicine, reliability engineering, finance, customer retention, epidemiology, and many other domains because the mathematics depends on the event-time structure rather than on the substantive label attached to the event.

## Event time, censoring, and the structure of the risk set

Let $T$ denote the event time of interest and let $C$ denote a right-censoring time. What we observe is

$$
Y = \min(T,C)
$$

together with the event indicator

$$
\Delta = \mathbf{1}\{T \le C\}.
$$

When $\Delta=1$, the event time is observed. When $\Delta=0$, the event has not been observed by time $C$, so the available information is only that $T>C$. The distinction sounds simple, but it changes the likelihood contribution of the observation and therefore changes estimation fundamentally. Replacing a censored observation by an event at $C$ systematically shortens observed survival, whereas dropping censored observations selectively removes long-lived units. Both operations bias the analysis.

The usual right-censoring methods rely on an assumption often described as **independent** or **non-informative censoring**. Informally, after conditioning on the covariates represented in the model, the censoring mechanism should not contain additional information about the unobserved event time. In notation, one often works with a condition resembling

$$
T \perp C \mid X.
$$

This does not mean that censoring must occur completely at random. A study may censor older participants earlier because of a staggered recruitment design, for example, provided the relevant variables are modeled appropriately. What would be problematic is a monitoring system in which high-risk customers disappear from the data stream because account closures or platform changes are themselves related to impending churn, while the analysis treats those disappearances as ordinary censoring. The censoring mechanism is therefore part of the statistical design, not merely a data-cleaning detail.

The two most important functions in continuous-time survival analysis are the survival function and the hazard function. The survival function is

$$
S(t)=P(T>t),
$$

the probability of remaining event-free beyond time $t$. The hazard function is an instantaneous event rate among units that have survived up to time $t$,

$$
h(t)
=
\lim_{\Delta t\to0}
\frac{
P(t\le T<t+\Delta t \mid T\ge t)
}{
\Delta t
}.
$$

Because the hazard is a rate rather than a probability, it is not restricted to the interval $[0,1]$. This point is routinely misunderstood. A hazard of 1.4 at a particular time does not mean a 140% event probability; it describes an instantaneous rate conditional on being event-free immediately before that time.

For a continuous event-time distribution, the cumulative hazard

$$
H(t)=\int_0^t h(u)\,du
$$

is related to survival through

$$
S(t)=\exp[-H(t)].
$$

This identity is useful because some models are most naturally described in terms of hazards while scientific interpretation may be easier on the survival-probability scale. It also makes clear that the hazard is not simply another way to write the same probability. Survival accumulates the entire hazard history up to time $t$.

The notion of a **risk set** follows naturally. At event time $t$, the relevant comparison population consists only of individuals who are still under observation and have not yet experienced the event immediately before $t$. Units that experienced the event earlier are no longer at risk of a first event; units censored earlier are no longer observed. Much of survival analysis can be understood as careful accounting of these changing risk sets.

## Kaplan-Meier estimation and non-parametric comparison

The Kaplan-Meier estimator provides a non-parametric estimate of the survival function under right censoring. Suppose events occur at ordered times $t_1<t_2<\cdots$. Let $n_j$ denote the number of units at risk just before $t_j$, and let $d_j$ denote the number of events at that time. The conditional probability of surviving past $t_j$, given survival up to $t_j$, is estimated by

$$
1-\frac{d_j}{n_j}.
$$

Multiplying these conditional survival probabilities gives

$$
\widehat S(t)
=
\prod_{t_j\le t}
\left(
1-\frac{d_j}{n_j}
\right).
$$

The product-limit form is not an arbitrary computational trick. It reflects the decomposition of survival into successive conditional probabilities as the risk set changes. Censored observations reduce later risk sets but do not create downward jumps in the survival curve, because no event has been observed at the censoring time.

Uncertainty around the Kaplan-Meier estimator also matters. Greenwood's approximation is commonly written as

$$
\widehat{\operatorname{Var}}
\left[
\widehat S(t)
\right]
\approx
\widehat S(t)^2
\sum_{t_j\le t}
\frac{d_j}{
n_j(n_j-d_j)
}.
$$

In practice, confidence intervals are often constructed on transformed scales to avoid impossible values below zero or above one. A plotted survival curve without uncertainty can be visually persuasive while providing little indication of how much information remains at later times, where the risk set may have become very small.

Comparing two Kaplan-Meier curves requires similar care. The log-rank test is frequently used to test whether event-time distributions differ between groups, but it should not be interpreted as a generic measure of clinical or operational importance. The test gives particular weight to differences consistent with proportional-hazards-type alternatives and can lose power when survival curves cross. Moreover, a statistically significant difference does not describe its magnitude. Differences in survival probability at a prespecified horizon, differences in median survival when estimable, or differences in restricted mean survival time can often communicate the practical effect more directly.

Restricted mean survival time is especially useful when proportional hazards are doubtful. For a horizon $\tau$,

$$
\operatorname{RMST}(\tau)
=
\int_0^\tau
S(t)\,dt.
$$

It represents the expected event-free time accumulated up to $\tau$. Comparing RMST between groups gives an effect in units of time and does not require a constant hazard ratio. The choice of $\tau$ must be scientifically justified and should generally lie within a range where both groups have adequate follow-up.

## Regression models: Cox, accelerated failure time, and changing covariates

The Cox proportional hazards model is the most widely used regression model in survival analysis. It specifies

$$
h(t\mid X)
=
h_0(t)
\exp(X^\top\beta),
$$

where $h_0(t)$ is an unspecified baseline hazard and the covariates act multiplicatively on the hazard. The model is semi-parametric because the regression coefficients are finite-dimensional parameters while the baseline hazard is left unspecified. Estimation of $\beta$ uses the partial likelihood, which compares the covariates of the individual experiencing an event with the covariates of everyone in the corresponding risk set. Ignoring ties for notational simplicity, the partial likelihood has the form

$$
L(\beta)
=
\prod_{i:\Delta_i=1}
\frac{
\exp(X_i^\top\beta)
}{
\sum_{j\in R(t_i)}
\exp(X_j^\top\beta)
},
$$

where $R(t_i)$ denotes the risk set immediately before event time $t_i$.

For a one-unit increase in covariate $X_j$, the factor $\exp(\beta_j)$ is a hazard ratio under the model. A hazard ratio is **not** a risk ratio, a probability ratio, a ratio of median survival times, or a statement that one group experiences the event a fixed percentage sooner. It compares instantaneous event rates among individuals who are still event-free at a given time. Because the composition of the risk sets changes over time, hazard ratios can be difficult to translate into intuitive probability differences, especially when the event is common or when hazards are non-proportional.

The proportional-hazards assumption is therefore central. In the basic Cox model, the ratio of hazards for two covariate patterns does not depend on time:

$$
\frac{
h(t\mid X_1)
}{
h(t\mid X_2)
}
=
\exp[
(X_1-X_2)^\top\beta
].
$$

If an exposure has a strong early effect and little late effect, a single constant hazard ratio is an inadequate summary. Schoenfeld residuals, explicit interactions with time, graphical diagnostics, and comparison with more flexible models can reveal departures from proportional hazards. Failure of proportional hazards is not a failure of survival analysis; it is evidence that the chosen regression structure is too restrictive.

Accelerated failure-time models provide a different interpretation. A common form is

$$
\log T
=
X^\top\beta
+
\sigma\varepsilon,
$$

where the distribution of $\varepsilon$ determines models such as Weibull or log-normal survival. Exponentiating a coefficient gives a multiplicative effect on the time scale under the model. This can be easier to interpret in settings where the scientific question concerns acceleration or deceleration of event time rather than relative hazard.

Two time-related modeling ideas are often confused. A **time-dependent covariate** is a predictor whose value changes during follow-up, written $X(t)$. Blood pressure, account balance, treatment status, or machine temperature may all vary with time. A **time-varying coefficient** means that the effect of a predictor changes with time, written $\beta(t)$. A customer balance can change while its effect remains constant, or a fixed baseline treatment can have an effect that weakens through time. These are different model structures and should not be conflated.

Time-dependent predictors create additional hazards of their own, particularly when they are affected by previous treatment or previous outcomes. Naively inserting such variables into a Cox model does not automatically produce a causal effect. In longitudinal causal problems with time-varying confounding, methods such as marginal structural models or other g-methods may be required.

## Truncation, competing risks, recurrent events, and multi-state processes

Right censoring is only one form of incomplete observation. **Left truncation**, also called delayed entry, occurs when a unit is observed only if it has survived long enough to enter the study. If individuals become eligible for inclusion at entry time $L_i$, then they should contribute to the risk set only after $L_i$. Ignoring delayed entry creates survivorship bias because people who experienced the event before they could enter the dataset are systematically absent.

Interval censoring is another distinct problem. Sometimes the exact event time is unknown but is known to lie between two examinations. If a disease is absent at one visit and present at the next, the event did not necessarily occur at the second visit. Treating that visit time as exact introduces measurement error into the event process. Methods designed for interval-censored data use the interval information directly.

Competing risks arise when several mutually exclusive event types can occur and one event prevents observation of another. If a patient can die from several causes, or a customer can leave because of cancellation, migration, or account closure, then the event type matters. Let $J$ denote the event cause. The cause-specific cumulative incidence function is

$$
F_k(t)
=
P(T\le t, J=k).
$$

It can be written in terms of the overall survival function and the cause-specific hazard $\lambda_k(t)$ as

$$
F_k(t)
=
\int_0^t
S(u-)\lambda_k(u)\,du.
$$

Treating competing events as ordinary censoring in a Kaplan-Meier estimator does not estimate this observed-world cumulative incidence; it instead constructs a hypothetical net-risk quantity in which the competing event is removed. This often overstates the actual probability of the event of interest.

Cause-specific hazard models and Fine-Gray subdistribution-hazard models answer different questions. A cause-specific Cox model is useful for etiological or mechanistic questions about the instantaneous event process among those currently event-free. A subdistribution model is connected more directly to the cumulative incidence function. Neither should be selected simply because software offers it; the estimand must be specified first.

Some applications contain repeated events rather than one terminal event. Hospital readmissions, machine faults, service calls, purchases, and insurance claims can all recur. Reducing the process to time until the first event may discard most of the relevant information. Andersen-Gill models, Prentice-Williams-Peterson models, frailty models, gap-time models, and multi-state models provide different ways of representing dependence between repeated events and transitions between states. The choice depends on whether event order matters, whether the risk clock resets, and whether unobserved heterogeneity is scientifically plausible.

## Prediction, validation, and causal interpretation

Survival prediction requires more than ranking individuals by risk. The concordance index is widely used because it measures whether individuals with earlier events tend to receive higher predicted risk scores. It is therefore a discrimination metric. A model can have high concordance and still produce badly calibrated survival probabilities. If a model predicts a 20% two-year event probability for a group of individuals, calibration asks whether roughly 20% of comparable individuals actually experience the event by two years, after accounting appropriately for censoring.

The Brier score at time $t$ compares the event-free indicator with the predicted survival probability,

$$
BS(t)
=
E
\left[
\left(
\mathbf 1\{T>t\}
-
\widehat S(t\mid X)
\right)^2
\right],
$$

with practical estimators typically using inverse-probability-of-censoring weights because the event status is not observed for everyone at every horizon. Time-dependent AUC, integrated Brier scores, calibration curves, and horizon-specific calibration intercepts or slopes provide complementary information. No single metric fully characterizes a survival model.

Validation design is equally important. Randomly splitting rows can produce optimistic results when multiple records belong to the same individual, when covariates are updated over time, or when the deployment setting is prospective and the data-generating process changes through calendar time. Bootstrap validation can estimate optimism in model development; temporal validation can better reflect future deployment; external validation across sites or populations provides stronger evidence of transportability. Every preprocessing and feature-selection step must occur inside the validation procedure.

The interpretation of survival regression must also remain distinct from causal inference. If customers with higher monthly bills have a higher estimated churn hazard, the model has identified an association conditional on its covariates. It has not shown that lowering the bill would cause churn to decrease by the corresponding hazard ratio. Price is likely associated with product tier, usage, tenure, customer selection, service quality, and many other variables. A causal claim requires a treatment definition, a counterfactual estimand, and identification assumptions.

Survival data are particularly susceptible to **immortal time bias**, which occurs when exposure classification uses information from the future. Suppose patients are classified as “treated” if they receive a treatment at any point during follow-up. To become treated they must survive until treatment, so the pre-treatment interval is an immortal period during which they could not have died and still entered the treated group. Assigning that time to the treated exposure creates a spurious survival advantage. Correct analyses treat exposure as time-varying or redesign the estimand and time origin.

## A reproducible implementation

The following example keeps the code deliberately small. It validates the basic event-time inputs, fits a Kaplan-Meier estimator, and leaves more complex regression modeling to a separately specified design. The point is not to reduce survival analysis to a library call, but to make the event and censoring semantics explicit before estimation.

~~~python
from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter


def validate_survival_data(
    durations: pd.Series,
    events: pd.Series,
) -> None:
    """Validate basic right-censored survival inputs."""
    if len(durations) != len(events):
        raise ValueError("durations and events must have equal length")
    if durations.empty:
        raise ValueError("survival data must not be empty")
    if durations.isna().any() or events.isna().any():
        raise ValueError("durations and events must not contain missing values")
    if (durations < 0).any():
        raise ValueError("durations must be non-negative")
    if not events.isin([0, 1, False, True]).all():
        raise ValueError("events must be coded as binary indicators")


def fit_kaplan_meier(
    durations: pd.Series,
    events: pd.Series,
) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier estimator for right-censored event times."""
    validate_survival_data(durations, events)

    model = KaplanMeierFitter()
    model.fit(
        durations=durations.astype(float),
        event_observed=events.astype(bool),
    )
    return model
~~~

A production analysis would also document the time origin, censoring mechanism, delayed entry if present, event definitions, competing events, and the calendar period represented by the data. Those details are not metadata around the model; they determine what the model estimates.

## Conclusion

Survival analysis is a coherent statistical framework for reasoning about event times when follow-up is incomplete and the population at risk changes through time. Its distinctive contribution is not a collection of specialized estimators but a way of preserving the information contained in censoring, delayed entry, event ordering, and changing risk sets. Kaplan-Meier estimation, Cox regression, accelerated failure-time models, competing-risk methods, recurrent-event models, and multi-state models are different tools built around that common structure.

A rigorous analysis therefore begins before choosing a model. The event must be defined precisely, the time origin must be scientifically meaningful, the censoring and truncation mechanisms must be understood, and the estimand must be stated on a scale that answers the actual question. Hazard ratios can be useful summaries, but they are not probabilities; concordance can measure ranking, but it is not calibration; predictive association can identify risk, but it is not a causal intervention effect. These distinctions matter because time-to-event data are unusually easy to compress into a convenient model while silently changing the question being answered.

The strength of survival analysis lies in refusing that compression. It keeps time explicit, respects partial observation, and makes the evolving risk set part of the inferential problem. When those elements are modeled carefully, the same mathematics can describe mortality, customer churn, machine failure, loan default, employee turnover, disease recurrence, and many other processes without pretending that their substantive mechanisms are identical. That combination of generality and discipline is what makes survival analysis one of the most useful branches of applied statistics.

## References

- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). *Statistical Models Based on Counting Processes*. Springer.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–220.
- Fine, J. P., & Gray, R. J. (1999). A Proportional Hazards Model for the Subdistribution of a Competing Risk. *Journal of the American Statistical Association*, 94(446), 496–509.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. *Journal of the American Statistical Association*, 53(282), 457–481.
- Royston, P., & Parmar, M. K. B. (2013). Restricted Mean Survival Time: An Alternative to the Hazard Ratio for the Design and Analysis of Randomized Trials with a Time-to-Event Outcome. *BMC Medical Research Methodology*, 13, 152.
- Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
