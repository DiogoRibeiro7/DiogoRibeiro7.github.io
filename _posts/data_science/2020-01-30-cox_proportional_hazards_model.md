---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-01-30'
excerpt: The Cox model estimates covariate effects on the hazard without specifying the baseline hazard, but hazard ratios are not risk ratios and proportional hazards must be checked rather than assumed.
header:
  image: /assets/images/headers/photo-healthcare-survival.jpg
  og_image: /assets/images/headers/photo-healthcare-survival.jpg
  overlay_image: /assets/images/headers/photo-healthcare-survival.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-healthcare-survival.jpg
  twitter_image: /assets/images/headers/photo-healthcare-survival.jpg
keywords:
- Cox proportional hazards
- survival analysis
- hazard ratio
- censoring
- Schoenfeld residuals
seo_description: A rigorous guide to Cox proportional hazards regression, partial likelihood, censoring, hazard-ratio interpretation, proportional-hazards diagnostics, and time-varying effects.
seo_title: 'Cox Proportional Hazards: Interpretation and Diagnostics'
seo_type: article
summary: A corrected guide to Cox regression that separates hazard from risk, explains partial likelihood and censoring assumptions, and shows how to diagnose and relax proportional hazards.
tags:
- Survival Analysis
- Statistical Modeling
- Healthcare
title: 'Cox Proportional Hazards: Interpretation and Diagnostics'
---

The Cox model is one of the most useful tools in survival analysis because it separates two parts of the event-time process:

1. an unspecified baseline hazard;
2. multiplicative covariate effects.

For covariate vector $X$,

$$
h(t\mid X)
=
h_0(t)\exp(X^\top\beta).
$$

The baseline hazard $h_0(t)$ can vary freely over time.

The covariate effect is summarized through $\beta$.

That flexibility is powerful, but it does not remove the need for assumptions.

## Hazard is not probability

For a continuous event time $T$, the hazard is

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

It is an instantaneous event rate conditional on survival to time $t$.

It is not a probability and can exceed 1 in its rate units.

The survival function is

$$
S(t)
=
P(T>t).
$$

These quantities are linked by the cumulative hazard

$$
H(t)
=
\int_0^t h(u)\,du
$$

and, for a continuous distribution,

$$
S(t)=\exp[-H(t)].
$$

This distinction matters when interpreting hazard ratios.

## Hazard ratios are not risk ratios

For a one-unit increase in predictor $X_j$, holding the other covariates fixed,

$$
\frac{
h(t\mid X_j+1,X_{-j})
}{
h(t\mid X_j,X_{-j})
}
=
\exp(\beta_j).
$$

That is a hazard ratio.

If

$$
\exp(\beta_j)=2,
$$

the instantaneous event rate is doubled at each time under the proportional-hazards model.

It does **not** mean the probability of experiencing the event by a fixed horizon is doubled.

Risk depends on the cumulative hazard over time.

A hazard ratio should therefore not be described casually as “twice the risk.”

## Proportional hazards is the defining slope restriction

For two covariate profiles $X_a$ and $X_b$,

$$
\frac{
h(t\mid X_a)
}{
h(t\mid X_b)
}
=
\exp\{(X_a-X_b)^\top\beta\}.
$$

The right-hand side does not depend on time.

That is the proportional-hazards assumption.

The baseline hazard can change dramatically.

The **ratio** between hazards for fixed covariate profiles is assumed constant.

If the treatment effect is strong early and weak later, a single constant coefficient can be misleading even if the fitted model converges without error.

## Partial likelihood

Suppose an event occurs for individual $i$ at time $t_i$.

Let $R(t_i)$ be the risk set immediately before that event.

Ignoring ties for the moment, the Cox partial-likelihood contribution is

$$
\frac{
\exp(X_i^\top\beta)
}{
\sum_{j\in R(t_i)}
\exp(X_j^\top\beta)
}.
$$

Multiplying these contributions over observed event times gives

$$
L_p(\beta)
=
\prod_{i:\delta_i=1}
\frac{
\exp(X_i^\top\beta)
}{
\sum_{j\in R(t_i)}
\exp(X_j^\top\beta)
}.
$$

The baseline hazard cancels from these conditional comparisons.

That is why $\beta$ can be estimated without specifying a parametric form for $h_0(t)$.

The method still uses event times through the risk sets.

It is not merely based on event ordering in an information-free sense.

## Tied event times require a convention

In real data, several events can occur at the same recorded time.

The simple partial-likelihood expression above assumes no ties.

Common approximations include Breslow and Efron methods.

For coarse time measurement or many ties, the choice can matter.

Software defaults should therefore be known rather than treated as an invisible implementation detail.

## Censoring

A right-censored observation contributes information that the event time exceeds its censoring time.

It remains in the risk set until censoring and leaves afterward.

The standard analysis relies on an appropriate independent-censoring assumption.

Informally, after conditioning on the variables required by the model and design, censoring should not contain additional information about the future event process.

Loss to follow-up due to deteriorating health can violate this.

Administrative censoring at a planned study end is often easier to justify.

## Schoenfeld residuals and proportional-hazards diagnostics

For each event, Schoenfeld residuals compare the covariate value of the subject experiencing the event with a risk-set-weighted expected covariate value.

Under proportional hazards, those residuals should not show systematic time trends.

A common diagnostic examines scaled Schoenfeld residuals against time and tests whether the slope is compatible with zero.

A small p-value is evidence against the constant-effect specification.

A large p-value does not prove proportional hazards.

Plots remain important because the shape of the time dependence matters.

## Time-dependent covariates are not the same as time-varying effects

These two ideas are often confused.

A **time-dependent covariate** changes value over follow-up:

$$
X_j=X_j(t).
$$

The model may still have a constant coefficient,

$$
h(t\mid X(t))
=
h_0(t)
\exp\{X(t)^\top\beta\}.
$$

A **time-varying effect** means the coefficient itself changes with time:

$$
h(t\mid X)
=
h_0(t)
\exp\{X^\top\beta(t)\}.
$$

The first allows the exposure value to evolve.

The second relaxes proportional hazards.

They solve different problems.

## Stratification

If a categorical nuisance factor violates proportional hazards and its coefficient is not itself the target, a stratified Cox model can use separate baseline hazards:

$$
h_s(t\mid X)
=
h_{0s}(t)
\exp(X^\top\beta).
$$

The covariate coefficients $\beta$ are shared across strata.

No hazard ratio is estimated for the stratification variable itself because its effect is absorbed into the stratum-specific baseline hazards.

## Frailty models

Shared frailty introduces a latent multiplicative random effect, for example

$$
h_{ij}(t)
=
u_j
h_0(t)
\exp(X_{ij}^\top\beta),
$$

where $u_j$ represents cluster-level unobserved heterogeneity.

This can model dependence among individuals within families, hospitals, or other clusters.

Frailty is not a generic repair for every omitted variable.

Its interpretation depends on the assumed random-effect distribution and clustering mechanism.

## Cox regression does not establish causation

In an observational study, a treatment coefficient can be confounded even if the proportional-hazards model fits perfectly.

The regression adjusts for included covariates.

It does not guarantee exchangeability between treatment groups.

Causal interpretation requires design or identification assumptions beyond the Cox likelihood.

This is especially important in medical applications, where “adjusted hazard ratio” is sometimes read as though it were automatically a causal treatment effect.

## A reproducible Python example

~~~python
from __future__ import annotations

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

data = load_rossi()

model = CoxPHFitter()
model.fit(
    data,
    duration_col="week",
    event_col="arrest",
)

model.print_summary()

model.check_assumptions(
    data,
    p_value_threshold=0.05,
    show_plots=False,
)
~~~

The fitted coefficients are log-hazard ratios conditional on the model and included covariates.

The diagnostics should be inspected before those ratios are summarized as constant effects over time.

## Absolute survival remains important

A hazard ratio can look impressive while the absolute difference in event probability is small, or vice versa.

For clinical interpretation, report quantities such as

$$
S(t\mid X)
$$

at meaningful horizons, absolute risk differences, or restricted mean survival time when appropriate.

Relative and absolute effects answer different questions.

## Conclusion

The Cox model is semiparametric because it leaves the baseline hazard unspecified while imposing a multiplicative covariate structure.

Its central coefficient interpretation is

$$
\exp(\beta_j)
=
\text{hazard ratio},
$$

not risk ratio.

The proportional-hazards assumption means that this hazard ratio is constant over time for time-fixed coefficients.

Censoring, ties, time-varying effects, and causal interpretation all require separate attention.

## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–220.
- Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
- Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*, 81(3), 515–526.
- Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data* (2nd ed.). Wiley.
