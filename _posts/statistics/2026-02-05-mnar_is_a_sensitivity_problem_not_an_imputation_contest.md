---
permalink: '/statistics/mnar_is_a_sensitivity_problem_not_an_imputation_contest/'
title: 'MNAR Is a Sensitivity Problem, Not an Imputation Contest'
date: '2026-02-05'
categories:
- Statistics
tags:
- Missing Data
- MNAR
- Sensitivity Analysis
- Pattern Mixture Models
- Selection Models
author_profile: false
classes: wide
seo_title: 'MNAR Is a Sensitivity Problem, Not an Imputation Contest'
seo_description: 'When missingness depends on unobserved values, the observed data generally do not identify the missing-data mechanism. MNAR analysis requires explicit assumptions, sensitivity parameters, tipping points, or additional identifying information.'
seo_type: article
excerpt: >-
  Under missing not at random, the unseen part of the distribution cannot
  generally be recovered from the observed sample alone. The central problem is
  therefore not which imputation algorithm predicts missing values best, but
  which assumptions identify the estimand and how conclusions change when those
  assumptions change.
summary: >-
  This article develops MNAR as an identification and sensitivity-analysis
  problem. An exact pattern-mixture construction shows that the same observed
  data can arise from both an MCAR mechanism and a strongly MNAR mechanism while
  implying different population means. The article then derives delta-adjusted
  pattern-mixture models, treatment-effect tipping points, selection models,
  partial-identification bounds, and the role of refreshment samples, validation
  data, instruments and external information. It also explains why artificial
  masking benchmarks do not validate an MNAR analysis.
keywords:
- missing not at random
- MNAR
- sensitivity analysis
- pattern mixture model
- selection model
- tipping point analysis
why_this_exists: >-
  Missing-data discussions often become competitions between imputation
  algorithms. That framing is inadequate under MNAR because the values that
  matter are unobserved precisely in a way that may depend on those values.
  Without extra information or restrictions, observed data do not determine the
  missing-data distribution uniquely.
evidence: >-
  Exact observed-data equivalence constructions, pattern-mixture and selection
  parameterisations, partial-identification bounds, controlled sensitivity
  calculations and standard missing-data theory.
methodology: >-
  Begin from the factorisation of the observed and missing outcome
  distributions, construct two incompatible full-data laws with the same
  observed-data distribution, then introduce explicit sensitivity parameters.
  Use a two-group treatment-effect example to derive a tipping point and a
  bounded-outcome example to show what can be learned without point
  identification.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-statistics-f-test.jpg
  og_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-f-test.jpg
  twitter_image: /assets/images/headers/photo-statistics-f-test.jpg
---

<!--
Development contract
Question: What can observed data identify when missingness still depends on unobserved values after conditioning on what was observed?
Claim: Under a nonparametric MNAR model, the observed-data law generally does not identify the distribution of the missing outcomes. Inference therefore requires additional restrictions, external information or sensitivity parameters whose influence should be reported explicitly.
Counterclaim: MNAR does not imply that useful inference is impossible. Validation samples, refreshment samples, instrumental or shadow variables, bounded outcomes, scientifically justified parametric structure and external information can identify or sharply constrain parts of the missing-data problem.
Evidence object: Exact observational-equivalence construction, delta-adjusted pattern-mixture model, treatment-effect tipping point and partial-identification interval for a bounded outcome.
Failure case: Treating an imputation algorithm that performs well under artificial random masking as validated for MNAR, interpreting a fitted selection parameter as empirically learned when it is driven mainly by functional-form restrictions, or presenting one sensitivity parameter as though it were observed.
Reader payoff: Separate what the data identify from what the missing-data assumptions supply, and report how substantive conclusions depend on those assumptions.
Exclusions: Repeating the general MCAR/MAR/MNAR overview, arguing that multiple imputation is inherently invalid, or presenting one MNAR model as universally correct.
-->

Missing data create two different statistical problems that are often collapsed into one. The first is computational: given a set of incomplete records, how should the analysis be performed? The second is inferential: what does the observed sample actually tell us about the part of the distribution that was not observed? Under missing completely at random or a plausible missing-at-random model, the second problem can sometimes be resolved by conditioning on observed information. Under missing not at random, the missingness process still depends on values that are absent, so the observed data alone generally do not determine the missing-data distribution.

That distinction changes the role of imputation. If the unobserved distribution is not identified, a more sophisticated prediction algorithm cannot manufacture identification. A neural network, random forest, chained-equations procedure or Bayesian model may produce numerically plausible completed data, but the values it generates are necessarily driven by assumptions about how missing cases differ from observed cases. The central question is therefore not which algorithm predicts missing entries most accurately. It is which assumptions connect the observed and unobserved parts of the population, whether those assumptions are scientifically credible, and how the target estimand changes when they are varied.

The blog already has a general discussion of [missing-data mechanisms](/machine-learning/missing_data_mechanisms_machine_learning/) and a separate article on [what multiple imputation gets right and where it can fail](/statistics/deep_dive_into_why_multiple_imputation_indefensible/). The present article takes a narrower and more mathematical position. It treats MNAR primarily as an identification and sensitivity problem.

## The observed data do not determine the missing distribution

Let $Y$ be an outcome and let

$$
R
=
\begin{cases}
1, & Y \text{ observed},\\
0, & Y \text{ missing}.
\end{cases}
$$

Suppose there are no covariates for the moment. The observed data identify two objects directly:

$$
\pi
=
P(R=1)
$$

and the conditional distribution

$$
f_1(y)
=
f(y\mid R=1).
$$

What is not observed is

$$
f_0(y)
=
f(y\mid R=0).
$$

The full population distribution is

$$
f(y)
=
\pi f_1(y)
+
(1-\pi)f_0(y).
$$

Without restrictions on $f_0$, the full distribution is not identified. Many different choices of $f_0$ produce exactly the same observed-data distribution because no value drawn from $f_0$ is ever seen.

An exact example makes this non-identification concrete. Suppose

$$
P(R=1)
=
0.8
$$

and among the observed cases,

$$
Y\mid R=1
\sim
N(0,1).
$$

Consider two full-data models.

In model A,

$$
Y\mid R=0
\sim
N(0,1).
$$

The missing and observed outcomes have the same distribution, so the population mean is

$$
E_A[Y]
=
0.8(0)
+
0.2(0)
=
0.
$$

In model B,

$$
Y\mid R=0
\sim
N(-3,1).
$$

The population mean is now

$$
E_B[Y]
=
0.8(0)
+
0.2(-3)
=
-0.6.
$$

The two models imply different full-data means, different lower tails and different scientific conclusions. They generate exactly the same observed sample: 80% of outcomes are visible, and the visible outcomes follow $N(0,1)$.

The difference lies entirely in a distribution that is never observed.

Model A corresponds to a situation in which missingness carries no outcome information in this simple construction. Model B induces outcome-dependent missingness. By Bayes' rule,

$$
P(R=1\mid Y=y)
=
\frac{
0.8\phi(y)
}{
0.8\phi(y)
+
0.2\phi(y+3)
},
$$

where $\phi$ is the standard-normal density. This probability varies with $y$, so missingness is MNAR.

The important fact is not that model B is more plausible than model A. It is that the observed data cannot choose between them. The same observed law is compatible with substantially different full-data laws and even with different missingness mechanisms.

This is why an observed-data test cannot in general establish MAR against MNAR. Relationships between missingness indicators and observed variables can provide evidence against MCAR, and study design can make some mechanisms more plausible than others, but the crucial dependence on the unseen value is not directly available for empirical comparison.

Covariates do not remove the logical problem. With observed covariates $X$, the missing-data distribution becomes

$$
f(Y\mid X,R=0),
$$

and the relevant assumption under MAR is that, after conditioning on the chosen observed information, this distribution can be linked to the observed cases without additional dependence on the unseen value. Whether the conditioning set is sufficient is a scientific assumption, not a property that can be verified completely from records in which the outcome is absent.

## Pattern-mixture models make the unobserved part explicit

One way to represent the problem is to factor the full-data distribution by missingness pattern,

$$
f(y,r)
=
P(R=r)
f(y\mid R=r).
$$

This is the pattern-mixture perspective. Instead of modelling why observations are missing, it models the outcome distribution separately among observed and missing cases.

The observed distribution

$$
f(y\mid R=1)
$$

is estimable. The missing distribution

$$
f(y\mid R=0)
$$

requires a restriction.

For a continuous outcome, a simple sensitivity model assumes that the missing and observed groups have the same distributional shape but different means,

$$
E[Y\mid R=0]
=
E[Y\mid R=1]
+
\Delta.
$$

The parameter $\Delta$ is not estimated from the missing values, because those values are missing. It is a sensitivity parameter.

If

$$
\mu_{\text{obs}}
=
E[Y\mid R=1],
$$

then the population mean becomes

$$
\mu
=
\pi\mu_{\text{obs}}
+
(1-\pi)
(\mu_{\text{obs}}+\Delta).
$$

Therefore,

$$
\mu
=
\mu_{\text{obs}}
+
(1-\pi)\Delta.
$$

This equation makes the assumption contribution completely transparent. The observed data determine $\mu_{\text{obs}}$ and $\pi$. The analyst supplies a plausible range for $\Delta$.

If 20% of the outcomes are missing,

$$
1-\pi
=
0.2,
$$

then every one-unit change in the assumed mean difference between missing and observed cases changes the population mean by

$$
0.2.
$$

The missing-data problem has not disappeared. It has been converted into a sensitivity curve whose slope is determined by the missing fraction.

This is a useful conversion because assumptions that would otherwise remain hidden inside an imputation algorithm become explicit quantities with scientific units. If $Y$ is a blood-pressure score, $\Delta$ is measured in blood-pressure units. If $Y$ is income, $\Delta$ is in currency. Subject-matter experts can reason about those values far more directly than about the name of an imputation method.

The same framework can be implemented through multiple imputation. One can generate MAR-based imputations and then shift imputed values by a specified $\Delta$, repeating the substantive analysis over a grid of sensitivity values. The important distinction is that the shift is not presented as an estimated truth. It is a controlled departure from MAR.

## A tipping point is often more informative than one MNAR estimate

Sensitivity analysis becomes especially useful when the substantive conclusion can be expressed as a threshold.

Consider a two-group study with treatment and control. Suppose the observed treatment mean is

$$
\mu_{T,\text{obs}}
=
0.6,
$$

and the observed control mean is

$$
\mu_{C,\text{obs}}
=
0.
$$

Assume 70% of outcomes are observed in each group, so

$$
\pi_T
=
\pi_C
=
0.7.
$$

Let missing outcomes differ from observed outcomes by

$$
\Delta_T
$$

in treatment and

$$
\Delta_C
$$

in control:

$$
E[Y_T\mid R=0]
=
0.6+\Delta_T,
$$

$$
E[Y_C\mid R=0]
=
0+\Delta_C.
$$

The full treatment mean is

$$
\mu_T
=
0.6
+
0.3\Delta_T,
$$

and the full control mean is

$$
\mu_C
=
0.3\Delta_C.
$$

The treatment contrast is therefore

$$
\tau
=
\mu_T-\mu_C
=
0.6
+
0.3(\Delta_T-\Delta_C).
$$

Under the MAR-like reference case

$$
\Delta_T
=
\Delta_C
=
0,
$$

the effect is

$$
\tau=0.6.
$$

If the missing outcomes are equally worse in both groups, so that

$$
\Delta_T=\Delta_C,
$$

the contrast remains

$$
0.6.
$$

The missing values can be substantially different from observed values without changing the treatment contrast if the departure is common to both groups.

The conclusion changes when the differential departure changes. The treatment effect reaches zero when

$$
0
=
0.6
+
0.3(\Delta_T-\Delta_C),
$$

which implies

$$
\Delta_T-\Delta_C
=
-2.
$$

The tipping point is therefore a two-unit differential disadvantage among missing treatment outcomes relative to missing control outcomes.

That is a more useful scientific statement than reporting one MNAR estimate chosen by convention. It asks whether a two-unit differential departure is plausible given the scale of the outcome, reasons for dropout, observed pre-dropout trajectories, auxiliary data and domain knowledge.

If a differential departure of only $-0.2$ would reverse the conclusion, the result is fragile. If reversal requires $-20$ on an outcome whose standard deviation is one, the conclusion is insensitive to a wide class of plausible missing-data deviations.

The same idea extends naturally to confidence intervals or posterior probabilities. Instead of finding the point at which the point estimate crosses zero, one can identify the sensitivity values at which a confidence interval includes zero, a clinical threshold is crossed, or a decision changes.

A sensitivity analysis is therefore most useful when it is tied to the decision boundary rather than displayed as a table of arbitrary alternative imputations.

## Selection models parameterise the missingness process instead

Pattern-mixture models start from

$$
f(Y\mid R).
$$

Selection models factor the joint distribution in the opposite direction,

$$
f(y,r)
=
f(y)
P(R=r\mid y).
$$

With covariates, a common selection model is

$$
\operatorname{logit}
P(R=1\mid Y,X)
=
\alpha
+
\beta^\top X
+
\gamma Y.
$$

The parameter $\gamma$ describes residual outcome dependence in the observation probability after conditioning on $X$. If

$$
\gamma=0,
$$

the observation process no longer depends directly on the unseen outcome under this model, corresponding to an MAR-type restriction. Nonzero $\gamma$ represents an MNAR departure.

This formulation can look attractive because the sensitivity assumption is expressed directly through the missingness process. The danger is to assume that fitting a parametric model makes $\gamma$ empirically identified in the same way as an ordinary regression coefficient.

In some fully parametric selection models, parameters can be formally identifiable because the assumed functional forms place enough restrictions on the joint distribution. That mathematical identification can be driven strongly by distributional shape assumptions rather than by direct information about missing outcomes. A normal outcome model combined with a logistic missingness model, for example, can sometimes distinguish parameter combinations that a nonparametric model cannot. If small deviations from those functional forms produce large changes in $\gamma$, the apparent identification is fragile.

For this reason, selection-model analyses commonly treat the MNAR parameter as a sensitivity parameter or place an externally justified prior over it rather than pretending that the observed likelihood alone has resolved the missingness mechanism.

Pattern-mixture and selection models are two factorisations of the same joint distribution. Neither is intrinsically more truthful. Their value is that they expose different scientific assumptions. Pattern-mixture models ask how missing outcomes differ from observed outcomes. Selection models ask how observation probability changes with the unseen outcome.

The more interpretable parameterisation depends on the application.

## Partial identification is sometimes more honest than a point estimate

Sensitivity analysis does not always need to select a narrow parametric family. If the outcome has known bounds, useful conclusions can follow from very weak assumptions.

Suppose

$$
0\le Y\le1.
$$

Let the observed fraction be

$$
\pi
$$

and the observed mean be

$$
\mu_{\text{obs}}.
$$

The missing-group mean

$$
\mu_{\text{mis}}
$$

must lie in

$$
[0,1].
$$

Therefore the full mean,

$$
\mu
=
\pi\mu_{\text{obs}}
+
(1-\pi)\mu_{\text{mis}},
$$

must satisfy

$$
\pi\mu_{\text{obs}}
\le
\mu
\le
\pi\mu_{\text{obs}}
+
(1-\pi).
$$

If 80% of values are observed and the observed mean is 0.70,

$$
\pi=0.8,
\qquad
\mu_{\text{obs}}=0.70,
$$

then

$$
0.56
\le
\mu
\le
0.76.
$$

This interval requires no MAR assumption and no imputation model. It is wide because the data genuinely leave uncertainty about the missing 20%.

Additional scientifically defensible restrictions can narrow the bounds. If missing outcomes are known to be no better than observed outcomes,

$$
\mu_{\text{mis}}
\le
0.70,
$$

the upper bound becomes

$$
0.70.
$$

If external information suggests

$$
0.4
\le
\mu_{\text{mis}}
\le
0.6,
$$

the full mean lies between

$$
0.8(0.70)+0.2(0.40)
=
0.64
$$

and

$$
0.8(0.70)+0.2(0.60)
=
0.68.
$$

The movement from

$$
[0.56,0.76]
$$

to

$$
[0.64,0.68]
$$

did not come from a better imputation algorithm. It came from additional identifying information.

Partial identification is valuable when a point estimate would create more certainty than the evidence supports. A bound can be decision-sufficient even if it does not collapse to one number. If every value in the identified set leads to the same substantive conclusion, missing-data uncertainty does not threaten the decision. If the set straddles the decision boundary, the uncertainty is real and should remain visible.

## Artificial masking does not validate an MNAR analysis

A common way to compare imputation algorithms is to start with complete observations, hide some known values artificially, impute them, and calculate RMSE or MAE against the values that were hidden. This can be useful for testing reconstruction under a specified masking mechanism. It does not reproduce the main inferential problem under MNAR.

Suppose the original complete subset consists only of people whose outcomes were observable in the real process. Randomly masking 20% of those observed values creates missingness by experiment. Conditional on the chosen masking rule, the artificially hidden values are not missing for the same reason as the real missing cases.

If the true study has

$$
P(R=1\mid Y)
$$

decreasing sharply for poor outcomes, random artificial masking replaces that mechanism with a known one that is independent of $Y$. An algorithm can perform excellently on this artificial task while remaining biased for the genuinely missing group because the latter comes from a different conditional distribution.

Even designing a synthetic MNAR masking mechanism does not prove that it matches the real one. It tests performance under the assumed mechanism chosen by the analyst.

Reconstruction error is also not the same as inferential validity. An imputation model with slightly better RMSE can distort a treatment contrast, tail probability, regression coefficient or standard error more severely than a model with worse pointwise prediction. The related draft *Imputation Accuracy Is Not Inferential Validity* is aimed at that broader distinction.

For MNAR specifically, algorithm benchmarking should therefore be subordinate to sensitivity analysis. The first question is whether the full-data estimand is identified under the assumed mechanism. The second is whether the computational method approximates that identified model well.

Reversing those questions can produce a highly accurate algorithm for the wrong estimand.

## Additional data can identify what the original sample cannot

MNAR should not be interpreted as a declaration that the problem is hopeless. Non-identification from one observed-data structure can sometimes be resolved by collecting different information.

A validation subsample may obtain outcomes for some units that would otherwise be missing. If the validation design is known, these observations provide direct information about

$$
f(Y\mid R=0,X).
$$

Refreshment samples can help in longitudinal studies by drawing a new cross-section from the target population at later times. The new sample can reveal marginal outcome distributions that dropout in the original panel concealed.

Administrative linkage can recover outcomes for study participants who stop responding. A survey respondent may decline to report income but still be linkable to a tax-record bracket. A patient who leaves a trial may have subsequent outcomes in electronic health records. These data do not make the missingness process disappear, but they change what is observed.

Instrumental or shadow-variable approaches use variables associated with the missing outcome but constrained in how they affect missingness, or vice versa. Under suitable assumptions, such variables can restore identification. The assumptions are strong and application-specific, but they illustrate a general principle: identification comes from information that distinguishes outcome values from observation propensity.

Repeated measurements can play a similar role. If pre-dropout trajectories are strongly predictive of the unseen follow-up outcome, they can make MAR more plausible conditional on observed history or sharply constrain an MNAR departure. They still do not prove that the departure is zero.

External studies, registries and expert elicitation can also inform sensitivity parameters. A prior distribution for $\Delta$ or $\gamma$ is most defensible when it encodes actual external knowledge rather than being chosen solely for mathematical convenience.

The best solution to a missing-data identification problem is often better data collection rather than a more complicated imputation engine.

## Sensitivity parameters should be expressed in scientific units

An MNAR sensitivity analysis becomes difficult to interpret when the sensitivity parameter has no direct relationship to the science.

A pattern-mixture shift

$$
\Delta
=
E[Y\mid R=0]
-
E[Y\mid R=1]
$$

has a clear unit. Experts can ask whether missing patients could plausibly have outcomes three points worse, whether missing income observations might be 20% higher, or whether unavailable sensor readings could correspond to temperatures five degrees above observed periods.

A log-odds selection parameter

$$
\gamma
$$

can also be interpreted, but the scale may be less intuitive. If

$$
\operatorname{logit}
P(R=1\mid Y,X)
=
\cdots+\gamma Y,
$$

then a one-unit increase in $Y$ multiplies the odds of observation by

$$
e^\gamma.
$$

Sensitivity ranges should therefore be translated into changes in observation probabilities over realistic outcome ranges. A prior such as

$$
\gamma\sim N(0,10^2)
$$

may look weak statistically while implying absurdly large differences in response probability scientifically.

Reference-based imputation in trials provides another example of interpretable structure. Instead of assuming missing post-dropout outcomes continue according to the treatment-arm model, one can impose assumptions such as jump to reference or copy reference. These are not neutral facts. They encode counterfactual post-dropout behaviour and should be understood as sensitivity scenarios.

A good MNAR analysis therefore describes each scenario in the language of the data-generating process, not only in software terminology.

## The estimand comes before the missing-data model

Different targets can have different sensitivity to the same missingness mechanism.

The population mean depends directly on the mean of the missing outcomes. A median can be insensitive to some tail departures and highly sensitive when missingness crosses the centre of the distribution. A regression slope can change because missingness distorts both marginal distributions and covariances. A treatment contrast may be robust to common shifts in both groups while being highly sensitive to differential shifts. A tail probability can be extremely sensitive to a small number of systematically missing extreme observations.

This means there is no universal statement that 10% missing data is harmless or 30% is fatal. The effect of missingness depends on where those missing values could lie and which functional of the distribution is being estimated.

The correct order is therefore to define the estimand, identify what part of it is supported by observed data, specify the weakest credible restrictions on the unobserved distribution, and then choose the computational method.

Imputation comes after this reasoning.

The same logic applies to machine learning. If the target is predictive performance for future records generated under the same missingness process, exploiting missingness indicators can be useful even when causal or population parameters are not identified. If the target is a policy effect under a changed data-collection regime, predictive success under the historical missingness pattern may be irrelevant.

"Missing data" is not one statistical task. Prediction, description and causal inference can require different assumptions even when the missing cells are identical.

## An honest MNAR result is a map, not one completed dataset

The temptation in missing-data analysis is to produce a complete rectangular dataset and proceed as if uncertainty has been resolved. Under MNAR, that presentation can hide the main source of uncertainty because the completed values depend on assumptions that the observed records did not determine.

A stronger analysis shows the map from assumptions to conclusions.

For a delta-adjusted pattern-mixture model, plot the estimand over a scientifically plausible range of $\Delta$. In a two-group problem, plot it over

$$
(\Delta_T,\Delta_C)
$$

or, when only the difference matters, over

$$
\Delta_T-\Delta_C.
$$

Mark the region in which the scientific conclusion is unchanged and the tipping boundary where it reverses.

For a selection model, vary the outcome dependence parameter and translate it into response-probability differences that domain experts can interpret. For bounded outcomes, report assumption-free or weak-assumption bounds before narrowing them with stronger restrictions. If external data inform the sensitivity parameter, show how much the conclusion changes with and without that information.

The final result can still include one primary analysis. Scientific work often requires a main estimate. The problem is presenting that estimate without showing how much of it was supplied by untestable assumptions.

The exact construction at the beginning of this article demonstrates why. The same observed data, 80% observed with

$$
Y\mid R=1
\sim
N(0,1),
$$

were compatible with a population mean of zero and a population mean of $-0.6$. No imputation contest could decide which full-data distribution generated those records because the distinguishing outcomes were precisely the ones that were absent.

The two-group example converted the unknown difference between missing and observed outcomes into a tipping point. With 30% missingness in each group and an observed treatment effect of $0.6$, the result reversed only when the missing treatment outcomes were two units worse relative to their observed group than the missing control outcomes were relative to theirs. Whether that departure is plausible is a substantive question that can be discussed.

That is the proper role of MNAR modelling. It does not conjure the unseen data. It makes explicit how much additional assumption is required to reach a conclusion.

Under MNAR, the missingness mechanism is not another hyperparameter waiting to be optimized. It is part of the scientific uncertainty.

## References

Daniels, M. J., & Hogan, J. W. (2008). *Missing Data in Longitudinal Studies: Strategies for Bayesian Modeling and Sensitivity Analysis*. Chapman & Hall/CRC.

Little, R. J. A. (1993). Pattern-mixture models for multivariate incomplete data. *Journal of the American Statistical Association*, 88(421), 125–134.

Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.

Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer.

Molenberghs, G., & Kenward, M. G. (2007). *Missing Data in Clinical Studies*. Wiley.

National Research Council. (2010). *The Prevention and Treatment of Missing Data in Clinical Trials*. National Academies Press.

Robins, J. M., Rotnitzky, A., & Scharfstein, D. O. (2000). Sensitivity analysis for selection bias and unmeasured confounding in missing data and causal inference models. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials*. Springer.

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.

Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for nonignorable drop-out using semiparametric nonresponse models. *Journal of the American Statistical Association*, 94(448), 1096–1120.
