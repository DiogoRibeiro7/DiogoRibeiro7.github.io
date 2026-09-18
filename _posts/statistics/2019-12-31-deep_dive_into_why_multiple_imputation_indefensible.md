---
author_profile: false
categories:
- Statistics
classes: wide
date: '2019-12-31'
excerpt: Multiple imputation is neither a universal cure for missing data nor an indefensible fiction. Its validity depends on the missingness assumptions, the imputation model, compatibility with the analysis, diagnostics, and sensitivity analysis.
header:
  image: /assets/images/headers/photo-statistics-f-test.jpg
  og_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_image: /assets/images/headers/photo-statistics-f-test.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-f-test.jpg
  twitter_image: /assets/images/headers/photo-statistics-f-test.jpg
keywords:
- Multiple imputation
- Missing data
- Missing at random
- Missing not at random
- Rubin's rules
- Sensitivity analysis
- Imputation model
seo_description: 'A critical but balanced review of multiple imputation: what it estimates, when it is valid, how it can fail, and why sensitivity analysis matters.'
seo_title: 'Multiple Imputation: What It Gets Right, What Can Go Wrong'
seo_type: article
summary: A revised guide to multiple imputation that separates valid criticisms from misconceptions and explains the assumptions, diagnostics, and sensitivity analyses needed for defensible missing-data inference.
tags:
- Missing Data
- Multiple Imputation
- Statistical Inference
title: 'Multiple Imputation: What It Gets Right, What Can Go Wrong'
---

> **Revision note (2026).** The original 2019 version of this article argued that multiple imputation was fundamentally indefensible and proposed single stochastic imputation as a superior alternative. That conclusion was too strong. The article has been rewritten to preserve the useful criticism — missing-data methods are assumption-dependent and require sensitivity analysis — while correcting the statistical claims about how multiple imputation works and when it is valid.

Missing data are not merely an inconvenience to be filled in before the “real” analysis begins.

They are part of the statistical problem.

The central question is not

> Which algorithm should I use to replace the missing values?

but

$$
\boxed{
\text{What assumptions make the unobserved part of the data informative enough for the target analysis?}
}
$$

Multiple imputation is one way to propagate uncertainty about missing values under an explicit statistical model. It is neither a universal gold standard nor an incoherent fiction.

Its validity depends on the assumptions connecting the observed and unobserved data, the quality of the imputation model, its compatibility with the substantive analysis, and whether uncertainty about unverifiable assumptions is explored rather than hidden.

## Start with the estimand and the missingness problem

Let $Y=(Y_{obs},Y_{mis})$ denote the complete data, where $Y_{obs}$ are observed values and $Y_{mis}$ are missing values. Let $R$ denote the missingness indicators.

Before choosing an imputation method, define the estimand. It might be

$$
E[Y],
$$

a regression coefficient,

$$
\beta,
$$

a treatment effect,

$$
E[Y(1)-Y(0)],
$$

or a predictive quantity.

The missing-data assumptions needed for valid inference depend on that target and on how the variables enter the analysis.

A method cannot rescue an estimand that the observed data and assumptions do not identify.

## MCAR, MAR, and MNAR are assumptions, not labels discovered from the data

The familiar taxonomy is useful if treated carefully.

### Missing completely at random

Under **MCAR**,

$$
P(R\mid Y_{obs},Y_{mis})=P(R).
$$

The probability of missingness does not depend on observed or missing values.

MCAR is strong and often implausible, but when it holds, complete-case analyses may remain unbiased for some targets, although usually less efficient because information is discarded.

### Missing at random

Under **MAR**,

$$
P(R\mid Y_{obs},Y_{mis})
=
P(R\mid Y_{obs}).
$$

Conditional on the observed variables included in the missingness or imputation model, missingness no longer depends on the unobserved values.

This is the assumption under which standard multiple-imputation procedures are most commonly justified.

But MAR is not something the observed data can generally prove. Whether it is plausible is a substantive modelling judgement.

### Missing not at random

Under **MNAR**,

$$
P(R\mid Y_{obs},Y_{mis})
$$

still depends on the unobserved values after conditioning on the observed data.

Examples include patients with unusually poor outcomes being more likely to drop out even after all observed prognostic variables have been accounted for.

MNAR does not mean analysis is impossible. It means the data alone do not identify the missingness mechanism without additional assumptions, restrictions, external information, or sensitivity parameters.

The crucial point is:

$$
\boxed{
\text{The missingness mechanism is part of the model.}
}
$$

## What multiple imputation actually does

The original version of this article described multiple imputation as averaging across several different missing-data models.

That is not the standard construction.

Under a specified imputation model, multiple imputation generates $M$ completed data sets by drawing plausible values for $Y_{mis}$ from a predictive distribution such as

$$
p(Y_{mis}\mid Y_{obs},R;\psi),
$$

where uncertainty in the model parameters $\psi$ is also propagated when the procedure is proper.

For each imputed data set $m=1,\dots,M$, the substantive analysis produces an estimate

$$
\hat\theta_m
$$

with estimated variance

$$
U_m.
$$

Rubin's combining rules then define

$$
\bar\theta
=
\frac{1}{M}\sum_{m=1}^{M}\hat\theta_m,
$$

within-imputation variance

$$
\bar U
=
\frac{1}{M}\sum_{m=1}^{M}U_m,
$$

and between-imputation variance

$$
B
=
\frac{1}{M-1}
\sum_{m=1}^{M}
(\hat\theta_m-\bar\theta)^2.
$$

The total variance is estimated by

$$
T
=
\bar U
+
\left(1+\frac{1}{M}\right)B.
$$

The meaning is clear:

- $\bar U$ represents uncertainty that would remain even if the missing values were known through a completed-data analysis;
- $B$ represents additional uncertainty induced by not knowing the missing values and model parameters;
- the finite-$M$ factor accounts for Monte Carlo error from using a finite number of imputations.

This is not “averaging incompatible realities.” It is integrating over uncertainty in unobserved data under a specified statistical model.

That model may be wrong. But that is a model-assessment problem, not a conceptual contradiction unique to multiple imputation.

## Fixed unknown parameters do not make parameter draws illegitimate

Another claim in the original article was that drawing model parameters was incoherent because the true parameters are fixed.

That criticism confuses the ontological status of a parameter with the mechanics of uncertainty propagation.

Frequentist procedures routinely use random quantities to represent sampling uncertainty around fixed parameters. Bootstrap distributions are one familiar example.

In proper multiple imputation, parameter uncertainty is introduced so that the variability across completed data sets reflects uncertainty induced by estimating the imputation model rather than pretending the fitted imputation parameters are known exactly.

The inferential question is not whether the population parameter is literally random in nature.

It is whether the procedure has the desired repeated-sampling properties or a coherent Bayesian interpretation under the stated model.

## Single imputation is not a safer default

The original article proposed single stochastic imputation as a more interpretable alternative.

That recommendation was incorrect as a general rule.

If missing values are imputed once and the completed data set is then analyzed as though the imputed values were observed, standard errors are typically too small because uncertainty about the missing values and imputation model has been ignored.

The completed data set looks more informative than it really is.

Formally, single imputation removes the between-imputation component

$$
B
$$

from the uncertainty calculation.

That can produce overconfident inference even when the single imputation itself is drawn from a reasonable predictive distribution.

Single imputation can still be useful for some prediction pipelines, visualization tasks, deterministic preprocessing systems, or carefully designed methods with external variance corrections. But it is not a general substitute for multiple imputation in inferential problems.

## The real problem: the imputation model can be wrong

A valid criticism of multiple imputation is that it is model-dependent.

Suppose the imputation model is

$$
Y_{mis}\mid Y_{obs}
\sim
p(\cdot\mid Y_{obs};\psi).
$$

If relevant predictors of missingness or the missing values are omitted, nonlinear structure is ignored, interactions are excluded, bounded variables are imputed by inappropriate Gaussian models, or longitudinal dependence is not represented, the completed data can be systematically distorted.

The number of imputations does not fix model misspecification.

Increasing $M$ reduces Monte Carlo error conditional on the imputation model. It does not make a bad imputation model good.

This distinction is important:

$$
\boxed{
\text{more imputations} \neq \text{more identification}
}
$$

and

$$
\boxed{
\text{more imputations} \neq \text{less model bias}.
}
$$

## The imputation model should be at least as rich as the analysis model

Suppose the substantive model is

$$
Y
=
\beta_0+
\beta_1X+
\beta_2Z+
\beta_3XZ+
\varepsilon.
$$

If the imputation model ignores the interaction $XZ$, the completed data may attenuate or otherwise distort the very relationship that the substantive analysis is trying to estimate.

Similar issues arise with nonlinear terms, transformations, time-varying effects, survival outcomes, hierarchical structure, and interactions.

This motivates the idea of **compatibility** or **congeniality** between the imputation and analysis models.

One practical lesson is:

> The imputation model should preserve the structures the analysis intends to estimate.

Substantive-model-compatible imputation methods were developed precisely because naive chained equations can otherwise conflict with nonlinear or interactive analysis models.

## Fully conditional specification is practical, but not automatically coherent

A common implementation is multiple imputation by chained equations, also called fully conditional specification.

For variables $Y_1,\ldots,Y_p$, one specifies conditional models such as

$$
p(Y_j\mid Y_{-j}),
$$

and iteratively imputes missing values variable by variable.

This is flexible and extremely useful in practice.

But an arbitrary set of conditional models need not correspond to a single coherent joint distribution.

That does not make chained equations unusable. It does mean the modeller should not confuse software convenience with automatic theoretical validity.

Diagnostics, subject-matter knowledge, and compatibility with the substantive model still matter.

## Diagnostics are not optional

Multiple imputation should not be a one-line preprocessing command followed by blind pooling.

At minimum, I would inspect:

### 1. Missingness patterns

Which variables are missing together? Does missingness depend strongly on observed covariates? Are there monotone or block structures?

### 2. Observed versus imputed distributions

Do imputed values occupy plausible ranges and reproduce important conditional relationships?

The goal is not for the marginal imputed distribution to match the observed one mechanically. Missing cases may genuinely differ. The question is whether the differences are scientifically and statistically plausible.

### 3. Trace and convergence diagnostics

For iterative methods such as chained equations, are imputations stable across iterations and starting values?

### 4. Influence of auxiliary variables

Variables that predict missingness or the incomplete variable can improve the plausibility of MAR and the efficiency of the imputation model, even when they do not appear in the final analysis.

### 5. Fraction of missing information

How much uncertainty is actually being introduced by missingness? A small percentage of missing cells can still produce substantial missing information for a particular estimand.

### 6. Monte Carlo error from finite $M$

The number of imputations should be large enough that simulation error from the imputation procedure is negligible relative to inferential uncertainty.

There is no statistical virtue in using exactly five imputations simply because early examples often did.

## MAR is often the beginning, not the end

A well-executed MAR analysis can still be fragile if plausible MNAR mechanisms would materially change the conclusion.

This is where the skeptical instinct in the original article was useful.

The problem was not that multiple imputation is inherently unfalsifiable. The deeper problem is that missing-data assumptions involving unobserved values are generally not identified by the observed data alone.

That means a defensible analysis should often ask:

$$
\boxed{
\text{How would the conclusion change under plausible departures from MAR?}
}
$$

## Sensitivity analysis should vary assumptions, not random seeds

Generating additional imputations under the same MAR model does **not** constitute sensitivity analysis.

It only improves the numerical approximation to inference under that model.

Sensitivity analysis requires changing the assumptions that connect observed and missing values.

### Delta adjustment

Suppose the MAR imputation model predicts

$$
Y_{mis}^{MAR}.
$$

An MNAR sensitivity model might instead use

$$
Y_{mis}^{MNAR}
=
Y_{mis}^{MAR}+\delta,
$$

where $\delta$ represents a systematic shift in the missing values relative to their MAR prediction.

Varying $\delta$ over a scientifically plausible range reveals how strongly the conclusion depends on departures from MAR.

### Pattern-mixture models

A pattern-mixture approach can model outcome distributions separately by missingness pattern and introduce identifying restrictions linking the observed and unobserved portions.

### Selection models

Selection models parameterize the outcome model and the missingness mechanism jointly, often making the dependence of missingness on unobserved outcomes explicit.

### Tipping-point analysis

Instead of asking whether one MNAR model is “correct,” we can ask:

> How extreme must the departure from MAR be before the substantive conclusion changes?

That question is often much more informative.

## Deterministic sensitivity analysis can complement multiple imputation

The original article framed deterministic sensitivity analysis as an alternative to multiple imputation.

A better view is that the two can be combined.

For each sensitivity parameter $\delta$, we can perform multiple imputation under the corresponding model and obtain

$$
\hat\theta(\delta)
$$

with its uncertainty.

Then the scientific result becomes a function

$$
\delta
\mapsto
\hat\theta(\delta),
$$

rather than a single answer presented as assumption-free.

This is more transparent because it separates two sources of uncertainty:

1. uncertainty **within** a chosen missing-data model;
2. uncertainty **about which missing-data model is plausible**.

Multiple imputation addresses the first. Sensitivity analysis addresses the second.

Confusing those roles is a common mistake.

## A worked conceptual example

Suppose the target is the mean outcome

$$
\mu=E[Y],
$$

and 20% of $Y$ is missing.

A naive complete-case estimate is

$$
\hat\mu_{CC}=12.1.
$$

After multiple imputation under an MAR model using observed covariates $X$ and $Z$, suppose we obtain

$$
\hat\mu_{MAR}=11.8,
$$

with a 95% interval

$$
[11.2,12.4].
$$

Now consider a delta-adjustment sensitivity model where missing outcomes are shifted downward by $\delta$ relative to the MAR prediction.

| $\delta$ | Estimated mean |
| ---: | ---: |
| 0.0 | 11.8 |
| -0.5 | 11.7 |
| -1.0 | 11.6 |
| -2.0 | 11.4 |

If the scientific conclusion changes only when $\delta<-1.7$, the relevant discussion is whether a shift that large is plausible.

That is more informative than arguing abstractly that MAR is either “true” or “false.”

## Imputation is not always the best tool

Multiple imputation is useful, but it is not mandatory.

Depending on the data-generating process and estimand, alternatives can include:

- maximum-likelihood estimation under an explicit incomplete-data model;
- inverse-probability weighting;
- doubly robust methods;
- Bayesian joint models;
- mixed models or likelihood-based longitudinal methods that naturally accommodate incomplete outcomes under their assumptions;
- partial-identification bounds when point identification is not defensible;
- explicit MNAR selection or pattern-mixture models;
- design-based methods that avoid imputation entirely.

The method should follow the estimand and assumptions, not fashion.

## Common failure modes in multiple imputation

The following practices deserve more criticism than multiple imputation itself:

1. **Calling MAR “verified.”** It usually cannot be established from observed data alone.
2. **Imputing before defining the analysis.** The imputation model must support the structures used by the substantive model.
3. **Using too few predictors.** Omitting variables related to missingness or the incomplete variable can make MAR less plausible and reduce efficiency.
4. **Ignoring interactions and nonlinearities.** This can destroy relationships the final analysis needs.
5. **Using deterministic mean imputation.** It suppresses variability and distorts associations.
6. **Using one stochastic imputation and ordinary standard errors.** It generally ignores imputation uncertainty.
7. **Treating $M=5$ as a rule.** The required number depends on missing information and desired Monte Carlo precision.
8. **Skipping diagnostics.** A pooled estimate is not evidence that the imputation model behaved sensibly.
9. **Reporting only an MAR analysis when MNAR departures are scientifically plausible.** Sensitivity analysis should be part of the inferential story.
10. **Believing more imputations repair misspecification.** They do not.

## What a defensible workflow looks like

A better sequence is

$$
\boxed{
\text{estimand}
\rightarrow
\text{missingness structure}
\rightarrow
\text{identification assumptions}
\rightarrow
\text{imputation or incomplete-data model}
\rightarrow
\text{diagnostics}
\rightarrow
\text{analysis}
\rightarrow
\text{sensitivity analysis}
}
$$

In practical terms:

1. define the target quantity before imputing;
2. understand which variables are incomplete and why;
3. identify predictors of missingness and incomplete values;
4. choose an imputation model compatible with the substantive analysis;
5. include auxiliary variables when useful;
6. use enough imputations to control Monte Carlo error;
7. inspect convergence and plausibility diagnostics;
8. pool estimates and uncertainty correctly;
9. challenge MAR with explicit sensitivity analyses when warranted;
10. report the assumptions as prominently as the numerical result.

## The broader lesson

The right criticism of multiple imputation is not

$$
\text{“it creates hypothetical data, therefore it is invalid.”}
$$

All missing-data inference relies on assumptions about information we did not observe.

The defensible criticism is more precise:

$$
\boxed{
\text{Multiple imputation is only as credible as the identification assumptions, imputation model, diagnostics, and sensitivity analysis that support it.}
}
$$

That is a much stronger standard than declaring the method either a gold standard or indefensible.

The aim is not to manufacture complete data.

It is to make uncertainty about incomplete data explicit enough that the scientific conclusion can be interrogated.

## References

- Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley, 1987.
- Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Wiley, 2019.
- Sterne JAC, White IR, Carlin JB, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*. 2009;338:b2393. DOI: [10.1136/bmj.b2393](https://doi.org/10.1136/bmj.b2393).
- White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. *Statistics in Medicine*. 2011;30(4):377-399. DOI: [10.1002/sim.4067](https://doi.org/10.1002/sim.4067).
- Bartlett JW, Seaman SR, White IR, Carpenter JR. Multiple imputation of covariates by fully conditional specification: accommodating the substantive model. *Statistical Methods in Medical Research*. 2015;24(4):462-487. DOI: [10.1177/0962280214521348](https://doi.org/10.1177/0962280214521348).
- National Research Council. *The Prevention and Treatment of Missing Data in Clinical Trials*. National Academies Press, 2010. DOI: [10.17226/12955](https://doi.org/10.17226/12955).
- Carpenter JR, Kenward MG. *Multiple Imputation and its Application*. Wiley, 2013.
