---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-01'
excerpt: Correlation is not causation, but the deeper question is how a causal effect becomes identifiable from a combination of design, assumptions, and data.
header:
  image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  og_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  overlay_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  twitter_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
keywords:
- Causal inference
- Correlation
- Confounding
- Colliders
- DAGs
- Potential outcomes
- Identification
seo_description: 'A modern introduction to causal inference: estimands, DAGs, confounding, colliders, exchangeability, positivity, consistency, and why identification comes before estimation.'
seo_title: 'Causality Beyond Correlation: Identification, DAGs, and Bias'
seo_type: article
summary: A revised introduction to causal inference that separates association from causal identification and explains how design, assumptions, DAGs, and estimands fit together.
tags:
- Causal Inference
- Correlation
- Statistics
title: 'Causality Beyond Correlation: Identification, DAGs, and Bias'
---

> **Revision note, September 2026.** The original 2020 version of this article framed causal inference too loosely, duplicated front matter inside the body, and overstated what randomized trials and causal graphs can establish on their own. This revision keeps the original URL but replaces that argument with a more careful identification-first treatment.

"Correlation does not imply causation" is true, but it is only the beginning of the problem.

The harder question is:

$$
\boxed{
\text{Under what assumptions does the observed data identify the causal quantity we care about?}
}
$$

That is the central problem of causal inference.

A correlation coefficient summarizes association in the observed distribution. A causal effect asks what would happen under an intervention or treatment assignment that may not have occurred naturally.

Those are different objects.

## 1. Start with the causal estimand

Suppose $A\in\{0,1\}$ denotes a treatment and $Y$ an outcome.

Let

$$
Y(1)
$$

be the outcome that would be observed for a unit under treatment and

$$
Y(0)
$$

under control.

A common causal estimand is the average treatment effect:

$$
\mathrm{ATE}
=
E[Y(1)-Y(0)].
$$

The difficulty is immediate: for any one unit, we observe at most one of these two potential outcomes.

This is the fundamental missing-data structure of causal inference.

Association, by contrast, concerns quantities such as

$$
E[Y\mid A=1]-E[Y\mid A=0],
$$

which are directly estimable from observed data.

The two coincide only under additional assumptions.

## 2. Identification comes before estimation

A useful separation is:

$$
\boxed{
\text{causal question}
\rightarrow
\text{estimand}
\rightarrow
\text{identification assumptions}
\rightarrow
\text{estimator}
}
$$

This order matters.

A sophisticated estimator cannot recover a causal effect that the design and assumptions do not identify.

Likewise, a simple estimator can be perfectly adequate when the causal structure is simple and the identification conditions are credible.

This is why causal inference is not merely regression with stronger language.

## 3. Randomization helps by creating exchangeability

Randomized experiments are powerful because random treatment assignment can make treatment independent of potential outcomes, at least in expectation and subject to correct implementation.

One version of the relevant condition is

$$
(Y(1),Y(0))\perp A.
$$

This is exchangeability.

Under randomization, plus appropriate consistency and positivity conditions, the observed treated and control groups can identify causal contrasts without having to model every prognostic factor.

But randomization does **not** magically remove every source of bias.

Trials can still suffer from:

- non-adherence;
- loss to follow-up;
- missing outcomes;
- measurement error;
- interference between units;
- protocol deviations;
- post-randomization selection;
- poor external validity.

So randomized treatment assignment is a powerful design feature, not a universal guarantee of unbiased inference.

## 4. Observational studies require stronger structural assumptions

When treatment is not randomized, treated and untreated units may differ systematically.

Suppose $L$ is a sufficient set of observed pre-treatment covariates. A conditional exchangeability assumption is

$$
(Y(1),Y(0))\perp A\mid L.
$$

This says that after conditioning on $L$, treatment assignment is as good as random with respect to the potential outcomes.

It is not something we can generally prove from the observed data.

It is an assumption about the causal structure.

That is the important shift from association to causal inference: the data alone do not decide which adjustment set is valid.

## 5. Three core identification conditions

For a standard treatment-effect analysis, three conditions appear repeatedly.

### Exchangeability

After conditioning on the chosen covariates, there is no unblocked common cause of treatment and outcome.

### Positivity

Every covariate pattern relevant to the target population must have a positive probability of receiving each treatment level being compared:

$$
0<P(A=a\mid L=l)<1.
$$

If some subgroup never receives one treatment, the corresponding causal contrast is not identified from those data without extrapolation.

### Consistency

If a unit actually receives treatment $A=a$, then the observed outcome equals the potential outcome under that treatment:

$$
Y=Y(a)\quad\text{when }A=a.
$$

This sounds obvious until treatment itself is poorly defined. "Exercise", "diet", "education", or "exposure" may correspond to many materially different interventions.

A causal estimand requires an intervention that is sufficiently well specified for the potential outcomes to be meaningful.

## 6. DAGs encode assumptions, not discoveries

Directed acyclic graphs are useful because they make causal assumptions explicit.

A simple confounding structure is

```text
L → A → Y
L → Y
```

Here $L$ is a common cause of treatment and outcome.

The path

$$
A\leftarrow L\rightarrow Y
$$

creates non-causal association between $A$ and $Y$.

Conditioning on $L$ can block that back-door path.

But a DAG does not become causal merely because it was drawn, and a causal graph is not generally identified from observational correlations alone.

Its arrows represent substantive assumptions informed by design, domain knowledge, temporal ordering, and prior evidence.

## 7. Why "adjust for everything" is wrong

Adjustment can reduce bias, but it can also create it.

The canonical example is a collider:

```text
A → C ← Y
```

The variable $C$ is caused by both $A$ and $Y$.

Without conditioning on $C$, the path

$$
A\rightarrow C\leftarrow Y
$$

is blocked.

Conditioning on $C$ can open that path and induce an association between $A$ and $Y$ even when none existed before.

This is why variable selection for causal inference cannot be reduced to:

> include every variable associated with treatment or outcome.

The role a variable plays in the causal graph matters.

## 8. Confounders, mediators, and colliders are different

Consider

```text
L → A → M → Y
L → Y
```

Here:

- $L$ may be a confounder;
- $M$ may be a mediator;
- another variable might be a collider.

Conditioning on each has a different consequence.

If the estimand is the **total effect** of $A$ on $Y$, adjusting for the mediator $M$ can remove part of the very effect we are trying to estimate.

If the estimand is a **direct effect**, mediator handling requires a different identification argument.

So there is no universal list of "control variables" independent of the estimand.

## 9. Selection bias is often collider bias

Berkson-type examples are not merely curiosities about strange correlations.

They are instances of conditioning on, or restricting to, a variable affected by two causes.

For example:

```text
A → S ← Y
```

where $S$ indicates selection into the observed sample.

If analysis is restricted to $S=1$, then treatment and outcome can become associated through the opened collider path even if they were independent in the source population.

This matters in:

- hospital-based studies;
- case-control sampling;
- complete-case analysis;
- platform or user-selection data;
- survivorship analyses;
- studies conditioned on employment, diagnosis, or admission.

Selection is therefore part of the causal model, not just a data-cleaning detail.

## 10. Simpson's paradox is about conditioning structure

Simpson's paradox describes situations where an aggregate association reverses after conditioning on another variable.

The important lesson is not:

> always stratify.

Nor is it:

> never trust aggregated data.

The correct action depends on the causal role of the stratifying variable.

If it is a confounder, conditioning may move us closer to the causal estimand.

If it is a mediator or collider, conditioning may answer a different question or introduce bias.

So Simpson's paradox is best viewed as a warning that association is conditional on the variables included in the analysis.

## 11. Granger causality is predictive precedence, not intervention causality

The original version of this article included Granger-causality code in a general causal-inference discussion.

That was misleading.

Granger causality asks whether past values of $X$ improve prediction of $Y$ beyond past values already in the model.

That is useful for time-series dependence and forecasting, but it does not by itself establish that intervening on $X$ would change $Y$.

A better label is often **Granger predictability** or **predictive causality**, while keeping it conceptually separate from intervention-based causal effects.

## 12. Estimation begins only after identification

Once a causal estimand has been identified under a credible set of assumptions, many estimators are possible.

Examples include:

- outcome regression;
- inverse-probability weighting;
- standardization or the g-formula;
- matching;
- doubly robust estimators;
- targeted maximum likelihood;
- instrumental-variable estimators;
- difference-in-differences;
- regression discontinuity;
- synthetic controls;
- panel and longitudinal g-methods.

These methods solve different identification and estimation problems.

No method can be judged in isolation from the estimand and assumptions that justify it.

## 13. Diagnostics cannot test away causal assumptions

Statistical diagnostics remain important.

We can inspect:

- propensity-score overlap;
- covariate balance;
- influential observations;
- model fit;
- residual patterns;
- positivity violations;
- sensitivity to alternative specifications.

But no goodness-of-fit test can establish that there is no unmeasured confounding.

That assumption is partly scientific and partly design-based.

The right response is therefore not to pretend the assumption is testable, but to make it explicit and study how conclusions change when it is weakened.

## 14. Sensitivity analysis belongs in causal inference

Suppose the estimated treatment effect is

$$
\hat\tau.
$$

The relevant question is not only whether $\hat\tau$ is statistically significant.

We should also ask:

- How strong would an unmeasured confounder need to be to materially change the conclusion?
- Does the result survive alternative reasonable adjustment sets?
- Is the estimate driven by regions with poor overlap?
- Does the estimand change when the target population changes?
- Are conclusions robust to missing-data assumptions?
- Are pre-treatment trends or placebo tests compatible with the identifying design?

Causal uncertainty is broader than standard error.

## 15. Causal discovery and causal effect estimation are not the same task

There is growing interest in algorithms that infer graph structure from data.

These methods can be valuable, but their outputs depend on assumptions such as:

- acyclicity;
- faithfulness;
- no hidden confounding, or explicit models for it;
- distributional or functional-form restrictions;
- correct temporal or intervention information.

Even when a Markov-equivalence class can be learned, multiple causal graphs may remain observationally indistinguishable.

So causal discovery should not be presented as an automatic route from correlation matrices to causal truth.

## 16. A practical workflow

A more defensible causal workflow is:

1. **Define the intervention or exposure.**
2. **Define the target population.**
3. **Define the causal estimand.**
4. **Write down the causal structure you are assuming.**
5. **Determine whether the estimand is identified under those assumptions.**
6. **Choose an estimator appropriate to that identification strategy.**
7. **Check overlap, model behavior, and data quality.**
8. **Run sensitivity analyses for assumptions the data cannot verify.**
9. **Separate statistical uncertainty from causal-structural uncertainty.**

In compact form:

$$
\boxed{
\text{question}
\rightarrow
\text{estimand}
\rightarrow
\text{assumptions}
\rightarrow
\text{identification}
\rightarrow
\text{estimation}
\rightarrow
\text{sensitivity}
}
$$

## 17. The practical rule

Do not ask only:

> Are $X$ and $Y$ correlated?

And do not jump immediately to:

> Which regression should I run?

Ask instead:

> What causal quantity do I want, and what assumptions would make it identifiable from the data I actually have?

That is the step that turns a statistical association into a causal analysis.

## References

- Hernán MA, Robins JM. *Causal Inference: What If*. Chapman & Hall/CRC, 2020.
- Greenland S, Pearl J, Robins JM. Causal diagrams for epidemiologic research. *Epidemiology*. 1999;10(1):37–48.
- Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. *Epidemiology*. 2004;15(5):615–625.
- Pearl J. *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge University Press, 2009.
- Pearl J, Glymour M, Jewell NP. *Causal Inference in Statistics: A Primer*. Wiley, 2016.
