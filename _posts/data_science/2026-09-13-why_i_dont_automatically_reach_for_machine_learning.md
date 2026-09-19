---
permalink: '/data-science/why_i_dont_automatically_reach_for_machine_learning/'
title: "Why I Don't Automatically Reach for Machine Learning"
categories:
- Data Science
- Statistics
tags:
- Statistical Modelling
- Machine Learning
- Explainability
- Applied Mathematics
- Model Selection
author_profile: false
seo_title: "Why I Don't Automatically Reach for Machine Learning"
seo_description: "Machine learning is useful, but it should not be the default answer to every data problem. Start from the scientific question, the data-generating structure, and the decision you need to make, then use ML when it earns its complexity."
excerpt: >-
  Machine learning is often treated as the default destination of a data project.
  I prefer the opposite order: understand the problem, write down the structure,
  build the simplest model that respects it, and only then ask whether additional
  predictive machinery buys something real.
summary: >-
  A modelling-first argument for using regression, state-space models, survival
  models, changepoint methods, Bayesian models, Gaussian processes and other
  transparent statistical tools before reaching automatically for machine learning.
  The point is not anti-ML. It is that complexity should solve a measured problem.
keywords:
  - statistical modelling
  - machine learning
  - model selection
  - explainability
  - data science
  - applied mathematics
classes: wide
date: '2026-09-13'
why_this_exists: >-
  In applied data science, algorithm choice often comes before problem formulation.
  This reverses that order. The article argues that many practical questions are
  better served by models whose assumptions, parameters, uncertainty and failure
  modes can be inspected directly, with machine learning added when measurable
  residual structure justifies it.
evidence: >-
  Draws on recurring modelling patterns across forecasting, time series, change
  detection, survival analysis, Bayesian inference and structured prediction. The
  article uses small mathematical examples rather than claiming one universal
  model family is superior.
methodology: >-
  Frames model choice as a sequence of questions about the estimand, data-generating
  process, loss function, extrapolation requirement, uncertainty, sample size and
  operational constraints. More complex models are judged by incremental predictive
  value, calibration, stability and out-of-distribution behaviour rather than by
  novelty alone.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
---

There is a habit in data science that I have never found particularly convincing.

A dataset arrives. Someone asks what model we should use. Within minutes the conversation has jumped to gradient boosting, neural networks, embeddings, transformers or whatever family happens to be fashionable at the time.

The algorithm has entered the room before the statistical question has been written down.

I prefer the opposite order.

Start with the problem.

Ask what is being estimated, predicted or decided. Ask what the observations actually represent. Ask which parts of the data-generating process are known, which assumptions are defensible, how uncertainty matters and whether the system will ever be asked to extrapolate.

Then choose a model.

Sometimes that model should absolutely be machine learning.

Often it should not.

My default is therefore not

$$
\text{data}
\rightarrow
\text{ML algorithm}.
$$

It is closer to

$$
\boxed{
\text{question}
\rightarrow
\text{structure}
\rightarrow
\text{estimand}
\rightarrow
\text{simple model}
\rightarrow
\text{measured residual problem}
\rightarrow
\text{additional complexity if justified}
}
$$

That distinction is more important than it may look.

## Prediction Is Not the Only Objective

Machine learning is especially powerful when the main objective is predictive accuracy under a well-defined future data distribution.

But many applied problems are not only prediction problems.

A researcher may want to know how an intervention changes an outcome. An engineer may need a parameter with physical meaning. A clinician may care about calibrated risk and uncertainty. A policy analyst may need a decomposition rather than a score. A forecasting system may need to explain which component changed. A scientific model may have to extrapolate outside the region in which observations are dense.

Those objectives are different.

Suppose I observe an outcome $Y$ and predictors $X$. A prediction problem asks for a function

$$
\widehat f(X)
$$

that minimizes some expected loss,

$$
R(f)
=
E\{L(Y,f(X))\}.
$$

That is already a complete and legitimate problem.

But an inferential problem may instead ask for a parameter $\beta_j$, a hazard ratio, a changepoint, a latent trend, a variance component, a posterior probability or a scientifically interpretable functional of the distribution.

Optimizing prediction error does not automatically answer those questions.

A black-box predictor can be excellent at

$$
Y\mid X
$$

while telling us very little about the parameter or mechanism we actually care about.

## A Model Is a Statement About Structure

Take the ordinary linear model,

$$
Y=X\beta+\varepsilon.
$$

Nobody should use it merely because it is simple.

But simplicity is not its main virtue.

The model tells us what structural statement we are making. Conditional on the chosen design and assumptions, changes in predictors act through a linear combination. The coefficients have a defined scale. Residual diagnostics have meaning. Interactions can be written explicitly. Restrictions can be imposed deliberately. Uncertainty can be propagated through a known inferential framework.

If I replace that model immediately with a flexible predictor, I may obtain lower test error. I may also lose the direct connection between the scientific statement and the fitted object.

That trade-off may be worthwhile.

It should still be a trade-off we consciously make.

## The Simplest Useful Model Is Not the Simplest Possible Model

There is an important difference between using a simple model and using a naive model.

If counts are the response, a Gaussian linear regression is often not the natural starting point. A Poisson or negative-binomial model may be structurally simpler because it respects the support and variance behaviour of the data.

If the outcome is time to an event, survival analysis gives us censoring-aware tools. Treating survival time as an ordinary regression target may make the computational pipeline look simpler while making the statistical problem wrong.

If data arrive over time, a state-space model may be more natural than a generic regressor because the latent dynamics are part of the problem.

If the scientific question concerns a structural break, a changepoint model addresses that question directly.

If observations are nested within schools, hospitals, regions or individuals, a hierarchical model can encode that repeated structure and partial pooling explicitly.

So when I say I prefer simpler, transparent models, I do not mean

$$
\text{always use linear regression}.
$$

I mean

$$
\boxed{
\text{use the simplest model that respects the important structure of the problem.}
}
$$

That can be mathematically sophisticated while remaining conceptually transparent.

## Structure Can Reduce the Amount of Learning We Need

Suppose a time series is generated approximately by

$$
y_t
=
\mu_t
+s_t
+x_t^\top\beta
+\varepsilon_t,
$$

where $\mu_t$ is a slowly varying level, $s_t$ is seasonality and $x_t$ contains known covariates.

A generic machine-learning model can attempt to learn the entire mapping from lagged inputs to future values.

A structural model says something stronger. It separates the forecasting problem into components whose behaviour we already partly understand.

This can be an enormous statistical advantage when data are limited.

Instead of asking an algorithm to rediscover seasonality, persistence and known effects from scratch, we encode those features and ask it to estimate the unknown quantities.

In statistical terms, structure restricts the function class.

That can increase bias when the restrictions are wrong, but it can sharply reduce variance when the restrictions are approximately right.

The familiar decomposition remains useful:

$$
E\left[(Y-\widehat f(X))^2\right]
=
\text{irreducible noise}
+
\text{bias}^2
+
\text{variance}.
$$

More flexibility mainly attacks bias. It can pay for that reduction with greater variance, more fragile tuning and more complicated failure modes.

With millions of representative observations, that trade may strongly favour flexibility.

With a few hundred observations and a good structural model, it may not.

## Small Data Changes the Calculation

A surprisingly large number of real data-science projects are small-data problems pretending to be big-data problems.

The raw database may contain millions of rows, but the effective sample size for the modelling question can be much smaller.

There may be only 40 independent patients, 20 stores, 15 years, 30 industrial batches, eight policy changes or a handful of genuine regime transitions.

Repeated measurements do not magically create independent information.

If subject $i$ contributes $T_i$ observations, the nominal sample size

$$
N=\sum_i T_i
$$

may dramatically overstate the effective information available for learning between-subject relationships.

A flexible model can fit the repeated structure beautifully while learning much less than its training score suggests.

In those settings I would rather model the dependence explicitly than celebrate a large row count.

## Extrapolation Is Where Structure Becomes Expensive to Ignore

Flexible machine-learning systems are usually strongest at interpolation.

They learn patterns in regions supported by training data.

Many scientific and engineering problems demand something harder.

What happens when temperature exceeds the observed range? What is the long-run effect of an intervention? How does a material behave at a pressure we have not measured? How does a demographic process evolve over decades? What happens to a financial quantity under an extreme stress scenario?

No model extrapolates safely by magic.

But explicit models make the extrapolation assumption visible.

Consider

$$
y=\beta_0+\beta_1 x+\varepsilon.
$$

Extrapolating beyond the observed $x$-range means asserting that the linear relationship continues.

That may be wrong, but at least the assumption is legible.

A flexible ensemble may extrapolate according to implementation-specific behaviour that is harder to describe and sometimes impossible to defend scientifically.

For extrapolation, I want the structure to come from the problem rather than emerge accidentally from the algorithm.

## Interpretability Is Not a Decoration Added Afterwards

There is a common workflow in which a complex model is fitted first and then interpretability tools are attached afterwards.

Feature importance, partial dependence, SHAP values or local explanations can all be useful.

But they answer a different question from fitting an interpretable model in the first place.

A model such as

$$
\log \lambda_i
=
\beta_0+x_i^\top\beta
$$

has parameters defined by the model itself.

Post-hoc explanation methods describe how a fitted predictor behaves under particular perturbations or conditioning choices. They can be informative, but they do not transform an arbitrary predictor into a structural model.

I therefore distinguish

$$
\text{model interpretability}
$$

from

$$
\text{prediction explanation}.
$$

Both can matter. They are not interchangeable.

## Uncertainty Is Part of the Output

In many problems, a point prediction is not enough.

Suppose the forecast is

$$
\widehat y_{t+h}=120.
$$

Whether that is useful depends heavily on whether the uncertainty is roughly

$$
[118,122]
$$

or

$$
[70,180].
$$

Likewise, a treatment effect estimate, failure probability or detected changepoint is incomplete without some representation of uncertainty.

Classical statistical models, Bayesian models, Gaussian processes and state-space models often make uncertainty a first-class object.

Machine-learning systems can certainly provide uncertainty too, but it must be designed and validated rather than assumed to exist because a predictive model is sophisticated.

A confidence score is not automatically a calibrated probability.

An ensemble spread is not automatically a prediction interval.

A neural posterior approximation is not automatically a valid posterior.

The uncertainty mechanism deserves the same scrutiny as the prediction mechanism.

## Start With a Baseline That Can Actually Win

One thing I dislike in benchmarking is the deliberately weak baseline.

If the machine-learning model is compared against a badly specified mean predictor or an obviously inappropriate regression, the experiment tells us almost nothing.

A baseline should represent a serious alternative.

For forecasting, that may mean seasonal naive forecasts, exponential smoothing, autoregressive models or a state-space formulation.

For tabular regression, it may mean a correctly specified generalized linear model with sensible transformations and interactions.

For count data, it may mean negative binomial regression rather than ordinary least squares.

For structured longitudinal data, it may mean mixed models or functional representations rather than flattening everything into independent rows.

Then the machine-learning model has to earn its place.

Let

$$
L_0
$$

be the out-of-sample loss of a strong transparent baseline and

$$
L_1
$$

the loss of the more complex model.

The quantity that matters is

$$
\Delta=L_0-L_1.
$$

Not whether model 1 is fashionable.

Not whether its training curve looks impressive.

Not whether it has more parameters.

The question is whether $\Delta$ is materially positive and stable enough to justify the additional complexity.

## Measure the Incremental Value of Complexity

Suppose repeated validation gives paired losses

$$
L_{0,r},\qquad L_{1,r},
$$

for folds, time windows or repeated samples $r=1,\ldots,R$.

Then examine

$$
\Delta_r=L_{0,r}-L_{1,r}.
$$

I want more than the mean.

I want to know whether the advantage persists across realistic data perturbations. Does it disappear in certain subgroups? Does it reverse under temporal drift? Does calibration deteriorate even while ranking improves? Does the complex model depend strongly on a few hyperparameters? Is its advantage smaller than the uncertainty in the validation procedure?

A model that improves average error from 0.184 to 0.182 may be worthwhile at enormous scale.

It may also be statistical noise attached to a much larger maintenance burden.

Context matters.

## Machine Learning Is Excellent When the Problem Actually Calls for It

None of this is an argument against machine learning.

There are problems where flexible learning is clearly the right tool.

Images, audio and natural language are obvious examples because useful representations themselves have to be learned from high-dimensional unstructured input.

Large-scale recommendation and ranking systems often involve interactions too numerous to specify manually.

Complex tabular systems with abundant representative data can contain nonlinearities and interactions that a compact parametric model cannot capture adequately.

In those cases the structure we can write down by hand may be weaker than the structure the data can teach us.

Then machine learning earns its complexity.

I would summarize that condition as

$$
\boxed{
\text{use ML when there is important predictive structure left after the defensible explicit structure has been used.}
}
$$

That is very different from using ML because the dataset has many columns.

## Hybrid Models Are Often Better Than the False Choice

The discussion is often framed as

$$
\text{statistics}
\quad\text{versus}\quad
\text{machine learning}.
$$

I do not find that distinction useful.

Many strong systems are hybrids.

A state-space model can handle known temporal structure while a flexible learner models the residual component.

A generalized additive model can capture interpretable nonlinear effects without becoming a black box.

A Gaussian process can encode smoothness and uncertainty while learning a highly flexible function.

A mechanistic model can produce physically meaningful features that feed a predictive model.

A neural network can estimate a nuisance function inside a broader semiparametric procedure.

A survival model can incorporate a learned representation while retaining the event-time likelihood.

We do not have to choose between mathematical structure and flexible prediction.

The useful question is which part of the problem should be specified and which part should be learned.

## Models Should Be Falsifiable

One reason I like explicit models is that they give us things to attack.

If residual variance is assumed constant, test that assumption.

If errors are assumed independent, inspect dependence.

If a time series model assumes one regime, look for structural breaks.

If a proportional-hazards model is used, investigate proportionality.

If a Gaussian process kernel encodes a length scale, inspect whether the inferred scale makes scientific sense.

If a hierarchical model assumes exchangeability, ask whether the groups really are exchangeable.

A useful model does not merely fit.

It exposes assumptions that can fail.

That gives us a cycle:

$$
\text{assumption}
\rightarrow
\text{fit}
\rightarrow
\text{diagnostic}
\rightarrow
\text{revision}.
$$

Machine-learning systems need analogous falsification tests: distribution-shift checks, subgroup performance, calibration, adversarial perturbations, sensitivity to seeds and hyperparameters, temporal validation and stress tests.

The more flexible the model, the more important those external contracts become because fewer constraints are visible in the model form itself.

## Sometimes the Right Result Is That Complexity Does Not Help

A data project can spend weeks trying increasingly complex algorithms and discover that a transparent model performs just as well.

That is not a failed project.

It is useful evidence.

If

$$
L_{\text{simple}}
\approx
L_{\text{complex}}
$$

under a credible evaluation design, then we have learned that the extra function-class flexibility is not buying much for this problem.

The simpler model may then win on deployment cost, stability, interpretability, reproducibility and ease of monitoring.

This is exactly the kind of result that gets lost when complexity is treated as progress by definition.

## What I Ask Before Reaching for ML

Before moving to a flexible machine-learning model, I usually want clear answers to a few questions.

What is the estimand or prediction target?

What is the unit of independent information?

What structure is already known from the domain?

Does the task require interpolation or extrapolation?

How important is calibrated uncertainty?

What is the strongest transparent baseline?

What failure mode of that baseline are we trying to solve?

How will we measure whether the more complex model solved it?

What happens under distribution shift?

What operational cost does the new model introduce?

Those questions often determine the model family more effectively than a leaderboard does.

## The Order Matters

For me, the most important principle is not simplicity itself.

It is the order in which the reasoning happens.

I do not want to begin with an algorithm and search for a justification.

I want to begin with the mathematical and scientific structure of the problem, build a model that makes that structure explicit, and then add flexibility where the evidence says the structure is incomplete.

So the workflow is not

$$
\text{simple model forever}.
$$

It is

$$
\boxed{
\text{structure first, complexity second.}
}
$$

Machine learning is one of the most useful tools in modern applied mathematics and data science.

That is precisely why I do not think it needs to be used automatically.

A method should be chosen because the problem demands what the method can do.

Not because the method is available.
