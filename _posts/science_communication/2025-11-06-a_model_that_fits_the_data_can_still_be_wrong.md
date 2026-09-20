---
permalink: '/science-communication/a_model_that_fits_the_data_can_still_be_wrong/'
title: 'A Model That Fits the Data Can Still Be Wrong'
date: '2025-11-06'
categories:
- Science Communication
tags:
- Statistical Modelling
- Identifiability
- Extrapolation
- Causal Inference
- Model Checking
author_profile: false
classes: wide
seo_title: 'Good Fit Does Not Establish a Correct Model'
seo_description: 'A model can reproduce observed data perfectly while giving the wrong extrapolation, leaving parameters unidentified, or supporting the wrong causal interpretation.'
seo_type: article
excerpt: >-
  Agreement with observed data is evidence that a model can reproduce those
  observations. It does not establish that the model is unique, that its
  parameters are identified, or that its causal interpretation is correct.
summary: >-
  This article develops three distinct ways in which excellent model fit can
  coexist with scientific error. An exact interpolation construction produces
  infinitely many models with zero training error but incompatible
  extrapolations. A nonlinear parameterisation shows that perfect prediction
  can coexist with structural non-identifiability. A pair of Gaussian
  structural models then demonstrates observational equivalence with opposite
  causal directions. The article concludes with practical model checking,
  sensitivity analysis, held-out prediction, experimental perturbation, and
  task-specific notions of model adequacy.
keywords:
- model fit
- model misspecification
- identifiability
- extrapolation
- observational equivalence
- causal models
why_this_exists: >-
  Scientific arguments frequently treat a high R-squared, small residuals,
  likelihood value, or accurate prediction as evidence that the underlying
  model has been established. This article separates empirical compatibility
  from uniqueness, parameter identification, extrapolation, and causal
  interpretation.
evidence: >-
  Three original mathematical counterexamples, classical work by Box on model
  criticism, White on misspecified likelihood models, Oreskes and colleagues
  on confirmation of numerical models, Gelman and Shalizi on model checking,
  Breiman on prediction and data modelling, and modern work on structural and
  practical identifiability.
methodology: >-
  Construct an infinite family of exact interpolants by adding a polynomial
  that vanishes at every observed design point. Derive a singular Fisher
  information matrix for a product-parameter model. Construct two Gaussian
  structural equation models with the same observational covariance matrix
  but different intervention distributions. Use these examples to distinguish
  fit, prediction, identifiability, and causal interpretation.
reviewed_at: '2025-11-06'
header:
  image: /assets/images/headers/photo-formulas.jpg
  og_image: /assets/images/headers/photo-formulas.jpg
  overlay_image: /assets/images/headers/photo-formulas.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-formulas.jpg
  twitter_image: /assets/images/headers/photo-formulas.jpg
---

<!--
Development contract
Question: What does good agreement between a model and observed data establish about the model?
Claim: Good fit establishes empirical compatibility with selected features of the observed data. It does not by itself establish uniqueness of functional form, parameter identifiability, extrapolation validity, or causal structure.
Counterclaim: Model fit is still essential evidence. A model that cannot reproduce relevant observations has failed an important test, and predictive validation can strongly support a model for a specified task and domain.
Evidence object: Infinite exact-interpolant construction, singular Fisher information in a product-parameter model, observationally equivalent Gaussian causal models, and a distribution-shift formulation of predictive risk.
Failure case: Concluding that because every model is imperfect no quantitative model deserves trust, or using philosophical limits on validation to dismiss strong predictive and experimental evidence.
Reader payoff: Separate fit from identification and explanation, recognise when held-out prediction is still insufficient, and choose model checks that target the scientific claim actually being made.
Exclusions: Arguing that mechanistic models are superior to statistical models in general, ranking modelling paradigms, and treating one goodness-of-fit statistic as universally appropriate.
-->

A model that fits observed data well has passed a test.

The difficulty is deciding which test.

A small residual sum of squares can show that predictions are numerically close to the observations used in fitting. A large likelihood can show that the observed data are relatively compatible with one set of model parameters. A high (R^2) can show that the model explains much of the observed variation according to a particular decomposition. Accurate held-out prediction can show that a fitted procedure generalises to data drawn under conditions similar to those used for validation.

These are meaningful achievements.

None of them automatically establishes that the mathematical form of the model is correct, that its parameters correspond uniquely to real processes, that it will extrapolate outside the observed domain, or that a causal story attached to the model has been identified.

The distinction matters because the word *fit* can quietly absorb all of those stronger claims.

George Box described model building as an iterative confrontation between theory and practice rather than a process in which one final equation is certified as true. Later work on model criticism, misspecification, identifiability, and causal inference has made the same problem precise in different mathematical languages.

The most direct way to see the issue is to construct models that fit perfectly and still disagree about what happens next.

## Perfect interpolation does not determine what happens between or beyond the data

Suppose we observe (n) input-output pairs

[
(x_1,y_1),ldots,(x_n,y_n).
]

Assume a function (f(x)) fits them exactly:

[
f(x_i)=y_i
]

for every observed point.

Now define a new family of functions

[
g_c(x)
=
f(x)
+
cprod_{i=1}^{n}(x-x_i),
]

where (c) can be any real number.

At every observed input (x_j), one factor in the product is zero. Therefore,

[
g_c(x_j)
=
f(x_j)
+
ccdot0
=
y_j.
]

Every value of (c) produces exactly the same fit to the observed data.

There are infinitely many such models.

Their residual sum of squares on the observed points is identical:

[
operatorname{RSS}(g_c)=0.
]

Their predictions away from the observed inputs can be arbitrarily different.

Take the simplest concrete example. Suppose the observed data are

[
(0,0),qquad(1,1),qquad(2,2).
]

One exact model is

[
f(x)=x.
]

Another family is

[
g_c(x)
=
x+c,x(x-1)(x-2).
]

At (x=0), (x=1), and (x=2), every member of this family produces exactly the observed values.

At (x=3),

[
g_c(3)
=
3+6c.
]

If

[
c=10,
]

then

[
g_{10}(3)=63.
]

If

[
c=-10,
]

then

[
g_{-10}(3)=-57.
]

The training data cannot distinguish a prediction of 63 from a prediction of (-57), even though both models achieve perfect fit at every observed point.

The example is deliberately extreme because the logical point should be visible without numerical ambiguity. In practical modelling, regularisation, smoothness assumptions, scientific theory, prior information, and model class restrictions prevent us from considering every possible interpolating function. Those constraints are precisely what make the prediction possible.

They are also assumptions.

The data alone did not select the continuation.

## Extrapolation always imports structure

Interpolation and extrapolation are often discussed as though they differ only in distance.

The deeper distinction is that extrapolation depends more strongly on structural assumptions that were weakly tested or not tested at all by the observed data.

Suppose observations cover

[
0le xle2.
]

Within that interval, a linear model and a mildly curved model may be nearly indistinguishable. Outside the interval, the terms that were numerically small can dominate.

A polynomial model illustrates this immediately:

[
Y
=
eta_0+eta_1x+eta_2x^2+arepsilon.
]

If all observed (x) values occupy a narrow range around zero, the contribution of

[
eta_2x^2
]

may be too small to distinguish reliably from noise. The fitted data can look almost linear whether (eta_2=0), (0.1), or a moderately different value.

At larger (x), the quadratic term grows as (x^2).

A component that was practically invisible in the calibration region can dominate the extrapolation.

This is why out-of-range prediction is not merely ordinary prediction with a larger confidence interval. The relevant functional behaviour may not have been empirically constrained at all.

The same issue appears in mechanistic models. Different parameter combinations can reproduce observed trajectories over one experimental regime while implying very different behaviour under a stronger perturbation, a longer time horizon, or a different initial condition.

Good fit within the observed regime supports local adequacy.

It does not automatically support structural transport outside that regime.

## Even parameters can remain unidentified when predictions fit perfectly

A second problem occurs when the model's observable predictions are well determined but the parameters used to explain those predictions are not.

Consider

[
Y_i
=
abx_i+arepsilon_i,
]

with

[
arepsilon_isimmathcal N(0,sigma^2).
]

The expected value is

[
mu_i=abx_i.
]

The likelihood depends on (a) and (b) only through their product

[
	heta=ab.
]

If the data strongly support

[
	heta=2,
]

then all of the following parameter pairs produce exactly the same fitted mean:

[
(a,b)=(1,2),
]

[
(a,b)=(2,1),
]

[
(a,b)=(4,0.5),
]

and infinitely many others satisfying

[
ab=2.
]

Prediction of (Y) can be excellent.

Separate interpretation of (a) and (b) is impossible from these data.

The geometry appears directly in the Fisher information.

The derivatives of the mean are

[
rac{partialmu_i}{partial a}
=
bx_i
]

and

[
rac{partialmu_i}{partial b}
=
ax_i.
]

For independent normal observations with known variance, the information matrix is

[
I(a,b)
=
rac{1}{sigma^2}
sum_i x_i^2
egin{pmatrix}
b^2 & ab\
ab & a^2
end{pmatrix}.
]

Its determinant is

[
det I(a,b)
=
left(
rac{1}{sigma^2}sum_i x_i^2
ight)^2
left(a^2b^2-a^2b^2ight)
=
0.
]

The matrix is singular.

No amount of additional data of exactly the same form resolves the individual parameters. More observations can estimate (ab) more precisely, but they cannot separate (a) from (b).

This is structural non-identifiability.

The problem differs from ordinary sampling uncertainty. A parameter can have a wide confidence interval because the sample is small. A structurally non-identifiable parameter is not uniquely recoverable even with ideal noise-free data under the same observation scheme.

Cobelli and DiStefano discussed this distinction in physiological models decades ago, and identifiability remains a central issue in modern dynamical systems modelling. Recent reviews distinguish structural non-identifiability from practical non-identifiability, where a parameter is theoretically recoverable but the available experiment does not contain enough information to estimate it precisely.

The distinction matters because parameters often carry scientific names.

If (a) represents one biological rate and (b) another, good prediction of their product does not justify claiming that both rates have been estimated.

## Reparameterisation can reveal what the data actually identify

The product model is not useless.

It is badly parameterised for the information supplied by the experiment.

Define

[
	heta=ab.
]

The model becomes

[
Y_i=	heta x_i+arepsilon_i.
]

Now the identifiable object is explicit.

This is often a productive response to non-identifiability. Rather than asking the data to estimate parameters they cannot separate, one can search for identifiable combinations.

In more complex models, these combinations may not be obvious. Profile likelihoods, symbolic methods, differential algebra, sensitivity analysis, rank tests, and other tools can reveal parameter directions that the data constrain weakly or not at all.

The conceptual lesson remains simple.

A fitted parameter value is not automatically an empirically identified quantity.

Optimisation software will often return a number even when many nearby or distant parameter combinations produce essentially the same likelihood.

Convergence of the optimiser is a numerical statement.

Identification is a statistical and structural statement.

They should not be confused.

## The same observational distribution can support opposite causal stories

A model can also fit every aspect of an observational distribution and still fail to identify the causal direction.

Let (X) and (Y) be standard normal variables with correlation

[
ho.
]

Consider the structural model

[
Xsimmathcal N(0,1),
]

[
Y=ho X+arepsilon_Y,
]

where

[
arepsilon_Y
sim
mathcal N(0,1-ho^2)
]

and (arepsilon_Y) is independent of (X).

This generates

[
operatorname{Var}(X)=1,
]

[
operatorname{Var}(Y)=1,
]

and

[
operatorname{Cov}(X,Y)=ho.
]

Now reverse the structural direction:

[
Ysimmathcal N(0,1),
]

[
X=ho Y+arepsilon_X,
]

with

[
arepsilon_X
sim
mathcal N(0,1-ho^2)
]

independent of (Y).

This model generates exactly the same bivariate normal observational distribution:

[
egin{pmatrix}
X\
Y
end{pmatrix}
sim
mathcal N
left[
egin{pmatrix}
0\
0
end{pmatrix},
egin{pmatrix}
1 & ho\
ho & 1
end{pmatrix}
ight].
]

No amount of passive observation of that joint distribution can distinguish the two models under the stated assumptions.

Their intervention predictions differ.

Under the first model, forcing

[
X=x
]

gives

[
mathbb E[Ymid do(X=x)]
=
ho x.
]

Under the second model, (Y) is generated upstream of (X). Intervening on (X) does not alter the distribution of (Y), so

[
mathbb E[Ymid do(X=x)]
=
0.
]

The observational fit is identical.

The causal conclusions are not.

Additional information can break the equivalence. Time ordering, experimental intervention, non-Gaussian structure, known physical constraints, instrumental variables, or other assumptions may distinguish the models.

The point is not that causal direction is always unknowable.

It is that observational fit alone did not identify it in this example.

## An excellent predictor can still be a poor explanation

Prediction and explanation are related scientific goals, but they place different demands on a model.

Suppose a black-box predictor estimates an outcome accurately in new observations from the same environment. That is strong evidence that the model contains useful predictive information.

It need not mean that internal variables or feature importance scores correspond to causal mechanisms.

Leo Breiman's discussion of the two cultures of statistical modelling emphasised the value of judging models by predictive performance rather than assuming that a chosen stochastic data model accurately represents the data-generating process.

That argument should not be converted into the opposite mistake.

Good prediction does not automatically establish mechanism either.

A variable can predict because it is a cause, an effect, a proxy, a collider-related artefact, a stable correlate, or a marker of the environment in which the data were collected.

For forecasting, some of those distinctions may be irrelevant.

For intervention, they are central.

The scientific interpretation therefore depends on the task.

A model can be adequate for prediction and inadequate for explanation.

A different model can be scientifically illuminating while predicting slightly less accurately.

Neither property can be inferred from fit alone.

## Held-out prediction solves one problem, not every problem

Training error is especially weak evidence because the model was selected using the same data.

Held-out prediction improves the situation.

Let (P) denote the distribution that generated the development data. For a prediction function (f) and loss (ell), define its target risk under (P) as

[
R_P(f)
=
mathbb E_P[ell(Y,f(X))].
]

A properly designed test set can estimate this quantity when test observations are independent and drawn from the same relevant distribution.

Now suppose deployment occurs under a different distribution (Q).

The relevant risk becomes

[
R_Q(f)
=
mathbb E_Q[ell(Y,f(X))].
]

Excellent estimation of

[
R_P(f)
]

does not determine

[
R_Q(f).
]

The change can arise from a different covariate distribution, altered measurement procedures, changed behaviour, a new policy environment, different prevalence, selection into the sample, or modification of the causal system itself.

Cross-validation therefore addresses overfitting to the observed sample.

It does not guarantee transport across distribution shift.

This matters because the phrase "validated model" can conceal the domain in which validation occurred.

Validation is always relative to a population, measurement process, outcome definition, and time period.

## A likelihood can be maximised under a false model

Suppose the true data distribution is (P), but every model in the fitted family

[
{Q_	heta:	hetainTheta}
]

is wrong.

Maximum likelihood still returns an estimate.

Under regularity conditions, the estimator can converge toward the parameter value whose model distribution is closest to the truth in Kullback-Leibler divergence.

That limiting parameter can be useful.

It is not proof that

[
P=Q_{	heta}.
]

White's 1982 analysis of maximum likelihood under model misspecification formalised this distinction. Standard inferential formulas that assume correct specification can also fail under misspecification, motivating robust covariance estimators in many settings.

This is another reason optimisation success should not be interpreted as model truth.

A numerical method answers

[
	ext{which member of this family fits best?}
]

It does not answer

[
	ext{is this family scientifically adequate?}
]

Those are different questions.

## Residuals test particular failures

Model checking attempts to find observations that the fitted model struggles to reproduce.

For a linear regression, residual plots may reveal curvature, heteroscedasticity, clustering, temporal dependence, or influential observations.

For probabilistic models, one can simulate replicated datasets from the fitted model and compare features of those simulations with the observed data.

If a Bayesian model is described by

[
p(	hetamid y),
]

the posterior predictive distribution is

[
p(y^{mathrm{rep}}mid y)
=
int
p(y^{mathrm{rep}}mid	heta)
p(	hetamid y)
,d	heta.
]

A discrepancy statistic (T) can then compare

[
T(y^{mathrm{rep}})
]

with

[
T(y).
]

Gelman and Shalizi emphasise this model-checking perspective. The aim is not merely to obtain posterior uncertainty conditional on a model, but to ask whether important features of the observed data resemble what the fitted model itself predicts.

A model can pass one check and fail another.

Matching the mean does not imply matching tail behaviour.

Matching marginal distributions does not imply reproducing dependence.

Matching short-term trajectories does not imply capturing long-term stability.

A model check is informative only about the aspect of the model that the check was capable of challenging.

## A single goodness-of-fit statistic compresses the wrong information for many tasks

Suppose two models have nearly identical root mean squared error:

[
operatorname{RMSE}_1
approx
operatorname{RMSE}_2.
]

One may systematically underpredict the upper tail while the other is well calibrated there.

If the scientific question concerns rare extreme events, those models are not equivalent.

Likewise, two probability models can have similar overall likelihood while differing in calibration for a subgroup that matters operationally.

A global fit statistic averages over discrepancies.

The scientific question may not.

This is why model adequacy should be defined relative to the intended use.

For estimating a mean, tail misspecification may have little consequence.

For estimating a one-in-a-thousand event probability, the same misspecification can dominate the answer.

For causal intervention, observational prediction can be nearly irrelevant if the model encodes the wrong direction of dependence.

The evaluation target should therefore resemble the inferential target.

## More parameters can improve fit while weakening identification

Increasing model flexibility almost always creates opportunities to fit observed data more closely.

Suppose model (M_1) is nested inside model (M_2). The larger model can often achieve

[
operatorname{RSS}(M_2)
le
operatorname{RSS}(M_1)
]

on the training data.

That inequality says nothing about whether the additional parameters are scientifically identified.

The extra flexibility may capture real structure.

It may also absorb noise, compensate for misspecification elsewhere, or create parameter trade-offs that leave interpretation unstable.

This is why penalised criteria, regularisation, cross-validation, prior structure, and experimental design exist.

The goal is not the smallest possible training residual.

The goal is to extract information that survives beyond the particular dataset used to tune the model.

## Structural and practical non-identifiability should be separated

Structural identifiability asks whether ideal observations generated by the model would uniquely determine its parameters.

Practical identifiability asks whether the actual data are sufficiently informative to estimate those parameters with useful precision.

The distinction is important.

In the product model

[
Y=abX+arepsilon,
]

(a) and (b) are structurally non-identifiable because only their product enters the observable distribution.

Now consider a different model in which the parameters are theoretically separable, but the experiment observes only a short time interval or a narrow input range. The likelihood may contain a long, nearly flat ridge. With ideal richer data the parameters could be separated. With the available experiment, they cannot.

That is practical non-identifiability.

Modern reviews in systems biology continue to emphasise both problems because fitting increasingly detailed dynamical models does not guarantee that the available experiments contain enough information to estimate every parameter.

A model can therefore reproduce a time series beautifully while its scientific interpretation remains highly unstable.

## Experimental perturbation tests more than passive fit

If several models explain the same observed regime, a useful scientific strategy is to create conditions under which their predictions diverge.

Suppose models (M_1) and (M_2) both fit historical data.

Under an intervention (A), they predict

[
mathbb E_{M_1}[Ymid do(A=a)]
=
5
]

and

[
mathbb E_{M_2}[Ymid do(A=a)]
=
12.
]

An experiment near that intervention is far more informative for discriminating between the models than collecting many more passive observations from a regime in which both predict approximately the same thing.

This is the connection between modelling and experimental design.

Data are informative when they stress the dimensions along which plausible models disagree.

The product-parameter example can also be solved this way. If an additional experiment measures a quantity that depends on (a) but not (b), or perturbs the system so that their effects enter separately, the ridge

[
ab=	heta
]

can be broken.

More data are valuable when they add new information.

Repeating the same uninformative experiment more precisely may only narrow uncertainty around the same unidentified combination.

## Sensitivity analysis asks whether the conclusion survives plausible alternatives

A model is often only one member of a defensible set of modelling choices.

Those choices may include the error distribution, link function, prior, covariate adjustment set, missing-data mechanism, functional form, treatment of outliers, lag structure, or boundary conditions.

If the scientific conclusion changes substantially across reasonable alternatives, that instability is part of the result.

Suppose one model estimates

[
hat	au=0.8
]

and a plausible alternative estimates

[
hat	au=-0.2.
]

Reporting only the first fit conceals the dependence of the conclusion on model specification.

Sensitivity analysis does not require treating every imaginable model as equally plausible.

Its purpose is to vary assumptions that are scientifically or statistically credible and determine which conclusions remain stable.

A robust conclusion survives changes that should not matter.

A fragile conclusion identifies where additional evidence is needed.

## Confirmation is always relative to alternatives and tests

Oreskes, Shrader-Frechette, and Belitz argued in the context of numerical models of natural systems that complete verification or validation is not available in the strong logical sense because natural systems are open and model-data agreement is nonunique. They distinguished that claim from empirical confirmation, where observations can support a model relative to the tests actually performed.

That distinction is useful well beyond earth science if it is not turned into scepticism about modelling itself.

Models accumulate credibility when they survive informative tests.

A model that predicts new data before seeing them is more credible than one fitted after the fact.

A model that predicts the result of an intervention gains evidence about its causal structure.

A model whose parameters remain stable across independent experiments gains evidence that they represent persistent features rather than compensating fit terms.

A model that fails targeted checks should be revised or abandoned for the use that exposed the failure.

The relevant standard is not metaphysical truth.

It is whether the model has survived tests capable of detecting errors that matter for the intended scientific task.

## Model usefulness is conditional on the question

The statement

[
	ext{this model fits well}
]

is incomplete.

We need to ask:

What feature of the data did it fit?

Over what range?

Under what distribution?

Using which observations for selection?

Which parameters are identified?

Which predictions were tested out of sample?

Which assumptions determine extrapolation?

Which causal claims require interventions rather than observations?

Which failure would materially change the scientific conclusion?

A model can be poor as a mechanistic explanation and excellent as a forecast.

Another can be adequate for estimating a population mean and poor for tail probabilities.

A third can predict the observed regime accurately but become useless after a policy intervention changes the data-generating process.

Calling all three simply "good models" discards the information needed to judge them.

## Fit is compatibility, not certification

Return to the three constructions.

The interpolation family

[
g_c(x)
=
f(x)
+
cprod_i(x-x_i)
]

shows that perfect fit to finite observations does not uniquely determine continuation away from those observations.

The product model

[
Y=abX+arepsilon
]

shows that perfect predictive fit can coexist with parameters that cannot be separately identified.

The two Gaussian structural models show that an entire observational distribution can be reproduced exactly by incompatible causal directions.

These are different failures.

No single diagnostic fixes all three.

Held-out prediction addresses overfitting within a target distribution.

Identifiability analysis addresses whether parameters can be uniquely recovered.

Experimental intervention can discriminate between observationally equivalent causal structures.

Sensitivity analysis tests dependence on modelling assumptions.

Posterior predictive checks and residual diagnostics test specific features of the fitted distribution.

Scientific theory restricts the set of functions and mechanisms considered plausible.

The appropriate question is therefore not whether a model fits.

It is what the fit has actually established.

A model deserves confidence when its assumptions, predictions, and interpretations have been exposed to tests that could have shown them to be inadequate.

Agreement with the data is necessary evidence in that process.

It is not the end of it.

## References

Box, G. E. P. (1976). Science and statistics. *Journal of the American Statistical Association*, 71(356), 791–799. https://doi.org/10.1080/01621459.1976.10480949

Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199–231. https://doi.org/10.1214/ss/1009213726

Cobelli, C., & DiStefano, J. J. (1980). Parameter and structural identifiability concepts and ambiguities: A critical review and analysis. *American Journal of Physiology*, 239(1), R7–R24. https://doi.org/10.1152/ajpregu.1980.239.1.R7

Gelman, A., & Shalizi, C. R. (2013). Philosophy and the practice of Bayesian statistics. *British Journal of Mathematical and Statistical Psychology*, 66(1), 8–38. https://doi.org/10.1111/j.2044-8317.2011.02037.x

Heinrich, M., Rosenblatt, M., Wieland, F. G., Stigter, H., & Timmer, J. (2025). On structural and practical identifiability: Current status and update of results. *Current Opinion in Systems Biology*, 41, 100546. https://doi.org/10.1016/j.coisb.2025.100546

Oreskes, N., Shrader-Frechette, K., & Belitz, K. (1994). Verification, validation, and confirmation of numerical models in the Earth sciences. *Science*, 263(5147), 641–646. https://doi.org/10.1126/science.263.5147.641

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25. https://doi.org/10.2307/1912526
