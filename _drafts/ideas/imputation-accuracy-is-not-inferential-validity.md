---
author_profile: false
categories:
- Statistics
classes: wide
excerpt: A low imputation RMSE tells us how well missing values were reconstructed under a chosen masking experiment. It does not tell us whether downstream estimates, standard errors, confidence intervals, or scientific conclusions are valid.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- missing data
- imputation
- multiple imputation
- statistical inference
- uncertainty
- coverage
- regression
seo_description: Why low imputation RMSE or MAE does not guarantee unbiased estimates, valid standard errors, or correct confidence-interval coverage in downstream statistical analyses.
seo_title: Imputation Accuracy Is Not Inferential Validity
seo_type: article
summary: Reconstruction accuracy, predictive performance, and inferential validity are different targets. An imputation method should be evaluated against the quantity the analysis is actually trying to preserve.
tags:
- Missing Data
- Statistical Inference
- Imputation
- Uncertainty
title: 'Imputation Accuracy Is Not Inferential Validity'
---

A missing-data benchmark often ends with a table like this:

| Method | RMSE | MAE |
| --- | ---: | ---: |
| Method A | 0.42 | 0.31 |
| Method B | 0.47 | 0.34 |
| Method C | 0.55 | 0.39 |

It is tempting to read the first row as the winner.

Sometimes it is.

But that conclusion is valid only if the scientific target is **reconstruction under the benchmark's loss function**.

If the completed data will later be used to estimate a regression coefficient, a treatment effect, a variance, a correlation, a survival model, or a confidence interval, the ranking above may answer the wrong question.

The central distinction is:

$$
\boxed{
\text{reconstructing missing values accurately}
\neq
\text{preserving valid downstream inference}.
}
$$

This is not an argument against imputation. It is an argument for matching the evaluation criterion to the inferential object we actually care about.

## Two different statistical targets

Let $Y_{\mathrm{mis}}$ denote the missing values and let an imputation procedure produce

$$
\widehat Y_{\mathrm{mis}}.
$$

A reconstruction benchmark may evaluate

$$
L_{\mathrm{rec}}
=
\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}
\left(Y_i-\widehat Y_i\right)^2,
$$

or its square root, RMSE.

That is a perfectly legitimate loss if the task is to predict the missing values themselves.

But suppose the actual scientific target is a parameter

$$
\theta = T(P),
$$

such as a population mean, regression coefficient, odds ratio, hazard ratio, variance component, or causal estimand.

Then the relevant questions are different:

$$
\operatorname{Bias}(\widehat\theta),
$$

$$
\operatorname{Var}(\widehat\theta),
$$

and, for an interval $C_n$,

$$
P(\theta\in C_n).
$$

A method can perform well under $L_{\mathrm{rec}}$ while performing badly on one or all of these quantities.

There is no theorem saying that

$$
\arg\min_m E[L_{\mathrm{rec}}(m)]
=
\arg\min_m E[L_{\mathrm{inf}}(m)].
$$

Usually there is no reason to expect the two optimizers to coincide.

## A simple counterexample: mean imputation

Consider independent observations

$$
Y_1,\ldots,Y_n
$$

with

$$
E[Y_i]=\mu,
\qquad
\operatorname{Var}(Y_i)=\sigma^2.
$$

Suppose each value is observed independently with probability $p$, so the missing fraction is approximately $1-p$.

Let $m$ be the number of observed values and let

$$
\bar Y_{\mathrm{obs}}
$$

be their mean.

Now fill every missing value with $\bar Y_{\mathrm{obs}}$.

The completed-data mean is then exactly

$$
\bar Y_{\mathrm{imp}}
=
\bar Y_{\mathrm{obs}}.
$$

Under missing completely at random, this is still an unbiased estimator of $\mu$.

So far, so good.

But the inferential problem appears in the variance.

Conditional on $m$,

$$
\operatorname{Var}(\bar Y_{\mathrm{imp}}\mid m)
=
\frac{\sigma^2}{m}.
$$

For large $n$, $m/n\to p$, so

$$
\operatorname{Var}(\bar Y_{\mathrm{imp}})
\approx
\frac{\sigma^2}{np}.
$$

Now inspect the variance of the **completed values**. Every imputed observation is placed exactly at the observed mean, so the missing cases contribute no squared deviation at all. The completed-data sample variance converges to approximately

$$
p\sigma^2.
$$

If we now pretend the imputed values were genuinely observed and calculate the usual standard error from all $n$ completed rows, we get

$$
\widehat{\operatorname{SE}}_{\mathrm{naive}}
\approx
\sigma\sqrt{\frac{p}{n}}.
$$

But the actual sampling standard error of the completed-data mean is approximately

$$
\operatorname{SE}_{\mathrm{true}}
\approx
\frac{\sigma}{\sqrt{np}}.
$$

Therefore

$$
\frac{
\widehat{\operatorname{SE}}_{\mathrm{naive}}
}{
\operatorname{SE}_{\mathrm{true}}
}
\approx p.
$$

That is a large distortion.

If half the values are observed, $p=0.5$, the naive completed-data standard error is asymptotically only about **half** the correct sampling standard error.

Under a normal approximation, a nominal 95% interval using that naive standard error has approximate coverage

$$
2\Phi(1.96p)-1.
$$

For several observed fractions:

| Observed fraction $p$ | Missing fraction | Approximate coverage of naive nominal 95% interval |
| ---: | ---: | ---: |
| 0.90 | 0.10 | 92.2% |
| 0.80 | 0.20 | 88.3% |
| 0.70 | 0.30 | 83.0% |
| 0.50 | 0.50 | 67.3% |

The point estimate can therefore be fine while the reported uncertainty is badly wrong.

This is exactly the sort of failure that an imputation RMSE leaderboard will not reveal.

## Why point prediction naturally removes uncertainty

The same issue appears more generally.

Suppose

$$
Y=\beta_0+\beta_1X+\varepsilon,
\qquad
E[\varepsilon\mid X]=0,
\qquad
\operatorname{Var}(\varepsilon\mid X)=\sigma^2.
$$

Under squared-error loss, the optimal point predictor is the conditional mean

$$
E[Y\mid X]
=
\beta_0+\beta_1X.
$$

If missing outcomes are replaced by this conditional mean, the imputed observations lie on the regression surface without residual noise.

That is not a bug in point prediction. It is what squared-error point prediction asks for.

But the residual variation is scientifically relevant if we later want to estimate

- $\sigma^2$;
- standard errors for $\beta$;
- predictive intervals;
- correlations;
- tail probabilities;
- heterogeneity;
- nonlinear downstream functionals.

The imputer has optimized one task by deliberately suppressing variation needed for another.

So the problem is deeper than saying that a particular imputation algorithm is too simple.

Even an excellent conditional-mean predictor can be the wrong object for inferential completion.

## A point prediction is not a draw from a predictive distribution

This distinction is easy to express mathematically.

A deterministic predictor returns something like

$$
\widehat y
\approx
E[Y_{\mathrm{mis}}\mid Y_{\mathrm{obs}},X].
$$

Inferential procedures may instead need to acknowledge the distribution

$$
Y_{\mathrm{mis}}
\mid
Y_{\mathrm{obs}},X,\psi,
$$

as well as uncertainty about the model parameter $\psi$.

Replacing a distribution with its conditional mean preserves neither its variance nor all of its relationships with other variables.

This is one reason single deterministic imputation usually makes conventional standard errors too small when the imputed values are subsequently treated as observed.

## Multiple imputation is designed for a different target

Multiple imputation does not try to find one completed dataset that is closest to the unknowable true dataset.

Instead, it creates multiple plausible completed datasets under an imputation model, analyzes each one, and combines the resulting estimates.

Let

$$
\widehat Q_1,\ldots,\widehat Q_M
$$

be estimates of the scientific quantity from $M$ imputed datasets, with within-imputation variance estimates

$$
U_1,\ldots,U_M.
$$

The pooled estimate is

$$
\bar Q
=
\frac{1}{M}\sum_{m=1}^{M}\widehat Q_m.
$$

The average within-imputation variance is

$$
\bar U
=
\frac{1}{M}\sum_{m=1}^{M}U_m,
$$

and the between-imputation variance is

$$
B
=
\frac{1}{M-1}
\sum_{m=1}^{M}
(\widehat Q_m-\bar Q)^2.
$$

Rubin's total variance combines the two:

$$
T
=
\bar U
+
\left(1+\frac{1}{M}\right)B.
$$

The important conceptual point is not the formula itself.

It is that the uncertainty between plausible completions is treated as **evidence about uncertainty**, rather than averaged away and forgotten.

That is a different goal from minimizing pointwise reconstruction error.

## But multiple imputation is not automatically valid

This does not mean that any stochastic imputation followed by Rubin's rules produces valid inference.

The imputation model matters.

Suppose the substantive model contains

$$
Y
=
\beta_0
+
\beta_1X
+
\beta_2X^2
+
\beta_3XZ
+
\varepsilon.
$$

An imputation procedure for missing $X$ that ignores the nonlinear term $X^2$ and interaction $XZ$ can fail to preserve the relationships required by the analysis model.

The imputation and substantive models can be **incompatible**.

This is why substantive-model-compatible approaches exist: the missing-data model has to respect the structure of the scientific analysis when that structure matters for the estimand.

A low imputation RMSE does not diagnose this incompatibility.

## Artificial masking is itself an assumption

A common benchmark starts from a complete dataset, hides a random subset of observed values, imputes them, and compares the imputations with the held-out truth.

That design is useful.

But it evaluates a specific missingness experiment.

If values are masked independently and uniformly, the benchmark is approximately asking:

> How well does this method reconstruct data under an MCAR-like masking process?

Real missingness may instead depend on observed variables:

$$
P(R=1\mid X,Y)
=
P(R=1\mid X),
$$

or may depend on the missing value itself even after conditioning on observed information.

A method that dominates under random masking may therefore perform differently under the actual observation process.

This matters because we can usually validate reconstruction only on values that were observed in the first place.

The missing values we most care about are exactly the values whose truth is unavailable.

So a masked-value benchmark should be described honestly as a **designed validation experiment**, not as direct evidence that the real missing values have been recovered correctly.

## The evaluation metric should follow the scientific target

A useful missing-data benchmark should begin by asking what the completed data will be used for.

| Downstream goal | Primary evaluation target | Useful diagnostics |
| --- | --- | --- |
| Recover individual missing values | reconstruction error | RMSE, MAE, calibration, interval coverage |
| Preserve a marginal distribution | distributional fidelity | means, variances, quantiles, tails, distributional distances |
| Build a predictor | out-of-sample predictive loss | log loss, Brier score, RMSE, calibration, subgroup performance |
| Estimate a model parameter | inferential validity | bias, empirical variance, estimated SE, confidence-interval coverage |
| Estimate a causal effect | estimand validity under missingness assumptions | bias, coverage, sensitivity to missingness model |
| Make a decision | expected decision loss | utility, cost, regret, intervention performance |

No single row dominates the others.

The correct metric depends on the task.

## Reconstruction metrics can still be extremely useful

RMSE and MAE are not bad metrics.

They answer a real question:

$$
\boxed{
\text{How close are the imputations to held-out values under this masking design?}
}
$$

That is useful when

- the missing value itself is the product;
- a downstream prediction system primarily needs accurate feature reconstruction;
- we are comparing point-prediction behavior;
- reconstruction is one component of a broader validation suite.

The mistake is not measuring RMSE.

The mistake is silently promoting RMSE into evidence for every downstream statistical claim.

## Distribution matching is also insufficient

A natural reaction is to add distributional diagnostics.

That helps, but it does not solve the inferential problem by itself.

Two completed datasets can have nearly identical marginal means, variances, and histograms while differing materially in

$$
\operatorname{Cov}(X,Y),
$$

conditional relationships,

$$
E[Y\mid X],
$$

interactions,

$$
E[Y\mid X,Z],
$$

or tail dependence.

If the scientific parameter depends on those structures, marginal distribution matching can still declare victory too early.

The object to preserve is the part of the **joint distribution relevant to the estimand**.

## A benchmark for inference should be simulation-based

When the goal is inference, one of the cleanest evaluation designs is a repeated simulation study.

Start from a known data-generating process

$$
W_i\sim P_{\theta_0}.
$$

Then, for each replicate:

1. generate a complete dataset;
2. impose a specified missingness mechanism;
3. apply each missing-data method;
4. fit the same substantive model;
5. store $\widehat\theta$ and its reported standard error or interval.

Across replications, calculate

$$
\operatorname{Bias}
=
E[\widehat\theta]-\theta_0,
$$

empirical standard deviation

$$
\operatorname{SD}_{\mathrm{emp}}(\widehat\theta),
$$

average model-based standard error

$$
E[\widehat{\operatorname{SE}}],
$$

and interval coverage

$$
P(\theta_0\in C_n).
$$

Then compare those quantities with reconstruction RMSE.

That comparison is scientifically interesting precisely because the rankings may disagree.

## The missingness mechanism should be part of the benchmark grid

A serious benchmark should not have only one masking probability.

At minimum, vary

- missing fraction;
- which variables are incomplete;
- MCAR versus plausible MAR mechanisms;
- strength of predictors of missingness;
- overlap or positivity in the observed-data patterns;
- nonlinearities and interactions in the substantive model;
- distribution tails and outliers;
- sample size.

For MNAR questions, no observed-data benchmark can prove that the unverifiable mechanism is correct. The role of the benchmark changes to **sensitivity analysis**: how much do conclusions move across scientifically plausible departures from the primary missingness assumption?

## Prediction pipelines have a different failure mode: leakage

When imputation is part of a machine-learning pipeline, the main inferential issue may not be Rubin's rules at all.

The critical requirement is often that imputation be fitted **inside the training split**.

If an imputer is trained on the entire dataset before cross-validation, information from validation or test rows can leak into the training process.

Then even the downstream predictive score is optimistic.

So for predictive tasks the correct object is usually the whole pipeline:

$$
\boxed{
\text{split}
\rightarrow
\text{fit imputer on training data}
\rightarrow
\text{transform}
\rightarrow
\text{fit predictor}
\rightarrow
\text{evaluate held-out data}.
}
$$

Again, the evaluation target follows the final use.

## Complete-case analysis is not merely the "bad baseline"

It is also a mistake to assume that the most sophisticated imputation method must always dominate complete-case analysis.

The validity and efficiency of complete-case estimation depend on

- which variables are missing;
- the missingness mechanism;
- the substantive model;
- whether the complete cases remain representative for the estimand;
- how much information is lost.

A useful benchmark should therefore include complete-case analysis when it is meaningful, not as a straw man but as a scientific reference point.

The comparison may reveal whether complexity is actually buying bias reduction, efficiency, or neither.

## Likelihood and Bayesian approaches do not require a completed dataset

Imputation is only one way to handle missing data.

Under a joint model with parameter $\theta$, a likelihood approach can integrate over the missing components:

$$
L(\theta;Y_{\mathrm{obs}})
=
\int
L(\theta;Y_{\mathrm{obs}},Y_{\mathrm{mis}})
\,dY_{\mathrm{mis}}.
$$

A Bayesian analysis can similarly average posterior uncertainty over missing quantities and model parameters.

These methods make an important conceptual point:

$$
\boxed{
\text{valid inference does not require pretending the missing values became observed.}
}
$$

Sometimes data completion is computationally convenient. It is not the scientific objective.

## The right question is not "Which imputer wins?"

The stronger question is

$$
\boxed{
\text{Which method preserves the quantity I need, under assumptions I can defend?}
}
$$

That question may produce different winners for different tasks.

A nearest-neighbor method may be excellent for reconstructing individual values.

A stochastic imputation model may be preferable for preserving uncertainty.

A substantive-model-compatible multiple-imputation procedure may be needed for nonlinear regression inference.

A likelihood analysis may avoid imputation entirely.

A sensitivity analysis may be more important than choosing between two algorithms when the missingness mechanism is the dominant uncertainty.

This is not a weakness of missing-data methodology.

It is what happens whenever different statistical losses correspond to different scientific goals.

## A practical evaluation protocol

Before benchmarking an imputation method, write down five things.

### 1. The downstream target

Is the goal a missing value, a prediction, a parameter, an interval, or a decision?

### 2. The missingness model

What observation process is the benchmark assuming, and why is it relevant to the real application?

### 3. The analysis model

Which relationships, nonlinearities, interactions, temporal dependencies, or censoring structures must the imputation preserve?

### 4. The uncertainty criterion

If inference matters, will the benchmark check estimated standard errors against empirical variation and test interval coverage?

### 5. The failure regime

Under what missing fractions, distribution shifts, weak overlap, or model misspecifications does the method stop being trustworthy?

Only then should we look at a leaderboard.

## The broader lesson

Imputation is a particularly clear example of a general statistical principle:

$$
\boxed{
\text{an algorithm is only optimal relative to a loss function.}
}
$$

A method that minimizes reconstruction loss has solved the reconstruction problem.

It has not automatically solved the inference problem.

If our final claim concerns a coefficient, uncertainty interval, causal effect, or decision, that object must appear explicitly in the validation design.

Otherwise we can produce a beautifully optimized answer to the wrong question.

## References

- Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley; 1987.
- Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Wiley; 2019.
- Sterne JAC, White IR, Carlin JB, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*. 2009;338:b2393. [doi:10.1136/bmj.b2393](https://doi.org/10.1136/bmj.b2393).
- White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. *Statistics in Medicine*. 2011;30(4):377–399. [doi:10.1002/sim.4067](https://doi.org/10.1002/sim.4067).
- Bartlett JW, Seaman SR, White IR, Carpenter JR. Multiple imputation of covariates by fully conditional specification: accommodating the substantive model. *Statistical Methods in Medical Research*. 2015;24(4):462–487. [doi:10.1177/0962280214521348](https://doi.org/10.1177/0962280214521348).
- van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. Chapman & Hall/CRC; 2018.
