---
permalink: '/statistics/error_coefficientes/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-01-01'
excerpt: Linear and logistic regression encode randomness differently. The distinction is between an additive disturbance model and a Bernoulli conditional distribution, not between having and not having error.
header:
  image: /assets/images/headers/photo-statistics-ecdf.jpg
  og_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-ecdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-ecdf.jpg
keywords:
- Error terms
- Linear regression
- Logistic regression
- Residuals
- Bernoulli likelihood
- Regression diagnostics
seo_description: How linear and logistic regression encode stochastic variation, why residuals differ from model errors, and how Gaussian and Bernoulli assumptions affect inference.
seo_title: 'Error Terms in Linear and Logistic Regression'
seo_type: article
summary: A rigorous comparison of stochastic variation in linear and logistic regression, separating latent model errors from fitted residuals and clarifying the assumptions needed for estimation and inference.
tags:
- Regression
- Statistical Modeling
title: 'Error Terms in Linear and Logistic Regression'
---

Linear and logistic regression both model a conditional distribution. The difference is not that one model has “error” and the other does not. The difference is how random variation is represented. For a Gaussian linear model,

$$
Y_i
=
X_i^\top\beta
+
\varepsilon_i.
$$

For binary logistic regression,

$$
Y_i\mid X_i
\sim
\operatorname{Bernoulli}(p_i),
$$

with

$$
\operatorname{logit}(p_i)
=
X_i^\top\beta.
$$

The first writes a disturbance term explicitly. The second places randomness directly in the Bernoulli conditional distribution.

## Model error is not the same thing as residual

In linear regression, the latent error is

$$
\varepsilon_i
=
Y_i
-
E(Y_i\mid X_i).
$$

It is part of the data-generating model and is not observed directly. After fitting, the residual is

$$
e_i
=
Y_i
-
\hat Y_i.
$$

Residuals depend on the estimated model and therefore are not identical to the latent errors. In matrix form,

$$
e
=
(I-H)Y,
$$

where

$$
H
=
X(X^\top X)^{-1}X^\top
$$

is the hat matrix. This means residuals have a covariance structure induced by the fitted model even when the original errors are independent.

## The linear conditional mean

The central regression model is

$$
E(Y\mid X)
=
X^\top\beta.
$$

Ordinary least squares estimates

$$
\hat\beta
=
\arg\min_b
\sum_i
(Y_i-X_i^\top b)^2.
$$

If

$$
E(\varepsilon\mid X)=0,
$$

then under the classical fixed-design argument,

$$
E(\hat\beta\mid X)=\beta.
$$

Normality is not required for this unbiasedness result.

## What Gaussian errors add

If we assume

$$
\varepsilon\mid X
\sim
\mathcal N(0,\sigma^2I),
$$

then OLS is also the maximum-likelihood estimator for $\beta$ and exact finite-sample t and F distributions become available. That is a stronger assumption than OLS estimation itself requires. The usual hierarchy is:

- conditional mean zero for unbiasedness;
- standard regularity conditions for consistency;
- homoskedastic uncorrelated errors for Gauss-Markov efficiency among linear unbiased estimators;
- Gaussian errors for exact normal-theory likelihood and finite-sample inference.

These conclusions should not be merged into one statement such as “normal errors make OLS unbiased and efficient.”

## Heteroskedasticity

If

$$
\operatorname{Var}(\varepsilon_i\mid X_i)
=
\sigma_i^2,
$$

OLS coefficients can remain unbiased or consistent under the appropriate exogeneity assumptions. What fails is the ordinary homoskedastic covariance formula. A heteroskedasticity-consistent covariance estimator can then be used for inference. So variance misspecification and conditional-mean misspecification are different problems.

## Logistic regression has a stochastic model

For binary outcomes,

$$
Y_i\mid X_i
\sim
\operatorname{Bernoulli}(p_i),
$$

where

$$
p_i
=
\frac{
1
}{
1+\exp(-X_i^\top\beta)
}.
$$

The conditional mean is

$$
E(Y_i\mid X_i)
=
p_i,
$$

and the conditional variance is

$$
\operatorname{Var}(Y_i\mid X_i)
=
p_i(1-p_i).
$$

The random variation is therefore explicit in the Bernoulli distribution. There is no need to add an independent Gaussian error to the logit equation.

## The likelihood

For independent observations, the Bernoulli likelihood is

$$
L(\beta)
=
\prod_{i=1}^{n}
p_i^{y_i}
(1-p_i)^{1-y_i}.
$$

The log-likelihood is

$$
\ell(\beta)
=
\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right].
$$

Maximum likelihood chooses

$$
\hat\beta
=
\arg\max_\beta
\ell(\beta).
$$

The likelihood is not “the error term.” It is the probability model used to estimate the coefficients.

## Logistic residuals exist

Logistic regression has several useful residual definitions. The response residual is

$$
e_i
=
y_i-\hat p_i.
$$

The Pearson residual is

$$
r_i^P
=
\frac{
y_i-\hat p_i
}{
\sqrt{
\hat p_i(1-\hat p_i)
}
}.
$$

Deviance residuals measure the signed contribution of each observation to model deviance. These residuals can reveal lack of fit, unusual observations, or systematic structure not captured by the model. So “logistic regression has no residuals” is false.

## Latent-variable representation

Logistic regression can also be represented through a latent variable:

$$
Y_i
=
I(Y_i^\ast>0),
$$

with

$$
Y_i^\ast
=
X_i^\top\beta
+
\varepsilon_i,
$$

where $\varepsilon_i$ follows a logistic distribution. This representation helps explain why the logit link appears. But the latent scale is not directly observed, so the coefficient scale depends on the fixed logistic error distribution.

## Coefficients live on different scales

In linear regression,

$$
\beta_j
$$

is a conditional change in the mean response per unit change in predictor $j$, holding the other modeled predictors fixed. In logistic regression,

$$
\beta_j
$$

is a conditional change in log-odds. Exponentiating gives an odds ratio:

$$
\exp(\beta_j).
$$

Neither coefficient is automatically causal. That interpretation requires a causal design or identification assumptions.

## Classification metrics do not replace model diagnostics

A logistic model can have high classification accuracy and still be poorly calibrated. AUC can be high while predicted probabilities are systematically too extreme. Useful diagnostics include:

- calibration plots;
- Brier score;
- log loss;
- residuals;
- leverage and influence;
- separation checks;
- out-of-sample validation.

The statistical model and the classification decision are related but distinct layers.

## Dependence

Both linear and logistic regression can be misspecified when observations are dependent. Examples include:

- repeated measurements;
- patients within hospitals;
- time series;
- spatial data;
- family clusters.

For binary repeated measures, options include GEE, mixed-effects logistic regression, or cluster-robust inference depending on the estimand. The Bernoulli mean model alone does not define dependence among observations.

## Overdispersion and binary data

For one Bernoulli observation,

$$
\operatorname{Var}(Y_i\mid X_i)
=
p_i(1-p_i)
$$

is fixed by the mean. But grouped binomial data can exhibit extra-binomial variation due to unmodeled heterogeneity or dependence. That can be handled through beta-binomial models, random effects, quasi-likelihood, or robust covariance methods depending on the source of variation.

## Conclusion

Linear and logistic regression do not differ because one “has error” and the other does not. They differ because they specify different conditional distributions:

$$
\boxed{
\text{linear regression}
\rightarrow
\text{additive conditional-mean model}
}
$$

$$
\boxed{
\text{logistic regression}
\rightarrow
\text{Bernoulli conditional distribution with logit link}
}
$$

Residuals are fitted diagnostics in both settings. The inferential assumptions belong to the full probability model, not to a vague idea of “error handling.”

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- Pregibon, D. (1981). Logistic regression diagnostics. *Annals of Statistics*, 9(4), 705–724.
