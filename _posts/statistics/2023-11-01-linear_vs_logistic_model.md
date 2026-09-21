---
permalink: '/statistics/linear_vs_logistic_model/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-11-01'
excerpt: The linear probability model and logistic regression target the same conditional event probability through different functional forms. The choice depends on estimands, extrapolation, inference, and communication.
header:
  image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  og_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  overlay_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  twitter_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
keywords:
- Linear probability model
- Logistic regression
- Marginal effects
- Risk difference
- Odds ratio
- Binary outcome
redirect_from:
- '/probability modeling/linear_vs_logistic_model/'
seo_description: A comparison of linear probability models and logistic regression, focusing on risk differences, odds ratios, heteroskedasticity, marginal effects, and prediction.
seo_title: Linear Probability Models vs Logistic Regression
seo_type: article
tags:
- Probability
- Regression
- Statistical Modeling
title: Linear Probability Models vs Logistic Regression
---

For a binary response Y, both the linear probability model and logistic regression attempt to describe

$$
p(x)=P(Y=1\mid X=x).
$$

They differ in how p(x) is parameterized.

## Linear probability model

The linear probability model writes

$$
E(Y\mid X)=X^\top\beta.
$$

Because E(Y|X) is the event probability for binary Y, each coefficient has an additive probability interpretation within the specified model.

For a continuous predictor X_j, beta_j is the change in predicted probability per one-unit change in X_j, holding other regressors fixed.

This direct risk-difference scale is often useful.

## The main limitations of the LPM

The conditional variance of a Bernoulli outcome is

$$
\operatorname{Var}(Y\mid X)=p(X)[1-p(X)],
$$

so errors are inherently heteroskedastic. Ordinary homoskedastic OLS standard errors are therefore inappropriate; heteroskedasticity-robust standard errors should generally be used.

The fitted line can also produce predictions below zero or above one, especially under extrapolation or strong covariate effects.

These are structural limitations, not merely cosmetic issues.

## Logistic regression

Logistic regression constrains probabilities to (0,1) through

$$
\operatorname{logit}p(x)
=
\log\frac{p(x)}{1-p(x)}
=
X^\top\beta.
$$

Hence

$$
p(x)=\frac{1}{1+\exp(-X^\top\beta)}.
$$

A coefficient beta_j is a change in log odds. Exponentiating it gives a conditional odds ratio for a one-unit predictor change under the model.

## Odds ratios are not risk ratios

If the odds ratio is 2, the probability does not generally double.

Starting from probability p_0, doubling the odds gives

$$
p_1=\frac{2p_0}{1-p_0+2p_0}.
$$

If p_0=0.40, then p_1 is about 0.571, not 0.80.

This is why communicating logistic coefficients only as odds ratios can obscure the practical magnitude of an effect.

## Marginal effects

Logistic regression can still be reported on the probability scale.

For a continuous predictor,

$$
\frac{\partial p(x)}{\partial x_j}
=
\beta_j p(x)[1-p(x)].
$$

The probability-scale effect therefore varies with x.

Average marginal effects summarize this quantity over the sample and can be easier to communicate than odds ratios.

## The models estimate different functional forms

The LPM assumes additivity on the probability scale. Logistic regression assumes additivity on the log-odds scale.

Neither functional form is automatically true. Flexible terms, splines, interactions, or nonparametric methods may be needed.

The fact that logistic regression respects probability bounds does not guarantee that its conditional mean specification is correct.

## Inference versus prediction

For prediction, compare out-of-sample calibration and loss. A logistic model is often a natural baseline because it returns bounded probabilities, but an LPM can predict well in a restricted covariate region.

For causal or explanatory analysis, the relevant scale matters. A risk difference, risk ratio, and odds ratio are different estimands and answer different questions.

Non-collapsibility also means an adjusted logistic odds ratio can differ from an unadjusted odds ratio even in the absence of confounding. This complicates comparisons across model specifications.

## Rare outcomes

When outcomes are rare, odds ratios can numerically approximate risk ratios, but this is an approximation and should not be generalized to common outcomes.

Rare events also create practical estimation problems such as separation, where maximum-likelihood logistic coefficients can diverge.

Penalized or bias-reduced methods may then be appropriate.

## Which should be used?

Use the model whose estimand and functional form match the scientific or predictive goal.

The LPM is useful when additive risk differences are central and predictions stay in a reasonable range. Logistic regression is useful when bounded probabilities and multiplicative odds structure are appropriate.

Both require model checking, robust validation, and careful interpretation.

## Conclusion

The real choice is not between a simple model and a sophisticated model. It is between different scales for the conditional probability.

Report effects on the scale people need to understand, and do not let coefficient convenience determine the scientific question.

## References

- Agresti, A. (2013). *Categorical Data Analysis*.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression*.
