---
permalink: '/mathematics/regularization_machine_learning/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-16'
header:
  image: /assets/images/headers/photo-mathematics-knot-projection.jpg
  og_image: /assets/images/headers/photo-mathematics-knot-projection.jpg
  overlay_image: /assets/images/headers/photo-mathematics-knot-projection.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-knot-projection.jpg
  twitter_image: /assets/images/headers/photo-mathematics-knot-projection.jpg
seo_description: "Regularization controls effective model complexity through penalties, priors, constraints, and early stopping; its effect depends on scale, geometry, and validation."
seo_title: "Regularization in Machine Learning: Geometry and Bias"
seo_type: article
tags:
- Machine Learning
- Regularization
- Statistics
title: "Regularization in Machine Learning: Geometry and Bias"
---

Regularization changes an estimation problem so that some solutions are preferred over others.

A generic penalized objective is

$$
\widehat\theta
=
\arg\min_\theta
\left[
L(\theta)
+
\lambda\Omega(\theta)
\right].
$$

The loss $L$ measures fit.

The penalty $\Omega$ encodes a preference.

The tuning parameter $\lambda$ controls the trade-off.

## Ridge regression

Ridge uses

$$
\Omega(\beta)=\|\beta\|_2^2.
$$

Thus

$$
\widehat\beta_{ridge}
=
\arg\min_\beta
\left[
\|y-X\beta\|_2^2
+
\lambda\|\beta\|_2^2
\right].
$$

Ridge shrinks coefficients toward zero but generally does not set them exactly to zero.

It is especially useful with correlated predictors and ill-conditioned design matrices.

## Lasso

Lasso uses

$$
\Omega(\beta)=\|\beta\|_1.
$$

The geometry of the $L_1$ constraint encourages sparse solutions.

But a zero coefficient does not prove a feature is scientifically irrelevant.

With correlated predictors, lasso can choose one variable and exclude another almost arbitrarily.

## Elastic net

Elastic net combines penalties:

$$
\lambda_1\|\beta\|_1
+
\lambda_2\|\beta\|_2^2.
$$

It can stabilize selection when groups of predictors are correlated.

## Standardization matters

Penalties act on coefficient magnitude.

If predictors have different numerical scales, equal coefficient penalties imply unequal effects on the original variables.

For linear models, standardized predictors are therefore often appropriate before ridge or lasso.

The scaling transformation must be fitted on training data only.

## Bias is intentional

Regularization introduces bias to reduce variance.

That is not a defect.

The relevant quantity is predictive or inferential error, not unbiasedness in isolation.

In finite samples, a biased estimator can have lower mean squared error than an unstable unbiased one.

## Bayesian interpretation

Ridge corresponds to a Gaussian prior on coefficients under a Gaussian likelihood.

Lasso corresponds to a Laplace prior in one common Bayesian interpretation.

This connection is useful, but penalized optimization and full Bayesian inference are not identical.

A posterior distribution contains uncertainty information that a penalized point estimate does not.

## Early stopping

Regularization does not require an explicit penalty.

Stopping gradient optimization before complete training can limit effective model complexity.

In overparameterized systems, optimization dynamics themselves can create implicit regularization.

## Dropout

Dropout randomly removes units during neural-network training.

It can reduce co-adaptation and acts as a stochastic regularizer.

Its effect is architecture- and optimization-dependent; it should not be described as a universal cure for overfitting.

## Data augmentation

Augmentation can be viewed as regularization when transformations encode invariances we want the model to learn.

If the augmentation changes the true label, it injects systematic error instead.

The transformation must reflect valid domain symmetry.

## Hyperparameter tuning

Choosing $\lambda$ on the final test set invalidates the test.

Regularization strength must be selected inside cross-validation or a validation set.

If feature selection is induced by lasso, that selection is also part of the tuning process.

## Regularization and causality

Regularization optimizes statistical behavior, not causal identification.

Shrinking or deleting a confounder because it improves predictive validation can bias a treatment-effect estimate.

Causal adjustment sets should follow identification logic.

## Double descent and modern models

The classic bias-variance story suggests a U-shaped test-error curve.

Modern overparameterized models can display double-descent behavior.

This does not make regularization obsolete.

It means model complexity, interpolation, optimization, and implicit bias interact in richer ways than the simplest textbook picture.

## Conclusion

Regularization is a structural assumption imposed on the solution.

The practical questions are:

- what behavior does the penalty favor?
- how does scaling affect it?
- how is the tuning parameter chosen?
- is the target prediction or inference?
- what uncertainty remains after selection?

Regularization works because it restricts or biases the model in useful ways, not because smaller coefficients are intrinsically better.

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net.
