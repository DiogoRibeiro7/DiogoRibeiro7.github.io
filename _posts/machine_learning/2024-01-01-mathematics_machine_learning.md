---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2024-01-01'
excerpt: "The mathematics of machine learning is a study of risk, approximation, optimization, and generalization under finite data."
header:
  image: /assets/images/headers/photo-network-cables.jpg
  og_image: /assets/images/headers/photo-network-cables.jpg
  overlay_image: /assets/images/headers/photo-network-cables.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-network-cables.jpg
  twitter_image: /assets/images/headers/photo-network-cables.jpg
keywords:
- Statistical learning theory
- Empirical risk minimization
- Generalization
- Bayes risk
- VC dimension
- Rademacher complexity
permalink: '/machine-learning/mathematics_machine_learning/'
redirect_from:
- '/machine learning/mathematics_machine_learning/'
seo_description: "A mathematical introduction to machine learning through risk minimization, Bayes decision rules, regularization, generalization, and model complexity."
seo_title: "Mathematics of Machine Learning: Risk and Generalization"
seo_type: article
tags:
- Machine Learning
- Mathematical Modeling
- Statistics
title: "Mathematics of Machine Learning: Risk and Generalization"
---

Machine learning can be formulated as learning a rule from finite data that performs well on future observations. The mathematical difficulty is that the distribution generating future data is unknown.

Let $(X,Y)\sim P$ and let a predictor $f$ incur loss $\ell(f(X),Y)$. The population risk is

$$
R(f)=E_P[\ell(f(X),Y)].
$$

Since $P$ is unknown, we observe data $(X_i,Y_i)_{i=1}^n$ and compute empirical risk

$$
\widehat R_n(f)
=
\frac{1}{n}
\sum_{i=1}^n
\ell(f(X_i),Y_i).
$$

The central problem is not merely minimizing training loss. It is controlling the gap

$$
R(f)-\widehat R_n(f).
$$

## Bayes decision rules

The optimal predictor depends on the loss.

Under squared loss,

$$
\ell(a,y)=(a-y)^2,
$$

the risk-minimizing predictor is

$$
f^*(x)=E[Y\mid X=x].
$$

Under absolute loss, the conditional median is optimal.

For binary classification with 0-1 loss, the Bayes classifier predicts the more probable class:

$$
f^*(x)
=
\mathbf 1\{P(Y=1\mid X=x)>1/2\}.
$$

This is an important lesson: the target function is defined jointly by the data distribution and the loss.

## Surrogate losses

The 0-1 loss is difficult to optimize directly because it is discontinuous.

Classification methods therefore often use surrogate losses such as logistic loss or hinge loss.

For labels $y\in\{-1,1\}$, logistic loss is

$$
\ell(z,y)=\log(1+e^{-yz}).
$$

A surrogate should be chosen because optimizing it produces useful decisions under the target loss, not merely because it is differentiable.

## Function classes

Learning always restricts the candidate predictors to a hypothesis class $\mathcal F$.

Examples include:

- linear functions
- bounded-depth trees
- reproducing-kernel Hilbert spaces
- neural networks with specified architectures

A richer class can approximate more functions but can also fit more accidental sample structure.

The relevant trade-off is approximation error versus estimation error.

## Regularization

Regularization controls effective complexity.

Ridge regression solves

$$
\widehat\beta
=
\arg\min_\beta
\left[
\|y-X\beta\|_2^2
+
\lambda\|\beta\|_2^2
\right].
$$

Lasso replaces the $L_2$ penalty with

$$
\lambda\|\beta\|_1.
$$

Regularization can be interpreted through optimization, geometry, Bayesian priors, or complexity control depending on context.

The tuning parameter must be selected without contaminating the final test set.

## Generalization bounds

Statistical learning theory studies conditions under which empirical performance approximates population performance.

A schematic uniform convergence result has the form

$$
\sup_{f\in\mathcal F}
|R(f)-\widehat R_n(f)|
\le
\text{complexity term}
+
\text{confidence term}.
$$

The exact complexity measure depends on the setting.

VC dimension is useful for binary classification classes. Rademacher complexity measures how strongly a function class can correlate with random signs on the observed sample.

These bounds explain principles, but practical model selection should not be reduced to plugging a neural network into a worst-case theoretical inequality.

## Bias-variance decomposition

For squared-error prediction, a classical decomposition at a fixed input separates noise, squared bias, and variance.

This decomposition is useful for understanding why increasing flexibility can reduce approximation bias while increasing sensitivity to training data.

It is not a universal decomposition for every loss or algorithm.

Modern overparameterized models can also exhibit behavior not captured well by the simple textbook U-shaped curve.

## Cross-validation

Cross-validation estimates out-of-sample performance under a resampling scheme.

It is only valid for the deployment target if folds respect the dependence structure.

Random folds can leak information when observations share patients, machines, households, documents, or future time.

Every data-dependent choice must occur inside the cross-validation loop, including preprocessing, feature selection, and hyperparameter tuning.

## Decision trees

A regression tree partitions feature space into regions and predicts a constant within each region.

A classification tree may optimize impurity criteria such as Gini impurity,

$$
G=1-\sum_k p_k^2.
$$

Greedy splitting does not find a globally optimal tree in general.

Pruning, depth constraints, minimum leaf size, and ensembles control instability.

## Random forests

Random forests reduce tree variance by averaging decorrelated trees built from bootstrap samples and randomized candidate features.

Their success is best understood through ensemble averaging and low correlation between constituent errors, not through a claim that they eliminate overfitting.

## Distribution shift

Classical supervised learning often starts with IID assumptions:

$$
(X_i,Y_i)\stackrel{iid}{\sim}P.
$$

Real deployment may instead have training distribution $P$ and future distribution $Q$.

If

$$
P\neq Q,
$$

generalization from training data becomes a transport problem.

Covariate shift, label shift, concept drift, and selection bias describe different changes and need different responses.

## Conclusion

The mathematics of machine learning is not a list of algorithms. It is the study of how finite observations constrain future prediction.

The recurring structure is

$$
\text{loss}
+
\text{function class}
+
\text{optimization}
+
\text{regularization}
+
\text{validation}
+
\text{distribution assumptions}.
$$

Once these are explicit, many apparently different algorithms become instances of the same statistical learning problem.

## References

- Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning*.
- Vapnik, V. N. (1998). *Statistical Learning Theory*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
