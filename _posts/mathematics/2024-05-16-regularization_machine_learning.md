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
seo_description: "A rigorous treatment of regularization as constrained estimation, covering ridge, lasso, elastic net, bias-variance trade-offs, geometry, scaling, Bayesian connections, tuning, selection, and causal caveats."
seo_title: "Regularization: Geometry, Bias, and Statistical Control"
seo_type: article
subtitle: "Penalized estimation, model complexity, and the assumptions hidden in shrinkage"
tags:
- Machine Learning
- Regularization
- Statistics
- Regression
title: "Regularization: Geometry, Bias, and Statistical Control"
---

Regularization is often introduced as a practical device for preventing overfitting, but that description is too shallow to explain what the method actually does. A regularized estimator modifies the original estimation problem so that some parameter configurations are preferred over others. In this sense, regularization is not merely a numerical trick applied after a model has been specified; it is part of the model specification itself. The penalty, constraint, prior, or optimization rule encodes assumptions about which solutions are plausible, stable, or desirable, and the resulting estimator deliberately trades fit to the observed sample for improved behavior outside that sample.

A generic penalized estimator can be written as

$$
\widehat\theta_\lambda
=
\arg\min_\theta
\left\{
L(\theta)
+
\lambda\Omega(\theta)
\right\},
$$

where $L(\theta)$ is a loss function, $\Omega(\theta)$ is a penalty, and $\lambda\ge0$ controls the strength of regularization. When $\lambda=0$, the estimator reduces to the unpenalized problem. As $\lambda$ increases, parameter configurations with large penalty values become progressively less attractive. The practical effect depends entirely on the geometry of $\Omega$, the scale of the predictors, the form of the loss, and the purpose of the analysis. Saying that regularization “reduces complexity” is therefore only the beginning; the central question is what kind of complexity is being penalized and why.

## Ridge regression and the geometry of shrinkage

For linear regression with response vector $y\in\mathbb R^n$ and design matrix $X\in\mathbb R^{n\times p}$, ridge regression solves

$$
\widehat\beta_{\text{ridge}}
=
\arg\min_\beta
\left\{
\|y-X\beta\|_2^2
+
\lambda\|\beta\|_2^2
\right\}.
$$

The $L_2$ penalty shrinks coefficients continuously toward zero. It is equivalent to solving the least-squares problem subject to a Euclidean constraint,

$$
\|\beta\|_2^2\le c,
$$

for an appropriate correspondence between $c$ and $\lambda$. Geometrically, ordinary least squares seeks the point minimizing the residual sum of squares, whereas ridge restricts the solution to lie inside an $L_2$ ball. The smooth boundary of that ball means that coefficients are usually reduced in magnitude rather than set exactly to zero.

When $X^\top X$ is invertible, the ridge estimator has the closed form

$$
\widehat\beta_{\text{ridge}}
=
\left(
X^\top X+\lambda I
\right)^{-1}
X^\top y.
$$

This expression reveals one reason ridge is effective in ill-conditioned problems. If $X^\top X$ has very small eigenvalues, the ordinary least-squares solution can be extremely sensitive to small perturbations in the data. Adding $\lambda I$ raises the eigenvalues away from zero and stabilizes the inversion. Ridge therefore reduces variance not because small coefficients are inherently preferable, but because unstable directions in parameter space are deliberately damped.

The singular-value decomposition makes this even clearer. Suppose

$$
X=UDV^\top,
$$

with singular values $d_1,\ldots,d_r$. Then ridge multiplies the least-squares contribution along singular direction $j$ by approximately

$$
\frac{d_j^2}{d_j^2+\lambda}.
$$

Directions with large singular values are only mildly shrunk, while weakly identified directions with small singular values are shrunk much more strongly. Ridge is therefore a form of spectral stabilization.

## Lasso, sparsity, and the ambiguity of selection

The lasso replaces the quadratic penalty with the $L_1$ norm,

$$
\widehat\beta_{\text{lasso}}
=
\arg\min_\beta
\left\{
\|y-X\beta\|_2^2
+
\lambda\|\beta\|_1
\right\}.
$$

Equivalently, it constrains the solution to an $L_1$ ball. The geometry of that constraint is fundamentally different from the smooth ridge ball. In two dimensions the $L_1$ ball is diamond-shaped, with corners aligned with the coordinate axes. When the contours of the loss first touch the constraint boundary at a corner, one or more coefficients become exactly zero. This is the geometric source of sparsity.

Sparsity makes lasso attractive when the analyst wants a compact predictive model, but sparsity should not be confused with scientific truth. A coefficient driven to zero is zero because of the interaction between the data, the penalty, the scale of the variables, and the chosen value of $\lambda$. It does not prove that the corresponding variable is irrelevant in a causal or substantive sense.

Correlated predictors make this limitation especially important. Suppose two variables carry nearly the same information about the response. Lasso may retain one and exclude the other, even though either variable could provide almost identical predictive performance. Small perturbations in the sample can reverse the choice. Variable selection can therefore be unstable even when prediction is stable. If scientific interpretation matters, the stability of the selected set should be examined through resampling, stability selection, grouped penalties, or domain constraints rather than inferred from one fitted lasso model.

Elastic net combines the two geometries,

$$
\widehat\beta_{\text{EN}}
=
\arg\min_\beta
\left\{
\|y-X\beta\|_2^2
+
\lambda_1\|\beta\|_1
+
\lambda_2\|\beta\|_2^2
\right\}.
$$

The $L_1$ term encourages sparsity while the $L_2$ term stabilizes groups of correlated predictors. This makes elastic net useful when the representation contains blocks of related variables, such as groups of biomarkers, text features, lagged predictors, or engineered variables derived from the same source.

## Regularization is an intentional bias-variance trade-off

The classical motivation for shrinkage is the bias-variance decomposition. Consider prediction under squared error for a target point $x$. If $\widehat f(x)$ is the estimator obtained from a random training sample, then expected prediction error can be decomposed schematically into

$$
E\left[
\left(
Y-\widehat f(x)
\right)^2
\right]
=
\sigma^2
+
\operatorname{Bias}\left[
\widehat f(x)
\right]^2
+
\operatorname{Var}\left[
\widehat f(x)
\right].
$$

Regularization intentionally increases bias in exchange for lower variance. In finite samples, this trade can reduce total prediction error substantially. An unregularized estimator may be unbiased under its model assumptions while remaining so unstable that its mean squared error is poor. The fact that regularization produces biased coefficient estimates is therefore not an accidental cost; it is the mechanism through which variance is controlled.

This point also explains why regularization strength depends on the prediction problem. In a low-noise setting with abundant data and a well-conditioned design, little shrinkage may be needed. In a high-dimensional setting with many weakly identified parameters, stronger regularization can improve stability considerably. There is no universally correct $\lambda$ because the optimal trade-off depends on signal strength, noise, dimension, collinearity, sample size, and the evaluation criterion.

Modern overparameterized models complicate the simplest U-shaped complexity story. In some settings, test error can decrease again after interpolation, producing the phenomenon known as double descent. This does not make regularization irrelevant. It shows that effective model complexity depends not only on parameter count but also on optimization dynamics, architecture, data geometry, initialization, and implicit biases in the learning algorithm. Explicit penalties are one form of complexity control among several.

## Scale is part of the penalty

Penalties act on coefficient magnitude, which means predictor scale directly affects the regularization problem. Suppose one predictor is measured in euros and another in thousands of euros. Their coefficients differ by a factor of one thousand even if they represent the same substantive relationship. Applying the same lasso or ridge penalty to those raw coefficients therefore imposes different effective shrinkage.

For this reason, linear models with coefficient penalties are usually fitted after standardizing continuous predictors,

$$
Z_j
=
\frac{
X_j-\mu_j
}{
\sigma_j
}.
$$

The parameters $\mu_j$ and $\sigma_j$ must be estimated from the training data only. If standardization is performed on the full dataset before cross-validation, information from the validation folds enters the training procedure and produces optimistic estimates.

The intercept is typically left unpenalized because shrinking the global location of the response is conceptually different from shrinking covariate effects. Categorical encodings and grouped variables may also require structured penalties so that the regularization respects how the representation was constructed. A one-hot encoded variable with many levels, for example, can be unfairly penalized relative to a single continuous predictor if each dummy coefficient is treated independently.

Regularization is therefore inseparable from preprocessing. The penalty acts on the chosen coordinate system, and a change of representation changes what “small” means.

## Bayesian connections and their limits

Penalized estimation has a useful Bayesian interpretation. Under a Gaussian likelihood for linear regression, ridge corresponds to a Gaussian prior on coefficients,

$$
\beta_j
\sim
N(0,\tau^2),
$$

with the relationship between $\tau^2$ and $\lambda$ determined by the likelihood variance. The posterior mode coincides with the ridge solution. Likewise, an independent Laplace prior,

$$
p(\beta_j)
\propto
\exp(-\lambda|\beta_j|),
$$

leads to a lasso-type posterior mode.

This connection is conceptually valuable because it makes the regularization assumption explicit: shrinkage toward zero corresponds to prior concentration around zero. It also clarifies why different penalties imply different beliefs about plausible parameter structures.

The equivalence, however, is only between particular penalized estimators and posterior modes under particular priors. A full Bayesian analysis produces a posterior distribution rather than only a single optimizer. Posterior uncertainty can be asymmetric, multimodal, or strongly correlated even when the mode resembles a penalized estimate. Treating lasso coefficients as though they were posterior means or credible intervals would therefore be incorrect.

The Bayesian viewpoint also highlights that zero is not always the natural shrinkage target. Hierarchical priors can shrink coefficients toward group means; structured priors can encourage smoothness, monotonicity, spatial dependence, or sparsity within predefined groups. Regularization is fundamentally about encoding structure, and zero-centered independent penalties are only one family of choices.

## Tuning regularization without leaking the test set

The regularization parameter is usually selected empirically, often by cross-validation. Let $\Lambda$ be a grid of candidate values. For each $\lambda\in\Lambda$, the model is trained on a subset of the data and evaluated on held-out observations. The chosen value minimizes an estimate of predictive loss,

$$
\widehat\lambda
=
\arg\min_{\lambda\in\Lambda}
\widehat R_{\text{CV}}(\lambda).
$$

The final test set must remain untouched during this process. If the analyst repeatedly examines test performance while adjusting $\lambda$, the test set becomes part of model development and no longer provides an unbiased assessment of generalization.

The same principle applies to every hyperparameter that affects regularization, including elastic-net mixing parameters, early-stopping patience, dropout rates, data-augmentation strength, weight decay, pruning thresholds, or architectural constraints. These choices belong inside model selection.

When feature selection is induced by lasso, tuning and selection are inseparable. The set of nonzero variables changes with $\lambda$, so any inference performed after choosing $\lambda$ has been conditioned on a data-dependent selection event. Standard errors from the final ordinary regression model cannot simply be interpreted as though the selected variables had been fixed in advance. Post-selection inference requires additional theory or independent data.

Nested cross-validation can be appropriate when both hyperparameter selection and unbiased performance estimation are required from the same dataset. The inner loop selects $\lambda$; the outer loop estimates the performance of the entire tuning procedure.

## Early stopping, weight decay, and implicit regularization

Regularization is broader than explicit penalties. In iterative optimization, stopping before convergence can constrain the effective solution. Gradient descent on least squares, for example, can initially fit directions associated with large singular values before slower directions are learned. Early stopping can therefore behave like a spectral filter, suppressing unstable components in a manner related to ridge regularization.

In neural networks, weight decay is often implemented by modifying the optimization update so that parameters are continuously pulled toward zero. In simple settings this corresponds closely to an $L_2$ penalty, although the equivalence depends on the optimizer. With adaptive methods, decoupled weight decay can behave differently from adding an $L_2$ term directly to the loss.

Dropout introduces a different form of regularization by randomly masking units during training. The resulting stochastic objective discourages fragile co-adaptations and can be interpreted through several approximate ensemble or Bayesian perspectives, depending on the formulation. It is not a universal remedy for overfitting, and its effect depends strongly on architecture, dataset size, normalization, and optimization.

Data augmentation also acts as regularization when the transformations encode invariances that the prediction problem should respect. Rotating an image is legitimate only if the label is invariant to rotation; perturbing a time series is legitimate only if the perturbation preserves the phenomenon being predicted. Incorrect augmentations impose false invariances and introduce systematic bias. As with explicit penalties, the regularizer encodes assumptions about which variation should be ignored.

## Prediction, inference, and causality are different objectives

Regularization is often optimized for predictive performance. That objective does not automatically align with classical parameter inference. Ridge and lasso deliberately bias coefficients, and the selected model can change discontinuously with the sample. If the goal is to estimate a scientifically meaningful coefficient, the analyst must consider whether shrinkage alters the estimand or whether debiasing and post-selection procedures are needed.

The distinction becomes even sharper in causal inference. Suppose the objective is the causal effect of treatment $A$ on outcome $Y$, and variable $C$ is a confounder. If regularization shrinks the coefficient of $C$ toward zero because $C$ is only weakly predictive of $Y$, the resulting treatment coefficient can become biased even though predictive loss improves. Confounders are selected because of the causal graph, not because they maximize cross-validated accuracy.

Conversely, a variable that is highly predictive may be a mediator or collider and should not necessarily be conditioned on in a causal analysis. Penalized prediction methods do not solve identification. They can be used within causal estimators, but only after the adjustment set and estimand have been justified.

This is why a statement such as “lasso selected the important variables” is incomplete. Important for what? Prediction, compression, scientific explanation, causal identification, or decision-making are different goals and can imply different variable sets.

## Regularization as structural control

The broadest view of regularization is that it restricts the set of solutions that the learning algorithm treats as equally plausible. Ridge prefers small Euclidean norm; lasso prefers sparsity; elastic net mixes these preferences; smoothing splines penalize roughness; total-variation penalties prefer piecewise-constant structure; graph penalties encourage neighboring parameters to be similar; monotonic constraints encode order; early stopping restricts optimization time; augmentation encodes invariance.

All of these methods trade unconstrained fit for structure. Their success depends on whether the structure being imposed is approximately correct for the problem. A penalty can improve generalization when it suppresses noise or unstable directions, but it can also create systematic error when it suppresses real complexity.

The central modeling task is therefore to align the regularizer with the scientific and computational structure of the problem. Regularization should not be selected because one technique is fashionable or because “smaller models generalize better” as a universal rule. The relevant questions are which directions are unstable, which patterns should be sparse, which transformations should be invariant, which groups of parameters should move together, and how strongly the available data support deviations from the preferred structure.

## Conclusion

Regularization is a form of statistical control imposed on an estimation problem. It works by making some solutions less attractive than others, thereby reducing sensitivity to noise, weak identification, collinearity, or excessive flexibility. Ridge, lasso, elastic net, weight decay, early stopping, dropout, augmentation, and structured penalties differ in implementation, but they share the same conceptual role: they encode preferences over functions or parameters.

The practical value of a regularizer cannot be judged from the penalty formula alone. Predictor scaling determines what coefficient magnitude means; hyperparameter tuning determines the strength of the imposed structure; correlated variables determine whether selection is stable; validation design determines whether the apparent gain generalizes; and the inferential objective determines whether biased shrinkage is acceptable.

For prediction, regularization can be one of the most effective ways to reduce variance and improve out-of-sample performance. For scientific inference, it requires more care because shrinkage changes parameter estimates and selection complicates uncertainty. For causal analysis, it cannot substitute for identification. These distinctions matter because the same numerical procedure can be entirely appropriate in one setting and misleading in another.

The correct question is therefore not whether a model is regularized, but what structure the regularization imposes, why that structure is defensible, and how the resulting bias-variance trade-off is validated against the actual objective of the analysis.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
