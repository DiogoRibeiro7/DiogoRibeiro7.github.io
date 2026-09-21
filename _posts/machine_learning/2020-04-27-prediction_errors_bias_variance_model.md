---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2020-04-27'
excerpt: Prediction error is a property of a fitted learning procedure under a deployment distribution. Cross-validation and bootstrap estimators target it differently, and their validity depends on how the data are split.
header:
  image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  og_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  overlay_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
  twitter_image: /assets/images/headers/photo-data-science-parallel-coordinates.jpg
keywords:
- prediction error
- cross-validation
- bootstrap
- .632 bootstrap
- generalization error
- Python
permalink: '/machine-learning/prediction_errors_bias_variance_model/'
redirect_from:
- '/machine learning/prediction_errors_bias_variance_model/'
seo_description: Prediction error estimation through cross-validation and bootstrap methods, including what they estimate, why data-splitting structure matters, and a correct implementation of the .632 bootstrap.
seo_title: 'Prediction Error: Cross-Validation, Bootstrap, and the Target Being Estimated'
seo_type: article
summary: A rigorous guide to estimating out-of-sample prediction error, with careful treatment of training-size bias, dependence, nested model selection, out-of-bag evaluation, and the .632 bootstrap.
tags:
- Model Evaluation
- Statistical Learning
- Bootstrap
- Python
title: 'Prediction Error: Cross-Validation, Bootstrap, and the Target Being Estimated'
---

Training error is not prediction error. If a model is fitted on observations

$$
D_n=
\{(X_i,Y_i)\}_{i=1}^{n},
$$

the empirical training loss is

$$
\widehat R_{\mathrm{train}}
=
\frac{1}{n}
\sum_{i=1}^{n}
L
\left(
Y_i,
\hat f_{D_n}(X_i)
\right).
$$

The same observations influenced both the fitted rule and its evaluation, so this quantity is usually optimistic for future data. The object we care about is closer to

$$
R(D_n)
=
E_{(X,Y)\sim P_{\mathrm{deploy}}}
\left[
L
\left(
Y,
\hat f_{D_n}(X)
\right)
\mid D_n
\right],
$$

the risk of the fitted model under the deployment distribution. If we also average over possible training samples, we obtain the expected risk of the learning procedure. Those are related but not identical targets.

## The split must mimic deployment

Cross-validation is often introduced as a generic recipe:

1. split rows;
2. train on some rows;
3. test on the rest.

The crucial assumption is hidden in step 1. The held-out observations must represent the way new observations will arrive. Random K-fold splitting is appropriate only when observations are exchangeable enough for random partitioning to mimic deployment. It is wrong when future data are structurally different from training data. Examples include:

- time series;
- multiple rows from the same patient;
- several windows from the same machine;
- spatially clustered observations;
- repeated measurements;
- grouped customer histories.

In those settings, splitting by row leaks information. The validation unit must be the unit that will be new at deployment.

## K-fold cross-validation

Partition the data into folds

$$
F_1,\ldots,F_K.
$$

Let

$$
\hat f^{-k}
$$

be the model fitted without fold $F_k$. The K-fold estimate is

$$
\widehat R_{CV}
=
\frac{1}{n}
\sum_{k=1}^{K}
\sum_{i\in F_k}
L
\left(
Y_i,
\hat f^{-k}(X_i)
\right).
$$

Every observation is evaluated by a model that did not use that observation for fitting. That removes direct resubstitution bias. It does not make the estimate unbiased for every target.

## Training-size bias

Each K-fold model is trained on approximately

$$
n\frac{K-1}{K}
$$

observations rather than all $n$ observations. If predictive performance improves with training size, K-fold CV can be pessimistic for the final model refitted on the full sample. LOOCV trains on $n-1$ observations, so this training-size discrepancy is small. That is why LOOCV often has low bias for the risk of a full-sample fit. The statement that LOOCV is simply unbiased is too strong.

The target, algorithm, and sampling scheme matter.

## Variance is more complicated than the textbook slogan

A common slogan says:

> LOOCV has low bias and high variance; 5-fold has higher bias and lower variance.

The first part is often directionally useful. The second is not a theorem that holds monotonically in every problem. Cross-validation folds overlap heavily in their training observations, so fold errors are correlated. Algorithmic instability, sample size, loss function, and data dependence all affect the variance. Five- and ten-fold CV are common because they work reasonably well in many applications, not because one universal mathematical optimum exists.

Repeated K-fold CV can estimate how sensitive the reported score is to the arbitrary fold partition.

## Model selection creates another layer

Suppose hyperparameters are selected by choosing the configuration with the smallest cross-validation error. If the same cross-validation results are then reported as the final performance estimate, the result is optimistic because the validation data influenced model selection. The clean separation is

$$
\boxed{
\text{inner loop: choose model}
\qquad
\text{outer loop: estimate performance}.
}
$$

This is nested cross-validation. A final untouched test set can serve the same outer role when enough data are available.

## Preprocessing belongs inside the folds

Leakage is not limited to fitting the final estimator. Operations learned from data must be fitted inside each training fold, including:

- standardization;
- imputation;
- feature selection;
- PCA;
- target encoding;
- resampling;
- learned embeddings.

If PCA is fitted on all observations before CV, information from the held-out folds has already influenced the representation. A pipeline should reproduce the entire training procedure inside each split.

## The bootstrap samples observations differently

A nonparametric bootstrap sample draws $n$ indices with replacement from the original $n$ observations. For a particular observation, the probability of being absent is

$$
\left(
1-\frac{1}{n}
\right)^n
\longrightarrow
e^{-1}
\approx
0.368.
$$

Therefore the probability of appearing at least once is approximately

$$
1-e^{-1}
\approx
0.632.
$$

This is the origin of the .632 weights. It is not the statement that every bootstrap sample contains exactly 63.2% of the observations. The number of distinct observations is random.

## Out-of-bag error

For bootstrap replicate $b$, let

$$
D_n^{\ast b}
$$

be the bootstrap sample. Observation $i$ is out-of-bag if it is not present in that replicate. A proper out-of-bag loss for observation $i$ averages predictions only over bootstrap models that did not train on $i$:

$$
\widehat e_i^{OOB}
=
\frac{
\sum_b
I(i\notin D_n^{\ast b})
L
\left(
Y_i,
\hat f_b(X_i)
\right)
}{
\sum_b
I(i\notin D_n^{\ast b})
}.
$$

Then

$$
\widehat R_{OOB}
=
\frac{1}{n}
\sum_i
\widehat e_i^{OOB}.
$$

Predicting the entire original training set after each bootstrap fit is not an out-of-bag estimate. Some of those observations were used to train that bootstrap model. The previous version of this article made exactly that mistake.

## The .632 estimator

Let

$$
\widehat R_{\mathrm{app}}
$$

be the apparent error from fitting and evaluating on the full original dataset. The .632 estimator is

$$
\widehat R_{.632}
=
0.368
\widehat R_{\mathrm{app}}
+
0.632
\widehat R_{OOB}.
$$

The method compensates for the excessive optimism of apparent error while using the bootstrap's effective training-sample structure. It was developed particularly for settings where ordinary resubstitution and leave-one-out bootstrap estimates have opposite biases. It is not automatically better than cross-validation for every modern prediction problem.

## The .632+ correction

The .632 estimator can still be optimistic for severe overfitting. Efron and Tibshirani introduced .632+, which increases the weight on out-of-bag error according to a relative overfitting measure. One common form is

$$
\widehat R_{.632+}
=
(1-w)
\widehat R_{\mathrm{app}}
+
w
\widehat R_{OOB},
$$

with

$$
w
=
\frac{0.632}
{1-0.368R}.
$$

The quantity $R$ compares observed overfitting with a no-information error scale. Its exact construction depends on the loss and prediction setting. That dependence is important enough that .632+ should not be implemented from a one-line formula copied without defining the no-information benchmark.

## A correct .632 implementation

The previous code in this post contained a serious bug: it trained the model on one bootstrap sample but generated the supposed OOB mask using a second, unrelated resample of indices. The code below uses the same sampled indices to define the OOB observations.

~~~python
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

FloatArray = NDArray[np.float64]

class Regressor(Protocol):
    def fit(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> "Regressor":
        ...

def predict(
        self,
        x: FloatArray,
    ) -> FloatArray:
        ...

def mse(
    observed: FloatArray,
    predicted: FloatArray,
) -> float:
    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted arrays "
            "must have the same shape."
        )

return float(
        np.mean(
            (observed - predicted) ** 2
        )
    )

def bootstrap_632_mse(
    x: FloatArray,
    y: FloatArray,
    estimator: Regressor,
    *,
    n_bootstraps: int = 1_000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    if x.ndim != 2:
        raise ValueError(
            "x must be a two-dimensional matrix."
        )

if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

if x.shape[0] != y.size:
        raise ValueError(
            "x and y must contain the same "
            "number of observations."
        )

n: int = y.size
    rng = np.random.default_rng(seed)

full_model = clone(estimator)
    full_model.fit(x, y)

apparent_error: float = mse(
        y,
        full_model.predict(x),
    )

oob_loss_sum = np.zeros(
        n,
        dtype=float,
    )
    oob_count = np.zeros(
        n,
        dtype=np.int64,
    )

for _ in range(n_bootstraps):
        sampled_index = rng.integers(
            0,
            n,
            size=n,
        )

in_bag = np.zeros(
            n,
            dtype=bool,
        )
        in_bag[sampled_index] = True

oob_index = np.flatnonzero(
            ~in_bag
        )

if oob_index.size == 0:
            continue

model = clone(estimator)
        model.fit(
            x[sampled_index],
            y[sampled_index],
        )

prediction = model.predict(
            x[oob_index]
        )

oob_loss_sum[oob_index] += (
            y[oob_index] - prediction
        ) ** 2

oob_count[oob_index] += 1

if np.any(oob_count == 0):
        raise RuntimeError(
            "Some observations were never OOB. "
            "Increase n_bootstraps."
        )

per_observation_oob = (
        oob_loss_sum / oob_count
    )

oob_error: float = float(
        per_observation_oob.mean()
    )

error_632: float = (
        0.368 * apparent_error
        + 0.632 * oob_error
    )

return (
        apparent_error,
        oob_error,
        error_632,
    )

rng = np.random.default_rng(7)

x = rng.normal(
    size=(300, 4)
)

beta = np.array(
    [1.5, -2.0, 0.5, 0.0],
    dtype=float,
)

y = (
    x @ beta
    + rng.normal(
        scale=1.0,
        size=300,
    )
)

result = bootstrap_632_mse(
    x,
    y,
    LinearRegression(),
)

print(result)
~~~

This estimates each observation's OOB loss only from models that did not train on that observation.

## Cross-validation code should reproduce the deployment split

For ordinary exchangeable regression data:

~~~python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=2026,
)

scores = cross_val_score(
    LinearRegression(),
    x,
    y,
    scoring="neg_mean_squared_error",
    cv=cv,
)

cv_mse: float = float(
    -scores.mean()
)

print(cv_mse)
~~~

For grouped, spatial, or temporal data, replace the ordinary K-fold splitter with the split appropriate to that structure. The estimator is only as honest as the split.

## Confidence in the performance estimate

Reporting

$$
\widehat R_{CV}=2.14
$$

to two decimal places can imply more certainty than the data justify. Fold scores are dependent, so the ordinary sample standard deviation of fold scores is not a simple standard error for generalization risk. Useful approaches include:

- repeated cross-validation to assess split sensitivity;
- bootstrap of the entire learning-and-evaluation procedure where appropriate;
- external test sets;
- model-comparison tests designed for repeated resampling structures.

The uncertainty target should be stated explicitly.

## Conclusion

Prediction-error estimation is not a contest between cross-validation and bootstrap. It is a problem of matching an estimator to a deployment target. The correct sequence is

$$
\boxed{
\text{deployment distribution}
\rightarrow
\text{validation unit}
\rightarrow
\text{training procedure}
\rightarrow
\text{held-out prediction}
\rightarrow
\text{loss}.
}
$$

Cross-validation fails when the split leaks information. Bootstrap fails when in-bag observations are mislabeled as out-of-bag. Neither method can rescue a validation design that does not resemble the future use of the model.

## References

- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B*, 36(2), 111–147.
- Efron, B., & Tibshirani, R. J. (1997). Improvements on cross-validation: The .632+ bootstrap method. *Journal of the American Statistical Association*, 92(438), 548–560.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
