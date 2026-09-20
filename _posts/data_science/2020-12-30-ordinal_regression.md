---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-12-30'
excerpt: Ordinal regression models cumulative probabilities without pretending ordered categories have equal spacing. This article derives the proportional-odds model and shows a reproducible Python implementation.
header:
  image: /assets/images/headers/photo-satellite-dish.jpg
  og_image: /assets/images/headers/photo-satellite-dish.jpg
  overlay_image: /assets/images/headers/photo-satellite-dish.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-satellite-dish.jpg
  twitter_image: /assets/images/headers/photo-satellite-dish.jpg
keywords:
- Python
- Statistical models
- Ordinal regression
- Proportional odds
- Marginal effects
permalink: '/data-science/ordinal_regression/'
redirect_from:
- '/data science/ordinal_regression/'
seo_description: A rigorous introduction to proportional-odds ordinal regression, threshold parameters, predicted category probabilities, diagnostics, and marginal effects in Python.
seo_title: 'Ordinal Regression: Proportional Odds and Marginal Effects'
seo_type: article
summary: A mathematical and computational guide to ordinal regression using the proportional-odds cumulative logit model, with a fully reproducible statsmodels example.
tags:
- Statistical Modeling
- Data Analysis
- Regression
- Python
title: 'Ordinal Regression: Proportional Odds and Marginal Effects'
---

Ordinal outcomes contain more information than nominal categories and less information than continuous measurements.

A response such as none, not much, some, and a good deal has an ordering, but the distance from one category to the next need not be equal. Coding those categories as the integers 1, 2, 3, 4 and fitting ordinary least squares silently imposes equal spacing.

A proportional-odds model avoids that assumption.

## Cumulative probabilities

Let

$$
Y\in\{1,\ldots,J\}
$$

be an ordered response. For each threshold $j=1,\ldots,J-1$, define

$$
P(Y\le j\mid x).
$$

The cumulative-logit model is

$$
\operatorname{logit}P(Y\le j\mid x)
=
\alpha_j-x^\top\beta.
$$

Each threshold has its own intercept $\alpha_j$, but the coefficient vector $\beta$ is shared across all thresholds. That shared slope is the proportional-odds assumption.

Equivalently,

$$
\operatorname{logit}P(Y>j\mid x)
=
x^\top\beta-\alpha_j.
$$

A one-unit increase in predictor $x_k$ therefore multiplies the odds of being above any threshold by

$$
\exp(\beta_k),
$$

provided the model is correctly specified.

## Thresholds are not ordinary intercepts

With $J$ categories there are $J-1$ ordered thresholds,

$$
\alpha_1<\alpha_2<\cdots<\alpha_{J-1}.
$$

They partition an underlying latent scale into observed categories.

Because the thresholds already act as intercepts, the design matrix must not contain an additional constant when using **statsmodels.miscmodels.ordinal_model.OrderedModel**.

## Category probabilities

For four categories,

$$
P(Y=1\mid x)=P(Y\le1\mid x),
$$

$$
P(Y=2\mid x)=P(Y\le2\mid x)-P(Y\le1\mid x),
$$

$$
P(Y=3\mid x)=P(Y\le3\mid x)-P(Y\le2\mid x),
$$

and

$$
P(Y=4\mid x)=1-P(Y\le3\mid x).
$$

This is usually the most useful scale for interpretation.

## Reproducible Python example

The previous version depended on an external survey file and used prediction and marginal-effect calls that do not match the fitted model interface. The example below is self-contained.

~~~python
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

rng = np.random.default_rng(2026)
n: int = 1_500

age_z: np.ndarray = rng.normal(size=n)
tertiary: np.ndarray = rng.binomial(1, 0.45, size=n)

latent: np.ndarray = (
    0.55 * age_z
    + 0.80 * tertiary
    + rng.logistic(size=n)
)

cut_points: np.ndarray = np.array([-1.0, 0.3, 1.4])
codes: np.ndarray = np.digitize(latent, cut_points)

labels: list[str] = ["None", "Not much", "Some", "A good deal"]

interest = pd.Series(
    pd.Categorical.from_codes(
        codes,
        categories=labels,
        ordered=True,
    ),
    name="interest",
)

x = pd.DataFrame(
    {
        "age_z": age_z,
        "tertiary": tertiary,
    }
)

model = OrderedModel(
    interest,
    x,
    distr="logit",
)

result = model.fit(method="bfgs", disp=False)
print(result.summary())
~~~

The simulation coefficient for standardized age is 0.55 and the coefficient for tertiary education is 0.80. A finite sample should recover values in the same neighborhood, not exactly the generating numbers.

## Predicting category probabilities correctly

Prediction requires a data frame with the same predictor columns used during estimation.

~~~python
age_grid: np.ndarray = np.linspace(-2.0, 2.0, 100)

new_data = pd.DataFrame(
    {
        "age_z": age_grid,
        "tertiary": np.zeros(age_grid.size, dtype=int),
    }
)

predicted: np.ndarray = result.model.predict(
    result.params,
    exog=new_data,
    which="prob",
)

assert predicted.shape == (100, 4)
assert np.allclose(predicted.sum(axis=1), 1.0)
~~~

The previous article passed only an age vector to a model fitted with several predictors. That design mismatch is not a valid prediction call.

## Marginal effects are category-specific

A positive coefficient does not imply that every category probability increases.

If age shifts probability toward higher categories, lower-category probabilities decrease while upper-category probabilities increase.

For category $j$, the relevant derivative is

$$
\frac{\partial P(Y=j\mid x)}
{\partial x_k}.
$$

Current statsmodels ordinal results do not expose the same **get_margeff()** interface used by some other model classes, so a direct finite-difference calculation is transparent and reproducible.

~~~python
h: float = 1e-4

x_plus = x.copy()
x_minus = x.copy()

x_plus["age_z"] += h
x_minus["age_z"] -= h

p_plus: np.ndarray = result.model.predict(
    result.params,
    exog=x_plus,
    which="prob",
)

p_minus: np.ndarray = result.model.predict(
    result.params,
    exog=x_minus,
    which="prob",
)

individual_effects: np.ndarray = (
    p_plus - p_minus
) / (2.0 * h)

average_marginal_effect: np.ndarray = (
    individual_effects.mean(axis=0)
)

for label, effect in zip(
    labels,
    average_marginal_effect,
    strict=True,
):
    print(f"{label:>12}: {effect:+.4f}")
~~~

Since category probabilities sum to one,

$$
\sum_{j=1}^{J}
\frac{\partial P(Y=j\mid x)}
{\partial x_k}
=
0.
$$

The estimated marginal effects should therefore sum to approximately zero.

## Binary predictors need contrasts

For a binary predictor such as tertiary education, a discrete contrast is more natural than a derivative:

$$
P(Y=j\mid tertiary=1,x_{-k})
-
P(Y=j\mid tertiary=0,x_{-k}).
$$

~~~python
x_one = x.copy()
x_zero = x.copy()

x_one["tertiary"] = 1
x_zero["tertiary"] = 0

p_one = result.model.predict(
    result.params,
    exog=x_one,
    which="prob",
)

p_zero = result.model.predict(
    result.params,
    exog=x_zero,
    which="prob",
)

average_contrast = (p_one - p_zero).mean(axis=0)
~~~

This produces a probability-scale effect for each response category.

## Checking proportional odds

The shared-slope restriction

$$
\beta^{(1)}
=
\beta^{(2)}
=
\cdots
=
\beta^{(J-1)}
$$

is substantive.

If a predictor has little effect on crossing the first threshold but a large effect on crossing the final threshold, the common-slope model can hide that pattern.

Useful checks include separate cumulative binary models, plots of observed and fitted cumulative probabilities, and comparison with partial proportional-odds or multinomial alternatives when justified.

## Coding and missing values

The outcome categories must be stored in the intended order. Alphabetical ordering is not a statistical definition.

Survey codes also need inspection before fitting. Values for "don't know", "refused", or "not applicable" must not be treated as legitimate levels of the ordinal response.

A correct likelihood cannot repair incorrectly coded categories.

## Conclusion

Ordinal regression preserves ordering without inventing equal distances between response categories.

The proportional-odds model is compact:

$$
\operatorname{logit}P(Y\le j\mid x)
=
\alpha_j-x^\top\beta.
$$

Its elegance comes from sharing one coefficient vector across thresholds. Its principal limitation comes from the same assumption.

Interpretation should therefore move from raw coefficients to predicted category probabilities and category-specific marginal effects, while checking whether proportional odds is plausible.

## References

- McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal Statistical Society: Series B*, 42(2), 109–142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- statsmodels documentation. **OrderedModel**, consulted September 2026.
