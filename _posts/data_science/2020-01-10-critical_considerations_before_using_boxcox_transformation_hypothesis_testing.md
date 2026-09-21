---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-01-10'
excerpt: Box-Cox is a transformation family inside a statistical model. It should not be used as a ritual for making raw data look normal, and changing the response scale changes the estimand and interpretation.
header:
  image: /assets/images/headers/photo-statistics-transformations.jpg
  og_image: /assets/images/headers/photo-statistics-transformations.jpg
  overlay_image: /assets/images/headers/photo-statistics-transformations.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-transformations.jpg
  twitter_image: /assets/images/headers/photo-statistics-transformations.jpg
keywords:
- Box-Cox transformation
- transformations
- regression
- variance stabilization
- retransformation bias
seo_description: Box-Cox transformations explained as part of a statistical model, including positivity, likelihood-based lambda selection, interpretation, hypothesis changes, and retransformation.
seo_title: 'Box-Cox Transformations: Model First, Transformation Second'
seo_type: article
summary: A rigorous guide to Box-Cox transformations that explains what the transformation is optimizing, why positivity matters, how interpretation changes, and when a GLM or explicit variance model is preferable.
tags:
- Regression
- Statistical Modeling
- Transformations
title: 'Box-Cox Transformations: Model First, Transformation Second'
---

The Box-Cox transformation is often used as a preprocessing ritual:

1. inspect a skewed response;
2. choose a transformation;
3. make the histogram look more normal;
4. run the original analysis.

That sequence misses the point of the method. Box and Cox introduced a **family of transformed regression models** and a likelihood-based way to compare them. The transformation is part of the model. It changes the scale on which the conditional mean is specified and therefore changes the interpretation of coefficients and hypotheses.

## The Box-Cox family

For a positive response $y>0$, a common Box-Cox transformation is

$$
T_\lambda(y)
=
\begin{cases}
\dfrac{y^\lambda-1}{\lambda},
&
\lambda\ne0,
\\[6pt]
\log y,
&
\lambda=0.
\end{cases}
$$

The limit as $\lambda\to0$ is

$$
\log y.
$$

Special cases include approximately:

$$
\lambda=1
\rightarrow
\text{original scale},
$$

$$
\lambda=\frac12
\rightarrow
\text{square-root-like scale},
$$

$$
\lambda=0
\rightarrow
\log y.
$$

The original method includes scaling conventions so likelihoods for different $\lambda$ values are comparable. That detail matters.

## What model is being assumed?

The classical Box-Cox regression assumes that, for some transformation parameter $\lambda$,

$$
T_\lambda(Y_i)
=
x_i^\top\beta
+
\varepsilon_i,
$$

with

$$
\varepsilon_i
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma^2)
$$

under the model. So the target is not

> find a transformation that makes $Y$ normal.

The target is closer to

> find a response scale on which the conditional mean is adequately modeled by the chosen linear predictor and the residual variance is approximately homogeneous and Gaussian enough for the intended likelihood analysis.

The residual structure is the object of interest.

## Raw-response normality is usually irrelevant

Suppose

$$
Y
=
\exp(
\beta_0+\beta_1X+\varepsilon
).
$$

The marginal distribution of $Y$ can be strongly skewed even if

$$
\log Y
=
\beta_0+\beta_1X+\varepsilon
$$

has well-behaved Gaussian errors. Testing the raw outcome for normality before fitting the regression answers the wrong question. Conditional modeling matters.

## Positivity is not optional

The standard Box-Cox family requires

$$
Y>0.
$$

A common workaround is to add a constant:

$$
Y^\ast=Y+c.
$$

That is not innocuous. Different choices of $c$ can produce different fitted transformations, especially when values are close to zero. If zero or negative values are scientifically meaningful, a shifted Box-Cox transformation should be justified explicitly or another model family should be used. A transformation should not erase the meaning of zero.

## Choosing lambda by likelihood

For each candidate $\lambda$, fit the transformed model and compute the corresponding profile likelihood. Then choose

$$
\hat\lambda
=
\arg\max_\lambda
\ell_p(\lambda).
$$

A likelihood interval for $\lambda$ is often more informative than the single optimum. If values near

$$
\lambda=0
$$

and

$$
\lambda=0.1
$$

fit almost equally well, reporting three decimal places for $\hat\lambda$ is false precision. Interpretability can justify choosing a nearby simple value such as 0, 1/2, or 1 when the likelihood supports it.

## The Jacobian matters

A transformation changes the density scale. If

$$
Z=T_\lambda(Y),
$$

then

$$
f_Y(y)
=
f_Z(T_\lambda(y))
\left|
\frac{dT_\lambda(y)}
{dy}
\right|.
$$

Likelihood comparison across transformation parameters must include this change-of-variable term, or use an equivalent scaled formulation. Simply fitting ordinary least squares after many arbitrary transforms and comparing residual sums of squares can give the wrong comparison.

## Transformations change hypotheses

Suppose the original scientific question concerns the arithmetic mean difference

$$
E[Y\mid X=1]
-
E[Y\mid X=0].
$$

After a log transformation, a linear model targets

$$
E[\log Y\mid X]
$$

on the transformed scale. A coefficient difference there is naturally multiplicative after exponentiation. For a simple log-linear model,

$$
\log Y
=
\beta_0+\beta_1X+\varepsilon,
$$

the quantity

$$
\exp(\beta_1)
$$

is associated with a ratio on a geometric-mean or median-like scale under the model, not automatically a ratio of arithmetic means. The scientific estimand has changed.

## Back-transformation is not as simple as exponentiating the fitted mean

If

$$
Z=\log Y,
$$

then

$$
E[Y\mid X]
=
E[\exp(Z)\mid X].
$$

In general,

$$
E[\exp(Z)\mid X]
\ne
\exp(E[Z\mid X]).
$$

Under a homoskedastic Gaussian log model,

$$
Z\mid X
\sim
\mathcal N(\mu_X,\sigma^2),
$$

we have

$$
E[Y\mid X]
=
\exp
\left(
\mu_X+\frac{\sigma^2}{2}
\right).
$$

That extra term is retransformation bias correction. For other transformations or heteroskedastic errors, the correction is more complicated.

## Outliers are not a reason to transform automatically

A monotone transformation can reduce the numerical leverage of large observations. That does not tell us whether those observations are errors, valid extremes, or evidence that the model is wrong. The sequence should be:

1. verify the observation;
2. understand the mechanism;
3. assess influence;
4. choose a model appropriate to the data-generating process.

Transformation is not data cleaning.

## Variance stabilization

Box-Cox can be useful when variance scales systematically with the mean. Suppose approximately

$$
\operatorname{SD}(Y\mid X)
\propto
E[Y\mid X]^q.
$$

A power transformation can sometimes make residual variability more nearly constant. But if the mean-variance relationship has a known stochastic origin, a generalized linear model may be more natural. For counts, for example, a Poisson or negative-binomial model uses the count distribution directly rather than trying to make counts Gaussian.

## GLMs are not merely “methods for non-normal residuals”

A generalized linear model specifies:

1. a response distribution from the exponential family;
2. a mean $\mu_i$;
3. a link function

$$
g(\mu_i)
=
x_i^\top\beta.
$$

The link transforms the **mean**, not the observed response. That is conceptually different from transforming $Y$ and fitting Gaussian least squares. Choosing between Box-Cox and a GLM is therefore a modeling decision, not two interchangeable normalization tricks.

## Comparing transformed models

Prediction error on the transformed scale is not directly comparable with prediction error on the original scale. If the operational loss is measured in euros, kilograms, or minutes, validation should return predictions to that scale and evaluate the relevant loss there. Likewise, $R^2$ values from different response transformations do not answer the same variance-explained question.

Model comparison must match the scientific and operational scale.

## A reproducible example

~~~python
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)

x: np.ndarray = rng.uniform(
    0.0,
    2.0,
    size=500,
)

noise: np.ndarray = rng.normal(
    loc=0.0,
    scale=0.35,
    size=x.size,
)

y: np.ndarray = np.exp(
    1.0 + 0.8 * x + noise
)

transformed, lambda_hat = stats.boxcox(y)

print(f"lambda = {lambda_hat:.3f}")
print(
    "corr(log(y), x) =",
    np.corrcoef(
        np.log(y),
        x,
    )[0, 1],
)
~~~

Because the data-generating process is log-linear, the estimated Box-Cox parameter should typically be near zero. The example works because we know the generating mechanism. With real data, diagnostics and subject-matter interpretation remain necessary.

## Conclusion

Box-Cox is not a test for whether data are normal. It is a family of transformed statistical models. The sequence should be

$$
\boxed{
\text{scientific estimand}
\rightarrow
\text{conditional model}
\rightarrow
\text{candidate transformation}
\rightarrow
\text{likelihood and diagnostics}
\rightarrow
\text{interpretation on the required scale}.
}
$$

A transformation is justified only if the transformed model answers a question we actually want answered.

## References

- Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211–243. https://doi.org/10.1111/j.2517-6161.1964.tb00553.x
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- Carroll, R. J., & Ruppert, D. (1988). *Transformation and Weighting in Regression*. Chapman & Hall.
