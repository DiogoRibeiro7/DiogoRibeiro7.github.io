---
permalink: '/statistics/measurement_errors/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-07-26'
excerpt: Measurement error is not just noise around a true value. Its consequences depend on whether error is random or systematic, which variable is measured with error, and what inferential target is being estimated.
header:
  image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  og_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
seo_description: Measurement error explained through statistical models, calibration, uncertainty, attenuation bias, repeated measurements, and the distinction between uncertainty and confidence intervals.
seo_title: 'Measurement Error: Bias, Precision, and Uncertainty'
seo_type: article
summary: A rigorous introduction to random and systematic measurement error, classical errors-in-variables models, uncertainty propagation, calibration, and inferential consequences.
tags:
- Statistical Modeling
- Data Quality
- Measurement Error
- Uncertainty
title: 'Measurement Error: Bias, Precision, and Uncertainty'
---

Measurement error is often introduced as a simple decomposition,

$$
X^{obs}=X^{true}+\text{error}.
$$

That notation is useful, but it hides most of the statistical difficulty.

Errors can be random or systematic. They can occur in the response, in the predictors, or in both. They may depend on the true value being measured, on the instrument, on the observer, on environmental conditions, or on previous measurements.

Those distinctions matter because measurement error changes both uncertainty and bias. The useful question is not merely

> How large is the measurement error?

but

$$
\boxed{
\text{How does the measurement process alter the quantity I am trying to estimate?}
}
$$

## A simple measurement model

Let $X^\ast$ denote the latent quantity of interest and let $X$ denote the recorded measurement. A basic additive model is

$$
X = X^\ast + U.
$$

If

$$
E(U\mid X^\ast)=0,
$$

the measurement is conditionally unbiased under this model. If instead

$$
E(U\mid X^\ast)=b(X^\ast),
$$

then the measurement process contains systematic bias. The terms "random" and "systematic" should therefore be understood relative to a model and a measurement procedure. A calibration offset that is fixed within one instrument may vary randomly across a fleet of instruments.

## Accuracy and precision are different properties

A measurement system can be precise but biased. Suppose repeated measurements satisfy

$$
X_j = X^\ast + b + U_j,
\qquad
E(U_j)=0,
\qquad
\operatorname{Var}(U_j)=\sigma_U^2.
$$

The systematic component $b$ shifts the center of the measurements away from the target. The random component $\sigma_U^2$ determines repeatability. Averaging repeated measurements,

$$
\bar X_m
=
\frac{1}{m}\sum_{j=1}^{m}X_j,
$$

reduces the random variance to

$$
\operatorname{Var}(\bar X_m)
=
\frac{\sigma_U^2}{m},
$$

if the random errors are independent. But the bias remains:

$$
E(\bar X_m-X^\ast)=b.
$$

Repeated measurements improve precision. They do not average away a systematic calibration error.

## Error in the response and error in a predictor behave differently

Consider the linear model

$$
Y^\ast=\beta_0+\beta_1X^\ast+\varepsilon.
$$

If the response is measured with additive mean-zero error,

$$
Y=Y^\ast+V,
\qquad
E(V\mid X^\ast)=0,
$$

ordinary least squares can remain unbiased for the slope under the usual exogeneity assumptions, although the residual variance increases. Now suppose the predictor is measured with classical error,

$$
X=X^\ast+U,
$$

with $U$ independent of $X^\ast$ and $\varepsilon$. Regressing $Y$ on the noisy $X$ produces attenuation. In the simple one-predictor case,

$$
\operatorname{plim}\hat\beta_1
=
\beta_1
\frac{\operatorname{Var}(X^\ast)}
{\operatorname{Var}(X^\ast)+\operatorname{Var}(U)}.
$$

The multiplicative factor

$$
\lambda
=
\frac{\operatorname{Var}(X^\ast)}
{\operatorname{Var}(X^\ast)+\operatorname{Var}(U)}
$$

lies between 0 and 1. So classical predictor error pulls the estimated slope toward zero. This is a statistical bias caused by measurement error, not merely a loss of precision.

## Systematic error is not always a constant offset

A scale that always adds 0.2 kg is a simple example. Real systems are often more complicated. An instrument may have:

- a zero-point offset,
- a multiplicative calibration error,
- nonlinear response,
- drift over time,
- saturation near physical limits,
- temperature dependence,
- observer effects,
- batch effects,
- rounding or digitization error.

A more realistic calibration model might be

$$
X = a + bX^\ast + U.
$$

If $a\neq 0$ or $b\neq 1$, the instrument is systematically distorted even if $U$ has mean zero. This is why calibration data are valuable: they provide observations where the reference value is independently known with substantially smaller uncertainty.

## Measurement uncertainty is not automatically a confidence interval

Statements such as

$$
32.3\pm0.5\text{ cm}
$$

are incomplete unless the meaning of $0.5$ is specified. It might denote:

- one standard uncertainty,
- an expanded uncertainty,
- a standard deviation of repeated measurements,
- a standard error of an estimated mean,
- a tolerance,
- or a confidence interval half-width.

Those objects are not interchangeable. The Guide to the Expression of Uncertainty in Measurement distinguishes **standard uncertainty** from **expanded uncertainty**. If $u_c$ is a combined standard uncertainty, an expanded uncertainty may be written

$$
U=ku_c,
$$

where $k$ is a coverage factor chosen for a stated coverage objective. That is a measurement-uncertainty statement. A statistical confidence interval has a different repeated-sampling interpretation and requires a specified statistical procedure. So the sentence

> the true value is likely between 31.8 and 32.8

cannot be justified from the notation $32.3\pm0.5$ alone.

## Combining uncertainty components

Suppose a reported quantity is

$$
Y=f(X_1,\ldots,X_p).
$$

For small uncertainties and a sufficiently smooth function, first-order propagation gives

$$
u_Y^2
\approx
\nabla f^\top
\Sigma
\nabla f,
$$

where $\Sigma$ is the covariance matrix of the input uncertainties. Written componentwise,

$$
u_Y^2
\approx
\sum_i
\left(
\frac{\partial f}{\partial x_i}
\right)^2u_i^2
+
2\sum_{i<j}
\frac{\partial f}{\partial x_i}
\frac{\partial f}{\partial x_j}
\operatorname{Cov}(X_i,X_j).
$$

The covariance terms matter. Treating all uncertainty sources as independent can understate or overstate total uncertainty. When the transformation is strongly nonlinear or the uncertainties are large, Monte Carlo propagation can be more appropriate than a first-order approximation.

## Repeated measurements estimate only some uncertainty sources

Suppose a device is used repeatedly under the same conditions. The empirical standard deviation estimates repeatability under those conditions. It does not automatically capture:

- calibration uncertainty,
- drift over months,
- between-device variability,
- operator changes,
- reference-standard uncertainty,
- environmental variation outside the experiment.

A precise laboratory repeatability estimate can therefore coexist with substantial real-world measurement uncertainty. The measurement protocol determines what uncertainty is being learned.

## Calibration and validation are different jobs

Calibration uses reference information to estimate the relationship between instrument readings and the target quantity. Validation asks whether the calibrated system performs adequately on new reference measurements or under new conditions. If the calibration equation is estimated on the same data used to judge its performance, the assessment can be optimistic.

The same train/test logic familiar from predictive modeling appears here too.

## Measurement error in data science

Measurement quality matters before any machine-learning algorithm sees the table. If predictors are measured inconsistently across sites, a model may learn site-specific instrumentation rather than the intended phenomenon. If the target label is noisy, apparent model error contains both prediction error and outcome-measurement error. Cross-validation cannot repair a biased measurement process when the same bias exists in every fold.

Data cleaning cannot reconstruct a latent true value unless there is information supporting that reconstruction. The measurement model is part of the statistical model.

## What to report

A defensible measurement analysis should state:

1. what quantity is intended to be measured;
2. how the instrument maps that quantity into a recorded value;
3. which uncertainty components are treated as random;
4. which systematic effects were calibrated or corrected;
5. how reference standards were obtained;
6. what repeated-measurement conditions were used;
7. whether uncertainty components are correlated;
8. whether the reported interval is a confidence interval, credible interval, standard uncertainty, or expanded uncertainty;
9. how measurement error propagates into downstream estimates.

Without those details, a numerical error bar can look much more informative than it is.

## Conclusion

Measurement error is not simply a nuisance term that makes observations fuzzy.

Random error reduces precision. Systematic error can create bias. Repeated measurements can average down some random components while leaving systematic effects intact. Error in predictors can bias regression coefficients even when the measurement error itself has mean zero.

The central lesson is

$$
\boxed{
\text{measurement process}
\rightarrow
\text{error model}
\rightarrow
\text{inferential consequence}
}
$$

A useful uncertainty statement must specify what uncertainty means and which sources of error it includes.

## References

- JCGM. (2008). *Evaluation of measurement data — Guide to the expression of uncertainty in measurement* (JCGM 100:2008).
- JCGM. (2012). *International vocabulary of metrology — Basic and general concepts and associated terms* (VIM), 3rd ed., JCGM 200:2012.
- Fuller, W. A. (1987). *Measurement Error Models*. Wiley.
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006). *Measurement Error in Nonlinear Models* (2nd ed.). Chapman & Hall/CRC.
