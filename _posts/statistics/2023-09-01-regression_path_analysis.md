---
permalink: '/statistics/regression_path_analysis/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-09-01'
excerpt: Path analysis is a system of linked regression equations. It can decompose associations into direct and indirect paths, but causal interpretation still depends on identification assumptions and study design.
header:
  image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  og_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  overlay_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-law-large-numbers.jpg
  twitter_image: /assets/images/headers/photo-statistics-law-large-numbers.jpg
keywords:
- Regression analysis
- Path analysis
- Structural equation models
- Mediation
- Causal inference
- Direct effects
- Indirect effects
seo_description: A rigorous comparison of regression and path analysis, including mediation, identification, direct and indirect effects, and the limits of causal interpretation.
seo_title: 'Regression and Path Analysis: Models, Mediation, and Causality'
seo_type: article
summary: Path analysis extends regression by fitting a system of linked equations and decomposing effects along prespecified paths. The diagram does not create causal identification: causal claims require assumptions about temporal ordering, confounding, measurement, and model specification.
tags:
- Regression
- Structural Equation Modeling
- Causal Inference
title: Regression and Path Analysis: What the Diagram Does Not Tell You
---

Regression and path analysis are often presented as two separate techniques, with regression described as a simple predictive method and path analysis as a more advanced causal method. That framing is misleading. Path analysis is built from regression equations. Its distinctive feature is that several equations are linked into a system so that a variable can be an outcome in one equation and a predictor in another.

The important distinction is therefore not "prediction versus causality." Both regression and path models can be used descriptively, predictively, or causally depending on the design and assumptions. A path diagram can encode a causal hypothesis, but arrows on a diagram do not make the hypothesis true.

## Regression as a conditional model

A linear regression model can be written as

$$
Y = \beta_0 + X^\top\beta + \varepsilon,
$$

with the conditional mean

$$
E(Y\mid X)=\beta_0+X^\top\beta.
$$

The coefficients describe conditional associations under the specified model. Their causal interpretation requires additional assumptions. For example, if $X_j$ is interpreted as an intervention, then confounding, selection, measurement error, post-treatment adjustment, and interference must all be considered.

Normality of residuals is not required for ordinary least-squares coefficients to exist or for the Gauss-Markov result. It becomes relevant for exact small-sample Gaussian inference. Likewise, multicollinearity does not generally bias OLS coefficients; it can make them unstable and imprecise.

Regression can also contain nonlinear terms, interactions, splines, fixed effects, random effects, and generalized response distributions. The phrase "regression analysis" therefore covers a much broader class than simple straight-line fitting.

## Path analysis as a system of regressions

Consider three variables $X$, $M$, and $Y$, where $M$ is hypothesized to mediate part of the relationship between $X$ and $Y$. A simple path model is

$$
M = aX + \varepsilon_M,
$$

$$
Y = c'X + bM + \varepsilon_Y.
$$

The direct path from $X$ to $Y$ is represented by $c'$. Under the linear model, the product

$$
ab
$$

is the model-based indirect effect through $M$. If a compatible total-effect model is

$$
Y = cX + \varepsilon,
$$

then under the standard linear decomposition,

$$
c = c' + ab.
$$

This algebra is useful, but it does not establish that $ab$ is a causal mediation effect. That interpretation needs assumptions about treatment assignment, mediator-outcome confounding, treatment-induced confounders, temporal ordering, consistency, and model specification.

## What path diagrams encode

A path diagram is a compact representation of the equations and covariance assumptions.

- A single-headed arrow usually represents a directed regression path.
- A double-headed arrow usually represents an allowed covariance.
- Exogenous variables have no directed causes inside the specified model.
- Endogenous variables are explained, at least partly, by other variables in the system.

These labels are properties of the model, not metaphysical properties of the variables. A variable can be exogenous in one model and endogenous in another.

The diagram is valuable because it forces assumptions into the open. If two residuals are allowed to correlate, that says something about omitted common causes or shared measurement structure. If no arrow or residual covariance connects two components, the model asserts a conditional independence structure that can have testable implications.

## Identification comes before estimation

A path model can contain more unknown parameters than the observed covariance structure can identify. Before interpreting fitted coefficients, one should ask whether the model is identified.

For $p$ observed variables, the covariance matrix supplies

$$
\frac{p(p+1)}{2}
$$

distinct second-order moments. A covariance-structure model cannot estimate an arbitrary number of free means, variances, covariances, and paths from those moments.

Identification is not merely a software issue. A model can converge numerically and still encode a causal quantity that is not identified from the study design. Statistical identification and causal identification are related but distinct questions.

## Direct and indirect effects are model dependent

The decomposition of an association into direct and indirect components depends on the variables included and on their causal roles. Conditioning on a mediator changes the estimand. Conditioning on a collider can create bias. Conditioning on a post-treatment common cause of mediator and outcome can destroy a simple mediation interpretation.

Even in linear models, an indirect effect is not a universal property of the data. It is relative to a specified causal graph and intervention. In nonlinear models, products of coefficients may not equal a marginal mediated effect at all.

This is why mediation analysis should begin with the causal question rather than with a path diagram template.

## Path analysis versus structural equation models

Observed-variable path analysis is commonly treated as a special case of structural equation modelling. Full SEM can additionally include latent variables and explicit measurement models. For example, a latent construct $\eta$ may be measured by several indicators,

$$
Y_j = \lambda_j \eta + \epsilon_j.
$$

This separates measurement error from structural relations among latent variables, at least under the assumptions of the measurement model.

Observed-variable path analysis generally treats measured variables as observed without an explicit latent measurement model. If a variable is measured unreliably, its path coefficients can be distorted just as regression coefficients can be distorted by measurement error.

## Model fit is not causal validation

SEM software reports fit statistics such as the chi-square test, RMSEA, CFI, TLI, and SRMR. These can be useful for diagnosing whether the covariance restrictions implied by the model are compatible with the observed covariance structure.

Good fit does not prove the causal graph. Distinct causal models can imply the same covariance structure, and a misspecified model can sometimes fit well because the data do not contain enough information to distinguish alternatives. Conversely, a scientifically useful approximation can be rejected in a very large sample because even tiny deviations become detectable.

Fit statistics should therefore be interpreted as evidence about model-data compatibility, not as proof of mechanism.

## When ordinary regression is enough

A system of path equations is unnecessary when the scientific target is a single conditional mean or a single treatment contrast. A regression model may be sufficient when there is one main outcome and no need to represent a network of mediating or reciprocal relations.

Adding a path diagram does not improve an analysis merely by making it look more structural. Complexity is justified when the joint system itself matters.

## When path analysis is useful

Path analysis becomes useful when several linked equations are substantively motivated. Common examples include longitudinal developmental models, mediation hypotheses, economic systems with intermediate mechanisms, and social-science theories involving multiple endogenous variables.

The strongest applications have three features: temporal and substantive ordering is specified before looking at the final estimates, the relevant confounders are addressed by design or explicit assumptions, and alternative plausible structures are considered rather than treating one diagram as uniquely determined by the data.

## Conclusion

Regression and path analysis belong to the same modelling family. Path analysis links several regressions so that direct, indirect, and total associations can be represented within one system. That extra structure can be scientifically useful, but it also adds assumptions.

The central lesson is

$$
\text{diagram}
\neq
\text{identification}.
$$

A causal path interpretation requires more than a fitted covariance model. It requires a defensible design, temporal logic, appropriate adjustment, measurement assumptions, and a clear estimand. Once those are stated, path analysis can be a concise language for the hypothesized mechanism rather than a substitute for causal reasoning.

## References

- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Kline, R. B. (2016). *Principles and Practice of Structural Equation Modeling* (4th ed.). Guilford Press.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
