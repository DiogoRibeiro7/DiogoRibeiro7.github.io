---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-11-10'
excerpt: Understand how simple linear regression models the relationship between two
  variables using a single predictor.
header:
  image: /assets/images/headers/photo-microscope.jpg
  og_image: /assets/images/headers/photo-microscope.jpg
  overlay_image: /assets/images/headers/photo-microscope.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-microscope.jpg
  twitter_image: /assets/images/headers/photo-microscope.jpg
keywords:
- Linear regression
- Least squares
- Data analysis
permalink: '/data-science/simple_linear_regression_intro/'
redirect_from:
- '/data science/simple_linear_regression_intro/'
seo_description: Discover the mechanics of simple linear regression and how to interpret
  slope and intercept when fitting a straight line to data.
seo_title: A Primer on Simple Linear Regression
seo_type: article
summary: This article introduces simple linear regression and the least squares method,
  showing how a single predictor explains variation in a response variable.
tags:
- Regression
- Statistics
- Data Science
title: A Primer on Simple Linear Regression
---

Simple linear regression is a foundational technique for modeling the relationship between a predictor variable and a response variable. By fitting a straight line, we can quantify how changes in one variable are associated with changes in another.

## The Model

The model states that the response $y_i$ is a linear function of the predictor $x_i$ plus an error term:

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad i = 1, \dots, n .
$$

Here $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\varepsilon_i$ captures everything the line does not explain. The quantities $\beta_0$ and $\beta_1$ are unknown population parameters; what we compute from data are estimates $\hat{\beta}_0$ and $\hat{\beta}_1$.

It is worth being precise about the conditioning convention. Classical derivations often condition on the observed predictor values and treat $x_i$ as fixed, but regression can also be formulated with random predictors. What matters for the standard slope interpretation is the conditional mean model $E(Y\mid X=x)=\beta_0+\beta_1x$. Regressing $y$ on $x$ and regressing $x$ on $y$ still give different lines because least squares minimizes vertical residuals in the chosen response variable.

## The Least Squares Method

The most common approach to estimating the regression line is **ordinary least squares (OLS)**. OLS finds the line that minimizes the sum of squared residuals between the observed data points and the line's predictions:

$$
S(\beta_0, \beta_1) = \sum_{i=1}^{n} \left(y_i - \beta_0 - \beta_1 x_i\right)^2 .
$$

Differentiating with respect to each parameter and setting both derivatives to zero yields the normal equations, whose solution has a clean closed form:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
= \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(x)}, \qquad
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} .
$$

The slope indicates the strength and direction of the relationship, while the intercept shows the expected value when the predictor is zero. Two structural facts fall directly out of these formulas. The fitted line always passes through the point of means $(\bar{x}, \bar{y})$, and the residuals always sum to zero when an intercept is included. Neither is evidence that the model fits well; both hold by construction.

The slope also connects directly to correlation:

$$
\hat{\beta}_1 = r \cdot \frac{s_y}{s_x},
$$

where $r$ is the Pearson correlation and $s_x, s_y$ are the sample standard deviations. Correlation is the slope you would get after standardising both variables, which is why it is unitless while the slope carries the units of $y$ per unit of $x$.

## Why Squared Errors

Squaring is a modelling choice, not a law. It is chosen because it makes the problem differentiable and yields the closed form above, and because under the assumption of normally distributed errors, least squares coincides with maximum likelihood estimation.

The cost is sensitivity to outliers. A residual twice as large contributes four times as much to the objective, so a single anomalous point can dominate the fit. If that behaviour is unwanted, minimising absolute deviations gives the more robust least-absolute-deviations estimator, at the price of losing the closed form.

## Assumptions and What Breaks

Four conditions underpin the standard inferential results:

- **Linearity.** The relationship between $x$ and $E[y]$ is genuinely a straight line. If it curves, the fitted slope is a weighted average of local slopes and may describe no part of the data well.
- **Dependence structure.** Standard textbook standard errors assume independent errors. Time-ordered or clustered data can violate this; depending on the correlation structure, naive standard errors may be too small or too large. The coefficient estimates can still be useful under suitable exogeneity conditions, but inference must account for the dependence.
- **Constant variance.** The spread of residuals does not depend on $x$. When it does, the coefficient estimates stay unbiased but their standard errors are wrong.
- **Normal errors.** Needed for exact $t$ and $F$ inference in small samples. In large samples the Central Limit Theorem makes this the least critical of the four.

These failures do not separate cleanly into “coefficient problems” and “standard-error problems.” Nonlinearity changes the estimand represented by the fitted slope; dependence can affect consistency when it is tied to omitted dynamics or endogeneity; heteroscedasticity primarily breaks the usual variance formula when the conditional mean is otherwise correct; and non-normality mainly matters for exact small-sample inference. Each assumption should be tied to the conclusion it supports.

Residual plots are the fastest diagnostic. Plotting residuals against fitted values should produce a structureless band; a curve indicates a linearity problem, and a funnel shape indicates non-constant variance.

## Fitting a Line in Practice

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=100)
y = 2.5 + 1.8 * x + rng.normal(scale=2.0, size=100)

X = sm.add_constant(x)          # adds the intercept column
model = sm.OLS(y, X).fit()

print(model.params)             # [intercept, slope]
print(model.conf_int())         # 95% confidence intervals
print(f"R-squared: {model.rsquared:.3f}")
```

Computing the estimates by hand confirms the formulas:

```python
b1 = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
b0 = y.mean() - b1 * x.mean()
print(b0, b1)                   # matches model.params
```

## Interpreting the Output

The slope says that a one-unit increase in $x$ is *associated with* a $\hat{\beta}_1$-unit change in the average of $y$. It does not say that intervening on $x$ would cause that change. Regression estimates a conditional mean; causal claims require assumptions about confounding that the arithmetic cannot supply.

The coefficient of determination

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

reports the share of variance in $y$ the model accounts for. It is easy to over-read. A high $R^2$ is compatible with a badly misspecified model, a low $R^2$ is normal and acceptable in fields with intrinsically noisy outcomes, and in ordinary least squares with an intercept $R^2$ never decreases when predictors are added. That makes raw $R^2$ a poor standalone criterion for comparing nested models because additional predictors are never penalized.

The intercept deserves particular care. It is the expected response when $x = 0$, which is meaningless if zero lies far outside the observed range. Centring the predictor at its mean makes the intercept the expected response at average $x$, which is usually the more interpretable quantity.

## Where to Go Next

Understanding simple linear regression is a stepping stone toward more complex modeling techniques, providing crucial intuition about correlation and causation. Multiple regression extends the same least-squares machinery to several predictors at once, where the coefficients become partial effects holding the other variables fixed. Generalised linear models keep the linear predictor but replace the normal error assumption, covering logistic regression for binary outcomes and Poisson regression for counts.

The core lesson transfers unchanged: the estimates are a projection of the data onto a chosen model, and their trustworthiness depends entirely on whether that model is a reasonable description of how the data arose.

## References

- Fox, J. (2015). *Applied Regression Analysis and Generalized Linear Models* (3rd ed.). SAGE.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- Weisberg, S. (2013). *Applied Linear Regression* (4th ed.). Wiley.
- Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician*, 27(1), 17-21.
