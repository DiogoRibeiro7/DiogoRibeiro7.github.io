---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2021-08-01'
excerpt: A step-by-step guide to implementing Linear Regression from scratch using
  the Normal Equation method, complete with Python code and evaluation techniques.
header:
  image: /assets/images/data_science_2.avif
  og_image: /assets/images/data_science_2.avif
  overlay_image: /assets/images/data_science_2.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.avif
  twitter_image: /assets/images/data_science_2.avif
keywords:
- Linear regression
- Normal equation
- Python
- Data science interviews
permalink: '/machine-learning/building_linear_regression_scratch/'
redirect_from:
- '/machine learning/building_linear_regression_scratch/'
seo_description: 'Build a linear regression model from scratch with the Normal Equation: theoretical foundations, algorithm design, and Python implementation.'
seo_title: 'Linear Regression from Scratch: Normal Equation'
seo_type: article
summary: This article provides a detailed algorithmic approach to building a Linear
  Regression model from scratch, covering theory, Python code implementation, and
  performance evaluation.
tags:
- Regression
- Python
title: 'Building Linear Regression from Scratch: A Detailed Algorithmic Approach'
---

One of the most commonly discussed topics in data science interviews—and one of the most practical—is the implementation of machine learning algorithms from scratch. A particularly interesting challenge that surfaces in interviews is building a **Linear Regression** model without relying on external libraries like Scikit-learn or TensorFlow.

Why is this problem so often encountered in interviews? This challenge tests a variety of skills that are critical for data science roles, including:

- Understanding the mathematics behind regression models.
- Knowledge of applied linear algebra and numerical methods.
- Object-Oriented Programming (OOP) practices.
- Designing algorithms from the ground up.
- Competence with numerical computing and performance optimizations.

This article works through the Normal Equation approach, then shows why the textbook version of it is the wrong way to actually compute the answer.

## The Fundamentals of Linear Regression

The model predicts a response as a linear combination of features:

$$
\hat{y} = X\beta, \qquad X \in \mathbb{R}^{n \times p},\ \beta \in \mathbb{R}^{p}.
$$

Fitting means choosing $\beta$ to minimise the residual sum of squares:

$$
S(\beta) = \lVert y - X\beta \rVert^2 .
$$

Setting the gradient to zero gives the **normal equations**:

$$
X^\top X \beta = X^\top y,
$$

whose textbook solution is $\hat{\beta} = (X^\top X)^{-1} X^\top y$.

There is a useful geometric reading. The fitted values $X\hat{\beta}$ are the orthogonal projection of $y$ onto the column space of $X$, and the residual $y - X\hat{\beta}$ is orthogonal to every column of $X$. That orthogonality *is* the normal equations — the name comes from "normal" in the geometric sense.

## Why Not to Invert the Matrix

The formula $(X^\top X)^{-1} X^\top y$ is correct mathematics and poor numerics. Two problems compound.

Forming $X^\top X$ **squares the condition number** of $X$. If $X$ is moderately ill-conditioned — which correlated features guarantee — the product can be numerically singular even though the least-squares problem is perfectly well posed. Precision is lost before any solving begins.

Explicitly inverting is then slower and less accurate than solving the system directly. `np.linalg.solve` uses a factorisation rather than computing an inverse and multiplying.

The robust approach skips $X^\top X$ altogether and factors $X$ directly. QR decomposition writes $X = QR$ with $Q$ orthonormal and $R$ upper triangular, reducing the problem to the triangular system $R\beta = Q^\top y$. SVD is more expensive still and handles rank deficiency gracefully by producing the minimum-norm solution, which is what `numpy.linalg.lstsq` does.

The practical ranking: `lstsq` (SVD) is the safe default, QR is a good balance, `solve` on the normal equations is acceptable when $X$ is well conditioned, and explicit `inv` is essentially never right.

## An Implementation

```python
import numpy as np


class LinearRegression:
    """Least squares by QR, with a normal-equations path for comparison."""

    def __init__(self, fit_intercept: bool = True, method: str = "qr"):
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = 0.0

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        A = self._design(X)
        y = np.asarray(y, dtype=float).ravel()
        if A.shape[0] < A.shape[1]:
            raise ValueError("fewer observations than parameters: underdetermined")

        if self.method == "qr":
            Q, R = np.linalg.qr(A)
            beta = np.linalg.solve(R, Q.T @ y)
        elif self.method == "svd":
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        elif self.method == "normal":
            beta = np.linalg.solve(A.T @ A, A.T @ y)
        else:
            raise ValueError(f"unknown method: {self.method}")

        if self.fit_intercept:
            self.intercept_, self.coef_ = beta[0], beta[1:]
        else:
            self.intercept_, self.coef_ = 0.0, beta
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Coefficient of determination."""
        y = np.asarray(y, dtype=float).ravel()
        resid = ((y - self.predict(X)) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        return 1 - resid / total
```

Separating `_design` from `fit` matters more than it looks: the intercept column must be added identically at fit and predict time, and doing it in two places is how that silently diverges.

## Seeing the Numerical Difference

Conditioning is not a theoretical concern. This constructs a design matrix with two nearly collinear columns:

```python
rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(scale=1e-6, size=n)      # almost a duplicate column
X = np.column_stack([x1, x2, rng.normal(size=n)])
beta_true = np.array([2.0, -1.0, 0.5])
y = X @ beta_true + rng.normal(scale=0.01, size=n)

print("condition number of X   :", f"{np.linalg.cond(X):.2e}")   # ~1.9e+06
print("condition number of XtX :", f"{np.linalg.cond(X.T @ X):.2e}") # ~3.5e+12

for m in ("svd", "qr", "normal"):
    beta = LinearRegression(method=m).fit(X, y).coef_
    print(f"{m:7} coefficients = {np.round(beta, 2)}")
```

The condition number of $X^\top X$ comes out as the square of that of $X$ — 1.9e6 becomes 3.5e12 — which is precisely the precision loss described above.

The coefficients are the striking part. The true values are $[2, -1, 0.5]$, and every method returns something near $[-1134, 1135, 0.499]$. That is not a numerical bug: with two nearly identical columns, the data genuinely cannot distinguish "2 times $x_1$ minus 1 times $x_2$" from "−1134 times $x_1$ plus 1135 times $x_2$", because those are almost the same function. The problem is ill-posed, and no algorithm can recover what the data does not contain.

What *is* recoverable is the identifiable combination. The sum of the first two coefficients comes back as 1.0007 against a true value of 1.0, under all three methods, and predictions are equally accurate in every case because the column space is unchanged.

The methods do differ, but modestly here: SVD and QR agree to full displayed precision while the normal equations diverge in the fourth significant figure. At this conditioning that gap is small; it widens as the condition number approaches the limits of double precision, which is the argument for preferring a factorisation by default rather than only when trouble is already visible.

The practical lesson is about interpretation more than arithmetic. If you need forecasts, collinearity may not matter. If you intend to read the coefficients, it matters completely — and a coefficient of −1134 where you expected 2 is the signal.

## Regularisation as the Fix

Ridge regression adds a penalty that both stabilises the numerics and controls variance:

$$
\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y .
$$

Adding $\lambda$ to the diagonal makes the system invertible even when $X^\top X$ is singular, which is why ridge works when $p > n$ and ordinary least squares has no unique solution at all.

Two implementation details are easy to get wrong. Do not penalise the intercept — shrinking it toward zero makes the fit depend on where the response happens to be centred. And standardise the features first, since the penalty applies equally to every coefficient and unscaled features are therefore penalised inconsistently.

## Checking the Implementation

Testing against a reference is the fastest way to catch mistakes:

```python
from sklearn.linear_model import LinearRegression as SKLinear

ours = LinearRegression().fit(X, y)
theirs = SKLinear().fit(X, y)

assert np.allclose(ours.predict(X), theirs.predict(X), atol=1e-8)
assert np.isclose(ours.score(X, y), theirs.score(X, y))
```

Worthwhile edge cases: a single feature, a perfectly collinear pair, a constant column, more parameters than observations, and `fit_intercept=False`. The constant column is a good test because it makes the design matrix rank-deficient once an intercept is added — SVD returns the minimum-norm solution, QR may not, and the normal equations fail outright. Knowing which behaviour you get is the point of writing it yourself.

## What the Exercise Is Actually For

You should use a library in production. The value in building this is understanding what the library is doing, so that when it warns about conditioning, returns implausible coefficients, or behaves differently from another implementation, the behaviour is legible rather than mysterious.

The transferable lesson is that the clean mathematical expression and the correct computation are not the same object. $(X^\top X)^{-1}X^\top y$ is how you write it; a QR or SVD factorisation is how you compute it. That gap recurs throughout numerical work.

## References

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
