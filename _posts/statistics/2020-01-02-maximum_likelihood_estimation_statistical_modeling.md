---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-02'
excerpt: Maximum likelihood is an estimation principle, not a guarantee of truth. Its properties depend on identifiability, regularity, model specification, and the geometry of the likelihood.
header:
  image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  og_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
keywords:
- Maximum likelihood estimation
- likelihood
- statistical inference
- asymptotic normality
- model misspecification
- Python
seo_description: Maximum likelihood estimation explained through likelihood geometry, identifiability, asymptotic theory, misspecification, and reproducible examples.
seo_title: 'Maximum Likelihood Estimation: What It Guarantees and What It Does Not'
seo_type: article
summary: A rigorous introduction to MLE that derives simple estimators, explains asymptotic properties under regularity conditions, and separates likelihood optimization from model validity.
tags:
- Statistical Modeling
- Probability
- Data Science
- Python
title: 'Maximum Likelihood Estimation: What It Guarantees and What It Does Not'
---

Maximum likelihood estimation is one of the central ideas in statistical modeling.

Its definition is simple.

Given observed data $x$ and a model indexed by parameter $\theta$, choose the parameter value that makes the observed data most likely under that model.

The difficulty is everything hidden inside the phrase **under that model**.

MLE does not tell us whether the model is scientifically appropriate.

It tells us which parameter value fits best within the family we chose.

## Likelihood is a function of the parameter

Suppose

$$
X_1,\ldots,X_n
$$

are modeled as independent observations with density or mass function

$$
p(x\mid\theta).
$$

After observing

$$
x_1,\ldots,x_n,
$$

the likelihood is

$$
L(\theta;x)
=
\prod_{i=1}^{n}
p(x_i\mid\theta).
$$

The observations are fixed inside this function.

The parameter varies.

This is why a likelihood is not a probability distribution over $\theta$.

Without a prior, it does not integrate to one over parameter space and need not be interpreted probabilistically as

$$
P(\theta\mid x).
$$

## Log-likelihood

Products of many small probabilities or densities are numerically inconvenient.

Because the logarithm is monotone,

$$
\arg\max_\theta L(\theta)
=
\arg\max_\theta \ell(\theta),
$$

where

$$
\ell(\theta)
=
\log L(\theta).
$$

For independent observations,

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log p(x_i\mid\theta).
$$

This turns products into sums and usually simplifies differentiation.

## Bernoulli example

Let

$$
X_i
\sim
\operatorname{Bernoulli}(p),
$$

independently.

If there are $k$ successes among $n$ observations,

$$
L(p)
=
p^k(1-p)^{n-k}.
$$

The log-likelihood is

$$
\ell(p)
=
k\log p
+
(n-k)\log(1-p).
$$

Differentiating,

$$
\frac{d\ell}{dp}
=
\frac{k}{p}
-
\frac{n-k}{1-p}.
$$

Setting the score to zero gives

$$
\hat p
=
\frac{k}{n}.
$$

So the sample proportion is the Bernoulli MLE.

The result is familiar, but the derivation shows the estimation principle explicitly.

## Normal example

Suppose

$$
X_i
\overset{\mathrm{iid}}{\sim}
\mathcal N(\mu,\sigma^2).
$$

The log-likelihood is

$$
\ell(\mu,\sigma^2)
=
-\frac{n}{2}\log(2\pi)
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{i=1}^{n}(x_i-\mu)^2.
$$

Maximizing over $\mu$ gives

$$
\hat\mu_{\mathrm{MLE}}
=
\bar x.
$$

Maximizing over $\sigma^2$ gives

$$
\hat\sigma^2_{\mathrm{MLE}}
=
\frac{1}{n}
\sum_{i=1}^{n}
(x_i-\bar x)^2.
$$

This is not the unbiased sample-variance estimator, whose denominator is $n-1$.

MLE and unbiasedness are different criteria.

## The score and information

The score is

$$
U(\theta)
=
\frac{\partial\ell(\theta)}
{\partial\theta}.
$$

Under regularity conditions and the correctly specified model,

$$
E_\theta[U(\theta)]
=
0.
$$

The Fisher information can be written as

$$
I(\theta)
=
E_\theta
\left[
U(\theta)U(\theta)^\top
\right],
$$

or, under additional regularity,

$$
I(\theta)
=
-
E_\theta
\left[
\frac{\partial^2\ell(\theta)}
{\partial\theta\partial\theta^\top}
\right].
$$

Information measures local curvature and parameter sensitivity.

Flat likelihood directions correspond to weak identification and large uncertainty.

## Consistency is not automatic

Textbook summaries often say that MLE is consistent.

The correct statement is conditional.

Consistency requires assumptions such as:

- the data-generating distribution belongs to, or is appropriately represented by, the model;
- the parameter is identifiable;
- the likelihood obeys suitable continuity and compactness or coercivity conditions;
- observations satisfy the dependence assumptions required by the theorem;
- the criterion converges uniformly enough to its population target.

When those conditions fail, MLE can be inconsistent, non-unique, or undefined.

## Identifiability

A model is identifiable if different parameter values imply different observable distributions.

Formally,

$$
P_{\theta_1}=P_{\theta_2}
\quad\Longrightarrow\quad
\theta_1=\theta_2.
$$

If two different parameter vectors generate exactly the same distribution, the data cannot distinguish them.

No optimizer can solve an identification problem.

This is a property of the model, not of the numerical algorithm.

## Asymptotic normality

Under standard regularity conditions,

$$
\sqrt n
(
\hat\theta-\theta_0
)
\xrightarrow{d}
\mathcal N
\left(
0,
I_1(\theta_0)^{-1}
\right),
$$

where $I_1$ denotes information per observation.

Equivalently,

$$
\operatorname{Var}(\hat\theta)
\approx
\frac{1}{n}
I_1(\theta_0)^{-1}.
$$

This approximation motivates Wald standard errors and confidence intervals.

It can fail near parameter boundaries, under weak identification, with mixture models, under nonregular likelihoods, or in small samples.

## Efficiency also needs qualification

MLE is often described as “efficient.”

Under the regular correctly specified parametric model, the MLE is asymptotically efficient in the usual Cramér-Rao sense.

That is not the statement that the MLE has minimum variance among all unbiased estimators in every finite sample.

Nor does it imply that an MLE from a wrong model is optimal for the scientific target.

The word **asymptotic** matters.

## Misspecification

Suppose the true distribution is $g(x)$ but we fit a family

$$
p(x\mid\theta).
$$

The MLE can still converge.

But it generally converges to the parameter value

$$
\theta^\ast
=
\arg\max_\theta
E_g[
\log p(X\mid\theta)
].
$$

Equivalently, this is the member of the model family minimizing Kullback-Leibler divergence from the truth, when the relevant quantities exist.

The parameter $\theta^\ast$ is a **pseudo-true parameter**.

That can be useful.

It is not evidence that the fitted model is literally true.

Under misspecification, the usual inverse-Fisher covariance formula also needs replacement by a sandwich form.

## Logistic regression

For binary outcomes,

$$
Y_i\sim\operatorname{Bernoulli}(p_i),
$$

with

$$
\operatorname{logit}(p_i)
=
x_i^\top\beta.
$$

The log-likelihood is

$$
\ell(\beta)
=
\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right].
$$

There is no general closed-form solution for $\hat\beta$.

Numerical optimization is used.

This is a genuine example of a common machine-learning loss arising directly as a negative log-likelihood.

## Not every machine-learning algorithm is MLE

The previous version of this article incorrectly grouped support vector machines, decision trees, and random forests as if they were likelihood-based MLE procedures.

They are not, in their standard forms.

A soft-margin SVM minimizes hinge loss plus a regularization term.

A decision tree recursively optimizes split criteria.

A random forest averages randomized trees.

These can be studied statistically, but they are not automatically maximum-likelihood estimators.

Neural networks are more nuanced.

A network trained with cross-entropy for a categorical outcome can be interpreted as maximizing a conditional likelihood.

A network trained under another objective may not have that interpretation.

The loss function determines the statistical connection.

## Optimization is not inference

Finding

$$
\hat\theta
=
\arg\max_\theta\ell(\theta)
$$

is an optimization problem.

Inference requires additional work.

Questions include:

- Is the optimum unique?
- Is it global or local?
- Is the parameter identifiable?
- What is the sampling distribution?
- Are standard errors valid?
- Is the model misspecified?
- Does the parameter answer the scientific question?

An optimizer returning “success” does not answer any of those.

## Local maxima and numerical issues

For a simple concave likelihood such as ordinary logistic regression without separation, optimization is well behaved.

Other likelihoods can contain:

- multiple local maxima;
- flat ridges;
- singularities;
- boundary optima;
- unbounded likelihoods.

Mixture models are a classic example.

Numerical diagnostics are part of statistical modeling.

Convergence flags, gradients, Hessians, starting values, and repeated initializations can matter.

## Reproducible Python examples

The closed-form Bernoulli and normal MLEs can be implemented without pretending an optimizer is necessary.

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def bernoulli_mle(
    observations: NDArray[np.int64],
) -> float:
    if observations.ndim != 1:
        raise ValueError(
            "observations must be one-dimensional"
        )

    if not np.all(
        (observations == 0)
        | (observations == 1)
    ):
        raise ValueError(
            "Bernoulli observations must be 0 or 1"
        )

    return float(observations.mean())


def normal_mle(
    observations: FloatArray,
) -> tuple[float, float]:
    if observations.ndim != 1:
        raise ValueError(
            "observations must be one-dimensional"
        )

    if observations.size == 0:
        raise ValueError(
            "observations cannot be empty"
        )

    mean_mle: float = float(
        observations.mean()
    )

    variance_mle: float = float(
        np.mean(
            (observations - mean_mle) ** 2
        )
    )

    return mean_mle, variance_mle


rng = np.random.default_rng(2026)

binary = rng.binomial(
    n=1,
    p=0.7,
    size=1_000,
)

normal = rng.normal(
    loc=5.0,
    scale=2.0,
    size=1_000,
)

print(bernoulli_mle(binary))
print(normal_mle(normal))
~~~

The normal variance uses denominator $n$ because it is the MLE.

That is intentional.

## Likelihood ratios

Likelihood also supports model comparison.

For nested models with maximized log-likelihoods

$$
\ell_0
$$

and

$$
\ell_1,
$$

the likelihood-ratio statistic is

$$
2(\ell_1-\ell_0).
$$

Under regular conditions, Wilks' theorem gives an asymptotic chi-square distribution with degrees of freedom equal to the difference in parameter dimension.

Again, “under regular conditions” matters.

Boundary parameters and non-identifiable models can invalidate the ordinary chi-square reference distribution.

## Bayesian inference uses the same likelihood differently

Bayesian inference combines the likelihood with a prior:

$$
p(\theta\mid x)
\propto
L(\theta;x)p(\theta).
$$

The maximum a posteriori estimator solves

$$
\hat\theta_{\mathrm{MAP}}
=
\arg\max_\theta
\left[
\ell(\theta)
+
\log p(\theta)
\right].
$$

This often resembles penalized likelihood.

But MLE and Bayesian inference answer different probability questions.

The likelihood is common to both.

The inferential framework is not.

## Conclusion

MLE is a disciplined way to estimate parameters inside a probabilistic model.

Its strongest properties are conditional:

$$
\boxed{
\text{model}
+
\text{identifiability}
+
\text{regularity}
+
\text{optimization}
\rightarrow
\text{MLE theory}
}
$$

The maximized likelihood cannot validate the model that generated it.

A good likelihood analysis therefore combines estimation with diagnostics, uncertainty, model comparison, and explicit discussion of misspecification.

## References

- Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. *Philosophical Transactions of the Royal Society A*, 222, 309–368.
- Myung, I. J. (2003). Tutorial on maximum likelihood estimation. *Journal of Mathematical Psychology*, 47(1), 90–100.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.
