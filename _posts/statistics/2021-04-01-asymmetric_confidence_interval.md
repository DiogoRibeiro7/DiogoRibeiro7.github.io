---
permalink: '/statistics/asymmetric_confidence_interval/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2021-04-01'
excerpt: Confidence intervals need not be symmetric around an estimate. Their shape follows the parameter space, sampling distribution, transformation, and interval construction method.
header:
  image: /assets/images/headers/photo-statistics-sampling-election.jpg
  og_image: /assets/images/headers/photo-statistics-sampling-election.jpg
  overlay_image: /assets/images/headers/photo-statistics-sampling-election.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-sampling-election.jpg
  twitter_image: /assets/images/headers/photo-statistics-sampling-election.jpg
keywords:
- Asymmetric confidence interval
- Wilson score interval
- Binomial proportion
- Profile likelihood
- Transformations
- Python
seo_description: Why confidence intervals can be asymmetric, how interval construction creates the asymmetry, and why Wilson, transformed, bootstrap, and profile-likelihood intervals differ.
seo_title: 'Why Confidence Intervals Can Be Asymmetric'
seo_type: article
summary: A precise guide to asymmetric confidence intervals that separates data skewness from estimator uncertainty and derives the Wilson interval for a binomial proportion.
tags:
- Confidence Intervals
- Probability
- Statistical Inference
- Python
title: 'Why Confidence Intervals Can Be Asymmetric'
---

Confidence intervals are often introduced in the form

$$
\hat\theta \pm z_{\alpha/2}\,\widehat{\mathrm{SE}}(\hat\theta).
$$

That construction is symmetric around the point estimate. It is not the definition of a confidence interval. A confidence interval is a procedure that maps data to a set

$$
C(X)
$$

with a specified repeated-sampling coverage property, ideally

$$
P_\theta\{\theta\in C(X)\}
\approx 1-\alpha.
$$

Nothing in that definition requires the lower and upper endpoints to be equally distant from $\hat\theta$. Asymmetry is therefore not an anomaly. It usually tells us something about the parameter space, the sampling distribution, the transformation used, or the way the interval was constructed.

## Symmetric Wald intervals

The familiar symmetric interval arises from an approximation such as

$$
\frac{\hat\theta-\theta}
{\widehat{\mathrm{SE}}(\hat\theta)}
\approx
\mathcal N(0,1).
$$

Inverting this approximation gives

$$
\hat\theta
\pm
z_{1-\alpha/2}
\widehat{\mathrm{SE}}(\hat\theta).
$$

This is a **Wald interval**. Its symmetry is inherited from the normal approximation on the $\theta$ scale. If that approximation is poor, the symmetry can be misleading.

## The distribution of the raw data is not the whole explanation

A common statement is:

> skewed data produce asymmetric confidence intervals.

Sometimes they do, but the mechanism is indirect. Confidence intervals concern the sampling distribution of an **estimator**, not the marginal shape of the raw observations by itself. A highly skewed population can still yield an approximately normal sampling distribution for a mean when the sample is sufficiently large. Conversely, even with a simple binomial model, an interval for a probability near 0 or 1 should respect the parameter bounds

$$
0\le p\le1.
$$

So the key object is the estimator and interval construction, not merely whether the histogram of the observed data is skewed.

## Transformations create asymmetry when transformed back

Suppose a positive parameter satisfies

$$
\theta>0
$$

and the log parameter

$$
\eta=\log\theta
$$

is approximately normal. A symmetric interval on the log scale is

$$
\hat\eta\pm z\,\mathrm{SE}(\hat\eta).
$$

Transforming back gives

$$
\left[
\exp\{\hat\eta-z\,\mathrm{SE}(\hat\eta)\},
\;
\exp\{\hat\eta+z\,\mathrm{SE}(\hat\eta)\}
\right].
$$

This interval is multiplicatively symmetric but additively asymmetric around

$$
\hat\theta=\exp(\hat\eta).
$$

Hazard ratios, odds ratios and rate ratios are commonly reported this way. The asymmetry is a consequence of the nonlinear transformation.

## Bounded parameters need bounded intervals

For a binomial proportion,

$$
X\sim\mathrm{Binomial}(n,p),
\qquad
\hat p=\frac{X}{n}.
$$

The naive Wald interval is

$$
\hat p
\pm
z
\sqrt{
\frac{\hat p(1-\hat p)}{n}
}.
$$

Near 0 or 1, this interval can extend outside the legal parameter space. It can also have poor coverage. The Wilson score interval avoids both problems by inverting the score test rather than placing a symmetric normal interval directly around $\hat p$.

## Deriving the Wilson interval

The score-test inequality for a candidate value $p$ is

$$
\frac{(\hat p-p)^2}
{p(1-p)/n}
\le z^2.
$$

Solving this quadratic inequality for $p$ gives

$$
\frac{
\hat p+\frac{z^2}{2n}
\pm
z
\sqrt{
\frac{\hat p(1-\hat p)}{n}
+
\frac{z^2}{4n^2}
}
}{
1+\frac{z^2}{n}
}.
$$

The midpoint is not generally $\hat p$, so the endpoints are not generally symmetric around the observed proportion. That asymmetry comes from test inversion and the bounded binomial parameter space, not from a vague statement that the data are "non-normal."

## Worked example

Take

$$
x=30,
\qquad
n=100,
\qquad
\hat p=0.30.
$$

The 95% Wald interval is

$$
0.30
\pm
1.96
\sqrt{
\frac{0.30(0.70)}{100}
},
$$

which gives approximately

$$
[0.2075,0.3925].
$$

The Wilson interval is approximately

$$
[0.2189,0.3959].
$$

The two intervals differ modestly here because the sample is not extremely small and the estimate is not close to a boundary. The reason to prefer Wilson is not that it is always narrower. It is that its coverage behavior is generally much better than the simple Wald interval for binomial proportions.

## Reproducible Python

~~~python
from __future__ import annotations

import math

from statsmodels.stats.proportion import proportion_confint

x: int = 30
n: int = 100
alpha: float = 0.05
z: float = 1.959963984540054

p_hat: float = x / n
standard_error: float = math.sqrt(
    p_hat * (1.0 - p_hat) / n
)

wald: tuple[float, float] = (
    p_hat - z * standard_error,
    p_hat + z * standard_error,
)

wilson_low, wilson_high = proportion_confint(
    count=x,
    nobs=n,
    alpha=alpha,
    method="wilson",
)

print(f"Wald   : {wald}")
print(
    "Wilson : "
    f"({wilson_low:.6f}, {wilson_high:.6f})"
)
~~~

For these values, current statsmodels returns a Wilson interval close to

$$
[0.2189,0.3959].
$$

The previous version of this article reported different Wilson endpoints.

## Profile-likelihood intervals

Maximum-likelihood problems provide another natural source of asymmetry. Let

$$
\ell(\theta)
$$

be the log likelihood and let $\hat\theta$ maximize it. A likelihood-ratio interval can be obtained by retaining values satisfying

$$
2\{\ell(\hat\theta)-\ell(\theta)\}
\le
\chi^2_{1,1-\alpha}.
$$

If the likelihood surface is steeper on one side of the maximum than the other, the resulting confidence interval is asymmetric. That shape can be informative because it reflects the local geometry of the likelihood rather than forcing a quadratic approximation to be symmetric on the original parameter scale.

## Bootstrap percentile intervals

Bootstrap intervals can also be asymmetric. If

$$
\hat\theta^{*(1)},\ldots,\hat\theta^{*(B)}
$$

are bootstrap estimates, a percentile interval uses empirical quantiles such as

$$
[
q_{0.025}^\ast,
q_{0.975}^\ast
].
$$

If the bootstrap distribution is skewed, the endpoints need not be equidistant from the original estimate. More refined methods such as BCa intervals also correct for bias and skewness in the bootstrap distribution. Again, the asymmetry belongs to the estimator's uncertainty distribution.

## Bayesian credible intervals are a different object

Bayesian posterior intervals are often asymmetric too. If the posterior density is

$$
p(\theta\mid y),
$$

an equal-tail 95% credible interval uses the 2.5% and 97.5% posterior quantiles. A highest-density interval may have different endpoints because it is constructed from posterior density rather than equal tails. Those intervals answer Bayesian probability questions. They should not be called confidence intervals merely because both are reported with two endpoints.

## Asymmetry is not evidence of bias

An asymmetric interval does not imply that the point estimator is biased. It can arise from:

- a nonlinear parameter transformation;
- a bounded parameter space;
- a skewed sampling distribution;
- inversion of a non-symmetric test;
- a non-quadratic likelihood;
- a bootstrap distribution;
- a Bayesian posterior.

Bias and interval asymmetry are separate concepts.

## What to report

When an interval is asymmetric, report the point estimate and the two endpoints directly:

$$
\hat\theta=1.7,
\qquad
95\%\ \mathrm{CI}=[1.2,2.8].
$$

Do not compress it to

$$
1.7\pm0.8
$$

because no single margin of error represents the interval. More importantly, state the method used:

- Wald,
- Wilson score,
- exact binomial,
- likelihood ratio,
- bootstrap percentile,
- BCa,
- transformed-scale interval,
- or another procedure.

The method determines the coverage properties and interpretation.

## Conclusion

Confidence intervals are not required to be symmetric. Symmetric intervals arise naturally when uncertainty is approximated by a normal distribution on the reporting scale. Asymmetry appears when the parameter is bounded, the relevant scale is nonlinear, the likelihood is non-quadratic, or the interval is constructed by test inversion, resampling or another non-Wald method.

The important question is not whether the interval looks balanced. It is whether the procedure has defensible coverage for the parameter and data-generating process being studied.

## References

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133. https://doi.org/10.1214/ss/1009213286
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
