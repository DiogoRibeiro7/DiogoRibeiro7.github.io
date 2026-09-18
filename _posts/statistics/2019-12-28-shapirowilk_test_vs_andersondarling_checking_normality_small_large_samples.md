---
author_profile: false
categories:
- Statistics
classes: wide
date: '2019-12-28'
excerpt: Shapiro-Wilk and Anderson-Darling are useful diagnostics for normality, but choosing between them by sample-size thresholds is misleading. The real question is what normality assumption matters for the analysis and whether deviations are consequential.
header:
  image: /assets/images/headers/photo-statistics-ecdf.jpg
  og_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-ecdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-ecdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-ecdf.jpg
keywords:
- Shapiro-Wilk test
- Anderson-Darling test
- normality testing
- Q-Q plot
- residual diagnostics
- sample size
- statistical assumptions
- Python
seo_description: A corrected guide to Shapiro-Wilk and Anderson-Darling tests, explaining why normality testing should not be reduced to sample-size cutoffs and how to assess whether departures from normality matter.
seo_title: 'Shapiro-Wilk vs Anderson-Darling: What Normality Tests Can Tell You'
seo_type: article
summary: Shapiro-Wilk and Anderson-Darling test different aspects of departure from normality, but neither should be selected by a simple small-sample versus large-sample rule. This revision explains what should actually be tested, how power changes with sample size, and why graphical and model-based diagnostics matter.
tags:
- Hypothesis Testing
- Diagnostics
- Sample Size
- Python
title: 'Shapiro-Wilk vs. Anderson-Darling: What Normality Tests Can and Cannot Tell You'
---

> **Revision note, September 2026.** The original 2019 version of this article recommended Shapiro-Wilk for small samples and Anderson-Darling for large samples. That rule is too simplistic and, in important cases, wrong. I have rewritten the article to distinguish the two tests more carefully and, more importantly, to explain why formal normality testing should not be used as an automatic gatekeeper for t-tests, ANOVA, regression, or other downstream analyses.

Normality tests are easy to run and surprisingly easy to misuse.

A common workflow is:

$$
\text{test normality}
\rightarrow
\begin{cases}
\text{use parametric method}, & p > 0.05,\\
\text{use nonparametric method}, & p \le 0.05.
\end{cases}
$$

That looks systematic. It is often bad statistical practice.

The first question should not be

> Which normality test should I use for this sample size?

It should be

$$
\boxed{
\text{What normality assumption, if any, is required by the analysis I actually want to perform?}
}
$$

Only after answering that does it make sense to discuss Shapiro-Wilk, Anderson-Darling, Q-Q plots, residual diagnostics, transformations, robust methods, or alternative models.

## What a normality test actually tests

For a sample

$$
X_1,\ldots,X_n,
$$

a normality test typically considers the null hypothesis

$$
H_0: X_i \sim \mathcal N(\mu,\sigma^2)
$$

for some unknown parameters $\mu$ and $\sigma$.

That is a very specific hypothesis: **exact distributional normality**.

Two consequences follow immediately.

First, failing to reject $H_0$ does not prove that the population is normal. A small sample may simply have too little power to detect the departure that exists.

Second, rejecting $H_0$ does not tell us whether the departure matters for the analysis we care about. With enough observations, a formal test can identify deviations from exact normality that are scientifically or inferentially negligible.

So the output of a normality test is not

$$
\text{normal} \quad\text{versus}\quad \text{non-normal}.
$$

It is evidence about the compatibility of the data with a particular normal model.

## The normality assumption is often attached to the wrong object

The original version of this article stated too broadly that t-tests and ANOVA require the observed data themselves to be normally distributed.

That is not the right general statement.

For a classical linear model,

$$
Y = X\beta + \varepsilon,
$$

the finite-sample normal-theory assumption concerns the errors

$$
\varepsilon \mid X,
$$

not the marginal distribution of $Y$ across every observation pooled together.

Similarly:

- in a **paired t-test**, the relevant object is the distribution of the paired differences;
- in a **two-sample t-test**, exact small-sample theory is based on assumptions about the distributions within groups, while large-sample inference can be much more robust;
- in **ANOVA**, diagnostics concern within-group errors or residuals rather than the unadjusted response distribution treated as one homogeneous sample;
- in **regression**, a histogram of the raw outcome may tell us very little about whether the model's error structure is adequate.

This distinction matters because testing the wrong object can reject a model that is perfectly reasonable for the scientific question.

## Why sample-size cutoffs are the wrong way to choose a test

The old rule

$$
\text{Shapiro-Wilk if }n<50,
\qquad
\text{Anderson-Darling if }n\text{ is large}
$$

should be discarded.

There is no general sample-size threshold at which Anderson-Darling becomes the correct replacement for Shapiro-Wilk.

Power depends on at least three things:

$$
\boxed{
\text{sample size}
+
\text{kind of departure from normality}
+
\text{test statistic}
}
$$

Simulation studies comparing normality tests repeatedly show that rankings change with the alternative distribution. Shapiro-Wilk often has very strong omnibus power, while Anderson-Darling is designed to place greater weight on discrepancies in the tails. Neither statement implies a universal sample-size switching rule.

And crucially, **both tests become more sensitive as the sample size increases**.

Anderson-Darling does not solve the large-sample problem of detecting tiny but irrelevant deviations. A large enough dataset can make either test reject a distribution that is close enough to normal for the intended inference.

## Shapiro-Wilk: an order-statistic correlation test

The Shapiro-Wilk statistic is

$$
W =
\frac{
\left(\sum_{i=1}^{n} a_i X_{(i)}\right)^2
}{
\sum_{i=1}^{n}(X_i-\bar X)^2
},
$$

where

$$
X_{(1)}\le\cdots\le X_{(n)}
$$

are the ordered observations and the coefficients $a_i$ are derived from the expected order statistics of a normal sample.

The test is sensitive to systematic departures in the ordered sample from what normal order statistics should look like.

Historically, Shapiro-Wilk earned its reputation because it performs very well across many non-normal alternatives. That does **not** mean it is only a small-sample test.

The useful interpretation is instead:

> Shapiro-Wilk is a strong general-purpose test of normality, but its p-value must still be interpreted in the context of sample size and the scientific consequences of non-normality.

There is also a software-specific caveat. Current SciPy documentation notes that for

$$
n>5000,
$$

the $W$ statistic is accurate but its reported p-value may not be. That is an implementation limitation, not a theoretical rule saying that Shapiro-Wilk suddenly becomes statistically inappropriate at $n=5001$.

## Anderson-Darling: an empirical-distribution test with tail emphasis

The Anderson-Darling statistic belongs to the family of empirical distribution function tests.

In simplified form it measures a weighted discrepancy between the empirical distribution

$$
F_n(x)
$$

and the fitted theoretical distribution

$$
F(x).
$$

For ordered observations, the familiar computational form includes terms such as

$$
\log F(X_{(i)})
$$

and

$$
\log\left[1-F(X_{(n+1-i)})\right],
$$

which gives observations near the tails substantial influence.

That is the key conceptual difference:

$$
\boxed{
\text{Anderson-Darling deliberately emphasizes tail departures.}
}
$$

This can be valuable when tail behaviour is substantively important. Examples include reliability, risk, extremes, or analyses in which tail probabilities drive decisions.

But “tail-sensitive” is not the same as “better for large samples.”

## Small samples: failure to reject may mean very little

Suppose $n=12$ and the Shapiro-Wilk test returns

$$
p=0.23.
$$

It is tempting to write:

> The data are normally distributed.

That conclusion is too strong.

A better statement is:

> The sample does not provide strong evidence against the fitted normal model.

With only twelve observations, many non-normal populations can easily produce samples that look unremarkable. Formal tests have limited power to discriminate among nearby distributions.

This is one reason graphical diagnostics are particularly important in small samples, even though the graphs themselves are also noisy.

## Large samples: statistical significance can become scientifically uninteresting

Now suppose $n=100{,}000$.

The true population distribution is almost normal but has a tiny amount of skewness.

A formal test may produce

$$
p < 10^{-10}.
$$

That result answers the narrow mathematical question:

$$
\text{Is the population exactly normal?}
$$

It does **not** answer the question:

$$
\text{Does this departure materially damage my estimator, interval, prediction, or decision?}
$$

Those are different questions.

For many estimators of means and regression coefficients, large-sample inference can remain useful far outside exact normality. Lumley, Diehr, Emerson, and Chen made this point forcefully in their review of large public-health datasets: the practical value of t-tests and linear regression does not come from real outcomes being exactly normal.

## Normality should not be a pre-test for choosing the scientific question

There is another subtle problem with the usual workflow.

Researchers sometimes do this:

1. test normality;
2. if $p>0.05$, perform a t-test;
3. otherwise, perform a rank-based test.

But a t-test and a rank-based test do not necessarily estimate or test the same scientific quantity.

A two-sample t-test is fundamentally about a difference in means.

A Wilcoxon-Mann-Whitney procedure is based on ranks and has a different probabilistic interpretation unless additional distributional assumptions are imposed.

So the choice between them should not be reduced to whether a normality test crossed $0.05$.

The estimand comes first.

## Q-Q plots answer a different and often more useful question

A normal Q-Q plot compares observed order statistics with theoretical normal quantiles.

The point is not merely to decide whether the dots lie perfectly on a line. They almost never do.

The useful questions are structural:

- Is there systematic skewness?
- Are both tails heavier than expected?
- Is one tail driving the discrepancy?
- Are there a few isolated outliers?
- Does the centre fit well while the extremes do not?
- Is there evidence of a mixture or several populations?

These patterns can tell us *how* the normal model fails, which is often more useful than knowing that a formal hypothesis test rejected it.

## A better diagnostic workflow

Instead of choosing Shapiro-Wilk or Anderson-Darling from a table of sample-size thresholds, I would use the following sequence.

### 1. Identify the statistical object

Ask what needs to be approximately normal, if anything:

$$
\text{raw observations? residuals? paired differences? transformed values?}
$$

### 2. Identify why normality matters

Is normality needed for:

- exact finite-sample inference;
- prediction intervals;
- a likelihood model;
- tail probabilities;
- residual modelling;
- merely a convenient approximation?

The consequence determines how strict the diagnostic needs to be.

### 3. Plot the data or residuals

Use a Q-Q plot together with context-specific residual plots. Look for the *shape* of the departure rather than merely a binary verdict.

### 4. Use a formal test if it answers a useful question

Shapiro-Wilk is a strong omnibus option. Anderson-Darling is attractive when tail discrepancies deserve extra weight.

But interpret

$$
p
$$

as evidence against exact normality, not as a certificate of whether a downstream method is valid.

### 5. Evaluate consequences directly

If the analysis matters, check the robustness of the actual inferential target.

That may mean:

- heteroskedasticity-robust standard errors;
- bootstrap intervals;
- permutation procedures appropriate to the null hypothesis;
- robust regression;
- alternative likelihoods;
- transformations;
- simulation under plausible non-normal distributions;
- sensitivity analyses for outliers or tail assumptions.

This is usually more informative than debating whether a normality-test p-value is $0.04$ or $0.06$.

## Python examples

### Shapiro-Wilk

```python
import numpy as np
from scipy.stats import shapiro

rng = np.random.default_rng(2026)
x = rng.normal(size=80)

result = shapiro(x)
print(result.statistic, result.pvalue)
```

### Anderson-Darling

Current SciPy versions can evaluate Anderson-Darling against a normal distribution directly:

```python
from scipy.stats import anderson

result = anderson(x, dist="norm")
print(result.statistic)
```

Depending on the SciPy version and method used, the result may expose critical values or a p-value. The important point is to consult the documentation for the installed version rather than assuming all releases expose the same interface.

### Q-Q plot

```python
import matplotlib.pyplot as plt
from scipy.stats import probplot

probplot(x, dist="norm", plot=plt)
plt.show()
```

For model diagnostics, replace `x` with the relevant residuals or differences rather than automatically testing the raw outcome.

## Shapiro-Wilk versus Anderson-Darling

A better comparison is therefore:

| Question | Shapiro-Wilk | Anderson-Darling |
| --- | --- | --- |
| General-purpose normality test | Strong choice | Strong choice |
| Tail discrepancies especially important | Sensitive | Explicitly tail-weighted |
| Choose solely because $n<50$ | No | No |
| Choose solely because $n>200$ | No | No |
| Becomes sensitive to small deviations as $n$ grows | Yes | Yes |
| Replaces graphical/model diagnostics | No | No |
| Determines automatically whether a t-test or ANOVA is valid | No | No |

## The main lesson

The useful distinction is not

$$
\boxed{
\text{small sample} \Rightarrow \text{Shapiro-Wilk}
}
$$

versus

$$
\boxed{
\text{large sample} \Rightarrow \text{Anderson-Darling}.
}
$$

It is

$$
\boxed{
\text{What departure from the model matters, and what would that departure change?}
}
$$

Shapiro-Wilk and Anderson-Darling are both useful tools. Neither should be treated as a mechanical permission slip for the rest of the analysis.

Normality is a modelling assumption whose importance depends on the estimand, model, sample size, and inferential procedure.

That is a much more useful question than asking which normality test wins a sample-size contest.

## References

- Shapiro, S. S., & Wilk, M. B. (1965). *An analysis of variance test for normality (complete samples).* Biometrika, 52(3/4), 591-611. https://doi.org/10.2307/2333709
- Anderson, T. W., & Darling, D. A. (1954). *A test of goodness of fit.* Journal of the American Statistical Association, 49(268), 765-769. https://doi.org/10.1080/01621459.1954.10501232
- Stephens, M. A. (1974). *EDF statistics for goodness of fit and some comparisons.* Journal of the American Statistical Association, 69(347), 730-737. https://doi.org/10.1080/01621459.1974.10480196
- Yap, B. W., & Sim, C. H. (2011). *Comparisons of various types of normality tests.* Journal of Statistical Computation and Simulation, 81(12), 2141-2155. https://doi.org/10.1080/00949655.2010.520163
- Lumley, T., Diehr, P., Emerson, S., & Chen, L. (2002). *The importance of the normality assumption in large public health data sets.* Annual Review of Public Health, 23, 151-169. https://doi.org/10.1146/annurev.publhealth.23.100901.140546
- SciPy documentation. `scipy.stats.shapiro` and `scipy.stats.anderson`, consulted September 2026.
