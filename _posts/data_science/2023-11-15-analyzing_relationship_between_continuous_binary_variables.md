---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-11-15'
excerpt: "A binary variable and a continuous variable can be related through mean differences, point-biserial correlation, regression, or latent-threshold models. The method should follow the estimand."
header:
  image: /assets/images/headers/photo-data-science-street-trees.jpg
  og_image: /assets/images/headers/photo-data-science-street-trees.jpg
  overlay_image: /assets/images/headers/photo-data-science-street-trees.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-street-trees.jpg
  twitter_image: /assets/images/headers/photo-data-science-street-trees.jpg
keywords:
- Point-biserial correlation
- Biserial correlation
- Binary variables
- Continuous variables
- Mean differences
- Logistic regression
permalink: '/data-science/analyzing_relationship_between_continuous_binary_variables/'
redirect_from:
- '/data analysis/analyzing_relationship_between_continuous_binary_variables/'
- '/data science/analyzing_relationship_between_continuous_binary_variables/'
seo_description: "How to analyze relationships between continuous and binary variables using mean differences, point-biserial correlation, regression, and latent-threshold models."
seo_title: "Continuous and Binary Variables: Correlation and Regression"
seo_type: article
tags:
- Correlation
- Statistics
title: "Continuous and Binary Variables: Correlation and Regression"
---

When one variable is continuous and the other is binary, there is no single privileged measure of association. The appropriate method depends on the question.

Are we comparing the mean of a continuous outcome between two groups? Are we predicting a binary outcome from a continuous predictor? Are we quantifying standardized association? Or is the observed binary variable assumed to arise by thresholding an unobserved continuous trait?

Those are different problems.

## Point-biserial correlation is Pearson correlation

If D is coded 0/1 and X is continuous, the point-biserial correlation is simply the ordinary Pearson correlation between X and D.

It can be written as

$$
r_{pb}
=
\frac{\bar X_1-\bar X_0}{s_X}
\sqrt{pq},
$$

where p and q are the sample proportions in the two groups.

The factor $\sqrt{pq}$ means that the correlation depends not only on the standardized mean difference but also on group balance.

Contrary to a common formula error, there is no extra division by $\sqrt n$. Adding n to the denominator would make the coefficient shrink mechanically with sample size and would no longer equal Pearson correlation.

## Relation to the two-sample t statistic

The point-biserial correlation and the ordinary two-sample t statistic are algebraically related under the corresponding assumptions.

For total sample size n,

$$
t
=
r\sqrt{\frac{n-2}{1-r^2}}.
$$

This means a point-biserial test of zero correlation and a two-group test of equal means encode the same one-predictor linear-model information.

## Mean differences can be more interpretable

If X is the outcome and D identifies groups, the regression

$$
X_i=\beta_0+\beta_1D_i+\varepsilon_i
$$

has

$$
\beta_1=\bar X_1-\bar X_0.
$$

This raw mean difference is often easier to interpret than a correlation.

A standardized effect such as Cohen's d can be useful when comparison across scales matters, but standardization changes the estimand.

## If the binary variable is the outcome

If Y is binary and X continuous, logistic regression is often a more natural model:

$$
\log\frac{P(Y=1\mid X)}{1-P(Y=1\mid X)}
=
\beta_0+\beta_1X.
$$

This directly models the event probability rather than summarizing the joint association through one symmetric correlation coefficient.

The direction of modelling therefore matters even though correlation itself is symmetric.

## What biserial correlation assumes

Biserial correlation is different from point-biserial correlation. It assumes that the observed dichotomy is produced by thresholding an unobserved continuous variable, typically under a latent normal model.

If latent variable Z is dichotomized at threshold c,

$$
D=\mathbf 1\{Z>c\},
$$

then biserial correlation attempts to estimate the correlation that X would have had with Z before dichotomization.

This is a strong model-based correction. It should not be used merely because one variable has two categories.

## A pass/fail example can be circular

Suppose pass/fail is obtained by thresholding the same test score that appears as the continuous variable. Correlating score with its own deterministic dichotomization mostly measures the consequence of the chosen threshold.

That is not evidence for a separate latent relationship.

Likewise, a hypertension diagnosis defined directly from blood pressure thresholds should not automatically be used as an example of biserial correlation with the same blood-pressure measure.

## Binary demographic variables

A binary group indicator can be correlated numerically with a continuous variable, but the interpretation is a standardized group difference, not evidence of a continuous latent trait behind the group label.

Modern analyses should also avoid presenting complex demographic constructs as universally binary when the measurement scheme itself is more nuanced.

## Covariate adjustment

Point-biserial correlation is unadjusted. If age, site, baseline severity, or other covariates matter, regression is more flexible:

$$
X=\beta_0+\beta_1D+Z^\top\gamma+\varepsilon.
$$

For causal interpretation, however, the covariate set must follow a causal identification strategy rather than statistical significance.

## Nonlinearity and heterogeneity

One correlation coefficient cannot represent every relationship. Group differences can vary across age, time, or other modifiers. Distributions can have equal means but different variances or tails.

Plotting the distributions and reporting group-specific summaries is therefore an important part of the analysis.

## Conclusion

The point-biserial correlation is not a special new correlation family; it is Pearson correlation with a 0/1 variable. Biserial correlation is a different, model-based quantity that assumes an underlying latent continuous dichotomy.

For most practical analyses, begin with the estimand: mean difference, risk model, standardized association, or latent-variable relationship. Then choose the method that corresponds to that target.

## References

- Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*.
- Tate, R. F. (1954). Correlation between a discrete and a continuous variable. Point-biserial correlation.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient.
