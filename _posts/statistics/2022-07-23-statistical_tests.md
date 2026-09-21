---
permalink: '/statistics/statistical_tests/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2022-07-23'
excerpt: Discover the universal structure behind statistical tests, highlighting the
  core comparison between observed and expected data that drives hypothesis testing
  and data analysis.
header:
  image: /assets/images/headers/photo-statistics-boxplots.jpg
  og_image: /assets/images/headers/photo-statistics-boxplots.jpg
  overlay_image: /assets/images/headers/photo-statistics-boxplots.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-boxplots.jpg
  twitter_image: /assets/images/headers/photo-statistics-boxplots.jpg
keywords:
- Statistical tests
- Hypothesis testing
- Structure of statistical tests
- Data analysis
- Observed vs expected data
- Statistical inference
- Test statistics
- P-value interpretation
- Statistical significance
- Common statistical test structure
- Hypothesis comparison
- Statistical methodologies
seo_description: The structure common to most statistical tests, and how comparing observed versus expected data forms the basis of hypothesis testing.
seo_title: Understanding the Universal Structure of Statistical Tests
seo_type: article
summary: This article explains the universal structure of statistical tests, focusing
  on the comparison between observed and expected data that forms the foundation of
  hypothesis testing and statistical inference.
tags:
- Hypothesis Testing
- Data Analysis
title: The Structure Behind Most Statistical Tests
---

## The Universal Structure of Statistical Tests

Statistical tests are fundamental tools in data analysis, used to make inferences about populations based on sample data. Interestingly, despite their diversity, most statistical tests follow a similar underlying structure. Understanding this structure can unlock a deeper comprehension of statistical methods and their implications.

### Observed Data vs. Expected Data

Many tests compare an observed statistic with the distribution that statistic would have under a null model. The generic structure is closer to

$$
T(X)
\quad\text{versus}\quad
T(X^\ast),\;X^\ast\sim H_0.
$$

Sometimes $T$ is literally an observed-minus-expected discrepancy, as in Pearson's chi-square test. In other tests it is a rank statistic, likelihood ratio, score, maximum deviation, or quadratic form. There is no single “observed minus expected” formula covering all tests. We begin with observed data, the actual measurements or outcomes collected from our sample. We then compare these observations to expected data, which are the theoretical values predicted under the null hypothesis.

#### The Null Hypothesis

The null hypothesis is a restriction on the data-generating model. It may represent no mean difference, no association, a specified probability, equal distributions, a parameter value, or another structured claim. The purpose of statistical testing is to evaluate whether the observed data provides sufficient evidence to reject the null hypothesis in favor of an alternative hypothesis ($H_a$).

For example, in a clinical trial comparing a new drug to a placebo, the null hypothesis might state that there is no difference in efficacy between the drug and the placebo. The expected data under the null hypothesis would reflect this lack of difference.

#### The Alternative Hypothesis

The alternative hypothesis specifies departures from the null that the test is designed to detect. It should not be described as the outcome researchers “aim to support,” because confirmatory inference should not encode a desired result. It suggests that there is a statistically significant effect, difference, or relationship. Continuing with the clinical trial example, the alternative hypothesis might state that the new drug is more effective than the placebo.

### Variability and Statistical Significance

The variability of data is central to determining the significance of the observed difference. Variability refers to how spread out the data points are. Highly variable data may require a larger difference between observed and expected values to reach statistical significance, while less variable data can achieve significance with a smaller difference.

#### Standard Deviation and Variance

Two common measures of variability are standard deviation and variance. The standard deviation is the square root of variance, not the average absolute distance from the mean, while variance ($\sigma^2$) is the average of the squared differences from the mean. Low standard deviation and variance indicate that the data points are close to the mean, while high values indicate greater spread.

#### P-Value and Statistical Significance

Statistical tests use the difference between observed and expected data, along with variability measures, to calculate a p-value. The p-value is the probability, under the null model and test procedure, of obtaining a test statistic at least as incompatible with the null as the observed statistic. It is not the probability that the observed data “occurred by chance” and not $P(H_0\mid X)$. A low p-value (typically less than 0.05) indicates that the observed data is unlikely under the null hypothesis, leading to its rejection.

### Common Statistical Tests

Many statistical tests adhere to the fundamental structure of comparing observed to expected data. Here are some widely used examples:

#### Student's t-test

A two-sample t procedure compares group means to determine if they are significantly different from each other. It uses the following formula:

$$ t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$

where $\bar{X_1}$ and $\bar{X_2}$ are the sample means, $s_1^2$ and $s_2^2$ are the sample variances, and $n_1$ and $n_2$ are the sample sizes.

#### Chi-Square Test

The Chi-Square Test assesses the association between categorical variables by comparing observed and expected frequencies. It uses the formula:

$$ \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} $$

where $O_i$ is the observed frequency and $E_i$ is the expected frequency.

#### ANOVA (Analysis of Variance)

ANOVA evaluates whether there are any statistically significant differences between the means of three or more independent groups. It decomposes the total variability into variability between groups and within groups, using the F-statistic:

$$ F = \frac{\text{Mean Square Between}}{\text{Mean Square Within}} $$

#### F-tests

An F statistic is a ratio of scaled quadratic forms. In one context it compares explained and residual mean squares in ANOVA; in another it can compare nested linear models. A variance-ratio test is one special case, not the definition of every F-test.

#### Z-test

A one-sample z-test is exact under a normal model with known population variance. In other settings, asymptotic z-tests arise because an estimator is approximately normal; “large sample” is not a standalone definition of a z-test. The formula is:

$$ Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} $$

where $\bar{X}$ is the sample mean, $\mu$ is the population mean, $\sigma$ is the population standard deviation, and $n$ is the sample size.

#### Binomial Test

The Binomial Test tests the success probability in a binomial distribution. It evaluates whether the observed proportion of successes in a sample matches a hypothesized proportion.

#### McNemar's Test

Used for paired nominal data, McNemar's Test determines if there are differences on a dichotomous trait. The test statistic is:

$$ \chi^2 = \frac{(|b - c| - 1)^2}{b + c} $$

where $b$ and $c$ are the counts of discordant pairs.

#### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test uses the ranks of paired differences. Its usual location-shift interpretation relies on a symmetric distribution of those differences; it is not a test of “population mean ranks.”

### Implications of the Structure

Understanding that these tests share a common structure allows for a more intuitive grasp of statistical analysis. It emphasizes that at the heart of hypothesis testing is the comparison of what we observe to what we expect. This insight can demystify complex statistical methods, making them more accessible and comprehensible.

For researchers and data analysts, recognizing this universal framework can enhance the design, execution, and interpretation of statistical tests. It promotes a clearer understanding of the assumptions and conditions underlying different tests, leading to more robust and reliable conclusions.

## Conclusion

The revelation that most statistical tests are built on the comparison of observed versus expected data is a powerful tool for anyone studying or applying statistics. It highlights the universality and simplicity underlying these analytical techniques, fostering a deeper understanding and more confident application of statistical tests in various fields of research and data analysis.

## References

- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 70(2), 129-133.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.

## The reference distribution is part of the test

A test statistic without its null distribution is incomplete. That null distribution may come from:

- an exact finite-sample model;
- asymptotic theory;
- randomization/permutation;
- Monte Carlo simulation;
- bootstrap calibration.

Two procedures using the same statistic can have different validity if they use different reference distributions.

## Design determines valid resampling

Permutation tests are exact only under the exchangeability induced by the null and study design. For paired data, treatment labels can usually be swapped **within pairs**, not across all rows. For randomized experiments, the randomization scheme defines the valid permutation set. This is why “use a permutation test when assumptions fail” is not a universal fallback.

## Tests should be tied to estimands

Before choosing a test, state the parameter or functional of interest:

$$
\text{mean difference},
\quad
\text{risk ratio},
\quad
\text{median},
\quad
\text{distributional equality},
\quad
\text{hazard ratio},
\ldots
$$

Different tests can reject for different reasons and therefore support different scientific claims. The common structure is not “observed minus expected.” It is

$$
\boxed{
\text{null model}
\rightarrow
\text{test statistic}
\rightarrow
\text{reference distribution}
\rightarrow
\text{decision rule}.
}
$$
