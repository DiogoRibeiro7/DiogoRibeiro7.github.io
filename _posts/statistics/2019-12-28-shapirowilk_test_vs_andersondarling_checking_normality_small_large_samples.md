---
author_profile: false
categories:
- Statistics
classes: wide
date: '2019-12-28'
excerpt: Explore the differences between the Shapiro-Wilk and Anderson-Darling tests, two common methods for testing normality, and how sample size and distribution affect their performance.
header:
  image: /assets/images/data_science_20.jpg
  og_image: /assets/images/data_science_20.jpg
  overlay_image: /assets/images/data_science_20.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_20.jpg
  twitter_image: /assets/images/data_science_20.jpg
keywords:
- Shapiro-wilk test
- Anderson-darling test
- Normality test
- Small sample size
- Large sample size
- Statistical distribution
- Python
- python
seo_description: A comparison of the Shapiro-Wilk and Anderson-Darling tests for normality, analyzing their strengths and weaknesses based on sample size and distribution.
seo_title: 'Shapiro-Wilk vs Anderson-Darling: Normality Tests for Small and Large Samples'
seo_type: article
summary: This article compares the Shapiro-Wilk and Anderson-Darling tests, emphasizing how sample size and distribution characteristics influence the choice of method when assessing normality.
tags:
- Normality testing
- Shapiro-wilk test
- Anderson-darling test
- Sample size
- Python
- python
title: 'Shapiro-Wilk Test vs. Anderson-Darling: Checking for Normality in Small vs. Large Samples'
---

## Shapiro-Wilk Test vs. Anderson-Darling: Checking for Normality in Small vs. Large Samples

Testing for normality is a crucial step in many statistical analyses, particularly when using parametric tests that assume data is normally distributed. Two of the most widely used normality tests are the **Shapiro-Wilk test** and the **Anderson-Darling test**. Although both are used to assess whether a dataset follows a normal distribution, they perform differently depending on sample size and the underlying distribution characteristics. This article explores these differences and guides how to choose the appropriate test based on your data.

### 1. Understanding the Basics of Normality Testing

In statistics, many parametric tests (such as t-tests or ANOVAs) require the assumption that the data follows a normal distribution. While visual methods like histograms or Q-Q plots are useful for assessing normality, formal statistical tests like Shapiro-Wilk and Anderson-Darling provide quantitative measures.

#### Why Is Normality Important?

- **Parametric tests** (like the t-test, ANOVA) are based on the assumption that the underlying data follows a normal distribution.
- **Non-normal data** can lead to inaccurate results in hypothesis testing, confidence intervals, and other statistical inferences.

The objective of normality tests is to determine whether to reject the hypothesis that a dataset is drawn from a normally distributed population.

### 2. Shapiro-Wilk Test: Best for Small Samples

The **Shapiro-Wilk test** is commonly regarded as the most powerful test for detecting deviations from normality, especially for **small sample sizes** (usually \( n < 50 \)). It was introduced in 1965 by Shapiro and Wilk and is based on the correlation between the data and the corresponding normal scores.

#### How Does It Work?

The Shapiro-Wilk test compares the ordered data points with the expected values of a normal distribution. The null hypothesis (\( H_0 \)) for the Shapiro-Wilk test states that the data is normally distributed. If the test produces a **p-value** below a predefined significance level (commonly 0.05), the null hypothesis is rejected, suggesting that the data is not normally distributed.

- **Test statistic**: The test statistic \( W \) is calculated using the equation:

  $$
  W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
  $$

  where \( a_i \) are constants generated from a normal distribution, \( x_{(i)} \) are the ordered sample values, and \( \bar{x} \) is the sample mean.

#### Strengths of Shapiro-Wilk

- **High power with small samples**: The Shapiro-Wilk test is highly effective in detecting non-normality in small datasets, typically outperforming other tests when \( n \) is below 50.
- **Sensitive to skewness and kurtosis**: It can detect deviations due to both the shape of the distribution and extreme values.

#### Limitations

- **Less effective for large samples**: When sample sizes increase significantly (e.g., \( n > 2000 \)), the Shapiro-Wilk test becomes overly sensitive and may flag trivial deviations as significant.
- **Slower computation**: The test involves more complex calculations, making it computationally heavier for larger datasets.

### 3. Anderson-Darling Test: Better for Large Samples

The **Anderson-Darling test** is another widely used normality test, which is a modification of the Kolmogorov-Smirnov test. It provides a more sensitive measure of the difference between the empirical distribution of the data and the expected cumulative distribution of a normal distribution. Unlike the Shapiro-Wilk test, the Anderson-Darling test performs well with **larger sample sizes**.

#### How Does It Work?

The Anderson-Darling test compares the observed cumulative distribution function (CDF) of the data to the expected CDF of the normal distribution. The test statistic \( A^2 \) is calculated based on the differences between these functions, giving more weight to the tails of the distribution:

- **Test statistic**: The Anderson-Darling statistic is computed as:

  $$
  A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} \left[ (2i-1) \left( \ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)})) \right) \right]
  $$

  where \( F(x) \) is the cumulative distribution function of the normal distribution.

#### Strengths of Anderson-Darling

- **More sensitive to tail behavior**: The Anderson-Darling test gives more weight to observations in the tails of the distribution, making it particularly useful for detecting deviations in the extremes.
- **Suitable for larger samples**: It performs well with larger datasets and remains powerful for both small and large samples, though it is especially reliable for larger datasets (e.g., \( n > 50 \)).

#### Limitations

- **Less powerful for small samples**: The Anderson-Darling test may not detect non-normality as effectively as the Shapiro-Wilk test for small datasets.
- **More prone to Type I errors**: In very large samples, it may detect statistically significant but practically negligible deviations from normality.

### 4. Choosing Between Shapiro-Wilk and Anderson-Darling

The choice between Shapiro-Wilk and Anderson-Darling tests depends primarily on the **sample size** and the **type of deviations** you expect from normality.

#### Small Samples (\( n < 50 \))

For small sample sizes, the Shapiro-Wilk test is generally preferred due to its higher power and reliability. It is more sensitive to deviations in both the center and tails of the distribution in smaller datasets.

- **Recommendation**: Use Shapiro-Wilk for \( n < 50 \).

#### Large Samples (\( n > 200 \))

As sample size increases, the Shapiro-Wilk test can become too sensitive, flagging minor deviations as statistically significant. The Anderson-Darling test, with its focus on tail behavior, often provides a more balanced view of normality for larger samples.

- **Recommendation**: Use Anderson-Darling for larger samples, especially if deviations in the tails are of particular interest.

#### Mid-range Samples (\( 50 \leq n \leq 200 \))

For datasets that fall in this mid-range, both tests can be useful, depending on the nature of the data. If your analysis is concerned with tail behavior or extreme values, the Anderson-Darling test may be more informative. However, the Shapiro-Wilk test remains a reliable choice if computational efficiency is not a concern.

### 5. Impact of Distribution Characteristics on Test Choice

Different distributions, especially those with heavy tails, skewness, or kurtosis, can influence the performance of normality tests. Both the Shapiro-Wilk and Anderson-Darling tests can detect non-normality, but their focus differs slightly.

- **Tail-heavy distributions**: The Anderson-Darling test is better suited for detecting deviations in the tails.
- **Symmetry and kurtosis**: The Shapiro-Wilk test is generally better at identifying issues related to skewness and kurtosis in smaller datasets.

### 6. Practical Considerations and Software Implementation

Both the Shapiro-Wilk and Anderson-Darling tests are widely implemented in statistical software such as R, Python (via SciPy), and SPSS. Here are examples of how to perform these tests in Python:

#### Shapiro-Wilk in Python

```python
from scipy.stats import shapiro

data = [4.5, 5.6, 7.8, 4.3, 6.1]
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

#### Anderson-Darling in Python

```python
from scipy.stats import anderson

data = [4.5, 5.6, 7.8, 4.3, 6.1]
result = anderson(data)
print('Statistic: %.3f' % result.statistic)
```

### 7. Conclusion: Which Test Should You Use?

Ultimately, the decision between the Shapiro-Wilk and Anderson-Darling tests depends on your sample size and the nature of the deviations you want to detect. For small samples, the Shapiro-Wilk test is a powerful and reliable option, while the Anderson-Darling test offers a more flexible and tail-sensitive approach, particularly useful for larger datasets.

Both tests provide valuable insights into the distribution of your data, ensuring you can make informed decisions in parametric testing and beyond.
