---
author_profile: false
categories:
- Statistics
- Data Analysis
- Hypothesis Testing
classes: wide
date: '2020-05-01'
excerpt: Learn about the Shapiro-Wilk and Anderson-Darling tests for normality, their
  differences, and how they guide decisions between parametric and non-parametric
  statistical methods.
header:
  image: /assets/images/data_science_3.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_3.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_3.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Anderson-Darling Test
- Non-Parametric Methods
- Shapiro-Wilk Test
- Normality Test
- Parametric Methods
seo_description: Explore the differences between the Shapiro-Wilk and Anderson-Darling
  tests for checking normality in data. Learn when to use each test and how to interpret
  the results.
seo_title: 'Shapiro-Wilk Test vs. Anderson-Darling Test: Normality Tests Explained'
seo_type: article
summary: This article explores two common normality tests—the Shapiro-Wilk test and
  the Anderson-Darling test—discussing their differences, when to use each, and how
  to interpret their results to determine the appropriate statistical method.
tags:
- Anderson-Darling Test
- Normality Tests
- Non-Parametric Methods
- Shapiro-Wilk Test
- Parametric Methods
title: 'Shapiro-Wilk Test vs. Anderson-Darling Test: Checking Normality in Data'
---

In statistics, the assumption of **normality** is critical for many parametric tests, such as t-tests and ANOVA. When this assumption is violated, the results of these tests can become unreliable, leading to incorrect conclusions. As such, it is important to test whether the data follows a normal distribution before applying parametric methods.

Two of the most commonly used tests for checking normality are the **Shapiro-Wilk test** and the **Anderson-Darling test**. Both tests assess whether a dataset is consistent with a normal distribution, but they differ in their methodologies and sensitivity to different aspects of the data.

This article provides a detailed comparison of the Shapiro-Wilk and Anderson-Darling tests, explains when to use each test, and discusses how to interpret their results in order to decide whether parametric or non-parametric methods should be used in further analysis.

---

## Why Check for Normality?

Many statistical methods, especially **parametric tests** (e.g., t-tests, ANOVA, regression), rely on the assumption that the data is normally distributed. A **normal distribution** is symmetrical, with most data points clustering around the mean and fewer data points appearing at the extremes (forming the characteristic bell curve).

If the data is not normally distributed, applying parametric tests could lead to:

- **Misleading p-values**: The significance levels might be inaccurate, leading to incorrect conclusions.
- **Increased risk of Type I and Type II errors**: There could be a higher risk of falsely rejecting or failing to reject the null hypothesis.

### Parametric vs. Non-Parametric Methods

- **Parametric methods** assume that the data follows a specific distribution (typically normal) and are generally more powerful when the assumptions are met.
- **Non-parametric methods** do not make distributional assumptions and can be used when the data violates normality or when working with ordinal data.

Testing for normality helps determine whether a parametric or non-parametric method is appropriate for further analysis.

---

## Shapiro-Wilk Test: A Powerful Normality Test

### What is the Shapiro-Wilk Test?

The **Shapiro-Wilk test** is a widely used test for assessing normality. Introduced by **Shapiro and Wilk** in 1965, it tests the null hypothesis that the data comes from a normally distributed population. If the test returns a **p-value** below a chosen significance level (e.g., 0.05), the null hypothesis is rejected, indicating that the data is not normally distributed.

### How the Shapiro-Wilk Test Works

The Shapiro-Wilk test calculates a statistic based on the correlation between the data and the corresponding normal distribution. It compares the **ordered** sample values with the corresponding expected values from a normal distribution to assess how closely the data follows a normal pattern. The test statistic **W** ranges from 0 to 1, where:

- **W close to 1** indicates that the data is approximately normally distributed.
- **W significantly less than 1** suggests deviation from normality.

$$
W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

Where:

- $$ x_{(i)} $$ are the ordered sample values.
- $$ a_i $$ are constants derived from the expected values of the normal order statistics.
- $$ \bar{x} $$ is the sample mean.

### When to Use the Shapiro-Wilk Test

- **Small to moderately sized samples**: The Shapiro-Wilk test is particularly effective for sample sizes less than 50, but can also be used for larger datasets.
- **General use**: It is a powerful and reliable test for checking normality in most situations.
- **High sensitivity to small deviations**: The Shapiro-Wilk test is sensitive to deviations from normality, making it a good default choice for checking normality.

### Interpretation of Shapiro-Wilk Test Results

- **p > 0.05**: Fail to reject the null hypothesis, meaning there is insufficient evidence to suggest the data is not normally distributed.
- **p < 0.05**: Reject the null hypothesis, suggesting that the data significantly deviates from a normal distribution.

---

## Anderson-Darling Test: A More Comprehensive Approach

### What is the Anderson-Darling Test?

The **Anderson-Darling test** is another popular method for testing whether data follows a normal distribution. It is a modification of the **Kolmogorov-Smirnov test**, and it provides a more sensitive test by giving more weight to the tails of the distribution, where deviations from normality often occur.

Unlike the Shapiro-Wilk test, the Anderson-Darling test calculates a statistic that measures the discrepancy between the **empirical cumulative distribution function (CDF)** of the data and the expected CDF of the normal distribution.

### How the Anderson-Darling Test Works

The Anderson-Darling test computes the squared differences between the observed and expected cumulative distributions, with a special focus on the tails of the distribution:

$$
A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} \left( (2i - 1) \left[ \log F(x_{(i)}) + \log(1 - F(x_{(n+1-i)})) \right] \right)
$$

Where:

- $$ n $$ is the sample size.
- $$ x_{(i)} $$ are the ordered sample values.
- $$ F(x) $$ is the CDF of the normal distribution.

The Anderson-Darling test returns an **A² statistic** and compares it against critical values to determine whether to reject the null hypothesis.

### When to Use the Anderson-Darling Test

- **More emphasis on tails**: The Anderson-Darling test is particularly useful when it is important to assess deviations in the tails of the distribution (e.g., financial data where extreme values are critical).
- **Larger samples**: The Anderson-Darling test performs well with larger sample sizes, making it a good choice when analyzing larger datasets.
- **Testing for different distributions**: While commonly used for normality testing, the Anderson-Darling test can be adapted to test other distributions (e.g., exponential, Weibull).

### Interpretation of Anderson-Darling Test Results

- **p > 0.05**: Fail to reject the null hypothesis, suggesting the data follows a normal distribution.
- **p < 0.05**: Reject the null hypothesis, indicating that the data does not follow a normal distribution, with particular focus on the tails.

---

## Shapiro-Wilk vs. Anderson-Darling: Key Differences

### 1. **Sensitivity to Sample Size**

- **Shapiro-Wilk**: Best suited for **small to moderate sample sizes** (e.g., less than 50). It may become overly sensitive with very large samples, detecting even trivial deviations from normality.
- **Anderson-Darling**: Handles **larger sample sizes** well and provides more robust results with large datasets. Its performance in smaller samples can still be strong but may not outperform Shapiro-Wilk in smaller datasets.

### 2. **Focus on Tails**

- **Shapiro-Wilk**: Provides a general assessment of normality across the entire distribution. While it detects deviations from normality, it does not specifically emphasize the tails.
- **Anderson-Darling**: Places more weight on the **tails of the distribution**, making it a better choice when tail behavior is particularly important (e.g., in financial data or extreme value analysis).

### 3. **Distributional Flexibility**

- **Shapiro-Wilk**: Primarily designed for testing normality.
- **Anderson-Darling**: Can be adapted for testing normality as well as other distributions (e.g., lognormal, Weibull, etc.).

### 4. **Interpretation of Results**

- **Shapiro-Wilk**: Tends to be more conservative, with lower power to detect small deviations from normality in larger samples.
- **Anderson-Darling**: More sensitive to deviations, particularly in the tails, and provides a broader perspective on distributional fit.

---

## Deciding When to Use Shapiro-Wilk vs. Anderson-Darling

Choosing between the Shapiro-Wilk and Anderson-Darling tests depends on several factors, including the sample size, the importance of tail behavior, and the distributional assumptions underlying the analysis.

### Use Shapiro-Wilk When

- You have a **small to moderate sample size** (e.g., n < 50).
- You need a **general test for normality** across the entire distribution.
- You are conducting basic hypothesis testing that requires normality (e.g., t-tests, ANOVA).

### Use Anderson-Darling When

- You are working with a **larger sample size**.
- Deviations in the **tails of the distribution** are critical to your analysis (e.g., extreme values in financial data).
- You are testing for normality as well as other types of distributions.

---

## Application of Normality Tests in Determining Parametric vs. Non-Parametric Methods

The results of normality tests guide the choice between **parametric** and **non-parametric** statistical methods:

1. **If the data is normally distributed (p > 0.05)**:
   - You can proceed with **parametric tests** such as t-tests, ANOVA, and Pearson correlation. These tests are more powerful and provide more precise estimates when the normality assumption holds.

2. **If the data is not normally distributed (p < 0.05)**:
   - You should consider using **non-parametric tests** such as the Wilcoxon signed-rank test, Mann-Whitney U test, or Kruskal-Wallis test. These tests do not assume normality and are better suited for skewed or ordinal data.

---

## Practical Example: Testing Normality in Real-World Data

Imagine you are analyzing the performance of three different marketing campaigns and want to compare their effects on sales across 50 stores. Before conducting a parametric test like ANOVA to compare the means, you need to check whether the sales data follows a normal distribution.

1. **Shapiro-Wilk Test**: You apply the Shapiro-Wilk test and obtain a **p-value of 0.03**, indicating that the sales data does not follow a normal distribution. This suggests you should consider non-parametric alternatives like the Kruskal-Wallis test.

2. **Anderson-Darling Test**: You also apply the Anderson-Darling test, which places more emphasis on the tails of the distribution, and obtain a **p-value of 0.01**. The Anderson-Darling test confirms that the data is non-normal, particularly highlighting deviations in the extreme values (e.g., outlier stores with very high or low sales).

Based on these results, you decide to use a non-parametric method to compare the marketing campaigns.

---

## Conclusion

Both the **Shapiro-Wilk test** and **Anderson-Darling test** are valuable tools for assessing whether data is normally distributed, a critical step in determining whether to apply parametric or non-parametric methods. The Shapiro-Wilk test is particularly useful for smaller sample sizes, offering a reliable general test for normality, while the Anderson-Darling test shines in larger datasets and when the behavior of the distribution's tails is of special interest.

Understanding the differences between these two tests allows you to make more informed decisions when analyzing data, ensuring that you choose the right statistical methods for your specific context.

### Further Reading

- **"Introduction to the Practice of Statistics"** by David S. Moore and George P. McCabe – A great resource for learning about normality tests and their applications in hypothesis testing.
- **"Nonparametric Statistical Methods"** by Myles Hollander and Douglas A. Wolfe – Offers an in-depth look at non-parametric alternatives when normality assumptions are violated.
- **Online Statistical Tools**: Try implementing normality tests in Python (`scipy.stats.shapiro` and `scipy.stats.anderson`) or R for hands-on practice with real data.

---
