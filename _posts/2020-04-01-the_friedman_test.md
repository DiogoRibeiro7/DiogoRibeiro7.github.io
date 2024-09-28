---
author_profile: false
categories:
- Statistics
- Data Analysis
- Non-Parametric Tests
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
keywords:
- Friedman Test
- Repeated Measures ANOVA
- Non-Parametric Test
- Ordinal Data
seo_description: Learn about the Friedman test, its application as a non-parametric
  alternative to repeated measures ANOVA, and its use with ordinal data or non-normal
  distributions.
seo_title: 'The Friedman Test: A Non-Parametric Alternative to Repeated Measures ANOVA'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Friedman Test
- Repeated Measures ANOVA
- Non-Parametric Tests
- Ordinal Data
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
---

In data analysis, we often encounter situations where we need to compare three or more related groups. When the assumptions of normality or homogeneity of variances are not met, using parametric methods such as repeated measures ANOVA may not be appropriate. In such cases, the **Friedman test** offers a robust **non-parametric alternative**.

The Friedman test is particularly useful for analyzing **ordinal data** or **non-normal distributions** in repeated measures designs, where the same subjects are measured under different conditions or across different time points. This article will provide a detailed explanation of the Friedman test, its application, and practical examples to help you understand when and how to use this method in your analyses.

---

## What is the Friedman Test?

The **Friedman test** is a non-parametric statistical test used to detect differences in treatments across multiple test attempts. It is a rank-based test that compares **three or more paired groups**. The test is named after the American statistician **Milton Friedman**, who introduced it as an extension of the Wilcoxon signed-rank test for more than two groups.

The Friedman test is often used as an alternative to **repeated measures ANOVA** when the assumptions of normality or homogeneity of variances are violated. Unlike ANOVA, which assumes that the data is normally distributed and uses the actual data values, the Friedman test works on the **ranks** of the data, making it a more flexible option for non-parametric data.

### Key Features of the Friedman Test

- **Non-parametric**: Does not assume normal distribution of the data.
- **Rank-based**: Compares the ranks of values within subjects rather than the raw data.
- **Used for dependent samples**: Measures differences within subjects across different conditions, time points, or treatments.
- **Alternative to repeated measures ANOVA**: When the assumptions of repeated measures ANOVA (normality and equal variances) are not met.

### Hypotheses for the Friedman Test

The Friedman test evaluates the null hypothesis:

- **H₀ (Null Hypothesis)**: The distributions of the groups are the same, meaning there is no significant difference between the conditions.
  
Against the alternative hypothesis:

- **H₁ (Alternative Hypothesis)**: At least one of the conditions is different from the others.

If the null hypothesis is rejected, it indicates that there are significant differences between at least two of the groups.

---

## When and How to Use the Friedman Test

The Friedman test is ideal for scenarios where:

1. **Data is ordinal**: The values can be ranked, but the distance between the ranks is not necessarily equal.
2. **Data is not normally distributed**: The test is robust to violations of normality, making it suitable for skewed or non-normal data.
3. **Repeated measurements on the same subjects**: When the same subjects are exposed to multiple conditions or measured at different time points.
4. **Small sample sizes**: Because it is non-parametric, the Friedman test can handle smaller sample sizes better than parametric alternatives.

### Assumptions of the Friedman Test

Despite being non-parametric, the Friedman test has its own set of assumptions:

- **Repeated measures**: The data must be from the same subjects, measured under different conditions.
- **Ordinal or continuous data**: The test can handle both ordinal and continuous data as long as ranks can be assigned.
- **Independence within groups**: While the measurements are related within subjects, the observations should be independent across subjects.

### How the Friedman Test Works

The Friedman test ranks the data within each subject across the different treatments or time points. Once the ranks are calculated, the test computes the sum of ranks for each treatment. If the treatment effects are similar across all conditions, the rank sums should be approximately equal. However, if there is a treatment effect, some treatments will consistently receive higher or lower ranks.

The test statistic for the Friedman test is calculated as follows:

$$
\chi_F^2 = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)
$$

Where:

- **n** is the number of subjects.
- **k** is the number of conditions.
- **R_j** is the sum of the ranks for condition j.

The test statistic follows a chi-square distribution with **k-1 degrees of freedom**. A p-value is computed from the test statistic to determine whether to reject the null hypothesis.

---

## Practical Applications of the Friedman Test

The Friedman test is particularly useful in situations where the same subjects are measured multiple times, such as:

- **Medical research**: Testing the effectiveness of different treatments on the same patients over time.
- **Psychology**: Measuring the response of individuals to different stimuli or conditions.
- **Education**: Comparing student performance across different teaching methods.
- **Consumer research**: Rating preferences for various products by the same group of consumers.

### Example 1: Analyzing Ranking Data

Consider a scenario where a company wants to test three different packaging designs for a product and asks 10 customers to rank the designs based on preference. Each customer ranks the designs from 1 (most preferred) to 3 (least preferred).

| Customer | Design A | Design B | Design C |
|----------|----------|----------|----------|
| 1        | 1        | 3        | 2        |
| 2        | 2        | 1        | 3        |
| 3        | 3        | 2        | 1        |
| 4        | 2        | 3        | 1        |
| ...      | ...      | ...      | ...      |

Here, each customer represents a **block**, and the packaging designs are the **conditions**. The Friedman test can be used to determine if there is a statistically significant difference in customer preferences for the three designs.

### Example 2: Repeated Measurements Over Time

Imagine a clinical trial where 15 patients are given three different drugs (A, B, and C) to treat a medical condition. Each patient’s response to the drugs is measured at different time points. Since the measurements are taken on the same patients across all three conditions, the Friedman test is appropriate for determining if there is a significant difference in the effectiveness of the drugs.

| Patient | Drug A | Drug B | Drug C |
|---------|--------|--------|--------|
| 1       | 10     | 12     | 9      |
| 2       | 8      | 15     | 11     |
| 3       | 14     | 13     | 10     |
| ...     | ...    | ...    | ...    |

The test will rank the responses within each patient and calculate whether there are significant differences between the drugs.

---

## Interpretation of Results and Post-Hoc Tests

If the Friedman test indicates that there is a significant difference between conditions, it does not specify **which** conditions are different. To determine this, you can use **post-hoc tests**, such as the **Wilcoxon signed-rank test** for pairwise comparisons between groups.

### Post-Hoc Testing

After performing the Friedman test, post-hoc testing helps identify where the significant differences lie between conditions. Some common methods for post-hoc analysis include:

- **Bonferroni correction**: This method adjusts the significance level to account for multiple comparisons.
- **Wilcoxon signed-rank test**: For pairwise comparisons between specific conditions.

### Interpretation of the Friedman Test Output

- **p-value**: If the p-value is below a chosen significance level (e.g., 0.05), you reject the null hypothesis and conclude that at least one condition is different.
- **Test statistic (χ²)**: The larger the test statistic, the greater the difference between the groups.

---

## Advantages and Limitations of the Friedman Test

### Advantages

- **No assumption of normality**: The Friedman test can be used when data is non-normally distributed, making it ideal for skewed data or small sample sizes.
- **Ordinal data compatibility**: The test works well with ordinal data, where the exact values are not important, only the ranks.
- **Handles repeated measures**: Designed specifically for repeated measures where the same subjects are used across multiple conditions.

### Limitations

- **Requires post-hoc tests**: The Friedman test itself does not indicate where the differences lie between groups; additional testing is needed to pinpoint specific differences.
- **Less powerful than parametric tests**: While the Friedman test is useful for non-parametric data, it can be less powerful than parametric tests like repeated measures ANOVA when the assumptions of normality and equal variances are met.

---

## Conclusion

The **Friedman test** is a valuable tool for analyzing **non-parametric data** in repeated measures designs. It provides a robust alternative to repeated measures ANOVA when the assumptions of normality or equal variances are not met. By comparing the ranks of data within subjects across different conditions, the Friedman test can identify whether significant differences exist between groups.

This test is particularly useful in situations where ordinal data, non-normal distributions, or small sample sizes make parametric methods inappropriate. Whether you’re comparing patient responses to different treatments over time or analyzing ranking data from surveys, the Friedman test is a flexible and reliable option.

### Further Reading

- **"Nonparametric Statistical Methods"** by Myles Hollander and Douglas A. Wolfe – A comprehensive resource on non-parametric methods, including the Friedman test.
- **"Practical Statistics for Medical Research"** by Douglas G. Altman – Offers practical guidance on using the Friedman test and other statistical methods in medical research.
- **Online Statistical Resources**: Many online tutorials and statistical software packages, like R or Python’s SciPy library, offer implementations of the Friedman test for practical use.

---
