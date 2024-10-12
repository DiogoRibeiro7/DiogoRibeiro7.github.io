---
author_profile: false
categories:
- Statistics
- Non-Parametric Tests
- Data Analysis
classes: wide
date: '2023-11-16'
excerpt: Learn how the Mann-Whitney U Test is used to compare two independent samples
  in non-parametric statistics, with applications in fields such as psychology, medicine,
  and ecology.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Mann-whitney u test
- Non-parametric test
- Independent samples
- Hypothesis testing
- Psychology
- Medicine
- Bash
- Python
seo_description: Explore the Mann-Whitney U Test, a non-parametric method for comparing
  two independent samples, with applications in fields like psychology, medicine,
  and ecology.
seo_title: 'Mann-Whitney U Test: Comparing Two Independent Samples'
seo_type: article
summary: The Mann-Whitney U Test is a non-parametric method used to compare two independent
  samples. This article explains the test's assumptions, mathematical foundations,
  and its applications in fields like psychology, medicine, and ecology.
tags:
- Mann-whitney u test
- Non-parametric statistics
- Two independent samples
- Hypothesis testing
- Data analysis
- Bash
- Python
title: 'Mann-Whitney U Test: Non-Parametric Comparison of Two Independent Samples'
---

The **Mann-Whitney U Test** is a non-parametric statistical test used to compare differences between two independent samples when the assumptions of parametric tests, such as normality, are not met. Also known as the **Wilcoxon rank-sum test**, this method is widely applied in fields like **psychology**, **medicine**, and **ecology**, where researchers often need to compare two groups without assuming the data follows a normal distribution.

The Mann-Whitney U Test is particularly useful when working with small sample sizes, ordinal data, or data that may contain outliers, and it provides a robust alternative to the **t-test** when the assumptions of normality and homoscedasticity (equal variances) are violated.

In this article, we will explore the theory behind the Mann-Whitney U Test, explain its mathematical formulation, and discuss its practical applications across various fields. We will also compare it to other non-parametric tests and outline when it should be used.

## 1. Introduction to the Mann-Whitney U Test

The Mann-Whitney U Test is designed to determine whether there is a significant difference between the distributions of two independent samples. Unlike parametric tests like the **independent samples t-test**, which compare the means of two groups, the Mann-Whitney U Test compares the **ranks** of the data points, making it suitable for non-normally distributed data or data that do not meet other parametric assumptions.

### 1.1 Hypotheses of the Mann-Whitney U Test

The Mann-Whitney U Test evaluates two competing hypotheses:

- **Null hypothesis ($H_0$):** The two independent samples come from the same population, or their distributions are identical. In this case, there is no difference between the two groups.
- **Alternative hypothesis ($H_1$):** The two samples come from different populations, or their distributions differ, implying that there is a significant difference between the two groups.

The test ranks all observations from both groups combined, then compares the sums of the ranks for each group to determine whether one group tends to have higher or lower values than the other.

### 1.2 When to Use the Mann-Whitney U Test

The Mann-Whitney U Test is appropriate when:

- The data are **ordinal**, **continuous**, or **non-normally distributed**.
- The two samples being compared are **independent** (i.e., there is no relationship between the participants or observations in each sample).
- The sample sizes are relatively small or the data contain outliers.
- The assumptions of parametric tests, such as the independent samples t-test, are violated (e.g., when data are skewed or variances are unequal).

### 1.3 Assumptions of the Mann-Whitney U Test

Despite being a non-parametric test, the Mann-Whitney U Test still has certain assumptions:

- **Independence:** The observations in each sample must be independent of one another.
- **Ordinal or continuous data:** The data should be ordinal or continuous in nature.
- **Comparability of distributions:** The test assumes that the two distributions have the same shape. If this assumption is violated, the test might compare the medians rather than the entire distribution.

## 2. Mathematical Foundation of the Mann-Whitney U Test

The Mann-Whitney U Test is based on the idea of ranking all the data points from both groups, then comparing the sum of the ranks for each group. Here's a breakdown of how the test works.

### 2.1 Ranking the Data

To conduct the Mann-Whitney U Test, the first step is to rank all the observations from both samples together in ascending order, assigning ranks from 1 to $n$, where $n$ is the total number of observations across both groups. If any values are tied, the average rank for those tied values is used.

### 2.2 Calculating the U Statistic

Once the ranks are assigned, the **U statistic** is calculated for each group. The U statistic represents the number of times a value from one sample precedes a value from the other sample in the ranked data.

The formula for calculating the U statistic for each group is:

$$
U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
$$

Where:

- $n_1$ is the number of observations in the first group.
- $n_2$ is the number of observations in the second group.
- $R_1$ is the sum of ranks for the first group.

Similarly, for the second group:

$$
U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2
$$

Where $R_2$ is the sum of ranks for the second group.

The smaller of $U_1$ and $U_2$ is used as the test statistic ($U$), representing the number of times one sample precedes the other in the rank order. This is compared against a critical value from the Mann-Whitney U distribution, or converted to a **Z-score** for large samples.

### 2.3 Z-Score for Large Samples

For larger sample sizes ($n_1 \geq 20$ or $n_2 \geq 20$), the U statistic can be approximated by a **normal distribution** and converted into a **Z-score**:

$$
Z = \frac{U - \mu_U}{\sigma_U}
$$

Where:

- $\mu_U = \frac{n_1 n_2}{2}$ is the mean of the U distribution.
- $\sigma_U = \sqrt{\frac{n_1 n_2 (n_1 + n_2 + 1)}{12}}$ is the standard deviation of the U distribution.

The Z-score is then compared to a standard normal distribution to determine the p-value, which indicates whether the observed difference between the two groups is statistically significant.

## 3. Applications of the Mann-Whitney U Test

The Mann-Whitney U Test is widely used in research where parametric assumptions cannot be met, and it has a range of applications across fields like **psychology**, **medicine**, and **ecology**.

### 3.1 Psychology

In psychology, the Mann-Whitney U Test is frequently used to compare groups on variables that are ordinal or non-normally distributed, such as survey responses, reaction times, or behavioral measures.

#### Example: Comparing Stress Levels

Researchers might use the Mann-Whitney U Test to compare stress levels (measured on a Likert scale) between two independent groups of participants, such as a treatment group and a control group. Since Likert scales are ordinal, and the distribution of responses may be skewed, the Mann-Whitney U Test is an appropriate choice for comparing the groups.

### 3.2 Medicine

In medical research, the Mann-Whitney U Test is commonly applied to compare treatment outcomes when the data do not meet the assumptions of parametric tests. For instance, it can be used to evaluate the effectiveness of different treatments when the outcome variable is non-normally distributed, such as patient recovery times, blood pressure measurements, or pain scores.

#### Example: Comparing Recovery Times

Suppose a clinical trial compares the recovery times (in days) between two groups of patients, one receiving a new drug and the other receiving a placebo. If the recovery times are skewed, the Mann-Whitney U Test can be used to determine whether the new drug leads to significantly faster recovery compared to the placebo.

### 3.3 Ecology

In ecology, the Mann-Whitney U Test is often used to compare environmental variables or species measurements between two different habitats or populations. Ecological data are frequently non-normally distributed, making the Mann-Whitney U Test a valuable tool for comparing groups in studies where the assumptions of parametric tests are violated.

#### Example: Comparing Species Abundance

Ecologists might apply the Mann-Whitney U Test to compare the abundance of a particular species in two different habitats. Since species abundance data are often skewed or zero-inflated, the Mann-Whitney U Test provides a robust method for assessing whether there is a significant difference in abundance between the two habitats.

## 4. Mann-Whitney U Test vs. Other Non-Parametric Tests

While the Mann-Whitney U Test is one of the most commonly used non-parametric tests, it is not the only option available for comparing two independent samples. Below are some comparisons with other non-parametric tests.

### 4.1 Mann-Whitney U Test vs. Wilcoxon Signed-Rank Test

The **Wilcoxon signed-rank test** is similar to the Mann-Whitney U Test, but it is used for **paired samples** or **dependent groups**. If the two samples being compared are not independent (for example, if the same participants are measured under two different conditions), the Wilcoxon signed-rank test should be used instead of the Mann-Whitney U Test.

### 4.2 Mann-Whitney U Test vs. Kruskal-Wallis Test

The **Kruskal-Wallis test** is an extension of the Mann-Whitney U Test that can be used to compare more than two independent groups. If your study involves more than two groups and you want to test for differences between them, the Kruskal-Wallis test is the appropriate non-parametric alternative to one-way ANOVA.

### 4.3 Mann-Whitney U Test vs. t-Test

The Mann-Whitney U Test is often used as a non-parametric alternative to the **independent samples t-test**. The t-test assumes that the data are normally distributed and have equal variances between groups. When these assumptions are not met, the Mann-Whitney U Test provides a reliable alternative for testing differences between two groups.

## 5. Implementing the Mann-Whitney U Test in Python

The **Mann-Whitney U Test** can be easily implemented using the `scipy` library in Python. Below is a step-by-step guide for performing the test on two independent samples.

### 5.1 Installing Required Libraries

If you don't already have `scipy` installed, you can install it using `pip`:

```bash
pip install scipy
```

### 5.2 Example Code

```python
from scipy.stats import mannwhitneyu

# Sample data: Two independent groups (e.g., scores from two different groups)
group1 = [50, 55, 60, 65, 70]
group2 = [30, 35, 40, 45, 50]

# Perform the Mann-Whitney U Test
stat, p_value = mannwhitneyu(group1, group2)

# Print the test statistic and p-value
print(f"U statistic: {stat}")
print(f"P-value: {p_value}")
```

### 5.3 Interpreting the Results

In the output, the U statistic provides the test statistic for the Mann-Whitney U Test, and the p-value indicates whether the difference between the two groups is statistically significant. If the p-value is below the chosen significance level (e.g., 0.05), you can reject the null hypothesis and conclude that there is a significant difference between the two groups.

The Mann-Whitney U Test is a powerful and widely used non-parametric test for comparing two independent samples, especially when the data do not meet the assumptions of parametric tests like the independent samples t-test. By ranking the data and comparing the sums of the ranks, the Mann-Whitney U Test provides a robust method for detecting differences between groups in a wide range of fields, including psychology, medicine, and ecology.

Its non-parametric nature, ease of use, and applicability to small sample sizes make the Mann-Whitney U Test an essential tool for researchers working with non-normally distributed data or ordinal data. By understanding when and how to apply the Mann-Whitney U Test, researchers can confidently analyze their data and draw meaningful conclusions about group differences.
