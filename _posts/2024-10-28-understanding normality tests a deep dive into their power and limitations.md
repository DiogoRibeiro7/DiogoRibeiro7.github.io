---
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2024-10-28'
excerpt: An in-depth look at normality tests, their limitations, and the necessity of data visualization.
header:
  image: /assets/images/data_science_5.jpg
  og_image: /assets/images/data_science_5.jpg
  overlay_image: /assets/images/data_science_5.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_5.jpg
  twitter_image: /assets/images/data_science_5.jpg
keywords:
- Normality Tests
- Statistics
- Data Analysis
- QQ Plots
- python
- r
- ruby
seo_description: An in-depth exploration of normality tests, their limitations, and the importance of visual inspection for assessing whether data follow a normal distribution.
seo_title: 'Understanding Normality Tests: A Deep Dive'
seo_type: article
summary: This article delves into the intricacies of normality testing, revealing the limitations of common tests and emphasizing the importance of visual tools like QQ plots and CDF plots.
tags:
- Normality Tests
- Statistical Methods
- Data Visualization
- python
- r
- ruby
title: 'Understanding Normality Tests: A Deep Dive into Their Power and Limitations'
---

**Abstract:**  
In statistical analysis, assessing whether data follow a normal distribution is a critical step that influences subsequent tests and interpretations. However, the concept of "normality tests" is often misunderstood. This article explores the intricacies of normality testing, highlighting the limitations of these tests, the variety of methods available, and the importance of understanding what each test measures. We will delve into a unique distribution that challenges several normality tests, demonstrating why visual inspection and a comprehensive understanding of the data are indispensable.

## Introduction

In the realm of statistics, the normal distribution holds a place of central importance. Many statistical tests and models assume that data follow a normal (Gaussian) distribution, which is characterized by its bell-shaped curve, symmetry around the mean, and specific properties of skewness and kurtosis. Assessing normality is thus a fundamental step in data analysis, ensuring the validity of inferential statistics that rely on this assumption.

However, the term "normality test" is somewhat of a misnomer. No statistical test can prove that data are normally distributed. Instead, these tests can only assess whether there is enough evidence to reject the null hypothesis ($H_0$) that the data come from a normal distribution. Failure to reject $H_0$ does not confirm normality; it merely suggests that the data are not significantly different from what would be expected under normality, within the test's power and limitations.

This article aims to shed light on the complexities of normality testing. We will examine a specially constructed distribution with properties that challenge several normality tests, revealing how different tests can yield contradictory results. By exploring the underlying mechanics of these tests and the nature of our example distribution, we will understand why visual tools like Quantile-Quantile (QQ) plots and empirical Cumulative Distribution Function (CDF) plots are essential complements to formal statistical tests.

## The Special Distribution: A Challenge to Normality Tests

Before diving into the tests themselves, let's consider the distribution that serves as our focal point. This distribution is crafted to exhibit certain characteristics that make it a suitable candidate for exploring the nuances of normality testing.

### Characteristics of the Distribution

- **Perfect Symmetry (Skewness ≈ 0):** The distribution is symmetric around its mean. Skewness, a measure of asymmetry, is approximately zero, indicating that the left and right tails of the distribution are mirror images of each other.
  
- **Normal Kurtosis (Kurtosis ≈ 3):** The kurtosis of the distribution is close to 3, which is the kurtosis of a normal distribution. Excess kurtosis (kurtosis minus 3) is approximately zero, suggesting that the tails of the distribution have a similar heaviness to those of a normal distribution.
  
- **Bimodality:** Despite its symmetry and normal kurtosis, the distribution is bimodal, meaning it has two distinct peaks or modes. There is a noticeable "hole" or dip in the middle of the distribution, deviating from the single-peaked nature of a normal distribution.

- **Non-normal Nature:** Given its bimodality and the separation between the modes, the distribution is not normal. The mean does not correspond to the mode, which is a key characteristic of a normal distribution.

### Implications

This distribution presents an interesting case for normality testing. Tests that rely solely on skewness and kurtosis may fail to detect the non-normality because these measures align closely with those of a normal distribution. However, tests that consider the overall shape or specific aspects of the distribution may reveal the deviation from normality.

## Understanding Normality Tests

Normality tests are statistical procedures used to determine whether a data set is well-modeled by a normal distribution. They involve formulating a null hypothesis ($H_0$) that the data come from a normal distribution and an alternative hypothesis ($H_1$) that they do not. The tests calculate a test statistic based on the data and compare it to a critical value or use it to compute a p-value, which informs the decision to reject or fail to reject $H_0$.

### The Null Hypothesis and Type II Errors

It's crucial to understand that failing to reject $H_0$ does not confirm that the data are normal. It merely indicates that there is insufficient evidence to conclude that the data are not normal, given the sample size and the test's sensitivity. This situation is known as a Type II error, where the test fails to detect a difference when one actually exists.

### Limitations of Normality Tests

Normality tests have inherent limitations:

1. **Sample Size Sensitivity:** Small sample sizes may not provide enough power to detect deviations from normality, leading to a failure to reject $H_0$ even when the data are not normal.

2. **Test Specificity:** Different tests are sensitive to different aspects of the distribution (e.g., skewness, kurtosis, overall shape). A test may not detect certain types of deviations.

3. **Multiple Testing Issues:** Using multiple normality tests can lead to contradictory results, as each test may respond differently to the data's characteristics.

## Tests Based on Moments: Skewness, Kurtosis, and Jarque-Bera

### Skewness and Kurtosis

- **Skewness** measures the asymmetry of a distribution. A skewness of zero indicates perfect symmetry.

- **Kurtosis** measures the "tailedness" of a distribution. A kurtosis of 3 corresponds to a normal distribution, with higher values indicating heavier tails and lower values indicating lighter tails.

### The Jarque-Bera Test

The **Jarque-Bera test** combines skewness and kurtosis to assess normality. It calculates a test statistic based on the sample skewness ($S$) and kurtosis ($K$):

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
$$

Where $n$ is the sample size.

### Limitations with the Special Distribution

In our special distribution, skewness and kurtosis are approximately equal to those of a normal distribution ($S \approx 0$, $K \approx 3$). Consequently, the Jarque-Bera test may yield a high p-value, leading us to fail to reject $H_0$. The test cannot detect the bimodality because it relies solely on skewness and kurtosis, which do not capture the distribution's multimodal nature.

### Interpretation

While the Jarque-Bera test is useful for detecting deviations in skewness and kurtosis, it is blind to other types of non-normality. In our example, the test's inability to detect bimodality highlights the importance of selecting appropriate tests based on the data's characteristics.

## Geary's Kurtosis and Its Power

### Geary's Kurtosis

**Geary's kurtosis**, also known as **Geary's ratio**, is defined as:

$$
\text{Geary's Ratio} = \frac{\text{Median Absolute Deviation (MAD)}}{\text{Standard Deviation (SD)}}
$$

This ratio compares the median absolute deviation to the standard deviation, providing a measure of kurtosis that is less sensitive to outliers than traditional kurtosis.

### Advantages

- **Robustness:** Geary's kurtosis is less affected by extreme values, making it a robust measure.
  
- **Sensitivity to Shape:** It captures aspects of the distribution's shape that traditional kurtosis may miss.

### Performance on the Special Distribution

Tests based on Geary's kurtosis are powerful in detecting non-normality in distributions like our special case. Despite the traditional kurtosis being normal ($K \approx 3$), Geary's ratio may differ from that expected under normality due to the distribution's bimodality and the "hole" in the middle.

### Implications

The success of Geary's kurtosis-based tests in detecting non-normality underscores the importance of using robust statistical measures. These tests can provide additional insights when traditional moment-based tests fail.

## Tests Based on CDF Comparison

### Overview

Tests that compare the empirical cumulative distribution function (ECDF) of the sample data to the theoretical CDF of a normal distribution can detect differences in the overall shape of the distributions. They consider the cumulative probabilities and assess deviations at all points, rather than focusing on specific moments.

### Kolmogorov-Smirnov Test

The **Kolmogorov-Smirnov (K-S) test** evaluates the maximum absolute difference between the ECDF and the theoretical CDF:

$$
D = \sup_x |F_n(x) - F(x)|
$$

Where $F_n(x)$ is the ECDF and $F(x)$ is the theoretical CDF.

### Limitations

- **Parameter Estimation Bias:** When parameters of the normal distribution (mean and standard deviation) are estimated from the data, the K-S test becomes biased towards failing to reject $H_0$.

### Lilliefors Test

The **Lilliefors test** adjusts the K-S test for cases where parameters are estimated from the data. It provides a corrected critical value or p-value, accounting for the bias.

### Cramér-von Mises Test

The **Cramér-von Mises test** considers the squared differences between the ECDF and the theoretical CDF, integrated over all values:

$$
W^2 = n \int_{-\infty}^{\infty} [F_n(x) - F(x)]^2 dF(x)
$$

### Anderson-Darling Test

The **Anderson-Darling test** improves upon the Cramér-von Mises test by giving more weight to the tails of the distribution:

$$
A^2 = -n - \frac{1}{n} \sum_{i=1}^n \left( (2i - 1) \left[ \ln F(X_i) + \ln(1 - F(X_{n + 1 - i})) \right] \right)
$$

### Performance on the Special Distribution

Tests based on CDF comparisons, particularly the **Anderson-Darling test**, are sensitive to deviations in the entire distribution, including the middle and tails. In our special distribution, the bimodality and the "hole" in the middle result in significant differences between the ECDF and the theoretical CDF of a normal distribution. Consequently, these tests are more likely to detect the non-normality.

### Recommendations

- **Prefer Anderson-Darling Over K-S:** The Anderson-Darling test is generally more powerful than the K-S test, especially for detecting deviations in the tails.
  
- **Use Lilliefors When Parameters Are Estimated:** The Lilliefors test corrects for bias when parameters are estimated from the data.

## Shapiro-Wilk and Shapiro-Francia Tests

### Shapiro-Wilk Test

The **Shapiro-Wilk test** assesses normality by examining the correlation between the data and the corresponding normal scores (expected values under normality). The test statistic $W$ is calculated as:

$$
W = \frac{\left( \sum_{i=1}^n a_i X_{(i)} \right)^2}{\sum_{i=1}^n (X_i - \bar{X})^2}
$$

Where $X_{(i)}$ are the ordered data and $a_i$ are constants derived from the expected values and variances of the order statistics of a normal distribution.

### Limitations

- **Sample Size Constraints:** The Shapiro-Wilk test is most effective for small to moderate sample sizes ($n \leq 2000$). For larger samples, its power may decrease.
  
- **Sensitivity to Symmetry and Tails:** The test is sensitive to deviations in symmetry and tail weight but may not detect bimodality effectively.

### Shapiro-Francia Test

The **Shapiro-Francia test** is a modification of the Shapiro-Wilk test, designed for larger sample sizes. It replaces the variance of the sample with the expected variance under normality.

### Shapiro-Chen Test

The **Shapiro-Chen test** is another variant that adjusts the weighting of the data to improve power against certain alternatives.

### Performance on the Special Distribution

In the case of our special distribution, the **Shapiro-Wilk test** may not be the most effective. Its test statistic relies on the correlation with normal order statistics, which may not sufficiently capture the bimodal nature of the distribution. The **Shapiro-Chen test**, however, might have improved power due to its adjustments.

### Implications

This example illustrates that even widely used tests like Shapiro-Wilk may not always be the best choice. Understanding the specific strengths of each test helps in selecting the most appropriate one for the data at hand.

## Contradictions Between Tests: Understanding the Discrepancies

### Observations

In testing our special distribution, we may encounter significant contradictions:

- Some tests yield high p-values (e.g., Jarque-Bera, failing to reject $H_0$).
- Other tests yield very low p-values (e.g., Anderson-Darling, rejecting $H_0$).

These discrepancies are not marginal; they can be substantial (e.g., $p \approx 1$ vs. $p < 0.001$).

### Reasons for Contradictions

- **Different Sensitivities:** Each test is sensitive to different aspects of the distribution (e.g., skewness, kurtosis, overall shape, tails).
  
- **Test Statistics Based on Different Principles:** Moment-based tests focus on skewness and kurtosis, while CDF-based tests consider the entire distribution.
  
- **Sample Size Effects:** Some tests perform differently depending on the sample size.

### Not an Error

Contradictory results do not indicate errors in the testing process. Instead, they reflect the multifaceted nature of statistical distributions and the varying focus of different tests.

### Choosing the Right Test

- **Define Concerns:** Determine what type of deviation from normality is most relevant for your analysis (e.g., tails, skewness, modality).
  
- **Select Appropriate Tests:** Choose tests that are sensitive to those specific deviations.
  
- **Use Multiple Tests Judiciously:** While using multiple tests can provide a comprehensive assessment, interpret results carefully to avoid confusion.

## Importance of Data Visualization

### Quantile-Quantile (QQ) Plots

A **QQ plot** compares the quantiles of the sample data to the quantiles of a theoretical normal distribution. If the data are normally distributed, the points should fall approximately along a straight line.

### Advantages

- **Visual Detection of Deviations:** QQ plots can reveal deviations from normality, such as skewness, kurtosis, and bimodality.
  
- **Easy Interpretation:** Patterns in the plot can indicate specific types of non-normality.

### Empirical CDF (ECDF) Plots

An **ECDF plot** displays the cumulative probabilities of the sample data. Overlaying the theoretical CDF allows for visual comparison.

### Advantages

- **Highlighting Differences in Distribution Shape:** ECDF plots can show where the sample data deviate from the theoretical distribution, including in the tails and middle.

### Application to the Special Distribution

Visualizing our special distribution using **QQ** and **ECDF** plots would likely reveal the bimodality and the "hole" in the middle. These deviations may not be apparent from statistical tests alone, especially those focusing on moments.

### Recommendations

- **Always Start with Visualization:** Before conducting formal tests, examine the data visually to identify potential issues.
  
- **Complement Tests with Plots:** Use visual tools to support and interpret the results of statistical tests.

## Conclusion

Assessing normality is a nuanced process that requires more than a one-size-fits-all approach. Our exploration of a specially crafted distribution has highlighted several key points:

1. **Normality Tests Cannot Prove Normality:** Failing to reject the null hypothesis does not confirm that data are normally distributed.

2. **Different Tests Have Different Sensitivities:** Understanding what each test measures is crucial. Tests based on moments may miss certain deviations, while CDF-based tests might detect them.

3. **Contradictory Results Are Informative:** Discrepancies between tests are not errors but reflections of the data's complexity. They inform us about different aspects of non-normality.

4. **Visualization Is Essential:** Visual tools like QQ plots and ECDF plots provide invaluable insights that complement formal tests.

5. **Select Tests Based on Specific Concerns:** Choose tests that align with the types of deviations most relevant to your analysis.

6. **Understand the Limitations:** Be aware of sample size effects, test assumptions, and the potential for Type II errors.

In practice, a comprehensive approach that combines statistical tests with visual inspection and a thorough understanding of the data will lead to more robust and reliable conclusions. By appreciating the strengths and limitations of various normality tests, statisticians and data analysts can make informed decisions that enhance the quality of their analyses.

## Appendix: Python Code for Normality Tests

```python
# Import necessary libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the special bimodal distribution
def generate_bimodal_distribution(size=1000):
    mean1, mean2 = 0, 3
    std1, std2 = 1, 0.5
    data1 = np.random.normal(mean1, std1, size // 2)
    data2 = np.random.normal(mean2, std2, size // 2)
    return np.concatenate([data1, data2])

# Generate data
data = generate_bimodal_distribution()

# Plot QQ plot
plt.figure(figsize=(8, 6))
stats.probplot(data, dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()

# Plot empirical CDF
plt.figure(figsize=(8, 6))
sns.ecdfplot(data, label='Empirical CDF')
x = np.linspace(min(data), max(data), 1000)
plt.plot(x, stats.norm.cdf(x, np.mean(data), np.std(data)), label='Theoretical CDF', linestyle='--')
plt.title('Empirical CDF vs Theoretical CDF')
plt.legend()
plt.show()

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(data)
print(f"Shapiro-Wilk Test: W = {shapiro_stat}, p-value = {shapiro_p}")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
print(f"Kolmogorov-Smirnov Test: D = {ks_stat}, p-value = {ks_p}")

# Anderson-Darling test
ad_result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling Test: A² = {ad_result.statistic}, critical values = {ad_result.critical_values}")

# Jarque-Bera test
jb_stat, jb_p = stats.jarque_bera(data)
print(f"Jarque-Bera Test: JB = {jb_stat}, p-value = {jb_p}")

# Geary's Kurtosis (using MAD and Standard Deviation)
mad = np.median(np.abs(data - np.median(data)))
sd = np.std(data)
geary_ratio = mad / sd
print(f"Geary's Kurtosis: {geary_ratio}")
```

# Appendix: R Code for Normality Tests

```r
# Load necessary libraries
library(MASS)
library(nortest)
library(moments)
library(ggplot2)

# Generate the special bimodal distribution
generate_bimodal_distribution <- function(size = 1000) {
  mean1 <- 0
  mean2 <- 3
  std1 <- 1
  std2 <- 0.5
  data1 <- rnorm(size / 2, mean = mean1, sd = std1)
  data2 <- rnorm(size / 2, mean = mean2, sd = std2)
  c(data1, data2)
}

# Generate data
data <- generate_bimodal_distribution()

# QQ Plot
qqnorm(data, main = "QQ Plot")
qqline(data, col = "blue")

# Empirical CDF vs Theoretical CDF
ggplot(data.frame(x = data), aes(x)) +
  stat_ecdf(geom = "step", color = "blue") +
  stat_function(fun = pnorm, args = list(mean = mean(data), sd = sd(data)),
                color = "red", linetype = "dashed") +
  labs(title = "Empirical CDF vs Theoretical CDF")

# Shapiro-Wilk Test
shapiro_test <- shapiro.test(data)
print(paste("Shapiro-Wilk Test: W =", shapiro_test$statistic, ", p-value =", shapiro_test$p.value))

# Kolmogorov-Smirnov Test
ks_test <- ks.test(data, "pnorm", mean(data), sd(data))
print(paste("Kolmogorov-Smirnov Test: D =", ks_test$statistic, ", p-value =", ks_test$p.value))

# Anderson-Darling Test
ad_test <- ad.test(data)
print(paste("Anderson-Darling Test: A² =", ad_test$statistic, ", p-value =", ad_test$p.value))

# Jarque-Bera Test
jb_test <- jarque.test(data)
print(paste("Jarque-Bera Test: JB =", jb_test$statistic, ", p-value =", jb_test$p.value))

# Geary's Kurtosis (using MAD and Standard Deviation)
mad <- mad(data)
sd <- sd(data)
geary_ratio <- mad / sd
print(paste("Geary's Kurtosis: ", geary_ratio))
```

# Appendix: Ruby Code for Normality Tests

```ruby
# Load necessary libraries
require 'distribution'
require 'gnuplotrb'
include GnuplotRB

# Generate the special bimodal distribution
def generate_bimodal_distribution(size = 1000)
  mean1, mean2 = 0, 3
  std1, std2 = 1, 0.5
  data1 = Array.new(size / 2) { Distribution::Normal.rng(mean1, std1).call }
  data2 = Array.new(size / 2) { Distribution::Normal.rng(mean2, std2).call }
  data1 + data2
end

# Generate data
data = generate_bimodal_distribution

# QQ plot (using Gnuplot)
x = Distribution::Normal.rng(0, 1).call
qq_plot = Plot.new([x.sort, data.sort], with: 'points', title: 'QQ Plot', style: 'points')
qq_plot.to_png('qq_plot.png')

# Empirical CDF vs Theoretical CDF (using Gnuplot)
sorted_data = data.sort
ecdf = sorted_data.each_with_index.map { |val, i| [val, (i + 1).to_f / sorted_data.size] }
cdf_plot = Plot.new(
  [ecdf, with: 'lines', title: 'Empirical CDF'],
  [sorted_data, sorted_data.map { |x| Distribution::Normal.cdf(x, data.mean, data.standard_deviation) },
  with: 'lines', title: 'Theoretical CDF', style: 'dashed']
)
cdf_plot.to_png('cdf_plot.png')

# Shapiro-Wilk Test (using R integration through RinRuby)
require 'rinruby'

R.eval <<-EOF
  shapiro_test <- shapiro.test(c(#{data.join(',')}))
  shapiro_stat <- shapiro_test$statistic
  shapiro_p_value <- shapiro_test$p.value
EOF

puts "Shapiro-Wilk Test: W = #{R.shapiro_stat}, p-value = #{R.shapiro_p_value}"

# Kolmogorov-Smirnov Test
ks_test = Distribution::Normal.kstest(data)
puts "Kolmogorov-Smirnov Test: D = #{ks_test[:statistic]}, p-value = #{ks_test[:p_value]}"

# Anderson-Darling Test (using R integration)
R.eval <<-EOF
  library(nortest)
  ad_test <- ad.test(c(#{data.join(',')}))
  ad_stat <- ad_test$statistic
  ad_p_value <- ad_test$p.value
EOF

puts "Anderson-Darling Test: A² = #{R.ad_stat}, p-value = #{R.ad_p_value}"

# Jarque-Bera Test
jb_test = Distribution::Normal.jarque_bera(data)
puts "Jarque-Bera Test: JB = #{jb_test[:statistic]}, p-value = #{jb_test[:p_value]}"

# Geary's Kurtosis (using MAD and Standard Deviation)
mad = data.map { |x| (x - data.median).abs }.median
sd = Math.sqrt(data.map { |x| (x - data.mean) ** 2 }.sum / data.size)
geary_ratio = mad / sd
puts "Geary's Kurtosis: #{geary_ratio}"
```
