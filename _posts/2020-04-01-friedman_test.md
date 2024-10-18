---
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Repeated measures anova
- Non-parametric test
- Friedman test
- Ordinal data
seo_description: Learn about the Friedman test, its application as a non-parametric
  alternative to repeated measures ANOVA, and its use with ordinal data or non-normal
  distributions.
seo_title: 'The Friedman Test: A Non-Parametric Alternative to Repeated Measures ANOVA'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Non-parametric tests
- Repeated measures anova
- Friedman test
- Ordinal data
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
---

In data analysis, we often encounter situations where we need to compare three or more related groups. When the assumptions of normality or homogeneity of variances are not met, using parametric methods such as repeated measures ANOVA may not be appropriate. In such cases, the **Friedman test** offers a robust **non-parametric alternative**.

The Friedman test is particularly useful for analyzing **ordinal data** or **non-normal distributions** in repeated measures designs, where the same subjects are measured under different conditions or across different time points. This article will provide a detailed explanation of the Friedman test, its application, and practical examples to help you understand when and how to use this method in your analyses.

---
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Repeated measures anova
- Non-parametric test
- Friedman test
- Ordinal data
seo_description: Learn about the Friedman test, its application as a non-parametric
  alternative to repeated measures ANOVA, and its use with ordinal data or non-normal
  distributions.
seo_title: 'The Friedman Test: A Non-Parametric Alternative to Repeated Measures ANOVA'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Non-parametric tests
- Repeated measures anova
- Friedman test
- Ordinal data
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
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
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Repeated measures anova
- Non-parametric test
- Friedman test
- Ordinal data
seo_description: Learn about the Friedman test, its application as a non-parametric
  alternative to repeated measures ANOVA, and its use with ordinal data or non-normal
  distributions.
seo_title: 'The Friedman Test: A Non-Parametric Alternative to Repeated Measures ANOVA'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Non-parametric tests
- Repeated measures anova
- Friedman test
- Ordinal data
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
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
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2020-04-01'
excerpt: The Friedman test is a non-parametric alternative to repeated measures ANOVA,
  designed for use with ordinal data or non-normal distributions. Learn how and when
  to use it in your analyses.
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Repeated measures anova
- Non-parametric test
- Friedman test
- Ordinal data
seo_description: Learn about the Friedman test, its application as a non-parametric
  alternative to repeated measures ANOVA, and its use with ordinal data or non-normal
  distributions.
seo_title: 'The Friedman Test: A Non-Parametric Alternative to Repeated Measures ANOVA'
seo_type: article
summary: This article provides an in-depth explanation of the Friedman test, including
  its use as a non-parametric alternative to repeated measures ANOVA, when to use
  it, and practical examples in ranking data and repeated measurements.
tags:
- Non-parametric tests
- Repeated measures anova
- Friedman test
- Ordinal data
title: 'The Friedman Test: Non-Parametric Alternative to Repeated Measures ANOVA'
---

## Conclusion

The **Friedman test** is a valuable tool for analyzing **non-parametric data** in repeated measures designs. It provides a robust alternative to repeated measures ANOVA when the assumptions of normality or equal variances are not met. By comparing the ranks of data within subjects across different conditions, the Friedman test can identify whether significant differences exist between groups.

This test is particularly useful in situations where ordinal data, non-normal distributions, or small sample sizes make parametric methods inappropriate. Whether you’re comparing patient responses to different treatments over time or analyzing ranking data from surveys, the Friedman test is a flexible and reliable option.

### Further Reading

- **"Nonparametric Statistical Methods"** by Myles Hollander and Douglas A. Wolfe – A comprehensive resource on non-parametric methods, including the Friedman test.
- **"Practical Statistics for Medical Research"** by Douglas G. Altman – Offers practical guidance on using the Friedman test and other statistical methods in medical research.
- **Online Statistical Resources**: Many online tutorials and statistical software packages, like R or Python’s SciPy library, offer implementations of the Friedman test for practical use.

---
