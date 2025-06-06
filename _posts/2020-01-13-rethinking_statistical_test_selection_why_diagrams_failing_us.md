---
author_profile: false
categories:
- Data Science
- Statistics
classes: wide
date: '2020-01-13'
excerpt: Most diagrams for choosing statistical tests miss the bigger picture. Here's a bold, practical approach that emphasizes interpretation over mechanistic rules, and cuts through statistical misconceptions like the N>30 rule.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Statistical Tests
- Welch t-test
- Data Science
- Hypothesis Testing
- Nonparametric Tests
seo_description: A bold take on statistical test selection that challenges common frameworks. Move beyond basic diagrams and N>30 pseudorules, and learn how to focus on meaningful interpretation and robust testing strategies.
seo_title: 'Rethinking Statistical Test Selection: A Bold Approach to Choosing Tests'
seo_type: article
summary: This article critiques popular frameworks for selecting statistical tests, offering a robust, more flexible alternative that emphasizes interpretation and realistic outcomes over pseudorules and data transformations. Learn why techniques like Welch’s t-test and permutation tests are better than many 'classics'.
tags:
- Statistical Analysis
- Data Science
- Testing Frameworks
- Welch Test
title: 'Rethinking Statistical Test Selection: Why the Diagrams Are Failing Us'
---

There are over **850** recognized statistical tests, and that number continues to grow. Yet, most diagrams and frameworks on how to choose a statistical test only scratch the surface, covering a narrow subset of options. Worse, many promote dangerous practices, like arbitrary data transformations or shallow rules like “N > 30” as if they are the ultimate truth.

This article is a **bold rethinking** of how we approach statistical test selection. I don’t follow the conventional flowcharts, and neither should you. We’ll dive into real-world approaches for comparing means, medians, and other data characteristics while respecting the integrity of the data. We’ll also explore why some traditional tests like the **t-test**, **Kruskal-Wallis**, and **Friedman test** are either obsolete or too limited for most modern applications. Instead, we’ll consider better alternatives like the **Welch t-test**, **ART-ANOVA**, and **permutation testing**, among others.

If you’re tired of the typical diagrams, pseudorules, and one-size-fits-all approaches, this article is for you. Let’s focus on practical methods that get to the core of understanding and interpreting data, not just blindly following the steps dictated by a formulaic chart.

## Why Most Statistical Diagrams Miss the Point

In every LinkedIn post, blog, or webinar about statistics, you’ll likely come across a diagram telling you which statistical test to use based on a few factors: **data type** (e.g., categorical vs. continuous), sample size, and whether your data is normally distributed. These flowcharts are popular, and they do serve as a useful starting point for newcomers to data science. But there’s a significant flaw: **they stop at the mechanics**, treating statistical tests as mechanistic processes that ignore the broader context of **interpretation**.

### Pseudorules Like “N > 30”

Take, for example, the rule “N > 30,” which claims that sample sizes greater than 30 allow for the use of parametric tests under the Central Limit Theorem. This is a **gross oversimplification**. Whether you have 25 or 100 data points, **assumptions about variance**, **normality**, and **independence** still need to be considered carefully. It’s not just about the number of data points; it’s about whether those data points are **representative** and **well-behaved** in the context of your study.

### Dangerous Data Transformations

Another common recommendation in these diagrams is to **transform the data** to meet the assumptions of parametric tests (e.g., log-transforming skewed data). But transforming data to fit a model often **distorts the interpretation** of results. If you have to twist your data into unnatural shapes to use a particular test, **maybe you’re using the wrong test** in the first place. Why not use a test that respects the data’s original structure?

I’m a firm believer that **tests should fit the data**, not the other way around. Instead of transforming the raw data, we can use methods that are more **robust** and **adaptive**, while still providing interpretable results.

## My Approach: Focus on Meaningful Comparisons

Here’s a breakdown of how I approach statistical test selection. Instead of relying on generic rules, I focus on these core tasks:

1. **Comparison of Conditional Means**: Either raw or link-transformed (logistic or Poisson link functions), but never transforming raw data.
2. **Comparison of Medians**: Particularly when the mean isn’t representative due to skewed distributions.
3. **Comparison of Other Aspects**: Like stochastic ordering, which is typically assessed through **rank-based tests** like **Mann-Whitney** and **Kruskal-Wallis**.
4. **Tests of Binary Data and Rates**: This includes binary outcome data (e.g., logistic regression) and counts or rates (e.g., Poisson models, survival analysis).

Let’s explore these categories in more detail and discuss which tests to use and why.

### 1. Comparison of Conditional Means: Raw or Link-Transformed (But Not the Data)

One of the most frequent tasks in data analysis is comparing means. This is where many fall into the trap of overusing the **t-test**. While the **t-test** is widely known, it’s limited by its assumption of equal variances across groups, which is almost never the case in real-world data.

#### Why the Welch t-test Should Be Your Default

When comparing the means of two groups, I recommend using the **Welch t-test** instead of the traditional t-test. The Welch t-test does not assume equal variances between groups, making it far more flexible. It should be your default whenever you’re comparing two means because, unlike the t-test, it’s robust to **heteroscedasticity** (unequal variances).

For example, let’s say you’re comparing the average customer satisfaction scores from two different user groups (e.g., users who received a new feature vs. those who did not). If these two groups have different variances (which is often the case in behavioral data), the Welch t-test will provide a more accurate picture of the differences between group means.

#### When to Use Link-Transformed Means

In cases where you’re dealing with **non-normal** data, or when your outcome is a rate or binary variable, you can apply **link functions** to the mean. For example, use the **log link** for count data (Poisson regression) or the **logit link** for binary data (logistic regression). These methods preserve the raw structure of the data while allowing you to model the relationship in a way that fits the data’s characteristics.

But note: this **doesn’t involve transforming the raw data** itself. Instead, the model applies a transformation to the mean or outcome, ensuring that interpretation remains clear.

### 2. Comparisons of Medians: When the Mean Won’t Do

There are many situations where the **mean** is not a reliable measure of central tendency—especially when the data is heavily skewed. In such cases, you’ll want to compare **medians** instead. For example, income data is typically skewed, with a few individuals earning much higher than the rest. The **median** provides a more accurate reflection of central tendency.

#### What About the Mann-Whitney Test?

The **Mann-Whitney test** (often called the **Wilcoxon rank-sum test**) is commonly used to compare the medians of two independent groups. But here's the catch—Mann-Whitney doesn't **strictly** compare medians. It tests whether one group tends to have larger values than the other, which can be interpreted as a form of **stochastic dominance**.

If you want a pure comparison of medians and are not interested in the entire distribution, there are alternatives like **quantile regression** that allow for more direct interpretation of median differences across groups.

### 3. Comparisons of Other Aspects: Beyond Means and Medians

In some cases, you’ll want to compare aspects of the distribution beyond the central tendency, such as the **ordering of values** across groups. For these tasks, rank-based tests like **Mann-Whitney** and **Kruskal-Wallis** are useful, but they have limitations that are often glossed over in flowcharts.

#### Kruskal-Wallis and Its Limits

The **Kruskal-Wallis test** is a nonparametric method for comparing medians across multiple groups, but its weakness is that it’s limited to **one categorical predictor**. In modern applications, where we often need to account for **multiple predictors**, **interactions**, or **repeated measures**, Kruskal-Wallis is simply too limited.

For more complex designs, you can use **ART-ANOVA** (Aligned Rank Transform ANOVA), **ATS** (Analysis of Treatments), or **WTS** (Wald-Type Statistics), all of which allow for greater flexibility in handling interactions and repeated measures. These techniques enhance the traditional Kruskal-Wallis framework by extending it to real-world data complexities.

### 4. Tests for Binary Data and Rates

When you’re dealing with **binary outcomes** (e.g., success/failure, alive/dead), traditional parametric tests like the **z-test** often show up in diagrams. But in real-world applications, these tests are limited in scope and are rarely the best choice.

#### Logistic Regression for Binary Data

For binary data, **logistic regression** is a far more robust option than the **z-test**. It allows you to model the probability of a binary outcome based on one or more predictors, giving you insights into how each variable affects the likelihood of success.

#### Count Data and Rates: Poisson and Beyond

For **count data** or **rate data** (e.g., number of occurrences per unit time), you can use **Poisson regression**. But be cautious—Poisson regression assumes that the mean and variance are equal, which is often not the case in real-world data. For overdispersed count data, you might want to use **Negative Binomial Regression**, which relaxes the equal-variance assumption and provides more accurate estimates.

### Survival Analysis and Binary Data Over Time

For time-to-event (survival) data, traditional approaches like the **Kaplan-Meier estimator** and the **log-rank test** are common but limited. A more powerful approach is to use **Cox proportional hazards regression**, which models the time to an event while accounting for various predictors, giving you a nuanced view of factors affecting survival times.

## Why I Avoid Some Popular Tests

I’ve covered some of the methods I frequently use, but it’s also important to explain why I avoid certain tests that are widely recommended in statistical diagrams.

### 1. The t-test

Let’s be honest—the **t-test** is overhyped. It’s limited to situations where variances are equal across groups, and as we’ve discussed, that’s rarely the case in real-world data. If you’re still using the t-test, it’s time to upgrade to **Welch’s t-test**, which is more robust and doesn’t make such restrictive assumptions about variance equality.

### 2. Kruskal-Wallis Test

As mentioned, the **Kruskal-Wallis test** is too limited for modern data analysis, especially when dealing with multiple groups or interactions. In most cases, it’s better to use alternatives like **ART-ANOVA** or **WTS**.

### 3. Friedman Test

The **Friedman test** is another nonparametric test often used for repeated measures. However, it’s limited in its ability to handle complex designs, such as interactions or multiple predictors. A more flexible approach is to use **ART-ANOVA**, which can handle these complexities with ease.

### 4. The z-test

The **z-test** is outdated and rarely useful in real-world data scenarios. Logistic regression or permutation testing are far better alternatives for binary data.

## A Word on Resampling Methods: Permutation vs. Bootstrap

Finally, I want to touch on **resampling methods**, which are often used when data doesn’t meet traditional parametric assumptions. You’ll often see **bootstrap tests** recommended in diagrams, but I prefer **permutation tests**.

Here’s why: **Permutation testing** naturally performs under the true null hypothesis by repeatedly shuffling data labels and recalculating the test statistic. This preserves the structure of the data and avoids some of the pitfalls of bootstrap testing, which requires assumptions about the null distribution. If you’re running an experiment and want a robust, nonparametric test, go with permutation testing.

## Break Free from the Diagrams

If you’ve been relying on the same diagrams and pseudorules for choosing statistical tests, it’s time to rethink your approach. These flowcharts may be a decent introduction, but they often ignore the complexities of real-world data. By focusing on meaningful interpretations, using robust methods like **Welch’s t-test**, and avoiding unnecessary data transformations, you can make better decisions and gain deeper insights from your data.

Remember, statistical tests are tools—not laws to be followed blindly. The real power lies in understanding what your data is telling you and choosing methods that respect its structure without distorting the interpretation.
