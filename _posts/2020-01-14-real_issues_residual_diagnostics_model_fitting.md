---
author_profile: false
categories:
- Data Science
- Statistics
classes: wide
date: '2020-01-14'
excerpt: Residual diagnostics often trigger debates, especially when tests like Shapiro-Wilk suggest non-normality. But should it be the final verdict on your model? Let's dive deeper into residual analysis, focusing on its impact in GLS, mixed models, and robust alternatives.
header:
  image: /assets/images/data_science_13.jpg
  og_image: /assets/images/data_science_13.jpg
  overlay_image: /assets/images/data_science_13.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_13.jpg
  twitter_image: /assets/images/data_science_13.jpg
keywords:
- Residual Diagnostics
- Shapiro-Wilk Test
- Generalized Least Squares
- Mixed Models
- Statistical Modeling
seo_description: An in-depth exploration of the limitations of Shapiro-Wilk and the real issues to consider in residual diagnostics when fitting models. Focusing on Generalized Least Squares and robust alternatives, this article provides insight into the complexities of longitudinal data analysis.
seo_title: 'Residual Diagnostics: Beyond the Shapiro-Wilk Test in Model Fitting'
seo_type: article
summary: In this article, we examine why the Shapiro-Wilk test should not be the final say in assessing model fit, particularly in complex models like Generalized Least Squares for longitudinal data. Instead, we explore alternative diagnostics, the role of kurtosis, skewness, and the practical impact of non-normality on parameter estimates.
tags:
- Residual Analysis
- Longitudinal Data
- Generalized Least Squares
- Parametric Models
title: 'Don''t Get MAD About Shapiro-Wilk: Real Issues in Residual Diagnostics and Model Fitting'
---

When fitting models, especially in longitudinal studies, residual diagnostics often become a contentious part of the statistical review process. It's not uncommon for a reviewer to wave the **Shapiro-Wilk test** in your face, claiming that the residuals' departure from normality invalidates your entire parametric model. But is this rigid adherence to normality testing warranted? 

Today, I'm going to walk you through a discussion I had with a statistical reviewer while analyzing data from a longitudinal study using a **Mixed-Model Repeated Measures** (MMRM) approach. We’ll examine why over-reliance on the **Shapiro-Wilk test** is misguided and how real-world data almost never meets theoretical assumptions perfectly. And more importantly, I’ll explain why **other diagnostic tools** and practical considerations should play a bigger role in determining whether your model is valid.

## The Problem with Over-Reliance on the Shapiro-Wilk Test

First, let’s talk about **Shapiro-Wilk**. It’s a test that measures the goodness-of-fit between your residuals and a normal distribution. When the p-value is below a certain threshold (usually 0.05), many take it as definitive evidence that the residuals are not normally distributed and, therefore, the model assumptions are violated. But here's the catch: this test becomes overly sensitive when sample sizes are large.

For instance, with **N ~ 360 observations**, the Shapiro-Wilk test will pick up **even the smallest deviations** from normality. This means that, although your data might not be perfectly normal (and in practice, it never is), it may still be **close enough** that the deviation has no practical effect on the validity of your model. Let’s not forget that **statistical models** are tools for approximation—not exact replicas of reality.

In my experience, using the Shapiro-Wilk test as a **litmus test for model validity** can be overly rigid and misguided. When my reviewer argued that the p-value for the Shapiro-Wilk test was less than 0.001, they essentially viewed this as grounds to dismiss the entire parametric model. However, I knew that other aspects of residual diagnostics—like **skewness**, **kurtosis**, and visual inspections (like **QQ plots**)—were far more indicative of the model’s practical robustness.

### Sample Size Sensitivity

Shapiro-Wilk is notorious for being **oversensitive** with large datasets. The irony is that, as your data size grows, this test is likely to reject normality due to minuscule deviations from the theoretical distribution. So, if you’re analyzing hundreds of data points, should you really be worried about a slight p-value drop below 0.05? Most likely not.

In my case, with **N = 360** residuals, the histogram of residuals overlapped almost perfectly with the normal curve. The **skewness** was practically zero, and while there was some **kurtosis** (~5.5 vs. the ideal of 3), it wasn’t extreme. A simple QQ plot showed only minor deviations in the tails, but the theoretical and empirical quantiles largely matched. Despite this, my reviewer was adamant that these results violated formal assumptions.

## Understanding Residual Diagnostics: More than Just Normality

The point I emphasized during this discussion was that **Shapiro-Wilk should not be the be-all and end-all** of model diagnostics. Residual analysis is about understanding the **behavior** of your data in relation to the assumptions of the model and ensuring that any deviations are not **practically significant**. Here are some of the diagnostic tools and metrics that can provide a clearer picture of what’s happening under the hood of your model:

### 1. **Skewness**: A Measure of Symmetry

One of the first checks I perform after running a model is to look at the **skewness** of the residuals. Skewness measures the asymmetry of the distribution of residuals. In an ideal world, residuals should have a skewness of zero, indicating a perfectly symmetrical distribution.

In the case of my longitudinal data, the skewness was around **0.05**, which is essentially **perfectly symmetrical** for practical purposes. A skewness value close to zero means there’s no need to worry about large asymmetries that could bias the results.

### 2. **Kurtosis**: Understanding Fat Tails

**Kurtosis** is another essential metric that often gets overlooked in favor of the Shapiro-Wilk test. Kurtosis tells you about the **heaviness of the tails** in the residuals' distribution. The normal distribution has a kurtosis of 3. If your residuals have a kurtosis higher than this, it indicates that the tails are fatter than those of a normal distribution, potentially signaling **outliers** or **extreme values**.

In my case, the kurtosis was around **5.5**—slightly above the ideal 3, but nowhere near the threshold where it would be a red flag (usually a kurtosis of **10+**). The small excess kurtosis here was not indicative of any serious issue.

### 3. **QQ Plots**: Visualizing Deviations from Normality

**QQ plots** (Quantile-Quantile plots) are another indispensable tool for diagnosing residuals. They plot the **empirical quantiles** of the residuals against the **theoretical quantiles** of a normal distribution. If the points fall along a straight line, the residuals are normally distributed.

In the conversation with my reviewer, the QQ plot showed minor deviations in the tails, but the **axes** made the deviations look far more dramatic than they actually were. In fact, apart from a few outliers, the theoretical and empirical quantiles were almost identical.

This is where the **practical significance** comes into play. Yes, there was a slight deviation from normality, but it was minor enough that it didn’t have a substantial impact on the **parameter estimates** of the model.

## Robustness Checks: Going Beyond Normality Assumptions

When fitting models—especially complex ones like **Mixed-Model Repeated Measures** (MMRM)—it’s often helpful to run **robustness checks** to see how much the residual distribution impacts your final results. In my case, I re-fitted the model using a **robust mixed-effects model** with **Huberized errors** (a method for reducing the influence of outliers by down-weighting them). This robust model essentially smooths out the impact of deviations in the residuals.

The result? The **parameter estimates** were nearly identical to those from the original parametric model, indicating that any deviation from normality had **little to no impact** on the overall conclusions of the model.

### Sensitivity Analysis: Non-Parametric Approaches

Another key part of the discussion involved conducting a **sensitivity analysis** using non-parametric methods to validate the parametric model’s results. I ran a **permutation paired t-test** (a non-parametric approach) and used **Generalized Estimating Equations** (GEE), which makes no assumptions about the normality of the residuals. Once again, the estimates were consistent across both parametric and non-parametric models, confirming that the original parametric approach was robust.

The **Shapiro-Wilk p-value** did not alter the **practical conclusions** of the study. In fact, the model produced **accurate and reliable results**, despite minor deviations from normality. 

## The Real Issue: Are the Estimates Reliable?

Here’s the heart of the matter: the **real issue** with residual diagnostics isn’t whether the p-value from Shapiro-Wilk is below 0.05 or if the QQ plot deviates slightly from a straight line. The real issue is whether these deviations have a **practical impact** on your parameter estimates and conclusions. 

In many cases, small deviations from normality will have **no meaningful effect** on your estimates. However, overly relying on strict statistical rules without understanding the **underlying behavior** of your model can lead to **overcorrection** and the use of inappropriate methods.

### Random Slopes and Residual Diagnostics

Another important issue that came up in the discussion was the use of **random slopes** in mixed models. In longitudinal studies, it’s common to include **random intercepts** and **random slopes** to account for the variation across individual subjects over time. However, in this particular study, I had difficulty getting the model to converge when adding random slopes.

Rather than forcing a **random slopes model** and risking **model convergence issues**, I opted for a **random intercept model**. Even though my reviewer initially criticized this choice, I showed that the estimates were practically identical to those from the more complex model (when it did converge). This brings us back to the main point: **practical validity** trumps the pursuit of perfect assumptions.

## Why the Shapiro-Wilk Test Alone Is Not Enough

The takeaway is this: **Shapiro-Wilk** is just one of many tools in the diagnostic toolbox. It’s not sufficient to look at a p-value below 0.05 and conclude that the model is flawed. Real data rarely conforms to perfect normality, and in most cases, **slight deviations from normality are inconsequential**. What’s more important is to assess the overall **robustness** of the model through **multiple diagnostic methods**:

- **Skewness** and **kurtosis** provide more nuanced insights into the distribution of residuals.
- **QQ plots** visually depict the nature of any deviations from normality.
- **Robust models** (such as Huberized models or GEE) allow you to test whether any deviation has a substantial impact on your estimates.
- **Sensitivity analyses** using non-parametric methods can confirm the stability of your results.

### When Normality Really Matters

That said, there are cases where normality really does matter—especially in small-sample studies or when extreme outliers are present. In these cases, deviations from normality can bias the results and lead to **misleading conclusions**. But in studies with larger samples or only slight deviations from normality, the impact on estimates is often minimal.

## The Role of Practicality in Statistical Modeling

Statistical models are ultimately **practical tools**—they’re designed to help us **approximate reality** and make informed decisions. They’re not meant to perfectly fit every theoretical assumption. When working with real-world data, the key is to strike a balance between meeting model assumptions and producing valid, interpretable results.

**Don’t get MAD** (Mean Absolute Deviation, for the pun-inclined) about Shapiro-Wilk when it flags deviations from normality. Look at the **broader picture**: how do your residuals behave? Are there any **outliers** or **heavy tails** that could distort your results? Is your model robust to minor deviations from assumptions?

By understanding these nuances, you can make informed decisions that go beyond mechanistic rules and focus on what really matters: the **interpretation** and **practical significance** of your findings.
