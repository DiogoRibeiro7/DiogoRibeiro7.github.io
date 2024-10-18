---
author_profile: false
categories:
- Data Analysis
classes: wide
date: '2023-11-15'
excerpt: Learn the differences between biserial and point-biserial correlation methods,
  and discover how they can be applied to analyze relationships between continuous
  and binary variables in educational testing, psychology, and medical diagnostics.
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_9.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_9.jpg
keywords:
- Biserial correlation
- Point-biserial correlation
- Educational testing
- Psychology
- Medical diagnostics
seo_description: Explore biserial and point-biserial correlation methods for analyzing
  relationships between continuous and binary variables, with applications in educational
  testing, psychology, and medical diagnostics.
seo_title: 'Biserial vs. Point-Biserial Correlation: Analyzing Continuous and Binary
  Variable Relationships'
seo_type: article
summary: Biserial and point-biserial correlation methods are used to analyze relationships
  between binary and continuous variables. This article explains the differences between
  these two correlation techniques and their practical applications in fields like
  educational testing, psychology, and medical diagnostics.
tags:
- Biserial correlation
- Point-biserial correlation
- Binary variables
- Continuous variables
- Educational testing
- Psychology
- Medical diagnostics
title: 'Biserial and Point-Biserial Correlation: Analyzing the Relationship Between
  Continuous and Binary Variables'
---

In statistical analysis, understanding the relationship between variables is essential for gaining insights and making informed decisions. When analyzing the relationship between **continuous** and **binary** variables, two specialized correlation methods are often employed: **biserial correlation** and **point-biserial correlation**. Both techniques are used to measure the strength and direction of association between these two types of variables, but they are applied in different contexts and are based on distinct assumptions.

In this article, we will explain the fundamental differences between biserial and point-biserial correlation, explore their mathematical formulations, and discuss their practical applications in various fields, including **educational testing**, **psychology**, and **medical diagnostics**.

## 1. Understanding Biserial and Point-Biserial Correlation

Before diving into the mathematical details, it is important to understand the distinction between **biserial correlation** and **point-biserial correlation** and when each method is used.

### 1.1 Biserial Correlation

**Biserial correlation** is a statistical method used when you are interested in measuring the relationship between a continuous variable and a binary variable that represents an underlying **latent continuous variable**. This means that the binary variable is an approximation or a discretization of a continuous variable that has been artificially divided into two categories.

For example:

- A student’s test performance (continuous variable: score) and their **pass/fail status** (binary variable) could be analyzed using biserial correlation, as the binary variable (pass/fail) represents an underlying latent continuous distribution of scores.

The **biserial correlation coefficient** assumes that the binary variable reflects an underlying normally distributed variable, and it attempts to recover this relationship to provide a more accurate estimate of the correlation.

### 1.2 Point-Biserial Correlation

The **point-biserial correlation** is a special case of the Pearson correlation coefficient, specifically used to measure the relationship between a **continuous variable** and a **dichotomous binary variable** (where the binary variable is truly categorical and not a representation of a continuous latent variable). The point-biserial correlation is applied when the binary variable does not arise from an underlying continuous distribution.

For example:

- **Gender** (binary variable: male/female) and **height** (continuous variable) could be analyzed using point-biserial correlation, as gender is truly categorical, and there is no underlying continuous variable.

The point-biserial correlation is mathematically equivalent to Pearson’s correlation when one variable is continuous and the other is dichotomous, making it straightforward to compute.

## 2. Mathematical Formulation of Biserial and Point-Biserial Correlation

Both biserial and point-biserial correlation coefficients aim to measure the strength and direction of the relationship between a continuous variable and a binary variable. However, the calculation of these two coefficients differs based on the assumptions about the binary variable.

### 2.1 Biserial Correlation Formula

The **biserial correlation coefficient ($$r_b$$)** is calculated as:

$$
r_b = \frac{\bar{X_1} - \bar{X_0}}{s} \cdot \frac{p_1 \cdot p_0}{\phi(z)}
$$

Where:

- $$\bar{X_1}$$ and $$\bar{X_0}$$ are the means of the continuous variable for the two binary groups.
- $$s$$ is the standard deviation of the continuous variable.
- $$p_1$$ and $$p_0$$ are the proportions of observations in the two binary categories.
- $$\phi(z)$$ is the height of the standard normal probability density function at the point $$z$$, which is the point on the continuous latent variable that corresponds to the threshold used to create the binary categories.

The **biserial correlation** adjusts for the fact that the binary variable represents a discretized version of a continuous variable, making it appropriate for use in cases where the binary variable reflects an underlying continuous trait.

### 2.2 Point-Biserial Correlation Formula

The **point-biserial correlation coefficient ($$r_{pb}$$)** is computed using the standard Pearson correlation formula, adapted for one continuous and one binary variable:

$$
r_{pb} = \frac{\bar{X_1} - \bar{X_0}}{s} \cdot \sqrt{\frac{p_1 \cdot p_0}{n}}
$$

Where:

- $$\bar{X_1}$$ and $$\bar{X_0}$$ are the means of the continuous variable for the two binary groups.
- $$s$$ is the standard deviation of the continuous variable.
- $$p_1$$ and $$p_0$$ are the proportions of the binary groups.
- $$n$$ is the total number of observations.

The **point-biserial correlation** does not assume an underlying continuous distribution for the binary variable. It is simply a measure of the difference in the continuous variable’s means between the two groups, standardized by the standard deviation and weighted by the proportions of the groups.

## 3. Practical Applications of Biserial and Point-Biserial Correlation

Biserial and point-biserial correlations have important applications in fields where researchers need to understand how a binary classification variable relates to a continuous outcome. These methods are particularly useful in **educational testing**, **psychology**, and **medical diagnostics**.

### 3.1 Educational Testing

In educational testing, both biserial and point-biserial correlations are widely used to assess the relationship between test scores (continuous variable) and categorical outcomes (binary variable). These correlations are crucial in test item analysis, where educators and psychometricians aim to evaluate the quality of test items and their relationship with overall performance.

#### 3.1.1 Biserial Correlation in Test Item Analysis

**Biserial correlation** is often used in item analysis to examine the relationship between students' total test scores (continuous variable) and their performance on individual test items (binary variable: correct/incorrect). This is because the binary outcome (correct/incorrect) reflects an underlying continuous distribution of ability.

For example:

- When analyzing whether a specific test question effectively differentiates between high- and low-performing students, the biserial correlation measures how well performance on the question correlates with the overall test score, providing insight into the quality of the test item.

#### 3.1.2 Point-Biserial Correlation in Test Reliability

The **point-biserial correlation** can also be used in educational testing, particularly to assess the relationship between a binary variable (e.g., **pass/fail**) and a continuous variable (e.g., **test score**). This allows researchers to determine how strongly overall test scores relate to categorical classifications, such as passing a grade level or failing an exam.

### 3.2 Psychology

In psychology, researchers often use biserial and point-biserial correlation to analyze relationships between psychological traits or behaviors (continuous variables) and categorical groupings (binary variables). These methods provide insights into how categorical factors, such as diagnostic status, relate to psychological measures like anxiety or cognitive performance.

#### 3.2.1 Biserial Correlation in Cognitive Testing

For example, **biserial correlation** may be used to examine the relationship between cognitive performance (e.g., **IQ score**) and a binary classification such as **presence or absence of learning disabilities**. Here, the binary variable (learning disability) is seen as representing an underlying continuous distribution of cognitive ability.

#### 3.2.2 Point-Biserial Correlation in Personality Research

In personality research, **point-biserial correlation** may be applied when comparing continuous psychological measures, such as **levels of neuroticism**, with a truly binary variable like **gender**. Since gender is not viewed as a latent continuous variable, point-biserial correlation is more appropriate in this context.

### 3.3 Medical Diagnostics

In medical diagnostics, biserial and point-biserial correlations are used to evaluate the relationship between **diagnostic test results** (continuous variables) and **binary health outcomes** (e.g., presence or absence of a disease). These correlations help medical researchers assess the effectiveness of diagnostic tools and predict patient outcomes.

#### 3.3.1 Biserial Correlation in Diagnostic Testing

For instance, **biserial correlation** may be used to explore the relationship between **blood pressure measurements** (continuous variable) and a **binary health outcome** such as **hypertension diagnosis** (yes/no). Since hypertension can be thought of as the result of an underlying continuous distribution of blood pressure values, biserial correlation is appropriate for analyzing this relationship.

#### 3.3.2 Point-Biserial Correlation in Treatment Efficacy

In medical studies, **point-biserial correlation** can be employed to examine how continuous measures (e.g., **tumor size reduction**) relate to binary outcomes such as **treatment success/failure**. By quantifying the correlation between a continuous treatment effect and a categorical classification, point-biserial correlation helps assess treatment efficacy.

## 4. Choosing Between Biserial and Point-Biserial Correlation

Choosing between biserial and point-biserial correlation depends on how the binary variable is conceptualized in the analysis:

- **Use biserial correlation** when the binary variable represents a **discretized version of an underlying continuous variable**. This is common in educational testing and some diagnostic contexts where the binary variable (e.g., correct/incorrect or pass/fail) reflects an underlying continuous process (e.g., ability or health condition severity).
  
- **Use point-biserial correlation** when the binary variable is truly categorical with **no underlying continuous distribution**. Examples include gender, treatment success/failure, or any other naturally dichotomous variable that does not reflect a latent continuous trait.

## 5. Conclusion

Both **biserial** and **point-biserial correlation** methods are valuable tools for analyzing relationships between binary and continuous variables. While **biserial correlation** is suited for situations where the binary variable reflects an underlying continuous trait, **point-biserial correlation** is appropriate for true dichotomous variables. Understanding the difference between these two correlation techniques is essential for accurately interpreting results in fields like **educational testing**, **psychology**, and **medical diagnostics**.

By applying the correct method in each context, researchers can derive meaningful insights and improve the robustness of their analyses, ultimately contributing to better decision-making in both academic and practical settings.
