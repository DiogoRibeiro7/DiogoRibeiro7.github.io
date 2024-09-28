---
author_profile: false
categories:
- Probability Theory
- Statistics
- Mathematical Analysis
classes: wide
date: '2024-07-19'
excerpt: An analysis of the Central Limit Theorem for m-dependent random variables
  under sub-linear expectations, applying Rosenthal’s Inequality.
header:
  image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
keywords:
- Central Limit Theorem
- m-dependence
- Sub-linear Expectations
- Rosenthal’s Inequality
seo_description: An in-depth exploration of the Central Limit Theorem for m-dependent
  random variables under sub-linear expectations, with a focus on Rosenthal's Inequality.
seo_title: Central Limit Theorem for m-dependent Random Variables
summary: This article discusses the Central Limit Theorem for m-dependent random variables
  under sub-linear expectations, highlighting key concepts like Rosenthal’s Inequality.
tags:
- Central Limit Theorem
- m-dependence
- Sub-linear Expectations
- Rosenthal’s Inequality
title: Central Limit Theorem for m-dependent Random Variables Under Sub-linear Expectations
---

## Abstract

This article investigates the extension of the Central Limit Theorem (CLT) for $$m$$-dependent random variables within the framework of sub-linear expectations. The study establishes Rosenthal’s inequality for $$m$$-dependent variables, using it to prove the convergence of sums of such random variables to a $$G$$-normal distribution under sub-linear expectations. By combining tools from dependence theory and sub-linear expectation, this work significantly broadens the application of CLTs to models characterized by uncertainty, ambiguity, and risk, making the results relevant for a wide range of practical applications in finance, insurance, and risk management.

## Background

The Central Limit Theorem (CLT) is a cornerstone of probability theory and statistics. It states that the sum of a sufficiently large number of independent and identically distributed (i.i.d.) random variables, regardless of their original distribution, converges in distribution to a normal (Gaussian) distribution. This result is pivotal because it allows for the approximation of the sum of random variables, even when the individual distributions are unknown or complex. The classical CLT plays a crucial role in various fields, including statistical inference, hypothesis testing, and the foundation of many machine learning algorithms.

However, the standard CLT assumes independence among random variables, an assumption that is not always realistic in practical scenarios. Many real-world phenomena exhibit dependencies, whether in time series data, spatial statistics, or econometric models. Addressing such dependencies requires extending the classical CLT to accommodate more flexible dependence structures.

## Motivation

In real-world data, perfect independence is often a rare occurrence. Time series data, for instance, typically exhibit correlations, as do spatial data in geostatistics or returns in financial markets. The notion of $$m$$-dependence provides a more general framework for modeling such dependencies. In an $$m$$-dependent sequence, a random variable depends only on the preceding $$m$$ variables, capturing localized dependence while maintaining some long-range independence.

Sub-linear expectations, introduced by Shige Peng, offer a framework for extending classical probabilistic concepts to settings where uncertainty and ambiguity are present. Traditional expectations are linear and additive, which assumes that all probabilistic models are well-defined and free from ambiguity. However, in many applications—such as financial markets or insurance models—there may be inherent uncertainty in model selection, parameter estimation, or even the probability measure itself. Sub-linear expectations allow for non-additivity, providing a way to quantify and manage such uncertainties. By combining $$m$$-dependence with sub-linear expectations, this research builds a more robust version of the CLT, which can better reflect real-world complexities.

## Objective

The goal of this research is to extend the Central Limit Theorem to $$m$$-dependent random variables within the sub-linear expectation framework. This extension requires developing new mathematical tools—specifically, inequalities and theorems that account for both the local dependence and the non-linear nature of sub-linear expectations. The primary challenge lies in handling the dependencies between random variables while ensuring convergence to a generalized normal distribution (the $$G$$-normal distribution) under sub-linear expectations.

## Framework and Notations

### Sub-linear Expectations

Sub-linear expectations generalize the traditional concept of expectation by allowing for non-additive measures, which are more suitable for dealing with model risk, uncertainty, and ambiguity. In classical probability theory, the expectation operator is linear, meaning that the expectation of the sum of two independent random variables equals the sum of their expectations. Sub-linear expectations relax this additivity property, enabling the incorporation of multiple probabilistic models simultaneously.

The concept of $$G$$-expectation, introduced by Peng, is a specific type of sub-linear expectation that is used to define the $$G$$-normal distribution, a generalization of the classical normal distribution. The $$G$$-normal distribution is pivotal in this framework, as it accommodates the uncertainty and risk embedded in the model.

### Notations

To formalize the setting, we define the following notations:

- **Sub-linear Expectation Space**: Denoted by $$(\Omega, \mathcal{H}, \mathbb{E})$$, where:
  - $$\Omega$$ is the sample space,
  - $$\mathcal{H}$$ is a linear space of real functions on $$\Omega$$,
  - $$\mathbb{E}$$ is a sub-linear expectation.
  
- **Capacity**: A non-additive set function $$v$$ that generalizes the classical probability measure, providing a way to quantify uncertainty. Capacity theory plays a key role in sub-linear expectation theory.
  
- **$$G$$-normal distribution**: A generalization of the classical normal distribution under the sub-linear expectation framework. It encapsulates both the traditional Gaussian distribution and the uncertainty or ambiguity of the model.

- **$$m$$-dependence**: A sequence of random variables $$X_1, X_2, \dots$$ is said to be $$m$$-dependent if each random variable depends only on the preceding $$m$$ variables. Formally, for any $$i$$, the random variable $$X_i$$ is conditionally independent of $$X_{i-m-1}, X_{i-m-2}, \dots$$ given $$X_{i-1}, X_{i-2}, \dots, X_{i-m}$$.

These notations are essential for formalizing the extension of the CLT to $$m$$-dependent sequences and for handling the complexities introduced by sub-linear expectations.

## Central Limit Theorem for $$m$$-dependent Random Variables

### Theorem Statement

The extended Central Limit Theorem asserts that, under certain regularity conditions, the sum of $$m$$-dependent random variables converges in distribution to a $$G$$-normal distribution when evaluated using sub-linear expectations. This result generalizes the classical CLT by incorporating both dependencies between the random variables and non-linear expectations.

### Rosenthal’s Inequality

A key component in the proof of this extended CLT is Rosenthal’s inequality, which provides bounds on the moments of sums of dependent random variables. Specifically, for $$m$$-dependent random variables, Rosenthal’s inequality establishes that the moments of the sum can be controlled by the moments of individual variables. This is crucial for ensuring the convergence of the distribution of the sum to the $$G$$-normal distribution under sub-linear expectations.

### Conditions for Convergence

The CLT for $$m$$-dependent random variables requires certain conditions to hold. These include:

- **Boundedness**: The random variables must satisfy certain boundedness conditions, such as bounded variance or bounded moments.
- **Moment Constraints**: Higher moments of the random variables need to be controlled, as these moments play a key role in ensuring the applicability of Rosenthal’s inequality.
- **Dependence Structure**: The $$m$$-dependence must be sufficiently localized, meaning that each variable can depend only on a finite number of preceding variables.

### Proof Outline

The proof of the extended CLT proceeds in several key steps:

1. **Establishing Rosenthal’s Inequality** for $$m$$-dependent random variables, providing the necessary moment bounds.
2. **Demonstrating boundedness** of the higher moments of the sums, ensuring that the moments do not grow too quickly as the number of random variables increases.
3. **Proving convergence** to the $$G$$-normal distribution by employing the framework of sub-linear expectations. This involves showing that the sum of $$m$$-dependent random variables behaves asymptotically like a $$G$$-normal distribution under sub-linear expectations.

## Truncated Conditions

### Truncated Variables

In many practical applications, it is necessary to work with truncated random variables, which exclude extreme values to avoid the influence of outliers. Truncated random variables are often used when the full distribution may contain extreme values that distort the results of statistical analysis.

### Theorem for Truncated Variables

The extension of the CLT to $$m$$-dependent random variables with truncated distributions follows a similar structure as the general case, but with additional steps to handle the truncation of extreme values. The truncated CLT asserts that, under certain regularity conditions, the sum of truncated $$m$$-dependent random variables converges to a $$G$$-normal distribution.

### Proof Outline

The proof for truncated variables involves the following steps:

1. **Defining the truncated variables** and establishing their moment properties.
2. **Adapting Rosenthal’s inequality** to the truncated setting, ensuring that the moment bounds still hold.
3. **Proving convergence** to the $$G$$-normal distribution for the truncated variables.

## Corollaries and Specific Cases

### Stationary Sequences

A particularly important application of the extended CLT is to stationary sequences of $$m$$-dependent random variables, where the joint distribution of the sequence is invariant under time shifts. The extended CLT provides a powerful tool for analyzing such sequences in settings with sub-linear expectations.

### Independent Case

When $$m = 0$$, the $$m$$-dependence assumption reduces to the case of independent random variables. In this case, the extended CLT recovers the classical Central Limit Theorem, providing a direct connection between the classical and generalized results.

### Mean Uncertainty

A notable corollary of the extended CLT applies to sequences without mean uncertainty. In such cases, the conditions for the CLT can be simplified, as the uncertainty in the mean behavior of the sequence is reduced. This result is particularly relevant in situations where the overall mean of the sequence is well-defined and less influenced by model uncertainty.

## Summary

This article extends the classical Central Limit Theorem to $$m$$-dependent random variables under the framework of sub-linear expectations. By introducing Rosenthal’s inequality for $$m$$-dependent sequences and proving convergence to a $$G$$-normal distribution, the study provides new mathematical tools for handling dependent random variables in uncertain environments. The results have significant implications for modeling in fields where dependencies and uncertainties are prevalent, such as finance, insurance, and risk management.

## Implications

The extension of the Central Limit Theorem to $$m$$-dependent random variables within the sub-linear expectation framework broadens the applicability of this fundamental theorem. It offers robust tools for dealing with uncertainty in complex models, which are essential in fields that require rigorous probabilistic analysis under ambiguous conditions.

## Future Work

Future research can build on this work by exploring applications in specific domains, such as financial risk modeling and actuarial science. Additionally, further extensions of these results to other types of dependencies or to non-stationary sequences could be investigated. Numerical methods for practical implementation and computation of $$G$$-normal distributions in real-world data would also be valuable contributions.

## References

1. Peng, S. (2007). $$G$$-expectation, $$G$$-Brownian motion and related stochastic calculus of Itô type. *Stochastic Analysis and Applications*.
2. Rosenthal, H. P. (1970). On the subspaces of $$L^p$$ (p > 2) spanned by sequences of independent random variables. *Israel Journal of Mathematics*.
3. Billingsley, P. (1995). *Probability and Measure*. John Wiley & Sons.
4. Durrett, R. (2019). *Probability: Theory and Examples*. Cambridge University Press.
5. Grimmett, G., & Stirzaker, D. (2001). *Probability and Random Processes*. Oxford University Press.
6. Kallenberg, O. (2002). *Foundations of Modern Probability*. Springer Science & Business Media.
7. Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and Statistical Modeling*. Springer Science & Business Media.
8. Shao, Q. M. (2003). *Mathematical Statistics*. Springer Science & Business Media.
9. Van der Vaart, A. W. (2000). *Asymptotic Statistics*. Cambridge University Press.
