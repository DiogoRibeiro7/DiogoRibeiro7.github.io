---
author_profile: false
categories:
- Probability Theory
- Statistics
- Mathematical Analysis
classes: wide
date: '2024-07-19'
header:
  image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
tags:
- Central Limit Theorem
- m-dependence
- Sub-linear Expectations
- Rosenthal’s Inequality
title: Central Limit Theorem for m-dependent Random Variables Under Sub-linear
  Expectations
---

## Abstract

This article explores the extension of the Central Limit Theorem (CLT) for $$m$$-dependent random variables under sub-linear expectations. By establishing Rosenthal’s inequality for $$m$$-dependent variables and employing the framework of sub-linear expectations, this study broadens the application of CLTs in models characterized by uncertainty.

## Background

The Central Limit Theorem (CLT) is a fundamental result in probability theory and statistics, asserting that the sum of a large number of independent and identically distributed random variables tends towards a normal distribution, regardless of the original distribution. This theorem underpins many statistical methods and justifies the approximation of complex distributions by the normal distribution.

## Motivation

In practical scenarios, independence among variables is often an unrealistic assumption. $$m$$-dependence provides a more flexible model for dependent sequences, where each variable is dependent only on the preceding $$m$$ variables. Sub-linear expectations, introduced by Peng, offer a robust framework for handling uncertainty and model risk, extending classical probabilistic concepts to situations where traditional expectations may fail.

## Objective

The aim of this study is to extend the Central Limit Theorem to $$m$$-dependent random variables within the framework of sub-linear expectations. This involves developing new inequalities and theorems that accommodate the dependencies and non-linearities inherent in these models.

## Framework and Notations

### Sub-linear Expectations

Sub-linear expectations are non-additive measures that generalize classical expectations, accommodating uncertainty and model risk. Peng’s framework introduces the notion of $$G$$-expectation and $$G$$-normal distribution, which are pivotal in this generalized setting.

### Notations

- **Sub-linear Expectation Space**: A triplet $$(\Omega, \mathcal{H}, \mathbb{E})$$, where $$\Omega$$ is the sample space, $$\mathcal{H}$$ is a linear space of real functions on $$\Omega$$, and $$\mathbb{E}$$ is a sub-linear expectation.
- **Capacity**: A non-additive set function that measures the size of sets in terms of uncertainty.
- **$$G$$-normal distribution**: A generalization of the normal distribution under sub-linear expectations.
- **$$m$$-dependence**: A property where each variable in a sequence depends only on the preceding $$m$$ variables.

These concepts are crucial for extending CLTs to dependent sequences, providing the necessary mathematical tools to handle dependencies and non-linearities.

## Central Limit Theorem for $$m$$-dependent Random Variables

### Theorem Statement

The main theorem states that under certain conditions, the sum of $$m$$-dependent random variables converges to a $$G$$-normal distribution under sub-linear expectations.

### Rosenthal’s Inequality

Rosenthal’s inequality for $$m$$-dependent variables is a key tool in the proof of the CLT. It provides bounds on the moments of sums of dependent random variables, which are essential for establishing convergence.

### Conditions

The CLT for $$m$$-dependent random variables holds under specific conditions, such as boundedness and certain moment constraints. These conditions ensure the applicability of Rosenthal’s inequality and the convergence of the distribution.

### Proof Outline

The proof involves several steps:
1. Establishing Rosenthal’s inequality for $$m$$-dependent variables.
2. Demonstrating the boundedness of higher moments.
3. Showing the convergence to the $$G$$-normal distribution using sub-linear expectations.

## Truncated Conditions

### Truncated Variables

Truncated random variables are those that have been modified to exclude extreme values. This concept is useful in scenarios where the full distribution includes outliers that can distort the results.

### Theorem Statement

The theorem under truncated conditions extends the CLT to $$m$$-dependent random variables with truncated distributions.

### Proof Outline

The proof under truncated conditions follows a similar structure but involves additional steps to handle the truncation:

1. Defining the truncated variables and their properties.
2. Adapting Rosenthal’s inequality to the truncated setting.
3. Proving convergence to the $$G$$-normal distribution with truncated variables.

## Corollaries and Specific Cases

### Stationary Sequences

The main theorem can be applied to stationary sequences of $$m$$-dependent random variables, where the joint distribution of the variables is invariant under time shifts.

### Independent Case

A special case of the theorem is when $$m = 0$$, corresponding to independent random variables. This recovers the classical CLT results within the framework of sub-linear expectations.

### Mean Uncertainty

For sequences without mean uncertainty, the conditions for the CLT can be simplified. This corollary highlights scenarios where the average behavior of the sequence is well-defined and less influenced by uncertainty.

## Summary

This study extends the Central Limit Theorem to $$m$$-dependent random variables under sub-linear expectations, establishing new inequalities and conditions for convergence. The results have significant implications for modeling dependent sequences and handling uncertainty in various fields.

## Implications

The extension of the CLT to $$m$$-dependent variables under sub-linear expectations broadens the applicability of this fundamental theorem. It offers robust tools for statistical analysis in finance, insurance, and other areas characterized by uncertainty and dependencies.

## Future Work

Future research could explore applications of these results in specific fields, develop numerical methods for practical implementation, and investigate further extensions to other types of dependencies and non-linearities.

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
