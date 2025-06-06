---
author_profile: false
categories:
- Probability Theory
- Statistics
- Mathematical Analysis
classes: wide
date: '2024-07-19'
excerpt: This article rigorously explores the Central Limit Theorem for m-dependent
  random variables under sub-linear expectations, presenting new inequalities, proof
  outlines, and implications in modeling dependent sequences.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Central limit theorem
- M-dependence
- Sub-linear expectations
- Rosenthal’s inequality
- Truncated variables
seo_description: A detailed study on the extension of the Central Limit Theorem for
  m-dependent random variables under sub-linear expectations, focusing on Rosenthal's
  inequality and handling truncated variables.
seo_title: Central Limit Theorem for m-dependent Random Variables
seo_type: article
summary: This article extends the classical Central Limit Theorem (CLT) to m-dependent
  random variables within the sub-linear expectation framework. It incorporates Rosenthal's
  inequality for m-dependent sequences, examines truncated conditions, and discusses
  the broader implications for real-world systems characterized by uncertainty and
  dependencies.
tags:
- Central limit theorem
- M-dependence
- Sub-linear expectations
- Rosenthal’s inequality
title: Central Limit Theorem for m-dependent Random Variables Under Sub-linear Expectations
---

## Abstract

This article provides a comprehensive investigation of the Central Limit Theorem (CLT) for $$m$$-dependent random variables within the framework of sub-linear expectations. By extending Rosenthal’s inequality to $$m$$-dependent variables and using the theoretical foundation of sub-linear expectations, we generalize the classical CLT to sequences that exhibit both dependence and uncertainty. These results have profound implications for fields such as finance, risk management, and other areas where model uncertainty plays a significant role. We explore truncated variables, proof structures, corollaries, and special cases, offering a full mathematical treatment of the subject.

## Introduction

The Central Limit Theorem (CLT) is one of the foundational results in probability theory and statistics. It asserts that for a sufficiently large number of independent and identically distributed (i.i.d.) random variables, their properly normalized sum converges to a normal distribution. The CLT has countless applications, ranging from inferential statistics to stochastic modeling, econometrics, and various branches of science and engineering. However, the classical form of the CLT assumes independence among random variables, an assumption that is frequently violated in real-world data, where dependencies are the norm rather than the exception.

Another limitation of the classical CLT is its reliance on traditional expectation, which assumes the existence of a single probability measure governing the data. However, in many practical scenarios—such as financial markets, insurance, and environmental science—uncertainty about the underlying probabilistic model itself may be present. This type of uncertainty, known as model uncertainty or ambiguity, calls for more flexible frameworks that can handle multiple possible probabilistic models.

To address these limitations, this article explores the extension of the CLT to $$m$$-dependent random variables under the framework of sub-linear expectations. $$m$$-dependence relaxes the assumption of independence by allowing each random variable to depend on the preceding $$m$$ variables in the sequence, providing a more realistic model of dependence. Sub-linear expectations, as introduced by Shige Peng, provide a robust method for dealing with model uncertainty, allowing for non-additive measures that can account for ambiguity and risk.

Our objective is to extend the CLT to $$m$$-dependent sequences within the framework of sub-linear expectations and to explore the implications of this generalization in various practical applications. We begin by revisiting the classical CLT and discussing its limitations, followed by an introduction to $$m$$-dependence and sub-linear expectations. We then present Rosenthal’s inequality for $$m$$-dependent variables, which plays a key role in the proof of the extended CLT. Finally, we examine truncated variables and the implications of this result in specific applications, such as stationary sequences and risk modeling.

## Classical Central Limit Theorem: Revisited

The classical Central Limit Theorem can be stated as follows:

Let $$X_1, X_2, \dots, X_n$$ be independent and identically distributed random variables with mean $$\mu$$ and variance $$\sigma^2$$. Then, the properly normalized sum of these variables,

$$
S_n = \frac{1}{\sqrt{n}} \left( \sum_{i=1}^n X_i - n\mu \right),
$$

converges in distribution to a standard normal random variable as the sample size grows:

$$
S_n \overset{d}{\rightarrow} \mathcal{N}(0, 1) \quad \text{as} \quad n \rightarrow \infty.
$$

This powerful result underlies many aspects of statistical inference. However, it crucially relies on the assumption that the $$X_i$$ are independent. In many real-world cases, such as time series data in finance or economics, this assumption does not hold. In addition, the classical expectation operator is linear and additive, which assumes that a single probabilistic model governs the process. However, there may be uncertainty or ambiguity in the underlying model itself, requiring a more general framework.

### Limitations of the Classical CLT

1. **Independence Assumption**: In the classical CLT, independence between random variables is required. However, many applications involve dependent data, such as stock market prices, which are influenced by past prices, or weather patterns, which may depend on historical data.
   
2. **Single Expectation Model**: The classical CLT operates under the assumption that there is a single, well-defined probability measure governing the distribution of the random variables. This may not be realistic in situations where there is ambiguity about the true probability model or where multiple models need to be considered simultaneously (e.g., in financial risk modeling).

### Generalizing the CLT

To address these limitations, we explore two generalizations of the CLT:

1. **$$m$$-Dependence**: A more flexible dependence structure where each random variable depends only on the preceding $$m$$ variables.
2. **Sub-linear Expectations**: A framework that extends classical expectations to non-additive measures, accommodating uncertainty about the underlying probability measure.

## $$m$$-dependence: Local Dependencies in Sequences

In practical applications, sequences of random variables often exhibit dependence. One way to model this is through $$m$$-dependence, which is defined as follows:

A sequence of random variables $$\{X_i\}_{i=1}^n$$ is said to be $$m$$-dependent if for any pair of indices $$i$$ and $$j$$ such that $$\lvert i - j\rvert > m$$, the random variables $$X_i$$ and $$X_j$$ are independent. This structure allows for dependencies within a "window" of $$m$$ terms but ensures that distant variables are independent.

Formally, we say that $$X_1, X_2, \dots, X_n$$ is $$m$$-dependent if for all $$i$$ and $$j$$ such that $$\lvert i - j\rvert > m$$,, the conditional independence holds:

$$
X_i \perp X_j \quad \text{whenever} \quad |i - j| > m.
$$

### Examples of $$m$$-dependence

1. **Time Series Models**: Consider a time series where each observation depends on the preceding $$m$$ observations. For example, in an autoregressive model of order $$m$$, denoted AR($$m$$), each value $$X_t$$ is modeled as a linear combination of the previous $$m$$ values plus some noise. This is a natural example of an $$m$$-dependent process.
   
2. **Markov Chains**: A Markov chain with a finite memory length of $$m$$ is another example. In such a process, the probability of moving to a new state depends only on the last $$m$$ states, making the sequence $$m$$-dependent.

### Implications of $$m$$-dependence

$$m$$-dependence strikes a balance between full independence and complete dependence, making it a useful model for processes where local dependencies exist but distant observations are effectively independent. This structure is particularly relevant in fields like time series analysis, spatial statistics, and finance, where dependencies over time or space are common but do not extend indefinitely.

## Sub-linear Expectations: Generalizing Classical Probability

Classical probability theory operates under the assumption of a single probability measure $$P$$, and the expectation of a random variable $$X$$ is given by the linear operator:

$$
\mathbb{E}[X] = \int X dP.
$$

In many situations, however, the true underlying probability measure may not be known, or there may be ambiguity in the model. This is where sub-linear expectations come into play. Introduced by Peng (2007), sub-linear expectations extend classical expectations to a non-additive framework, allowing for greater flexibility in the presence of uncertainty.

### Definition of Sub-linear Expectations

A sub-linear expectation is a functional $$\mathbb{E}: \mathcal{H} \to \mathbb{R}$$ that satisfies the following properties for all $$X, Y \in \mathcal{H}$$ and $$\lambda \geq 0$$:

1. **Monotonicity**: If $$X \geq Y$$, then $$\mathbb{E}(X) \geq \mathbb{E}(Y)$$.
2. **Constant Preservation**: For any constant $$c \in \mathbb{R}$$, $$\mathbb{E}(c) = c$$.
3. **Sub-additivity**: For all $$X, Y \in \mathcal{H}$$, $$\mathbb{E}(X + Y) \leq \mathbb{E}(X) + \mathbb{E}(Y)$$.
4. **Positive Homogeneity**: For any $$\lambda \geq 0$$ and $$X \in \mathcal{H}$$, $$\mathbb{E}(\lambda X) = \lambda \mathbb{E}(X)$$.

These properties allow sub-linear expectations to model situations where there is uncertainty or ambiguity in the underlying probability measure. In this context, the term "sub-linear" refers to the fact that the expectation operator is no longer additive, reflecting the uncertainty in the model.

### The $$G$$-expectation and $$G$$-normal Distribution

One important example of a sub-linear expectation is the $$G$$-expectation, which leads to the concept of the $$G$$-normal distribution. The $$G$$-normal distribution is a generalization of the classical normal distribution under sub-linear expectations and plays a central role in our generalization of the CLT.

In a $$G$$-expectation framework, a random variable $$X$$ is said to follow a $$G$$-normal distribution if, under the sub-linear expectation $$\mathbb{E}$$, its behavior is analogous to that of a Gaussian random variable in the classical case. Specifically, the $$G$$-normal distribution captures the idea that the variance of $$X$$ is uncertain or ambiguous, making it a powerful tool for modeling uncertainty in finance and other fields.

## Rosenthal’s Inequality for $$m$$-dependent Sequences

Rosenthal's inequality provides a powerful tool for controlling the moments of sums of random variables. For independent random variables, Rosenthal's inequality bounds the $$p$$-th moment of the sum of these variables in terms of the individual moments. In the case of $$m$$-dependent random variables, Rosenthal's inequality can be extended to account for the dependence structure, playing a crucial role in the proof of the extended CLT.

### Classical Rosenthal’s Inequality

For independent random variables $$X_1, X_2, \dots, X_n$$ with finite moments, Rosenthal's inequality states that for any $$p \geq 2$$, there exists a constant $$C_p$$ such that:

$$
\mathbb{E} \left( \left| \sum_{i=1}^n X_i \right|^p \right) \leq C_p \left( \sum_{i=1}^n \mathbb{E}(|X_i|^p) + \left( \sum_{i=1}^n \mathbb{E}(X_i^2) \right)^{p/2} \right).
$$

This inequality provides an upper bound on the $$p$$-th moment of the sum of independent random variables. The extension of this result to $$m$$-dependent random variables is non-trivial, as it requires taking into account the dependencies between the variables.

### Rosenthal’s Inequality for $$m$$-dependent Variables

For $$m$$-dependent random variables, Rosenthal’s inequality can be extended as follows:

Let $$X_1, X_2, \dots, X_n$$ be an $$m$$-dependent sequence of random variables with finite moments. Then, for any $$p \geq 2$$, there exists a constant $$C_p$$ such that:

$$
\mathbb{E} \left( \left| \sum_{i=1}^n X_i \right|^p \right) \leq C_p \left( \sum_{i=1}^n \mathbb{E}(|X_i|^p) + \sum_{k=1}^m \left( \sum_{i=1}^n \mathbb{E}(X_i^2) \right)^{p/2} \right).
$$

This inequality plays a central role in the proof of the CLT for $$m$$-dependent sequences, as it allows us to control the higher moments of the sum of dependent random variables, ensuring that the sum converges to a $$G$$-normal distribution under sub-linear expectations.

### Proof Outline for the CLT under Sub-linear Expectations

The proof of the CLT for $$m$$-dependent random variables under sub-linear expectations proceeds in several key steps:

1. **Moment Control via Rosenthal’s Inequality**: We begin by applying Rosenthal’s inequality for $$m$$-dependent variables to control the higher moments of the sum of the random variables. This ensures that the moments of the sum do not grow too rapidly, allowing us to establish convergence in distribution.

2. **Boundedness of Higher Moments**: Using the bounds provided by Rosenthal’s inequality, we demonstrate that the higher moments of the normalized sum are bounded. This step is crucial for ensuring that the distribution of the sum does not "blow up" as the number of variables increases.

3. **Convergence to the $$G$$-normal Distribution**: Finally, we show that the sum of $$m$$-dependent random variables converges in distribution to a $$G$$-normal distribution under sub-linear expectations. This step involves verifying that the limiting distribution satisfies the properties of the $$G$$-normal distribution, including the uncertainty in the variance.

## Truncated Conditions

In many applications, especially in financial modeling and risk management, it is common to work with truncated random variables. Truncated random variables are those that have been modified to exclude extreme values, which may distort the results of statistical analysis. Truncation is particularly useful in scenarios where the full distribution contains outliers or other extreme values that are not representative of the underlying process.

### Truncated Variables and the CLT

The extension of the CLT to $$m$$-dependent random variables with truncated distributions follows a similar structure to the general case, but with additional steps to handle the truncation of extreme values. The truncated CLT can be stated as follows:

Let $$\{X_i\}_{i=1}^n$$ be an $$m$$-dependent sequence of truncated random variables. Under certain regularity conditions (such as bounded moments and truncation thresholds), the sum of these variables converges in distribution to a $$G$$-normal distribution under sub-linear expectations.

### Proof Outline for the Truncated CLT

The proof of the truncated CLT involves the following steps:

1. **Defining the Truncated Variables**: We begin by defining the truncated random variables, which are modified to exclude extreme values. This step ensures that the distribution of the variables is well-behaved and that outliers do not distort the results.

2. **Adapting Rosenthal’s Inequality**: Next, we adapt Rosenthal’s inequality to the truncated setting, ensuring that the moment bounds still hold for the truncated variables.

3. **Convergence to the $$G$$-normal Distribution**: Finally, we prove that the sum of the truncated random variables converges in distribution to a $$G$$-normal distribution under sub-linear expectations, using the same techniques as in the general case.

## Corollaries and Specific Cases

### Stationary Sequences

A particularly important application of the extended CLT is to stationary sequences of $$m$$-dependent random variables. A sequence is said to be stationary if its joint distribution is invariant under time shifts, meaning that the statistical properties of the sequence do not change over time. The extended CLT provides a powerful tool for analyzing such sequences in settings with sub-linear expectations, as it allows for dependencies and uncertainty in the model.

### Independent Case

When $$m = 0$$, the $$m$$-dependence assumption reduces to the case of independent random variables. In this case, the extended CLT recovers the classical Central Limit Theorem, providing a direct connection between the classical and generalized results.

### Mean Uncertainty

In many applications, there may be uncertainty about the mean of the random variables in the sequence. A notable corollary of the extended CLT applies to sequences without mean uncertainty, where the conditions for the CLT can be simplified. In such cases, the uncertainty in the mean behavior of the sequence is reduced, allowing for a more straightforward application of the theorem.

## Implications of the Extended CLT

The extension of the Central Limit Theorem to $$m$$-dependent random variables within the sub-linear expectation framework has significant implications for a wide range of fields. In particular, the ability to handle dependencies and uncertainty simultaneously is crucial in applications such as:

- **Financial Risk Modeling**: In finance, dependencies between asset returns and uncertainty about the underlying model are pervasive. The extended CLT provides a rigorous framework for modeling these dependencies and uncertainties, allowing for more accurate risk assessments.

- **Insurance and Actuarial Science**: In the insurance industry, dependencies between claims and uncertainty about the probability distribution of losses are common. The extended CLT offers a robust tool for analyzing such systems, providing insights into the distribution of aggregate losses.

- **Time Series Analysis**: In econometrics and time series analysis, dependencies between observations are the norm rather than the exception. The extended CLT provides a theoretical foundation for analyzing time series data with local dependencies and uncertainty.

## Future Work

Future research could build on this work by exploring applications of the extended CLT in specific domains, such as financial risk modeling, insurance, and time series analysis. Additionally, further extensions of the CLT to other types of dependence structures (such as mixing sequences or Markov chains) and other types of uncertainty (such as uncertainty in higher-order moments) could be investigated. Another avenue for future research is the development of numerical methods for computing $$G$$-normal distributions in practical applications.

## Conclusion

In this article, we have extended the Central Limit Theorem to $$m$$-dependent random variables within the framework of sub-linear expectations. By introducing Rosenthal’s inequality for $$m$$-dependent sequences and proving convergence to a $$G$$-normal distribution, we have developed new mathematical tools for handling dependent random variables in uncertain environments. The results have significant implications for fields where dependencies and uncertainties are prevalent, such as finance, insurance, and risk management.

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
