---
title: "Advanced Sequential Change-Point Detection for Univariate Models"
categories:
- Statistics
- Machine Learning
- Data Analysis
tags:
- Change-Point Detection
- Univariate Models
- Sequential Analysis
author_profile: false
classes: wide
# toc: true
# toc_label: The Complexity of Real-World Data Distributions
---

Sequential change-point detection is a dynamic field that deals with real-time monitoring of data sequences to detect points where the statistical properties change. This process is crucial in various domains such as finance, quality control, signal processing, and biostatistics. This document explores the theoretical background, advanced methodologies, practical implementations, and real-world applications of sequential change-point detection for univariate models.

## Theoretical Foundations

### Change-Point Model

Consider a sequence of observations $$\{X_t\}_{t=1}^n$$ where a change-point $$\tau$$ might occur such that:

For $$t \leq \tau$$, $$X_t \sim F_0$$

For $$t > \tau$$, $$X_t \sim F_1$$

Here, $$F_0$$ and $$F_1$$ are the distributions before and after the change, respectively. The change-point $$\tau$$ is unknown and needs to be detected.

### Hypothesis Testing Framework

The change-point detection problem can be framed as a hypothesis testing problem:

- Null hypothesis ($$H_0$$): No change, i.e., $$F_0 = F_1$$
- Alternative hypothesis ($$H_1$$): There is a change, i.e., $$F_0 \neq F_1$$

## Advanced Methods for Sequential Change-Point Detection

### Likelihood Ratio Test (LRT)

The likelihood ratio test is a powerful method for detecting changes in statistical properties. The likelihood ratio for a potential change-point $$k$$ is given by:

$$
\Lambda_k = \frac{\prod_{t=1}^k f_0(X_t) \prod_{t=k+1}^n f_1(X_t)}{\prod_{t=1}^n f_0(X_t)}
$$

The maximum likelihood ratio test statistic is:

$$
\Lambda_n = \max_{1 \leq k < n} \Lambda_k
$$

If $$\Lambda_n$$ exceeds a critical value, a change is detected.

### Cumulative Sum (CUSUM) Method

The CUSUM method accumulates deviations from a target value and signals a change when the cumulative sum exceeds a threshold. The CUSUM statistic is defined as:

$$
C_n = \max_{0 \leq k < n} \left| \sum_{t=k+1}^n (X_t - \mu) \right|
$$

where $$\mu$$ is the target mean. A change is detected if $$C_n$$ exceeds a predefined threshold $$h$$.

### Page's Cumulative Sum (CUSUM) Test

Page's test, a variation of the CUSUM method, is used to detect shifts in the mean of a sequence of observations. The test statistic is:

$$
W_n = \max_{1 \leq k < n} \left( \sum_{t=k+1}^n (X_t - \mu_0) - \frac{(n-k)}{2}\delta \right)
$$

where $$\delta$$ is the magnitude of the shift. A change-point is signaled if $$W_n$$ exceeds a threshold.

### Shiryaev-Roberts Procedure

The Shiryaev-Roberts procedure is a sequential method based on a Bayesian approach. The statistic is updated as follows:

$$
R_t = \left( 1 + R_{t-1} \right) \frac{f_1(X_t)}{f_0(X_t)}
$$

A change is detected if $$R_t$$ exceeds a threshold. This method balances the detection delay and false alarm rate effectively.

## Practical Implementations

### Algorithm Design

1. **Initialize Parameters**: Set initial values for the parameters such as means ($$\mu_0, \mu_1$$), variances ($$\sigma_0^2, \sigma_1^2$$), and thresholds ($$h$$).
2. **Real-Time Data Collection**: Collect data in a sequential manner.
3. **Compute Test Statistics**: Update the chosen test statistic (LRT, CUSUM, Page's, or Shiryaev-Roberts) with each new observation.
4. **Apply Decision Rule**: Compare the test statistic to the predefined threshold. If it exceeds the threshold, signal a change-point.
5. **Adapt and Iterate**: Continue collecting data and updating the test statistics until a change is detected or the sequence ends.

### Example Implementation: CUSUM Method

To implement the CUSUM method, follow these steps:

1. **Initialize**: $$C_0 = 0$$
2. **Update Statistic**: For each new observation $$X_n$$:
   $$
   C_n = \max(0, C_{n-1} + (X_n - \mu_0 - \delta))
   $$
3. **Decision Rule**: Signal a change-point if $$C_n > h$$.

## Real-World Applications

### Quality Control in Manufacturing

In manufacturing, sequential change-point detection is used to monitor the production process for deviations from the desired quality standards. For instance, changes in the mean or variance of a critical dimension can indicate a problem with the machinery or materials.

**Example**: Monitoring the diameter of produced parts:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

Using the CUSUM method, deviations from the target diameter are accumulated, and a change is signaled when the cumulative sum exceeds a threshold.

### Financial Market Analysis

In finance, change-point detection helps identify shifts in market trends or volatility. Detecting these shifts early can inform trading strategies and risk management.

**Example**: Monitoring stock returns:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

The likelihood ratio test can be used to detect changes in the mean return or volatility.

### Environmental Monitoring

Environmental scientists use change-point detection to identify changes in environmental parameters such as temperature, pollution levels, or water quality.

**Example**: Monitoring air pollution levels:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

Page's test can detect shifts in pollution levels, indicating potential changes in emissions or meteorological conditions.

### Biostatistics and Epidemiology

In biostatistics, change-point detection is used to monitor changes in clinical trial data, disease incidence rates, or other health-related metrics.

**Example**: Monitoring disease incidence rates:

Before change: $$X_t \sim Poisson(\lambda_0)$$

After change: $$X_t \sim Poisson(\lambda_1)$$

The Shiryaev-Roberts procedure can detect increases in incidence rates, signaling a potential outbreak.

## Mathematical Details

### Likelihood Ratio Test (LRT) Derivation

The likelihood ratio for a change-point at $$k$$ is:

$$
\Lambda_k = \frac{\prod_{t=1}^k f_0(X_t) \prod_{t=k+1}^n f_1(X_t)}{\prod_{t=1}^n f_0(X_t)}
$$

Taking the logarithm of the likelihood ratio:

$$
\log \Lambda_k = \sum_{t=k+1}^n \log \left( \frac{f_1(X_t)}{f_0(X_t)} \right)
$$

The test statistic is:

$$
\Lambda_n = \max_{1 \leq k < n} \log \Lambda_k
$$

### CUSUM Statistic Derivation

The CUSUM statistic is defined as:

$$
C_n = \max_{0 \leq k < n} \left| \sum_{t=k+1}^n (X_t - \mu) \right|
$$

This can be rewritten iteratively as:

$$
C_n = \max(0, C_{n-1} + (X_n - \mu))
$$

### Page's Test Derivation

Page's test statistic is:

$$
W_n = \max_{1 \leq k < n} \left( \sum_{t=k+1}^n (X_t - \mu_0) - \frac{(n-k)}{2} \delta \right)
$$

This incorporates the cumulative sum of deviations from the mean and adjusts for the expected shift $$\delta$$.

Sequential change-point detection is a vital tool in various fields for real-time monitoring and timely detection of changes in data sequences. Advanced methods such as Likelihood Ratio Tests, CUSUM, Page's Test, and the Shiryaev-Roberts procedure offer robust solutions tailored to different types of changes and data characteristics. Understanding the theoretical foundations, implementing practical algorithms, and applying these methods in real-world scenarios enable effective monitoring and response to changes in diverse applications.

## References

- Basseville, M., & Nikiforov, I. V. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice Hall.
- Shiryaev, A. N. (1963). "On optimum methods in quickest detection problems". *Theory of Probability and Its Applications*.
- Page, E. S. (1954). "Continuous inspection schemes". *Biometrika*.
- Pollak, M. (1985). "Optimal detection of a change in distribution". *The Annals of Statistics*.
- Tartakovsky, A. G., & Veeravalli, V. V. (2005). "Asymptotically quickest change detection in distributed sensor systems". *Sequential Analysis*.
- Tartakovsky, A. G., Rozovskii, B. L., & Blazek, R. E. (2014). *Sequential Analysis: Hypothesis Testing and Changepoint Detection*. CRC Press.
- Lai, T. L. (1995). "Sequential changepoint detection in quality control and dynamical systems". *Journal of the Royal Statistical Society: Series B*.
- Lorden, G. (1971). "Procedures for reacting to a change in distribution". *The Annals of Mathematical Statistics*.
