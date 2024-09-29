---
author_profile: false
categories:
- Statistics
- Machine Learning
- Data Analysis
classes: wide
date: '2024-02-14'
excerpt: Sequential change-point detection plays a crucial role in real-time monitoring
  across industries. Learn about advanced methods, their practical applications, and
  how they help detect changes in univariate models.
header:
  image: /assets/images/data_science_3.jpg
  og_image: /assets/images/data_science_4.jpg
  overlay_image: /assets/images/data_science_3.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_3.jpg
  twitter_image: /assets/images/data_science_4.jpg
keywords:
- change-point detection
- univariate models
- sequential analysis
- real-time monitoring
- statistical detection methods
- data analysis
- machine learning models
- anomaly detection
- sequential change-point algorithms
- time series analysis
- python
seo_description: Explore advanced methods and practical implementations for sequential
  change-point detection in univariate models, covering theoretical foundations, real-world
  applications, and key statistical techniques.
seo_title: Advanced Techniques for Sequential Change-Point Detection in Univariate
  Models
seo_type: article
tags:
- Change-Point Detection
- Univariate Models
- Sequential Analysis
- python
title: Advanced Sequential Change-Point Detection for Univariate Models
---

## Overview

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

### Monitoring for Multiple Change-Points

In real-world applications, sequences may contain multiple change-points. To handle this scenario:

1. **Reset After Detection**: After a change-point is detected, reset the detection process to look for additional changes in the subsequent data.
2. **Segment-Based Methods**: Use segment-based methods to partition the sequence into homogeneous segments and detect multiple change-points in each segment.
3. **Sliding Window Approach**: Apply a sliding window technique that continuously monitors overlapping subsets of the data, allowing for the detection of changes that might be missed by analyzing the full sequence.

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

## Appendix: Python Implementation of the CUSUM Method

The following Python code demonstrates how to implement the CUSUM method for sequential change-point detection. In this example, we simulate a univariate sequence with a mean shift, apply the CUSUM method, and detect when a change-point occurs.

```python
import numpy as np
import matplotlib.pyplot as plt

# CUSUM function
def cusum(data, mu_0, delta, h):
    """
    CUSUM implementation for sequential change-point detection.

    Parameters:
    data (np.array): The observed data sequence.
    mu_0 (float): The target mean before the change.
    delta (float): The magnitude of the shift after the change.
    h (float): The decision threshold for signaling a change-point.

    Returns:
    C (np.array): CUSUM statistic over time.
    t_change (int): The time index where the change-point is detected, or None if not detected.
    """
    C = np.zeros(len(data))  # Initialize the CUSUM statistic array
    t_change = None  # Change-point time index

    for t in range(1, len(data)):
        # Update the CUSUM statistic
        C[t] = max(0, C[t - 1] + (data[t] - mu_0 - delta))

        # Check if the CUSUM statistic exceeds the threshold
        if C[t] > h:
            t_change = t
            break  # Stop when the first change-point is detected

    return C, t_change

# Simulated data
np.random.seed(42)
n = 100
mu_0 = 0  # Mean before change
mu_1 = 2  # Mean after change
change_point = 60  # Actual change-point

# Data before and after the change
data = np.concatenate([np.random.normal(mu_0, 1,
```

### Explanation of the Code

#### CUSUM Function:

The `cusum` function calculates the CUSUM statistic iteratively. It checks for a change-point when the CUSUM statistic exceeds the decision threshold `h`.

**Inputs**:

- `data`: The observed sequence of data points.
- `mu_0`: The target mean before the change.
- `delta`: The magnitude of the shift in the mean after the change.
- `h`: The threshold for signaling a change.

**Outputs**:

- `C`: The CUSUM statistic over time.
- `t_change`: The index at which a change-point is detected, or `None` if no change-point is detected.

#### Simulated Data:

A sequence of 100 observations is generated. The first 60 points follow a normal distribution with mean `mu_0 = 0`, and the last 40 points follow a normal distribution with mean `mu_1 = 2`, representing a shift in the data.

#### CUSUM Parameters:

We set `delta = 1`, the expected magnitude of the mean shift, and a decision threshold of `h = 5`.

#### CUSUM Detection:

The CUSUM method is applied to the data, and the CUSUM statistic is updated sequentially. The detected change-point is marked once the statistic exceeds the threshold.

#### Plotting the Results:

The data, the actual change-point, the detected change-point, and the CUSUM statistic are plotted for visualization.

### Output

The output of the code is a plot that shows the following:

- The observed data sequence.
- The actual change-point (marked by a red dashed line).
- The detected change-point using the CUSUM method (marked by a green dashed line).
- The CUSUM statistic over time (shown in orange).

In addition, the index of the detected change-point is printed to the console.

This example demonstrates how the CUSUM method can be used to detect changes in the statistical properties of a univariate time series in a sequential manner.

## References

- Basseville, M., & Nikiforov, I. V. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice Hall.
- Shiryaev, A. N. (1963). "On optimum methods in quickest detection problems". *Theory of Probability and Its Applications*.
- Page, E. S. (1954). "Continuous inspection schemes". *Biometrika*.
- Pollak, M. (1985). "Optimal detection of a change in distribution". *The Annals of Statistics*.
- Tartakovsky, A. G., & Veeravalli, V. V. (2005). "Asymptotically quickest change detection in distributed sensor systems". *Sequential Analysis*.
- Tartakovsky, A. G., Rozovskii, B. L., & Blazek, R. E. (2014). *Sequential Analysis: Hypothesis Testing and Changepoint Detection*. CRC Press.
- Lai, T. L. (1995). "Sequential changepoint detection in quality control and dynamical systems". *Journal of the Royal Statistical Society: Series B*.
- Lorden, G. (1971). "Procedures for reacting to a change in distribution". *The Annals of Mathematical Statistics*.
