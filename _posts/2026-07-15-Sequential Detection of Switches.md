---
title: "Sequential Detection of Switches in Models with Changing Structures"
categories:
- Statistics
- Machine Learning
- Data Analysis
tags:
- Change-Point Detection
- Sequential Analysis
- Structural Change
author_profile: false
---

Sequential detection of switches in models with changing structures is a critical aspect of real-time data analysis. It involves identifying points in a data sequence where the underlying statistical properties shift, signaling a change in the model structure. This process is essential in fields like finance, signal processing, and engineering, where timely detection of structural changes can significantly impact decision-making and system performance. This document delves into the theoretical framework, advanced techniques, practical implementations, and real-world applications of detecting switches in models with changing structures.

## Theoretical Foundations

### Model Switching Framework

In sequential change-point detection, we consider a sequence of observations $$\{X_t\}_{t=1}^n$$ that may undergo structural changes at unknown points $$\{\tau_i\}$$. These change-points mark the transitions between different regimes or models:

For $$t \leq \tau_1$$: $$X_t \sim F_0$$

For $$\tau_1 < t \leq \tau_2$$: $$X_t \sim F_1$$

...

For $$t > \tau_m$$: $$X_t \sim F_m$$

Here, $$F_i$$ represents the distribution of the observations in the $$i$$-th regime. The goal is to detect the points $$\{\tau_i\}$$ where the model switches from $$F_{i-1}$$ to $$F_i$$.

### Hypothesis Testing for Structural Changes

Detecting switches in models can be framed as a hypothesis testing problem for each potential change-point $$\tau$$:

- Null hypothesis ($$H_0$$): No change at $$\tau$$, i.e., $$F_{i-1} = F_i$$
- Alternative hypothesis ($$H_1$$): A change occurs at $$\tau$$, i.e., $$F_{i-1} \neq F_i$$

This hypothesis testing framework underpins many of the techniques used for sequential detection of structural changes.

## Advanced Techniques for Sequential Detection

### Generalized Likelihood Ratio Test (GLRT)

The Generalized Likelihood Ratio Test extends the likelihood ratio test to handle multiple potential change-points. For a candidate change-point $$k$$, the GLRT statistic is:

$$
\Lambda_k = \frac{\max_{\theta_0} \prod_{t=1}^k f_0(X_t; \theta_0) \cdot \max_{\theta_1} \prod_{t=k+1}^n f_1(X_t; \theta_1)}{\max_{\theta} \prod_{t=1}^n f(X_t; \theta)}
$$

where $$\theta_0$$ and $$\theta_1$$ are the parameters of the distributions before and after the change, respectively. The test statistic for the entire sequence is:

$$
\Lambda_n = \max_{1 \leq k < n} \Lambda_k
$$

A change is detected if $$\Lambda_n$$ exceeds a critical value.

### Adaptive Cumulative Sum (CUSUM) Method

The adaptive CUSUM method modifies the traditional CUSUM approach to account for changing structures. The statistic is updated based on the most recent observations, allowing for dynamic adaptation:

$$
C_n = \max_{0 \leq k < n} \left| \sum_{t=k+1}^n (X_t - \mu_k) \right|
$$

where $$\mu_k$$ is the estimated mean for the $$k$$-th segment. This method detects changes when $$C_n$$ exceeds a predefined threshold.

### Bayesian Change-Point Detection

Bayesian methods provide a probabilistic framework for change-point detection, incorporating prior information about potential change-points and model parameters. The posterior probability of a change-point at time $$t$$ is updated sequentially:

$$
P(\tau = t | X_{1:n}) \propto P(X_{1:t} | \tau = t) \cdot P(X_{t+1:n} | \tau = t) \cdot P(\tau = t)
$$

A change is signaled when the posterior probability exceeds a threshold.

### Hidden Markov Models (HMM)

HMMs are well-suited for modeling sequences with switches between multiple states. Each state represents a different regime, and the transition probabilities capture the likelihood of switching between states. The Viterbi algorithm can be used to decode the most probable sequence of states, identifying the change-points.

## Practical Implementations

### Algorithm Design

1. **Initialize Parameters**: Set initial values for parameters such as means, variances, and thresholds.
2. **Real-Time Data Collection**: Collect data sequentially.
3. **Compute Test Statistics**: Update the chosen test statistic (GLRT, adaptive CUSUM, Bayesian posterior, or HMM state) with each new observation.
4. **Apply Decision Rule**: Compare the test statistic to the threshold. If it exceeds the threshold, signal a change-point.
5. **Adapt and Iterate**: Continue collecting data and updating test statistics until a change is detected or the sequence ends.

### Example Implementation: Bayesian Change-Point Detection

To implement Bayesian change-point detection:

1. **Initialize**: Set prior probabilities for change-points and initial parameters.
2. **Update Posterior**: For each new observation $$X_n$$, update the posterior probability of change-points:
   $$
   P(\tau = t | X_{1:n}) \propto P(X_{1:t} | \tau = t) \cdot P(X_{t+1:n} | \tau = t) \cdot P(\tau = t)
   $$
3. **Decision Rule**: Signal a change-point if $$P(\tau = t | X_{1:n})$$ exceeds a threshold.

## Real-World Applications

### Quality Control in Manufacturing

In manufacturing, detecting structural changes in processes is crucial for maintaining product quality. Sequential change-point detection helps identify shifts in production parameters, such as mean or variance, which can indicate issues with machinery or materials.

**Example**: Monitoring the thickness of manufactured sheets:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

Using adaptive CUSUM, deviations from the target thickness are accumulated, and a change is signaled when the cumulative sum exceeds a threshold.

### Financial Market Analysis

In finance, structural changes in market conditions can impact trading strategies and risk management. Detecting these changes early allows for timely adjustments to investment portfolios.

**Example**: Monitoring volatility in stock prices:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

Bayesian change-point detection can identify shifts in volatility, informing risk management decisions.

### Environmental Monitoring

Environmental monitoring often involves tracking parameters like temperature, pollution levels, or water quality, which can exhibit structural changes due to natural or anthropogenic factors.

**Example**: Monitoring river water quality:

Before change: $$X_t \sim N(\mu_0, \sigma^2)$$

After change: $$X_t \sim N(\mu_1, \sigma^2)$$

Hidden Markov Models can detect changes in water quality parameters, signaling potential contamination events.

### Biostatistics and Epidemiology

In health sciences, detecting structural changes in disease incidence rates or clinical trial data is crucial for timely intervention and analysis.

**Example**: Monitoring disease outbreak:

Before change: $$X_t \sim Poisson(\lambda_0)$$

After change: $$X_t \sim Poisson(\lambda_1)$$

GLRT can detect increases in incidence rates, indicating the onset of an outbreak.

## Mathematical Details

### Generalized Likelihood Ratio Test (GLRT) Derivation

The GLRT statistic for a change-point at $$k$$ is:

$$
\Lambda_k = \frac{\max_{\theta_0} \prod_{t=1}^k f_0(X_t; \theta_0) \cdot \max_{\theta_1} \prod_{t=k+1}^n f_1(X_t; \theta_1)}{\max_{\theta} \prod_{t=1}^n f(X_t; \theta)}
$$

Taking the logarithm:

$$
\log \Lambda_k = \log \left( \max_{\theta_0} \prod_{t=1}^k f_0(X_t; \theta_0) \right) + \log \left( \max_{\theta_1} \prod_{t=k+1}^n f_1(X_t; \theta_1) \right) - \log \left( \max_{\theta} \prod_{t=1}^n f(X_t; \theta) \right)
$$

The test statistic is:

$$
\Lambda_n = \max_{1 \leq k < n} \log \Lambda_k
$$

### Bayesian Posterior Probability Update

The posterior probability of a change-point at time $$t$$ given the data $$X_{1:n}$$ is updated as follows:

$$
P(\tau = t | X_{1:n}) \propto P(X_{1:t} | \tau = t) \cdot P(X_{t+1:n} | \tau = t) \cdot P(\tau = t)
$$

This involves computing the likelihoods of the data before and after the change, as well as the prior probability of the change-point.

### HMM State Estimation

In HMMs, the probability of being in a particular state at time $$t$$ given the observations $$X_{1:t}$$ is updated using the forward algorithm:

$$
\alpha_t(i) = P(X_{1:t}, S_t = i) = \sum_{j=1}^N \alpha_{t-1}(j) \cdot P(S_t = i | S_{t-1} = j) \cdot P(X_t | S_t = i)
$$

where $$\alpha_t(i)$$ is the forward probability, $$S_t$$ is the state at time $$t$$, and $$N$$ is the number of states.

Sequential detection of switches in models with changing structures is a powerful tool for real-time monitoring and adaptive response in various fields. By understanding the theoretical foundations, employing advanced detection techniques, and implementing practical algorithms, practitioners can effectively manage and respond to structural changes in diverse applications.

## References

- Basseville, M., & Nikiforov, I. V. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice Hall.
- Shiryaev, A. N. (1963). "On optimum methods in quickest detection problems". *Theory of Probability and Its Applications*.
- Page, E. S. (1954). "Continuous inspection schemes". *Biometrika*.
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition". *Proceedings of the IEEE*.
