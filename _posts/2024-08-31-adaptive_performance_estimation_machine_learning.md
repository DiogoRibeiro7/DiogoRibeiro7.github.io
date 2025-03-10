---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2024-08-31'
excerpt: Explore adaptive performance estimation techniques in machine learning, including
  methods like CBPE and PAPE. Learn how these approaches help monitor model performance
  and detect issues like data drift and covariate shift.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_9.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_9.jpg
keywords:
- Adaptive performance estimation
- Machine learning monitoring
- Cbpe
- Pape
- Data drift detection
- Covariate shift management
- Model performance tracking
seo_description: Learn about adaptive performance estimation in machine learning with
  a focus on methods like CBPE and PAPE. Understand how to manage performance monitoring,
  data drift, and covariate shift for better model outcomes.
seo_title: 'Adaptive Machine Learning Performance Estimation: CBPE and PAPE'
seo_type: article
summary: This article dives into adaptive performance estimation techniques in machine
  learning, comparing methods such as Confidence-Based Performance Estimation (CBPE)
  and Predictive Adaptive Performance Estimation (PAPE). It covers their roles in
  detecting data drift, covariate shift, and maintaining optimal model performance.
tags:
- Machine learning
- Performance monitoring
- Data drift
- Covariate shift
- Cbpe
- Pape
title: 'Adaptive Performance Estimation in Machine Learning: From CBPE to PAPE'
---

Monitoring the performance of Machine Learning (ML) models post-deployment is crucial to ensure they continue to provide value. While performance can be directly calculated when labels are available immediately after prediction, this is often not the case. Labels may be delayed or unavailable, necessitating alternative monitoring strategies. Mature data science teams often monitor changes in input data distribution as a proxy for performance stability. However, data drift is common and does not always negatively impact performance. Existing methods that measure input data drift struggle to accurately quantify its impact, even when used in complex models. This motivates our research into more effective algorithms.

In this article, we introduce the Probabilistic Adaptive Performance Estimation (PAPE) algorithm, designed to estimate the performance of classification models without labels under covariate shift. We will also compare it with its predecessor, Confidence-based Performance Estimation (CBPE).

## Understanding Data Drift and Covariate Shift

Data drift refers to changes in the statistical properties of input data over time. This can happen due to various reasons such as seasonal effects, changes in user behavior, or external events. Covariate shift is a specific type of data drift where the input data distribution $P(X)$ changes, but the conditional distribution of the target given the input $P(Y|X)$ remains the same.

To illustrate these concepts, let's consider a credit default model. This binary classifier uses a continuous feature—credit applicant’s income. Here, $X$ represents the income, and $P(X) = P(x)$. The target $Y$ can be 0 (no default) or 1 (default). The concept $P(y=1|x)$ is the probability of default given the income.

## Example Scenario: Credit Default Model

In our example, the input distribution $P(x)$ and the concept $P(y=1|x)$ are as follows:

- The income is normally distributed with a mean of 80 kEUR.
- The probability of default is high (95%) for low incomes (0 kEUR) and decreases sigmoidally, reaching 5% for incomes above 200 kEUR. The default probability is 50% at around 80 kEUR.

The classifier's predicted probabilities, $\hat{p} = f(x)$, approximate the true concept. A well-trained model should predict probabilities close to the true $P(y=1|x)$, although it may underperform in regions with sparse data.

## From Predicted Probabilities to Performance Metrics

Predicted probabilities are often converted into binary predictions using a threshold. For instance, if $f(x) > 0.5$, the model predicts a default ($\hat{y} = 1$); otherwise, it predicts no default ($\hat{y} = 0$). This thresholding helps in calculating performance metrics like accuracy.

Under covariate shift, while the conditional distribution $P(Y|X)$ remains unchanged, the input distribution $P(X)$ shifts. For example, if the income distribution shifts towards higher values, the distribution of predicted probabilities will also shift.

## Confidence-Based Performance Estimation (CBPE)

The CBPE method relies on the intuition that high-confidence predictions (probabilities close to 0 or 1) are more likely to be correct. For CBPE:

- Raw predicted probabilities from a classifier are calibrated using a calibrator $c$, typically a regressor trained on predicted probabilities and true labels.
- The calibrated probability $c(\hat{p})$ represents the expected probability of a positive label.
- Given calibrated probabilities, performance metrics can be estimated without labels. For instance, if $c(\hat{p})=0.9$ for a positive prediction, we expect 0.9 True Positives and 0.1 False Positives.

## Probabilistic Adaptive Performance Estimation (PAPE)

PAPE extends CBPE to handle shifts in data distributions more effectively. It uses Importance Weighting (IW) to adjust for covariate shifts. The key steps are:

1. Estimate the density ratio $P(X)_{\text{shifted}} / P(X)_{\text{reference}}$ to understand how the input distribution has changed.
2. Train a Density Ratio Estimation (DRE) classifier to distinguish between reference and shifted data, using this classifier to assign weights to observations.
3. The PAPE calibrator uses these weights to adapt to the shifted distribution, ensuring that calibrated probabilities remain accurate under the new distribution.

## Comparing CBPE and PAPE

Let’s consider different scenarios of covariate shift and compare the performance estimates from CBPE and PAPE:

| Covariate Shift Description          | CBPE Accuracy Estimate | PAPE Accuracy Estimate | True Accuracy (Oracle) |
|--------------------------------------|------------------------|------------------------|------------------------|
| Shift in income distribution         | 0.76                   | 0.76                   | 0.76                   |
| Shift in applicant's sex distribution| 0.69                   | 0.74                   | 0.74                   |
| Shift in both income and sex distribution | 0.76                   | 0.87                   | 0.87                   |

In scenarios with strong covariate shifts, PAPE provides more accurate performance estimates than CBPE by adapting to the new data distribution.

## Importance of Calibration

Calibration ensures that predicted probabilities reflect the true likelihood of an event. Perfectly calibrated models will have their predicted probabilities equal to the observed frequencies. This calibration is crucial for accurate performance estimation, especially under covariate shift.

## Conclusion

CBPE and PAPE are robust methods for performance estimation in ML models. While CBPE works well under mild shifts, PAPE excels in scenarios with significant covariate shifts by adapting the calibration process. PAPE's adaptive nature makes it a preferable choice for environments with dynamic data distributions.

For real-world applications, PAPE offers a reliable way to monitor model performance without relying on immediate label availability, making it a valuable tool for data scientists and engineers.
