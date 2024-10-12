---
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-10-01'
excerpt: 'Understanding coverage probability in statistical estimation and prediction: its role in constructing confidence intervals and assessing their accuracy.'
header:
  image: /assets/images/data_science_9.jpg
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_9.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_9.jpg
  twitter_image: /assets/images/data_science_1.jpg
keywords:
- Coverage probability
- Confidence interval
- Nominal confidence level
- Prediction intervals
- Statistical estimation
seo_description: A detailed explanation of coverage probability, its role in statistical estimation theory, and its relationship to confidence intervals and prediction intervals.
seo_title: Coverage Probability in Statistical Estimation Theory
seo_type: article
summary: In statistical estimation theory, coverage probability measures the likelihood that a confidence interval contains the true parameter of interest. This article explains its importance in statistical theory, prediction intervals, and nominal coverage probability.
tags:
- Confidence intervals
- Statistical theory
- Estimation
title: 'Coverage Probability: Explained'
---

## What is Coverage Probability?

Coverage probability is a fundamental concept in **statistical estimation theory**, referring to the probability that a confidence interval or confidence region contains the true value of the parameter of interest. In essence, it quantifies the likelihood that the confidence interval, constructed based on a sample, will enclose the unknown population parameter (e.g., a mean or variance) under repeated sampling.

Mathematically, for a given confidence interval $$C(\mathbf{X})$$, where $$\mathbf{X}$$ represents the data, the coverage probability is defined as:

$$
P(\theta \in C(\mathbf{X})) = \text{coverage probability}
$$

Here, $$\theta$$ denotes the true parameter value, and $$C(\mathbf{X})$$ represents the confidence interval computed from a sample of data. In this context, the coverage probability is often interpreted over repeated sampling, where the goal is to assess how frequently the interval would contain the true parameter across many hypothetical repetitions of the experiment.

## Long-Run Frequency Interpretation

In practice, the **coverage probability** is evaluated in terms of **long-run frequency**. This approach assumes that if an experiment were repeated infinitely, a certain proportion of the constructed confidence intervals would contain the true parameter. For example, if the confidence level is 95%, then 95 out of every 100 intervals constructed would be expected to enclose the true parameter.

This long-run interpretation is essential in statistical inference, as it provides a probabilistic measure of the reliability of the interval estimation process. However, it is crucial to note that this interpretation is hypothetical and depends on the underlying assumptions of the model.

## Coverage Probability in Statistical Prediction

In the context of **statistical prediction**, coverage probability extends to **prediction intervals**. A prediction interval is used to estimate a future out-of-sample observation, rather than a fixed parameter. The coverage probability in this scenario refers to the probability that the prediction interval will contain the value of the future random variable.

Just as with confidence intervals, the coverage probability of a prediction interval reflects how often, in repeated experiments, the interval would include the future value of the random variable of interest. If the actual coverage probability is 90%, then 90% of the prediction intervals constructed should contain the future observation in repeated sampling.

## Nominal Coverage Probability vs. True Coverage Probability

When constructing a confidence interval, the analyst specifies a fixed level of certainty—known as the **nominal coverage probability** or **confidence level**. Commonly, this is set at 0.95 (95%), meaning that the confidence interval should contain the true parameter 95% of the time under repeated sampling.

The nominal coverage probability, however, is not always equal to the **true coverage probability**, which reflects the actual proportion of intervals that contain the true parameter. If all assumptions used to derive the confidence interval are satisfied, the nominal and true coverage probabilities will coincide. However, if the assumptions are violated, the true coverage probability may deviate from the nominal value.

### Key Definitions:

- **Nominal Coverage Probability**: The pre-specified probability (e.g., 95%) set by the analyst when constructing a confidence interval.
- **True Coverage Probability**: The actual probability that the confidence interval contains the parameter, accounting for any violations of assumptions.

### Conservative vs. Anti-Conservative Intervals

When the **true coverage probability** exceeds the **nominal coverage probability**, the confidence interval is said to be **conservative**. This means the interval is wider than necessary, and it contains the parameter more often than expected by the nominal confidence level.

Conversely, if the **true coverage probability** is lower than the nominal value, the interval is considered **anti-conservative** or **permissive**. This implies that the interval is too narrow and fails to contain the parameter as frequently as intended.

### Example

Consider a clinical study where the goal is to estimate the mean remission duration for cancer patients after treatment. A 95% confidence interval is constructed to capture the mean remission duration. Here, the **coverage probability** would indicate how often, across many such studies, the confidence interval captures the true mean remission time. If the interval construction method is correct, the nominal and true coverage probabilities would match, ensuring accurate inference.

## The Role of Hypothetical Repetitions

The interpretation of **coverage probability** is inherently tied to the notion of **hypothetical repetitions**. In this framework, the data collection and analysis procedure are assumed to be repeatable, and independent data sets from the same probability distribution are considered. For each of these hypothetical data sets, a confidence interval is calculated, and the fraction of intervals that contain the true parameter value is used to determine the coverage probability.

This concept reinforces the understanding that coverage probability is not a one-time guarantee for a specific data set. Instead, it is a property of the statistical procedure under repeated sampling from the same population.

## Conclusion

Coverage probability plays a crucial role in statistical inference by providing a probabilistic framework for evaluating the accuracy of confidence and prediction intervals. While the **nominal coverage probability** is often set by the analyst, the **true coverage probability** depends on the validity of the underlying assumptions. Understanding the distinction between nominal and true coverage probabilities is essential for ensuring that statistical conclusions are reliable and accurately reflect the uncertainty inherent in parameter estimation.
