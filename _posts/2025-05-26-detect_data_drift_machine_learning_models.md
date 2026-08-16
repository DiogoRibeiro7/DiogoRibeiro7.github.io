---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2025-05-26'
excerpt: Data drift is one of the primary threats to model reliability in production.
  This article walks through how to detect it using both statistical techniques and
  modern monitoring tools.
header:
  image: /assets/images/data_science_2.avif
  og_image: /assets/images/data_science_2.avif
  overlay_image: /assets/images/data_science_2.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.avif
  twitter_image: /assets/images/data_science_2.avif
keywords:
- Data drift detection
- Kullback-leibler divergence
- Population stability index
- Chi-square test
- Evidently ai
- Nannyml
permalink: '/machine-learning/detect_data_drift_machine_learning_models/'
redirect_from:
- '/machine learning/model monitoring/detect_data_drift_machine_learning_models/'
- '/machine learning/detect_data_drift_machine_learning_models/'
seo_description: How to detect data drift with KL Divergence and PSI, and tools like NannyML and Evidently AI for maintaining accuracy in production.
seo_title: 'Detecting Data Drift in Machine Learning: Methods and Tools'
seo_type: article
summary: Explore how to detect data drift in machine learning systems, including core
  techniques like KL Divergence, PSI, and Chi-square tests, as well as practical tools
  like NannyML and Evidently AI.
tags:
- Data Drift
- Model Monitoring
- Hypothesis Testing
- MLOps
title: How to Detect Data Drift in Machine Learning Models
related_posts:
- /machine-learning/data_drift_concept_drift_understanding_differences_implications/
- /machine-learning/detect_multivariate_data_drift/
- /machine-learning/monitoring_drift/
- /machine-learning/prevalence_shift_base_rate_drift/
---

Data drift—the change in the distribution of input data over time—is one of the most common and insidious causes of model performance degradation in production environments. A model trained on a historical dataset might face real-world data that no longer reflects past patterns, leading to inaccurate predictions and diminished business value.

Detecting data drift early is critical to maintaining model integrity. This article provides a practical guide to identifying drift using both classical statistical tests and modern machine learning tools designed for production systems.

## What Is Data Drift?

Data drift, also known as **covariate shift**, occurs when the statistical properties of the features in the input data change over time. This does not necessarily mean the target variable changes (that's concept drift), but it does mean the inputs the model relies on have shifted in ways that can undermine its validity.

For example, a model trained on retail customer behavior during the holiday season may perform poorly in the summer due to changes in purchasing patterns, even though the target variable (e.g., purchase made: yes/no) remains consistent.

## Statistical Techniques for Drift Detection

Several statistical methods can be used to compare the distribution of features in incoming (production) data with those in the training or validation dataset. Below are the most commonly used techniques:

### 1. Kullback-Leibler (KL) Divergence

KL Divergence measures how one probability distribution diverges from a second, reference probability distribution. For discrete variables:

$$
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

Here, $P$ is the observed distribution in production, and $Q$ is the reference distribution from training. A KL divergence of 0 indicates no drift, while higher values suggest significant differences.

KL Divergence is sensitive to zero values, so smoothing techniques or binning are often required when computing it in practice.

### 2. Population Stability Index (PSI)

PSI is widely used in industries like finance and insurance to monitor scorecard model stability. It quantifies changes in the distribution of a variable across two datasets.

The formula is:

$$
\text{PSI} = \sum_{i=1}^{n} (P_i - Q_i) \log \frac{P_i}{Q_i}
$$

Where:

- $P_i$ is the proportion of records in bin $i$ from the production data.  
- $Q_i$ is the proportion in bin $i$ from the training data.

**Interpretation of PSI values**:

- < 0.1: No significant change  
- 0.1–0.25: Moderate drift  
- > 0.25: Significant drift  

### 3. Chi-Square Test

The Chi-square test assesses whether observed frequency distributions differ from expected distributions. It's effective for categorical variables and can be used to compare feature value distributions between datasets.

The test statistic is:

$$
\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}
$$

Where $O_i$ and $E_i$ are observed and expected counts, respectively. A low p-value indicates a statistically significant difference between distributions, suggesting drift.

### 4. Kolmogorov-Smirnov (K-S) Test

The K-S test is a non-parametric method that measures the maximum distance between the cumulative distributions of two datasets. It's particularly suited for continuous numerical features. A significant K-S statistic indicates that the feature's distribution has changed.

## Practical Tools for Drift Detection

In addition to statistical methods, there are open-source tools designed to monitor and report drift automatically within machine learning pipelines.

### NannyML

NannyML is a powerful open-source Python library designed for post-deployment data and performance monitoring without requiring actual labels.

**Features**:

- Detects data drift, concept drift, and performance degradation.  
- Supports unlabelled data monitoring using confidence-based estimators.  
- Generates comprehensive visual reports and dashboards.  

NannyML is especially useful in high-stakes settings where labels are delayed or expensive to obtain.

GitHub: [https://github.com/NannyML/nannyml](https://github.com/NannyML/nannyml)

### Evidently AI

Evidently AI is a monitoring tool that creates rich dashboards and reports to monitor model performance, data quality, and drift.

**Features**:

- Real-time and batch monitoring of models.  
- Pre-built statistical tests for drift, outliers, and data quality.  
- Interactive visualizations for exploratory drift analysis.  

It integrates easily into both local development and production pipelines, making it suitable for both experimentation and operations.

GitHub: [https://github.com/evidentlyai/evidently](https://github.com/evidentlyai/evidently)

## Best Practices for Drift Monitoring

- **Baseline Everything**: Always capture and log the training dataset distribution as a reference for comparison.  
- **Monitor Regularly**: Set automated checks (e.g., daily, weekly) to evaluate feature distributions.  
- **Track Key Features**: Prioritize monitoring features that have high feature importance or are historically unstable.  
- **Visualize Changes**: Use tools like Evidently AI to graphically assess where and how drift is occurring.  
- **Respond to Drift**: Define thresholds and triggers for retraining or alerting based on drift severity.  

## Final Thoughts

Detecting data drift is not just about protecting model accuracy—it’s about preserving the integrity of decisions made from your ML system. By combining statistical rigor with modern monitoring tools, teams can catch distributional shifts early and take proactive steps before model performance deteriorates.

Data is never static, and neither should your monitoring strategy be. Embrace drift detection as a continuous process, not a one-time diagnostic. In doing so, you ensure your models remain as adaptive as the environments they serve.

## References

- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset Shift in Machine Learning*. MIT Press.
- Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79-86.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 70(2), 129-133.
- Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association*, 46(253), 68-78.
