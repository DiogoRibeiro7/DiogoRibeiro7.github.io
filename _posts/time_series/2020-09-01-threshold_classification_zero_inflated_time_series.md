---
author_profile: false
categories:
- Time Series
classes: wide
date: '2020-09-01'
excerpt: This article explores the use of stationary distributions in time series
  models to define thresholds in zero-inflated data, improving classification accuracy.
header:
  image: /assets/images/headers/photo-data-science-theater.jpg
  og_image: /assets/images/headers/photo-data-science-theater.jpg
  overlay_image: /assets/images/headers/photo-data-science-theater.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-theater.jpg
  twitter_image: /assets/images/headers/photo-data-science-theater.jpg
keywords:
- Time series stationarity
- Zero-inflated data
- Threshold classification
- Statistical modeling
permalink: '/time-series/threshold_classification_zero_inflated_time_series/'
redirect_from:
- '/time series analysis/threshold_classification_zero_inflated_time_series/'
- '/time series/threshold_classification_zero_inflated_time_series/'
seo_description: A method for threshold classification in zero-inflated time series using stationary distributions and parametric modeling.
seo_title: Threshold Classification for Zero-Inflated Series
seo_type: article
summary: A novel approach for threshold classification in zero-inflated time series
  data using stationary distributions derived from time series models. This method
  addresses the limitations of traditional techniques by leveraging parametric distribution
  quantiles for better accuracy and generalization.
tags:
- Statistical Modeling
- Probability
- Time Series
title: A Generalized Approach to Threshold Classification for Zero-Inflated Time Series
  Data Using Stationary Distributions
---

## Abstract

Zero-inflated time series data, characterized by a high frequency of zero values, are common in fields such as meteorology, finance, and traffic analysis. This paper introduces an enhanced methodology for classifying event intensities by leveraging the stationary distribution of a fitted time series model, where applicable. By modeling the time series, obtaining the stationary distribution parameters, and using the quantiles of this distribution for classification, the method ensures that thresholds reflect the true underlying process. Applications to precipitation data, wind speed, and financial transactions demonstrate the effectiveness of this approach in capturing event intensities.

## 1. Time Series Stationarity and Zero-Inflation

### Stationarity in Time Series Data

Weak stationarity requires a constant mean and an autocovariance depending only on lag, not merely constant mean and variance. More importantly, a stationary marginal distribution does not by itself justify useful alert thresholds when zeros and positive values arise from different mechanisms.

However, zero-inflated data complicates traditional analysis due to an overabundance of zeros. Many time series, like precipitation or traffic flow data, have long stretches of zeros interrupted by bursts of non-zero values. If the series is stationary, the excess zeros and the non-zero observations can still be described using a suitable time series model that captures both components.

### Objective

This paper aims to integrate the concept of stationarity into the threshold classification of zero-inflated time series data. By fitting a suitable time series model, we can derive the stationary distribution, use its quantiles for classification, and ensure that the thresholds reflect the true underlying process, providing a more meaningful categorization.

## 2. Challenges in Traditional Threshold Classification

### Zero-Inflated Time Series Data

In zero-inflated time series data, zeros dominate the distribution, making it difficult to apply traditional thresholding techniques like percentiles or standard deviation-based intervals. This is especially problematic for classification tasks, where a meaningful distinction between event intensities (such as low, moderate, or severe precipitation) is needed. A percentile threshold does not inherently assume normality. The real difficulty is that a zero-inflated series often represents a mixture of two processes: whether an event occurs at all and, conditional on occurrence, how large it is.

### Limitations of Existing Methods

Common methods such as percentile and standard deviation thresholds struggle in the face of zero inflation:

- **Percentile-Based Thresholds**: The large number of zeros skews the distribution, leading to uninformative thresholds.
- **Standard Deviation Intervals**: Mean and standard deviation metrics are heavily influenced by zeros, often resulting in unrepresentative or even negative thresholds.

These limitations motivate the need for an alternative methodology that considers the structure and stationarity of the time series.

## 3. Methodology: Fitting Time Series Models and Using Stationary Distributions

### Step 1: Fit a Suitable Time Series Model

The first step involves fitting a time series model that accounts for the zero-inflated nature of the data. Depending on the characteristics of the time series, different models may be appropriate:

- **ARMA/ARIMA Models**: Autoregressive Moving Average (ARMA) or Autoregressive Integrated Moving Average (ARIMA) models can be used for stationary or differenced stationary time series.
- **Zero-Inflated Time Series Models**: For highly zero-inflated data, specialized models like **Zero-Inflated ARMA (ZIARMA)** or **Hurdle Models** can be used to capture both the zero and non-zero components.

The model fitting process yields parameters that describe the time series, such as autoregressive coefficients, moving average terms, and noise variance.

### Step 2: Model occurrence and positive magnitude separately

Let

$$
Z_t=I(Y_t>0).
$$

A two-part model can write

$$
P(Y_t=0)=1-p_t,
$$

and model

$$
Y_t\mid Y_t>0
$$

with a positive-valued distribution. The occurrence probability $p_t$ and positive magnitude can each depend on seasonality, covariates, and past values.

### Step 3: Define thresholds from the operational loss

With the stationary distribution in hand, thresholds for classification can be defined using its quantiles. This ensures that the thresholds are meaningful and reflect the underlying distribution of the time series, including the effect of zeros. For example, thresholds can be set at the 25th, 50th, and 75th percentiles of the stationary distribution:

1. **None**: Zero values, represented by the point mass at zero.
2. **Low**: Values between the 25th percentile and the median (50th percentile).
3. **Moderate**: Values between the median and the 75th percentile.
4. **Intense**: Values above the 75th percentile.

These thresholds capture the full range of data, providing a more nuanced and accurate classification scheme.

### Step 4: Adjust for Domain-Specific Needs

Thresholds can be further refined based on the specific characteristics of the time series or the domain requirements. For example, in the context of precipitation data, expert meteorological knowledge may suggest adjustments to ensure thresholds align with standard categories of rainfall intensity.

## 4. Application to Various Domains

### Precipitation Data

Precipitation data is typically zero-inflated, with many hours showing no rainfall and occasional periods of light, moderate, or heavy precipitation. Fitting an ARMA model to a stationary precipitation series allows us to derive the stationary distribution, including the influence of zeros.

- **Model Fit**: A Zero-Inflated ARMA (ZIARMA) model can be used to capture both zero and non-zero precipitation events.
- **Stationary Distribution**: The stationary distribution combines a point mass at zero with a continuous distribution for positive precipitation values.
- **Thresholds**: Quantile-based thresholds derived from this distribution accurately reflect rainfall intensity, providing a robust classification method.

### Wind Speed Data

Wind speed data often exhibits periods of calm (zero wind speed) interspersed with gusts of varying intensities. An ARMA or Zero-Inflated ARMA model can be used to model wind speed over time, capturing both calm and gusty periods.

- **Model Fit**: An ARMA model is fitted to the wind speed data, and the stationary distribution is obtained.
- **Thresholds**: Quantiles of the stationary distribution are used to define wind speed categories, such as calm, breezy, and windy conditions.

### Financial Transactions

In financial markets, trading volumes often fluctuate between periods of inactivity (zero transactions) and active trading. A Hurdle or Zero-Inflated ARMA model can model both components.

- **Model Fit**: A Hurdle model can separate zero transaction periods from positive transactions.
- **Thresholds**: Quantiles of the stationary distribution for non-zero transactions define thresholds for low, moderate, and high trading volumes.

## 5. Evaluation and Validation

### Performance Metrics

The effectiveness of the stationary distribution approach is evaluated using several performance metrics:

- **Accuracy**: The proportion of correctly classified observations.
- **Precision and Recall**: These metrics are especially important when certain categories (e.g., severe precipitation) are rare but critical.
- **Model Fit Indicators**: Metrics such as the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) evaluate the goodness of fit of the time series model.

### Cross-Dataset Validation

The methodology is tested across multiple datasets to assess generalizability:

- **Precipitation Data**: Datasets from different geographic regions and climates.
- **Wind Speed Data**: Urban and rural wind patterns.
- **Financial Data**: Transaction data from various market conditions.

### Comparison with Baseline Methods

The stationary distribution approach is compared to traditional methods, such as percentile-based thresholds, showing improvements in classification accuracy and the interpretability of thresholds.

## 6. Discussion

### Insights

The use of stationary distributions for threshold classification offers several advantages:

- **Incorporating Time Series Properties**: By leveraging the fitted time series model, the approach captures temporal dependencies and the overall distribution more effectively than simple statistical measures.
- **Adapting to Zero Inflation**: The method naturally handles zero-inflated data, providing more meaningful classifications.

### Limitations

- **Model Complexity**: The process of fitting time series models and deriving stationary distributions requires expertise and computational resources.
- **Assumption of Stationarity**: If the time series is non-stationary, differencing or other transformations may be necessary before applying this approach.

## 7. Conclusion

This paper presents a novel approach to threshold classification in zero-inflated time series data by fitting time series models and using the resulting stationary distributions to define thresholds. This method improves upon traditional classification techniques by incorporating the time series structure, providing more accurate and meaningful event intensity classifications. Applications to domains such as meteorology, wind speed, and finance demonstrate the versatility of the methodology.

---

## Zero-inflated and hurdle models

A zero-inflated model assumes some zeros arise from a structural-zero process while the count process can also generate zeros. A hurdle model separates zero versus positive occurrence and then uses a zero-truncated positive distribution. Those assumptions are different. The choice should follow the mechanism. For time series, dependence can enter both components:

$$
\operatorname{logit}(p_t)
=
x_t^\top\beta
+
\phi Z_{t-1},
$$

while the positive magnitude model can have its own temporal structure. Differencing the raw series is not a generic solution to zero inflation.
