---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2020-10-01'
excerpt: A comparison between machine learning models and univariate time series models
  for predicting emergency department visit volumes, focusing on predictive accuracy.
header:
  image: /assets/images/headers/photo-factory.jpg
  og_image: /assets/images/headers/photo-factory.jpg
  overlay_image: /assets/images/headers/photo-factory.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-factory.jpg
  twitter_image: /assets/images/headers/photo-factory.jpg
keywords:
- Time series models
- Emergency department prediction
- Gradient boosted machines
- Resource allocation
- Random forest
permalink: '/machine-learning/time_series_models_predicting_emergency/'
redirect_from:
- '/machine learning/time_series_models_predicting_emergency/'
seo_description: Comparing machine learning and univariate time series models for predicting emergency department visit volumes, with random forest performing best.
seo_title: 'Forecasting ED Visit Volumes: ML vs Time Series'
seo_type: article
summary: A study comparing machine learning models (random forest, GBM) with univariate
  time series models (ARIMA, ETS, Prophet) for predicting emergency department visits.
  Results show machine learning models perform better, though not substantially so.
tags:
- Time Series
- Machine Learning
- Decision Trees
title: Machine Learning vs. Univariate Time Series Models in Predicting Emergency
  Department Visit Volumes
---

## 1. Introduction

Accurately predicting emergency department (ED) visit volumes is crucial for efficient hospital management. Emergency departments are the first point of contact for many patients, and fluctuations in patient volumes can create challenges in resource allocation, staffing, and patient care. Knowing when surges in patient visits are likely to occur allows hospital administrators to schedule staff more effectively, manage the availability of beds and equipment, and ensure that adequate resources are in place for potential emergencies.

Historically, predictions of ED visits have relied on univariate time series models that use past visit data to forecast future values. These methods include ARIMA (AutoRegressive Integrated Moving Average), Exponential Smoothing (ETS), and Facebook's Prophet algorithm. However, these models are limited in that they only take into account past visit data and do not include other variables that may influence visit patterns, such as weather conditions, holidays, or the day of the week.

In recent years, machine learning has emerged as a promising alternative to traditional time series models. By incorporating multiple variables and learning complex patterns in the data, machine learning models have the potential to improve predictive accuracy. A fair comparison between machine-learning and time-series models must hold the forecast target, horizon, training window, and information set fixed. A model with weather and calendar covariates is not directly comparable with a deliberately univariate baseline unless the purpose is explicitly to measure the value of those additional predictors.

## 2. Methods

### 2.1. Data Collection

This article does not include the underlying hospital dataset or a reproducible data source for the claimed 2017–2019 experiment. The previous version nevertheless presented institution-specific results as though they had been reproduced here. Those claims should not be treated as evidence. The defensible content is the design of a forecasting comparison.

### 2.2. Time Series Models

For comparison, three univariate time series models were used as baselines:

- **ARIMA (AutoRegressive Integrated Moving Average)**: A well-established time series model that combines autoregressive and moving average components to capture trends and seasonality in time series data. ARIMA assumes that future values can be predicted as a linear function of past observations and residuals.

- **Exponential Smoothing (ETS)**: ETS models capture trend and seasonality by applying smoothing to past observations. ETS includes three components—error, trend, and seasonality—allowing it to adapt to different time series patterns.

- **Facebook's Prophet**: Prophet is a flexible time series forecasting tool developed by Facebook. It is designed to handle time series data with seasonality and holidays, making it a useful baseline for this study. Prophet can automatically detect yearly, weekly, and daily seasonality and accommodate missing data and irregular trends.

These models were trained using the daily ED visit data from 2017 to 2018, without additional external features, and then tested on out-of-sample data from 2019 to evaluate their predictive accuracy.

### 2.3. Machine Learning Models

Two machine learning models were used for comparison with the univariate time series models:

- **Random Forest**: A powerful ensemble learning technique that constructs multiple decision trees during training and outputs the mean prediction (for regression) from individual trees. Random forests handle large amounts of data and can capture non-linear relationships between features, making them well-suited for this task. In this study, the random forest model was trained using not only past ED visit data but also external features such as day of the week and weather variables.

- **Gradient Boosted Machines (GBM)**: GBM is another ensemble learning method that builds models sequentially, with each tree trying to correct the errors of the previous one. GBM models are known for their high accuracy but can be prone to overfitting if not properly tuned. Similar to random forest, the GBM model in this study used both past visit data and external features to make predictions.

### 2.4. Model Training and Testing

The models were trained on data from 2017 to 2018 and then tested on data from 2019 to assess their out-of-sample predictive accuracy. This approach ensured that the models were evaluated on data they had not previously seen, providing a more accurate measure of their generalization capabilities.

To assess model performance, the **root mean squared error (RMSE)** was used as the primary evaluation metric. RMSE is defined as the square root of the average squared differences between predicted and actual values. A lower RMSE indicates higher predictive accuracy.

### 2.5. Feature Importance in Machine Learning Models

One advantage of machine learning models over traditional time series models is the ability to assess the importance of different features in making predictions. By analyzing the trained random forest and GBM models, we can determine which variables had the greatest influence on ED visit predictions. This feature importance analysis provides insights into which factors, beyond past ED visits, are most predictive of patient volume patterns.

## 3. What a Reproducible Comparison Should Report

A valid benchmark should publish, for every model and forecast horizon:

- the exact training and test dates;
- all features available at each forecast origin;
- hyperparameter-tuning procedure;
- rolling-origin or blocked evaluation design;
- point and probabilistic forecast metrics;
- uncertainty in score differences.

For point forecasts, MAE and RMSE answer different loss questions. For staffing decisions, quantile loss can be more useful because underforecasting and overforecasting may have asymmetric costs. A strong baseline set includes:

$$
\hat y_{t+h}=y_t
$$

for persistence where meaningful, seasonal naive forecasts such as

$$
\hat y_{t+h}=y_{t+h-s},
$$

and a well-tuned exponential-smoothing or regression-with-ARIMA-errors model. Machine-learning models should use lagged demand features and external covariates only when those covariates are genuinely known at the forecast origin.

### Feature importance is not causal explanation

Random-forest or boosting feature importance measures predictive contribution inside a fitted model. They do not establish that temperature, pressure, or weekday **causes** ED volume changes. Correlated predictors can split importance unpredictably, and impurity-based importance can be biased toward high-cardinality or noisy continuous variables.

Permutation importance or SHAP values can describe model dependence, but causal interpretation still requires a causal design.

## 4. Discussion

### 4.1. Comparison of Model Performance

Without the original data, fitted models, and numerical results, this article cannot support a claim that random forest or GBM outperformed ARIMA, ETS, or Prophet. The appropriate conclusion is methodological: flexible models can exploit external predictors and nonlinear interactions, while statistical time-series models provide strong structured baselines. Which performs better is an empirical question for a reproducible rolling-origin benchmark.

### 4.2. Practical Implications

The ability to accurately predict ED visit volumes has important practical implications for hospital administrators. By anticipating surges in patient load, hospitals can optimize staffing levels, reduce patient wait times, and ensure that sufficient resources are available. The inclusion of weather data and day-of-week effects in machine learning models provides additional insights that can help hospitals prepare for periods of higher demand, such as during extreme weather events or holidays.

### 4.3. Limitations and Future Research

While the results of this study are promising, several limitations should be noted. First, the study relied on data from a single hospital, which may limit the generalizability of the findings. Additionally, the machine learning models could potentially benefit from further tuning and feature engineering. For example, incorporating more granular weather data or including other variables, such as local event data or public health indicators, could improve predictive accuracy.

Future research should explore the use of more advanced machine learning techniques, such as neural networks, which have the potential to capture even more complex patterns in the data. Additionally, testing the models across multiple hospitals and geographic regions would provide a more comprehensive understanding of their generalizability and effectiveness in different contexts.

## 5. Conclusions

Emergency-department forecasting should be evaluated as an operational forecasting problem, not as a contest between “machine learning” and “time series.” The strongest model is the one that produces calibrated, reproducible forecasts at the horizons that staffing and capacity decisions actually use. Claims of superiority require the data and results to be published or reproducible.

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
