---
author_profile: false
categories:
- Healthcare Analytics
- Machine Learning
classes: wide
date: '2020-10-01'
excerpt: A comparison between machine learning models and univariate time series models
  for predicting emergency department visit volumes, focusing on predictive accuracy.
header:
  image: /assets/images/data_science_5.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_5.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_5.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Time Series Models
- Emergency Department Prediction
- Gradient Boosted Machines
- Resource Allocation
- Random Forest
seo_description: This study examines machine learning and univariate time series models
  for predicting emergency department visit volumes, highlighting the superior predictive
  accuracy of random forest models.
seo_title: Comparing Machine Learning and Time Series Models for Predicting ED Visit
  Volumes
seo_type: article
summary: A study comparing machine learning models (random forest, GBM) with univariate
  time series models (ARIMA, ETS, Prophet) for predicting emergency department visits.
  Results show machine learning models perform better, though not substantially so.
tags:
- Emergency Department
- Time Series Forecasting
- Machine Learning
- Gradient Boosted Machines
- Random Forest
title: Machine Learning vs. Univariate Time Series Models in Predicting Emergency
  Department Visit Volumes
---

## 1. Introduction

Accurately predicting emergency department (ED) visit volumes is crucial for efficient hospital management. Emergency departments are the first point of contact for many patients, and fluctuations in patient volumes can create challenges in resource allocation, staffing, and patient care. Knowing when surges in patient visits are likely to occur allows hospital administrators to schedule staff more effectively, manage the availability of beds and equipment, and ensure that adequate resources are in place for potential emergencies.

Historically, predictions of ED visits have relied on univariate time series models that use past visit data to forecast future values. These methods include ARIMA (AutoRegressive Integrated Moving Average), Exponential Smoothing (ETS), and Facebook's Prophet algorithm. However, these models are limited in that they only take into account past visit data and do not include other variables that may influence visit patterns, such as weather conditions, holidays, or the day of the week.

In recent years, machine learning has emerged as a promising alternative to traditional time series models. By incorporating multiple variables and learning complex patterns in the data, machine learning models have the potential to improve predictive accuracy. This study seeks to compare machine learning models, specifically random forests and gradient boosted machines (GBM), with traditional univariate time series models to predict daily ED visits at St. Joseph Mercy Ann Arbor. We hypothesize that machine learning models will outperform univariate time series models by leveraging additional features and capturing more complex relationships in the data.

## 2. Methods

### 2.1. Data Collection

The dataset used for this study consists of daily ED visit records from St. Joseph Mercy Ann Arbor for the years 2017 and 2018. These records capture the number of patients who visited the emergency department each day, along with information such as the day of the week and whether the day was a holiday.

To enhance the predictive power of machine learning models, additional external variables were collected, including weather data such as:

- **Maximum Temperature**: Daily high temperatures, which may influence patient volumes due to weather-related health conditions like heat stroke or hypothermia.
- **Surface (SFC) Pressure**: Atmospheric pressure, which has been associated with health conditions such as headaches or respiratory issues.
- **Humidity**: The amount of moisture in the air, which can exacerbate conditions like asthma or other respiratory problems.
- **Precipitation**: Rain or snow, which may lead to accidents or falls, resulting in increased ED visits.

These weather variables were obtained from the National Weather Service for the same time period. This allowed for the creation of a richer dataset, combining both hospital ED data and external environmental factors that could influence visit patterns.

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

## 3. Results

### 3.1. Predictive Performance

The performance of each model was assessed based on its ability to predict daily ED visit volumes in the 2019 test dataset. The results, measured using RMSE, are summarized below:

- **Random Forest**: The random forest model produced the lowest RMSE, indicating that it was the most accurate at predicting ED visit volumes in the test set. By incorporating both past visit data and external variables, the random forest was able to capture complex patterns in the data that the univariate time series models missed.
  
- **Gradient Boosted Machines (GBM)**: The GBM model also performed well, achieving a slightly higher RMSE than random forest but still outperforming the time series models. GBM's ability to iteratively correct prediction errors made it particularly effective at capturing non-linear relationships in the data.
  
- **Exponential Smoothing (ETS)**: Among the time series models, ETS performed the best. Although it did not include external features, its ability to model seasonality and trends allowed it to produce reasonably accurate predictions.
  
- **Prophet**: The Prophet model performed moderately well, capturing some seasonal patterns but not to the same extent as the machine learning models.
  
- **ARIMA**: The ARIMA model had the highest RMSE, indicating that it struggled to capture the underlying patterns in ED visit volumes. This result suggests that ARIMA's reliance solely on past visit data limited its predictive power in this context.

### 3.2. Feature Importance in Machine Learning Models

An analysis of feature importance in the random forest model revealed several key insights:

- **Day of the Week**: The day of the week was the most important predictor of ED visit volumes, with significantly higher patient loads on weekdays compared to weekends. This finding is consistent with known patterns in hospital operations, where non-emergency visits tend to increase during weekdays when primary care facilities are open.

- **Maximum Temperature**: Weather-related variables, particularly maximum daily temperature, were also important predictors. Extreme heat or cold can lead to health issues that result in higher ED visits, such as heatstroke in the summer or respiratory issues in the winter.

- **Surface Pressure**: Surface atmospheric pressure emerged as another important feature, possibly due to its association with certain health conditions like migraines and respiratory issues.

This feature importance analysis highlights the ability of machine learning models to incorporate and learn from multiple factors, leading to more accurate predictions than time series models that rely solely on past ED visits.

## 4. Discussion

### 4.1. Comparison of Model Performance

The results of this study support the hypothesis that machine learning models can outperform traditional univariate time series models in predicting ED visit volumes. By incorporating additional features such as weather variables and day of the week, the random forest and GBM models were able to capture patterns that the time series models missed. However, the improvement in predictive accuracy was modest, suggesting that time series models still have value, particularly in settings where external data is not available or where simpler models are preferred.

The random forest model was the most accurate overall, likely due to its ability to handle large datasets and capture non-linear relationships between features. The GBM model performed slightly worse but still outperformed the time series models. Surprisingly, ETS performed well relative to the machine learning models, suggesting that time series models with well-defined seasonal patterns can still provide accurate forecasts.

### 4.2. Practical Implications

The ability to accurately predict ED visit volumes has important practical implications for hospital administrators. By anticipating surges in patient load, hospitals can optimize staffing levels, reduce patient wait times, and ensure that sufficient resources are available. The inclusion of weather data and day-of-week effects in machine learning models provides additional insights that can help hospitals prepare for periods of higher demand, such as during extreme weather events or holidays.

### 4.3. Limitations and Future Research

While the results of this study are promising, several limitations should be noted. First, the study relied on data from a single hospital, which may limit the generalizability of the findings. Additionally, the machine learning models could potentially benefit from further tuning and feature engineering. For example, incorporating more granular weather data or including other variables, such as local event data or public health indicators, could improve predictive accuracy.

Future research should explore the use of more advanced machine learning techniques, such as neural networks, which have the potential to capture even more complex patterns in the data. Additionally, testing the models across multiple hospitals and geographic regions would provide a more comprehensive understanding of their generalizability and effectiveness in different contexts.

## 5. Conclusions

This study compared machine learning models (random forest and GBM) with traditional univariate time series models (ARIMA, ETS, and Prophet) for predicting daily ED visit volumes at St. Joseph Mercy Ann Arbor. The results show that machine learning models, particularly random forest, outperformed time series models in terms of predictive accuracy, though the improvement was modest. The day of the week and weather-related variables were found to be important predictors of ED visits, highlighting the advantages of incorporating external factors into predictive models.

While machine learning models provide a slight edge in accuracy, time series models like ETS still performed well and may be sufficient in some cases. Further research is needed to refine these models and explore additional features that could improve predictions. In practice, accurate ED visit forecasts can help hospitals allocate resources more effectively, ultimately improving patient care and operational efficiency.
