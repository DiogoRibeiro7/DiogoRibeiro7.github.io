---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2025-06-09'
excerpt: Learn specialized feature engineering techniques to make time series data
  more predictive for machine learning models.
header:
  image: /assets/images/data_science_12.jpg
  og_image: /assets/images/data_science_12.jpg
  overlay_image: /assets/images/data_science_12.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_12.jpg
  twitter_image: /assets/images/data_science_12.jpg
keywords:
- Time series features
- Lag variables
- Rolling windows
- Seasonality
- Python
seo_description: Discover practical methods for crafting informative features from
  time series data, including lags, moving averages, and trend extraction.
seo_title: Feature Engineering for Time Series Data
seo_type: article
summary: This post explains how to engineer features such as lagged values, rolling
  statistics, and seasonal indicators to improve model performance on sequential data.
tags:
- Feature engineering
- Time series
- Machine learning
- Forecasting
- Python
title: Crafting Time Series Features for Better Models
---

Time series data contains rich temporal dynamics—trends, seasonality, cycles, shocks, and evolving variance—that standard tabular features often miss. By crafting features that explicitly encode these patterns, you empower models (from ARIMA to gradient boosting and deep learning) to learn more nuanced signals and deliver significantly more accurate forecasts. This article dives deep into both classic and cutting-edge feature engineering techniques for time series, complete with conceptual explanations and code snippets in Python.

## 1. Lagged Variables

### 1.1 Basic Lag Features

The simplest way to give your model “memory” is to include previous observations as new predictors. For a series $$y_t$$, you might create:  

```python
df['lag_1'] = df['y'].shift(1)
df['lag_7'] = df['y'].shift(7)   # weekly lag
```

These features let the model learn autoregressive relationships: how yesterday or last week influences today.

### 1.2 Distributed Lags

Rather than picking arbitrary lags, you can create a range of lags and let the model pick which matter:

```python
for lag in range(1, 15):
    df[f'lag_{lag}'] = df['y'].shift(lag)
```

When used with regularized models (e.g., Lasso or tree-based methods), the model will zero-out irrelevant lags automatically.

## 2. Rolling Statistics

Rolling (or moving) statistics smooth the data, revealing local trends and variability.

### 2.1 Moving Averages

A $k$-period rolling mean captures local trend:

```python
df['roll_mean_7'] = df['y'].rolling(window=7).mean()
```

Experiment with short and long windows (7, 30, 90 days) to capture different granularities.

### 2.2 Rolling Variance and Other Aggregations

Volatility often matters as much as level. Rolling standard deviation, quantiles, min/max, and even custom functions can be computed:

```python
df['roll_std_14'] = df['y'].rolling(14).std()
df['roll_max_30'] = df['y'].rolling(30).max()
df['roll_q25_30'] = df['y'].rolling(30).quantile(0.25)
```

These features help the model detect regime changes or anomalous behavior.

## 3. Seasonal Indicators

Even simple flags for calendar units can boost performance dramatically.

### 3.1 Calendar Features

Extract month, day of week, quarter, and more:

```python
df['month'] = df.index.month
df['dow']   = df.index.dayofweek   # 0=Monday, …, 6=Sunday
df['quarter'] = df.index.quarter
```

### 3.2 Cyclical Encoding

Treating these as numeric can introduce artificial discontinuities at boundaries (e.g., December→January). Instead encode them cyclically with sine/cosine:

```python
df['sin_month'] = np.sin(2*np.pi * df['month']/12)
df['cos_month'] = np.cos(2*np.pi * df['month']/12)
```

This preserves the circular nature of time features.

## 4. Trend and Seasonality Decomposition

Decompose the series into trend, seasonal, and residual components (e.g., with STL) and use them directly.

```python
from statsmodels.tsa.seasonal import STL
stl = STL(df['y'], period=365)
res = stl.fit()
df['trend']    = res.trend
df['seasonal'] = res.seasonal
df['resid']    = res.resid
```

Feeding the trend and seasonal components separately lets your model focus on each pattern in isolation.

## 5. Fourier and Spectral Features

To capture complex periodicities without manually creating many dummies, build Fourier series terms:

```python
def fourier_terms(series, period, K):
    t = np.arange(len(series))
    terms = {}
    for k in range(1, K+1):
        terms[f'sin_{period}_{k}'] = np.sin(2*np.pi*k*t/period)
        terms[f'cos_{period}_{k}'] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(terms, index=series.index)

fourier_df = fourier_terms(df['y'], period=365, K=3)
df = pd.concat([df, fourier_df], axis=1)
```

This approach succinctly encodes multiple harmonics of yearly seasonality (or whatever period you choose).

## 6. Date-Time Derived Features

Beyond basic calendar units, derive:

* Time since event: days since last promotion, hours since last maintenance.
* Cumulative counts: how many times a threshold was breached to date.
* Time to next event: days until holiday or known scheduled event.

These features are domain-specific but often highly predictive.

## 7. Holiday and Event Effects

Special days often trigger spikes or drops. Incorporate known events:

```python
import holidays
us_holidays = holidays.US()
df['is_holiday'] = df.index.to_series().apply(lambda d: d in us_holidays).astype(int)
```

You can also add “days until next holiday” and “days since last holiday” to capture lead/lag effects.

## 8. Interaction and Lagged Interaction Features

Combine time features with lagged values to model varying autocorrelation:

```python
df['lag1_sin_month'] = df['lag_1'] * df['sin_month']
```

Such interactions can help the model learn that the effect of yesterday’s value depends on the season or trend.

## 9. Window-Based and Exponential Weighted Features

Instead of fixed‐window rolling stats, use exponentially weighted moving averages (EWMA) to prioritize recent observations:

```python
df['ewm_0.3'] = df['y'].ewm(alpha=0.3).mean()
```

Experiment with different decay rates to find the optimal memory length.

## 10. Domain-Specific Signals

In finance: technical indicators (RSI, MACD); in retail: days since last promotion; in IoT: time since device reboot. Leverage your domain knowledge to craft bespoke features that capture critical drivers.

## 11. Feature Selection and Validation

With hundreds of engineered features, guard against overfitting:

* Correlation analysis: drop highly collinear features.
* Model-based importance: use tree-based methods to rank features.
* Regularization: L1/L2 penalties to zero out irrelevant predictors.
* Cross-validation: time-aware CV (e.g., expanding window) to test generalization.

## 12. Integrating Engineered Features

Finally, assemble your features into a single DataFrame (aligning on time index), handle missing values (common after shifts/rolls), and feed into your chosen model:

```python
df.dropna(inplace=True)
X = df.drop(columns=['y'])
y = df['y']
```

Use pipelines (e.g., scikit-learn’s Pipeline) to keep preprocessing, feature engineering, and modeling reproducible and version-controlled.

By thoughtfully engineering temporal features—from simple lags to spectral and event-driven signals—you unlock hidden structures in your data. Paired with rigorous validation and domain expertise, these techniques can transform raw time series into powerful predictors, elevating model performance across forecasting tasks.
