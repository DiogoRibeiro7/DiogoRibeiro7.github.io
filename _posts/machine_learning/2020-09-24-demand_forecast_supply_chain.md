---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2020-09-24'
excerpt: Leveraging customer behavior through predictive modeling, the BG/NBD model
  offers a more accurate approach to demand forecasting in the supply chain compared
  to traditional time-series models.
header:
  image: /assets/images/headers/photo-logistics-warehouse.jpg
  og_image: /assets/images/headers/photo-logistics-warehouse.jpg
  overlay_image: /assets/images/headers/photo-logistics-warehouse.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-logistics-warehouse.jpg
  twitter_image: /assets/images/headers/photo-logistics-warehouse.jpg
keywords:
- Supply chain
- Repurchase model
- Time series
- Demand forecasting
- Python
permalink: '/machine-learning/demand_forecast_supply_chain/'
redirect_from:
- '/machine learning/demand_forecast_supply_chain/'
seo_description: How customer behavior and predictive models improve supply chain demand forecasting, using the BG/NBD model for better accuracy.
seo_title: Demand Forecasting in Supply Chain Using Customer Behavior
seo_type: article
summary: This article explores the use of customer behavior modeling to improve demand
  forecasting in the supply chain industry. We demonstrate how the BG/NBD model and
  the Lifetimes Python library are used to predict repurchases and optimize sales
  predictions over a future period.
tags:
- Customer Analytics
- Python
title: A Predictive Approach for Demand Forecasting in the Supply Chain Using Customer
  Behavior Modeling
---

## Introduction

In this analysis, I explore a predictive approach to forecasting demand in the **supply chain industry**. The goal is to forecast sales for multiple products over the next N days. Traditional methods such as **ARIMA** and **Prophet** were tested but found inadequate in capturing the complexity and variability of the product lines. As a result, I turned to a customer-centric approach by leveraging the **Lifetimes Python library** to model **customer behavior** for more accurate predictions.

Unlike traditional time series data that focuses solely on product and time, **transactional data** used here incorporates customers, products, and time, offering richer insights into purchasing behavior. By using this customer-level data, we can build a model that predicts demand more accurately, tailored to individual purchasing patterns.

## Transaction Data: A Different Perspective on Forecasting

Instead of purely focusing on product-level forecasting, this method incorporates **transaction-level data**, which includes the customer, product, and transaction time. This approach shifts the focus from a one-dimensional product-time perspective to a more comprehensive customer-product-time framework. By incorporating customer purchasing behavior, we gain valuable insights that improve the **demand forecasting** process.

Even though the ultimate goal is still to predict the future sales volume of products, the structural differences in the data require a different approach. Here, considering **customer-level factors** like purchase frequency, recency, and average spending helps build a more precise forecasting model.

For instance, in this supply chain forecasting problem, individual customer behavior becomes a critical component of the model, providing an added layer of detail beyond traditional time series methods.

## Applying a Repurchase Predictive Model

The **Repurchase Predictive Model** is used to estimate the likelihood of a customer making a purchase within the next N days. This model is based on past behaviors, such as how often customers buy and what products they typically purchase. These insights are invaluable in helping businesses optimize inventory, adjust marketing efforts, and anticipate demand more accurately.

To achieve this, I used the **BG/NBD model (Beta-Geometric/Negative Binomial Distribution)**. This model assumes that each customer has a unique purchase frequency (how often they buy) and a probability of stopping purchases (churn). These characteristics are modeled using Gamma and Beta distributions, respectively. While the customer is active, their purchases follow a **Poisson process**.

The model takes into account:

- **Recency**: How recently the customer made their last purchase.
- **Frequency**: How often they make purchases.
- **Monetary Value**: How much they spend per purchase.

This model allows us to predict the number of future transactions for each customer, which is then aggregated to forecast sales volumes for each product.

## Building the Repurchase Model

### Step 1: Data Preparation and Initialization

```python
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Load and prepare the data
df = transaction_data.copy()
df['trans_date'] = pd.to_datetime(df['trans_date'])

# Clean the data: Remove missing or invalid entries
df.dropna(subset=['Customer ID', 'Quantity'], inplace=True)
df = df[df['Quantity'] > 0]

# Do not remove large orders mechanically as "outliers".
# Large quantities can be genuine demand and may be exactly
# what inventory planning needs to forecast.

# Split data into training and validation sets
cutoff_date = pd.to_datetime('2011-11-30')
train_df = df[df['trans_date'] < cutoff_date]
valid_df = df[(df['trans_date'] >= cutoff_date) & (df['trans_date'] < cutoff_date + pd.Timedelta(days=10))]
```

In this step, the data was cleaned, outliers removed, and split into training and validation sets. Missing values were handled, and only valid transactions were retained.

### Step 2: Fitting the BG/NBD and Gamma-Gamma Models

```python
# Prepare summary data for the BG/NBD model
summary = summary_data_from_transaction_data(train_df, 'Customer ID', 'trans_date', monetary_value_col='Quantity')

# Fit the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.05)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Predict purchases over the next 10 days
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    10, summary['frequency'], summary['recency'], summary['T']
)

# Fit the Gamma-Gamma model for monetary value
ggf = GammaGammaFitter(penalizer_coef=0.02)
ggf.fit(summary['frequency'], summary['monetary_value'])
summary['expected_avg_sales'] = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])
```

Here, the BG/NBD model predicts how likely a customer is to make another purchase, and the Gamma-Gamma model estimates the average sales value per transaction.

### Step 3: Forecasting Expected Sales

```python
# Calculate expected sales by multiplying predicted purchases by average sales
summary['expected_sales'] = summary['predicted_purchases'] * summary['expected_avg_sales']

# Merge predictions with customer-product data
customer_product = train_df.groupby(['Customer ID', 'Description'])['Quantity'].sum().reset_index()

# A customer-level purchase forecast cannot be converted into
# a product-level forecast merely by multiplying historical
# product shares unless those shares are assumed stable.
# Estimate product choice explicitly or forecast product demand
# directly at the product level.
```

This step calculates expected sales per customer, then aggregates those predictions at the product level.

### Step 4: Seasonal Adjustments

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition of product sales
seasonal_indices = {}
for product in daily_sales_pivot.columns:
    product_series = daily_sales_pivot[product]
    model_type = 'multiplicative' if (product_series > 0).all() else 'additive'
    
    if len(product_series.dropna()) >= 90:
        decomposition = seasonal_decompose(
            product_series,
            model=model_type,
            period=30,
        )
        # The mean of an additive seasonal component is approximately
        # zero, so it is not a usable seasonal adjustment factor.
        seasonal_indices[product] = decomposition.seasonal
```

By adjusting sales forecasts based on seasonal patterns, we ensure that periodic trends are factored into the predictions.

### Step 5: Validation of Predictions

```python
# Validate the predictions using actual sales data
actual_sales = valid_df.groupby('Description')['Quantity'].sum().reset_index()
validation_df = product_sales_forecast.merge(actual_sales, on='Description', how='left')
validation_df['actual_sales'].fillna(0, inplace=True)

# Error calculation
mae = mean_absolute_error(validation_df['actual_sales'], validation_df['adjusted_expected_sales'])
rmse = np.sqrt(mean_squared_error(validation_df['actual_sales'], validation_df['adjusted_expected_sales']))
```

The model is validated by comparing the predicted sales with actual sales data from the validation period, calculating error metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Final Thoughts

A customer-level repurchase model and a product-level demand forecast answer different questions. Neither is inherently more accurate than a time-series model; the comparison must be made on the same forecast target, horizon, information set, and validation period. The BG/NBD model provides valuable insights into customer behavior, enabling businesses to forecast demand and optimize inventory more effectively.

The combination of repurchase models, seasonality adjustments, and product-level forecasts enhances the precision of predictions, offering a robust framework for supply chain management.

For further refinement, future models could incorporate external factors like promotions, holidays, or inventory levels to capture even more variability in demand. With ongoing exploration and refinement, this approach can lead to better business decisions and more efficient supply chain operations.


## The forecast target must be explicit

Supply-chain forecasting may target

$$
Y_{s,t+h},
$$

demand for SKU $s$ at horizon $h$, or a hierarchy such as SKU-store, category-store, region, and total demand.

A customer repurchase model estimates a different object:

$$
E[N_i(t,t+h)\mid\mathcal F_t],
$$

the expected number of future transactions for customer $i$.

Turning one into the other requires a product-choice model or an assumption that future product mix follows historical proportions. That assumption should be tested rather than hidden inside a groupby.

## BG/NBD assumptions

BG/NBD models repeat purchasing under assumptions about transaction rates and dropout. They are useful for non-contractual customer behavior, but they are not generic demand models.

The Gamma-Gamma model is designed for positive monetary value under assumptions about independence between transaction frequency and monetary value. Passing item quantity as if it were monetary value changes the model meaning.

If the target is unit demand, model units directly.

## Inventory decisions need distributions

Point forecasts are not enough for safety-stock decisions.

For lead time $L$, inventory policy depends on the distribution

$$
P(
D_{t+1:t+L}
\mid
\mathcal F_t
).
$$

Quantiles, prediction intervals, and service-level loss matter more than RMSE alone.

## Validation must be time ordered

The training period must precede the validation period, and every feature must be available at the forecast origin.

Compare against simple baselines:

- seasonal naive;
- moving average;
- exponential smoothing;
- direct SKU-level models.

A complex customer model should earn its complexity by beating those baselines on the same horizon and loss function.
