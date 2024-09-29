---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2020-09-24'
excerpt: Leveraging customer behavior through predictive modeling, the BG/NBD model offers a more accurate approach to demand forecasting in the supply chain compared to traditional time-series models.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_7.jpg
keywords:
- Supply Chain
- Repurchase Model
- Time Series
- Demand Forecasting
- python
seo_description: Explore how using customer behavior and predictive models can improve demand forecasting in the supply chain industry, leveraging the BG/NBD model for better accuracy.
seo_title: Demand Forecasting in Supply Chain Using Customer Behavior
seo_type: article
summary: This article explores the use of customer behavior modeling to improve demand forecasting in the supply chain industry. We demonstrate how the BG/NBD model and the Lifetimes Python library are used to predict repurchases and optimize sales predictions over a future period.
tags:
- Customer Behavior
- python
- Demand Forecasting
- Repurchase Models
title: A Predictive Approach for Demand Forecasting in the Supply Chain Using Customer Behavior Modeling
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

# Outlier removal using IQR method
Q1, Q3 = df['Quantity'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['Quantity'] >= Q1 - 1.5 * IQR) & (df['Quantity'] <= Q3 + 1.5 * IQR)]

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

# Proportionally distribute predicted sales for each product
customer_product['product_proportion'] = customer_product['Quantity'] / customer_product.groupby('Customer ID')['Quantity'].transform('sum')
customer_product['expected_product_sales'] = customer_product['product_proportion'] * summary['expected_sales']
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
        decomposition = seasonal_decompose(product_series, model=model_type, period=30)
        seasonal_indices[product] = decomposition.seasonal.mean()
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

Using a customer-centric approach to predict sales demand in the supply chain offers a more accurate forecasting model than traditional time-series methods. The BG/NBD model provides valuable insights into customer behavior, enabling businesses to forecast demand and optimize inventory more effectively.

The combination of repurchase models, seasonality adjustments, and product-level forecasts enhances the precision of predictions, offering a robust framework for supply chain management.

For further refinement, future models could incorporate external factors like promotions, holidays, or inventory levels to capture even more variability in demand. With ongoing exploration and refinement, this approach can lead to better business decisions and more efficient supply chain operations.
