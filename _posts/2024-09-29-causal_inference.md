---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2024-09-29'
excerpt: Monotonic constraints are crucial for building reliable and interpretable
  machine learning models. Discover how they are applied in causal ML and business
  decisions.
header:
  image: /assets/images/Causal-Inference-Hero.png
  overlay_image: /assets/images/Causal-Inference-Hero.png
  teaser: /assets/images/Causal-Inference-Hero.png
keywords:
- machine learning
- causal inference
- monotonic constraints
- decision trees
- gradient boosting
- business analytics
seo_description: Learn how monotonic constraints improve predictions in causal machine
  learning and real-world applications like real estate, healthcare, and marketing.
seo_title: Causal Machine Learning with Monotonic Constraints
tags:
- Causal ML
- Monotonic Constraints
- Business Applications
title: 'Causal Insights in Machine Learning: Monotonic Constraints for Better Predictions'
---

![Example Image](/assets/images/causal_inference.jpeg)

## 1. Introduction to Causal Machine Learning

In recent years, machine learning (ML) has been revolutionizing industries by providing predictive power to a wide range of business applications. However, traditional machine learning models, while adept at making accurate predictions, often fall short when tasked with answering causal questions—particularly when these questions involve "what-if" scenarios that impact business decisions.

Enter causal machine learning, an emerging field that bridges the gap between prediction and decision-making by integrating causal relationships into machine learning models. This integration allows businesses to reliably answer questions like, "What will happen if we take action X?" or "Will improving Y lead to better outcomes?" This is critical for making data-driven decisions in areas like marketing, healthcare, finance, and real estate.

One of the key techniques in causal machine learning is the application of **monotonic constraints**. These constraints ensure that relationships between input variables and outcomes behave in ways that align with our knowledge of the world, thereby enabling machine learning models to produce more realistic and actionable predictions.

In this article, we’ll explore:

- The basics of causal ML and why it’s important for business decisions.
- The limitations of traditional machine learning models in answering causal questions.
- The concept of monotonic constraints and how they improve model reliability.
- Step-by-step examples of how to apply monotonic constraints in real-world scenarios.
- The practical implications of using causal models in business applications.

We’ll conclude with a detailed appendix that includes Python code snippets for implementing monotonic constraints using popular libraries like Scikit-learn and CatBoost.

## 2. Traditional Machine Learning vs. Causal Machine Learning

### 2.1 What is Traditional Machine Learning?

Traditional machine learning models are built to identify patterns in data and make predictions based on those patterns. These models are trained on historical data and can be very effective for forecasting future outcomes. For example, a model might predict house prices based on features like square footage, location, and overall condition.

However, traditional ML models have one major limitation: they treat all input features equally and fail to differentiate between **covariate features** (those that cannot be changed) and **treatment features** (those that can be changed by decision-makers). This becomes problematic when we need to answer causal questions, such as "What will happen if we change feature X?"

### 2.2 The Importance of Causal Questions in Business

In a business context, predictive models are often used to guide decision-making. For instance, a company might ask, "If we invest $30,000 in renovating a house, will we be able to sell it for a higher price?" Answering this question requires understanding the causal effect of the renovation (the treatment) on the sale price (the outcome).

Causal questions are particularly important in fields like marketing, healthcare, and finance. Traditional ML models, however, are not equipped to answer these questions because they focus solely on prediction rather than the effect of interventions.

### 2.3 Introduction to Causal Machine Learning

Causal machine learning goes beyond correlation and focuses on understanding cause-and-effect relationships in the data. Unlike traditional models, causal models distinguish between **covariate** and **treatment** features. By explicitly modeling these relationships, causal ML enables businesses to make informed decisions about interventions. 

For example, instead of merely predicting the sale price of a house, a causal model can tell us whether renovating the house will increase its value—and by how much.

## 3. The Concept of Monotonic Constraints

### 3.1 What Are Monotonic Constraints?

Monotonic constraints ensure that certain logical relationships between input features and the target variable are respected by the model. Specifically, a monotonic constraint guarantees that an increase in one or more input features leads to a **non-decreasing** or **non-increasing** effect on the predicted outcome.

For instance, common sense dictates that improving a house's condition should never decrease its value. Yet, traditional ML models might sometimes produce counterintuitive predictions, such as predicting a lower sale price for an improved property. Monotonic constraints prevent such illogical outcomes.

### 3.2 The Benefits of Monotonic Constraints

Monotonic constraints are particularly useful when the relationship between input features and the outcome is **monotonic** but not necessarily linear.

Examples include:
- **Salary and experience**: More experience typically leads to higher salaries, but the relationship may not be linear.
- **Ice cream sales and temperature**: As temperatures rise, ice cream sales generally increase, though not linearly.

Monotonic constraints allow us to incorporate this kind of domain knowledge into models, making them more aligned with real-world expectations.

### 3.3 Types of Monotonic Constraints

Monotonic constraints are expressed as follows:

- **+1**: When a feature increases, the prediction must be greater than or equal to the original prediction.
- **0**: No monotonic constraint is applied (default).
- **-1**: When a feature increases, the prediction must be less than or equal to the original prediction.

By selectively applying these constraints to specific features, we ensure that models behave in ways that align with intuition and domain expertise.

## 4. Implementing Monotonic Constraints in Machine Learning Models

### 4.1 When to Use Monotonic Constraints

Monotonic constraints are particularly useful in scenarios where the relationship between an input feature and the target outcome is well understood. In real estate, for example, improving the condition of a house should always increase its value.

However, not all features have a clear monotonic relationship with the target. For example, the relationship between square footage and house price might be more complex. In such cases, a monotonic constraint might not be appropriate for that feature.

### 4.2 Applying Monotonic Constraints in Decision Trees

Decision trees are popular for regression tasks due to their interpretability. Adding monotonic constraints enhances their reliability in answering causal questions. For Python code examples, refer to the appendix.

### 4.3 Monotonic Constraints in Gradient Boosting Models

Gradient boosting models, like **CatBoost** and **LightGBM**, are known for their performance in machine learning tasks. Both libraries support monotonic constraints, allowing for the integration of domain knowledge without sacrificing model performance.

## 5. Real-World Applications of Causal ML and Monotonic Constraints

### 5.1 Real Estate

Causal ML models with monotonic constraints can help answer questions like:

- Will renovating this property increase its sale price?
- What level of renovation would be most cost-effective?

### 5.2 Healthcare

Causal models help determine the impact of treatments on patient outcomes. Monotonic constraints ensure that more aggressive treatments lead to non-decreasing recovery probabilities.

### 5.3 Marketing and Advertising

Marketers often ask, "What will happen if we increase our ad spend?" A causal ML model with monotonic constraints ensures that increased ad spending leads to non-decreasing sales figures.

## 6. The Benefits and Challenges of Causal ML

### 6.1 Benefits

- **Improved decision-making**: Causal ML models guide businesses based on the impact of interventions, not just correlations.
- **Greater model interpretability**: Monotonic constraints make models behave in ways that make sense to stakeholders.
- **Reduced overfitting**: Monotonic constraints introduce domain knowledge, helping guide predictions and reduce overfitting.

### 6.2 Challenges

- **Selecting appropriate constraints**: Knowing when and to which features to apply monotonic constraints can be difficult.
- **Complexity**: Causal ML models require a deeper understanding of both the business domain and machine learning principles.

## 7. The Future of Causal Machine Learning

Causal machine learning is transforming predictive analytics by incorporating causal relationships and monotonic constraints. This allows businesses to move beyond mere predictions to answering critical "what-if" questions.

Monotonic constraints improve the reliability and interpretability of models, making them not only more accurate but also more trustworthy. As causal ML evolves, it will play an increasingly important role in data-driven decision-making.

## Appendix: Code Examples for Implementing Monotonic Constraints

### A1. Implementing Monotonic Constraints in Scikit-learn

```python
from sklearn.tree import DecisionTreeRegressor

# Define the decision tree with monotonic constraints
decision_tree_with_constraints = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=10,
    monotonic_cst=[1, 1]  # Monotonic constraints applied to both features
)

# Train the model
decision_tree_with_constraints.fit(X_train, y_train)

# Make predictions
predictions = decision_tree_with_constraints.predict(X_test)
```

### A2. Implementing Monotonic Constraints in CatBoost

```python
from catboost import CatBoostRegressor

# Define the CatBoost model with monotonic constraints
catboost_with_constraints = CatBoostRegressor(
    silent=True,
    monotone_constraints={"square feet": 1, "overall condition": 1}
)

# Train the model
catboost_with_constraints.fit(X_train, y_train)

# Make predictions
predictions = catboost_with_constraints.predict(X_test)
```

### A3. Implementing Monotonic Constraints in LightGBM

```python
import lightgbm as lgb

# Define the dataset and model with monotonic constraints
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "regression",
    "monotone_constraints": [1, 1],  # Constraints applied to both features
}

# Train the model
model = lgb.train(params, train_data)

# Make predictions
predictions = model.predict(X_test)
```