---
author_profile: false
categories:
- Time-Series
- Machine Learning
classes: wide
date: '2024-10-08'
excerpt: Explore time-series classification in Python with step-by-step examples using
  simple models, the catch22 feature set, and UEA/UCR repository benchmarking with
  statistical tests.
header:
  image: /assets/images/data_science_2.jpg
  og_image: /assets/images/data_science_2.jpg
  overlay_image: /assets/images/data_science_2.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.jpg
  twitter_image: /assets/images/data_science_2.jpg
keywords:
- Time-Series Classification
- Catch22
- Python
- UEA/UCR
seo_description: Learn how to implement time-series classification in Python using
  simple models, catch22 features, and benchmarking with statistical tests using UEA/UCR
  datasets.
seo_title: 'Python Code for Time-Series Classification: Simple Models to Catch22'
seo_type: article
summary: This article provides Python code for time-series classification, covering
  simple models, catch22 features, and benchmarking with UEA/UCR repository datasets
  and statistical significance testing.
tags:
- Python
- Time-Series Classification
- Catch22
- UEA/UCR
title: 'Implementing Time-Series Classification: From Simple Models to Advanced Feature
  Sets'
---

---
title: "Implementing Time-Series Classification: From Simple Models to Advanced Feature Sets"
categories:
- Time-Series
- Machine Learning
tags:
- Python
- Time-Series Classification
- Catch22
- UEA/UCR
author_profile: false
seo_title: "Python Code for Time-Series Classification: Simple Models to Catch22"
seo_description: "Learn how to implement time-series classification in Python using simple models, catch22 features, and benchmarking with statistical tests using UEA/UCR datasets."
excerpt: "Explore time-series classification in Python with step-by-step examples using simple models, the catch22 feature set, and UEA/UCR repository benchmarking with statistical tests."
summary: "This article provides Python code for time-series classification, covering simple models, catch22 features, and benchmarking with UEA/UCR repository datasets and statistical significance testing."
keywords: 
- Time-Series Classification
- Catch22
- Python
- UEA/UCR
classes: wide
---

# Implementing Time-Series Classification: From Simple Models to Advanced Feature Sets

Time-series classification is an essential task in machine learning, with applications ranging from finance to healthcare and industrial monitoring. While deep learning models offer high accuracy in many cases, simpler models based on basic statistical features, like the mean and standard deviation, often provide a solid foundation. For more complex tasks, feature sets like **catch22** can be introduced to capture subtle dynamics in the data.

This article provides Python implementations for:
1. A **simple time-series classification** model using mean and standard deviation.
2. Extending the model with the **catch22 feature set** for added complexity.
3. **Benchmarking and statistical testing** using the UEA/UCR repository, which allows for fair comparisons between different models.

Let’s dive into the code!

## 1. Simple Time-Series Classification using Mean and Standard Deviation

In this first example, we build a basic time-series classifier using only the **mean** and **standard deviation** of the time-series as features. We’ll train a logistic regression model to classify time-series data.

### Code: `simple_model.py`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample time-series dataset (replace with actual data)
def generate_synthetic_data(n_samples=100, n_time_steps=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_time_steps)  # Time-series data
    y = np.random.randint(0, 2, size=n_samples)   # Binary labels
    return X, y

# Feature extraction: mean and standard deviation
def extract_features(X):
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    return np.column_stack([mean, std])

# Main script
if __name__ == "__main__":
    # Generate or load dataset
    X, y = generate_synthetic_data()
    
    # Extract simple features: mean and standard deviation
    X_features = extract_features(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Train a simple linear classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
```

In this example:

- We generate synthetic time-series data using `numpy` and extract the **mean** and **standard deviation** as features.
- The `LogisticRegression` classifier is used to predict the class labels.
- The performance of the model is evaluated with **accuracy**.

## 2. Adding catch22 Features for Enhanced Classification

Next, we extend the basic model by incorporating additional features from the **catch22** feature set. The catch22 set includes 22 canonical features that capture more complex time-series dynamics, such as **periodicity**, **outliers**, and **non-linearity**.

### Code: `catch22_model.py`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catch22 import catch22_all

# Load sample time-series dataset (replace with actual data)
def generate_synthetic_data(n_samples=100, n_time_steps=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_time_steps)  # Time-series data
    y = np.random.randint(0, 2, size=n_samples)   # Binary labels
    return X, y

# Feature extraction: mean, standard deviation, and catch22 features
def extract_features_with_catch22(X):
    basic_features = extract_basic_features(X)
    catch22_features = np.array([catch22_all(ts)['values'] for ts in X])
    return np.hstack([basic_features, catch22_features])

# Extract mean and standard deviation
def extract_basic_features(X):
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    return np.column_stack([mean, std])

# Main script
if __name__ == "__main__":
    # Generate or load dataset
    X, y = generate_synthetic_data()
    
    # Extract features: mean, std, and catch22
    X_features = extract_features_with_catch22(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression classifier
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with catch22 features: {accuracy:.4f}")
```

Here, we:

- Extend the model by adding the **catch22** features.
- Use the same `LogisticRegression` model to classify the time-series and compare its performance with the previous model.

## 3. Benchmarking with the UEA/UCR Repository and Statistical Testing

Finally, we benchmark the models using the **UEA/UCR repository** (or synthetic data) and perform **statistical tests** to compare model performance. We use **cross-validation** to compute accuracy scores and a **paired t-test** to determine whether the performance improvement is statistically significant.

### Code: `benchmark_model.py`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from catch22 import catch22_all

# Load a real dataset from UEA/UCR repository or use synthetic data
def generate_synthetic_data(n_samples=100, n_time_steps=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_time_steps)  # Time-series data
    y = np.random.randint(0, 2, size=n_samples)   # Binary labels
    return X, y

# Feature extraction: mean and standard deviation
def extract_features(X):
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    return np.column_stack([mean, std])

# Feature extraction with catch22
def extract_features_with_catch22(X):
    basic_features = extract_features(X)
    catch22_features = np.array([catch22_all(ts)['values'] for ts in X])
    return np.hstack([basic_features, catch22_features])

# Perform t-test to compare models
def perform_statistical_test(model1_acc, model2_acc):
    t_stat, p_value = ttest_rel(model1_acc, model2_acc)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The performance difference is statistically significant.")
    else:
        print("No statistically significant difference in performance.")

# Main script
if __name__ == "__main__":
    # Generate or load dataset
    X, y = generate_synthetic_data()
    
    # Extract basic features and catch22 features
    X_basic = extract_features(X)
    X_catch22 = extract_features_with_catch22(X)
    
    # Split data for cross-validation testing
    model_basic = LogisticRegression(max_iter=200)
    model_catch22 = LogisticRegression(max_iter=200)
    
    # Cross-validation on basic features
    basic_scores = cross_val_score(model_basic, X_basic, y, cv=5)
    print(f"Mean accuracy (Basic): {np.mean(basic_scores):.4f}")
    
    # Cross-validation on catch22 features
    catch22_scores = cross_val_score(model_catch22, X_catch22, y, cv=5)
    print(f"Mean accuracy (Catch22): {np.mean(catch22_scores):.4f}")
    
    # Statistical significance test
    perform_statistical_test(basic_scores, catch22_scores)
```

In this code:

- We benchmark both the **simple** and **catch22-enhanced** models using **cross-validation**.
- We then compare the results using a **paired t-test** to assess the statistical significance of the performance differences.

## Conclusion

This article demonstrates how to implement time-series classification using **simple models** and extend them with **catch22 features** for added complexity. We also highlight the importance of **benchmarking** and **statistical testing** to validate the improvements made by more complex models.

In many real-world applications, starting with simple features like the **mean** and **standard deviation** can be highly effective. When needed, additional complexity (such as the **catch22** feature set) can be introduced, but the key is to ensure that this added complexity yields **meaningful improvements**.

Feel free to experiment with these scripts on your own datasets and explore how well these methods perform in different domains!
