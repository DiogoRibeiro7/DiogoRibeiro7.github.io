---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2021-08-01'
excerpt: A step-by-step guide to implementing Linear Regression from scratch using
  the Normal Equation method, complete with Python code and evaluation techniques.
header:
  image: /assets/images/data_science_2.jpg
  og_image: /assets/images/data_science_2.jpg
  overlay_image: /assets/images/data_science_2.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.jpg
  twitter_image: /assets/images/data_science_2.jpg
keywords:
- Linear Regression
- Normal Equation
- Python
- Data Science Interviews
- python
seo_description: Learn how to build a Linear Regression model from scratch using the
  Normal Equation approach. This article covers the theoretical foundations, algorithm
  design, and Python implementation.
seo_title: Building Linear Regression from Scratch Using the Normal Equation
seo_type: article
summary: This article provides a detailed algorithmic approach to building a Linear
  Regression model from scratch, covering theory, Python code implementation, and
  performance evaluation.
tags:
- Linear Regression
- Python
- Normal Equation
- python
title: 'Building Linear Regression from Scratch: A Detailed Algorithmic Approach'
---

One of the most commonly discussed topics in data science interviews—and one of the most practical—is the implementation of machine learning algorithms from scratch. A particularly interesting challenge that surfaces in interviews is building a **Linear Regression** model without relying on external libraries like Scikit-learn or TensorFlow.

Why is this problem so often encountered in interviews? This challenge tests a variety of skills that are critical for data science roles, including:

- Understanding the mathematics behind regression models.
- Knowledge of applied linear algebra and numerical methods.
- Object-Oriented Programming (OOP) practices.
- Designing algorithms from the ground up.
- Competence with numerical computing and performance optimizations.

This article explores a step-by-step method to implement **Linear Regression** from scratch using the **Normal Equation** approach. Along the way, we’ll touch on the theoretical foundations, algorithm design, performance considerations, and techniques for evaluating the model.

## The Fundamentals of Linear Regression

Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features). The relationship is modeled using a straight line (in simple linear regression) or a hyperplane (in multiple linear regression). The mathematical formulation for a linear regression model is given by:

$$ y = X \theta $$

Where:

- $$y$$ is the vector of target values (dependent variable).
- $$X$$ is the matrix of feature values (independent variables).
- $$\theta$$ (theta) is the vector of coefficients (parameters or weights).

The goal of linear regression is to estimate the vector of parameters $$\theta$$ such that the error between the predicted and actual values of $$y$$ is minimized. This is typically done by minimizing the **sum of squared residuals**.

### Solving Linear Regression: The Normal Equation

We will focus on the **Normal Equation** approach to solve for the parameters $$\theta$$. The formula for estimating $$\theta$$ is:

$$ \theta = (X^T X)^{-1} X^T y $$

Where:

- $$X^T$$ is the transpose of the feature matrix $$X$$.
- $$(X^T X)^{-1}$$ is the inverse of the matrix $$X^T X$$.
- $$y$$ is the vector of observed outputs.

This method provides an exact solution without requiring iterative methods like **gradient descent**. However, matrix inversion can be computationally expensive, especially for large datasets, and may introduce numerical instability.

## Step-by-Step Implementation

### Step 1: Set Up the Data

The first step is setting up the dataset. For linear regression, we add an **intercept** term by inserting a column of ones to the feature matrix. This intercept term is critical as it allows the regression model to fit data that doesn't pass through the origin.

```python
import numpy as np

# Add a column of ones for the intercept term
def add_intercept(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))  # Create a column of ones
    return np.hstack((ones, X))  # Horizontally stack the ones with the feature matrix
```

### Step 2: Implement the Normal Equation

Now, we implement the Linear Regression class that encapsulates the functionality of fitting the model using the Normal Equation.

```python
class LinearRegression:
    def __init__(self):
        self.theta = None  # Parameters to be learned
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using the normal equation.
        X: Feature matrix (without intercept term)
        y: Target vector
        """
        X_b = add_intercept(X)  # Add the intercept term
        # Normal Equation: theta = (X^T * X)^-1 * X^T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions based on the learned parameters.
        X: Feature matrix
        Returns predicted values
        """
        X_b = add_intercept(X)  # Add the intercept term
        return X_b.dot(self.theta)
```

### Step 3: Evaluate Model Performance

To evaluate the model, we calculate the **R-squared** metric, which represents the proportion of variance in the dependent variable that can be explained by the independent variables.

The formula for $$R^2$$ is:

$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$

Where:

- $$SS_{res}$$ is the residual sum of squares.
- $$SS_{tot}$$ is the total sum of squares.

```python
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the R-squared value for the model.
    y_true: Actual target values
    y_pred: Predicted target values
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)
```

### Step 4: Testing the Implementation

Let's test our linear regression implementation using synthetic data:

```python
# Create a toy dataset
np.random.seed(42)  # Set random seed for reproducibility
X = 2 * np.random.rand(100, 1)  # 100 data points with one feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with some noise

# Instantiate and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
r2 = r_squared(y, y_pred)
print(f"R-squared: {r2:.4f}")
```

This will print the $$R^2$$ value, which indicates how well the model fits the data. For our synthetic dataset, you can expect a high $$R^2$$ value as the data follows a simple linear relationship.

### A Closer Look at Algorithm Design and OOP

This implementation showcases essential skills in:

- **Matrix Operations**: The Normal Equation involves matrix transposition, multiplication, and inversion, all of which require a solid grasp of linear algebra.
- **Object-Oriented Design**: The class structure ensures that the regression logic is modular and reusable.
- **Performance Considerations**: Although the Normal Equation provides an exact solution, matrix inversion has a time complexity of $$O(n^3)$$, making it inefficient for large datasets. In such cases, iterative methods like **Gradient Descent** are more appropriate.

### Extending the Algorithm: Regularization and Stability

#### Regularization

To prevent issues with multicollinearity (when features are highly correlated), we can use **Ridge Regression**. This introduces a regularization term, modifying the Normal Equation as follows:

$$ \theta = (X^T X + \lambda I)^{-1} X^T y $$

Here, $$\lambda$$ is the regularization strength, and $$I$$ is the identity matrix, ensuring numerical stability by making $$X^T X + \lambda I$$ invertible.

#### Numerical Stability

Matrix inversion can be unstable, especially for large datasets. Libraries like **NumPy** internally use robust algorithms (e.g., **LAPACK**) to handle such operations efficiently, but issues can still arise with poorly conditioned matrices.

### Final Thoughts

Implementing a **Linear Regression** algorithm from scratch offers deep insights into machine learning's mathematical and computational foundations. It demonstrates your ability to build a solution from the ground up, highlighting skills in linear algebra, object-oriented design, and numerical computing.

In interview settings, this exercise not only tests your coding abilities but also your understanding of trade-offs between **simplicity, performance, and scalability**.

Having a solid grasp of these fundamentals will make you a more versatile and effective data scientist—so the next time you're asked to implement linear regression from scratch, you'll know exactly how to approach it!
