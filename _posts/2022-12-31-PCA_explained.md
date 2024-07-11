---
title: "Understanding PCA: A Step-by-Step Guide to Principal Component Analysis"
categories:
- Data Science
- Machine Learning
tags:
- PCA
- Feature Selection
- Dimensionality Reduction
author_profile: false
---

At the end of this blog, you can (visually) explain the variance in your data, select the most informative features, and create insightful plots. We will go through the following topics:

1. Feature Selection vs. Extraction
2. Dimension Reduction Using PCA
3. Explained Variance and the Scree Plot
4. Loadings and the Biplot
5. Extracting the Most Informative Features
6. Outlier Detection

## Gentle Introduction to PCA

The main purpose of Principal Component Analysis (PCA) is to reduce dimensionality in datasets by minimizing information loss. In general, there are two manners to reduce dimensionality: Feature Selection and Feature Extraction. The latter is used, among others, in PCA where a new set of dimensions or latent variables are constructed based on a (linear) combination of the original features. 

In the case of feature selection, a subset of features is selected that should be informative for the task ahead. Reducing dimensionality is important for several reasons such as reducing complexity, improving run time, determining feature importance, visualizing class information, and preventing the curse of dimensionality. This means that, for a given sample size, and above a certain number of features, the classifier will degrade in performance rather than improve. In most cases, a lower-dimensional space results in more accurate mapping and compensates for the “loss” of information.

In the next section, we will explain how to choose between feature selection and feature extraction techniques because there are reasons to choose one over the other.

## Feature Selection

Feature selection is necessary for several situations:

1. In case the features are not numeric (e.g., strings).
2. In case you need to extract meaningful features.
3. To keep measurements intact (a transformation would make a linear combination of measurements and the unit to be lost).

A disadvantage is that feature selection procedures require a search strategy and/or objective function to evaluate and select the potential candidates. For example, it may require a supervised approach with class information to perform a statistical test or a cross-validation approach to select the most informative features. Nevertheless, feature selection can also be done without class information, such as by selecting the top N features based on variance (higher is better).

## Feature Extraction

Feature extraction approaches can reduce the number of dimensions while minimizing the loss of information. To do this, we need a transformation function \( y = f(x) \). In the case of PCA, the transformation is limited to a linear function which we can rewrite as a set of weights that make up the transformation step: \( y = Wx \), where \( W \) are the weights, \( x \) are the input features, and \( y \) is the final transformed feature space.

A linear transformation with PCA has some disadvantages. It will make features less interpretable, and sometimes even useless for follow-up in certain use cases. For example, if potential cancer-related genes were discovered using a feature extraction technique, it may describe that the gene was partially involved together with other genes. A follow-up in the laboratory would not make sense, e.g., to partially knock out/activate genes.

## How Are Dimensions Reduced in PCA?

### Part 1: Center Data Around the Origin

The first part is computing the average of the data, which can be done in four smaller steps:

1. Compute the average per feature.
2. Compute the overall center.
3. Shift the data so that it is centered around the origin.

This transformation step does not change the relative distance between the points but only centers the data around the origin.

### Part 2: Fit the Line Through Origin and Data Points

The next part is to fit a line through the origin and the data points (or samples). This can be done by:

1. Drawing a random line through the origin.
2. Projecting the samples on the line orthogonally.
3. Rotating until the best fit is found by minimizing the distances.

### Part 3: Computing the Principal Components and the Loadings

We determined the best-fitted line in the direction with maximum variation, which is now the 1st Principal Component (PC1). The next step is to compute the slope of PC1 that describes the contribution of each feature for PC1. The slope of the line is representative of our visual observation; for every unit of feature contribution, the principal component is formed.

### Part 4: The Transformation and Explained Variance

We computed the Principal Components (PCs) and can now rotate the entire dataset such that the x-axis is the direction where the largest variance is seen (aka PC1). Each PC will contain a proportion of the total variation. To compute the explained variance, divide the sum of squared distances (SS) for each PC by the number of data points minus one.

### Standardization

Before we do parts 1 to 4, it is crucial to get the data in the right shape by standardization. PCA is sensitive to variables that have different value ranges or the presence of outliers. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.

## The PCA Library

A few words about the `pca` library used for the upcoming analysis. The `pca` library is designed to tackle several challenges such as:

- Analyze different types of data.
- Computing and plotting the explained variance (scree plot).
- Extraction of the best-performing features.
- Insights into the loadings with the Biplot.
- Outlier Detection.
- Removal of unwanted (technical) bias.

### Benefits of `pca` Library

- Maximizes compatibility and integration in pipelines.
- Built-in standardization.
- Contains the most-wanted output and plots.
- Simple and intuitive.
- Open-source with comprehensive documentation.

## A Practical Example to Understand the Loadings

### Creating a Synthetic Dataset

For demonstration purposes, create a synthetic dataset containing 8 features and 250 samples. Each feature will contain random integers but with increasing variance. All features are independent of each other. The dataset is ideal for demonstrating PCA principles, the loadings, the explained variance, and the importance of standardization.

### PCA Results

After fitting and transforming the data, examine the features and the variation in the data. The explained (cumulative) variance can be examined with a scree plot. For a 2-dimensional plot, visualize the loadings and scatter.

### Real Dataset: Wine Dataset

Analyze a realistic dataset (wine dataset) with 178 samples, 13 features, and 3 wine classes. Normalize the data to ensure each variable contributes equally to the analysis. Using PCA, extract the top-performing features and visualize the explained variance with a scree plot. 

## Outlier Detection

The `pca` library contains two methods to detect outliers: Hotelling’s T2 and SPE/DmodX. Both methods are complementary and computing the overlap can point towards the most deviant observations.

### Wrapping Up

Each Principal Component is a linear vector that contains proportions of the contribution of the original features. The interpretation of the contribution of each variable to the principal components can be retrieved using the loadings and visualized with the biplot. This analysis provides intuition about the variation of the variables and class separability. The `pca` library also provides functionalities to remove outliers and unwanted technical variance (or bias) using a normalization strategy.
