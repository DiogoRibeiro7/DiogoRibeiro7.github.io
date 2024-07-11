---
title: "Testing and Evaluating Outlier Detectors Using Doping"
categories:
- Data Science
- Machine Learning
tags:
- Outlier Detection
- Data Doping
- Model Evaluation
author_profile: false
---

Outlier detection is a challenging problem, especially when it comes to evaluating the effectiveness of outlier detectors. This article presents a method known as doping, where real data rows are modified to create synthetic outliers. These doped records help in assessing the performance of outlier detectors by providing a benchmark for evaluation.

We focus specifically on tabular data, although the concept of doping can be applied to various data modalities including text, image, audio, and network data.

## Challenges in Evaluating Outlier Detectors

### Evaluating Predictive Models

In predictive modeling for regression and classification, the availability of labeled data simplifies model evaluation. Techniques like train-validation-test splits or cross-validation help in tuning models and estimating accuracy. The presence of labels allows for straightforward assessment of model performance on unseen data.

### Evaluating Outlier Detection

Outlier detection lacks labeled data, making the problem significantly more difficult. Unlike clustering, where metrics like the Silhouette score can measure clustering quality, outlier detection has no equivalent. Any method to evaluate detected outliers essentially becomes an outlier detection algorithm itself, leading to circular reasoning.

Without a definitive way to determine if the highest-scoring records are truly outliers, evaluating outlier detectors with real data is nearly impossible.

## Synthetic Test Data and Doping

One solution is to create synthetic test data where we assume the generated data are outliers. By evaluating how well detectors identify these synthetic outliers, we can gauge their effectiveness. Among various methods to create synthetic data, this article focuses on doping.

### Doping Data Records

Doping involves modifying existing data records, typically altering one or a few cells per record. For example, in a financial dataset of franchise performance, a typical record might include:

- Age of the franchise
- Number of years with the current owner
- Number of sales last year
- Total dollar value of sales last year

A doped version of this record might set the age to 100 years, making it an obvious outlier. More subtle doping might change the total sales from $500,000 to $100,000, creating an unusual combination of sales numbers and total sales value.

### Approaches to Doping

#### Including Doped Records in Training Data

Including doped records in training data helps test the ability to detect outliers within the current dataset. By adding a small number of doped records to the training set, we can evaluate if detectors score these records higher than their original versions. This method helps ensure that the detector can identify outliers similar to those doped records.

#### Including Doped Records Only in Testing Data

Training with real data and testing with both real and doped data tests the ability to detect future outliers. This method trains on a clean dataset and evaluates performance on a separate test set that includes doped records. This approach allows for a larger and more reliable test dataset without contaminating the training data.

## Creating Doped Data

One method to create doped data involves randomly selecting cells to modify. Although some doped records may not be true outliers, random values typically disrupt feature associations, making most doped records anomalous.

### Example: Abalone Dataset

We use the abalone dataset from OpenML for an example. After preprocessing the data, we test three outlier detectors: Isolation Forest, LOF, and ECOD, using the PyOD library.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD

# Collect the data
data = fetch_openml('abalone', version=1) 
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

# Use an Isolation Forest to clean the data
clf = IForest() 
clf.fit(df)
if_scores = clf.decision_scores_
top_if_scores = np.argsort(if_scores)[::-1][:10]
clean_df = df.loc[[x for x in df.index if x not in top_if_scores]].copy()

# Create a set of doped records
doped_df = df.copy() 
for i in doped_df.index:
  col_name = np.random.choice(df.columns)
  med_val = clean_df[col_name].median()
  if doped_df.loc[i, col_name] > med_val:
    doped_df.loc[i, col_name] = \
      clean_df[col_name].quantile(np.random.random()/2)
  else:
    doped_df.loc[i, col_name] = \
      clean_df[col_name].quantile(0.5 + np.random.random()/2)

# Define a method to test a specified detector. 
def test_detector(clf, title, df, clean_df, doped_df, ax): 
  clf.fit(clean_df)
  df = df.copy()
  doped_df = doped_df.copy()
  df['Scores'] = clf.decision_function(df)
  df['Source'] = 'Real'
  doped_df['Scores'] = clf.decision_function(doped_df)
  doped_df['Source'] = 'Doped'
  test_df = pd.concat([df, doped_df])
  sns.boxplot(data=test_df, orient='h', x='Scores', y='Source', ax=ax)
  ax.set_title(title)

# Plot each detector in terms of how well they score doped records 
# higher than the original records
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 3)) 
test_detector(IForest(), "IForest", df, clean_df, doped_df, ax[0])
test_detector(LOF(), "LOF", df, clean_df, doped_df, ax[1])
test_detector(ECOD(), "ECOD", df, clean_df, doped_df, ax[2])
plt.tight_layout()
plt.show()
```

In this example, we see that Isolation Forest scores doped records higher, but not significantly. LOF effectively distinguishes doped records, while ECOD struggles since the doping method does not create extreme values but unusual combinations.

### Alternative Doping Methods

Other doping techniques can enhance the likelihood of creating anomalous records. For instance, for categorical data, we might select a new value different from both the original and predicted values. For numeric data, selecting values in different quartiles from the original and predicted quartiles can create outliers.

## Creating a Suite of Test Datasets

To accurately evaluate outlier detectors, creating multiple test datasets with varying levels of doping difficulty is beneficial. This helps in differentiating detectors' performance and understanding their strengths and weaknesses.

The doping should represent the types of outliers of interest, covering the range of what needs to be detected. Multiple test sets allow for a thorough evaluation and help in creating effective ensembles of outlier detectors.

## Conclusion

Evaluating outlier detectors is challenging, especially without labeled data. Synthetic data creation, particularly through doping, offers a practical solution. Although not perfect, doping provides a means to benchmark and compare outlier detectors, helping select the best-performing detectors and estimate their future performance.