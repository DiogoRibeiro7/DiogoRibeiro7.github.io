---
author_profile: false
categories:
- Data Science
- Machine Learning
classes: wide
date: '2024-07-15'
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- outlier detection
- data doping
- model evaluation
- anomaly detection
- machine learning testing
- evaluating ML models
- robust data models
seo_description: Learn how to test and evaluate outlier detection models using data
  doping techniques. Understand the impact of doping on model performance and outlier
  identification.
seo_title: Evaluating Outlier Detectors with Data Doping Techniques
seo_type: article
summary: This article explores techniques for testing and evaluating outlier detection
  models using data doping, highlighting key methodologies and their impact on model
  performance.
tags:
- Outlier Detection
- Data Doping
- Model Evaluation
title: Testing and Evaluating Outlier Detectors Using Doping
---

Outlier detection presents significant challenges, particularly in evaluating the effectiveness of outlier detection algorithms. Traditional methods of evaluation, such as those used in predictive modeling, are often inapplicable due to the lack of labeled data. This article introduces a method known as doping, where existing data rows are intentionally modified to create synthetic outliers. These doped records serve as a benchmark for evaluating the performance of outlier detectors.

Doping involves altering one or more attributes of real data records to ensure they appear anomalous compared to the rest of the dataset. This process allows us to create a controlled environment where the effectiveness of different outlier detection techniques can be rigorously tested and compared.

In this article, we concentrate specifically on tabular data, as it is a common format in many practical applications. However, the concept of doping is not limited to tables. It can be adapted to other data modalities, including text, images, audio, and network data. For instance, in text data, doping could involve changing certain words to uncommon synonyms, while in image data, it might involve altering pixel values to create visually noticeable anomalies.

By applying doping across different data types, we can develop a more comprehensive understanding of how well various outlier detectors perform under diverse conditions. This method provides a systematic approach to tackling one of the most difficult aspects of outlier detection: the absence of a clear and objective ground truth.

## Challenges in Evaluating Outlier Detectors

### Evaluating Predictive Models

In predictive modeling for regression and classification, the availability of labeled data greatly simplifies model evaluation. Techniques such as train-validation-test splits or cross-validation are commonly employed to fine-tune models and estimate their accuracy. These methods provide a straightforward way to assess model performance on unseen data because the labels offer a clear benchmark against which predictions can be measured.

### Evaluating Outlier Detection

Outlier detection, on the other hand, lacks the luxury of labeled data, making the problem significantly more challenging. Unlike clustering, where metrics like the Silhouette score can be used to measure the quality of clusters, outlier detection does not have an analogous metric. The absence of a clear definition of what constitutes an outlier complicates the evaluation process. 

In clustering, we can assess the internal consistency of clusters and the separation between them using distance metrics such as Manhattan or Euclidean distances. This allows for the calculation of scores that help in selecting the best clustering solution. However, for outlier detection, there is no equivalent objective measure. Any attempt to evaluate detected outliers would essentially involve defining a new outlier detection algorithm, leading to circular reasoning.

Without a definitive method to verify if the highest-scoring records are indeed outliers, evaluating outlier detectors using real data becomes nearly impossible. This lack of a clear and objective ground truth poses a major hurdle in the development and fine-tuning of effective outlier detection systems.

## Synthetic Test Data and Doping

One practical solution to the evaluation problem in outlier detection is the creation of synthetic test data, which can be assumed to be outliers. By assessing how well detectors identify these synthetic outliers, we can evaluate their effectiveness. Among the various methods to generate synthetic data, this article focuses on doping.

### Doping Data Records

Doping involves modifying existing data records by altering one or a few cells per record. This technique ensures that the modified records stand out as outliers. For example, consider a financial dataset of franchise performance, where a typical record might include:

- Age of the franchise
- Number of years with the current owner
- Number of sales last year
- Total dollar value of sales last year

A doped version of this record could set the age to 100 years, making it an obvious outlier. More subtle doping might change the total sales from $500,000 to $100,000, creating an unusual combination of sales numbers and total sales value. By manipulating the data in this way, we can create a controlled environment to test and compare the performance of different outlier detectors.

### Approaches to Doping

#### Including Doped Records in Training Data

One approach is to include doped records in the training data. This helps to test the detector's ability to identify outliers within the current dataset. By adding a small number of doped records to the training set, we can evaluate if the detectors score these records higher than their original versions. This method ensures that the detector can identify outliers similar to those doped records.

#### Including Doped Records Only in Testing Data

Another approach is to train on real data and test with both real and doped data. This tests the detector's ability to identify future outliers. By training on a clean dataset and evaluating performance on a separate test set that includes doped records, this method allows for a larger and more reliable test dataset without contaminating the training data. This approach is useful for assessing how well the detector can generalize to new, unseen data that may contain outliers.

By incorporating these approaches, we can systematically evaluate and improve the performance of outlier detectors, ensuring they are robust and reliable for practical applications.

## Creating Doped Data

One method to create doped data involves randomly selecting cells to modify within the dataset. Although not all doped records will be true outliers, the randomness in value selection typically disrupts the inherent feature associations, making most doped records anomalous. This approach ensures a diverse set of anomalies, which can be used to test the robustness of outlier detectors.

### Example: Abalone Dataset

To illustrate the process of creating doped data and evaluating outlier detectors, we use the abalone dataset from OpenML. After preprocessing the data, we test three outlier detectors: Isolation Forest, Local Outlier Factor (LOF), and ECOD, utilizing the PyOD library.

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

In this example, we see that Isolation Forest scores doped records higher, but not significantly. LOF effectively distinguishes doped records, while ECOD struggles since the doping method does not create extreme values but rather unusual combinations.

### Alternative Doping Methods

Other doping techniques can enhance the likelihood of creating anomalous records. For instance:

- **Categorical Data:** Select a new value that is different from both the original and predicted values using a predictive model. For example, if the original value in a categorical column is "A," and the predicted value is "B," select a value like "C" that deviates from the expected pattern.

- **Numeric Data:** Choose new values from different quartiles than the original and predicted values. For instance, if the original value is in the first quartile (Q1) and the predicted value is in the second quartile (Q2), select a value from the third (Q3) or fourth quartile (Q4). This approach increases the likelihood of creating anomalous records by disrupting normal feature associations.

By employing these alternative doping methods, we can generate a diverse set of test datasets that more accurately reflect potential real-world anomalies. This allows for a comprehensive evaluation of outlier detectors and helps in identifying their strengths and weaknesses across different types of data anomalies.

## Creating a Suite of Test Datasets

To accurately evaluate outlier detectors, creating multiple test datasets with varying levels of doping difficulty is beneficial. This approach helps differentiate the performance of detectors and provides a clearer understanding of their strengths and weaknesses.

### Varying Levels of Doping Difficulty

Creating test datasets with different levels of doping difficulty involves modifying the data to various extents. For instance, one test set might include very obvious anomalies where multiple features are significantly altered, while another test set might contain subtle anomalies with minimal changes. This variation allows us to evaluate how well detectors can identify both blatant and nuanced outliers.

### Representing Types of Outliers of Interest

The doping process should mimic the types of outliers that are of interest in practical applications. By ensuring that the doped records cover the range of anomalies we aim to detect, we create more relevant and effective test datasets. These datasets can include a mix of single-feature anomalies, unusual combinations of features, and rare value occurrences.

### Comprehensive Evaluation

Using multiple test sets enables a thorough evaluation of outlier detectors. It allows us to assess:

- **Detection Sensitivity:** How well the detector identifies different levels of anomalies.
- **False Positives:** The rate at which normal records are incorrectly flagged as outliers.
- **False Negatives:** The rate at which actual outliers are missed by the detector.

### Creating Effective Ensembles

Multiple test sets also aid in creating effective ensembles of outlier detectors. Since different detectors may excel at identifying different types of anomalies, combining their strengths can lead to a more robust detection system. By understanding the specific strengths and weaknesses of each detector through comprehensive testing, we can design ensembles that maximize detection accuracy while minimizing false positives and negatives.

Creating a suite of test datasets with varying levels of doping difficulty is crucial for accurately evaluating outlier detectors. This approach not only helps in understanding the performance of individual detectors but also in designing effective ensembles that can tackle a wide range of anomalies in real-world applications.

## Conclusion

Evaluating outlier detectors poses significant challenges, particularly due to the absence of labeled data. Synthetic data creation, especially through the method of doping, provides a practical and effective solution. By modifying existing data to create synthetic outliers, doping allows us to establish benchmarks and systematically compare the performance of various outlier detectors.

Although doping is not a perfect method, it offers a valuable means to gauge the effectiveness of outlier detectors. By creating and using doped records, we can identify the best-performing detectors and gain insights into their potential performance on future datasets. This approach not only aids in the selection of robust outlier detection models but also enhances our ability to develop comprehensive and reliable detection systems for diverse real-world applications.

## References

1. Aggarwal, C. C. (2016). Outlier Analysis. Springer.
2. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data (pp. 93-104). ACM.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. ACM Computing Surveys, 41(3), 1-58.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
5. Li, D., Chen, D., Jin, B., Shi, L., Goh, J., & Ng, S.-K. (2019). MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks. In Proceedings of the 2019 International Joint Conference on Artificial Intelligence (IJCAI) (pp. 4251-4257).
6. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
7. Zhao, Y., Nasrullah, Z., & Li, Z. (2019). PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of Machine Learning Research, 20(96), 1-7.
8. OpenML. (n.d.). Abalone Dataset. Retrieved from [https://www.openml.org/search?type=data&sort=runs&id=42726&status=active](https://www.openml.org/search?type=data&sort=runs&id=42726&status=active).
