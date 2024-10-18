---
author_profile: false
categories:
- Data Science
- Machine Learning
classes: wide
date: '2019-12-30'
excerpt: AUC-ROC and Gini are popular metrics for evaluating binary classifiers, but
  they can be misleading on imbalanced datasets. Discover why AUC-PR, with its focus
  on Precision and Recall, offers a better evaluation for handling rare events.
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
keywords:
- Auc-pr
- Precision-recall
- Binary classifiers
- Imbalanced data
- Machine learning metrics
seo_description: When evaluating binary classifiers on imbalanced datasets, AUC-PR
  is a more informative metric than AUC-ROC or Gini. Learn why Precision-Recall curves
  provide a clearer picture of model performance on rare events.
seo_title: 'AUC-PR vs. AUC-ROC: Evaluating Classifiers on Imbalanced Data'
seo_type: article
summary: In this article, we explore why AUC-PR (Area Under Precision-Recall Curve)
  is a superior metric for evaluating binary classifiers on imbalanced datasets compared
  to AUC-ROC and Gini. We discuss how class imbalance distorts performance metrics
  and provide real-world examples of why Precision-Recall curves give a clearer understanding
  of model performance on rare events.
tags:
- Binary classifiers
- Imbalanced data
- Auc-pr
- Precision-recall
title: 'Evaluating Binary Classifiers on Imbalanced Datasets: Why AUC-PR Beats AUC-ROC
  and Gini'
---

When working with binary classifiers, metrics like **AUC-ROC** and **Gini** have long been the default for evaluating model performance. These metrics offer a quick way to assess how well a model discriminates between two classes, typically a **positive class** (e.g., detecting fraud or predicting defaults) and a **negative class** (e.g., non-fraudulent or non-default cases). 

However, when dealing with **imbalanced datasets**, where one class is much more prevalent than the other, these metrics can **mislead** us into believing a model is better than it truly is. In such cases, **AUC-PR**—which focuses on **Precision** and **Recall**—offers a more meaningful evaluation of a model’s ability to handle rare events, providing a clearer picture of how the model performs on the **minority class**.

In this article, we'll explore why **AUC-PR** (Area Under the Precision-Recall Curve) is more informative than **AUC-ROC** and **Gini** when evaluating models on imbalanced datasets. We’ll delve into why AUC-ROC often **overstates model performance**, and how AUC-PR shifts the focus to the model’s performance on the **positive class**, giving a more reliable assessment of how well it handles **imbalanced classes**.

## The Challenges of Imbalanced Data

Before diving into metrics, it’s important to understand the **challenges of imbalanced data**. In many real-world applications, the class distribution is highly skewed. For instance, in **fraud detection**, **medical diagnosis**, or **default prediction**, the positive class (e.g., fraudulent transactions, patients with a disease, or customers defaulting on loans) represents only a **tiny fraction** of the total cases.

In these scenarios, models tend to **focus heavily on the majority class**, often leading to deceptive results. A model might show high accuracy by correctly identifying many **True Negatives** but fail to adequately detect the **True Positives**—the rare but critical cases. This is where traditional metrics like AUC-ROC and Gini can fall short.

### Imbalanced Data Example: Fraud Detection

Imagine you’re building a model to detect fraudulent transactions. Out of 100,000 transactions, only 500 are fraudulent. That’s a **0.5% positive class** and a **99.5% negative class**. A model that predicts **all transactions as non-fraudulent** would still achieve **99.5% accuracy**, despite **failing completely** to detect any fraud.

While accuracy alone is clearly misleading, even metrics like **AUC-ROC** and **Gini**, which aim to balance True Positives and False Positives, can still provide an **inflated sense of performance**. This is because they take **True Negatives** into account, which, in imbalanced datasets, dominate the metric and obscure the model’s struggles with the positive class.

## Why AUC-ROC and Gini Can Be Misleading

The **AUC-ROC curve** (Area Under the Receiver Operating Characteristic Curve) is widely used to evaluate binary classifiers. It plots the **True Positive Rate** (TPR) against the **False Positive Rate** (FPR) at various classification thresholds. The **Gini coefficient** is closely related to AUC-ROC, as it is simply **2 * AUC-ROC - 1**.

While AUC-ROC is effective for **balanced datasets**, it becomes problematic when applied to **imbalanced data**. Here’s why:

### 1. **Over-Emphasis on True Negatives**

The ROC curve incorporates the **True Negative Rate** (TNR), which means that a model can appear to perform well by simply classifying the majority of non-events (True Negatives) correctly. In imbalanced datasets, where the negative class is abundant, even a model with **poor performance on the positive class** can still achieve a high AUC-ROC score, giving a **false sense of effectiveness**.

For example, a model that classifies all non-fraudulent transactions correctly while missing most fraudulent transactions will still show a **high AUC-ROC**. This is because the **False Positive Rate** (FPR) will remain low, and the **True Positive Rate** (TPR) can look decent even if many fraud cases are missed.

### 2. **Sensitivity to Class Imbalance**

In imbalanced datasets, the **majority class** dominates the calculation of the ROC curve. As a result, the metric often emphasizes performance on the negative class rather than the positive class. For highly skewed datasets, this can result in a **high AUC-ROC score**, even if the model is **failing** to correctly classify the minority class.

For instance, if 95% of your dataset consists of **True Negatives**, a model that excels at classifying the negative class but performs poorly on the positive class can still produce a high **AUC-ROC** score. In this way, AUC-ROC can **overstate** how well your model is really doing when you care most about the positive class.

## Why AUC-PR Is Better for Imbalanced Data

When evaluating binary classifiers on imbalanced datasets, a better approach is to use the **AUC-PR curve** (Area Under the Precision-Recall Curve). The **Precision-Recall curve** plots **Precision** (the proportion of correctly predicted positive cases out of all predicted positive cases) against **Recall** (the proportion of actual positive cases that are correctly identified).

### 1. **Focus on the Positive Class**

The key advantage of **AUC-PR** is that it **focuses on the positive class**, without being distracted by the abundance of True Negatives. This is particularly important when dealing with **rare events**, where identifying the minority class (e.g., fraud, defaults, or disease) is the primary goal. 

**Precision** measures how many of the predicted positive cases are correct, and **Recall** measures how well the model identifies actual positive cases. Together, they provide a clearer picture of the model's performance when dealing with **imbalanced classes**.

For example, in fraud detection, the **Precision-Recall curve** will give a more accurate sense of how well the model balances **finding fraud cases** (high Recall) with ensuring that **predicted fraud cases are actually fraudulent** (high Precision).

### 2. **Ignoring True Negatives**

One of the strengths of **AUC-PR** is that it **ignores True Negatives**—which are often overwhelmingly present in imbalanced datasets. This means that the model’s performance is evaluated **solely** on its ability to handle the positive class (the class of interest in most real-world applications). 

By ignoring True Negatives, the **Precision-Recall curve** gives a more direct view of the model’s performance on **rare events**, making it **far more suitable** for tasks like **fraud detection**, **default prediction**, or **medical diagnoses** where false positives and false negatives carry different risks and costs.

## A Real-World Example: Comparing AUC-ROC and AUC-PR

Let’s look at a real-world example to illustrate how AUC-PR offers a better assessment of model performance on imbalanced data. Imagine you’re building a classifier to predict loan defaults.

### Step 1: Evaluating with AUC-ROC

When you plot the **ROC curve**, you see that the model achieves a **high AUC-ROC score** of 0.92. Based on this, it might seem that the model is excellent at distinguishing between default and non-default cases. The **Gini coefficient**, calculated as **2 * AUC-ROC - 1**, is similarly high, suggesting strong model performance.

### Step 2: Evaluating with AUC-PR

Now, you turn to the **Precision-Recall curve** and find a different story. Although Recall is high (the model identifies most default cases), **Precision is much lower**, suggesting that many of the predicted defaults are actually **false positives**. This means that while the model is good at detecting defaults, it’s not as confident in its predictions. As a result, the **AUC-PR** score is significantly lower than the AUC-ROC score, reflecting the model’s **struggle with class imbalance**.

### Step 3: What This Tells Us

This discrepancy between AUC-ROC and AUC-PR tells us that while the model might appear to perform well overall (high AUC-ROC), its **actual performance** in identifying and confidently predicting defaults is **suboptimal** (low AUC-PR). In practice, this could lead to **incorrect predictions**, where too many non-default cases are classified as defaults, resulting in unnecessary interventions or loss of trust in the model.

## Conclusion: Why AUC-PR Should Be Your Go-To for Imbalanced Data

For **imbalanced datasets**, AUC-ROC and Gini can **mislead** you into thinking your model performs well when, in fact, it struggles with the **minority class**. Metrics like **AUC-PR** offer a more focused evaluation by prioritizing **Precision** and **Recall**—two critical metrics for rare events where misclassification can be costly.

In practice, when evaluating models on tasks like **fraud detection**, **default prediction**, or **disease diagnosis**, where the positive class is rare but crucial, the **Precision-Recall curve** and **AUC-PR** give a more honest reflection of the model’s performance. While AUC-ROC might inflate the model's effectiveness by focusing on the majority class, AUC-PR shows how well the model **balances** Precision and Recall—two metrics that matter most in real-world applications where **rare events** have significant consequences.

### Key Takeaways:

- **AUC-ROC** and **Gini** are suitable for balanced datasets but can **overstate** model performance on imbalanced data.
- **AUC-PR** focuses on the **positive class**, providing a clearer view of how well the model handles **rare events**.
- When evaluating binary classifiers on **imbalanced datasets**, always consider using **AUC-PR** as it offers a more honest assessment of your model's strengths and weaknesses.

In your next machine learning project, especially when handling imbalanced datasets, prioritize **AUC-PR** over AUC-ROC and Gini for a clearer, more accurate evaluation of your model’s ability to manage rare but critical events.
