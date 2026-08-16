---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2021-11-10'
excerpt: Explore key metrics for evaluating classification and regression models.
header:
  image: /assets/images/data_science_8.avif
  og_image: /assets/images/data_science_8.avif
  overlay_image: /assets/images/data_science_8.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.avif
  twitter_image: /assets/images/data_science_8.avif
keywords:
- Model evaluation
- Accuracy
- Precision
- Recall
- Regression metrics
permalink: '/machine-learning/model_evaluation_metrics/'
redirect_from:
- '/machine learning/model_evaluation_metrics/'
seo_description: A concise overview of essential metrics like precision, recall, F1-score,
  and RMSE for measuring model performance.
seo_title: Essential Metrics for Evaluating Machine Learning Models
seo_type: article
summary: Learn how to interpret common classification and regression metrics to choose
  the best model for your data.
tags:
- Model Evaluation
title: A Guide to Model Evaluation Metrics
---

Choosing the right evaluation metric is critical for comparing models and selecting the best one for your problem.

## The Metric Encodes Your Priorities

A metric is not a neutral measurement, it is a statement about which mistakes matter. Every choice implicitly weighs false positives against false negatives, and picking one before understanding that trade-off means the model will be optimised for the wrong thing.

The question to answer first is what happens downstream of a prediction. If a false positive means a wasted marketing email, it is cheap. If it means an unnecessary biopsy, it is not. If a false negative means a missed fraudulent transaction, the cost is the transaction value. Those answers determine the metric; the metric does not determine them.

## Classification Metrics

- **Accuracy** measures the fraction of correct predictions. It works well when classes are balanced but can be misleading with imbalanced datasets.
- **Precision** and **recall** capture how well the model retrieves relevant instances without producing too many false positives or negatives. The **F1-score** provides a balance between the two.

Written out from the confusion matrix:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} .
$$

Precision answers "when the model says yes, how often is it right". Recall answers "of everything that was actually positive, how much did we catch". They pull against each other, because lowering the decision threshold catches more positives while admitting more false alarms.

The accuracy trap is worth quantifying. In fraud detection with a 0.1% positive rate, a model predicting "not fraud" for every transaction achieves 99.9% accuracy and is completely worthless. Whenever classes are imbalanced, accuracy measures the base rate rather than the model.

The $F_1$ score weights precision and recall equally, which is itself an assumption. The general form lets you tilt it:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} .
$$

Setting $\beta = 2$ weights recall higher, appropriate for screening where a miss is worse than a false alarm. Setting $\beta = 0.5$ favours precision.

## Threshold-Free Measures

Most classifiers output a score, and the threshold turning that score into a label is a separate decision. Metrics that integrate over all thresholds evaluate the ranking rather than one operating point.

**ROC AUC** plots true positive rate against false positive rate and reports the area beneath. It equals the probability that a randomly chosen positive is ranked above a randomly chosen negative. Its weakness under heavy imbalance is that the false positive rate has a huge denominator, so a large absolute number of false positives barely moves the curve.

**PR AUC**, the area under the precision-recall curve, uses precision instead and therefore stays sensitive when positives are rare. For imbalanced problems it is the more informative summary, and its baseline is the positive class prevalence rather than 0.5.

**Log loss** evaluates the probabilities themselves rather than any thresholded decision:

$$
\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n} \big[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \big] .
$$

It punishes confident errors severely, which makes it the right choice when the predicted probability feeds into an expected-value calculation rather than a yes/no action.

That last case raises calibration, which the ranking metrics ignore entirely. A model can have excellent AUC while its probabilities are systematically too extreme. If you multiply a predicted probability by a monetary value, you need calibration, and a reliability diagram or Brier score is the diagnostic.

## Regression Metrics

- **Mean Absolute Error (MAE)** evaluates the average magnitude of errors.
- **Root Mean Squared Error (RMSE)** penalizes larger errors more heavily, making it useful when large deviations are particularly undesirable.

$$
\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|, \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2} .
$$

The difference is not cosmetic. Minimising squared error targets the conditional mean, while minimising absolute error targets the conditional median. On skewed targets these differ substantially, and RMSE will chase outliers that MAE largely ignores. Choose RMSE when a single large miss really is disproportionately costly, MAE when all errors scale linearly with cost.

RMSE is also always at least as large as MAE, so the two are never directly comparable across reports. When errors should be judged relative to magnitude, MAPE is tempting but breaks when actual values approach zero and penalises over-prediction and under-prediction asymmetrically; MASE, which scales error against a naive baseline, avoids both problems.

$R^2$ reports the share of variance explained, but it is a comparison against predicting the mean, not an absolute quality measure, and it should never be the sole criterion.

## Validating Honestly

The metric is only as trustworthy as the split it is computed on. Random k-fold cross-validation assumes exchangeable observations. That assumption fails for time series, where future data must never inform predictions about the past, and for grouped data, where the same customer or patient appearing in both training and test folds leaks information.

Use forward-chaining splits for temporal data and grouped splits when records cluster. And keep a genuinely held-out test set that is consulted once, at the end, because a validation set used repeatedly for tuning gradually becomes part of training.

Report variability alongside the point estimate. A mean cross-validation score without its standard deviation hides whether a 0.02 improvement is real or noise.

## Connecting Back to the Decision

Selecting evaluation metrics that align with business goals will help you make informed decisions about which model to deploy.

The most direct way to do that is to skip the proxy where possible. If each error type carries a known cost, compute expected cost directly:

$$
\text{Cost} = C_{FP} \cdot FP + C_{FN} \cdot FN,
$$

then choose the threshold that minimises it. This converts an abstract metric debate into an explicit economic one, and it makes the assumptions visible enough to argue about.

Track one primary metric that drives decisions, plus a small set of guardrails that must not degrade. Optimising a single number without constraints tends to produce models that win on the metric and fail in use.

## References

- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.
