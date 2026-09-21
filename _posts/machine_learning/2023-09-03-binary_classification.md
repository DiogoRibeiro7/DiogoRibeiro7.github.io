---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2023-09-03'
excerpt: Binary classification is not just choosing an algorithm. It requires a clear target, calibrated probabilities, realistic validation, and thresholds tied to decision costs.
header:
  image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  og_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  overlay_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  twitter_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
keywords:
- Binary classification
- Logistic regression
- ROC AUC
- Precision recall
- Calibration
- Classification thresholds
- Class imbalance
- Cost-sensitive classification
permalink: '/machine-learning/binary_classification/'
redirect_from:
- '/machine learning/binary_classification/'
seo_description: A rigorous introduction to binary classification, including probabilistic prediction, thresholds, calibration, class imbalance, validation, and decision costs.
seo_title: 'Binary Classification: Probabilities, Thresholds, and Decisions'
seo_type: article
tags:
- Classification
- Supervised Learning
- Machine Learning
title: 'Binary Classification: Probabilities Before Labels'
---

Binary classification is often described as the task of assigning observations to one of two classes. That description hides an important layer. Most useful classifiers do not begin with a hard label. They estimate a score or probability and only then convert that quantity into an action using a threshold.

The statistical problem and the decision problem are therefore separate. A model may rank observations well but be poorly calibrated. A model may produce accurate probabilities but still be used with an inappropriate threshold. A classifier should be evaluated at all three levels: discrimination, calibration, and decision utility.

## The probabilistic target

Let

$$
Y\in\{0,1\}
$$

and let $X$ denote the available predictors. A probabilistic classifier estimates

$$
p(x)=\Pr(Y=1\mid X=x).
$$

A hard prediction is then produced by a threshold $t$,

$$
\hat Y =
\begin{cases}
1, & \hat p(x)\ge t,\\
0, & \hat p(x)<t.
\end{cases}
$$

The common threshold $t=0.5$ has no universal justification. If false negatives are much more costly than false positives, or if only a limited number of cases can be acted upon, the appropriate threshold can be very different.

## Confusion matrix and threshold-dependent metrics

For a fixed threshold, predictions can be summarized as true positives, false positives, true negatives, and false negatives.

From these counts,

$$
\text{Sensitivity}
=
\frac{TP}{TP+FN},
$$

$$
\text{Specificity}
=
\frac{TN}{TN+FP},
$$

$$
\text{Precision}
=
\frac{TP}{TP+FP}.
$$

Precision depends on prevalence as well as model performance. It therefore changes when the same classifier is deployed in populations with different base rates.

The F1 score is

$$
F_1
=
2\frac{\text{precision}\times\text{recall}}
{\text{precision}+\text{recall}}.
$$

It can be useful when precision and recall are both relevant, but it encodes a particular symmetric trade-off and ignores true negatives. It should not be treated as a universal classification objective.

## Accuracy and class imbalance

Accuracy is

$$
\frac{TP+TN}{N}.
$$

It can be misleading when one class is common, because a trivial majority-class classifier may achieve high accuracy while detecting none of the minority class.

Class imbalance does not, however, make every standard metric invalid. ROC AUC remains a valid ranking measure under imbalance, although it can be operationally uninformative when false positives are costly and the negative class is extremely large. Precision-recall curves are often more directly informative in rare-event settings because precision reflects the burden of false alerts.

The important point is to choose metrics that correspond to the deployment problem rather than to classify metrics as universally "good" or "bad" for imbalanced data.

## Discrimination is not calibration

Two models can have similar ROC AUC and very different probability estimates. If among cases assigned risk $0.8$, only about 40% are actually positive, the probabilities are badly calibrated even if the ranking is strong.

Calibration asks whether

$$
\Pr(Y=1\mid \hat p(X)\approx p)
\approx p.
$$

Useful diagnostics include reliability diagrams, the Brier score, calibration intercepts and slopes, and calibration curves.

Calibration matters whenever probabilities are interpreted as risks, used in expected-cost calculations, or combined with other decision models.

## Logistic regression

Logistic regression models the log odds as

$$
\log\frac{p(x)}{1-p(x)}
=
\beta_0+x^\top\beta.
$$

This is a probabilistic model, not merely a classification rule. Its strengths include interpretability, stable estimation in moderate dimensions, and direct probability output. Its limitations include the assumed functional form on the log-odds scale and sensitivity to separation or severe misspecification.

Nonlinearity can be introduced through splines, interactions, basis expansions, or generalized additive models without abandoning the regression framework.

## Trees and ensembles

Decision trees partition predictor space recursively. They are easy to visualize but can be unstable: small changes in the data can produce different trees.

Random forests reduce that instability by averaging over many randomized trees. Gradient-boosted trees build an additive ensemble sequentially and can achieve strong predictive performance on structured tabular data.

Neither method is automatically superior to logistic regression. Performance depends on sample size, signal structure, interactions, missingness, noise, and the validation design.

## Support vector machines

Support vector machines search for separating boundaries with large margins. With kernels, they can represent nonlinear decision surfaces.

Their native output is a decision score rather than a calibrated probability. If calibrated probabilities are needed, a separate calibration procedure such as Platt scaling or isotonic regression may be applied using data not used to fit the original classifier.

## Neural networks

Neural networks can represent highly flexible decision functions and are valuable when the predictors have complex structure such as images, text, audio, or large-scale high-dimensional interactions.

That flexibility does not remove the need for calibration, regularization, representative training data, and deployment-like validation. A more flexible model can overfit more subtle forms of dataset-specific structure.

## Threshold choice as a decision problem

Suppose a false positive costs $C_{FP}$ and a false negative costs $C_{FN}$. Under a simple two-action model with calibrated probabilities, the expected-loss threshold can be derived from those costs.

Predict positive when

$$
(1-p)C_{FP}
<
pC_{FN}.
$$

Solving for $p$,

$$
p >
\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

This simple result shows why 0.5 is not privileged. If missing a true case is ten times more costly than investigating a false alert, the optimal threshold can be much lower.

Real systems often add capacity constraints, delayed outcomes, fairness requirements, or multiple downstream actions. The threshold should then be chosen inside the actual operational decision framework.

## Validation must match deployment

Random train-test splits assume that future observations are exchangeable with randomly held-out historical observations. That assumption often fails.

Examples include:

- repeated observations from the same person
- multiple rows from the same machine
- temporal forecasting
- site-to-site deployment
- geographic generalization
- policy changes
- sensor replacement
- concept drift

Grouped, temporal, or external validation may therefore be required. Leakage can make a classifier appear excellent even when it will fail immediately after deployment.

## Choosing a classifier

The correct question is not "Which algorithm is best?" A useful workflow is:

1. define the outcome and observation unit
2. define the deployment population
3. identify the costs of different errors
4. build a simple baseline
5. compare candidate models under realistic validation
6. assess both discrimination and calibration
7. select thresholds from the decision problem
8. monitor performance after deployment

This process often makes model choice less dramatic than it first appears. A simple model that is calibrated, stable, and operationally aligned can be preferable to a more complex model with slightly better AUC.

## Conclusion

Binary classification is a probability-estimation problem followed by a decision rule. Confusion-matrix metrics describe one threshold. ROC and precision-recall curves describe families of thresholds. Calibration evaluates whether predicted probabilities can be trusted as probabilities. The final threshold belongs to the decision problem.

A sound classifier is therefore not merely a function that returns zero or one. It is a model embedded in a measurement, validation, and decision system.

## References

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
- Hand, D. J. (2009). Measuring classifier performance: a coherent alternative to the area under the ROC curve. *Machine Learning*, 77, 103-123.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of ICML*.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
