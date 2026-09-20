---
author_profile: false
categories:
- Data Science
classes: wide
date: '2019-12-30'
excerpt: ROC and precision-recall curves answer different questions under class imbalance. ROC AUC does not become invalid when prevalence is low, but precision exposes the operational burden of false positives.
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
keywords:
- Precision-recall
- ROC AUC
- Binary classifiers
- Imbalanced data
- Prevalence
- Model evaluation
permalink: '/data-science/evaluating_binary_classifiers_imbalanced_datasets/'
redirect_from:
- '/data science/evaluating_binary_classifiers_imbalanced_datasets/'
seo_description: ROC AUC and precision-recall curves behave differently under class imbalance. This article derives the role of prevalence and explains when each metric is informative.
seo_title: ROC and Precision-Recall Under Class Imbalance
seo_type: article
summary: A mathematical comparison of ROC AUC, Gini, precision-recall curves and threshold-specific decision metrics for rare-event classification.
tags:
- Classification
- Model Evaluation
title: 'ROC and Precision-Recall Under Class Imbalance'
---

Class imbalance creates real evaluation problems, but not for the reason usually given.

A common explanation says that ROC AUC is "dominated by true negatives" and therefore becomes invalid when the positive class is rare. That statement is too crude. ROC AUC is built from the **true-positive rate** and **false-positive rate**, both of which are conditional rates. If the conditional score distributions remain unchanged, changing the number of positives relative to negatives does not by itself change the population ROC curve.

Precision behaves differently. It depends directly on class prevalence.

That distinction is the reason precision-recall curves can be much more informative in rare-event applications.

The correct conclusion is therefore not

\[
\boxed{\text{imbalanced data} \Rightarrow \text{ignore ROC AUC}}
\]

but

\[
\boxed{
\text{choose a metric whose conditioning matches the decision problem}
}
\]

## Start from the confusion matrix

For a fixed threshold, define

\[
\mathrm{TPR}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
=
P(\hat Y=1\mid Y=1)
\]

and

\[
\mathrm{FPR}
=
\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
=
P(\hat Y=1\mid Y=0).
\]

The ROC curve plots \(\mathrm{TPR}\) against \(\mathrm{FPR}\) while the classification threshold varies.

Precision is

\[
\mathrm{Precision}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
=
P(Y=1\mid \hat Y=1),
\]

while recall is simply the true-positive rate,

\[
\mathrm{Recall}=\mathrm{TPR}.
\]

The change in conditioning is the entire story.

ROC asks how the classifier behaves **within the positive and negative classes**.

Precision asks what fraction of the cases **flagged by the classifier** are actually positive.

Those are not interchangeable questions.

## Why prevalence enters precision but not the ROC coordinates

Let

\[
\pi=P(Y=1)
\]

be the prevalence of the positive class.

Bayes' rule gives

\[
P(Y=1\mid \hat Y=1)
=
\frac{
P(\hat Y=1\mid Y=1)P(Y=1)
}{
P(\hat Y=1)
}.
\]

Substituting \(\mathrm{TPR}\), \(\mathrm{FPR}\) and \(\pi\),

\[
\boxed{
\mathrm{Precision}
=
\frac{
\pi\,\mathrm{TPR}
}{
\pi\,\mathrm{TPR}
+
(1-\pi)\,\mathrm{FPR}
}
}
\]

This equation explains why a classifier can have an attractive ROC operating point and still create an unacceptable number of false alerts in a rare-event problem.

Suppose fraud prevalence is

\[
\pi=0.005,
\]

the classifier achieves

\[
\mathrm{TPR}=0.80
\]

and

\[
\mathrm{FPR}=0.01.
\]

A 1% false-positive rate may sound excellent. But precision is

\[
\frac{0.005(0.80)}
{0.005(0.80)+0.995(0.01)}
\approx 0.287.
\]

Only about 29% of the alerts are fraud.

The ROC point

\[
(0.01,0.80)
\]

has not changed. The operational interpretation has.

## What ROC AUC actually measures

ROC AUC summarizes ranking performance across thresholds.

For continuous scores, it has the probabilistic interpretation

\[
\mathrm{AUC}_{ROC}
=
P(S^+>S^-)
+
\frac{1}{2}P(S^+=S^-),
\]

where \(S^+\) is the score assigned to a randomly selected positive observation and \(S^-\) is the score assigned to a randomly selected negative observation.

This is why ROC AUC is largely insensitive to class prevalence: the comparison is made between conditional score distributions.

That property is not a defect.

If the scientific question is

> How well does the score rank positives above negatives?

ROC AUC is a coherent answer.

The problem begins when that ranking statistic is interpreted as though it described the expected burden of positive predictions in a population where positives are extremely rare.

It does not.

## Gini adds no new information to ROC AUC

For binary ranking, the commonly used model Gini coefficient is

\[
G=2\,\mathrm{AUC}_{ROC}-1.
\]

Therefore,

\[
\mathrm{AUC}_{ROC}=0.80
\quad\Longleftrightarrow\quad
G=0.60.
\]

Gini does not solve or worsen the class-imbalance problem. It is a linear rescaling of the same ranking statistic.

Reporting both as though they were independent evidence of model quality is redundant.

## Why precision-recall plots are useful for rare events

A precision-recall curve plots precision against recall while the threshold varies.

Because precision contains prevalence explicitly, the curve exposes the cost of false positives in the population being evaluated.

For a random classifier whose predictions are independent of the outcome, expected precision is the prevalence,

\[
\mathrm{Precision}_{\mathrm{baseline}}=\pi.
\]

If prevalence is 0.5%, a precision of 5% is ten times the random baseline even though 95% of alerts are false positives.

That is a much more useful statement than calling 5% precision simply "low."

The baseline moves with prevalence, which is both a strength and a limitation.

## AUPRC is not prevalence-invariant

The fact that precision depends on \(\pi\) means that the area under a precision-recall curve also depends on prevalence.

This matters when comparing experiments.

Suppose the same conditional score model is evaluated once in a case-control sample containing 50% positives and once in the real deployment population containing 1% positives. The ROC curve can remain essentially unchanged while the precision-recall curve changes dramatically.

Therefore,

\[
\boxed{
\text{AUPRC values from different prevalences are not automatically comparable}
}
\]

If deployment prevalence matters, evaluation should use a test set representative of that population or adjust predictive values to the target prevalence.

Saito and Rehmsmeier's central point is precisely that precision-recall plots make the consequences of imbalance visible in a way ROC plots do not.

That does not imply that ROC AUC is mathematically corrupted by imbalance.

## Neither area metric chooses an operating threshold

AUC metrics average over many thresholds, including thresholds that may never be used.

Production systems operate at one threshold, or under a policy that changes thresholds according to capacity or cost.

At a candidate threshold, the decision may depend on quantities such as

\[
\mathrm{TP},\quad
\mathrm{FP},\quad
\mathrm{FN},
\]

expected cost,

\[
C(t)
=
c_{\mathrm{FP}}P(\mathrm{FP}\mid t)
+
c_{\mathrm{FN}}P(\mathrm{FN}\mid t),
\]

or constraints such as

\[
\mathrm{Recall}(t)\ge 0.95
\]

with precision maximized subject to that requirement.

A fraud team that can investigate 500 transactions per day has a capacity constraint. A screening program may care about sensitivity at a prespecified false-positive rate. A credit model may be evaluated by expected loss or profit.

No single AUC number contains those decisions.

## Accuracy is the genuinely obvious failure under severe imbalance

If 0.5% of transactions are fraudulent, the classifier

\[
\hat Y=0
\]

for every transaction has

\[
99.5\%
\]

accuracy.

Its recall is zero.

This is a direct consequence of class prevalence because ordinary accuracy weights every observation equally and the negative class dominates the sample count.

That argument should not be transferred mechanically to ROC AUC. The two metrics have different denominators and different meanings.

## A better evaluation stack

For rare-event classification, I would normally keep several layers separate.

### Ranking

Use ROC AUC when the question is whether the score ranks positives above negatives.

### Positive-prediction quality

Use precision-recall curves when the fraction of useful alerts matters and evaluate them at the target prevalence.

### Calibration

Check whether predicted probabilities correspond to observed event frequencies. A perfectly ranked model can still be badly calibrated.

### Threshold performance

Report precision, recall, specificity and the confusion matrix at thresholds that could actually be deployed.

### Decision performance

When costs or benefits can be stated, evaluate expected utility, expected cost, net benefit or another domain-specific objective directly.

The best metric depends on what the classifier is for.

## Conclusion

Class imbalance does not make ROC AUC meaningless.

It makes **some interpretations of ROC performance incomplete**.

ROC coordinates condition on the true class and are therefore insensitive to prevalence in a way that precision is not. Precision asks the operationally different question of how many positive predictions are correct. When positives are rare, even a small false-positive rate can produce many more false alerts than true alerts, and a precision-recall curve makes that visible.

So the useful rule is not "AUPRC beats ROC AUC."

It is:

\[
\boxed{
\begin{aligned}
\text{ROC} &\rightarrow \text{ranking across classes},\\
\text{PR} &\rightarrow \text{quality of positive predictions at a prevalence},\\
\text{decision metric} &\rightarrow \text{what the system is actually built to optimize}.
\end{aligned}
}
\]

That distinction is more accurate, and it survives changes in class balance.

## References

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
