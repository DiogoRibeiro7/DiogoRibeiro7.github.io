---
permalink: '/machine-learning/annotator_agreement_accuracy_ceiling/'
title: 'Annotator Disagreement Sets the Ceiling: What Label Noise Does to Every Number You Report'
categories:
- Machine Learning
tags:
- Model Evaluation
- Data Quality
- Label Noise
- Machine Learning
author_profile: false
seo_title: 'Inter-Annotator Agreement and the Accuracy Ceiling'
seo_description: 'If annotators disagree ten percent of the time, a perfect model scores 90 percent and a five-point difference between two models measures four. A simulation shows the ceiling, the compressed gap, the evaluation set it costs, and what majority voting recovers.'
excerpt: >-
  Two annotators agree on 82 percent of items. A model that predicts the
  truth perfectly will score 90 percent against their labels, and two
  models five points apart will look four points apart. The evaluation
  set needed to tell them apart grows by 60 percent.
summary: >-
  Why disagreement between annotators is a property of the task rather
  than a defect to be scolded away, what raw agreement and Cohen's kappa
  each measure and why they diverge under class imbalance, a simulation
  showing the accuracy ceiling that label error imposes, the compression
  of differences between models by a factor of one minus twice the error
  rate, the evaluation set that costs, how majority voting recovers most
  of it, and what asymmetric annotation error does to recall and
  precision.
keywords:
  - inter-annotator agreement
  - Cohen's kappa
  - label noise
  - accuracy ceiling
  - model evaluation
  - majority vote
  - data quality
classes: wide
date: '2025-12-15'
why_this_exists: >-
  Evaluation sets are treated as ground truth even when the people who
  made them disagreed with each other, and the consequences are quiet:
  models appear to plateau, real improvements fail to reach significance,
  and leaderboards compress. This post quantifies each of those effects
  and shows what buys them back.
evidence: >-
  Simulated labelling of 200,000 items at 50 and 10 percent prevalence,
  with independent annotator error rates from 2 to 30 percent, majority
  votes of one to seven annotators, models of known true accuracy scored
  against the noisy labels, and an asymmetric error model in which
  annotators miss positives more often than they invent them.
methodology: >-
  Measures raw agreement and Cohen's kappa against the annotator error
  rate at two prevalences, the measured accuracy of a model that predicts
  the truth perfectly, the measured gap between two models five points
  apart against the one minus twice the error prediction, the evaluation
  set required for 80 percent power, the effective label error after
  majority voting, and the measured recall and precision under asymmetric
  error.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-library.jpg
  og_image: /assets/images/headers/photo-library.jpg
  overlay_image: /assets/images/headers/photo-library.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-library.jpg
  twitter_image: /assets/images/headers/photo-library.jpg
---
The model has been stuck at 89 percent for three months. Four architectures, a larger training set and a round of hyperparameter search have each bought a point and given it back. The team starts to talk about the task being saturated.

Two annotators labelled a sample of the evaluation set independently and agreed on 82 percent of items. That corresponds to each of them being wrong about ten percent of the time, and it means a model that predicted the true label of every item would score 90 percent against those labels. The model is not stuck below a ceiling of 100; it is pressed against a ceiling of 90, and every point of apparent improvement is being measured with an instrument whose own error is larger than the effect.

## Disagreement Is Information About the Task

Annotators disagree because tasks are ambiguous, guidelines are incomplete, and judgement varies. For some tasks the disagreement rate is essentially the task's difficulty, and no amount of training removes it. That makes agreement a measurement to take and report, not a failure to hide.

Two summaries are usual. Raw agreement is the share of items two annotators label the same way. Cohen's kappa corrects for the agreement that would occur by chance given each annotator's label frequencies. The two diverge sharply when classes are imbalanced, and the direction of the divergence is the source of a great deal of confusion.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def annotate(truth, err, r=rng):
    """Each annotator flips the true label independently with probability err."""
    return np.where(r.random(len(truth)) < err, 1 - truth, truth)

def kappa(a, b):
    """Cohen's kappa between two label vectors."""
    po = np.mean(a == b)
    pe = np.mean(a == 1) * np.mean(b == 1) + np.mean(a == 0) * np.mean(b == 0)
    return (po - pe) / (1 - pe)

n = 200_000
for prev in (0.5, 0.1):
    truth = (rng.random(n) < prev).astype(int)
    for err in (0.02, 0.05, 0.10, 0.20):
        a, b = annotate(truth, err), annotate(truth, err)
        print(f"prevalence {prev:.0%}, error {err:.0%}: agreement {np.mean(a == b):.1%}, kappa {kappa(a, b):.2f}")
```

| Annotator error | Agreement | Kappa at 50% prevalence | Kappa at 10% prevalence |
| --- | --- | --- | --- |
| 2% | 96.1% | 0.92 | 0.81 |
| 5% | 90.5% | 0.81 | 0.61 |
| 10% | 82.0% | 0.64 | 0.39 |
| 20% | 68.0% | 0.36 | 0.16 |

Raw agreement does not depend on prevalence at all: the same annotators with the same error rate agree on 82 percent of items whether the positive class is half the data or a tenth. Kappa falls steeply as the class becomes rarer, because when almost everything is negative, agreeing is easy and chance agreement is high, so a given amount of real agreement is worth less credit. A kappa of 0.39 on a rare class and a kappa of 0.64 on a balanced one describe annotators of identical quality. Reporting kappa without the prevalence invites the reader to conclude the wrong thing, which is why both belong in the same sentence.

## The Ceiling

The consequence for evaluation is direct. If the reference label is one annotator's judgement and that annotator is wrong with probability $e$, then a model predicting the true label scores $1 - e$.

```python
truth = (rng.random(n) < 0.5).astype(int)
for err in (0.0, 0.02, 0.05, 0.10, 0.20):
    labels = annotate(truth, err)
    print(f"error {err:.0%}: a perfect model scores {np.mean(truth == labels):.1%}")
```

| Annotator error | Measured accuracy of a perfect model |
| --- | --- |
| 0% | 100.0% |
| 2% | 98.0% |
| 5% | 95.0% |
| 10% | 89.8% |
| 20% | 79.9% |

At ten percent annotator error, 90 percent is not a plateau to break through; it is the maximum the instrument can display. A model that scored 95 would be a model that agreed with the annotator's mistakes, which is worse, not better.

![Highest accuracy a model can appear to reach against the error rate of a single annotator, for labels from one annotator and from majority votes of three, five and seven. One annotator caps the score at one minus the error rate; majority voting lifts the ceiling sharply.](/assets/images/figures/annotator_noise_ceiling.png){: width="1152" height="672" loading="lazy"}

## Differences Get Compressed Too

The ceiling is the visible problem. The quieter one is that label noise shrinks the gap between models, which is what evaluation is usually for.

```python
truth = (rng.random(n) < 0.5).astype(int)
for err in (0.0, 0.05, 0.10, 0.20):
    labels = annotate(truth, err)
    preds_a = np.where(rng.random(n) < 0.90, truth, 1 - truth)      # 90% true accuracy
    preds_b = np.where(rng.random(n) < 0.85, truth, 1 - truth)      # 85% true accuracy
    ma, mb = np.mean(preds_a == labels), np.mean(preds_b == labels)
    print(f"error {err:.0%}: A {ma:.1%}, B {mb:.1%}, gap {ma - mb:.2%}, predicted {(1 - 2 * err) * 0.05:.2%}")
```

| Annotator error | Model A measured | Model B measured | Measured gap | Predicted: (1 − 2e) × 5 points |
| --- | --- | --- | --- | --- |
| 0% | 89.9% | 85.0% | 4.98% | 5.00% |
| 5% | 86.0% | 81.5% | 4.46% | 4.50% |
| 10% | 82.0% | 78.1% | 3.93% | 4.00% |
| 20% | 74.1% | 71.0% | 3.10% | 3.00% |

A true five-point difference measures four points at ten percent error and three at twenty, following $(1 - 2e)$ closely. The mechanism is simple: on the items the annotator got wrong, the better model is penalised more often than the worse one, because it agrees with the truth more often. Noise does not merely add variance here, it shrinks the signal.

That compression has a price in sample size, since the same effect must be detected from a smaller measured gap.

```python
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
for err in (0.0, 0.05, 0.10, 0.20):
    gap = (1 - 2 * err) * 0.05
    print(f"error {err:.0%}: measurable gap {gap:.2%}, items {(z / gap) ** 2 * 0.25:,.0f}, "
          f"relative {(1 / (1 - 2 * err) ** 2):.1f}x")
```

| Annotator error | Measurable gap | Items needed for 80% power | Relative to clean labels |
| --- | --- | --- | --- |
| 0% | 5.00% | 785 | baseline |
| 5% | 4.50% | 969 | 1.2× |
| 10% | 4.00% | 1,226 | 1.6× |
| 20% | 3.00% | 2,180 | 2.8× |

The requirement grows as $(1 - 2e)^{-2}$. Twenty percent annotator error nearly triples the evaluation set needed to distinguish two models that genuinely differ by five points, and no amount of extra data changes the ceiling, only the ability to resolve differences below it.

## What Majority Voting Buys

If annotator errors are independent, a majority vote has to be wrong in more than half the annotators at once, which is much rarer.

```python
for m in (1, 3, 5, 7):
    for err in (0.05, 0.10, 0.20):
        votes = np.zeros(n)
        for _ in range(m):
            votes += annotate(truth, err)
        maj = (votes > m / 2).astype(int)
        print(f"{m} annotators at error {err:.0%}: effective error {np.mean(maj != truth):.2%}")
```

| Annotators | Effective error at 5% | At 10% | At 20% |
| --- | --- | --- | --- |
| 1 | 5.04% | 10.15% | 20.09% |
| 3 | 0.71% | 2.79% | 10.43% |
| 5 | 0.11% | 0.84% | 5.83% |
| 7 | 0.02% | 0.25% | 3.34% |

Three annotators cut a ten percent error to 2.8 percent and lift the ceiling from 90 to 97.2 percent; five take it to 99.2. The returns are steep at first and then flatten, and the practical reading is that triple-labelling an evaluation set is usually worth far more than tripling its size. An evaluation set of a thousand items labelled three times each is a better instrument than three thousand items labelled once, whenever the question is about differences near the ceiling.

The independence assumption is doing real work here. Annotators who share a misleading guideline, or who all misread the same ambiguous category the same way, make correlated errors that no vote removes. Correlated error is why majority voting raises the ceiling without ever reaching 100 percent in practice, and why adjudicating disagreements against a written definition beats adding a fourth voter.

## Asymmetric Error Hits Recall and Precision Differently

Annotators rarely make symmetric mistakes. On a rare, effortful class they miss positives far more often than they invent them, and the consequences differ by metric.

```python
truth10 = (rng.random(n) < 0.10).astype(int)
for miss, false_pos in ((0.20, 0.01), (0.30, 0.02)):
    lab = truth10.copy()
    lab = np.where((truth10 == 1) & (rng.random(n) < miss), 0, lab)
    lab = np.where((truth10 == 0) & (rng.random(n) < false_pos), 1, lab)
    print(f"miss {miss:.0%}, invent {false_pos:.0%}: labelled prevalence {lab.mean():.1%}, "
          f"perfect model's measured recall {np.mean(lab[truth10 == 1] == 1):.0%}, "
          f"precision {np.mean(truth10[lab == 1] == 1):.0%}")
```

| Annotation error | Labelled prevalence (true 10%) | Perfect model's measured recall | Measured precision |
| --- | --- | --- | --- |
| Miss 20% of positives, invent 1% | 8.9% | 80% | 90% |
| Miss 30% of positives, invent 2% | 8.9% | 71% | 79% |

A model that finds every true positive is scored at 80 percent recall, because a fifth of the true positives are not labelled as positive. Worse, the model's correct detections of missed positives are counted as false positives, so its measured precision falls too. The prevalence in the labelled data understates the truth, which then propagates into any threshold tuned on that data and any business estimate of how often the event occurs.

This is the case where the ceiling is not merely low but misleading about which model is better: a model tuned to maximise measured precision will learn to avoid exactly the hard positives the annotators missed.

## What to Do

1. **Double-label a sample of every evaluation set** and report raw agreement and kappa with the class prevalence beside them. Without that number the accuracy figure has no scale.
2. **State the ceiling in the evaluation report.** If annotators disagree ten percent of the time, say that a perfect model scores about 90, so nobody spends a quarter chasing the last ten points.
3. **Expect measured differences to be compressed by (1 − 2e)** and size the evaluation set accordingly; at twenty percent error it takes nearly three times the items.
4. **Prefer more labels per item over more items** when the decision is between close models, since voting raises the ceiling while extra items only reduce variance.
5. **Adjudicate disagreements against a written definition** rather than adding voters, because correlated annotator error survives any vote.
6. **Check whether error is asymmetric** on rare classes, and treat measured recall and prevalence as understated when annotators miss positives.

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46.
- Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543-549.
- Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics*, 34(4), 555-596.
- Frénay, B., & Verleysen, M. (2014). Classification in the presence of label noise: a survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845-869.
- Northcutt, C. G., Athalye, A., & Mueller, J. (2021). Pervasive label errors in test sets destabilize machine learning benchmarks. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*.
- Snow, R., O'Connor, B., Jurafsky, D., & Ng, A. Y. (2008). Cheap and fast, but is it good? Evaluating non-expert annotations for natural language tasks. *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing*, 254-263.
- Aroyo, L., & Welty, C. (2015). Truth is a lie: crowd truth and the seven myths of human annotation. *AI Magazine*, 36(1), 15-24.
