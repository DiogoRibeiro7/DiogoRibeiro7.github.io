---
permalink: '/machine-learning/feature_selection_before_cross_validation_leakage/'
title: 'Preprocessing Inside the Fold: How Feature Selection Before Cross-Validation Invents Accuracy'
categories:
- Machine Learning
tags:
- Model Evaluation
- Cross-Validation
- Data Leakage
- Feature Selection
author_profile: false
seo_title: 'Feature Selection Before Cross-Validation'
seo_description: 'Choosing features on the full dataset and then cross-validating a model on them reports accuracy the model does not have. On pure noise the leak produces 80 to 95 percent accuracy. A simulation measures it for feature selection, target encoding and scaling, and shows the fix.'
excerpt: >-
  One hundred samples, five thousand features, labels assigned by coin
  toss. Keep the ten features most correlated with the label, then
  cross-validate a logistic regression on them: 82 percent accuracy. The
  features are noise, the labels are noise, and the number is real.
summary: >-
  Why any step that looks at the labels is part of the model and must be
  fitted inside each training fold, a simulation of feature selection on
  pure noise showing cross-validated accuracy from 67 to 95 percent when
  the selection sees the held-out labels and 50 percent when it does not,
  how the leak scales with the number of candidate features and shrinks
  with sample size, what happens when a real signal is present, the same
  leak through target encoding, the harmless case of scaling, and the
  pipeline discipline that prevents all of it.
keywords:
  - data leakage
  - cross-validation
  - feature selection
  - selection bias
  - target encoding
  - pipeline
  - model evaluation
classes: wide
date: '2026-03-10'
why_this_exists: >-
  Selecting features, encoding categories or tuning on the whole dataset
  before cross-validation is the most common way a model's reported
  accuracy comes apart in production. This post measures the inflation
  under conditions typical of small, wide datasets, so that the size of
  the risk is concrete and the fix is obviously worth its cost.
evidence: >-
  Simulated datasets with balanced binary labels independent of every
  feature, 50 to 500 samples and 1,000 to 5,000 features, with the k
  most label-correlated features selected either on all data or inside
  each training fold; a version with five informative features and an
  independent test set of 10,000; target encoding of a 200-level category
  on 500 samples; standardisation on 100 samples; 100 replications each.
methodology: >-
  Compares five-fold cross-validated accuracy of a logistic regression
  with selection outside and inside the folds against chance and, in the
  signal case, against the true accuracy of the final model on fresh
  data; repeats the comparison for target encoding and for scaling.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-wafer.jpg
  og_image: /assets/images/headers/photo-wafer.jpg
  overlay_image: /assets/images/headers/photo-wafer.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-wafer.jpg
  twitter_image: /assets/images/headers/photo-wafer.jpg
---
The dataset has a hundred patients and five thousand measurements each. The pipeline is sensible: rank the measurements by their correlation with the diagnosis, keep the ten strongest, and run five-fold cross-validation on a logistic regression over those ten. The cross-validated accuracy is 82 percent. The paper is written.

The diagnosis in this simulation was assigned by a coin toss, and every one of the five thousand measurements is independent noise. The 82 percent is not a fluke of one dataset; it is the average over a hundred of them, and it comes entirely from the order of two steps. The feature selection saw all hundred labels, including the twenty that each cross-validation fold was about to hold out, and among five thousand noise features there are always ten that correlate with a hundred coin tosses by chance. The cross-validation then dutifully measured how well those ten predict the labels they were chosen to predict.

## Anything That Sees the Labels Is the Model

Cross-validation estimates how a procedure performs on data it has not seen. The estimate is honest only if the held-out fold was genuinely unseen by every step that produced the predictions for it. Fitting the classifier is obviously such a step. Selecting features by their relationship to the label is one too: it uses the labels, it changes what the classifier receives, and if it ran on the whole dataset then the held-out labels shaped the features used to predict them. The same is true of encoding a category by the mean label within it, of tuning a hyperparameter on the whole dataset, of choosing a transformation because it improved the fit, and of any decision the analyst made after looking at the labels.

The rule is that the entire pipeline, from raw data to prediction, is refitted from scratch inside each training fold and applied unchanged to the held-out fold. Steps that do not use the labels, such as scaling by the feature's own mean and spread, leak information too, but not the kind that matters, and the simulation below shows the difference.

## A Simulation

Balanced binary labels, features independent of them, and a selection step that keeps the $k$ features with the largest absolute correlation with the label. The cross-validation is run twice per dataset: once with the selection done on all the data before the folds are formed, once with it done inside each training fold.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(0)

def noise_data(n, p, r=rng):
    """Pure noise: features independent of a balanced binary label."""
    X = r.normal(0, 1, (n, p))
    y = np.repeat([0, 1], n // 2)
    r.shuffle(y)
    return X, y

def top_k(X, y, k):
    """Indices of the k features most correlated with the label."""
    yc = y - y.mean()
    corr = np.abs((X - X.mean(0)).T @ yc) / (X.std(0) * yc.std() * len(y) + 1e-12)
    return np.argsort(corr)[-k:]

def cv_accuracy(X, y, k, select_inside, r=rng):
    skf = StratifiedKFold(5, shuffle=True, random_state=int(r.integers(1e9)))
    if not select_inside:
        cols = top_k(X, y, k)                    # the selection sees every label, held-out ones included
    correct = 0
    for tr, te in skf.split(X, y):
        c = top_k(X[tr], y[tr], k) if select_inside else cols
        clf = LogisticRegression(max_iter=1000).fit(X[tr][:, c], y[tr])
        correct += (clf.predict(X[te][:, c]) == y[te]).sum()
    return correct / len(y)

reps = 100
for n, p, k in ((100, 1000, 10), (100, 5000, 10), (100, 5000, 50), (500, 5000, 10), (50, 5000, 10)):
    out, inn = [], []
    for _ in range(reps):
        X, y = noise_data(n, p)
        out.append(cv_accuracy(X, y, k, select_inside=False))
        inn.append(cv_accuracy(X, y, k, select_inside=True))
    print(f"n={n} p={p} k={k}: selection on all data {np.mean(out):.1%}, inside folds {np.mean(inn):.1%}")
```

**Cross-validated accuracy on pure noise**, where the right answer is 50 percent.

| Samples | Candidate features | Features kept | Selection on all data | Selection inside folds |
| --- | --- | --- | --- | --- |
| 100 | 1,000 | 10 | 78.4% | 50.2% |
| 100 | 5,000 | 10 | 82.5% | 49.5% |
| 100 | 5,000 | 50 | 95.1% | 51.0% |
| 500 | 5,000 | 10 | 66.6% | 49.7% |
| 50 | 5,000 | 10 | 91.8% | 51.8% |

With the selection inside the folds, every row is at chance, which is what a hundred coin tosses and five thousand noise features deserve. With the selection outside, the reported accuracy runs from 67 to 95 percent, and it moves in exactly the directions the mechanism predicts. More candidate features give more chances for a spurious correlation, so 5,000 beats 1,000. Keeping more of them lets the classifier combine more spurious signal, so 50 beats 10. Fewer samples make each spurious correlation larger, so 50 samples beat 100 and 500 is the mildest case. The most dangerous dataset, small and wide, is the one where the leak is worst, and it is the kind of dataset on which feature selection is most tempting.

![Cross-validated accuracy on pure noise against the number of candidate features, with the ten most correlated features selected on all the data or inside each training fold. Outside the folds, accuracy on noise climbs above 80 percent as the pool grows; inside, it stays at chance.](/assets/images/figures/cv_selection_leakage.png){: width="1152" height="672" loading="lazy"}

## When There Is Something to Find

A real signal does not make the leak go away; it hides it inside a number that is merely too good.

```python
def signal_data(n, p, r=rng):
    X = r.normal(0, 1, (n, p))
    y = np.repeat([0, 1], n // 2); r.shuffle(y)
    X[:, :5] += 0.6 * (2 * y[:, None] - 1)           # five informative features
    return X, y

out, inn, truth = [], [], []
for _ in range(reps):
    X, y = signal_data(100, 5000)
    out.append(cv_accuracy(X, y, 10, select_inside=False))
    inn.append(cv_accuracy(X, y, 10, select_inside=True))
    c = top_k(X, y, 10); clf = LogisticRegression(max_iter=1000).fit(X[:, c], y)   # the final model
    Xt, yt = signal_data(10000, 5000)
    truth.append((clf.predict(Xt[:, c]) == yt).mean())
print(f"all data {np.mean(out):.1%}, inside folds {np.mean(inn):.1%}, fresh data {np.mean(truth):.1%}")
```

| Five informative features among 5,000, 100 samples | Accuracy |
| --- | --- |
| Cross-validated, selection on all data | 92.0% |
| Cross-validated, selection inside folds | 82.8% |
| Final model on 10,000 fresh samples | 85.3% |

The leaky estimate overstates the deployed model's accuracy by seven points, and the direction of the error is always the same. The estimate with selection inside the folds is slightly below the truth, by two and a half points, because each fold's pipeline is trained on 80 samples rather than 100 and picks the five real features a little less reliably. That is the ordinary pessimism of cross-validation, and it is the safe side to err on. A team choosing between the two numbers to put in a report is choosing between a number that will be beaten in production and one that will be met.

## The Same Leak Through Encoding

Target encoding replaces a category with the average label of the rows in that category. Computed on the whole dataset, it hands each row a summary that includes its own label, and for rare categories the summary is almost the label itself.

```python
def target_encode_cv(n, n_cat, inside, r=rng):
    cat = r.integers(0, n_cat, n)
    y = np.repeat([0, 1], n // 2); r.shuffle(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=int(r.integers(1e9)))
    correct = 0
    if not inside:
        means = np.array([y[cat == c].mean() if np.any(cat == c) else 0.5 for c in range(n_cat)])
    for tr, te in skf.split(cat, y):
        if inside:
            means = np.array([y[tr][cat[tr] == c].mean() if np.any(cat[tr] == c) else 0.5 for c in range(n_cat)])
        clf = LogisticRegression().fit(means[cat[tr]][:, None], y[tr])
        correct += (clf.predict(means[cat[te]][:, None]) == y[te]).sum()
    return correct / n
print(f"encoded on all data {np.mean([target_encode_cv(500, 200, False) for _ in range(reps)]):.1%}, "
      f"inside folds {np.mean([target_encode_cv(500, 200, True) for _ in range(reps)]):.1%}")
```

A category with 200 levels and no relationship to the label, encoded on all 500 rows, gives a cross-validated accuracy of 73 percent; encoded inside the folds, 50 percent. The category carries no information, and the encoding manufactured 23 points of accuracy from the labels it was allowed to see. High-cardinality categories, customer identifiers, postcodes, product codes, are where target encoding is most used and where the leak is largest.

## The Leak That Does Not Matter

Not every fit on the whole dataset is a problem of the same size.

```python
def scale_cv(inside, r=rng):
    X, y = noise_data(100, 20, r)
    skf = StratifiedKFold(5, shuffle=True, random_state=int(r.integers(1e9)))
    correct = 0
    if not inside:
        mu, sd = X.mean(0), X.std(0)
    for tr, te in skf.split(X, y):
        if inside:
            mu, sd = X[tr].mean(0), X[tr].std(0)
        clf = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        correct += (clf.predict((X[te] - mu) / sd) == y[te]).sum()
    return correct / 100
print(f"scaled on all data {np.mean([scale_cv(False) for _ in range(reps)]):.1%}, inside folds {np.mean([scale_cv(True) for _ in range(reps)]):.1%}")
```

Standardising features with the mean and spread of the whole dataset gives 49.7 percent on noise against 50.1 percent inside the folds. The scaling uses the held-out rows, but it does not use their labels, and the feature means it borrows from them are a negligible influence on a classifier. It is still wrong in principle, and a pipeline object fixes it for free, but the reader who wants to know where to look first should look at the steps that touch the labels.

## Why It Keeps Happening

The leak survives because the leaky pipeline is easier to write. Selecting features once and passing the reduced matrix to a cross-validation function is two lines; selecting inside every fold means the selection has to be part of the object that gets cross-validated, and the analyst has to have thought of the selection as part of the model. Every modern library makes this possible, through a pipeline that chains the selector, the encoder and the classifier so that cross-validation refits all three on each training fold. What the library cannot do is make the analyst put the step inside the pipeline instead of before it.

The second reason is that the inflated number is the one people want. A 92 percent model is a paper and a launch; an 83 percent model is a project. The discipline of the inside-fold estimate is the discipline of preferring the number that will be true.

## What to Do

1. **Treat every label-using step as part of the model**: feature selection, target encoding, hyperparameter tuning, and any choice made after looking at the labels.
2. **Put the whole pipeline inside the cross-validation**, so that each training fold refits every step and the held-out fold sees none of it. Use a pipeline object; do not hand-roll it.
3. **Expect small, wide datasets to be the worst case**, and be most suspicious of high accuracy exactly where the sample is small and the feature pool is large.
4. **Use nested cross-validation** when a selection or tuning step is itself chosen by cross-validation; the outer loop estimates the whole procedure.
5. **Keep a test set that nothing has touched**, and score the final pipeline on it once. It is the check that catches whatever the pipeline object missed.
6. **Run the procedure on shuffled labels** as a sanity check; a pipeline that scores well above chance on permuted labels has a leak.

## References

- Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *Proceedings of the National Academy of Sciences*, 99(10), 6562-6566.
- Simon, R., Radmacher, M. D., Dobbin, K., & McShane, L. M. (2003). Pitfalls in the use of DNA microarray data for diagnostic and prognostic classification. *Journal of the National Cancer Institute*, 95(1), 14-18.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), section 7.10.2, "The wrong and right way to do cross-validation". Springer.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
