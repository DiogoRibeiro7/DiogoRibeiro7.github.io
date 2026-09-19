---
permalink: '/machine-learning/when_unlabelled_data_makes_semi_supervised_learning_worse/'
title: 'When Unlabelled Data Makes Semi-Supervised Learning Worse'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Covariate Shift
- Negative Transfer
- Self-Training
- Distribution Shift
author_profile: false
seo_title: 'When Unlabelled Data Hurts Semi-Supervised Learning'
seo_description: 'Unlabelled data is not automatically helpful. A controlled self-training experiment shows how covariate shift can turn a small semi-supervised gain into negative transfer while the supervised baseline stays fixed.'
excerpt: >-
  More data is only useful when the assumptions connecting it to the labelled
  problem are reasonable. In a controlled experiment, self-training begins with a
  small positive gain, but shifting only the unlabelled covariates turns that gain
  negative even though the labelled sample and test distribution never change.
summary: >-
  A reproducible example of negative transfer in semi-supervised learning. The
  article separates two questions that are often conflated: whether covariate shift
  makes an SSL method worse relative to its own no-shift control, and whether the
  shifted method actually falls below the supervised baseline.
keywords:
  - semi-supervised learning
  - negative transfer
  - covariate shift
  - self-training
  - unlabelled data
  - distribution shift
classes: wide
date: '2026-09-05'
why_this_exists: >-
  Semi-supervised learning is often introduced with the intuition that additional
  unlabelled observations should improve a classifier when labels are scarce. That
  intuition hides assumptions about how labelled, unlabelled and target data are
  related. A small controlled experiment shows what happens when only the
  unlabelled covariates move.
evidence: >-
  Fifty independently generated binary classification problems with 400 training
  observations, 1,000 test observations and 10 percent exposed training labels.
  The labelled sample and test distribution are fixed within each seed while only
  hidden-label training covariates are translated by increasing amounts.
methodology: >-
  Compares logistic regression trained only on exposed labels with confidence-based
  logistic self-training. For each seed, the unshifted and shifted experiments are
  paired exactly. The article reports semi-supervised gain, shift response, the
  empirical probability of shift harm and the empirical probability of negative
  transfer.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
---

Suppose you have 400 training observations but labels for only 40 of them.

The obvious temptation is to use the other 360 observations somehow. They contain feature information, they reveal the geometry of the training distribution, and throwing them away feels wasteful.

That intuition is the entry point to semi-supervised learning.

It is also where a dangerous sentence often appears:

> Unlabelled data cannot hurt because it contains additional information.

That sentence is false.

Unlabelled observations contain information about a feature distribution. Whether that information helps classification depends on assumptions connecting that distribution to the class boundary we actually care about.

When those assumptions fail, additional unlabelled data can reduce accuracy.

The important word is not *unlabelled*.

It is *relevant*.

## The Paired Comparison

Let

$$
A_S
$$

be the test accuracy of a supervised classifier trained only on the exposed labels, and let

$$
A_{SSL}(\delta)
$$

be the accuracy of a semi-supervised method when the unlabelled covariates have been shifted by a vector $\delta$.

Define the semi-supervised gain

$$
G(\delta)
=
A_{SSL}(\delta)-A_S.
$$

Positive gain means the semi-supervised method beats the supervised comparator.

Negative gain means negative transfer:

$$
\boxed{G(\delta)<0.}
$$

Now define a second quantity,

$$
R(\delta)
=
G(\delta)-G(0).
$$

This is the response to unlabelled covariate shift.

Negative response means the shift made the semi-supervised method worse relative to its own no-shift version:

$$
\boxed{R(\delta)<0.}
$$

These two events are not the same.

A method can be harmed by shift and still remain better than supervised learning:

$$
R(\delta)<0,
\qquad
G(\delta)>0.
$$

Or it can cross all the way into negative transfer:

$$
R(\delta)<0,
\qquad
G(\delta)<0.
$$

That distinction turns out to be useful.

## Hold the Supervised Problem Fixed

If we want to isolate the effect of unlabelled data, the comparison must be paired.

Within each simulated dataset I keep fixed:

- every exposed labelled observation,
- every training label,
- the held-out test sample,
- the supervised model,
- the number of exposed labels.

Only the covariates of observations whose labels are hidden are translated.

If $L$ is the set of labelled training points and $U$ the unlabelled set, then

$$
x_i(\delta)
=
\begin{cases}
x_i, & i\in L,\\
x_i+\delta, & i\in U.
\end{cases}
$$

The supervised logistic regression therefore sees exactly the same labelled data for every $\delta$.

Its accuracy cannot change because of the intervention.

The semi-supervised learner does see the shifted observations, so any change in its result comes from the altered unlabelled sample.

That is the experiment.

## A Small Reproducible Example

The data are deliberately simple.

There are two balanced classes. The first feature contains a separated class signal and the second is noise. Ten percent of the training labels are exposed.

The supervised comparator is logistic regression. The semi-supervised method is confidence-based self-training using the same logistic model as its base estimator.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier

N_TRAIN = 400
N_TEST = 1_000
LABELLED_FRACTION = 0.10
SEPARATION = 4.0
NOISE = 1.0


def make_dataset(seed: int):
    """Generate one balanced binary classification problem."""
    rng = np.random.default_rng(seed)

    def sample(n: int):
        y = np.arange(n, dtype=int) % 2
        rng.shuffle(y)

        x1 = (2 * y - 1) * (SEPARATION / 2)
        x1 = x1 + rng.normal(0.0, NOISE, n)
        x2 = rng.normal(0.0, NOISE, n)
        return np.column_stack((x1, x2)), y

    x_train, y_train = sample(N_TRAIN)
    x_test, y_test = sample(N_TEST)

    labelled = np.zeros(N_TRAIN, dtype=bool)
    for class_label in (0, 1):
        indices = np.flatnonzero(y_train == class_label)
        count = round(indices.size * LABELLED_FRACTION)
        chosen = rng.choice(indices, size=count, replace=False)
        labelled[chosen] = True

    return x_train, y_train, x_test, y_test, labelled
```

The evaluation deliberately uses the same logistic model family on both sides.

```python
def evaluate(x_train, y_train, x_test, y_test, labelled):
    """Return supervised accuracy, self-training accuracy and SSL gain."""
    supervised = LogisticRegression(max_iter=1_000)
    supervised.fit(x_train[labelled], y_train[labelled])
    supervised_accuracy = accuracy_score(
        y_test,
        supervised.predict(x_test),
    )

    partially_observed = np.full_like(y_train, -1)
    partially_observed[labelled] = y_train[labelled]

    self_training = SelfTrainingClassifier(
        LogisticRegression(max_iter=1_000),
        threshold=0.75,
        criterion="threshold",
        max_iter=10,
    )
    self_training.fit(x_train, partially_observed)
    ssl_accuracy = accuracy_score(
        y_test,
        self_training.predict(x_test),
    )

    return (
        supervised_accuracy,
        ssl_accuracy,
        ssl_accuracy - supervised_accuracy,
    )
```

For every seed I first evaluate the unshifted data. I then move only the unlabelled observations along the informative feature.

```python
rows = []

for seed in range(50):
    x_train, y_train, x_test, y_test, labelled = make_dataset(seed)

    sup0, ssl0, gain0 = evaluate(
        x_train,
        y_train,
        x_test,
        y_test,
        labelled,
    )

    for shift in (0.0, 0.5, 1.0, 2.0):
        shifted = x_train.copy()
        shifted[~labelled, 0] += shift

        supervised, ssl, gain = evaluate(
            shifted,
            y_train,
            x_test,
            y_test,
            labelled,
        )

        rows.append(
            {
                "seed": seed,
                "shift": shift,
                "supervised": supervised,
                "self_training": ssl,
                "gain": gain,
                "response": gain - gain0,
            }
        )

results = pd.DataFrame(rows)
```

Because the labelled points are untouched, the supervised result is identical across shifts within each seed.

That is not an incidental property. It is the control that makes the comparison interpretable.

## What Happens

Across the 50 repeated datasets, the average results are:

| Unlabelled shift | Supervised accuracy | Self-training accuracy | Mean SSL gain | Mean shift response | $P(R<0)$ | $P(G<0)$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.9735 | 0.9755 | +0.0020 | 0.0000 | 0% | 22% |
| 0.5 | 0.9735 | 0.9720 | -0.0015 | -0.0035 | 62% | 56% |
| 1.0 | 0.9735 | 0.9666 | -0.0069 | -0.0089 | 76% | 68% |
| 2.0 | 0.9735 | 0.9685 | -0.0050 | -0.0070 | 78% | 66% |

The no-shift semi-supervised method is slightly better on average:

$$
G(0)
\approx
0.0020.
$$

The advantage is small, but positive.

A shift of only 0.5 already changes the sign of the mean gain:

$$
G(0.5)
\approx
-0.0015.
$$

At shift one,

$$
G(1)
\approx
-0.0069,
$$

and the self-training classifier is harmed relative to its own no-shift behaviour in 76 percent of the repeated datasets.

It falls below the supervised comparator in 68 percent.

The supervised accuracy remains

$$
\approx0.9735
$$

throughout because the supervised task has not changed.

Only the unlabelled sample changed.

## More Unlabelled Data Is Not the Same as More Labelled Data

A labelled observation contributes both

$$
x_i
\quad\text{and}\quad
y_i.
$$

An unlabelled observation contributes only

$$
x_i.
$$

The learning algorithm has to infer how that feature distribution is related to the class structure.

Self-training does this through the model's own predictions. High-confidence predictions are promoted to pseudo-labels, after which they become part of the training process.

If the unlabelled covariates come from the same relevant distribution, those pseudo-labels can reinforce a useful boundary.

If they are shifted, confidence can become a liability.

A wrong pseudo-label is no longer merely one incorrect prediction. It becomes training data.

That creates a feedback loop:

$$
\text{biased prediction}
\rightarrow
\text{pseudo-label}
\rightarrow
\text{refit}
\rightarrow
\text{more biased predictions}.
$$

The unlabeled sample has not become useless. It has become informative about the wrong feature distribution.

## Shift Harm and Negative Transfer Are Different

The table contains two probabilities because they answer different questions.

The first is

$$
P\{R(\delta)<0\}.
$$

This asks how often the distribution shift made the semi-supervised method worse than its own no-shift version.

The second is

$$
P\{G(\delta)<0\}.
$$

This asks how often the shifted semi-supervised method became worse than simply ignoring the unlabelled observations.

Those are not interchangeable.

Imagine that no-shift self-training gains five percentage points and the shifted version gains only two. Then

$$
R(\delta)=-0.03,
$$

so the shift harmed the method, but

$$
G(\delta)=+0.02,
$$

so there is no negative transfer.

Conversely, a method can already be poor before the shift. In that case negative transfer may be common even if $R(\delta)$ is close to zero.

The probabilities should therefore be reported separately.

## Why the Response Does Not Have to Be Monotone

There is another detail in the table that I would not smooth away.

The mean response at shift one is

$$
-0.0089,
$$

while at shift two it is slightly less negative:

$$
-0.0070.
$$

At the same time, the empirical probability of harm rises from 76 percent to 78 percent.

So the larger shift is harmful on slightly more datasets but has a less negative mean response.

That is perfectly possible.

Different summaries depend on different parts of the response distribution.

There is therefore no reason to assume

$$
\|\delta_1\| < \|\delta_2\|
\quad\Longrightarrow\quad
R(\delta_1) > R(\delta_2).
$$

Covariate shift interacts with the geometry of the labelled and unlabelled observations, the current decision boundary, confidence thresholds and the pseudo-label sequence.

The response can be directional and non-monotone.

## Direction Matters Too

Writing only the magnitude

$$
r=\|\delta\|
$$

can also be misleading.

A shift of one unit toward a decision boundary is not equivalent to a shift of one unit parallel to it.

In more than one dimension,

$$
\delta_1=(1,0)
$$

and

$$
\delta_2=(0,1)
$$

have the same norm but may interact with the classifier in completely different ways.

For that reason, I would keep the full vector $\delta$ in the experimental record and only compare magnitudes within a fixed direction.

## A Better Evaluation Contract

A comparison such as

$$
A_{SSL} > A_S
$$

on one train-test split is not enough.

At minimum I would keep four invariants: the same labelled subset, the same target distribution, paired seeds between no-shift and shifted conditions, and explicit retention of negative transfer rather than clipping it away.

For repeated experiments I would report at least

$$
E[G],
\qquad
\operatorname{median}(G),
\qquad
P(G<0).
$$

Under an explicit shift experiment I would add

$$
E[R(\delta)]
\qquad\text{and}\qquad
P\{R(\delta)<0\}.
$$

Averages describe magnitude. Sign probabilities describe frequency. Both matter.

## The Supervised Baseline Is More Than a Benchmark

The supervised model is also a control arm.

Because it never sees the hidden-label observations, it should remain invariant when only those observations are shifted.

If the supervised accuracy changes, then something else in the experiment changed too.

This gives a simple falsification check:

$$
A_S(\delta)=A_S(0)
$$

for every shift applied exclusively to the unlabelled sample.

## What This Example Does Not Prove

This small experiment is not an argument against semi-supervised learning.

The no-shift result actually shows the opposite: self-training provides a small average improvement when its assumptions are reasonably aligned with the data.

Nor does the example imply that every covariate shift will hurt.

The effect depends on direction, magnitude, labelled fraction, class geometry, model family, pseudo-label confidence and many other factors.

What the experiment establishes is narrower:

$$
\boxed{
\text{unlabelled observations can cause negative transfer even when the labelled problem itself is unchanged.}
}
$$

That is enough to reject the idea that more unlabelled data is automatically benign.

## The Practical Question

Before adding unlabelled observations to a production training pipeline, I would ask a question more basic than which SSL algorithm to use:

> Why should these unlabelled observations help with the decision boundary represented by the labelled data?

Sometimes the answer is strong. They come from the same acquisition process, time period and population. The cluster or smoothness assumptions are plausible. Pseudo-label confidence is calibrated.

Sometimes the answer is weak. The unlabelled pool was scraped from somewhere else, collected later, selected through another process, or contains a different mixture of populations.

In that situation the dataset is not simply "more data."

It is a distributional assumption.

And assumptions should be stress-tested.

The useful comparison is therefore not only

$$
\text{supervised}
\quad\text{versus}\quad
\text{semi-supervised}.
$$

It is

$$
\boxed{
\text{semi-supervised gain under controlled changes to the unlabelled distribution}.
}
$$

If a modest shift turns a small gain into negative transfer, that tells us something important before deployment does.
