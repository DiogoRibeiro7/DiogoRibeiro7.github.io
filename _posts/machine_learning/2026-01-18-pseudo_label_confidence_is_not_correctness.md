---
permalink: '/machine-learning/pseudo_label_confidence_is_not_correctness/'
title: 'Pseudo-Label Confidence Is Not the Same as Correctness'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Pseudo-Labels
- Self-Training
- Calibration
- Confirmation Bias
- Uncertainty
author_profile: false
seo_title: 'Why High-Confidence Pseudo-Labels Can Still Be Wrong'
seo_description: 'A 0.95 model probability is not a 95 percent correctness guarantee. Pseudo-label selection depends on calibration, distribution shift, selection effects and repeated feedback through retraining.'
excerpt: >-
  Self-training promotes model predictions into training labels. The usual
  safeguard is confidence thresholding, but confidence is produced by the same
  model that created the pseudo-label. Unless that confidence is calibrated on the
  relevant distribution, a threshold such as 0.95 has no direct interpretation as
  a 95 percent probability that accepted pseudo-labels are correct.
summary: >-
  A mathematical analysis of confidence-based pseudo-labelling. The article
  separates prediction probability from empirical correctness, shows why
  confidence thresholding creates a selected subpopulation, quantifies how even a
  modest pseudo-label error rate can dominate a small labelled set, and explains
  why retraining can amplify systematic mistakes. It develops calibration,
  selective-risk and stability checks for evaluating self-training pipelines.
keywords:
- pseudo-label confidence
- self-training
- semi-supervised learning
- calibration
- confirmation bias
- selective classification
- pseudo-label error
classes: wide
date: '2026-01-18'
why_this_exists: >-
  Confidence thresholds are often treated as though they convert uncertain model
  predictions into nearly reliable labels. That interpretation is only justified
  under calibration assumptions that are rarely checked on the accepted
  unlabelled population. Because accepted pseudo-labels are then reused as
  training data, small systematic errors can become self-reinforcing.
evidence: >-
  Classical pseudo-labelling and self-training, modern calibration literature,
  consistency-based semi-supervised methods and work on confirmation bias in
  pseudo-label pipelines.
methodology: >-
  Separate model score from correctness probability, condition explicitly on the
  acceptance event, quantify erroneous pseudo-label mass relative to the labelled
  sample, and treat each self-training iteration as a feedback system rather than
  as one independent prediction step.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
---

A common semi-supervised learning pipeline looks reassuringly conservative.

Train a classifier on the labelled observations. Predict class probabilities for the unlabelled observations. Keep only predictions above a confidence threshold such as

$$
0.95.
$$

Treat those predictions as labels. Retrain.

The intuition is obvious:

> if the model is 95 percent confident, the pseudo-label is probably correct.

That sentence contains an assumption.

A model score of

$$
0.95
$$

is not, by itself, a statement that 95 percent of such predictions are correct.

It is a number produced by a fitted model.

To interpret it probabilistically, we need calibration.

To use it safely for pseudo-labelling, we need something stronger: calibration on the particular unlabelled population from which pseudo-labels are selected.

And because those pseudo-labels are fed back into training, even a small systematic error can become part of the next model rather than remaining an isolated prediction mistake.

The issue is not that pseudo-labelling is inherently unreliable.

The issue is that confidence and correctness are different statistical objects.

## The Basic Self-Training Rule

Suppose we have labelled observations

$$
\mathcal D_L
=
\{(x_i,y_i)\}_{i=1}^{n_L}
$$

and unlabelled observations

$$
\mathcal D_U
=
\{x_j\}_{j=1}^{n_U}.
$$

Let a classifier produce class probabilities

$$
\hat p_k(x)
=
\widehat P(Y=k\mid X=x).
$$

For each unlabelled observation, define the predicted class

$$
\hat y(x)
=
\arg\max_k \hat p_k(x)
$$

and its reported confidence

$$
q(x)
=
\max_k \hat p_k(x).
$$

Confidence-based pseudo-labelling accepts the observation when

$$
q(x)\geq\tau
$$

for some threshold $\tau$, perhaps

$$
\tau=0.95.
$$

The new pseudo-labelled set is

$$
\widehat{\mathcal D}_U(\tau)
=
\left\{
(x,\hat y(x)):
q(x)\geq\tau
\right\}.
$$

The next model is then trained using some combination of

$$
\mathcal D_L
$$

and

$$
\widehat{\mathcal D}_U(\tau).
$$

Nothing in this construction guarantees

$$
P\{\hat y(X)=Y\mid q(X)\geq0.95\}
=
0.95.
$$

That equality is a calibration statement.

It has to be earned.

## Confidence Is a Model Output

Consider binary classification.

Let

$$
\hat p(x)
$$

be the model's estimated probability that

$$
Y=1.
$$

If the model outputs

$$
\hat p(x)=0.95,
$$

then the mathematical fact is simply that the fitted model assigned the value 0.95 to that observation.

The empirical interpretation

$$
P(Y=1\mid \hat p(X)=0.95)=0.95
$$

requires calibration.

A perfectly calibrated probabilistic classifier satisfies, informally,

$$
P(Y=1\mid \hat p(X)=p)=p
$$

for relevant values of $p$.

For predicted-class confidence, define

$$
C
=
\mathbb 1\{\hat Y=Y\}.
$$

A corresponding calibration condition is

$$
E[C\mid q(X)=q]
=
q.
$$

Only under such a relation can model confidence be interpreted directly as correctness frequency.

Neural networks, boosted trees, logistic models under misspecification, and many other classifiers can all be miscalibrated.

A model can rank observations correctly while being systematically overconfident.

That distinction matters greatly for pseudo-label selection.

## A Threshold Is Not a Correctness Guarantee

Suppose the accepted pseudo-labels all have reported confidence between

$$
0.95
$$

and

$$
0.99.
$$

One might expect their error rate to be at most about five percent.

But imagine that the classifier is overconfident on the unlabelled population and that accepted predictions are correct only 80 percent of the time.

Then

$$
P\{\hat Y=Y\mid q\geq0.95\}
=
0.80,
$$

not 0.95.

The threshold has selected observations according to the model's own score.

It has not changed the model's score into a calibrated probability.

This difference is easy to miss because the number

$$
0.95
$$

already looks probabilistic.

The semantic interpretation comes from calibration, not from the presence of a decimal between zero and one.

## Perfect Calibration Would Help

It is useful to state the favourable case clearly.

Suppose predicted-class confidence is perfectly calibrated on the relevant population:

$$
E[C\mid q]=q.
$$

Then for the selected set

$$
S_\tau
=
\{x:q(x)\geq\tau\},
$$

we have

$$
E[C\mid S_\tau]
=
E[q\mid q\geq\tau].
$$

Because every accepted confidence is at least $\tau$,

$$
E[q\mid q\geq\tau]
\geq\tau.
$$

Therefore,

$$
\boxed{
E[C\mid q\geq\tau]
\geq\tau
}
$$

under perfect calibration.

So if calibration genuinely holds on the accepted unlabelled population, a threshold such as 0.95 has a useful interpretation.

The problem is not confidence thresholding itself.

The problem is assuming the condition without checking it.

## Calibration on the Labelled Set Is Not Enough

Suppose the model is well calibrated on a held-out labelled sample.

That is good evidence.

It still does not establish calibration on the unlabelled pool.

Let

$$
P_L(X,Y)
$$

denote the labelled-data distribution and

$$
P_U(X,Y)
$$

the population from which unlabelled observations were drawn.

Calibration estimated under

$$
P_L
$$

need not transfer to

$$
P_U.
$$

If the feature distribution changes,

$$
P_L(X)\neq P_U(X),
$$

or if the conditional relationship changes,

$$
P_L(Y\mid X)\neq P_U(Y\mid X),
$$

the calibration relationship can change as well.

Even if the classifier remains accurate overall, its high-confidence region may behave differently.

That is exactly the region self-training selects.

## Pseudo-Label Selection Creates a New Population

The accepted pseudo-labels are not a random sample from the unlabelled pool.

They satisfy

$$
q(X)\geq\tau.
$$

So the pseudo-labelled training distribution is conditional:

$$
P_U(X,Y\mid q(X)\geq\tau).
$$

That distribution can differ sharply from the full unlabelled distribution.

For example, the model may be most confident on:

- majority-class observations,
- observations close to familiar training examples,
- regions with low epistemic uncertainty,
- regions where its inductive bias is strongest,
- or groups over-represented in the labelled data.

The pseudo-labelled set can therefore be heavily selected even when the original unlabelled pool is representative.

Self-training does not merely add data.

It adds data through a model-dependent sampling mechanism.

## High Confidence Can Be Concentrated in the Wrong Places

Suppose a binary classifier is trained from a small labelled sample.

The true class boundary is nonlinear, but the fitted model is linear.

Far from its estimated linear boundary, the model may produce probabilities near zero or one.

Those predictions can be highly confident because the model extrapolates strongly.

Yet some of those regions may lie on the wrong side of the true nonlinear boundary.

The problem is systematic.

The model is not uncertain because, inside its own model class, the prediction is clear.

Confidence reflects certainty conditional on the fitted representation and model.

It does not measure whether the model class itself is correct.

This distinction becomes important whenever model misspecification is plausible.

## Confidence and Epistemic Uncertainty Are Different

A softmax probability or logistic probability is often treated as though it summarises all relevant uncertainty.

It does not.

A model can output

$$
0.999
$$

in a region poorly represented by training data.

The score may reflect a large logit magnitude rather than strong empirical support.

In discriminative classification, the predictive score is generated by the fitted function.

It need not increase uncertainty simply because the observation lies far from the labelled support.

Some model families extrapolate with increasing confidence.

For pseudo-labelling, this means that thresholding can preferentially accept points that are confidently wrong because of extrapolation.

Distance from labelled support, ensemble disagreement, posterior uncertainty, density estimates or other diagnostics may therefore provide information that the class probability alone does not.

## Accepted Pseudo-Labels Can Outnumber Real Labels Very Quickly

Suppose we begin with

$$
n_L=100
$$

human-labelled observations.

Now assume the unlabelled pool is large and the 0.95 threshold accepts

$$
m=5000
$$

pseudo-labels.

If the accepted pseudo-label error rate is only

$$
\varepsilon=0.05,
$$

then the expected number of incorrect pseudo-labels is

$$
m\varepsilon
=
5000\times0.05
=
250.
$$

So the training set contains roughly

$$
250
$$

incorrect synthetic labels versus only

$$
100
$$

human labels.

This does not imply that the pseudo-labelled model must fail.

Correct pseudo-labels also contribute information, and observations differ in leverage.

But the arithmetic exposes an important asymmetry.

A seemingly small error rate can correspond to a large absolute amount of wrong supervision when

$$
n_U\gg n_L.
$$

The relevant quantity is not only pseudo-label accuracy.

It is pseudo-label error mass relative to the trusted labelled signal.

## A Weighted Objective Makes the Trade-Off Explicit

A typical self-training objective can be written as

$$
L(\theta)
=
\sum_{i\in L}
\ell\bigl(f_\theta(x_i),y_i\bigr)
+
\lambda
\sum_{j\in U_\tau}
w_j
\ell\bigl(f_\theta(x_j),\hat y_j\bigr),
$$

where

- $U_\tau$ is the accepted pseudo-labelled set,
- $w_j$ is an optional confidence or reliability weight,
- $\lambda$ controls the contribution of pseudo-labelled observations.

This expression makes the central balance visible.

The labelled term contains fewer observations but trusted targets.

The pseudo-labelled term may contain far more observations but noisy targets generated by the current model.

If the second term becomes numerically dominant, the next model can be trained more strongly to reproduce its own previous decisions than to respect the original labelled evidence.

That is the mechanism behind confirmation bias in self-training.

## One Wrong Prediction Is Not the Main Problem

A standard supervised model makes prediction errors.

Those errors do not normally become new training labels automatically.

Self-training changes that.

At iteration $t$, suppose the model predicts

$$
\hat y_j^{(t)}
$$

for an unlabelled observation.

If that pseudo-label is accepted, iteration $t+1$ is fitted using

$$
\hat y_j^{(t)}
$$

as a target.

The process is

$$
f^{(t)}
\rightarrow
\hat y^{(t)}
\rightarrow
\mathcal D^{(t+1)}
\rightarrow
f^{(t+1)}.
$$

A prediction error has therefore crossed the boundary from output to input.

That is why self-training should be understood as a feedback system.

## A Simple Feedback Model

Let

$$
e_t
$$

denote the error rate among pseudo-labels accepted at iteration $t$.

Suppose the next model's pseudo-label error behaves approximately like

$$
e_{t+1}
=
\alpha e_t+\beta,
$$

where

- $\alpha$ represents amplification or persistence of existing pseudo-label errors,
- $\beta$ represents fresh error introduced by limited labelled information, noise or model misspecification.

If

$$
0\leq\alpha<1,
$$

the process has a stable fixed point

$$
e^\star
=
\frac{\beta}{1-\alpha}.
$$

If $\alpha$ is small, errors are corrected quickly.

If $\alpha$ is close to one, errors persist.

If the effective feedback is stronger than this simple stable model permits, self-training can move away from the labelled solution rather than toward it.

The formula is only a schematic model.

Its purpose is to emphasise that repeated pseudo-labelling is dynamic.

Evaluating only the accuracy of the first pseudo-label batch can miss what happens after those labels influence later models.

## Class Imbalance Makes Thresholding Asymmetric

Suppose class zero is common and class one is rare.

A classifier trained on few labels may be highly confident about the majority class and cautious about the minority class.

With a common threshold $\tau$, accepted pseudo-labels may therefore satisfy

$$
|\widehat{\mathcal D}_{U,0}|
\gg
|\widehat{\mathcal D}_{U,1}|.
$$

The retrained model now sees an even more majority-heavy target distribution.

That can increase the next round of majority-class confidence, creating a feedback loop in class proportions.

The process can be written schematically as

$$
\text{initial imbalance}
\rightarrow
\text{asymmetric confidence}
\rightarrow
\text{asymmetric acceptance}
\rightarrow
\text{stronger effective imbalance}.
$$

A global confidence threshold does not control this mechanism.

Class-specific thresholds, distribution alignment or explicit prior constraints may be needed when imbalance matters.

## Calibration Can Differ by Class

Even when global expected calibration error looks acceptable, class-conditional calibration can be poor.

For class $k$, one may examine

$$
P(Y=k\mid \hat Y=k,q=t).
$$

A classifier can be well calibrated on average while being overconfident for one class and underconfident for another.

That matters for pseudo-labelling because accepted samples are partitioned by predicted class.

If class-specific calibration differs, the pseudo-label error rate also differs.

Reporting only one global calibration statistic can therefore conceal the part of the model that creates most pseudo-label noise.

## Calibration Can Differ by Subgroup

The same issue appears across populations.

Suppose

$$
G
$$

denotes a subgroup, device type, geography, acquisition source or time period.

A model may satisfy approximate global calibration,

$$
E[C\mid q]\approx q,
$$

while failing conditionally:

$$
E[C\mid q,G=g]\neq q.
$$

If pseudo-label acceptance rates also differ by group, the training data can become concentrated in the groups where the model is most confident rather than the groups where additional supervision is most needed.

The resulting feedback is not merely a calibration problem.

It can become a representation problem because under-represented regions receive fewer pseudo-labels and therefore less influence during retraining.

## Distribution Shift Makes Confidence Thresholds Fragile

Suppose a threshold was selected using labelled validation data from distribution

$$
P_0.
$$

Later, unlabelled observations come from

$$
P_1.
$$

Even if the model continues producing scores in the same numerical range, the mapping

$$
q
\mapsto
P(C=1\mid q)
$$

may change.

A threshold calibrated under $P_0$ can therefore have a different selective risk under $P_1$.

Define the pseudo-label error rate at threshold $\tau$ as

$$
R_{\text{PL}}(\tau)
=
P\{\hat Y\neq Y\mid q\geq\tau\}.
$$

The relevant quantity for self-training is not merely $\tau$.

It is

$$
R_{\text{PL}}(\tau).
$$

Under shift, that risk must be re-estimated or at least stress-tested.

## Precision-Coverage Trade-Off

Raising the confidence threshold usually reduces the number of accepted pseudo-labels.

Define coverage

$$
\Gamma(\tau)
=
P(q\geq\tau)
$$

and pseudo-label error

$$
R_{\text{PL}}(\tau)
=
P(\hat Y\neq Y\mid q\geq\tau).
$$

A useful pseudo-label selection curve is therefore

$$
\tau
\mapsto
\left(
\Gamma(\tau),
R_{\text{PL}}(\tau)
\right).
$$

High thresholds generally reduce coverage.

They may reduce error as well, but that relationship should be measured rather than assumed.

The relevant engineering question is not

> Which threshold sounds conservative?

It is

> How much pseudo-labelled coverage do we obtain for a tolerable error rate?

That is a selective-classification problem.

## Why 0.95 Is Often an Arbitrary Number

Thresholds such as

$$
0.90,\quad0.95,\quad0.99
$$

look principled because they are familiar probability levels.

But without calibration they are simply hyperparameters.

A threshold of 0.95 in one model can correspond to lower empirical correctness than 0.80 in another.

Even within the same model, the meaning can vary by class, subgroup and time period.

The threshold should therefore be selected through an explicit validation objective, not because 0.95 feels safe.

## Calibration Methods Help, but They Do Not Solve Everything

Post-hoc calibration methods include:

- temperature scaling,
- Platt scaling,
- isotonic regression,
- beta calibration,
- and more flexible calibration models.

For neural classifiers, temperature scaling is often a useful baseline.

If logits are

$$
z_k(x),
$$

temperature scaling replaces them with

$$
\frac{z_k(x)}{T}
$$

before applying softmax.

A temperature

$$
T>1
$$

usually softens overconfident predictions.

This can improve calibration on validation data without changing the predicted class ranking.

But post-hoc calibration does not solve distribution shift automatically.

A calibration map fitted on one labelled distribution can fail on another.

Nor does it remove model misspecification.

It improves the interpretation of scores under the conditions where the calibration relationship remains valid.

## Use Trusted Labels to Measure Selective Accuracy

If enough labelled validation data exist, evaluate accuracy conditionally on the same acceptance rule that will be used for pseudo-labelling.

For threshold $\tau$, estimate

$$
\widehat A(\tau)
=
\frac{
\sum_i
\mathbb 1\{q_i\geq\tau\}
\mathbb 1\{\hat y_i=y_i\}
}{
\sum_i
\mathbb 1\{q_i\geq\tau\}
}.
$$

Also estimate coverage,

$$
\widehat\Gamma(\tau)
=
\frac{1}{n}
\sum_i
\mathbb 1\{q_i\geq\tau\}.
$$

Plotting

$$
\widehat A(\tau)
$$

against

$$
\widehat\Gamma(\tau)
$$

is more informative than quoting one confidence threshold.

It directly evaluates the selection rule.

Confidence becomes operational only after we know how it behaves empirically.

## Confidence Bins Are Useful but Can Hide the Tail

Reliability diagrams group predictions into bins.

For bin $B_m$, compare

$$
\operatorname{conf}(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m}
q_i
$$

with

$$
\operatorname{acc}(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m}
\mathbb 1\{\hat y_i=y_i\}.
$$

That is helpful.

But pseudo-labelling often depends almost entirely on the extreme right tail of the confidence distribution.

A model can have a reasonable global calibration error while the bin

$$
q\geq0.95
$$

is poorly calibrated because it contains few validation observations.

For self-training, tail calibration matters disproportionately.

Report uncertainty there.

## A Confidence Threshold Should Have an Interval Around Its Accuracy

Suppose only 80 labelled validation predictions satisfy

$$
q\geq0.95.
$$

If 76 are correct, the observed selective accuracy is

$$
\frac{76}{80}
=
0.95.
$$

That does not mean the true selective accuracy is exactly 0.95.

The estimate itself has sampling uncertainty.

This matters because high thresholds often leave few labelled validation examples.

A pseudo-label rule should therefore be judged using confidence or credible intervals for selective accuracy, not only the point estimate.

The irony is easy to miss:

the more selective the confidence threshold becomes, the fewer labelled observations may remain for estimating how trustworthy that threshold actually is.

## Agreement Between Models Can Add Information

One way to reduce reliance on a single model's confidence is to require agreement across independently trained models or perturbations.

Suppose models

$$
f_1,\ldots,f_M
$$

produce predictions

$$
\hat y^{(1)}(x),\ldots,\hat y^{(M)}(x).
$$

A pseudo-label might be accepted only when

$$
\hat y^{(1)}(x)
=
\cdots
=
\hat y^{(M)}(x)
$$

and the confidence criteria are also satisfied.

This can reduce some idiosyncratic errors.

But agreement is not independence.

Models trained on the same data, features and architecture can share the same systematic bias.

Ten identical mistakes do not become correct because ten models agree.

Diversity of errors matters.

## Consistency Regularisation Changes the Problem but Not the Principle

Modern semi-supervised methods often combine pseudo-labels with consistency regularisation.

The model is encouraged to produce similar predictions under perturbations of the same observation.

If

$$
T_1(x)
$$

and

$$
T_2(x)
$$

are two augmentations, the method may penalise disagreement between

$$
f(T_1(x))
$$

and

$$
f(T_2(x)).
$$

This can be powerful when the augmentation is label-preserving.

But label preservation is itself an assumption.

An augmentation that changes the target class turns consistency into a wrong constraint.

Once again, the unlabelled method works by introducing structure that is not contained in the labels alone.

The structural assumption should be tested where possible.

## Thresholding Can Hide Hard Regions

Confidence-based self-training preferentially selects easy observations.

That can be useful early in training.

It also means that hard regions may remain unlabelled indefinitely.

Suppose the true decision boundary passes through a region where confidence remains around

$$
0.55\text{ to }0.75.
$$

If the threshold stays at

$$
0.95,
$$

those observations never enter the pseudo-labelled set.

The model repeatedly reinforces regions it already understands while receiving no new supervision near the boundary where its errors are concentrated.

This is one reason active learning and pseudo-labelling solve different information problems.

Pseudo-labelling tends to exploit confident regions.

Active learning often seeks informative uncertain regions for human labelling.

The two can complement each other.

## Monitoring Only Final Test Accuracy Misses the Mechanism

Suppose a semi-supervised model improves test accuracy by one percentage point.

That is useful evidence.

But if we want to understand whether the pipeline is robust, we should inspect the pseudo-label process itself.

Track at least:

$$
\text{acceptance rate by iteration},
$$

$$
\text{pseudo-label class proportions},
$$

$$
\text{estimated pseudo-label accuracy},
$$

$$
\text{confidence distribution},
$$

$$
\text{calibration on trusted labels},
$$

and

$$
\text{agreement or stability across retraining runs}.
$$

A final metric can hide a pipeline that succeeds only because one early pseudo-label batch happened to be favourable.

Repeated-seed behaviour is especially important.

## A Better Experimental Contract

For confidence-based self-training, I would keep the following controls.

### 1. Preserve a purely supervised baseline

Train the same model family on the labelled sample only.

This isolates the contribution of pseudo-labels.

### 2. Keep a trusted validation set untouched

Do not use every available label for initial training.

Some real labels are needed to estimate calibration, selective accuracy and pseudo-label risk.

### 3. Evaluate multiple thresholds

Do not report only the threshold that produced the best final result.

Show the trade-off across

$$
\tau.
$$

### 4. Record pseudo-label accuracy where ground truth is available

In simulation or retrospective experiments, hide labels rather than deleting them.

That allows direct measurement of

$$
R_{\text{PL}}(\tau).
$$

### 5. Track class and subgroup composition

Measure who gets pseudo-labelled, not only how many observations are accepted.

### 6. Repeat across seeds

Pseudo-label feedback can make the path dependent on the initial labelled sample.

Report the distribution of gains and failures.

### 7. Stress-test distribution shift

Alter the unlabelled pool while keeping the labelled test problem fixed.

This reveals whether confidence continues to mean the same thing when the unlabelled distribution changes.

## Confidence Should Be Treated as a Measurement

A model confidence score is a measurement produced by an instrument.

The instrument has assumptions.

It can be biased.

Its calibration can drift.

Its error can depend on population and operating conditions.

Seen this way, a threshold such as

$$
q\geq0.95
$$

is not fundamentally different from any other measurement threshold.

Before using it to create training data, ask:

- What does this score measure?
- On which population was that interpretation validated?
- How uncertain is the calibration estimate?
- Does the interpretation hold by class and subgroup?
- What happens under plausible shift?
- How much training weight will accepted pseudo-labels receive?

That framing is more useful than treating confidence as an intrinsic property of a prediction.

## Pseudo-Labels Are Not Labels

The notation itself can encourage overconfidence.

Once we write

$$
\hat y_j
$$

next to

$$
y_i,
$$

the two targets can look interchangeable.

They are not.

A human or experimentally observed label is itself imperfect in many applications, but its error mechanism is different from that of a pseudo-label generated by the model being trained.

A pseudo-label carries model dependence.

Its uncertainty should not disappear simply because it has been converted to an integer class ID.

If the model estimates class probabilities, retaining soft targets can sometimes preserve more information than collapsing immediately to

$$
\arg\max.
$$

Even then, the probabilities remain model outputs and should not be confused with known conditional probabilities.

## The Link to Negative Transfer

In an earlier article, I showed that unlabelled covariate shift can turn a small semi-supervised gain into negative transfer while the supervised baseline remains unchanged.

The article is available here:

[When Unlabelled Data Makes Semi-Supervised Learning Worse](/machine-learning/when_unlabelled_data_makes_semi_supervised_learning_worse/)

Confidence-based pseudo-labelling is one mechanism through which that failure can occur.

If distribution shift changes the relationship between

$$
q(X)
$$

and correctness, the acceptance rule can continue selecting many observations while silently admitting more wrong pseudo-labels.

The threshold has not changed.

Its meaning has.

That is why confidence calibration should be monitored as part of the semi-supervised system rather than treated as a one-time model property.

## The Main Distinction

Three quantities are easy to confuse:

### Reported confidence

$$
q(x)
=
\max_k \hat p_k(x).
$$

This is what the model outputs.

### Conditional correctness probability

$$
P\{\hat Y=Y\mid q(X)=q\}.
$$

This is what calibration connects to the model score.

### Selective pseudo-label accuracy

$$
P\{\hat Y=Y\mid q(X)\geq\tau\}.
$$

This is what matters directly when thresholded pseudo-labels are turned into training targets.

They are related.

They are not identical.

A sound pseudo-labelling pipeline measures the third quantity rather than inferring it from the first.

## Conclusion

Confidence thresholding is a sensible idea.

If a model is well calibrated on the relevant population, high-confidence predictions are natural candidates for pseudo-labelling.

But the number on the probability output is not a correctness guarantee.

A threshold such as

$$
0.95
$$

means only that the fitted model assigned at least 0.95 probability to its preferred class.

To interpret that threshold operationally, we need evidence about

$$
P\{\hat Y=Y\mid q(X)\geq0.95\}.
$$

That quantity can change with model misspecification, class imbalance, subgroup composition, distribution shift and retraining.

And because pseudo-labels are fed back into the model, their errors are not passive.

They become supervision.

The important distinction is therefore

$$
\boxed{
\text{model confidence}
\neq
\text{empirical correctness}
}
$$

unless calibration makes the connection valid.

For self-training, I would go one step further:

$$
\boxed{
\text{validate the acceptance rule, not merely the probability output.}
}
$$

That means measuring selective accuracy, coverage, calibration uncertainty, class composition, subgroup behaviour and iteration-to-iteration stability.

A pseudo-label should enter the training set because its reliability has been demonstrated under the operating conditions of the pipeline.

Not because the model printed a large number beside it.

## References

- Arazo, E., Ortego, D., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. *International Joint Conference on Neural Networks*. https://doi.org/10.1109/IJCNN48605.2020.9207304
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*, 1321–1330.
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop on Challenges in Representation Learning*.
- Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., & Raffel, C. (2020). FixMatch: Simplifying semi-supervised learning with consistency and confidence. *Advances in Neural Information Processing Systems*, 33, 596–608.
