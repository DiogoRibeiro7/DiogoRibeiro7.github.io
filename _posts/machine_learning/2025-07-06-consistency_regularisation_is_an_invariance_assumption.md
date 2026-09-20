---
permalink: '/machine-learning/consistency_regularisation_is_an_invariance_assumption/'
title: 'Consistency Regularisation Is an Invariance Assumption, Not Free Supervision'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Consistency Regularisation
- Data Augmentation
- Invariance
- Pseudo-Labels
- Statistical Learning
author_profile: false
seo_title: 'Consistency Regularisation Requires Valid Label Invariance'
seo_description: 'Consistency-based semi-supervised learning assumes that selected perturbations preserve the target label. If an augmentation crosses the true decision boundary, even the Bayes-optimal classifier is penalised.'
excerpt: >-
  Consistency regularisation is often presented as a way to extract supervision
  from unlabelled data by requiring stable predictions under perturbation. The
  supervision does not come for free. It comes from an invariance assumption:
  the perturbation must preserve the target we are trying to learn.
summary: >-
  A mathematical analysis of consistency-based semi-supervised learning. A simple
  threshold-classification example shows that valid Bayes predictions can receive
  positive consistency loss whenever augmentations cross a true class boundary.
  The article develops augmentation validity, local smoothness, orbit invariance,
  strong versus weak perturbations, confidence gating and practical stress tests.
keywords:
- consistency regularisation
- semi-supervised learning
- augmentation invariance
- FixMatch
- MixMatch
- pseudo-labels
- label preserving augmentation
classes: wide
date: '2025-07-06'
why_this_exists: >-
  Modern semi-supervised methods often rely on consistency under perturbation.
  The phrase sounds weaker than it is. Requiring f(x) and f(Tx) to agree asserts
  that the target itself should remain unchanged under T. This article isolates
  that assumption and shows what happens when it fails.
evidence: >-
  Classical consistency regularisation, perturbation-based semi-supervised
  learning, FixMatch-style weak and strong augmentation, and an exact
  one-dimensional counterexample in which the oracle classifier incurs positive
  consistency loss.
methodology: >-
  Write the consistency objective explicitly, derive the probability that a
  perturbation crosses a deterministic label boundary, interpret augmentation
  families as equivalence relations, and separate empirical augmentation
  invariance from confidence and predictive performance.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/constellation.jpg
  og_image: /assets/images/headers/constellation.jpg
  overlay_image: /assets/images/headers/constellation.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/constellation.jpg
  twitter_image: /assets/images/headers/constellation.jpg
---

A modern semi-supervised learning recipe often looks like this:

1. take an unlabelled observation,
2. perturb or augment it,
3. require the model to make nearly the same prediction,
4. repeat this over a large unlabelled sample.

The idea is attractive.

If a photograph is still the same photograph after mild cropping, colour jitter or noise, then the classifier should not change its mind.

The same intuition appears in text, audio, sensor data, time series and tabular learning.

But the phrase

> predictions should be consistent under perturbation

already contains a substantive assumption.

The assumption is not about the model.

It is about the target.

If a transformation $T$ is used for consistency regularisation, then we are asserting approximately that

$$
Y(Tx)=Y(x).
$$

That statement can be correct.

It can also be false.

When it is false, the consistency objective does not merely fail to add useful supervision.

It penalises the classifier for representing the true label change.

## The Generic Consistency Objective

Let

$$
f_\theta(x)
$$

denote a model prediction.

For classification, this may be a probability vector.

Let

$$
T
$$

be a random augmentation drawn from some distribution over transformations.

A generic consistency loss has the form

$$
L_{\text{cons}}(\theta)
=
E_{X,T}
\left[
d
\left(
f_\theta(X),
f_\theta(TX)
\right)
\right],
$$

where $d$ measures disagreement.

For probabilities, $d$ might be:

- squared Euclidean distance,
- Kullback-Leibler divergence,
- cross entropy,
- Jensen-Shannon divergence,
- or another discrepancy.

The full semi-supervised objective often looks like

$$
L(\theta)
=
L_{\text{sup}}(\theta)
+
\lambda
L_{\text{cons}}(\theta),
$$

where $L_{\text{sup}}$ uses trusted labels and $\lambda$ controls the strength of the unlabelled consistency term.

The key question is therefore:

> When should $f_\theta(X)$ and $f_\theta(TX)$ agree?

The answer cannot come from the unlabelled observations alone.

## The Hidden Invariance Assumption

Suppose the ideal target function is

$$
f^\star(x).
$$

For consistency regularisation to align with the target, we need approximately

$$
f^\star(Tx)
=
f^\star(x)
$$

for the transformations being used.

For deterministic classification,

$$
Y=g(X),
$$

this becomes

$$
g(Tx)=g(x).
$$

For probabilistic classification, the stronger relevant statement is

$$
P(Y\mid X=x)
\approx
P(Y\mid X=Tx).
$$

This is an invariance assumption.

Consistency regularisation converts it into training pressure.

That is where the extra information comes from.

## A Threshold Example

Consider the simplest binary classification problem:

$$
X\sim\operatorname{Unif}(-1,1),
$$

with deterministic label

$$
Y
=
\mathbb 1\{X>0\}.
$$

The Bayes classifier is

$$
f^\star(x)
=
\mathbb 1\{x>0\}.
$$

Now define an augmentation

$$
T_\varepsilon(x)
=
x+\varepsilon,
$$

where

$$
\varepsilon
\sim
\operatorname{Unif}(-\delta,\delta).
$$

The augmentation looks mild.

It adds bounded noise.

But for observations close to zero, it can cross the true decision boundary.

If $x>0$ and $x<\delta$, the label changes whenever

$$
\varepsilon<-x.
$$

Therefore,

$$
P
\left(
Y(T_\varepsilon x)\neq Y(x)
\mid X=x
\right)
=
\frac{\delta-x}{2\delta}
$$

for

$$
0<x<\delta.
$$

By symmetry, for

$$
-\delta<x<0,
$$

the probability is

$$
\frac{\delta-|x|}{2\delta}.
$$

Outside the interval

$$
|x|\geq\delta,
$$

the augmentation cannot cross the threshold.

## The Oracle Classifier Has Positive Consistency Loss

Now ask how often the true Bayes classifier changes under augmentation.

For

$$
0<\delta\leq1,
$$

the probability of crossing the label boundary is

$$
P
\left(
Y(T_\varepsilon X)\neq Y(X)
\right).
$$

Using the uniform density of $X$,

$$
P
=
\int_{-\delta}^{\delta}
\frac12
\frac{\delta-|x|}{2\delta}
\,dx.
$$

By symmetry,

$$
P
=
\frac{1}{2\delta}
\int_0^\delta
(\delta-x)
\,dx.
$$

The integral is

$$
\frac{\delta^2}{2},
$$

so

$$
\boxed{
P
\left(
Y(T_\varepsilon X)\neq Y(X)
\right)
=
\frac{\delta}{4}.
}
$$

This is already enough to make the point.

If

$$
\delta=0.2,
$$

then

$$
5\%
$$

of augmented pairs cross the true class boundary.

If

$$
\delta=0.5,
$$

then

$$
12.5\%
$$

do.

The Bayes-optimal classifier is correct.

The consistency objective still penalises it.

## The Problem Is Not Model Error

This counterexample removes several common excuses.

There is no:

- model misspecification,
- label noise,
- calibration error,
- high-dimensional geometry,
- optimization failure,
- distribution shift,
- or finite-sample ambiguity.

The classifier

$$
f^\star(x)
=
\mathbb 1\{x>0\}
$$

is exactly correct.

The augmentation is the problem.

It encodes an invariance that the target does not satisfy.

That distinction matters because practitioners often respond to consistency failure by changing the model.

Sometimes the right object to change is the augmentation distribution.

## Stronger Augmentation Is Not Automatically Better

Suppose $\delta$ increases.

The augmentation explores a larger neighbourhood around each observation.

That may create stronger regularisation.

It also raises the probability of crossing the true boundary:

$$
P(\text{label change})
=
\frac{\delta}{4}
$$

for

$$
0\leq\delta\leq1.
$$

So in this example, stronger augmentation linearly increases the fraction of pairs for which consistency is wrong.

There is a genuine trade-off:

$$
\text{stronger invariance pressure}
\quad\text{versus}\quad
\text{higher probability of invalid invariance}.
$$

The phrase

> use stronger augmentation

has no universal statistical meaning.

Strength is useful only relative to the target's actual invariances.

## A Large Consistency Weight Can Move the Boundary

Consider the full objective

$$
L(\theta)
=
L_{\text{sup}}(\theta)
+
\lambda
L_{\text{cons}}(\theta).
$$

If $\lambda$ is small, labelled evidence can dominate near the true boundary.

If $\lambda$ becomes large, the optimizer is increasingly rewarded for making

$$
f_\theta(x)
\approx f_\theta(x+\varepsilon)
$$

even where the true label changes.

One way to reduce consistency cost is to smooth the classifier near zero.

Another extreme is to make the prediction nearly constant.

A constant function satisfies

$$
f(x)=f(Tx)
$$

perfectly.

It is also a useless classifier.

This reveals an important fact:

$$
\boxed{
\text{consistency alone does not identify the target}.
}
$$

The supervised term and structural assumptions are what prevent trivial solutions.

## Consistency Is a Smoothness Prior

For small perturbations, consistency regularisation can be understood as local smoothness.

Suppose $f$ is differentiable and

$$
T_\varepsilon(x)=x+\varepsilon
$$

with small $\varepsilon$.

A first-order expansion gives

$$
f(x+\varepsilon)
\approx
f(x)
+
\nabla f(x)^T\varepsilon.
$$

Then

$$
f(x+\varepsilon)-f(x)
\approx
\nabla f(x)^T\varepsilon.
$$

For squared consistency loss,

$$
E_\varepsilon
\left[
\{f(x+\varepsilon)-f(x)\}^2
\right]
$$

is approximately

$$
\nabla f(x)^T
\Sigma_\varepsilon
\nabla f(x),
$$

where

$$
\Sigma_\varepsilon
=
E[\varepsilon\varepsilon^T].
$$

So consistency regularisation penalises directional derivatives of the prediction function.

It encourages smoothness in the directions explored by the augmentation.

That is useful when those directions are nuisance variation.

It is harmful when they contain target-relevant variation.

## Augmentation Defines Which Directions Should Not Matter

Suppose

$$
x\in\mathbb R^p
$$

and augmentation noise has covariance

$$
\Sigma_\varepsilon.
$$

Then the local penalty

$$
\nabla f(x)^T
\Sigma_\varepsilon
\nabla f(x)
$$

is large when the model changes in directions with large augmentation variance.

So the augmentation distribution defines a geometry of invariance.

If one feature is perturbed heavily, the method is told that predictions should be insensitive to it.

If another feature is never perturbed, no such constraint is imposed.

Augmentation design is therefore closely related to feature relevance.

It is not merely data multiplication.

## An Invariance Can Remove the Signal

Suppose the target is

$$
Y
=
\mathbb 1\{X_1>0\}.
$$

Now use augmentation that adds large noise to $X_1$ but leaves all other coordinates unchanged.

The consistency objective tells the classifier:

> changing $X_1$ should not change the prediction.

The supervised labels tell it:

> $X_1$ determines the prediction.

The two parts of the training objective contradict each other.

No amount of unlabelled data resolves that contradiction.

More unlabelled data makes the invalid regularisation more precisely estimated.

## Augmentations Define Equivalence Classes

There is another useful way to think about consistency.

Let $\mathcal T$ be a set of transformations.

For an observation $x$, define its augmentation orbit

$$
\mathcal O(x)
=
\{T(x):T\in\mathcal T\}.
$$

Strong consistency training encourages

$$
f(x')
\approx f(x)
$$

for all

$$
x'\in\mathcal O(x).
$$

So the method is asserting that observations in the same orbit are label-equivalent.

This is a powerful structural statement.

If the orbit stays inside one true class, the assumption is useful.

If the orbit crosses classes, consistency collapses distinctions the target actually needs.

## Group Invariance Makes the Assumption Precise

For some problems, transformations form a mathematical group $G$ acting on the input space.

An invariant target satisfies

$$
f^\star(gx)
=
f^\star(x)
$$

for all

$$
g\in G.
$$

Then quotienting out the group action can reduce irrelevant variation.

This is one reason translation, rotation or permutation invariance can be so effective in the right domains.

But group structure does not guarantee label invariance.

The group may be mathematically valid while the target changes under its action.

The application determines whether the invariance is legitimate.

## Rotation Is Not Always Label-Preserving

In general image recognition, a small rotation may preserve object identity.

In other settings, orientation is the label.

Examples include:

- handwritten digits such as 6 and 9,
- arrows,
- traffic signs,
- mechanical part orientation,
- medical imaging where anatomical direction matters,
- and remote-sensing tasks with directional structure.

So the rule

$$
Y(R_\theta x)=Y(x)
$$

cannot be assumed merely because rotations are easy to generate.

The augmentation has to match the semantics of the task.

## Horizontal Flips Are Context-Dependent

Horizontal flipping is common in vision.

For many object categories, it is harmless.

For other tasks it can change meaning.

Examples include:

- text,
- left-versus-right anatomical labels,
- driving side,
- asymmetric logos,
- directional signs,
- and medical laterality.

An augmentation library cannot know which distinction the target uses.

The modeller has to know.

## Time-Series Augmentation Is Especially Delicate

Suppose $x(t)$ is a time series.

Common augmentations include:

- time shifts,
- cropping,
- warping,
- amplitude scaling,
- jitter,
- masking,
- permutation of segments,
- frequency filtering.

Whether these preserve labels depends entirely on the problem.

If the label is

> did an event occur within ten seconds before failure?

then shifting the event in time can change the label.

If the label is defined by absolute amplitude, scaling is not invariant.

If transient ordering matters, segment permutation destroys the target.

Time-series augmentation is therefore not a generic recipe.

It is a hypothesis about which aspects of temporal structure are irrelevant.

## Cropping Can Delete the Evidence

Suppose a long signal receives a positive label because it contains one rare event.

A random crop may remove that event.

If the cropped version is forced to retain the positive label or prediction, the model is being trained to classify an observation as positive without the evidence that made the original positive.

Sometimes that is desirable because contextual information remains.

Sometimes it is plainly wrong.

Consistency does not distinguish the two cases.

Only the task definition can.

## Noise Injection Has a Measurement Interpretation

Adding noise can be justified when the noise distribution reflects measurement uncertainty.

Suppose the observed variable is

$$
X_{\text{obs}}
=
X_{\text{true}}
+
\eta.
$$

If repeated measurements would naturally vary according to $\eta$, then enforcing some prediction stability across comparable perturbations can be reasonable.

But if artificial augmentation noise is much larger or differently structured than real measurement error, the invariance is no longer anchored to the data-generating process.

A useful rule is:

> synthetic perturbations should have a defensible relationship to plausible nuisance variation.

Convenience is not enough.

## Weak and Strong Augmentations Encode Different Claims

Methods such as FixMatch distinguish weak and strong augmentations.

A weakly augmented observation produces a pseudo-label.

A strongly augmented version is trained to match that pseudo-label.

Schematically,

$$
\hat y
=
\arg\max
f_\theta(T_w x),
$$

then

$$
f_\theta(T_s x)
$$

is pushed toward $\hat y$ when confidence is high.

This setup contains two structural assumptions:

1. the weak augmentation preserves the label,
2. the strong augmentation also preserves the label.

The second assumption is stronger.

Confidence in the weak view does not prove that the strong transformation is label-preserving.

## Confidence Gating Solves a Different Problem

Suppose pseudo-labels are used only when

$$
\max_k f_\theta(T_wx)_k
\geq\tau.
$$

This protects against uncertain predictions.

It does not directly protect against invalid augmentations.

The model can be extremely confident about $x$ while

$$
T_sx
$$

belongs to another true class.

Confidence answers

> how strongly does the model prefer this class?

Augmentation validity asks

> should the target remain the same after this transformation?

These are different questions.

A confidence threshold cannot substitute for an invariance test.

## A Perfectly Calibrated Model Can Still Suffer

Imagine a perfectly calibrated classifier on the original distribution.

Suppose the strong augmentation moves some observations across true class boundaries.

The original confidence can still be valid.

The augmentation is still invalid.

Calibration says something about

$$
P(Y\mid X=x).
$$

Consistency assumes something about the relation between

$$
P(Y\mid X=x)
$$

and

$$
P(Y\mid X=Tx).
$$

Calibration does not imply invariance.

## Consistency Can Amplify Shortcut Features

Suppose a model has learned a shortcut feature that is stable under the chosen augmentations.

The true causal or semantic feature may vary more.

Consistency training then rewards the shortcut because it produces stable predictions across augmented views.

The method can become more confident in a spurious rule precisely because that rule is augmentation-invariant.

This is a subtle failure mode.

Consistency does not prefer truth.

It prefers stability under the transformation family.

If nuisance structure is more stable than target-relevant structure, the regulariser can reinforce the nuisance.

## The Augmentation Distribution Matters, Not Just the Transformation Type

Saying

> we use additive noise

is incomplete.

We need its distribution.

For example,

$$
\varepsilon
\sim N(0,\sigma^2)
$$

with small $\sigma$ expresses a different invariance assumption from large $\sigma$.

The same is true for rotation range, crop size, masking rate and temporal warp strength.

The probability that an augmentation changes the true label is a function of augmentation intensity.

In the threshold example,

$$
P(\text{label change})
=
\frac{\delta}{4}.
$$

So augmentation validity is quantitative.

It should be studied as a curve, not a binary property.

## Estimate an Augmentation Failure Curve

If enough trusted labels are available, define

$$
r(\alpha)
=
P
\left(
Y(T_\alpha X)\neq Y(X)
\right),
$$

where $\alpha$ controls augmentation strength.

For deterministic labels, this can sometimes be estimated directly by applying transformations and verifying whether the target changes.

In other applications, domain rules or expert annotation may be needed.

The function

$$
\alpha\mapsto r(\alpha)
$$

is an augmentation validity curve.

It answers a more useful question than

> Is this augmentation good?

It asks

> At what intensity does the invariance begin to fail?

## Some Labels Are Naturally Invariant Only Locally

A transformation need not be globally label-preserving to be useful.

It may be valid for some observations and invalid for others.

In the threshold example, additive perturbation is safe whenever

$$
|x|\geq\delta.
$$

It is risky only near the boundary.

This suggests conditional augmentation policies.

Instead of using one global perturbation strength, the model can adapt augmentation to local uncertainty or estimated distance from a boundary.

That is more complicated.

It is also closer to the actual invariance structure.

## Boundary-Aware Consistency

Suppose a model estimates a margin

$$
m(x).
$$

One might reduce augmentation strength when

$$
|m(x)|
$$

is small.

For example,

$$
\delta(x)
=
c|m(x)|
$$

for some factor $c$.

This is only a heuristic because the estimated margin can itself be wrong.

But it reflects a sensible principle:

> stronger perturbations are safer farther from suspected label boundaries.

Uniform augmentation ignores this geometry.

## Consistency and the Low-Density Assumption Are Connected

Why can consistency work well in semi-supervised learning?

One reason is that, under a low-density separation assumption, most unlabelled observations lie away from class boundaries.

Then local perturbations are unlikely to cross the true boundary.

The consistency assumption becomes approximately valid over most of the marginal distribution.

In that regime,

$$
P(Y(TX)\neq Y(X))
$$

can be small.

So consistency regularisation is not independent of the cluster or low-density assumptions.

It is another way of exploiting them.

## If the Boundary Cuts Through Dense Regions, Consistency Becomes Risky

Consider again

$$
X\sim N(0,1)
$$

with

$$
Y=\mathbb 1\{X>0\}.
$$

The decision boundary is at the highest-density point.

Many observations lie near zero.

Even modest perturbations have a non-negligible chance of crossing the boundary.

This is exactly the setting where a low-density assumption fails.

Consistency regularisation now receives many contradictory pairs.

The same geometry undermines both assumptions.

## More Unlabelled Data Can Strengthen the Wrong Invariance

Suppose

$$
n_U
$$

becomes very large.

The empirical consistency loss converges more precisely to its population value.

If the augmentation invariance is correct, that is useful.

If it is wrong, the method estimates the wrong regularisation objective more accurately.

This is the same general principle that appears throughout semi-supervised learning:

$$
\boxed{
\text{more unlabelled data strengthens the assumptions through which it is used}.
}
$$

It does not validate those assumptions.

## Augmentation Search Can Overfit Too

Suppose an analyst tries many augmentation policies:

- several noise levels,
- several crop sizes,
- multiple masking rates,
- different transformations,
- and combinations of all of them.

Then the policy with the best validation performance is selected.

That is a model-selection problem.

The final validation gain is optimistic if the same validation set was used repeatedly.

Augmentation design creates researcher degrees of freedom just like hyperparameter tuning.

A serious pipeline should separate augmentation exploration from final evaluation.

## Domain Knowledge Is Statistical Information

In many discussions of semi-supervised learning, domain knowledge is treated as an informal extra.

For consistency methods, it is often central.

Knowing that:

- translation preserves object identity,
- monotone rescaling preserves a clinical risk category,
- sensor jitter within calibration tolerance is nuisance variation,
- or phase shift changes a time-localised event label,

is information about the invariance structure of the target.

That information is what makes unlabelled consistency useful.

The model is not learning those facts from unlabelled data.

We are supplying them through augmentation design.

## Test the Transformation on Labelled Data

The simplest validation is also one of the most important.

Take trusted labelled observations.

Apply the proposed augmentation.

Ask whether the target should remain the same.

Where direct relabelling is possible, estimate the empirical violation rate.

For transformation family $\mathcal T$, estimate

$$
\widehat r
=
\frac1m
\sum_{i=1}^m
\mathbb 1
\left\{
y(T_i x_i)\neq y_i
\right\}.
$$

If the task has deterministic labels and augmentation can be labelled reliably, this is direct evidence.

If not, use domain-specific constraints or expert review.

Do not assume invariance merely because the transformation is common in another benchmark.

## Compare Weak and Strong Views on Trusted Labels

For each labelled observation $x_i$, generate:

$$
T_wx_i
$$

and

$$
T_sx_i.
$$

Then evaluate:

- label retention,
- model accuracy,
- confidence,
- calibration,
- and prediction disagreement.

If strong augmentation sharply increases true label changes while only modestly increasing useful diversity, the trade-off is poor.

This test directly mirrors the mechanism used on unlabelled observations.

## Track Consistency Accuracy, Not Only Consistency Loss

A low consistency loss can be misleading.

A constant predictor has excellent consistency.

So monitor supervised performance jointly with consistency.

On trusted augmented labels, one can compute

$$
A_{\text{aug}}
=
P
\left(
\hat Y(TX)=Y(TX)
\right).
$$

Also compute invariance agreement,

$$
C_{\text{aug}}
=
P
\left(
\hat Y(TX)=\hat Y(X)
\right).
$$

These answer different questions.

High

$$
C_{\text{aug}}
$$

with low

$$
A_{\text{aug}}
$$

means the model is consistently wrong.

Consistency is not accuracy.

## Report the Joint Table

A useful diagnostic is the joint frequency of:

1. original prediction correct or incorrect,
2. augmented prediction correct or incorrect,
3. predictions agree or disagree.

For example, the problematic event is

$$
\hat Y(X)=Y(X),
$$

but

$$
\hat Y(TX)=\hat Y(X)\neq Y(TX).
$$

That means consistency preserved the model output while the true label changed.

This is direct evidence that the augmentation invariance is invalid for those cases.

Aggregate loss values can hide this.

## Invariance Should Be Defined at the Target Level

Sometimes labels are coarse.

A transformation may change a fine-grained property while preserving the coarse target.

That is acceptable.

For example, a crop may alter exact object position while preserving object category.

The relevant requirement is not that the input be semantically identical in every respect.

It is that the transformation preserve the target variable used for learning.

Formally,

$$
Y(TX)=Y(X)
$$

is enough.

Other attributes may change.

This distinction helps avoid making augmentation unnecessarily weak.

## Multi-Task Learning Complicates Invariance

Suppose one model predicts multiple targets:

$$
Y_1,\ldots,Y_m.
$$

A transformation may preserve some targets and change others.

For example, rotation may preserve object class but change orientation.

Cropping may preserve disease presence but change lesion size.

Time shifting may preserve event type but change time-to-event.

A single consistency objective across all outputs can therefore be inappropriate.

Invariance may need to be target-specific.

## Regression Needs the Same Care

Consistency regularisation is not limited to classification.

For regression, one may impose

$$
f(Tx)\approx f(x).
$$

But the target may legitimately change continuously under the transformation.

Suppose

$$
Y=X
$$

and augmentation adds noise:

$$
TX=X+\varepsilon.
$$

Then the correct regression function is

$$
f^\star(TX)=X+\varepsilon,
$$

not

$$
X.
$$

Enforcing exact consistency is wrong.

A transformation can preserve category while not preserving a continuous target.

The loss should reflect the task.

## Equivariance Can Be Better Than Invariance

Sometimes the correct relationship is not

$$
f(Tx)=f(x).
$$

It is

$$
f(Tx)
=
\rho(T)f(x),
$$

where $\rho(T)$ describes how the output should transform.

This is equivariance.

Examples include:

- rotating an image and rotating a predicted orientation,
- translating an object and translating a segmentation mask,
- shifting a time series and shifting an event timestamp,
- permuting graph nodes and permuting node-level outputs.

If the target transforms predictably, invariance throws away information.

Equivariance encodes the correct structural relationship.

## Consistency Should Match the Symmetry of the Task

The broader principle is:

$$
\boxed{
\text{regularise according to the target symmetry, not the input transformation alone}.
}
$$

Some transformations imply invariance.

Some imply equivariance.

Some have no valid target relation.

Choosing among them is part of model specification.

## A Better Semi-Supervised Objective

Conceptually, the consistency term should be conditional on valid transformations:

$$
L_{\text{cons}}
=
E
\left[
d
\left(
f(X),
f(TX)
\right)
\,
I\{T\text{ preserves }Y\}
\right].
$$

Of course the indicator is unknown for unlabelled data.

That is exactly the problem.

In practice we replace it with prior knowledge, constrained augmentation families, confidence heuristics or learned policies.

The hidden assumption should remain visible even when the implementation cannot evaluate it directly.

## Adaptive Augmentation Needs Independent Validation

Some methods learn or search augmentation policies automatically.

That can improve performance.

It does not remove the invariance issue.

The learned policy is another model.

If it is optimized only for consistency or training loss, it may discover transformations that make the learner's job easier rather than transformations that preserve the real target.

Policy learning should therefore be evaluated against trusted labels or task-specific invariance checks.

Automation does not make the assumption disappear.

## The Best Augmentation Is Task-Specific

There is no universal ranking of augmentations.

A strong image transformation can be excellent for object recognition and disastrous for optical character recognition.

A temporal crop can help activity classification and break forecasting.

Feature masking can regularise redundant tabular data and erase a rare decisive variable.

Noise injection can model measurement error and destroy threshold-sensitive signals.

The correct question is not

> Which augmentations work well in semi-supervised learning?

It is

> Which transformations preserve this target under this data-generating process?

That question has to be answered locally.

## A Practical Validation Contract

Before using consistency regularisation with substantial unlabelled weight, I would require several checks.

### 1. Define the intended invariance

Write down why the target should remain unchanged under each transformation.

### 2. Measure augmentation violations on trusted labels

Estimate how often the transformation changes the true target where that can be checked.

### 3. Sweep augmentation strength

Report performance and violation rates across a range, not only at the chosen value.

### 4. Keep a supervised baseline

The unlabelled consistency term should demonstrate an improvement over the same model trained only on trusted labels.

### 5. Separate confidence from invariance

A high-confidence pseudo-label does not prove that a strong augmentation preserves the label.

### 6. Evaluate by class and subgroup

An augmentation may be safe for one class and harmful for another.

### 7. Test under distribution shift

The invariance may fail when the population or acquisition process changes.

### 8. Report repeated seeds

Consistency methods can be path-dependent through pseudo-labels, initialization and augmentation sampling.

### 9. Prefer equivariance when the output should transform

Do not force invariance onto a target that has a known transformation law.

## The Link to Pseudo-Label Confidence

In a previous article I argued that pseudo-label confidence is not the same as correctness.

The article is available here:

[Pseudo-Label Confidence Is Not the Same as Correctness](/machine-learning/pseudo_label_confidence_is_not_correctness/)

Consistency regularisation adds another layer.

Even if the pseudo-label for the weak view is correct,

$$
\hat y(T_wx)=Y(T_wx),
$$

the strong augmentation may change the true label:

$$
Y(T_sx)\neq Y(T_wx).
$$

Then training the strong view toward the weak pseudo-label is wrong by construction.

Confidence and augmentation validity are separate gates.

Both matter.

## The Link to Semi-Supervised Identifiability

In another article, I separated

$$
P_X
$$

from

$$
P(Y\mid X)
$$

and argued that unlabelled data helps only through assumptions connecting the two.

Consistency regularisation is one such assumption.

It says that the conditional label distribution is approximately constant along selected perturbation directions:

$$
P(Y\mid X=x)
\approx
P(Y\mid X=Tx).
$$

That is how unlabelled observations acquire supervisory value.

The method does not avoid assumptions.

It makes one geometrically explicit.

## The Central Counterexample

The threshold example captures the whole issue:

$$
X\sim\operatorname{Unif}(-1,1),
$$

$$
Y=\mathbb 1\{X>0\},
$$

$$
T_\varepsilon(X)=X+\varepsilon,
\qquad
\varepsilon\sim\operatorname{Unif}(-\delta,\delta).
$$

The Bayes classifier is exact.

Yet

$$
\boxed{
P
\left(
Y(T_\varepsilon X)\neq Y(X)
\right)
=
\frac{\delta}{4}
}
$$

for

$$
0\leq\delta\leq1.
$$

So even perfect supervision and perfect modelling do not make the consistency assumption valid.

Only the transformation does.

## Conclusion

Consistency regularisation is powerful because it injects structural information into a learning problem with few labels.

The structural information is an invariance claim.

For a transformation $T$, the method assumes approximately that

$$
\boxed{
P(Y\mid X=x)
\approx
P(Y\mid X=Tx).
}
$$

When that statement is correct, unlabelled data can constrain the classifier productively.

When it is false, the consistency term penalises correct variation.

Stronger augmentation then strengthens the wrong constraint.

More unlabelled data estimates it more precisely.

Higher pseudo-label confidence does not repair it.

The right workflow is therefore not

> choose strong augmentations and enforce consistency.

It is

> define the target invariances, test them where labels exist, quantify how validity changes with augmentation strength, and then use consistency only inside the region where the assumption is defensible.

The supervision is not free.

It is purchased with an invariance assumption.

## References

- Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., & Raffel, C. (2019). MixMatch: A holistic approach to semi-supervised learning. *Advances in Neural Information Processing Systems*, 32.
- Laine, S., & Aila, T. (2017). Temporal ensembling for semi-supervised learning. *International Conference on Learning Representations*.
- Sajjadi, M., Javanmardi, M., & Tasdizen, T. (2016). Regularization with stochastic transformations and perturbations for deep semi-supervised learning. *Advances in Neural Information Processing Systems*, 29.
- Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., & Raffel, C. (2020). FixMatch: Simplifying semi-supervised learning with consistency and confidence. *Advances in Neural Information Processing Systems*, 33, 596–608.
