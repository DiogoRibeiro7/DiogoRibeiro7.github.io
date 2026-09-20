---
permalink: '/machine-learning/entropy_minimisation_can_make_the_wrong_answer_more_confident/'
title: 'Entropy Minimisation Can Make the Wrong Answer More Confident'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Entropy Minimisation
- Unlabelled Data
- Confidence
- Low-Density Separation
- Statistical Learning
author_profile: false
seo_title: 'Why Entropy Minimisation Does Not Guarantee Correct Predictions'
seo_description: 'Predictive entropy is minimised by confident probabilities near zero or one, regardless of correctness. With weak supervision, an unlabelled entropy term can favour confidently wrong solutions.'
excerpt: >-
  Entropy minimisation encourages decisive predictions on unlabelled data. That can
  be useful when decision boundaries should avoid high-density regions, but low
  entropy is not the same as low classification error. A simple balanced example
  shows that a sufficiently strong entropy term can destabilise the neutral
  supervised solution and create confident alternatives.
summary: >-
  A mathematical critique of over-interpreting entropy minimisation in
  semi-supervised learning. The article derives the binary entropy objective,
  constructs a constant-classifier example with balanced trusted labels, shows a
  bifurcation when the effective unlabelled entropy weight exceeds two, and
  explains how class imbalance, misspecification and low-density assumptions
  determine whether confident predictions are useful.
keywords:
- entropy minimisation
- semi-supervised learning
- predictive entropy
- low-density separation
- confidence
- unlabelled data
- negative transfer
classes: wide
date: '2025-06-15'
why_this_exists: >-
  Predictive entropy is often used as though reducing uncertainty on unlabelled
  observations were intrinsically beneficial. The entropy term does not know which
  class is correct. It only rewards certainty. This article isolates that fact
  mathematically and shows when the unlabelled term can overwhelm weak supervision.
evidence: >-
  Binary entropy geometry, an exact one-parameter counterexample with balanced
  labels, the low-density separation interpretation of entropy minimisation, and
  classical semi-supervised learning literature.
methodology: >-
  Analyse the supervised plus entropy objective in closed form, inspect the second
  derivative at the symmetric solution, identify the threshold where that solution
  loses local stability, and then translate the result into practical diagnostics
  for semi-supervised pipelines.
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

Entropy minimisation sounds like a sensible semi-supervised principle.

If the model sees many unlabelled observations, encourage it to make confident predictions on them.

In binary classification, this means pushing predicted probabilities away from

$$
0.5
$$

and toward

$$
0
$$

or

$$
1.
$$

The intuition is that class boundaries should pass through low-density regions, not through regions filled with uncertain predictions.

That intuition can be useful.

But there is a basic asymmetry hiding inside it:

$$
\text{entropy knows how uncertain a prediction is;}
$$

it does not know whether the prediction is correct.

A confidently wrong prediction has very low entropy.

So does a confidently correct one.

The entropy term cannot distinguish them.

## Binary Predictive Entropy

For a binary class probability

$$
p=P(Y=1\mid X=x),
$$

the Shannon entropy is

$$
H(p)
=
-p\log p
-
(1-p)\log(1-p).
$$

Its maximum occurs at

$$
p=\frac12,
$$

where uncertainty is highest.

Its minimum occurs at the extremes,

$$
p=0
$$

and

$$
p=1.
$$

Formally,

$$
H\left(\frac12\right)=\log 2,
$$

while

$$
H(0)=H(1)=0.
$$

So minimizing entropy encourages decisive predictions.

That is all it guarantees.

## The Entropy Function Is Symmetric About the Wrong Answer

Suppose the true label for an observation is

$$
Y=1.
$$

Then

$$
p=0.99
$$

is excellent.

But

$$
p=0.01
$$

has exactly the same entropy:

$$
H(0.99)=H(0.01).
$$

The entropy term is indifferent between them.

It prefers both to

$$
p=0.5.
$$

This single identity already gives the central warning:

$$
\boxed{
\text{low predictive entropy}
\not\Rightarrow
\text{high predictive accuracy}.
}
$$

The labelled loss has to supply the direction.

## The Typical Semi-Supervised Objective

A simplified entropy-minimisation objective is

$$
L(\theta)
=
L_{\text{sup}}(\theta)
+
\lambda
\sum_{j\in U}
H
\left(
p_\theta(x_j)
\right),
$$

where

- $L_{\text{sup}}$ is the supervised loss,
- $U$ is the unlabelled sample,
- $\lambda$ controls the strength of the entropy penalty.

The supervised term says:

> fit the trusted labels.

The entropy term says:

> make the unlabelled predictions decisive.

Those two instructions are compatible only if the direction of confidence induced by the supervised signal is already approximately correct.

## A One-Parameter Counterexample

Consider an intentionally simple classifier.

It predicts the same class-one probability

$$
q
$$

for every observation.

The model is obviously underpowered.

That is useful because it lets us see the objective exactly.

Suppose the trusted labelled set contains exactly two observations:

- one with label zero,
- one with label one.

The supervised cross-entropy loss is

$$
L_{\text{sup}}(q)
=
-\log q
-
\log(1-q).
$$

Because the labels are balanced, the supervised optimum is

$$
q=\frac12.
$$

The classifier cannot distinguish the observations, so the best it can do is report the class frequency.

Now add

$$
m
$$

unlabelled observations and entropy weight

$$
\lambda.
$$

Because the classifier predicts the same $q$ for every unlabelled point, the entropy contribution is

$$
\lambda m H(q).
$$

Define

$$
\alpha=\lambda m.
$$

The full objective becomes

$$
\boxed{
L_\alpha(q)
=
-\log q
-
\log(1-q)
+
\alpha H(q).
}
$$

This one-dimensional function contains the entire conflict.

## The Balanced Supervised Solution Is Always Stationary

Differentiate.

The supervised part gives

$$
\frac{d}{dq}
\left[
-\log q-\log(1-q)
\right]
=
-\frac1q
+
\frac1{1-q}.
$$

For the entropy term,

$$
H'(q)
=
\log
\frac{1-q}{q}.
$$

Therefore,

$$
L_\alpha'(q)
=
-\frac1q
+
\frac1{1-q}
+
\alpha
\log
\frac{1-q}{q}.
$$

At

$$
q=\frac12,
$$

the first two terms cancel and

$$
\log1=0.
$$

So

$$
L_\alpha'\left(\frac12\right)=0
$$

for every

$$
\alpha.
$$

The symmetric solution remains a stationary point no matter how much unlabelled entropy pressure we add.

But its stability changes.

## The Second Derivative Reveals the Transition

Differentiate again.

For the supervised term,

$$
\frac{d^2}{dq^2}
\left[
-\log q-\log(1-q)
\right]
=
\frac1{q^2}
+
\frac1{(1-q)^2}.
$$

For entropy,

$$
H''(q)
=
-\frac1q
-
\frac1{1-q}.
$$

Thus,

$$
L_\alpha''(q)
=
\frac1{q^2}
+
\frac1{(1-q)^2}
-
\alpha
\left(
\frac1q
+
\frac1{1-q}
\right).
$$

At

$$
q=\frac12,
$$

we obtain

$$
L_\alpha''\left(\frac12\right)
=
8-4\alpha.
$$

Therefore:

$$
L_\alpha''\left(\frac12\right)>0
\quad\text{when}\quad
\alpha<2,
$$

and

$$
L_\alpha''\left(\frac12\right)<0
\quad\text{when}\quad
\alpha>2.
$$

So the balanced supervised solution changes from a local minimum to a local maximum at

$$
\boxed{
\alpha=2.
}
$$

That is the key result.

## The Entropy Term Can Destabilise the Supervised Solution

Recall that

$$
\alpha=\lambda m.
$$

So the transition occurs when the total effective unlabelled entropy pressure exceeds

$$
2.
$$

For

$$
\alpha<2,
$$

the objective is locally happy with the uncertain probability

$$
q=\frac12.
$$

For

$$
\alpha>2,
$$

that same point becomes locally unstable.

The entropy term now pushes the model toward one of two more confident alternatives.

Because the problem is symmetric, there is no information in the objective telling it which direction is substantively correct.

The unlabelled term has created confidence pressure.

It has not created class information.

## The Two Confident Directions Are Symmetric

The objective satisfies

$$
L_\alpha(q)
=
L_\alpha(1-q).
$$

So whenever a minimum appears at

$$
q^\star<\frac12,
$$

there is another at

$$
1-q^\star>\frac12.
$$

One solution predicts class zero confidently for almost everyone.

The other predicts class one confidently for almost everyone.

The trusted labels are perfectly balanced.

The unlabelled entropy term creates no preference between the two collapse directions.

This is not a numerical artifact.

It follows from symmetry.

## More Unlabelled Data Can Strengthen the Collapse

Because

$$
\alpha=\lambda m,
$$

increasing the size of the unlabelled sample increases the total entropy contribution unless the loss is normalized differently.

Even when implementations average the unlabelled term, an equivalent effect appears through the explicit weighting parameter.

The important quantity is the relative strength of the unlabelled regularizer compared with trusted supervision.

If that ratio grows too large, the model can be rewarded more strongly for becoming certain than for respecting the limited labelled evidence.

This gives another version of a recurring semi-supervised principle:

$$
\boxed{
\text{more unlabelled data strengthens the assumption through which it is used}.
}
$$

Entropy minimisation assumes that decisive predictions are useful.

It does not verify that assumption.

## Confidence Collapse Is Not Always Visible in Accuracy

Suppose the true population is highly imbalanced.

Imagine

$$
P(Y=0)=0.95.
$$

A model collapsing confidently to class zero can obtain

$$
95\%
$$

accuracy.

Its entropy is near zero.

Its accuracy looks strong.

Its minority-class recall is catastrophic.

So a low-entropy solution can look excellent under a coarse aggregate metric.

This is especially dangerous when the labelled set is small enough that minority examples are rare.

Entropy minimisation can reinforce the majority solution because confidence and prevalence point in the same direction.

## Class Balance Changes the Geometry of the Objective

Now suppose the labelled sample contains

$$
n_1
$$

positive labels and

$$
n_0
$$

negative labels.

For the same constant classifier,

$$
L_{\text{sup}}(q)
=
-n_1\log q
-
n_0\log(1-q).
$$

The supervised optimum is

$$
q_{\text{sup}}
=
\frac{n_1}{n_0+n_1}.
$$

This already reflects the observed class balance.

Add entropy minimisation and the optimum is pushed further away from

$$
0.5.
$$

So if the labelled class proportion is itself noisy or unrepresentative, the unlabelled entropy term can amplify that sampling accident.

The mechanism is simple:

$$
\text{small labelled imbalance}
\rightarrow
\text{slightly asymmetric predictions}
\rightarrow
\text{entropy pressure}
\rightarrow
\text{more extreme asymmetry}.
$$

## Entropy Minimisation Is a Decision-Boundary Assumption

Why can entropy minimisation work well in practice?

Suppose class probabilities are uncertain mainly near the decision boundary.

Then high entropy marks a region where the classifier changes class.

If the unlabelled data density is low there, pushing probabilities away from

$$
0.5
$$

encourages the boundary to move into the density valley.

This is closely related to low-density separation.

The method is therefore useful when the geometry looks roughly like

$$
\text{high-density class region}
\quad
\text{low-density boundary}
\quad
\text{high-density class region}.
$$

In that setting, unlabelled density contains information about where the decision boundary should not go.

## The Assumption Can Fail at Maximum Density

Consider

$$
X\sim N(0,1),
$$

with deterministic class rule

$$
Y
=
\mathbb 1\{X>0\}.
$$

The Bayes decision boundary is

$$
x=0.
$$

But zero is also the mode of the feature distribution.

The correct boundary passes through the region of highest density.

Any method that strongly prefers low-density separation is being pushed away from the true classifier.

The failure is not because the model is poorly trained.

The structural assumption is wrong.

## Low Entropy Can Be Wrong Everywhere

Take the extreme classifier

$$
p_\theta(Y=1\mid x)=0.999
$$

for every $x$.

Its predictive entropy is tiny:

$$
H(0.999)
\approx0.
$$

If the true population is balanced, its classification error is approximately

$$
50\%.
$$

Now take the opposite constant classifier,

$$
p_\theta(Y=1\mid x)=0.001.
$$

It has essentially the same entropy and the same error.

Entropy cannot distinguish between them.

A supervised signal is required to orient the decision.

## Entropy Is Not Calibration

A classifier can be confidently wrong and badly calibrated.

Entropy minimisation tends to sharpen probabilities.

That can reduce predictive entropy while worsening calibration.

Suppose true conditional probability is

$$
P(Y=1\mid X=x)=0.7.
$$

A calibrated model should predict approximately

$$
0.7.
$$

Entropy minimisation may push the output toward

$$
0.9
$$

or

$$
1.
$$

Classification may remain correct under a 0.5 threshold.

Calibration becomes worse.

So even when accuracy improves, probabilistic quality can deteriorate.

## Proper Scoring Rules and Entropy Serve Different Roles

Cross entropy on labelled observations is a proper scoring rule.

In expectation, it is minimized by the true conditional probability.

Predictive entropy on unlabelled observations is not a proper scoring rule for the unknown labels.

It contains no observed outcome.

Its optimum is simply certainty.

That distinction is fundamental.

The supervised loss estimates

$$
P(Y\mid X).
$$

The unlabelled entropy term imposes a structural preference on that estimate.

It is regularization, not additional observed truth.

## A Useful Decomposition

For binary prediction, one can conceptually separate three goals:

1. **discrimination**: choose the correct class,
2. **calibration**: match predicted probability to empirical frequency,
3. **confidence**: move probabilities away from 0.5.

Entropy minimisation directly encourages the third.

It may help the first under suitable geometry.

It can hurt the second.

Those outcomes should not be conflated.

## The Entropy Term Can Fight Label Noise in Either Direction

Suppose a few trusted labels are wrong.

Entropy minimisation might help by preventing the classifier from bending sharply around isolated mislabeled observations.

That can be beneficial.

But the same mechanism can also suppress a rare but correct class.

The method does not know whether a conflicting labelled point is:

- noise,
- a minority population,
- a boundary case,
- or evidence that the low-density assumption fails.

The interpretation comes from the data-generating context.

## Entropy and Pseudo-Labelling Are Closely Related

Pseudo-labelling converts high-confidence predictions into hard targets.

Entropy minimisation instead directly rewards low-entropy predictions.

Both encourage the model to become more decisive on unlabelled observations.

In the binary case, pushing

$$
p
$$

toward

$$
0
$$

or

$$
1
$$

makes an eventual hard pseudo-label more stable.

So the two mechanisms can reinforce each other:

$$
\text{slightly confident prediction}
\rightarrow
\text{entropy sharpening}
\rightarrow
\text{hard pseudo-label}
\rightarrow
\text{retraining}.
$$

If the initial direction is wrong, the feedback can strengthen the error.

## Consistency Regularisation Can Reinforce the Same Mistake

Suppose a model is confidently wrong on an unlabelled observation.

Entropy minimisation rewards the confidence.

Consistency regularisation then asks augmented versions of the observation to preserve that prediction.

The combination can produce a coherent but wrong local region.

This is why semi-supervised losses should not be evaluated one term at a time.

Their assumptions interact.

## Entropy Minimisation Does Not Create Missing Classes

Suppose the labelled sample contains no examples from one rare class.

An unlabelled entropy term does not know that the class exists.

If the model's current representation assigns those observations confidently to a known class, entropy minimisation rewards that assignment.

The unlabelled data may contain the missing population geometrically.

But the entropy objective alone has no semantic mechanism for inventing a new class label.

This matters in open-set and class-mismatch settings.

## Class-Mismatch Can Be Dangerous

Suppose the unlabelled pool contains observations from classes absent from the labelled problem.

A closed-set classifier still has to allocate them among known classes.

Entropy minimisation encourages that allocation to be confident.

The result can be confidently wrong by construction.

So before using entropy minimisation, one should ask whether labelled and unlabelled samples share the same label space.

That assumption is often left implicit.

## Out-of-Distribution Inputs Can Receive Low Entropy

Many discriminative models produce high-confidence predictions far from the labelled support.

If such observations enter the unlabelled pool, entropy minimisation can make the problem worse by explicitly rewarding those confident extrapolations.

Low entropy therefore does not imply that an observation is in distribution.

Confidence, density and support are separate concepts.

## Entropy Minimisation and Temperature

Suppose logits are

$$
z_k(x).
$$

A softmax with temperature

$$
T
$$

uses

$$
p_k(x;T)
=
\frac{
\exp(z_k/T)
}{
\sum_j\exp(z_j/T)
}.
$$

Lower temperature sharpens probabilities.

As

$$
T\to0,
$$

the distribution becomes nearly one-hot.

Entropy falls.

But class decisions may not change at all.

So one can reduce entropy dramatically without improving classification.

This gives a simple conceptual demonstration that entropy is not synonymous with accuracy.

## Confidence Sharpening Can Be Purely Cosmetic

Suppose the predicted class is already fixed:

$$
\arg\max_k p_k(x)=c.
$$

Changing

$$
p_c(x)
$$

from

$$
0.7
$$

to

$$
0.99
$$

does not change the hard classification.

It only changes confidence.

If class $c$ is correct, the classifier becomes more decisive.

If class $c$ is wrong, it becomes more decisively wrong.

Entropy minimisation cannot tell which case it is in.

## Why the Low-Density Assumption Helps

The strongest argument for entropy minimisation is not

> confident predictions are good.

It is

> under the assumed geometry, correct boundaries should lie in low-density regions, so uncertainty on high-density unlabelled observations is evidence that the current boundary is misplaced.

That is a much more specific statement.

It gives the method a mechanism.

It also gives us something to test.

## Examine Entropy Against Feature Density

If entropy minimisation is justified through low-density separation, inspect whether high predictive entropy is actually concentrated in low-density or transition regions.

Let

$$
\hat \rho(x)
$$

be a density proxy or local-neighbour score.

Then examine the joint relationship between

$$
H\{p_\theta(x)\}
$$

and

$$
\hat\rho(x).
$$

If high entropy is concentrated in dense, well-supported regions, the low-density rationale is questionable.

Density estimation in high dimensions is difficult, so this is not a universal diagnostic.

But the justification and the diagnostic should at least point in the same direction.

## Use Labelled Data to Check the Assumption

On a trusted labelled validation set, examine whether points near the estimated decision boundary tend to lie in lower-density regions than confidently classified observations.

For example, define a margin proxy

$$
M(x)
=
\left|
p_\theta(Y=1\mid x)-\frac12
\right|.
$$

Small

$$
M(x)
$$

indicates uncertainty.

Compare neighbourhood density for small- and large-margin observations.

If uncertainty frequently occurs in dense regions containing both classes, entropy minimisation may impose the wrong geometry.

## Monitor Calibration Before and After Entropy Regularisation

Because entropy minimisation sharpens predictions, calibration should be measured explicitly.

Useful metrics include:

- reliability diagrams,
- class-conditional calibration,
- Brier score,
- log loss,
- expected calibration error,
- and selective accuracy at confidence thresholds.

An accuracy gain accompanied by severe overconfidence may be unacceptable in applications where probabilities drive decisions.

## Compare Predictive Entropy With Error

On labelled validation data, estimate

$$
P(\hat Y\neq Y\mid H\leq h).
$$

If low-entropy predictions are genuinely reliable, this conditional error should fall as entropy decreases.

Do not assume the relationship.

Measure it.

A model whose lowest-entropy predictions still contain systematic errors is a poor candidate for entropy-driven self-training.

## Sweep the Entropy Weight

The parameter

$$
\lambda
$$

controls how strongly unlabelled certainty competes with trusted supervision.

A serious experiment should report performance over a range of values.

The one-parameter example showed why.

The effective objective can change qualitatively as the relative entropy weight grows.

In the toy problem, the symmetric solution changes stability at

$$
\alpha=2.
$$

Real models need not have such a clean threshold.

They can still undergo analogous changes in optimization behaviour.

## Record Class Proportions

Entropy minimisation can amplify class imbalance.

Track the predicted class distribution on unlabelled data as

$$
\lambda
$$

changes.

If one class suddenly absorbs most of the unlabelled pool while supervised validation does not improve, that is a warning sign.

The model may be reducing entropy through class collapse rather than discovering useful structure.

## Conditional Entropy Alone Can Encourage Collapse

In some objectives, one minimizes conditional predictive entropy

$$
H(Y\mid X)
$$

on unlabelled data.

A constant deterministic classifier has

$$
H(Y\mid X)=0.
$$

So conditional entropy alone admits complete class collapse.

Some methods counter this by also encouraging diversity in the marginal prediction distribution.

That distinction matters.

## Mutual Information Shows the Missing Piece

Recall

$$
I(X;Y)
=
H(Y)-H(Y\mid X).
$$

Minimizing conditional entropy

$$
H(Y\mid X)
$$

encourages confident predictions.

Maximizing mutual information also rewards large marginal entropy

$$
H(Y),
$$

which discourages collapse to a single class.

This does not solve every problem.

But it reveals exactly what pure entropy minimisation omits.

A model that confidently predicts one class for everything has low conditional entropy but low marginal class entropy as well.

## Marginal Entropy Introduces Another Assumption

Encouraging high

$$
H(Y)
$$

can prevent trivial collapse.

But it may implicitly favour balanced class usage.

If the true class distribution is highly imbalanced, forcing high marginal entropy can be wrong.

So the correction introduces another structural assumption.

There is no free objective.

Every regularizer expresses a preference.

## Distribution Alignment Makes Priors Explicit

Some semi-supervised methods explicitly align predicted class frequencies with an estimated class prior.

If the true prior

$$
\pi_k=P(Y=k)
$$

is known or reliably estimated, this can constrain entropy collapse.

But if the unlabelled population has a different class prior from the labelled sample, using the labelled prior can introduce bias.

Prior alignment should therefore be justified by the sampling design.

## A Good Negative Control

One useful experiment is to destroy the relationship between labels and unlabelled geometry while preserving the feature distribution.

For example:

1. keep the unlabelled features fixed,
2. permute trusted labels in a controlled experiment,
3. fit the semi-supervised method,
4. inspect whether entropy minimisation still produces confident structure.

It often will.

That is expected.

The negative control reveals how much apparent certainty can be generated by the regularizer without genuine label information.

## Another Negative Control: Unimodal Data

Use a continuous unimodal distribution with a label boundary that cuts through its dense region.

This directly violates low-density separation.

Compare:

- supervised learning,
- entropy-minimised learning,
- calibration,
- boundary location,
- and error.

If the entropy method degrades, the experiment confirms that the failure is structural rather than implementation-specific.

## The Supervised Baseline Is the Control Arm

The simplest question remains:

$$
\text{Does the unlabelled entropy term improve over the same model trained only on trusted labels?}
$$

The baseline should use:

- the same architecture,
- the same labelled data,
- the same optimization budget where possible,
- and the same evaluation set.

Otherwise, gains can be attributed to unrelated changes.

Semi-supervised learning needs a clean control condition.

## Report Negative Transfer

Define

$$
G
=
R_{\text{sup}}
-
R_{\text{SSL}},
$$

where lower risk is better.

Then

$$
G<0
$$

indicates negative transfer.

Do not report only the average gain.

Report:

$$
E[G],
$$

$$
\operatorname{median}(G),
$$

and

$$
P(G<0)
$$

across seeds or resamples.

Entropy minimisation can help on average while harming a substantial subset of datasets or subgroups.

## Entropy by Subgroup Matters Too

Suppose a model is confident on the majority population but uncertain on a minority subgroup.

Entropy minimisation may preferentially force that subgroup into the majority decision geometry.

So examine predictive entropy and error by subgroup.

Global entropy reduction can hide concentrated harm.

This is especially important when representation quality differs across populations.

## Confidence Is Not Evidence of Support

An observation can have:

- low predictive entropy,
- high model confidence,
- and almost no nearby labelled support.

Those are not contradictory.

A discriminative model can extrapolate confidently.

So entropy-based semi-supervised learning should be paired with support diagnostics where extrapolation is a concern.

Possible proxies include:

- nearest-labelled distance,
- ensemble disagreement,
- density ratio estimates,
- representation-space coverage,
- or conformal-style nonconformity scores.

None is universal.

The point is to avoid treating certainty as evidence that the model has seen enough relevant supervision.

## Entropy Minimisation Can Be Useful

The critique should not be overextended.

When:

- labelled and unlabelled distributions are aligned,
- the label space is shared,
- boundaries lie in relatively low-density regions,
- the initial supervised model is directionally reasonable,
- and class proportions are handled appropriately,

entropy minimisation can improve semi-supervised learning.

The method is not flawed.

Its success is conditional.

## The Correct Interpretation

The entropy term does not say

> these predictions are probably correct.

It says

> among otherwise plausible solutions, prefer one that is more decisive on the unlabelled sample.

That is a regularization principle.

Its validity depends on the structure of the problem.

Once stated that way, the method becomes easier to evaluate honestly.

## A Practical Validation Contract

Before trusting entropy minimisation, I would require several checks.

### 1. Preserve the supervised baseline

Measure exactly what the entropy term adds.

### 2. Sweep the entropy weight

Look for collapse, instability and calibration changes.

### 3. Track predicted class proportions

A sudden drift toward one class can indicate entropy-driven collapse.

### 4. Measure calibration

Low entropy should not be mistaken for reliable probability estimates.

### 5. Examine low-entropy error

Check whether the most confident predictions are actually the most accurate on trusted labels.

### 6. Stress-test class imbalance

Repeat experiments under plausible changes in class prior.

### 7. Stress-test low-density separation

Use synthetic or controlled examples where the true boundary crosses dense regions.

### 8. Check class-space alignment

Do not assume the unlabelled pool contains only known classes.

### 9. Report negative transfer

Retain seeds and subgroups where the entropy term hurts.

## The Core Counterexample

The constant-classifier example is deliberately crude because it isolates the objective.

With one labelled example from each class,

$$
L_\alpha(q)
=
-\log q
-
\log(1-q)
+
\alpha H(q).
$$

The balanced solution

$$
q=\frac12
$$

is always stationary.

Its curvature is

$$
L_\alpha''\left(\frac12\right)
=
8-4\alpha.
$$

Therefore,

$$
\boxed{
\alpha>2
\quad\Longrightarrow\quad
q=\frac12
\text{ becomes a local maximum}.
}
$$

A sufficiently strong entropy term creates pressure toward confident alternatives even though the trusted labels are exactly balanced.

That is the whole warning in one equation.

The unlabelled term creates certainty.

It does not create direction.

## Conclusion

Entropy minimisation is useful because many classification problems do have low-density decision boundaries.

When that assumption is approximately correct, unlabelled data can reveal where uncertain boundaries are geometrically implausible.

But predictive entropy itself contains no notion of correctness.

For binary probability $p$,

$$
H(p)=H(1-p).
$$

So a confidently wrong prediction is just as attractive to the entropy term as a confidently correct one.

The key principle is therefore

$$
\boxed{
\text{low entropy means confidence, not truth}.
}
$$

The supervised loss must orient the classifier.

The geometry of the unlabelled data must make that orientation compatible with decisive predictions.

And the relative weight of the entropy term must not overwhelm the trusted signal.

Use entropy minimisation as a structural prior.

Do not interpret it as supervision magically extracted from uncertainty itself.

## References

- Chapelle, O., Schölkopf, B., & Zien, A. (Eds.). (2006). *Semi-Supervised Learning*. MIT Press.
- Grandvalet, Y., & Bengio, Y. (2005). Semi-supervised learning by entropy minimization. *Advances in Neural Information Processing Systems*, 17.
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop on Challenges in Representation Learning*.
