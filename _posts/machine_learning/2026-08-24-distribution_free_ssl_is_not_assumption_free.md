---
permalink: '/machine-learning/distribution_free_ssl_is_not_assumption_free/'
title: 'Distribution-Free Semi-Supervised Learning Is Not Assumption-Free'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Risk Estimation
- UAI 2026
- Unbiased Estimation
- Variance Reduction
- Statistical Learning
author_profile: false
seo_title: 'A Critical Reading of Distribution-Free SSL with Risk Rewrite'
seo_description: 'A technical reading of Hirose, Irobe and Kanamori (UAI 2026): generalized risk rewriting, variance-optimal unbiased estimators, multiclass SSL, and what distribution-free does and does not mean.'
excerpt: >-
  Hirose, Irobe and Kanamori propose a generalized risk-rewriting framework for
  semi-supervised learning that avoids cluster, manifold and augmentation
  assumptions. The paper is mathematically interesting because unlabelled data are
  used to reduce the variance of an unbiased risk estimator rather than to impose
  geometric structure. But distribution-free is not the same as assumption-free.
summary: >-
  A critical technical reading of the UAI 2026 paper Generalized
  Distribution-Free Semi-Supervised Learning with Risk Rewrite. The article
  derives the basic risk identity, interprets the method as a control-variate
  construction, explains the minimum-variance result and its multiclass extension,
  and examines the assumptions behind the distribution-free terminology,
  including shared class-conditional distributions, class-prior knowledge,
  covariance estimation and the asymptotic treatment of the unlabelled sample.
keywords:
- distribution-free semi-supervised learning
- risk rewriting
- PNU learning
- unbiased risk estimator
- variance reduction
- UAI 2026
- multiclass semi-supervised learning
classes: wide
date: '2026-08-24'
why_this_exists: >-
  Much of modern semi-supervised learning extracts supervision by assuming that
  labels vary smoothly along the geometry of unlabelled data. This paper takes a
  different route: rewrite the same population risk using labelled and unlabelled
  expectations, then choose an unbiased estimator with lower variance. That
  statistical distinction is important and worth separating from the broader
  claim of being distribution-free.
evidence: >-
  Hirose, Irobe and Kanamori, UAI 2026, PMLR 337:2152-2178; the earlier PNU
  framework of Sakai et al., ICML 2017; and the paper's reported binary and
  multiclass experiments across UCI tabular datasets, MNIST and CIFAR-10.
methodology: >-
  Re-derive the risk-rewriting identity from the mixture representation of the
  unlabelled marginal, interpret the resulting estimator family as a
  control-variate problem, inspect the variance formula and generalization
  argument, and then separate the paper's formal guarantees from its empirical
  evidence and practical assumptions.
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

Semi-supervised learning is usually introduced through geometry.

Nearby observations should have similar labels. Decision boundaries should avoid
high-density regions. Predictions should remain stable under perturbations. A graph
built from unlabelled data should carry label information.

All of those ideas can work.

They also require assumptions connecting the marginal feature distribution

$$
P_X
$$

to the conditional label mechanism

$$
P(Y\mid X).
$$

Hirose, Irobe and Kanamori take a different route in their UAI 2026 paper,
*Generalized Distribution-Free Semi-Supervised Learning with Risk Rewrite*.

The unlabelled sample is not used to infer a manifold, construct pseudo-labels or
enforce local smoothness.

It is used to estimate the same classification risk in a different way.

That sounds modest.

Statistically, it is the most interesting part of the paper.

## The Problem Setting

Consider a multiclass problem with labels

$$
Y\in\{1,\ldots,k\}.
$$

Let

$$
p_i(x)
=
p(x\mid Y=i)
$$

be the class-conditional distribution and

$$
\theta_i
=
P(Y=i)
$$

the class prior.

The unlabelled marginal is therefore

$$
\boxed{
p(x)
=
\sum_{i=1}^{k}
\theta_i p_i(x).
}
$$

For a classifier $g$ and loss $\ell$, define the usual population risk

$$
R(g)
=
E_{(X,Y)}
\left[
\ell(g(X),Y)
\right].
$$

It can be decomposed as

$$
R(g)
=
\sum_{i=1}^{k}
\theta_i R_{ii}(g),
$$

where

$$
R_{ij}(g)
=
E_{X\sim p_i}
\left[
\ell(g(X),j)
\right].
$$

The unusual notation is useful.

The first index tells us which class-conditional distribution generated $X$.

The second tells us which label is inserted into the loss.

Now define the corresponding unlabelled risk component

$$
R_{Uj}(g)
=
E_{X\sim p}
\left[
\ell(g(X),j)
\right].
$$

Because the unlabelled distribution is the class mixture,

$$
\boxed{
R_{Uj}(g)
=
\sum_{i=1}^{k}
\theta_i R_{ij}(g).
}
$$

This identity is the engine of the paper.

## Risk Rewriting Is an Algebraic Identity

Start from

$$
R(g)
=
\sum_i
\theta_i R_{ii}(g).
$$

For any coefficient attached to label $j$, the quantity

$$
\sum_i
\theta_i R_{ij}(g)
-
R_{Uj}(g)
$$

is exactly zero.

So we can add arbitrary linear combinations of these zero-valued terms to the
risk without changing the population quantity being estimated.

A convenient parameterization can be written schematically as

$$
R_{\mathrm{lin}}^{a}(g)
=
R(g)
+
\sum_j
c_j(a)
\left[
\sum_i
\theta_i R_{ij}(g)
-
R_{Uj}(g)
\right].
$$

At the population level,

$$
R_{\mathrm{lin}}^{a}(g)
=
R(g)
$$

for every admissible $a$.

The population target is unchanged.

The empirical estimator is not.

Different choices of the coefficients combine noisy labelled and unlabelled
sample averages differently.

That creates a variance optimisation problem.

## The Connection to Control Variates

This is how I would interpret the paper.

Suppose

$$
\widehat R_{\mathrm{sup}}
$$

is an unbiased supervised risk estimator.

Now construct a random quantity

$$
\widehat Z
$$

with

$$
E[\widehat Z]=0.
$$

Then

$$
\widehat R_c
=
\widehat R_{\mathrm{sup}}
+
c\widehat Z
$$

is unbiased for every $c$.

Its variance is

$$
\operatorname{Var}(\widehat R_c)
=
\operatorname{Var}(\widehat R_{\mathrm{sup}})
+
2c
\operatorname{Cov}(\widehat R_{\mathrm{sup}},\widehat Z)
+
c^2
\operatorname{Var}(\widehat Z).
$$

The optimal coefficient is the familiar control-variate coefficient

$$
c^\star
=
-
\frac{
\operatorname{Cov}(\widehat R_{\mathrm{sup}},\widehat Z)
}{
\operatorname{Var}(\widehat Z)
}.
$$

Risk rewriting generalizes this idea to a vector of zero-mean identities built
from class-conditional and unlabelled risks.

The unlabelled data are valuable because they help construct estimators of the
same target with smaller sampling variance.

That is fundamentally different from saying that the geometry of unlabelled data
reveals the decision boundary.

## The Multiclass Generalization

Earlier PNU learning combines positive-negative supervised risk with
positive-unlabelled or negative-unlabelled risk.

That construction is naturally binary.

Hirose and colleagues define a larger family of linear risk rewrites containing
all class-to-label component risks and the corresponding unlabelled risks.

Under a linear-independence condition on the component risks, that family can be
parameterized by a vector

$$
a\in\mathbb R^k.
$$

This is one of the useful conceptual contributions of the paper.

Instead of designing one rewritten risk for each special case, define the entire
linear family of unbiased risk estimators and optimise inside it.

PNU then becomes one member, or a restricted subfamily, of a more general
construction.

## The Minimum-Variance Result

For a fixed classifier $g$, define for each true class $m$ a covariance matrix

$$
C_m,
$$

whose entries are covariances between losses evaluated with different candidate
labels under

$$
X\sim p_m.
$$

The paper forms the weighted covariance matrix

$$
S
=
\sum_{m=1}^{k}
\frac{\theta_m^2}{n_m}
C_m
$$

and a corresponding weighted vector

$$
u
=
\sum_{m=1}^{k}
\frac{\theta_m^2}{n_m}
C_m e_m,
$$

where $e_m$ is the $m$th coordinate vector.

In the asymptotic regime

$$
n_U\rightarrow\infty,
$$

the minimum variance over the linear estimator family has the form

$$
\boxed{
\operatorname{Var}_{\min}
=
\sum_{m=1}^{k}
\frac{\theta_m^2}{n_m}
(C_m)_{mm}
-
u^T S^{-1}u.
}
$$

The first term is the variance of the corresponding supervised estimator for
fixed $g$.

The second term is non-negative.

So the optimized rewrite can only reduce, or in the degenerate case leave
unchanged, the pointwise asymptotic variance.

This is the mathematical result that gives the framework substance.

## Why Asymmetric Losses Matter

The paper reports an interesting distinction between symmetric and asymmetric
losses.

For symmetric losses such as the zero-one loss, the risk components satisfy
additional linear relations.

The larger estimator family then loses some of the degrees of freedom that make
the generalized rewrite more powerful.

In the binary experiments, the optimized generalized estimator and PNU become
essentially equivalent in variance under symmetric loss.

Under asymmetric loss such as binary cross entropy, the larger estimator family
can do better.

That is exactly what one would expect from the control-variate interpretation.

More independent covariance structure gives more room to choose useful
coefficients.

## A Concrete Reported Example

The paper includes a controlled binary variance experiment.

With a balanced class prior and binary cross entropy, PNU provides only a small
variance reduction in one reported configuration.

At

$$
n_U=1000,
$$

the reported PNU variance ratio relative to supervised estimation is about

$$
0.953,
$$

while the generalized linear estimator reaches about

$$
0.883.
$$

The numbers are less important than the mechanism.

PNU restricts the admissible rewrite.

The generalized method searches a larger unbiased family.

When the covariance structure is asymmetric, the additional degrees of freedom
can matter.

## Variance Reduction Is Not Yet Learning

A lower-variance estimate of

$$
R(g)
$$

for one fixed classifier is useful.

But a learning algorithm chooses

$$
\widehat g
=
\arg\min_{g\in\mathcal G}
\widehat R(g).
$$

Pointwise variance reduction does not automatically imply that the empirical
minimizer generalizes better.

The paper addresses this explicitly.

That step is important.

## The Generalization Argument

The authors derive an excess-risk bound for the empirical minimizer based on their
rewritten estimator.

The main qualitative message is that the dominant estimation term depends on a
maximum variance quantity over the hypothesis class.

If the labelled sample sizes grow at the same order, the paper notes that this
dominant term scales like

$$
\mathcal O(n^{-1/2}),
$$

while another finite-sample term scales like

$$
\mathcal O(n^{-1}).
$$

Within the assumptions of the theorem, reducing the relevant maximum variance
tightens the bound.

That provides the missing bridge:

$$
\text{lower estimator variance}
\rightarrow
\text{tighter uniform risk estimation}
\rightarrow
\text{potentially better learned predictor}.
$$

This is stronger than arguing from pointwise variance alone.

## What "Distribution-Free" Means Here

This is where terminology needs care.

The paper is distribution-free in an important sense.

It does not assume that:

- classes correspond to density clusters,
- the decision boundary lies in a low-density region,
- the data lie on a useful manifold,
- augmentations preserve labels,
- graph neighbours share labels,
- or confident predictions are correct.

That is a meaningful contrast with a large part of modern SSL.

But distribution-free does not mean assumption-free.

## The Labelled and Unlabelled Samples Must Belong to the Same Mixture Model

The identity

$$
R_{Uj}
=
\sum_i
\theta_i R_{ij}
$$

requires the unlabelled marginal to be

$$
p(x)
=
\sum_i
\theta_i p_i(x),
$$

using the same class-conditional distributions represented by the labelled
samples.

If labelled and unlabelled data come from different acquisition processes, time
periods or populations, then the empirical terms may estimate different objects.

Covariate shift is not automatically neutralized by calling the estimator
distribution-free.

The method avoids a geometric assumption.

It still needs the sampling identity behind the risk rewrite.

## Class Priors Matter

The construction uses

$$
\theta_i
=
P(Y=i).
$$

The experiments assume these priors are known.

The paper notes that class-prior estimation methods can be used in practice.

That is reasonable.

It also means a practical implementation inherits another estimation problem.

If

$$
\widehat\theta_i
$$

is biased or noisy, the exact unbiasedness of the rewritten empirical risk no
longer holds automatically.

A deployment-oriented extension should therefore propagate class-prior
uncertainty rather than treating it as a side issue.

## The Clean Variance Theory Uses an Infinite-Unlabelled Limit

The minimum-variance derivation is presented under

$$
n_U\rightarrow\infty.
$$

This removes the variance of the unlabelled-risk estimates and isolates how
labelled-sample covariance can be exploited.

That is mathematically clean.

In finite data, the unlabelled components are noisy too.

The paper does study finite

$$
n_U
$$

experimentally and reports decreasing variance ratios as the unlabelled sample
grows.

Still, the asymptotic simplification matters when interpreting the theorem.

The finite-unlabelled optimum is a different covariance problem.

## The Generalization Theorem and Cross Entropy Need to Be Kept Distinct

The generalization theorem described in the paper assumes a bounded loss,

$$
0
\leq
\ell
\leq
c_\ell,
$$

together with Lipschitz regularity.

The experiments use cross-entropy loss.

Ordinary unrestricted cross entropy is not globally bounded.

So the theorem, as stated, should not be read as a direct guarantee for every
cross-entropy experiment unless additional restrictions control the prediction
range.

This does not invalidate the experiments.

It does mean that the theoretical and empirical sections operate under slightly
different regimes.

That distinction is worth stating rather than smoothing over.

## The Linear-Independence Assumption Is Also Structural

The convenient $k$-dimensional parameterization requires a linear-independence
assumption on the component risks.

The authors explicitly note that symmetric losses violate this condition because
their loss components obey deterministic linear relations.

They handle that case separately.

This is good mathematical hygiene.

It also shows why the framework should not be summarized as a completely
assumptionless algebraic trick.

The structure of the loss function matters.

## The Two Practical Methods

The paper proposes two main practical variants for multiclass SSL.

The iterative method estimates covariance-related quantities and alternates between
updating the classifier and the rewrite coefficients.

I will refer to it as the iterative estimator.

The second uses an equal-covariance approximation.

This avoids the need to estimate the full covariance structure used by the optimal
coefficient calculation.

The approximation is attractive because the exact variance-optimal coefficients
depend on quantities that are unknown before the classifier is trained.

This is the gap between the clean population result and a usable algorithm.

## Covariance Estimation Is Not Free Either

The iterative method needs data to estimate covariance terms.

The paper includes an ablation on the number of covariance-estimation examples per
class and reports that performance stabilizes with a modest number, with strong
regularization helping even at very small sizes.

That is useful evidence.

But the important accounting question remains:

> Where do those trusted examples come from?

If they are taken from a scarce labelled budget, the method is trading training
labels for estimator calibration.

That can still be an excellent trade.

It should be part of the sample-efficiency calculation.

## The Equal-Covariance Approximation Is Pragmatic

The equal-covariance variant is particularly interesting because it provides a
data-light rule analogous to the equal-variance simplification used in binary PNU.

The paper also checks how closely the covariance matrices satisfy that
approximation.

The reported diagnostic is not especially small in several datasets, and the
authors do not find a simple relationship between that local covariance mismatch
and predictive performance.

That result is actually useful.

It warns against treating a convenient covariance approximation as a literal
empirical truth.

An approximation can work without being descriptively exact.

## What the Experiments Show

The empirical section covers:

- binary tabular datasets,
- binary image tasks,
- seven-class tabular datasets,
- ten-class MNIST,
- balanced labelled samples,
- mild class imbalance,
- and severe class imbalance.

The multiclass experiments use

$$
n_U=5000
$$

unlabelled observations.

Results are reported over 30 seeds.

The proposed methods frequently match or outperform supervised learning,
pseudo-labelling and VAT, and the multiclass extension is clearly viable.

That supports the practical value of the framework.

It does not support a stronger statement that the method uniformly dominates
supervised learning.

There are rows where the supervised baseline is slightly better than the
iterative method.

That is exactly what one should expect from an estimator whose benefit depends on
covariance structure.

## The Baseline Choice Matters

The paper compares against supervised learning, PNU where applicable,
pseudo-labelling and VAT.

Those are useful baselines for isolating the paper's statistical contribution.

They are not an exhaustive representation of modern SSL in 2026.

There is no large comparison against the current generation of
augmentation-heavy, teacher-student or foundation-model-based SSL systems.

I do not see that as a defect in the theory paper.

But it limits the interpretation of empirical competitiveness.

The result is better read as

> risk rewriting is practically competitive in the tested controlled regimes

than as

> risk rewriting is the new state of the art in semi-supervised learning.

Those are very different claims.

## The Most Interesting Contribution Is Not Accuracy

For me, the strongest contribution is conceptual.

The paper reminds us that unlabelled data can help without pretending to contain
missing labels.

The benefit can be purely statistical:

$$
\boxed{
\text{same target risk}
+
\text{different unbiased estimator}
+
\text{lower variance}.
}
$$

That is a clean alternative to the usual SSL narrative.

No pseudo-label needs to be believed.

No manifold needs to be identified.

No augmentation needs to preserve the target.

No graph needs to encode class smoothness.

The unlabelled marginal participates through an exact expectation identity.

## Why This Fits the Earlier Identifiability Argument

I previously argued that unlabelled data identifies

$$
P_X
$$

rather than

$$
P(Y\mid X).
$$

At first glance, distribution-free risk rewriting might appear to contradict that
statement.

It does not.

The method does not infer

$$
P(Y\mid X)
$$

from

$$
P_X
$$

alone.

It already has labelled samples from each class and class-prior information.

The unlabelled marginal is then used to estimate a known linear combination of
class-conditional expectations more efficiently.

That is a very different information structure.

The distinction is worth preserving.

## A Small Derivation Makes the Point

Suppose for one label $j$ we know

$$
R_{Uj}
=
\sum_i
\theta_i R_{ij}.
$$

Then the quantity

$$
Z_j
=
\sum_i
\theta_i \widehat R_{ij}
-
\widehat R_{Uj}
$$

has expectation zero when all empirical samples represent the stated population
components.

An estimator

$$
\widehat R_c
=
\widehat R_{\mathrm{sup}}
+
c Z_j
$$

remains unbiased.

If

$$
Z_j
$$

is correlated with the supervised estimation error, choosing $c$ appropriately
reduces variance.

This is not label inference.

It is variance cancellation.

That is the paper in its simplest statistical form.

## Where I Would Extend the Work

Several extensions seem natural.

### Finite-unlabelled optimality

Derive and optimize the full covariance expression without taking

$$
n_U\rightarrow\infty.
$$

That would make the theory directly informative about how many unlabelled
observations are actually enough.

### Estimated class priors

Treat

$$
\widehat\theta
$$

as random and derive the additional bias and variance it introduces.

This is important under class-prior shift.

### Distribution shift

Study what happens when

$$
p_U(x)
\neq
\sum_i
\theta_i p_i(x)
$$

because labelled and unlabelled samples come from different domains.

The exact risk identity then becomes misspecified.

A robust rewrite would be more useful than simply assuming the identity survives.

### Cross-fitting covariance estimation

The iterative method estimates covariance quantities from trusted data.

Cross-fitting could reduce reuse bias and make the sample accounting more explicit.

### Modern pretrained representations

The method is model-agnostic in principle.

It would be interesting to test whether variance reduction remains useful when the
classifier is a frozen or parameter-efficiently tuned foundation model, where the
labelled-sample regime and covariance structure can look very different.

## A Stronger Evaluation Question

The next empirical question should not only be:

> Does the method improve test accuracy?

It should be:

> When does the estimated variance reduction predict the actual gain over supervised learning?

The theory says variance is the mechanism.

So variance reduction should have predictive value across tasks, seeds and
labelled-data regimes.

If it does, the framework becomes diagnostically useful.

If it does not, then the bound may be too loose to guide practice even if it is
formally correct.

That is the kind of theory-to-experiment link I would like to see tested next.

## What I Take From the Paper

The paper makes three contributions that I think are genuinely important.

First, it turns binary PNU-style rewriting into a general multiclass linear family.

Second, it gives a variance-optimal interpretation rather than choosing rewrite
coefficients heuristically.

Third, it connects that variance reduction to a learning-theoretic excess-risk
bound.

Those pieces fit together.

The main caution is linguistic.

"Distribution-free" should be read as

> no cluster, manifold or comparable geometric assumption on the feature
> distribution is needed for the risk-rewriting identity.

It should not be read as

> labelled and unlabelled data may come from arbitrary unrelated distributions,
> class priors need not be known, losses need no regularity, and finite-sample
> covariance estimation is irrelevant.

The paper does not claim all of that.

Readers should not silently promote the phrase into those stronger statements.

## Conclusion

Hirose, Irobe and Kanamori provide a useful alternative way to think about
semi-supervised learning.

Instead of asking unlabelled data to reveal class geometry, they ask it to help
estimate the same classification risk more efficiently.

The key identity is simple:

$$
R_{Uj}
=
\sum_i
\theta_i R_{ij}.
$$

The important step is recognizing that this identity generates a family of
unbiased empirical risk estimators with different variances.

Optimizing that family gives a multiclass generalization of PNU learning and, in
the asymptotic unlabelled regime, a closed-form variance reduction relative to the
supervised estimator.

That is a real statistical contribution.

My main interpretation is therefore:

$$
\boxed{
\text{unlabelled data need not supply labels to supply information}.
}
$$

It can supply covariance information that improves risk estimation.

But the method is not assumption-free.

Its assumptions are simply different from the geometric assumptions that dominate
modern SSL.

That distinction is precisely why the paper is worth reading.

## References

- Hirose, Y., Irobe, H., & Kanamori, T. (2026). Generalized Distribution-Free Semi-Supervised Learning with Risk Rewrite. *Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence*, PMLR 337, 2152–2178. https://proceedings.mlr.press/v337/hirose26a.html
- Sakai, T., du Plessis, M. C., Niu, G., & Sugiyama, M. (2017). Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 2998–3006. https://proceedings.mlr.press/v70/sakai17a.html
- Chapelle, O., Schölkopf, B., & Zien, A. (Eds.). (2006). *Semi-Supervised Learning*. MIT Press.
