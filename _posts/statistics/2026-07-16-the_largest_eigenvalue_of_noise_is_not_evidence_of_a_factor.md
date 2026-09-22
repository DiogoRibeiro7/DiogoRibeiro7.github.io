---
permalink: '/statistics/the_largest_eigenvalue_of_noise_is_not_evidence_of_a_factor/'
title: 'The Largest Eigenvalue of Noise Is Not Evidence of a Factor'
date: '2026-07-16'
categories:
- Statistics
tags:
- Random Matrix Theory
- Principal Component Analysis
- Covariance Estimation
- High Dimensional Statistics
- Marchenko Pastur
author_profile: false
classes: wide
seo_title: 'The Largest Eigenvalue of Noise Is Not Evidence of a Factor'
seo_description: 'In high dimensions, pure noise produces a nontrivial spectrum of sample covariance eigenvalues. Marchenko-Pastur theory gives the baseline, and spiked covariance models show when a real factor actually separates from noise.'
seo_type: article
excerpt: >-
  When the number of variables is not small relative to the sample size, sample
  covariance eigenvalues spread dramatically even if the true covariance is the
  identity. Large eigenvalues can therefore arise from noise alone.
summary: >-
  This article develops random matrix theory as a baseline for high-dimensional
  covariance analysis. For p/n = 0.5 and identity covariance, Marchenko-Pastur
  theory predicts sample eigenvalues between about 0.086 and 2.914 even though
  every population eigenvalue equals one. The article derives this support,
  explains why sample covariance becomes badly conditioned, connects the result
  to PCA, and develops the spiked covariance phase transition: a population
  spike must exceed 1 + sqrt(c) before its sample eigenvalue separates
  asymptotically from the noise bulk. The discussion then covers eigenvector
  inconsistency, Tracy-Widom edge fluctuations, shrinkage covariance, factor
  selection, and the limits of classical PCA in p/n-not-small regimes.
keywords:
- random matrix theory
- Marchenko Pastur
- sample covariance
- PCA
- spiked covariance
- high dimensional statistics
why_this_exists: >-
  Classical multivariate statistics is often taught under fixed dimension and
  increasing sample size. In modern problems the number of variables can be a
  substantial fraction of the sample size, and then sample covariance behaves
  qualitatively differently. Random matrix theory supplies the correct noise
  baseline.
evidence: >-
  Marchenko-Pastur asymptotics for Wishart covariance matrices, exact support
  calculations, the Baik-Ben Arous-Peche phase transition for spiked covariance,
  and standard results on spectral shrinkage and high-dimensional PCA.
methodology: >-
  Start from iid Gaussian observations with identity population covariance and
  derive the Marchenko-Pastur support. Use p/n = 0.5 as a numerical example,
  then introduce a rank-one population spike and show when the leading sample
  eigenvalue remains buried in noise and when it separates. Connect these
  asymptotics to PCA, covariance conditioning and factor selection.
reviewed_at: '2026-09-22'
header:
  image: /assets/images/kernel_math.webp
  og_image: /assets/images/kernel_math.webp
  overlay_image: /assets/images/kernel_math.webp
  show_overlay_excerpt: false
  teaser: /assets/images/kernel_math.webp
  twitter_image: /assets/images/kernel_math.webp
---

<!--
Development contract
Question: When is a large sample covariance eigenvalue actually evidence of structure rather than an ordinary fluctuation of high-dimensional noise?
Claim: When p/n is not negligible, sample covariance eigenvalues have a broad random spectrum even under identity covariance. The Marchenko-Pastur law provides the null bulk, and a population spike must cross a detectability threshold before it produces an asymptotically separated sample eigenvalue.
Counterclaim: Random matrix asymptotics are not an automatic significance test. Finite samples, non-Gaussian tails, dependence, heteroskedasticity, missing data and preprocessing can shift the spectral baseline materially.
Evidence object: Identity-covariance Marchenko-Pastur support at p/n=0.5, condition-number inflation, and a rank-one spiked covariance model below and above the separation threshold.
Failure case: Keeping principal components because their eigenvalues exceed one, applying Marchenko-Pastur thresholds after arbitrary standardization or temporal dependence without adjustment, or treating a separated eigenvalue as proof of a scientifically meaningful latent factor.
Reader payoff: Use random matrix theory to distinguish high-dimensional sampling noise from genuine covariance structure and understand why classical PCA heuristics break when dimension grows with sample size.
Exclusions: A complete random-matrix-theory survey, proofs of universality, and a finance-only treatment of covariance cleaning.
-->

Principal component analysis begins with a deceptively simple object: the eigenvalues and eigenvectors of a sample covariance matrix. In low-dimensional classical statistics, the intuition is straightforward. If the population covariance is the identity, every true eigenvalue equals one. With enough observations, the sample covariance should be close to the identity, so sample eigenvalues should cluster near one. A conspicuously large eigenvalue therefore looks like evidence of a latent factor.

That intuition depends on an asymptotic regime that modern datasets often violate.

If the number of variables \(p\) grows with the sample size \(n\) so that the ratio

$$
c
=
\frac{p}{n}
$$

does not vanish, then even pure noise produces a broad and highly structured eigenvalue distribution. The sample covariance matrix does not converge to the identity in operator norm. Its smallest and largest eigenvalues move away from one by amounts that remain substantial no matter how large \(n\) becomes, provided \(p/n\) approaches a positive constant.

Random matrix theory is the mathematical framework that describes this regime.

The key practical consequence is uncomfortable but important: a large principal-component eigenvalue is not automatically evidence of a factor. Before interpreting spectral structure, one needs a noise baseline appropriate to the dimensionality of the problem.

## Identity covariance does not produce sample eigenvalues near one in high dimension

Let

$$
X_1,\ldots,X_n
$$

be independent \(p\)-dimensional Gaussian vectors with

$$
X_i
\sim
N(0,I_p).
$$

The population covariance is exactly

$$
\Sigma
=
I_p.
$$

Every population eigenvalue equals one.

Form the sample covariance matrix

$$
S
=
\frac1n
\sum_{i=1}^n
X_iX_i^\top.
$$

Equivalently, if \(X\) is the \(p\times n\) data matrix,

$$
S
=
\frac1n
XX^\top.
$$

For fixed \(p\) and

$$
n\to\infty,
$$

the law of large numbers implies

$$
S
\to
I_p
$$

entrywise and in stronger matrix senses. This is the regime behind classical covariance intuition.

Now change the asymptotics.

Let

$$
p,n\to\infty
$$

together with

$$
\frac pn
\to
c
\in
(0,\infty).
$$

Then the empirical distribution of the eigenvalues of \(S\) converges to the Marchenko-Pastur law.

For

$$
0<c\le1,
$$

the limiting density is

$$
f_c(\lambda)
=
\frac{
\sqrt{
(\lambda_+-\lambda)(\lambda-\lambda_-)
}
}{
2\pi c\lambda
},
$$

for

$$
\lambda\in[\lambda_-,\lambda_+],
$$

where

$$
\lambda_\pm
=
(1\pm\sqrt c)^2.
$$

The average population eigenvalue is still one.

The sample eigenvalues are not concentrated at one.

They occupy an interval whose width depends on \(c\).

This is a qualitative change in covariance geometry.

## At p/n = 0.5, pure noise produces eigenvalues up to about 2.914

Take

$$
c
=
\frac pn
=
0.5.
$$

Then

$$
\sqrt c
=
\frac1{\sqrt2}
\approx
0.7071.
$$

The Marchenko-Pastur edges are

$$
\lambda_-
=
(1-\sqrt{0.5})^2
\approx
0.0858,
$$

and

$$
\lambda_+
=
(1+\sqrt{0.5})^2
\approx
2.9142.
$$

So even though the true covariance matrix is exactly

$$
I_p,
$$

high-dimensional sampling noise alone produces sample covariance eigenvalues spread approximately across

$$
[0.086,2.914].
$$

A sample eigenvalue of

$$
2.5
$$

may look enormous relative to the true eigenvalue one.

Under this high-dimensional null model, it is not remarkable at all.

It lies inside the ordinary noise bulk.

This immediately undermines several common heuristics.

The Kaiser rule in PCA retains components with eigenvalues above one after standardization. Under the Marchenko-Pastur null with \(c=0.5\), a large fraction of pure-noise eigenvalues exceed one.

A scree plot can show apparently substantial leading components even when every population direction has exactly the same variance.

The phenomenon is not caused by non-Gaussianity, hidden factors or numerical instability.

It occurs in the ideal Gaussian identity-covariance model.

## High-dimensional noise also creates artificial ill-conditioning

The population covariance

$$
\Sigma=I_p
$$

has condition number

$$
\kappa(\Sigma)=1.
$$

The sample covariance condition number is approximately

$$
\kappa(S)
\approx
\frac{
\lambda_+
}{
\lambda_-
}
=
\left(
\frac{
1+\sqrt c
}{
1-\sqrt c
}
\right)^2
$$

when

$$
c<1.
$$

At

$$
c=0.5,
$$

this gives

$$
\kappa(S)
\approx
\frac{
2.9142
}{
0.0858
}
\approx
33.97.
$$

The true covariance is perfectly conditioned.

Its high-dimensional sample estimate behaves as though some directions have roughly 34 times the variance of others.

That spread is entirely sampling noise.

As

$$
c\uparrow1,
$$

the lower edge

$$
\lambda_-
=
(1-\sqrt c)^2
$$

approaches zero and the condition number explodes.

At

$$
c=1,
$$

the limiting support begins at zero.

If

$$
p>n,
$$

the sample covariance matrix has rank at most \(n\), so at least

$$
p-n
$$

eigenvalues are exactly zero.

This is not a small-sample inconvenience that disappears with careful numerical linear algebra.

It is a structural consequence of trying to estimate a \(p\times p\) covariance matrix with only \(n\) observations.

When \(p\) is of the same order as \(n\), classical covariance inversion becomes intrinsically unstable.

## PCA diagonalizes both signal and sampling noise

Principal component analysis finds eigenvectors

$$
v_j
$$

and eigenvalues

$$
\hat\lambda_j
$$

of the sample covariance,

$$
Sv_j
=
\hat\lambda_jv_j.
$$

The leading eigenvector maximizes sample variance,

$$
v_1
=
\arg\max_{\|v\|=1}
v^\top Sv.
$$

That optimization matters.

Even when every direction has the same population variance, PCA searches over many directions and selects the one that looks most variable in the sample.

In high dimension there are many opportunities for one random direction to look unusually strong.

The leading eigenvalue is therefore an extreme statistic, not an ordinary unbiased estimate of one fixed variance.

This is conceptually similar to multiple comparisons. Searching over many candidate directions induces an optimism effect.

Random matrix theory quantifies that search effect through the eigenvalue spectrum.

Under identity covariance, the largest eigenvalue does not converge to one.

It converges to

$$
(1+\sqrt c)^2
$$

at first order.

PCA is not "finding a factor" in this null case.

It is finding the direction in which high-dimensional noise happened to align most strongly.

## The spiked covariance model asks when a real factor escapes the noise bulk

Now introduce genuine structure.

Suppose the population covariance has one elevated eigenvalue,

$$
\Sigma
=
\operatorname{diag}
(
\ell,
1,
1,
\ldots,
1
),
$$

with

$$
\ell>1.
$$

This is the rank-one spiked covariance model.

The first coordinate contains a real population factor with variance

$$
\ell,
$$

while all orthogonal directions retain variance one.

One might expect any

$$
\ell>1
$$

to produce a leading sample eigenvalue separated from the noise bulk.

In high dimension, it does not.

There is a phase transition.

For the standard spiked Wishart model with aspect ratio

$$
c=\frac pn,
$$

the critical population spike is

$$
\ell_c
=
1+\sqrt c.
$$

If

$$
1<\ell
\le
1+\sqrt c,
$$

the leading sample eigenvalue does not separate asymptotically from the Marchenko-Pastur upper edge.

The factor exists in the population covariance.

Its sample eigenvalue is nevertheless buried in the noise bulk.

If

$$
\ell
>
1+\sqrt c,
$$

an outlier eigenvalue emerges.

Its asymptotic sample location is

$$
\hat\ell
\to
\ell
\left(
1+
\frac{
c
}{
\ell-1
}
\right).
$$

This is the Baik-Ben Arous-Peche type transition in the covariance setting.

The result is one of the clearest examples of a high-dimensional detectability threshold.

## At c = 0.5, a true eigenvalue of 1.5 can remain invisible

Again take

$$
c=0.5.
$$

The critical spike is

$$
\ell_c
=
1+\sqrt{0.5}
\approx
1.7071.
$$

Suppose the population has a genuine leading eigenvalue

$$
\ell=1.5.
$$

This is 50% larger than the noise eigenvalues.

Yet

$$
1.5
<
1.7071.
$$

Asymptotically, the corresponding sample eigenvalue does not detach from the noise edge.

It remains buried near

$$
\lambda_+
\approx
2.914.
$$

This is deeply counterintuitive from a low-dimensional perspective.

The true signal eigenvalue is larger than one.

The sample leading eigenvalue can be larger still because noise itself creates eigenvalues near 2.9.

The real factor is therefore spectrally undetectable by simple eigenvalue separation.

Now take

$$
\ell=3.
$$

This exceeds the threshold comfortably.

The asymptotic sample outlier is

$$
\hat\ell
\to
3
\left(
1+
\frac{
0.5
}{
3-1
}
\right)
=
3(1.25)
=
3.75.
$$

This lies beyond the Marchenko-Pastur edge

$$
2.914.
$$

The signal separates.

The population spike must be strong enough not merely to exceed the baseline eigenvalue one, but to overcome the high-dimensional noise geometry.

## Sample eigenvalues are biased upward after separation

The spiked-model formula also reveals another high-dimensional effect.

If the population spike is

$$
\ell=3,
$$

the separated sample eigenvalue converges to

$$
3.75,
$$

not to

$$
3.
$$

The leading sample eigenvalue is asymptotically biased upward.

For very strong spikes,

$$
\ell\gg1,
$$

the correction

$$
\frac{
c
}{
\ell-1
}
$$

becomes small and classical intuition gradually returns.

Near the separation threshold, the bias can be substantial.

This matters whenever PCA eigenvalues are interpreted as direct estimates of explained population variance.

In high dimension,

$$
\hat\lambda_j
$$

and

$$
\lambda_j
$$

are not interchangeable.

Spectral shrinkage methods attempt to invert or correct this high-dimensional distortion.

The problem is not simply to denoise matrix entries.

It is to denoise eigenvalues under nonlinear spectral bias.

## Eigenvectors also undergo a phase transition

An eigenvalue can separate without its sample eigenvector perfectly recovering the population factor direction.

Let

$$
u
$$

be the true spike eigenvector and

$$
\hat u
$$

the leading sample eigenvector.

A useful measure of alignment is

$$
|\langle
\hat u,
u
\rangle|^2.
$$

Below the spike threshold, the sample eigenvector becomes asymptotically uninformative about the true factor direction under the standard model.

Its squared overlap tends to zero.

Above the threshold, nonzero asymptotic alignment appears.

One common expression for the limiting squared cosine in the rank-one model is

$$
|\langle
\hat u,
u
\rangle|^2
\to
\frac{
1-
\frac{
c
}{
(\ell-1)^2
}
}{
1+
\frac{
c
}{
\ell-1
}
},
$$

for

$$
\ell>1+\sqrt c.
$$

Take

$$
c=0.5
$$

and

$$
\ell=3.
$$

Then

$$
\ell-1=2,
$$

so

$$
|\langle
\hat u,
u
\rangle|^2
\to
\frac{
1-\frac{0.5}{4}
}{
1+\frac{0.5}{2}
}
=
\frac{
0.875
}{
1.25
}
=
0.70.
$$

Even with a clearly separated eigenvalue, the sample eigenvector captures only about 70% squared alignment with the true factor direction asymptotically.

The principal component is informative and noisy.

This matters when loadings are interpreted scientifically.

A leading PC can have a stable-looking eigenvalue and unstable individual coefficients.

High-dimensional PCA uncertainty is not summarized by the eigenvalue alone.

## The Marchenko-Pastur law is a null model, not a universal truth

The clean derivation assumes independent observations with identity covariance and regular tail behaviour.

Real data can violate every part of that setup.

Serial dependence reduces the effective sample size.

Heteroskedasticity changes the spectral distribution.

Heavy tails create extreme eigenvalues that can overwhelm the classical bulk.

Missing-data handling and imputation alter covariance structure.

Standardizing each variable by an estimated sample variance introduces additional dependence.

Nonlinear relationships can produce covariance patterns that PCA does not represent well.

Block dependence among variables changes the population spectrum before sampling noise is added.

The correct use of Marchenko-Pastur theory is therefore not:

> every eigenvalue below \(\lambda_+\) is noise and every eigenvalue above it is signal.

The correct interpretation is conditional:

> under an appropriate high-dimensional null model, this is the expected noise spectrum.

One should first ask whether the assumptions defining that null model are scientifically credible.

For time-series data, effective aspect ratio may differ substantially from

$$
p/n
$$

because observations are autocorrelated.

For heavy-tailed data, robust covariance estimators or heavy-tail random matrix theory may be more appropriate.

The noise baseline itself is a model.

## Edge fluctuations live on a smaller scale than the bulk

The Marchenko-Pastur upper edge gives the first-order location of the largest noise eigenvalue.

Finite samples fluctuate around that edge.

For Gaussian Wishart matrices, after appropriate centering and scaling, the largest eigenvalue has Tracy-Widom fluctuations.

Schematically,

$$
\lambda_{\max}
=
\lambda_+
+
O_p
\left(
n^{-2/3}
\right)
$$

on the normalized covariance scale, with constants depending on the precise finite-\(n,p\) normalization.

This matters if one wants a formal significance threshold for a leading eigenvalue.

Comparing only with the deterministic asymptotic edge

$$
\lambda_+
$$

ignores finite-sample variability.

A value slightly above the edge is not automatically significant.

Johnstone's work on the largest eigenvalue of Wishart matrices provides finite-sample centering and scaling formulas that make the Tracy-Widom approximation useful in practice.

Permutation or bootstrap methods can also construct empirical null spectra, especially when preprocessing or dependence makes the classical Wishart model questionable.

The broad lesson is familiar from hypothesis testing.

A null expectation is not the same as a null distribution.

## Parallel analysis is a simulation version of the same idea

Parallel analysis in PCA compares observed sample eigenvalues with eigenvalues generated under a noise reference model.

Instead of using the Marchenko-Pastur law analytically, simulate datasets with no latent factor structure but similar dimensions, compute their eigenvalues, and retain components that exceed a chosen null quantile.

The logic is essentially random-matrix logic.

The observed spectrum should be interpreted relative to the spectrum that dimensionality alone would produce.

The method can accommodate preprocessing choices more flexibly than a closed-form Marchenko-Pastur threshold.

Its quality depends entirely on the simulated null.

If real variables are heavy-tailed, autocorrelated or constrained but the null uses iid Gaussian noise, the reference spectrum can be wrong.

A good parallel analysis therefore reproduces relevant nuisance structure while removing the latent-factor structure being tested.

The same principle applies beyond PCA.

A randomization baseline is useful only when it preserves the aspects of the data that are not under investigation.

## Covariance shrinkage addresses the same high-dimensional instability from another direction

The sample covariance

$$
S
$$

is unbiased entrywise under iid sampling:

$$
E[S]
=
\Sigma.
$$

That does not make it a good high-dimensional matrix estimator under operator norms or downstream inversion.

A simple shrinkage estimator is

$$
\hat\Sigma_\alpha
=
(1-\alpha)S
+
\alpha T,
$$

where

$$
T
$$

is a structured target such as a scaled identity.

Shrinkage pulls extreme sample eigenvalues toward a more stable spectrum.

Ledoit-Wolf shrinkage chooses the intensity data-adaptively under a quadratic matrix-loss criterion.

Nonlinear shrinkage goes further and adjusts sample eigenvalues differently according to their location in the spectrum.

The random-matrix perspective explains why this helps.

The observed spectral spread contains both genuine population heterogeneity and high-dimensional sampling distortion.

If

$$
p/n
$$

is not small, treating raw sample eigenvalues as population eigenvalues exaggerates dispersion.

Shrinkage attempts to undo that exaggeration.

The same issue appears in portfolio optimization, discriminant analysis, Mahalanobis distance, Gaussian likelihoods and any method requiring

$$
S^{-1}.
$$

A poorly conditioned sample covariance can destabilize everything downstream.

Covariance regularization is therefore not merely a numerical fix.

It is statistical estimation adapted to the high-dimensional regime.

## Explained-variance percentages can be deceptive under pure noise

PCA summaries often report

$$
\frac{
\hat\lambda_j
}{
\sum_{k=1}^{p}
\hat\lambda_k
}
$$

as the fraction of variance explained by component \(j\).

Under identity covariance,

$$
\sum_k
\hat\lambda_k
$$

is close to

$$
p,
$$

while the largest eigenvalue is close to

$$
(1+\sqrt c)^2.
$$

At

$$
c=0.5
$$

and large

$$
p,
$$

the leading noise component therefore explains roughly

$$
\frac{
2.914
}{
p
}.
$$

For

$$
p=20,
$$

that is about 14.6%.

For

$$
p=100,
$$

it is about 2.9%.

The percentage depends strongly on dimension.

A scree plot can still show a visually dominant first component because the leading edge of the noise spectrum is systematically above the centre.

Interpreting "variance explained" without a null baseline can therefore overstate evidence for structure.

The relevant question is not whether the first eigenvalue is larger than the average.

It must be larger than what high-dimensional noise predicts.

## Standardization does not make the noise problem disappear

Analysts often switch from covariance PCA to correlation PCA by standardizing every variable to sample variance one.

This removes scale differences across variables.

It does not restore low-dimensional asymptotics.

The sample correlation matrix still has a noisy high-dimensional spectrum.

The exact limiting law differs slightly because the diagonal normalization is estimated from the same data, but under broad conditions the Marchenko-Pastur picture remains relevant.

Standardization also changes the scientific estimand.

A low-variance variable and a high-variance variable receive equal marginal scale after normalization.

That can be appropriate when units are arbitrary or incomparable.

It can be inappropriate when variance magnitude is itself meaningful.

The random-matrix question comes after this modelling decision.

First decide whether covariance or correlation is the scientifically relevant object.

Then compare its spectrum to an appropriate high-dimensional null.

## Factor detectability is not the same as factor usefulness

A spike above the random-matrix threshold is statistically detectable in the asymptotic model.

That does not prove the factor is scientifically meaningful.

A batch effect can create a strong separated eigenvalue.

A scanner or instrument effect can dominate biological structure.

A time trend can create a large component.

A preprocessing artifact can create covariance.

A genuine latent mechanism can lie below the spectral detection threshold and remain scientifically important despite being difficult to recover from the available sample.

Random matrix theory separates one question:

> Is this spectral feature larger than the high-dimensional noise bulk expected under a null covariance model?

It does not answer:

> What generated the feature?

That second question requires design information, external variables, replication, mechanistic interpretation and validation.

A separated eigenvalue is evidence of covariance structure under the model.

It is not a label for that structure.

## High-dimensional asymptotics change the default PCA question

In low-dimensional PCA, a common workflow is:

1. compute the covariance matrix;
2. inspect eigenvalues;
3. retain large components;
4. interpret loadings.

When

$$
p/n
$$

is not small, the first question should change.

Before interpreting the observed spectrum, ask what spectrum pure noise would produce at the same aspect ratio.

For identity covariance and

$$
c=0.5,
$$

the answer is already wide:

$$
[0.086,2.914].
$$

That range exists even though every true eigenvalue is exactly one.

A real population spike of 1.5 is not enough to escape it.

A spike of 3 is.

The sample outlier produced by that spike is not 3 but approximately 3.75.

Its eigenvector is informative but still imperfectly aligned with the true direction.

These are not edge cases.

They are the ordinary geometry of covariance estimation when dimension grows with sample size.

Random matrix theory therefore changes the interpretation of PCA from

> which sample eigenvalues are large?

to

> which sample eigenvalues are larger than high-dimensional noise can plausibly create, and how much bias remains after they separate?

That is a much better question.

## References

Baik, J., Ben Arous, G., & Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *The Annals of Probability*, 33(5), 1643–1697.

Bai, Z. D., & Silverstein, J. W. (2010). *Spectral Analysis of Large Dimensional Random Matrices* (2nd ed.). Springer.

Johnstone, I. M. (2001). On the distribution of the largest eigenvalue in principal components analysis. *The Annals of Statistics*, 29(2), 295–327.

Johnstone, I. M., & Paul, D. (2018). PCA in high dimensions: An orientation. *Proceedings of the IEEE*, 106(8), 1277–1292.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457–483.

Paul, D. (2007). Asymptotics of sample eigenstructure for a large dimensional spiked covariance model. *Statistica Sinica*, 17(4), 1617–1642.

Vershynin, R. (2018). *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press.
