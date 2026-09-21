---
permalink: '/mathematics/gaussian_processes_are_distributions_over_functions/'
title: 'Gaussian Processes Are Distributions Over Functions'
date: '2026-06-25'
categories:
- Mathematics
tags:
- Gaussian Processes
- Bayesian Statistics
- Kernel Methods
- Statistical Modeling
- Uncertainty Quantification
author_profile: false
classes: wide
seo_title: 'Gaussian Processes Are Distributions Over Functions'
seo_description: 'A Gaussian process is a prior over functions defined by a mean and covariance kernel. Kernel choice determines smoothness, extrapolation, uncertainty and which structures the model can represent.'
seo_type: article
excerpt: >-
  Gaussian processes are often presented as flexible regressors with uncertainty
  bands. Their real mathematical content is stronger: a kernel defines a prior
  over functions, and posterior inference is conditioning in an infinite-
  dimensional Gaussian model.
summary: >-
  This article develops Gaussian processes from finite-dimensional Gaussian
  consistency, derives the posterior mean and covariance by conditioning a joint
  Gaussian distribution, and interprets kernels as structural assumptions rather
  than interchangeable similarity measures. It compares squared-exponential and
  Matérn smoothness, shows why extrapolation depends strongly on the mean and
  covariance structure, explains marginal-likelihood hyperparameter estimation,
  identifies amplitude-lengthscale-noise confounding, and discusses nonstationary
  kernels, sparse approximations and the distinction between latent-function and
  predictive uncertainty.
keywords:
- Gaussian process
- covariance kernel
- Matérn kernel
- marginal likelihood
- Bayesian nonparametrics
- function-space prior
why_this_exists: >-
  Gaussian processes are frequently introduced through code libraries and kernel
  menus, which makes them look like another flexible regression algorithm.
  Their real value comes from treating an unknown function as a random object and
  encoding structural assumptions directly through covariance.
evidence: >-
  Gaussian conditioning identities, positive-semidefinite kernel theory,
  regularity properties of squared-exponential and Matérn processes, exact
  marginal likelihood, and standard Gaussian-process approximation theory.
methodology: >-
  Define a Gaussian process through all finite-dimensional marginals, derive the
  posterior for noisy observations, interpret several kernel families through
  their implied sample-path properties, and examine hyperparameter
  identifiability and extrapolation using the geometry of the covariance matrix.
reviewed_at: '2026-09-21'
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
Question: What does a Gaussian process actually assume about an unknown function?
Claim: A Gaussian process is a probability distribution over functions whose mean and covariance kernel encode structural assumptions about smoothness, scale, correlation and extrapolation. Posterior regression is Gaussian conditioning, not merely interpolation with error bars.
Counterclaim: Exact Gaussian-process models can be computationally expensive, sensitive to kernel misspecification and weakly identified in their hyperparameters; a flexible kernel does not eliminate the need for model criticism.
Evidence object: Finite-dimensional GP definition, exact posterior derivation, smoothness contrast between squared-exponential and Matérn kernels, marginal-likelihood decomposition, and amplitude-lengthscale-noise confounding.
Failure case: Choosing a kernel from a software menu without a generative interpretation, treating posterior credible bands as model-robust uncertainty, or interpreting a well-optimized marginal likelihood as proof that hyperparameters are uniquely identified.
Reader payoff: Understand Gaussian processes as function-space statistical models and choose kernels, means and approximations based on the structure of the scientific problem.
Exclusions: A Python tutorial, a catalogue of every covariance family, and repetition of the existing time-series implementation post.
-->

Gaussian processes are often introduced as flexible regression models that draw smooth curves through data and attach uncertainty bands around the prediction. That description is operationally correct and mathematically incomplete. A Gaussian process is not fundamentally a curve-fitting algorithm. It is a probability distribution on functions.

The distinction matters because it changes how the model should be designed. In ordinary parametric regression one specifies a finite vector of coefficients and places a probability model on the observations conditional on those coefficients. In a Gaussian process, the unknown function itself is random. The prior says which functions are plausible before observing data, and the covariance kernel encodes how values of that function are expected to co-vary across the input space.

Once observations arrive, posterior inference is obtained by conditioning a joint Gaussian distribution. The familiar posterior mean and variance formulas are therefore consequences of multivariate Gaussian algebra. The apparent flexibility of Gaussian-process regression comes from moving the modelling assumptions into the covariance operator rather than eliminating them.

This is why the kernel matters so much. A squared-exponential kernel does not merely say that nearby points are similar. It says much more: sample paths are extraordinarily smooth, correlations decay according to a Gaussian function of distance, and extrapolation reverts toward the mean according to that structure. A Matérn kernel makes a different statement about differentiability. A periodic kernel asserts repetition. An additive kernel asserts decomposability. A nonstationary kernel says correlation structure changes with location.

The model is flexible only inside the function class its covariance makes plausible.

## A Gaussian process is defined through every finite collection of inputs

Let

$$
f:\mathcal X\to\mathbb R
$$

be an unknown function. We say

$$
f
\sim
\mathcal{GP}(m,k)
$$

when, for every finite collection of inputs

$$
x_1,\ldots,x_n,
$$

the vector

$$
\mathbf f
=
\begin{pmatrix}
f(x_1)\\
\vdots\\
f(x_n)
\end{pmatrix}
$$

has a multivariate normal distribution,

$$
\mathbf f
\sim
N(\mathbf m,K),
$$

where

$$
m_i
=
m(x_i)
$$

and

$$
K_{ij}
=
k(x_i,x_j).
$$

The mean function is

$$
m(x)
=
E[f(x)],
$$

and the covariance kernel is

$$
k(x,x')
=
\operatorname{Cov}
\left(
f(x),f(x')
\right).
$$

The crucial consistency requirement is that these finite-dimensional Gaussian distributions agree when points are added, removed or reordered. A valid covariance kernel guarantees that consistency through positive semidefiniteness.

For any finite coefficients

$$
a_1,\ldots,a_n,
$$

a valid kernel must satisfy

$$
\sum_{i=1}^n
\sum_{j=1}^n
a_i a_j
k(x_i,x_j)
\ge
0.
$$

Equivalently, every kernel matrix

$$
K
$$

must be positive semidefinite.

This is not a technical nuisance imposed by software. It is what makes

$$
K
$$

a covariance matrix for every possible finite set of inputs.

The process is infinite-dimensional in the sense that no fixed finite parameter vector describes the unknown function values everywhere. Yet every finite collection is governed by ordinary multivariate Gaussian probability. This is why Gaussian processes sit naturally between functional analysis and statistics: the model is defined over a function space, while inference at observed and prediction points reduces to linear algebra.

## Posterior regression is Gaussian conditioning

Suppose observations satisfy

$$
y_i
=
f(x_i)
+
\varepsilon_i,
$$

with independent Gaussian noise

$$
\varepsilon_i
\sim
N(0,\sigma_n^2).
$$

Let the training inputs be

$$
X
=
(x_1,\ldots,x_n),
$$

and prediction inputs be

$$
X_\star
=
(x_1^\star,\ldots,x_m^\star).
$$

For simplicity, first take a zero mean function. Define

$$
K
=
K(X,X),
$$

$$
K_\star
=
K(X,X_\star),
$$

and

$$
K_{\star\star}
=
K(X_\star,X_\star).
$$

Because observations include Gaussian noise,

$$
\mathbf y
\sim
N
\left(
0,
K+\sigma_n^2I
\right).
$$

The joint distribution of observed values and latent function values at the prediction points is

$$
\begin{pmatrix}
\mathbf y\\
\mathbf f_\star
\end{pmatrix}
\sim
N
\left[
\begin{pmatrix}
0\\
0
\end{pmatrix},
\begin{pmatrix}
K+\sigma_n^2I & K_\star\\
K_\star^\top & K_{\star\star}
\end{pmatrix}
\right].
$$

Conditioning one block of a multivariate Gaussian on the other gives

$$
\mathbf f_\star
\mid
\mathbf y
\sim
N
\left(
\mu_\star,
\Sigma_\star
\right),
$$

with posterior mean

$$
\mu_\star
=
K_\star^\top
\left(
K+\sigma_n^2I
\right)^{-1}
\mathbf y,
$$

and posterior covariance

$$
\Sigma_\star
=
K_{\star\star}
-
K_\star^\top
\left(
K+\sigma_n^2I
\right)^{-1}
K_\star.
$$

These are the standard Gaussian-process regression formulas.

They should be read structurally.

The posterior mean is a linear combination of observed outcomes,

$$
\mu_\star
=
\sum_{i=1}^n
w_i(x_\star)y_i,
$$

where the weights depend entirely on the covariance geometry. Observations judged strongly correlated with the prediction point receive larger influence. Noise reduces their leverage through

$$
K+\sigma_n^2I.
$$

The posterior covariance is prior uncertainty minus the amount removed by observing correlated data.

If a prediction point is far from all observations under the chosen kernel, then

$$
K_\star
\approx0,
$$

and the posterior approaches the prior:

$$
\mu_\star
\to0,
$$

$$
\Sigma_\star
\to
K_{\star\star}.
$$

With a nonzero mean function, the posterior returns toward that mean instead.

This is what Gaussian-process extrapolation means. Outside the region linked strongly to the data by the kernel, the model does not "continue the trend" unless the prior mean or covariance structure says it should.

## The mean function controls extrapolation more than many implementations admit

Setting

$$
m(x)=0
$$

is convenient because the algebra becomes cleaner. It is not a neutral choice.

Suppose observations show a clear linear trend. A zero-mean stationary kernel can fit the observed region by bending a smooth function through the data. Far from the data, covariance with the observed points decreases and the posterior mean eventually returns toward zero.

If zero is not a plausible long-range level, the extrapolation is structurally wrong no matter how well the local fit looks.

A more appropriate model might use

$$
m(x)
=
\beta_0+\beta_1x
$$

and place the Gaussian process on residual structure,

$$
f(x)
=
m(x)
+
g(x),
$$

where

$$
g
\sim
\mathcal{GP}(0,k).
$$

Now the long-range behaviour reverts toward the parametric trend rather than toward zero.

This separation is especially useful when some extrapolative structure is known mechanistically. A physical scaling law, seasonal baseline, dose-response form or conservation relation can enter the mean, while the GP represents deviations around it.

The same idea appears in universal kriging and semiparametric regression. Gaussian processes do not force a choice between rigid parametric structure and nonparametric flexibility. They can combine them.

The important point is that extrapolation comes from assumptions. A kernel fit to local data cannot infer arbitrary behaviour beyond the support of the inputs.

## The squared-exponential kernel assumes extreme smoothness

The squared-exponential kernel is

$$
k_{\mathrm{SE}}(x,x')
=
\sigma_f^2
\exp
\left[
-
\frac{
(x-x')^2
}{
2\ell^2
}
\right].
$$

The amplitude

$$
\sigma_f^2
$$

sets the marginal prior variance, while the lengthscale

$$
\ell
$$

controls how rapidly correlation decays with distance.

If

$$
|x-x'|
\ll
\ell,
$$

the function values are strongly correlated.

If

$$
|x-x'|
\gg
\ell,
$$

their covariance is negligible.

What matters more is the implied regularity. Sample paths from a squared-exponential process are almost surely infinitely differentiable. The kernel's spectral density decays faster than any polynomial, strongly suppressing high-frequency variation.

This is an extremely smooth prior.

In many applications that is exactly what is wanted. Spatial physical fields, slowly varying latent functions and smooth calibration curves can justify it.

In other settings it is too strong. A biological process can be continuous but rough. A material property can have local irregularity. A temporal process can be nondifferentiable even though it remains strongly correlated.

Using the squared-exponential kernel because it is the software default can therefore produce posterior functions that look more regular than the underlying phenomenon.

A narrow credible band around an oversmoothed posterior does not fix that misspecification.

## Matérn kernels separate scale from differentiability

The Matérn family adds a smoothness parameter

$$
\nu>0.
$$

In one common parameterization,

$$
k_\nu(r)
=
\sigma_f^2
\frac{
2^{1-\nu}
}{
\Gamma(\nu)
}
\left(
\frac{
\sqrt{2\nu}r
}{
\ell
}
\right)^\nu
K_\nu
\left(
\frac{
\sqrt{2\nu}r
}{
\ell
}
\right),
$$

where

$$
r=|x-x'|
$$

and

$$
K_\nu
$$

is the modified Bessel function of the second kind.

The parameter

$$
\ell
$$

still controls correlation range, while

$$
\nu
$$

controls smoothness.

For

$$
\nu=\frac12,
$$

the kernel becomes exponential,

$$
k(r)
=
\sigma_f^2
\exp
\left(
-\frac r\ell
\right).
$$

Sample paths are continuous but not mean-square differentiable.

For

$$
\nu=\frac32,
$$

the kernel is

$$
k(r)
=
\sigma_f^2
\left(
1+
\frac{
\sqrt3 r
}{
\ell
}
\right)
\exp
\left(
-
\frac{
\sqrt3 r
}{
\ell
}
\right).
$$

For

$$
\nu=\frac52,
$$

the kernel becomes

$$
k(r)
=
\sigma_f^2
\left(
1+
\frac{
\sqrt5 r
}{
\ell
}
+
\frac{
5r^2
}{
3\ell^2
}
\right)
\exp
\left(
-
\frac{
\sqrt5 r
}{
\ell
}
\right).
$$

As

$$
\nu\to\infty,
$$

the Matérn family approaches the squared-exponential kernel.

The smoothness parameter therefore separates two ideas that are easy to conflate: how far dependence extends and how rough the function is locally.

This matters because two kernels can have similar correlation at moderate distances and very different high-frequency behaviour. A model that estimates only one lengthscale but assumes infinite differentiability has silently fixed an important structural property.

In scientific applications, choosing

$$
\nu
$$

from known regularity can be more defensible than optimizing it solely through marginal likelihood.

## Kernel sums and products encode function decomposition

Covariance kernels can be combined while preserving positive semidefiniteness.

If

$$
k_1
$$

and

$$
k_2
$$

are valid kernels, then

$$
k_1+k_2
$$

and

$$
k_1k_2
$$

are also valid under standard constructions.

A sum corresponds naturally to additive latent functions. If

$$
f_1
\sim
\mathcal{GP}(0,k_1)
$$

and

$$
f_2
\sim
\mathcal{GP}(0,k_2)
$$

independently, then

$$
f=f_1+f_2
$$

is a GP with kernel

$$
k=k_1+k_2.
$$

This gives a probabilistic interpretation to additive kernel design.

A time-series covariance might be written as

$$
k
=
k_{\text{trend}}
+
k_{\text{seasonal}}
+
k_{\text{short}}
$$

to represent long-range smooth variation, periodic structure and short-range residual correlation.

Products represent interaction. A periodic kernel multiplied by a slowly varying kernel can produce locally periodic behaviour whose pattern gradually changes over time.

The algebra therefore encodes assumptions about decomposition and interaction.

Kernel engineering is most defensible when each term corresponds to a meaningful structural component rather than an arbitrary search through combinations.

## Hyperparameters are estimated through the marginal likelihood

Let

$$
\theta
$$

denote kernel and noise hyperparameters. Under the Gaussian model,

$$
\mathbf y
\sim
N
\left(
\mathbf m,
K_\theta+\sigma_n^2I
\right).
$$

The log marginal likelihood is

$$
\log p(\mathbf y\mid\theta)
=
-
\frac12
(\mathbf y-\mathbf m)^\top
K_y^{-1}
(\mathbf y-\mathbf m)
-
\frac12
\log|K_y|
-
\frac n2
\log(2\pi),
$$

where

$$
K_y
=
K_\theta+\sigma_n^2I.
$$

The first term rewards data fit.

The second penalizes covariance structures that allocate large volume to functions not needed to explain the observed data.

This is sometimes described as an automatic Occam penalty.

The interpretation is useful if not overstated. The marginal likelihood compares models within the chosen kernel family and prior parameterization. It does not guarantee that the family is scientifically correct, and its optimizer need not be sharply identified.

Multiple hyperparameter combinations can yield similar covariance matrices over the observed input configuration.

This is particularly common with sparse or narrowly spaced designs.

## Amplitude, lengthscale and noise can be weakly identified

Suppose observations cover only a short interval relative to the true correlation length.

A long lengthscale with moderate amplitude can produce nearly constant latent functions over the observed domain.

A somewhat shorter lengthscale with smaller amplitude can generate similar covariance among the observed points.

Additional white noise can absorb local variation that would otherwise be attributed to a shorter lengthscale.

The likelihood can therefore contain ridges or broad valleys in

$$
(\sigma_f,\ell,\sigma_n)
$$

space.

One symptom is optimization sensitivity to initialization. Another is a flat profile likelihood or posterior distribution for one hyperparameter after conditioning on the others.

The problem is structural rather than numerical when the design does not contain enough interpoint distances to distinguish the scales.

Consider the covariance between two points separated by distance

$$
r.
$$

Under the squared-exponential kernel,

$$
\frac{
k(r)
}{
k(0)
}
=
\exp
\left(
-
\frac{
r^2
}{
2\ell^2
}
\right).
$$

If all observed distances satisfy

$$
r\ll\ell,
$$

then

$$
\exp
\left(
-
\frac{
r^2
}{
2\ell^2
}
\right)
\approx
1
-
\frac{
r^2
}{
2\ell^2
}.
$$

Many large values of

$$
\ell
$$

produce nearly the same correlations.

The data can tell us that the process is smooth over the observed range without identifying the exact long-range correlation scale.

A single optimized lengthscale should therefore not automatically be interpreted as a physical distance parameter.

## Hyperparameter uncertainty should propagate into prediction

The standard GP regression formula often plugs in a point estimate

$$
\hat\theta
$$

of the kernel hyperparameters and then conditions on it as though known.

This understates predictive uncertainty when the hyperparameters are weakly identified.

A fully Bayesian treatment places a prior

$$
p(\theta)
$$

and integrates,

$$
p(f_\star\mid y)
=
\int
p(f_\star\mid y,\theta)
p(\theta\mid y)
\,d\theta.
$$

The resulting predictive distribution is generally no longer exactly Gaussian because it is a mixture over hyperparameters.

In many well-informed datasets, plug-in uncertainty is a reasonable approximation. In sparse extrapolation problems, the difference can be substantial.

Lengthscale uncertainty is especially important because it controls how quickly posterior predictions revert toward the prior away from observations. Two plausible lengthscales can agree inside the data range and diverge rapidly outside it.

This is another reason uncertainty bands should not be interpreted as model-robust confidence.

They are conditional on the kernel family, mean function, likelihood and treatment of hyperparameters.

## Latent-function uncertainty is not predictive observation uncertainty

The posterior covariance

$$
\Sigma_\star
$$

derived earlier describes uncertainty about the latent function values

$$
f_\star.
$$

A future observation satisfies

$$
y_\star
=
f_\star+\varepsilon_\star.
$$

Therefore,

$$
\operatorname{Var}
(
y_\star\mid y
)
=
\Sigma_\star
+
\sigma_n^2I.
$$

The predictive interval for future noisy observations is wider than the credible interval for the latent function.

This distinction is easy to miss in plotting libraries because both are often shown as shaded bands.

If the scientific target is the underlying smooth signal, the latent interval is relevant.

If the target is what the next measurement will be, observation noise must be added.

In heteroskedastic problems,

$$
\sigma_n^2
$$

may depend on input location. In count, binary or survival settings, the observation model is non-Gaussian and exact conditioning is lost.

Then Gaussian-process latent functions can still be used, but posterior inference requires approximations such as Laplace methods, expectation propagation or variational inference.

The GP prior over functions survives. The convenience of Gaussian conjugacy does not.

## Nonstationary processes require nonstationary covariance

A stationary kernel depends on displacement rather than absolute location,

$$
k(x,x')
=
k(x-x').
$$

If the process is also isotropic in multiple dimensions, dependence depends only on distance,

$$
k(x,x')
=
k(\|x-x'\|).
$$

These assumptions are powerful and often unrealistic.

A spatial field can be smooth in one region and rough in another. A time series can change correlation length after an intervention. A physiological signal can have state-dependent variability. A material surface can contain boundaries across which correlation changes sharply.

One route is input warping: transform the input space so that stationary covariance in the warped coordinates corresponds to nonstationary covariance in the original coordinates.

Another is to let kernel parameters vary with location,

$$
\ell=\ell(x),
$$

or build covariance from latent processes.

Change-point kernels can combine two covariance regimes with a smooth transition.

Deep Gaussian processes compose latent GPs, creating highly nonstationary mappings at the cost of more difficult inference.

The need for nonstationarity should be driven by residual structure and domain knowledge rather than by complexity for its own sake.

A stationary Matérn model can be far more interpretable and stable when its assumptions are approximately correct.

## Gaussian processes interpolate only in the noiseless limit

A common description says GPs interpolate the observations.

With observation noise

$$
\sigma_n^2>0,
$$

the posterior mean does not generally pass exactly through every observed point.

At the training inputs,

$$
\mu
=
K
\left(
K+\sigma_n^2I
\right)^{-1}
y.
$$

If

$$
\sigma_n^2
\to0
$$

and $K$ is nonsingular, then

$$
\mu\to y.
$$

The model interpolates in the noiseless limit.

With positive noise, the posterior smooths because the model treats deviations from the latent function as measurement variation.

This gives a Bayesian interpretation of regularization. The kernel controls which functions are plausible, while noise controls how strongly observations must be followed.

There is a close connection with kernel ridge regression. For Gaussian likelihood and fixed kernel hyperparameters, the GP posterior mean has the same algebraic form as a kernel ridge predictor for an appropriate regularization parameter.

The interpretations differ.

Kernel ridge regression is typically framed as optimization in a reproducing-kernel Hilbert space.

Gaussian-process regression is framed probabilistically and supplies a posterior covariance in addition to the mean.

The same matrix formula can therefore arise from different conceptual routes.

## The GP and its RKHS are related but not identical

Every positive-definite kernel defines a reproducing-kernel Hilbert space.

It is tempting to say that a Gaussian-process sample path lives in that RKHS.

For many common kernels, this is false with probability one.

The RKHS associated with the kernel is typically a set of particularly regular functions relative to the GP sample paths. It plays the role of a Cameron-Martin space: shifts by RKHS functions interact nicely with the Gaussian measure, but a typical random draw need not belong to the RKHS itself.

This distinction matters conceptually when connecting Bayesian GP inference with penalized optimization.

The posterior mean can lie in the span of kernel sections

$$
k(x_i,\cdot),
$$

while random functions drawn from the GP prior have different regularity properties.

The connection between kernels, RKHS norms and Gaussian measures is deep and useful, but identifying all three as the same object hides important functional-analytic structure.

## Exact inference scales poorly because covariance matrices are dense

For $n$ observations, exact Gaussian-process regression requires factorizing

$$
K+\sigma_n^2I.
$$

A dense Cholesky factorization costs approximately

$$
O(n^3)
$$

operations and

$$
O(n^2)
$$

memory.

This is manageable for thousands of observations and problematic for hundreds of thousands or millions.

The computational limitation has generated several approximation families.

Inducing-point methods introduce

$$
m\ll n
$$

representative latent variables and approximate the covariance structure through them, often reducing cost toward

$$
O(nm^2).
$$

Structured kernel interpolation exploits grid or algebraic structure.

Random Fourier features approximate stationary kernels through finite random basis expansions.

State-space representations convert certain one-dimensional kernels, especially Matérn families, into linear stochastic differential equations and permit Kalman-filter inference with cost linear in time length under appropriate conditions.

Nearest-neighbour Gaussian processes and sparse precision formulations exploit conditional independence approximations.

These are not merely engineering tricks. Every approximation changes the prior or posterior representation.

A sparse GP should therefore be evaluated statistically as well as computationally.

## Experimental design can be expressed directly through posterior variance

Because the GP provides an explicit posterior covariance, it naturally supports design decisions.

Suppose one can choose the next input

$$
x_{\mathrm{new}}.
$$

A simple criterion is maximum posterior variance,

$$
x_{\mathrm{new}}
=
\arg\max_x
\operatorname{Var}
[
f(x)\mid y
].
$$

This chooses the point where the current function estimate is most uncertain.

Other criteria target integrated variance reduction, expected improvement, entropy, level-set uncertainty or a downstream decision objective.

Bayesian optimization uses a GP surrogate together with an acquisition function to decide where an expensive objective should be evaluated next.

The distinction between uncertainty and lack of data becomes operational here. Posterior variance is not merely an error bar to plot after fitting. It can determine where the next measurement is most informative under the assumed kernel.

If the kernel is misspecified, that design can be poor. A squared-exponential model that believes the function is globally smooth may see little value in sampling near a sharp boundary it cannot represent.

Experimental design is therefore another place where covariance assumptions have real consequences.

## Gaussian processes are structured priors, not automatic uncertainty machines

A Gaussian process gives uncertainty because the model is probabilistic. That uncertainty is only as credible as the model.

If the kernel is too smooth, credible bands can be narrow around an oversmoothed function.

If the mean function is wrong, extrapolation can revert toward an implausible baseline.

If the noise model is wrong, uncertainty can be allocated incorrectly between latent variation and measurement error.

If hyperparameters are weakly identified, plug-in bands can understate uncertainty.

If the process is nonstationary and the covariance is stationary, the model can appear well calibrated in dense regions and fail in sparse ones.

The strength of Gaussian processes is not that they eliminate assumptions. It is that many important assumptions are explicit in mathematically interpretable objects.

The mean function says what the model expects before local evidence.

The covariance kernel says how function values relate across inputs.

The likelihood says how observations deviate from the latent process.

Hyperpriors say which covariance scales are plausible.

Once these are specified, conditioning is exact in the Gaussian case.

That is a remarkably coherent modelling framework.

It is also why kernel choice should never be treated as a cosmetic dropdown.

A Gaussian process is a distribution over functions.

The kernel defines the geometry of that distribution.

## References

Adler, R. J., & Taylor, J. E. (2007). *Random Fields and Geometry*. Springer.

Berlinet, A., & Thomas-Agnan, C. (2004). *Reproducing Kernel Hilbert Spaces in Probability and Statistics*. Kluwer.

Genton, M. G. (2001). Classes of kernels for machine learning: a statistics perspective. *Journal of Machine Learning Research*, 2, 299–312.

Gramacy, R. B. (2020). *Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences*. CRC Press.

Matérn, B. (1986). *Spatial Variation* (2nd ed.). Springer.

Quinonero-Candela, J., & Rasmussen, C. E. (2005). A unifying view of sparse approximate Gaussian process regression. *Journal of Machine Learning Research*, 6, 1939–1959.

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer.

Williams, C. K. I., & Rasmussen, C. E. (1996). Gaussian processes for regression. In *Advances in Neural Information Processing Systems 8*.
