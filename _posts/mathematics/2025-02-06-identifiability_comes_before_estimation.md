---
permalink: '/mathematics/identifiability_comes_before_estimation/'
title: 'Identifiability Comes Before Estimation'
date: '2025-02-06'
categories:
- Mathematics
tags:
- Inverse Problems
- Identifiability
- Parameter Estimation
- Ill-Posed Problems
- Regularization
author_profile: false
classes: wide
seo_title: 'Identifiability Comes Before Estimation'
seo_description: 'A model can fit observations almost perfectly while its parameters remain non-identifiable. Inverse problems require uniqueness and stability before numerical estimation can be interpreted.'
seo_type: article
excerpt: >-
  Parameter estimation is an inverse problem. Before asking an optimiser for a
  numerical answer, one must ask whether the observations determine that answer
  uniquely and whether small perturbations in the data can change it drastically.
summary: >-
  This article develops identifiability from the forward map of an inverse
  problem. It distinguishes structural identifiability, practical
  identifiability and Hadamard stability; gives exact examples of non-unique and
  ill-conditioned inverse maps; connects Jacobian singular values to Fisher
  information and likelihood geometry; and explains why regularisation and
  Bayesian priors stabilise an inverse problem without creating information that
  the data did not contain.
keywords:
- identifiability
- inverse problems
- structural identifiability
- practical identifiability
- ill posed inverse problems
- regularization
why_this_exists: >-
  Applied modelling often moves directly from a forward model to parameter
  optimisation. That reverses the logical order. If the parameter-to-observation
  map is non-injective, no optimiser can recover a unique parameter vector from
  the data; if the inverse map is unstable, arbitrarily small observational
  perturbations can produce large parameter changes.
evidence: >-
  Classical inverse-problem theory, structural and practical identifiability
  literature, profile-likelihood methods, information-matrix geometry and
  Bayesian inverse-problem formulations.
methodology: >-
  Begin from a general forward operator and define structural identifiability as
  injectivity. Use an exact multiplicative-parameter example to exhibit a
  non-identifiable manifold, then a two-by-two linear inverse problem to
  distinguish uniqueness from stability. Relate local sensitivity to singular
  values of the Jacobian and the Fisher information matrix, and analyse how
  regularisation, priors and experimental design alter the inverse problem.
reviewed_at: '2026-09-20'
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
Question: When is parameter estimation mathematically meaningful?
Claim: Estimation is meaningful only after the observation map is shown to identify the target parameter, at least to the degree required by the scientific question, and after the inverse map is sufficiently stable for the available data quality.
Counterclaim: Prediction can remain well determined even when individual parameters are not identifiable, and regularised or Bayesian solutions can be useful if their dependence on penalties or priors is made explicit.
Evidence object: A multiplicative parameter model with an exact non-identifiable ridge, a nearly singular 2x2 linear inverse problem, singular-value geometry, Fisher information, profile likelihood and experiment-design arguments.
Failure case: Treating successful optimisation, narrow numerical tolerances, a unique optimiser output, or a proper Bayesian posterior as proof that the data uniquely determine the parameters.
Reader payoff: Separate fit from identification, structural non-identifiability from practical uncertainty, and data information from assumptions introduced through regularisation or priors.
Exclusions: Treating all inverse problems as linear, claiming regularisation is improper, or assuming parameter non-identifiability implies useless predictions.
-->

Parameter estimation is often presented as an optimisation problem. A model is written down, observations are collected, a loss function is defined, and an algorithm searches for the parameter vector that minimises the discrepancy between model and data. That workflow is computationally natural and logically incomplete. Before asking how to estimate a parameter, one must ask whether the observations determine it at all.

The distinction is fundamental in inverse problems. A forward problem begins with parameters or latent quantities and computes observable consequences. An inverse problem starts from those consequences and tries to reconstruct what produced them. If the forward map is not one-to-one, several parameter values produce exactly the same observations. If the map is technically one-to-one but nearly singular, tiny perturbations in the observations can produce very large changes in the reconstructed parameters. In either case, an optimiser can return a number. The existence of a numerical answer does not imply that the answer was identified by the data.

Let the parameter vector be $\theta\in\Theta$ and let the model map parameters into an observation space through

$$
y=F(\theta)+\varepsilon,
$$

where $F$ is the forward operator and $\varepsilon$ represents measurement error, model discrepancy or both. Parameter estimation attempts to invert $F$ from observed $y$. The mathematical questions begin before optimisation: is $F$ injective on the relevant parameter space, and if it is, how sensitive is $F^{-1}$ to perturbations in $y$?

## Structural identifiability is a uniqueness question

In its simplest form, structural identifiability asks whether perfect, noise-free observations generated by the model determine the parameter uniquely. A parameterisation is globally identifiable if

$$
F(\theta_1)=F(\theta_2)
\quad\Longrightarrow\quad
\theta_1=\theta_2
$$

for all admissible parameter values, apart from exceptional sets allowed by the precise definition. If different parameter vectors produce exactly the same model output, the inverse problem is structurally non-identifiable. More data of the same observational type cannot repair the problem because the ambiguity is built into the model map itself. citeturn366452view0turn216997search2

A trivial example makes the issue visible. Suppose the observed trajectory is

$$
y(t)=ab\,e^{-kt},
$$

with unknown positive parameters $a$, $b$ and $k$. Even if $y(t)$ were observed continuously and without noise for all $t\ge0$, the data can identify $k$ and the product

$$
c=ab,
$$

but they cannot separate $a$ from $b$. For every $\lambda>0$,

$$
(a,b,k)
\mapsto
(\lambda a,b/\lambda,k)
$$

leaves the complete trajectory unchanged:

$$
(\lambda a)(b/\lambda)e^{-kt}
=
ab e^{-kt}.
$$

The data therefore identify a one-dimensional manifold of equivalent parameter vectors rather than one point. This is not a numerical failure. No optimiser, additional decimal precision, larger computer or alternative loss function can recover information that is absent from the observation map.

The likelihood geometry reflects the same fact. If the noise model depends on the predicted trajectory only through $ab$ and $k$, then the likelihood is constant along curves satisfying

$$
ab=c.
$$

An optimisation routine may choose one point on that ridge because of initialisation, parameter bounds, numerical tolerances or implicit regularisation. Repeating the optimisation from several starting points can reveal the ridge, but convergence to the same point does not prove uniqueness. An optimiser is a procedure for selecting a numerical solution. Identifiability is a property of the model and observation scheme.

The distinction between global and local identifiability becomes important in nonlinear models. A model may have finitely many parameter values that reproduce the same observations, in which case parameters can be locally identifiable while not globally identifiable. Symmetries are a common source. If an output depends on $\theta^2$, then $\theta$ and $-\theta$ are observationally equivalent unless the parameter space or additional information breaks the symmetry. Local curvature around either solution can look perfectly regular while the global inverse remains ambiguous.

Structural identifiability is especially important in dynamical systems because model complexity can hide non-identifiable combinations. Differential-equation models may contain parameters that appear separately in the equations but only enter the measured output through certain combinations. Structural-identifiability methods analyse this algebraically or through observability-related rank conditions. The point is not specific to systems biology, where much of the applied literature developed. The same issue appears in compartment models, kinetic models, geophysical inversion, inverse PDEs, control, econometrics and any setting where latent mechanisms are reconstructed from indirect measurements. citeturn366452view0turn216997search5

## Uniqueness is not stability

Identifiability addresses whether a solution is unique. Hadamard's classical notion of well-posedness asks for more: a solution should exist, be unique and depend continuously on the data. An inverse problem can therefore be identifiable and still be unusable because the inverse map is extremely unstable. Inverse problems are frequently difficult precisely because observational noise is amplified in directions where the forward map changes very little. citeturn216997search7turn569338search3

A two-dimensional linear example separates uniqueness from stability cleanly. Let

$$
A_\delta
=
\begin{pmatrix}
1 & 1\\
1 & 1+\delta
\end{pmatrix},
\qquad
y=A_\delta\theta,
$$

with $\delta\ne0$. Since

$$
\det(A_\delta)=\delta,
$$

the matrix is invertible for every nonzero $\delta$. The parameter vector is therefore structurally identifiable in the exact linear problem. Yet as $\delta\to0$, the two rows of $A_\delta$ become nearly identical. The inverse becomes increasingly sensitive to noise.

The inverse is

$$
A_\delta^{-1}
=
\frac{1}{\delta}
\begin{pmatrix}
1+\delta & -1\\
-1 & 1
\end{pmatrix}.
$$

The factor $1/\delta$ makes the instability explicit. An observational perturbation of order $\varepsilon$ can induce a parameter perturbation of order

$$
\frac{\varepsilon}{|\delta|}.
$$

If $\delta=10^{-3}$, measurement error at the fourth decimal place can easily become parameter error at the first decimal place. The inverse exists and is unique, but it is badly conditioned.

This is the simplest algebraic version of practical non-identifiability. In nonlinear models, the same geometry appears through directions in parameter space along which predictions change only weakly relative to the observational noise. Structural identifiability is a theoretical property of the ideal model and observation scheme. Practical identifiability concerns whether the available amount, location and precision of data constrain parameters sufficiently for the intended inference. citeturn216997search3turn569338search0

## The Jacobian reveals weak directions

For a nonlinear forward map $F$, local behaviour around parameter $\theta_0$ can be approximated by

$$
F(\theta_0+\Delta\theta)
\approx
F(\theta_0)
+
J(\theta_0)\Delta\theta,
$$

where

$$
J(\theta_0)
=
\frac{\partial F}{\partial\theta}
$$

is the sensitivity matrix. The singular values of $J$ describe how strongly different parameter directions change the observations. If $J$ has an exact null direction $v$,

$$
Jv=0,
$$

then infinitesimal movement along $v$ does not change the output to first order. If the rank deficiency reflects an exact model symmetry, this is a local signature of structural non-identifiability. If the smallest singular value is positive but extremely small, the corresponding direction is weakly informed and noise can be strongly amplified.

For Gaussian measurement error with covariance $\Sigma$, the local Fisher information matrix takes the familiar form

$$
\mathcal I(\theta)
=
J(\theta)^\top
\Sigma^{-1}
J(\theta).
$$

If $\mathcal I$ is singular, some local parameter combination carries zero information under the model. If it is badly conditioned, uncertainty becomes highly anisotropic. The confidence region is not a compact, roughly spherical neighbourhood around the optimum but a long narrow valley. Large movement along one parameter combination may produce almost no change in likelihood, while tiny movement in another direction is strongly penalised.

This geometry explains why standard errors derived from an inverse Hessian can become enormous or numerically unstable near non-identifiable directions. It also explains why reporting only the fitted parameter vector can be misleading. The numerical optimum contains little information about how sharply the data localise that optimum.

Profile likelihood provides one way to expose this structure in nonlinear problems. For parameter $\theta_j$, fix $\theta_j$ at a sequence of values and re-optimise the remaining parameters. The profile

$$
\ell_p(\theta_j)
=
\max_{\theta_{-j}}
\ell(\theta_j,\theta_{-j})
$$

shows how much fit deteriorates when that parameter is forced away from its optimum while the rest of the model compensates. A flat profile indicates that the data permit a wide range of values. An unbounded profile can indicate practical or structural non-identifiability. Raue and colleagues developed this approach precisely to distinguish these cases in partially observed dynamical systems. citeturn569338search0

## Good fit and identifiable parameters are different achievements

An important consequence is that predictive fit and parameter identification need not move together. A model can predict the observed output extremely well while its internal parameters remain uncertain or non-identifiable. In the multiplicative example, every point satisfying $ab=c$ produces exactly the same trajectory, so prediction of that trajectory can be perfect even though neither $a$ nor $b$ is individually recoverable.

This is not a contradiction. The data may identify a lower-dimensional function of the parameters even when they do not identify the full parameter vector. If the scientific target is the identifiable combination

$$
\psi(\theta)=ab,
$$

there may be no practical problem. If the scientific claim concerns the separate physical interpretation of $a$ and $b$, the same fit is insufficient.

The same distinction occurs in larger mechanistic models. Several rate constants may trade off while preserving a measured concentration trajectory. A state prediction may remain stable across the entire likelihood ridge. A control policy may depend only on an identifiable combination. Conversely, an apparently minor non-identifiable parameter may be central to a mechanistic interpretation even when forecasts are unaffected.

This is why identifiability should be defined relative to the scientific target. "The model is identifiable" is often too coarse. Which parameters, combinations, latent states or predictions are identifiable from which observations and inputs? A model can contain identifiable and non-identifiable components simultaneously.

## Regularisation stabilises a choice, not the data

Ill-posed inverse problems are often solved with regularisation. In a linear problem

$$
y=A\theta+\varepsilon,
$$

Tikhonov regularisation replaces least squares with

$$
\hat\theta_\lambda
=
\arg\min_\theta
\left\{
\|A\theta-y\|_2^2
+
\lambda\|L\theta\|_2^2
\right\}.
$$

When $L=I$, the solution is

$$
\hat\theta_\lambda
=
(A^\top A+\lambda I)^{-1}A^\top y.
$$

The term $\lambda I$ improves numerical conditioning because singular directions of $A^\top A$ no longer need to be inverted at arbitrarily small eigenvalues. This is valuable and often essential. It also changes the problem. The regularised estimate balances agreement with the observations against a preference encoded by the penalty.

If $A$ has a true null space, the data alone cannot distinguish parameter vectors differing along that null space. The regulariser selects among them by preferring, for example, smaller norm, greater smoothness or some other structural property encoded by $L$. The resulting solution can be useful, physically sensible and statistically optimal under a chosen criterion. It should not be described as though the missing information had suddenly appeared in the measurements.

The distinction matters in scientific interpretation. If the recovered field, coefficient or parameter changes substantially when the regularisation strength or penalty structure changes, then part of the reconstruction is assumption-driven. Sensitivity to regularisation is therefore evidence about the inverse problem, not merely a tuning inconvenience.

Inverse PDE problems make this especially clear. Suppose a field $u$ satisfies

$$
-\nabla\cdot(\kappa(x)\nabla u(x))=f(x),
$$

and observations of $u$ are used to infer the spatial coefficient $\kappa(x)$. Depending on which parts of $u$ are observed, boundary conditions, forcing and admissible function class, many coefficient fields may produce nearly indistinguishable observations. Smoothness penalties can produce a unique stable reconstruction, but the smoothness is additional information supplied through the inverse method.

## Bayesian inference changes the object but not the information content

Bayesian inverse problems replace the search for one parameter value with inference about a posterior distribution,

$$
\pi(\theta\mid y)
\propto
\pi(y\mid\theta)\pi(\theta).
$$

This is an elegant way to incorporate prior information, quantify uncertainty and regularise ill-posed problems. In function-space inverse problems, the Bayesian formulation can itself possess strong well-posedness properties even when the underlying deterministic inverse problem is unstable. Stuart's formulation made this connection particularly influential in modern inverse-problem theory. citeturn569338search3turn216997search7

A proper posterior should not, however, be confused with data identifiability. Return to the model

$$
y(t)=ab\,e^{-kt}.
$$

If the likelihood depends on $a$ and $b$ only through their product, the likelihood retains a ridge. Independent proper priors on $a$ and $b$ can turn the posterior into a proper probability distribution and may produce finite posterior means and credible intervals for each parameter. The posterior has become mathematically well defined because prior information resolves the ambiguity. The data still did not distinguish $a$ from $b$.

This is not a criticism of Bayesian inference. It is exactly what priors are supposed to do: combine information external to the current likelihood with the observations. The interpretive requirement is to distinguish likelihood information from prior information. If posterior concentration along a weakly identified direction is driven mainly by the prior, the resulting certainty should not be attributed to the experiment.

Posterior geometry can reveal the same issues as profile likelihood. Strong parameter correlations, long ridges, multimodality and sensitivity to prior scale can all indicate weak identification. A sampler that mixes poorly along such a ridge is partly exposing a statistical problem rather than merely a computational one.

## Better experiments can change identifiability

Structural non-identifiability is defined for a specific model, input and observation scheme. Changing what is measured can change the inverse problem. In the multiplicative example, observing

$$
y_1(t)=ab\,e^{-kt}
$$

alone identifies only $ab$ and $k$. If an additional experiment measures

$$
y_2=a,
$$

then $a$ becomes known and

$$
b=\frac{ab}{a}
$$

is identified as well. The ambiguity was not repaired by collecting more samples of $y_1$. It was repaired by adding an observation that breaks the symmetry.

The same logic drives optimal experimental design. Measurements should be chosen not merely to increase sample size but to distinguish parameter directions that the current design confounds. For a local linearised problem, this can mean increasing the smallest singular values of the sensitivity matrix or improving an information criterion such as the determinant or minimum eigenvalue of the Fisher information matrix.

Time placement also matters. Consider an exponential decay

$$
y(t)=Ae^{-kt}.
$$

If measurements are concentrated in a very narrow early-time window, the approximation

$$
e^{-kt}\approx1-kt
$$

can make combinations of amplitude and decay rate poorly separated under noise. Extending observations over a time range in which curvature becomes visible can dramatically improve practical identifiability without changing the model or estimator.

Inputs can play the same role. In dynamical systems, a constant input may leave parameters confounded while a sufficiently rich excitation separates their effects. Repeated measurements of the same uninformative experiment are not equivalent to a redesigned experiment.

## Estimation should come after an identifiability argument

No single diagnostic solves every inverse problem. Structural identifiability may require symbolic algebra, differential geometry, differential elimination or observability methods. Practical identifiability can be examined through profile likelihoods, likelihood contours, sensitivity matrices, Fisher information, bootstrap distributions, posterior geometry or repeated simulation under realistic noise. Linear problems allow direct singular-value and condition-number analysis. Nonlinear and infinite-dimensional problems require more care.

The order of reasoning is nevertheless stable. First define the scientific quantity to be recovered and the forward map that connects it to observations. Then determine whether distinct values of that quantity can generate the same ideal observations. If uniqueness holds, examine whether the inverse is stable at the available data quality and design. Only then does numerical estimation have a clear interpretation.

This order prevents several common mistakes. A low training error does not prove parameter identification. A successful optimiser does not prove uniqueness. A nonsingular numerical Hessian at one point does not establish global identifiability. A narrow interval produced after strong regularisation does not show that the likelihood was informative. A proper posterior does not imply that the data alone identified every parameter.

Identifiability is therefore not a technical check to perform after fitting. It determines what fitting can mean. The inverse problem begins with the question of whether the observations contain enough information to distinguish the objects we want to infer. Estimation is what we do once that question has been answered.

## References

Cobelli, C., & DiStefano, J. J. (1980). Parameter and structural identifiability concepts and ambiguities: a critical review and analysis. *American Journal of Physiology*, 239(1), R7–R24. https://doi.org/10.1152/ajpregu.1980.239.1.R7

Kaipio, J., & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer.

Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., & Timmer, J. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics*, 25(15), 1923–1929. https://doi.org/10.1093/bioinformatics/btp358

Stuart, A. M. (2010). Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559. https://doi.org/10.1017/S0962492910000061

Villaverde, A. F., Barreiro, A., & Papachristodoulou, A. (2016). Structural identifiability of dynamic systems biology models. *PLOS Computational Biology*, 12(10), e1005153. https://doi.org/10.1371/journal.pcbi.1005153

Wanika, L., et al. (2024). Structural and practical identifiability analysis in bioengineering: a beginner's guide. *Journal of Biological Engineering*, 18, 20. https://doi.org/10.1186/s13036-024-00410-x
