---
permalink: '/mathematics/ill_posed_problems_and_what_regularization_really_does/'
title: 'Ill-Posed Problems and What Regularization Really Does'
date: '2025-04-17'
categories:
- Mathematics
tags:
- Inverse Problems
- Regularization
- Singular Value Decomposition
- Ill-Posed Problems
- Numerical Analysis
author_profile: false
classes: wide
seo_title: 'Ill-Posed Problems and What Regularization Really Does'
seo_description: 'Regularization does not recover information that unstable inverse problems have lost. It controls noise amplification by replacing exact inversion with a biased but stable approximation.'
seo_type: article
excerpt: >-
  In an ill-posed inverse problem, small observational errors can become large
  errors in the reconstructed parameters or fields. Regularization works by
  suppressing unstable directions, trading exact inversion for controlled bias.
summary: >-
  This article develops regularization from the singular-value decomposition of
  a linear inverse problem. It shows how small singular values amplify noise,
  derives Tikhonov and truncated-SVD filter factors, and explains the
  bias-stability trade-off introduced by regularization. A numerical example
  demonstrates how an observational perturbation of 1e-5 can create an order-ten
  parameter error under exact inversion. The discussion then covers parameter
  choice through discrepancy principles, generalized cross-validation and the
  L-curve, before extending the same ideas to nonlinear and Bayesian inverse
  problems.
keywords:
- ill posed inverse problems
- regularization
- Tikhonov regularization
- truncated SVD
- singular value decay
- inverse problems
why_this_exists: >-
  Regularization is often introduced as a generic penalty used to prevent
  overfitting. In inverse problems its role is more fundamental. It replaces an
  unstable inverse with a family of stable approximations whose conclusions
  depend explicitly on additional structural assumptions.
evidence: >-
  Classical regularization theory, singular-value analysis of discrete inverse
  problems, generalized cross-validation, the L-curve and standard monographs on
  deterministic and statistical inverse problems.
methodology: >-
  Analyse a compact discrete inverse problem through its singular-value
  decomposition, derive exact noise amplification under unregularized inversion,
  then derive Tikhonov and truncated-SVD filter factors and compare their action
  on stable and unstable singular directions.
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
Question: What mathematical problem does regularization solve in an inverse problem?
Claim: Regularization controls instability by attenuating parameter directions that the data determine only weakly. It replaces exact inversion with a biased but stable approximation and therefore introduces assumptions that must be separated from information supplied by the observations.
Counterclaim: Regularization is not merely a compromise forced by bad numerics. In genuinely ill-posed problems a stable inverse cannot exist without restricting the admissible solution class or incorporating additional information.
Evidence object: SVD decomposition of a linear inverse problem, an exact three-dimensional noise-amplification example, Tikhonov and truncated-SVD filter factors, and parameter-selection criteria.
Failure case: Treating the regularized solution as though it were uniquely implied by the data, selecting the regularization parameter only to improve visual smoothness, or comparing regularizers without acknowledging that they encode different prior structure.
Reader payoff: Understand where instability comes from, how regularization changes the inverse, why the regularization parameter controls bias and variance, and what claims can legitimately be made from a regularized reconstruction.
Exclusions: Repeating generic machine-learning explanations of L1 and L2 penalties, claiming one parameter-selection method is universally optimal, or treating all numerical ill-conditioning as evidence of structural non-identifiability.
-->

The phrase *ill-posed inverse problem* is sometimes used loosely for any estimation problem that is numerically difficult. The classical meaning is more precise. A problem is well posed in the sense associated with Hadamard when a solution exists, is unique and depends continuously on the observations. Inverse problems often fail the third condition even when the first two are satisfied. The forward model may map many distinct parameter directions into observations of very different magnitudes, so that some features of the unknown are transmitted strongly while others are almost erased. When the inverse tries to reconstruct those weak directions, measurement noise is amplified together with the signal.

This is the setting in which regularization becomes mathematically necessary rather than merely convenient. Regularization does not make the missing information reappear. It changes the inverse problem so that weakly determined directions are suppressed, constrained or replaced by assumptions about what kinds of solutions are acceptable. The price is bias. The benefit is stability. Understanding this trade-off requires looking directly at the singular values of the forward operator.

Consider the linear observation model

$$
y=A\theta+\varepsilon,
$$

where $A\in\mathbb R^{m\times n}$ is the discrete forward operator, $\theta$ is the unknown parameter or discretised field, $y$ is the observation vector and $\varepsilon$ represents noise or model discrepancy. If $A$ has full column rank and $m\ge n$, the least-squares solution is

$$
\hat\theta
=
(A^\top A)^{-1}A^\top y.
$$

Algebraically, the expression is unremarkable. Numerically, everything depends on the spectrum of $A$.

## Ill-posedness appears in the singular spectrum

Write the singular-value decomposition as

$$
A=U\Sigma V^\top,
$$

with singular values

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0.
$$

For simplicity, first assume $A$ is square and invertible. Then

$$
A^{-1}
=
V\Sigma^{-1}U^\top
$$

and the exact inverse solution can be written as

$$
\hat\theta
=
\sum_{i=1}^{n}
\frac{
u_i^\top y
}{
\sigma_i
}
v_i.
$$

This expression explains the instability directly. The data are projected onto the left singular vectors $u_i$. Each projection is then divided by $\sigma_i$. Components associated with large singular values are reconstructed stably. Components associated with very small singular values are multiplied by very large factors.

If

$$
y=A\theta_\star+\varepsilon,
$$

then the reconstruction error is

$$
\hat\theta-\theta_\star
=
\sum_i
\frac{
u_i^\top\varepsilon
}{
\sigma_i
}
v_i.
$$

Noise aligned with singular direction $u_i$ is amplified by $1/\sigma_i$. The condition number

$$
\kappa_2(A)
=
\frac{
\sigma_{\max}
}{
\sigma_{\min}
}
$$

summarises the worst relative amplification in the finite-dimensional linear problem. A large condition number means that some parameter directions are much less observable than others.

A three-dimensional example makes the scale of the problem explicit. Let

$$
A
=
\begin{pmatrix}
1 & 0 & 0\\
0 & 10^{-3} & 0\\
0 & 0 & 10^{-6}
\end{pmatrix},
$$

and suppose the true parameter vector is

$$
\theta_\star
=
\begin{pmatrix}
1\\
1\\
1
\end{pmatrix}.
$$

The exact noise-free observation is

$$
y_\star
=
\begin{pmatrix}
1\\
10^{-3}\\
10^{-6}
\end{pmatrix}.
$$

Now perturb only the third observation by $10^{-5}$,

$$
\varepsilon
=
\begin{pmatrix}
0\\
0\\
10^{-5}
\end{pmatrix}.
$$

The observational error is small in absolute terms. Exact inversion produces

$$
A^{-1}\varepsilon
=
\begin{pmatrix}
0\\
0\\
10
\end{pmatrix},
$$

so the reconstructed third parameter becomes approximately

$$
11
$$

instead of $1$. An error at the fifth decimal place in the observation has created an order-ten error in the parameter because the inverse divided that error by

$$
10^{-6}.
$$

Nothing failed computationally. The inverse was evaluated correctly. The instability is a property of the problem.

The same phenomenon becomes more severe in many discretisations of integral equations, tomography, deconvolution and inverse partial differential equations because the singular values do not merely differ by a fixed factor. They can decay systematically towards zero as the discretisation resolves finer scales. Increasing the numerical resolution then introduces parameter directions that are increasingly difficult to observe. The forward problem may become more detailed while the inverse becomes less stable.

This explains why simply collecting a higher-resolution discretisation does not necessarily improve reconstruction. A finer mesh can enlarge the space of unknowns faster than the data constrain it. The new degrees of freedom often live in precisely the high-frequency directions that the forward operator attenuates most strongly.

## Regularization changes the inverse

Tikhonov regularization replaces exact least squares with

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

In the singular-vector basis this becomes

$$
\hat\theta_\lambda
=
\sum_i
\frac{
\sigma_i
}{
\sigma_i^2+\lambda
}
(u_i^\top y)
v_i.
$$

It is useful to rewrite the coefficient as

$$
\frac{
\sigma_i
}{
\sigma_i^2+\lambda
}
=
\frac{1}{\sigma_i}
\frac{
\sigma_i^2
}{
\sigma_i^2+\lambda
}.
$$

The first factor,

$$
\frac{1}{\sigma_i},
$$

is the unstable exact inverse. The second,

$$
f_i(\lambda)
=
\frac{
\sigma_i^2
}{
\sigma_i^2+\lambda
},
$$

is a filter factor. When $\sigma_i^2\gg\lambda$, the filter factor is close to one and the corresponding singular direction is reconstructed almost as under exact inversion. When $\sigma_i^2\ll\lambda$, the filter factor is close to zero and the unstable direction is strongly attenuated.

Regularization is therefore not a mysterious penalty added to an optimisation problem. In the singular basis it is a frequency-dependent modification of the inverse. Stable directions survive. Weak directions are damped.

Return to the diagonal example and choose

$$
\lambda=10^{-8}.
$$

The three filter factors are approximately

$$
f_1\approx1,
$$

$$
f_2
=
\frac{
10^{-6}
}{
10^{-6}+10^{-8}
}
\approx0.9901,
$$

and

$$
f_3
=
\frac{
10^{-12}
}{
10^{-12}+10^{-8}
}
\approx10^{-4}.
$$

The first two parameter directions are almost untouched. The third direction, which exact inversion amplified catastrophically, is nearly removed. The regularized solution becomes stable because the procedure refuses to reconstruct a component that the data determine too weakly.

That refusal creates bias even in noise-free data. If the true parameter has substantial mass in the suppressed direction, the regularized reconstruction will shrink it. The method is stable precisely because it does not attempt exact recovery of every component.

This produces the central regularization trade-off. Small $\lambda$ gives low bias but high sensitivity to noise. Large $\lambda$ gives greater stability but stronger bias. There is no choice of $\lambda$ that simultaneously reconstructs arbitrarily weak singular directions and prevents noise in those same directions from being amplified.

The structure of $L$ matters just as much as the magnitude of $\lambda$. Choosing $L=I$ penalises large parameter norm. Choosing a first-difference matrix penalises roughness. A second-difference operator favours small curvature. In an imaging problem, total variation favours piecewise constant structure. Each regularizer declares some solutions more plausible than others. The regularized estimate is therefore determined jointly by the data and by the geometry imposed through the penalty.

This is the inverse-problem interpretation of regularization. It is not merely a device for controlling model complexity. It is an explicit way of supplying information in directions where the forward operator is weak.

## Truncated SVD makes the same logic visible in a harder form

Tikhonov regularization attenuates unstable directions continuously. Truncated singular-value decomposition uses a sharper rule. For threshold $\tau$, define

$$
\hat\theta_\tau
=
\sum_{\sigma_i\ge\tau}
\frac{
u_i^\top y
}{
\sigma_i
}
v_i.
$$

Directions with singular values above the threshold are inverted exactly. Directions below the threshold are removed entirely.

The filter factors are therefore

$$
f_i(\tau)
=
\begin{cases}
1, & \sigma_i\ge\tau,\\
0, & \sigma_i<\tau.
\end{cases}
$$

The method is less smooth than Tikhonov regularization but conceptually transparent. It divides the parameter space into directions trusted by the data and directions regarded as too unstable to reconstruct.

This also clarifies why different regularizers can produce different reconstructions from the same observations. They implement different rules about which directions are trusted and how weak directions should be treated. A comparison between regularizers is therefore not simply a comparison of optimisation algorithms. It is a comparison of assumptions.

For many compact inverse problems, the singular vectors themselves have an interpretable structure. High-index singular vectors often oscillate more rapidly than low-index vectors, while the corresponding singular values decrease. Noise is therefore amplified most strongly in fine-scale components. Smoothness regularization works well when the true solution is expected to have little energy in those components. If the true solution contains sharp features, the same regularizer can erase genuine structure.

The regularizer must therefore be judged relative to the scientific object being reconstructed. A penalty that is appropriate for a smooth diffusion coefficient may be inappropriate for a piecewise-constant geological boundary. Stability is not achieved for free. It is achieved by restricting the class of solutions.

## Choosing the regularization parameter is part of the inference

Once a family of regularized solutions

$$
\hat\theta_\lambda
$$

has been defined, the value of $\lambda$ becomes an inferential decision. Choosing it only because one reconstruction looks visually attractive is difficult to defend because appearance can hide both overfitting and oversmoothing.

One approach is the discrepancy principle. If the noise level is known or credibly estimated, choose $\lambda$ so that the residual norm is comparable with the amount of noise expected in the observations,

$$
\|A\hat\theta_\lambda-y\|_2
\approx
\delta,
$$

where $\delta$ represents the expected noise magnitude. The logic is that fitting the observations much more closely than the noise level requires the inverse to reproduce noise, while a residual much larger than the noise level indicates excessive regularization or model discrepancy.

The method has a clear interpretation when the noise scale is known. Its weakness is equally clear when the error model is poorly understood. In real scientific inverse problems, measurement noise and model discrepancy are often mixed, and forcing the residual to match a nominal instrumental error can produce misleading confidence.

Generalized cross-validation takes a different route. In ridge-type linear problems, write the fitted data as

$$
\hat y_\lambda
=
H_\lambda y,
$$

where

$$
H_\lambda
=
A(A^\top A+\lambda I)^{-1}A^\top.
$$

Golub, Heath and Wahba proposed choosing $\lambda$ by minimizing

$$
G(\lambda)
=
\frac{
\|(I-H_\lambda)y\|_2^2/m
}{
\left[
1-\operatorname{tr}(H_\lambda)/m
\right]^2
}.
$$

The numerator measures residual error while the denominator corrects for the effective flexibility of the smoother. GCV is attractive because it does not require a known noise variance and is invariant to orthogonal transformations of the observation coordinates. Its success still depends on the error structure and the relationship between the chosen regularization family and the true inverse problem.

The L-curve gives a geometric diagnostic. For a range of $\lambda$, plot the residual norm against the regularization seminorm, commonly

$$
\left(
\log\|A\hat\theta_\lambda-y\|_2,
\log\|L\hat\theta_\lambda\|_2
\right).
$$

Many problems produce an L-shaped curve. One arm corresponds to weak regularization, where the residual is small but the solution norm becomes large or rough. The other corresponds to strong regularization, where the solution is constrained heavily and the residual grows. The corner is interpreted as a compromise between those regimes. Hansen developed the L-curve as a systematic tool for analysing discrete ill-posed problems, and Hansen and O'Leary studied its use for selecting regularization parameters.

The L-curve is useful because it visualises the trade-off directly. It is not a theorem that the visually sharpest corner produces the scientifically best reconstruction. Some problems have no clear corner. Noise can be correlated. Different penalties produce different curves. Parameter choice should therefore be treated as part of the uncertainty analysis rather than hidden as a preprocessing detail.

In many applications, several parameter-selection methods should be compared. If GCV, a discrepancy principle and an L-curve criterion all select broadly similar regularization strengths, that agreement is useful evidence. If they imply substantially different reconstructions, the disagreement reveals that the data do not determine the regularization scale strongly.

## Bias and variance can be separated in the singular basis

The singular-value representation also makes the bias-stability trade-off explicit. Suppose the true parameter is

$$
\theta_\star
=
\sum_i
\theta_i^\star v_i
$$

and the observation noise is zero-mean with variance $\sigma_\varepsilon^2$ in the left-singular-vector basis. Under Tikhonov regularization with $L=I$,

$$
\hat\theta_\lambda
=
\sum_i
f_i(\lambda)\theta_i^\star v_i
+
\sum_i
\frac{
\sigma_i
}{
\sigma_i^2+\lambda
}
(u_i^\top\varepsilon)v_i.
$$

The expected reconstruction bias in direction $v_i$ is

$$
\left[
f_i(\lambda)-1
\right]
\theta_i^\star,
$$

while the noise variance in that direction is proportional to

$$
\sigma_\varepsilon^2
\frac{
\sigma_i^2
}{
(\sigma_i^2+\lambda)^2
}.
$$

Increasing $\lambda$ suppresses the variance term in weak directions but increases the magnitude of the bias term. The trade-off is not metaphorical. It is visible term by term.

This representation also shows why regularization can be effective even when the unregularized estimator is unbiased. Unbiasedness alone is not enough when the variance is enormous. In an ill-posed inverse problem, a small amount of bias can produce a much smaller mean squared reconstruction error by eliminating unstable noise amplification.

The same reasoning underlies shrinkage methods in statistics, but the inverse-problem setting gives the phenomenon a direct geometric interpretation. The weak singular directions are not merely parameters with large variance. They are directions that the observation operator almost removes before the data are measured.

## Nonlinear problems inherit the same structure locally

Most scientific inverse problems are nonlinear,

$$
y=F(\theta)+\varepsilon.
$$

Near a current parameter value $\theta_0$, linearise the forward map,

$$
F(\theta_0+\Delta\theta)
\approx
F(\theta_0)
+
J(\theta_0)\Delta\theta,
$$

where $J$ is the Jacobian. The singular values of $J$ then describe local parameter directions that are strongly or weakly observed. Small singular values create the same local noise-amplification problem as in the linear case.

Regularized nonlinear least squares can be written as

$$
\hat\theta_\lambda
=
\arg\min_\theta
\left\{
\|F(\theta)-y\|_2^2
+
\lambda R(\theta)
\right\},
$$

where $R$ encodes the preferred solution structure. Gauss-Newton and Levenberg-Marquardt methods can themselves be understood partly through this lens. Damping stabilises steps when the local curvature or Jacobian is poorly conditioned.

The nonlinear setting adds complications because singular directions can change with $\theta$, multiple local minima can exist, and a regularizer can alter not only stability but which basin of attraction an optimiser reaches. Parameter-choice sensitivity should therefore be examined together with initialization sensitivity and profile geometry.

Iterative regularization provides another perspective. Some algorithms become unstable only after too many iterations because early iterates reconstruct well-determined large-scale components before later iterates begin fitting weak singular directions and noise. Stopping the iteration early can therefore act as a regularization method. The iteration count plays a role analogous to $\lambda$.

This phenomenon is particularly clear in Landweber iteration and related gradient methods for inverse problems. Early stopping is not only a machine-learning heuristic for generalization. In an inverse problem it can be a mathematically explicit way to avoid inverting unstable components too aggressively.

## Bayesian regularization makes the additional information explicit

For Gaussian linear inverse problems, Tikhonov regularization has a direct Bayesian interpretation. Suppose

$$
y\mid\theta
\sim
\mathcal N(A\theta,\Sigma_\varepsilon)
$$

and assign a Gaussian prior

$$
\theta
\sim
\mathcal N(0,\Sigma_\theta).
$$

The negative log posterior, up to constants, is

$$
(y-A\theta)^\top
\Sigma_\varepsilon^{-1}
(y-A\theta)
+
\theta^\top
\Sigma_\theta^{-1}
\theta.
$$

The maximum a posteriori estimator is therefore a regularized least-squares solution. When

$$
\Sigma_\varepsilon=\sigma^2I
$$

and

$$
\Sigma_\theta^{-1}
\propto
L^\top L,
$$

the prior precision generates the regularization penalty.

This equivalence is useful because it clarifies the source of the stabilizing information. The regularizer corresponds to assumptions about the plausible scale or structure of $\theta$. In the Bayesian formulation those assumptions are stated as a probability distribution rather than only as an optimisation penalty.

The posterior can quantify uncertainty in directions that deterministic regularization returns as one point estimate. It still cannot make the likelihood informative where the forward operator is not. When the data are weak, the posterior depends more strongly on the prior. Prior sensitivity is therefore an important diagnostic of how much of the apparent reconstruction is observation-driven.

In function-space inverse problems this point becomes even more important. A discretisation-dependent penalty can behave differently as the mesh is refined, whereas a well-specified prior on the underlying function space can define a coherent infinite-dimensional inverse problem. The numerical discretisation should approximate that problem rather than silently define a new prior every time the grid changes.

## Regularization should be reported as part of the scientific model

A reconstructed parameter field or latent signal is often presented visually as though it were a transformed version of the data. That presentation hides how strongly the result may depend on the regularization assumptions. A scientifically complete report should state the forward model, observation model, penalty or prior, regularization parameter, parameter-selection rule and sensitivity of the reconstruction to plausible alternatives.

If two reasonable values of $\lambda$ produce materially different scientific conclusions, then the conclusion is regularization-sensitive. If changing from a smoothness penalty to total variation changes the location of a recovered boundary, then the boundary is partly assumption-dependent. If a Bayesian reconstruction narrows dramatically only under an informative prior, that fact is part of the inference.

The point is not to avoid regularization. Many inverse problems cannot be solved meaningfully without it. The point is to describe honestly what the regularizer contributes.

In the diagonal example, exact inversion transformed a perturbation of $10^{-5}$ into an error of $10$ in the third parameter direction. Tikhonov regularization prevented that amplification by suppressing the component associated with singular value $10^{-6}$. The method succeeded because it declined to reconstruct what the data could not determine reliably. The regularized value in that direction was therefore not an improved measurement of the original parameter. It was a stable estimate obtained by combining weak data information with the structural preference encoded by the penalty.

That distinction is the essence of regularization in inverse problems. The forward operator determines which directions the observations reveal. Noise determines how far those directions can be trusted. The regularizer determines what is done with the remainder.

Regularization does not recover lost information. It makes an underdetermined or unstable inference usable by stating, explicitly or implicitly, what kinds of solutions we are prepared to believe.

## References

Engl, H. W., Hanke, M., & Neubauer, A. (1996). *Regularization of Inverse Problems*. Kluwer Academic Publishers.

Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as a method for choosing a good ridge parameter. *Technometrics*, 21(2), 215–223. https://doi.org/10.1080/00401706.1979.10489751

Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the L-curve. *SIAM Review*, 34(4), 561–580. https://doi.org/10.1137/1034115

Hansen, P. C. (1998). *Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion*. SIAM.

Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-curve in the regularization of discrete ill-posed problems. *SIAM Journal on Scientific Computing*, 14(6), 1487–1503. https://doi.org/10.1137/0914086

Kaipio, J., & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer.

Tikhonov, A. N., & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems*. V. H. Winston & Sons.
