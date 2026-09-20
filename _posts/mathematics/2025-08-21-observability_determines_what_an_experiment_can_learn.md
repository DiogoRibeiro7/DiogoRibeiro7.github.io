---
permalink: '/mathematics/observability_determines_what_an_experiment_can_learn/'
title: 'Observability Determines What an Experiment Can Learn'
date: '2025-08-21'
categories:
- Mathematics
tags:
- Inverse Problems
- Observability
- Experimental Design
- Fisher Information
- System Identification
author_profile: false
classes: wide
seo_title: 'Observability Determines What an Experiment Can Learn'
seo_description: 'More measurements do not necessarily mean more information. Observability and experimental design determine which states and parameters can actually be recovered from data.'
seo_type: article
excerpt: >-
  Repeating the same measurement can reduce noise without changing what is
  identifiable. An informative experiment changes the geometry of the inverse
  problem by revealing directions that the original observation scheme could
  not distinguish.
summary: >-
  This article connects observability, identifiability and optimal experimental
  design. It develops the observability matrix for linear dynamical systems,
  shows with exact examples why repeated measurements cannot repair rank
  deficiency, derives the Fisher information matrix for parameter estimation,
  and compares D-, A- and E-optimal design criteria. The discussion then extends
  to nonlinear systems, sensor placement, sampling time, input design and robust
  experiments under model discrepancy.
keywords:
- observability
- experimental design
- inverse problems
- Fisher information
- sensor placement
- system identification
why_this_exists: >-
  Applied modelling often treats data quantity as a proxy for information.
  Inverse problems show why that is wrong. A thousand repeated measurements of
  an uninformative output can leave an entire state or parameter combination
  unobservable, while one strategically chosen measurement can remove the
  ambiguity.
evidence: >-
  Classical observability theory for linear and nonlinear systems, Fisher
  information based optimal design, structural-identifiability theory and
  modern work on informative experiment design for nonlinear dynamic models.
methodology: >-
  Start from a linear state-space system and derive the observability matrix.
  Use exact rank-deficient examples to separate repeated precision from new
  information, then formulate parameter information through local sensitivities
  and the Fisher information matrix. Compare D-, A- and E-optimality before
  extending the discussion to nonlinear and robust design.
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
Question: Why can more observations fail to improve an inverse problem?
Claim: Information depends on how observations interact with the forward model, not simply on sample size. Repeated measurements can reduce noise in already observed directions while leaving unobservable or weakly identifiable directions untouched.
Counterclaim: Repetition still matters. Once the observation scheme spans the relevant directions, repeated measurements can improve precision substantially. The point is not that more data are useless, but that design determines which dimensions more data can inform.
Evidence object: Linear observability matrix, double-integrator example, rank-one Fisher-information example, exact comparison with a second experiment that restores full rank, and classical optimality criteria.
Failure case: Maximising sample size without examining rank, using D-optimality mechanically under a wrong local model, or assuming the observation design that is optimal for parameter estimation is also optimal for prediction or model discrimination.
Reader payoff: Understand when to collect more of the same data and when to change sensors, sampling times, inputs or experimental conditions instead.
Exclusions: Treating observability and identifiability as identical in every nonlinear system, claiming one optimality criterion is universally best, or assuming Fisher-information design remains valid far from the parameter values used to construct it.
-->

The amount of data collected in an experiment is often used as shorthand for how informative the experiment is. Larger samples reduce sampling variability, repeated measurements can average out noise, and denser time series can reveal structure that sparse observations miss. None of this implies that information increases equally in every direction of a model. An experiment can contain thousands of measurements and still be incapable of distinguishing two states or two parameter combinations that generate the same observed output.

This is the role of observability and experimental design in inverse problems. The central question is not only how many observations are available, but whether the observation operator exposes the parts of the system one wants to recover. If the experiment projects several physically distinct states onto the same measured quantity, repetition improves precision about that projection without separating the hidden components. A new sensor, a different input, or a strategically chosen measurement time can be more informative than a large increase in sample size because it changes the geometry of the inverse map.

The distinction is easiest to see in dynamical systems. Consider the linear state-space model

$$
\dot x(t)
=
Ax(t)+Bu(t),
$$

with observations

$$
y(t)=Cx(t).
$$

The state vector $x(t)\in\mathbb R^n$ contains the internal variables of the system, while $y(t)$ contains only what the experiment measures. Even when the dynamics $A$ are known exactly, recovering the initial state $x(0)$ from the observed trajectory is possible only if the output carries enough information about every state direction.

## Observability is a rank property

For the autonomous linear system

$$
\dot x(t)=Ax(t),
\qquad
y(t)=Cx(t),
$$

the solution is

$$
x(t)=e^{At}x(0),
$$

so the complete output trajectory is

$$
y(t)=Ce^{At}x(0).
$$

Differentiating the output at $t=0$ gives

$$
y(0)=Cx(0),
$$

$$
\dot y(0)=CAx(0),
$$

$$
\ddot y(0)=CA^2x(0),
$$

and in general

$$
y^{(k)}(0)=CA^kx(0).
$$

Collecting the first $n$ such relationships produces the observability matrix

$$
\mathcal O
=
\begin{pmatrix}
C\\
CA\\
CA^2\\
\vdots\\
CA^{n-1}
\end{pmatrix}.
$$

The linear system is observable when

$$
\operatorname{rank}(\mathcal O)=n.
$$

This criterion makes the relevant notion of information explicit. Observability is not determined by the number of rows of recorded data alone. It depends on whether the measured outputs and their dynamical evolution span all state directions.

A two-dimensional example is enough to show the difference. Consider the double-integrator system

$$
A
=
\begin{pmatrix}
0 & 1\\
0 & 0
\end{pmatrix},
$$

so that

$$
\dot x_1=x_2,
\qquad
\dot x_2=0.
$$

If the experiment measures the first state,

$$
C
=
\begin{pmatrix}
1 & 0
\end{pmatrix},
$$

then

$$
CA
=
\begin{pmatrix}
0 & 1
\end{pmatrix},
$$

and therefore

$$
\mathcal O
=
\begin{pmatrix}
1 & 0\\
0 & 1
\end{pmatrix}.
$$

The system is observable. Measuring position through $x_1$ over time also reveals the constant velocity $x_2$ because the slope of the position trajectory contains that information.

Now suppose the experiment measures only the second state,

$$
C
=
\begin{pmatrix}
0 & 1
\end{pmatrix}.
$$

Then

$$
CA
=
\begin{pmatrix}
0 & 0
\end{pmatrix},
$$

and

$$
\mathcal O
=
\begin{pmatrix}
0 & 1\\
0 & 0
\end{pmatrix}.
$$

Its rank is one. The velocity is observed, but the initial position is invisible because changing $x_1(0)$ has no effect on the measured output. Recording the velocity one thousand times with negligible measurement error does not reveal the missing initial position. The experiment is not short of precision. It is missing an observation direction.

This is the essential difference between collecting more data and collecting different information.

## Repetition reduces noise without changing rank

The same geometry appears directly in parameter estimation. Suppose an experiment measures

$$
y
=
\theta_1+\theta_2+\varepsilon,
$$

with

$$
\varepsilon\sim\mathcal N(0,\sigma^2).
$$

The sensitivity of the mean response to the parameter vector is

$$
s
=
\begin{pmatrix}
1 & 1
\end{pmatrix}.
$$

For one observation, the Fisher information matrix is

$$
\mathcal I_1
=
\frac{1}{\sigma^2}
s^\top s
=
\frac{1}{\sigma^2}
\begin{pmatrix}
1 & 1\\
1 & 1
\end{pmatrix}.
$$

Its determinant is zero and its rank is one. Only the combination

$$
\theta_1+\theta_2
$$

is identified.

If the same experiment is repeated independently $n$ times, the information matrix becomes

$$
\mathcal I_n
=
\frac{n}{\sigma^2}
\begin{pmatrix}
1 & 1\\
1 & 1
\end{pmatrix}.
$$

Every nonzero eigenvalue grows by a factor of $n$, so uncertainty about the identifiable combination shrinks. The rank remains one for every finite or infinite $n$. No amount of replication separates $\theta_1$ from $\theta_2$ because the experiment always observes the same linear combination.

A second experiment can change the situation immediately. Suppose an additional condition produces

$$
z
=
\theta_1-\theta_2+\eta,
$$

with

$$
\eta\sim\mathcal N(0,\sigma^2).
$$

Its sensitivity is

$$
r
=
\begin{pmatrix}
1 & -1
\end{pmatrix}.
$$

One observation from each experiment gives

$$
\mathcal I
=
\frac{1}{\sigma^2}
\left(
s^\top s+r^\top r
\right)
=
\frac{1}{\sigma^2}
\begin{pmatrix}
2 & 0\\
0 & 2
\end{pmatrix}.
$$

The information matrix is now full rank. The determinant changes from zero to

$$
\det(\mathcal I)
=
\frac{4}{\sigma^4}.
$$

One carefully chosen new measurement condition has done something that arbitrarily many repetitions of the first condition could never do: it has rotated the sensitivity geometry and separated the two parameters.

This example is deliberately simple, but the same principle governs much larger inverse problems. Sampling the same time point more densely, measuring the same output variable with a second identical sensor, or repeating the same excitation can improve precision without resolving parameter confounding. The useful question is therefore not merely how many measurements are available, but how linearly independent their sensitivities are with respect to the quantities being inferred.

## Fisher information turns design into geometry

For a nonlinear model with observations

$$
y_j
=
g_j(\theta,\xi)+\varepsilon_j,
$$

let $\xi$ denote the experimental design. It may encode sampling times, sensor locations, input trajectories, doses, boundary conditions or any other controllable feature of the experiment. If the observation errors are Gaussian with covariance $\Sigma$, the local sensitivity matrix is

$$
S(\theta,\xi)
=
\frac{\partial g(\theta,\xi)}{\partial\theta}.
$$

The Fisher information matrix is then

$$
\mathcal I(\theta,\xi)
=
S(\theta,\xi)^\top
\Sigma^{-1}
S(\theta,\xi).
$$

The eigenvectors of $\mathcal I$ describe local parameter combinations, while the corresponding eigenvalues describe how strongly the experiment constrains those combinations. A zero eigenvalue represents a locally invisible parameter direction. A very small eigenvalue represents a direction that is technically observable but practically weak. A balanced spectrum indicates that the experiment constrains the parameter space more uniformly.

Optimal experimental design chooses $\xi$ to improve some function of this information matrix. The choice of function depends on the scientific objective.

D-optimal design maximises

$$
\det\mathcal I(\theta,\xi).
$$

For linear Gaussian models, this is equivalent to minimising the volume of the asymptotic confidence ellipsoid for the parameter vector. Because the determinant is the product of the eigenvalues,

$$
\det\mathcal I
=
\prod_i\lambda_i,
$$

D-optimality rewards experiments that enlarge the total information volume and strongly penalises exact rank deficiency.

A-optimal design minimises

$$
\operatorname{tr}
\left(
\mathcal I^{-1}
\right),
$$

which is related to the sum of marginal parameter variances under the local Gaussian approximation.

E-optimal design maximises the smallest eigenvalue,

$$
\lambda_{\min}(\mathcal I),
$$

and therefore focuses on the weakest identified direction. This can be attractive when one poorly observed parameter combination dominates the uncertainty.

These criteria are not equivalent. An experiment that maximises determinant can tolerate one relatively weak direction if gains elsewhere compensate sufficiently. E-optimality is more explicitly concerned with protecting the worst direction. A-optimality weights the average parameter variance. The appropriate criterion therefore depends on whether the goal is overall parameter-volume reduction, balanced identifiability, prediction or some other decision.

The design problem is also model dependent. In nonlinear systems,

$$
\mathcal I(\theta,\xi)
$$

depends on the unknown parameter value $\theta$. A design that is optimal near one nominal parameter vector may be poor if the true system lies elsewhere. Local optimal design therefore inherits the uncertainty of the model parameters it is intended to estimate.

Bayesian experimental design addresses this by averaging utility over a prior distribution,

$$
U(\xi)
=
\mathbb E_{\theta\sim\pi(\theta)}
[
u(\theta,\xi)
].
$$

More generally, one can choose an experiment to maximise expected information gain,

$$
\mathbb E_{y\mid\xi}
\left[
D_{\mathrm{KL}}
\left(
\pi(\theta\mid y,\xi)
\|
\pi(\theta)
\right)
\right].
$$

This formulation asks which experiment is expected to move the posterior furthest from the prior. It is conceptually attractive because it treats information directly, although the required nested integration and optimisation can be computationally expensive.

## Sampling time and input shape can matter more than sample size

Dynamic systems make the importance of design especially clear because sensitivities change over time. Suppose an exponential decay is observed,

$$
y(t)
=
Ae^{-kt}.
$$

The sensitivities are

$$
\frac{\partial y}{\partial A}
=
e^{-kt},
$$

and

$$
\frac{\partial y}{\partial k}
=
-At e^{-kt}.
$$

At

$$
t=0,
$$

the sensitivity to $k$ is zero. Measurements concentrated arbitrarily close to zero can estimate $A$ well while containing little information about the decay rate. Repeating the earliest measurement more precisely does not change that fact.

At very large times, both sensitivities approach zero because the signal itself disappears into noise. The most informative times for $k$ lie between these extremes, where the trajectory has evolved enough for the decay rate to influence the response but the signal remains measurable.

This is why dense uniform sampling is not automatically efficient. Ten carefully chosen times can contain more information about a parameter than one hundred measurements placed where the sensitivity is nearly zero or nearly collinear with another parameter's sensitivity.

The same argument applies to input design. In a dynamical system

$$
\dot x
=
f(x,\theta,u),
$$

the control input $u(t)$ changes the trajectory and therefore changes the sensitivity matrix. A constant input can leave several parameters confounded, while a time-varying or multi-level input can force the system into regimes where those parameters have different observable consequences.

System identification describes this in terms of excitation. If the input does not excite the relevant modes, the corresponding dynamics remain difficult or impossible to recover. The principle is broader than linear control theory. A perturbation experiment is informative when competing parameter values or model structures respond differently to the perturbation.

This is why intervention often reveals more than passive observation. Two mechanisms can produce nearly identical equilibrium behaviour and very different transient responses after a controlled change. Observing the transient can separate them.

Sensor placement has the same geometry. Suppose a spatial field depends on a parameter through a sensitivity function

$$
s(x)
=
\frac{\partial y(x,\theta)}{\partial\theta}.
$$

Placing multiple sensors where $s(x)$ has the same shape or where two parameter sensitivities are nearly proportional gives redundant information. Placing sensors where the sensitivity vectors differ can improve rank and conditioning.

The best sensor location is therefore not necessarily where the measured signal is largest. A location with moderate signal but strong discrimination among parameter directions may be more informative for the inverse problem.

## Nonlinear observability requires more than one matrix

For nonlinear systems

$$
\dot x
=
f(x,u),
$$

$$
y
=
h(x),
$$

the linear observability matrix is no longer sufficient globally. Hermann and Krener developed an observability rank condition based on Lie derivatives of the output. The first derivative is

$$
L_fh(x)
=
\frac{\partial h}{\partial x}
f(x,u),
$$

and successive Lie derivatives describe how the dynamics expose hidden state directions through time.

An observability map can be formed from

$$
h(x),
\quad
L_fh(x),
\quad
L_f^2h(x),
\ldots
$$

and local observability is analysed through the rank of its Jacobian with respect to the state. Parameters can be appended as constant states, linking structural identifiability and observability in nonlinear dynamic models.

The nonlinear case introduces several complications. Rank can depend on the current state, input and parameter values. A system can be observable for one experiment and unobservable for another. Degenerate initial conditions can remove information that exists generically. An input can either reveal or hide state directions. This makes observability not only a property of the equations but of the equations together with the experimental configuration.

That dependence is scientifically useful. If a model is structurally non-identifiable under one input but identifiable under another, the appropriate response is not always to fix a parameter or simplify the model. It may be to redesign the experiment.

Recent work on dynamic biological systems uses exactly this connection between observability conditions and experimental information. The same idea applies to chemical kinetics, pharmacokinetics, mechanical systems, robotics, electrical systems and any nonlinear state-space model in which only part of the state is measured. citeturn953148search0turn953148search2

## Optimal design is only optimal relative to a model and a target

Information-based design can be powerful, but it can also create false confidence if the design criterion is interpreted too literally. A D-optimal design maximises the determinant of a Fisher information matrix computed under a model. If the model is wrong, the experiment can be optimal for distinguishing parameter values inside the wrong model while being poor for detecting the discrepancy itself.

This connects experimental design directly to the problem of model discrepancy. Suppose two candidate parameter vectors differ strongly under the assumed model at one sensor location, so the Fisher information is large there. If both are wrong in the same way because a missing mechanism dominates that location, the measurement may be excellent for local parameter precision and poor for scientific discrimination.

A more robust design may deliberately include observations that challenge the model rather than merely sharpen its parameters. Measurements under new forcing regimes, additional output variables, boundary locations or external validation conditions can expose discrepancy that a locally optimal calibration design would ignore.

The target quantity matters just as much. Parameter-optimal design is not automatically prediction-optimal design. If the scientific objective is to predict a future state

$$
q(\theta),
$$

then the relevant uncertainty is

$$
\nabla q(\theta)^\top
\mathcal I^{-1}
\nabla q(\theta),
$$

not necessarily the volume of the full parameter confidence ellipsoid. Parameters that are weakly identified individually may combine into a well-determined prediction, while a small uncertainty in one particular parameter direction may dominate the predictive quantity of interest.

Model-discrimination design is different again. If the scientific question is which of two mechanisms is more plausible, an informative experiment should maximise the separation between their predicted observations, not necessarily the Fisher information within either model.

There is therefore no universal best experiment. Experimental design is always conditional on the inferential target, model class, noise assumptions and practical constraints.

## More data help after the design becomes informative

The claim that more measurements are not the same as more information should not be pushed too far. Once an experiment observes the relevant directions, repetition matters. If

$$
\mathcal I_1
$$

is full rank, $n$ independent repetitions under the same design often give

$$
\mathcal I_n
=
n\mathcal I_1,
$$

so asymptotic covariance scales like

$$
\mathcal I_n^{-1}
=
\frac{1}{n}
\mathcal I_1^{-1}.
$$

Standard errors then shrink at the familiar rate

$$
n^{-1/2}.
$$

The point is not that sample size is unimportant. It is that sample size multiplies the information already present in the design. It cannot create missing directions when the information matrix is rank deficient.

The practical workflow should therefore separate two questions. First ask whether the experiment observes the quantities required by the scientific problem. Then ask how much replication is needed to estimate those quantities at useful precision.

That order is often reversed. Power calculations are performed for a design whose observation structure has not been checked for identifiability. More participants, more time points or more simulation runs are then proposed as the remedy for wide intervals. If the uncertainty comes from a weak or missing sensitivity direction, redesigning the experiment can be far more effective than increasing $n$.

This is particularly important in expensive scientific experiments. Biological assays, clinical measurements, field sensors, physical prototypes and high-fidelity simulations can all have substantial marginal cost. An information-based design can reduce cost by identifying which measurements are redundant and which new conditions change the rank or conditioning of the inverse problem.

The same reasoning applies to observational studies even when the investigator cannot manipulate the system directly. Data collection can still vary by location, population, time, instrument or covariate distribution. Choosing observations that broaden the support of the design can improve identifiability more than repeated sampling from the same narrow regime.

The inverse-problem perspective therefore changes what it means to ask for more data. More data can mean more repetitions of the same projection, or it can mean new observations that reveal previously hidden dimensions. Only the second changes what the experiment is capable of learning.

## References

Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007). *Optimum Experimental Designs, with SAS*. Oxford University Press.

Hermann, R., & Krener, A. J. (1977). Nonlinear controllability and observability. *IEEE Transactions on Automatic Control*, 22(5), 728–740. https://doi.org/10.1109/TAC.1977.1101601

Kalman, R. E. (1960). On the general theory of control systems. *Proceedings of the First International Congress of Automatic Control*, 481–492.

Pukelsheim, F. (1993). *Optimal Design of Experiments*. Wiley.

Villaverde, A. F. (2019). Observability and structural identifiability of nonlinear biological systems. *Complexity*, 2019, 8497093. https://doi.org/10.1155/2019/8497093

Villaverde, A. F., Pathirana, D., Fröhlich, F., Hasenauer, J., & Banga, J. R. (2022). A protocol for dynamic model calibration. *Briefings in Bioinformatics*, 23(1), bbab387. https://doi.org/10.1093/bib/bbab387

Walter, E., & Pronzato, L. (1997). *Identification of Parametric Models from Experimental Data*. Springer.
