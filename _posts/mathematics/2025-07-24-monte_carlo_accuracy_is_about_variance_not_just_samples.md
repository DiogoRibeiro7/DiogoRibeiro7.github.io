---
permalink: '/mathematics/monte_carlo_accuracy_is_about_variance_not_just_samples/'
title: 'Monte Carlo Accuracy Is About Variance, Not Just Samples'
date: '2025-07-24'
categories:
- Mathematics
tags:
- Monte Carlo
- Variance Reduction
- Importance Sampling
- Control Variates
- Numerical Methods
author_profile: false
classes: wide
seo_title: 'Monte Carlo Accuracy Is About Variance, Not Just Samples'
seo_description: 'Monte Carlo error depends on estimator variance as well as sample size. Control variates, antithetic sampling, stratification and importance sampling can reduce computational cost by orders of magnitude.'
seo_type: article
excerpt: >-
  The familiar square-root convergence rate does not mean that all Monte Carlo
  estimators are equally useful. Good estimator design can reduce variance by
  factors that brute-force sampling would require orders of magnitude more
  computation to match.
summary: >-
  This article develops classical Monte Carlo integration from the central limit
  theorem and treats variance as the central computational quantity. Exact
  examples show a fourfold gain from a control variate and roughly a thirtyfold
  gain from antithetic sampling at equal function-evaluation cost. A six-sigma
  Gaussian tail example then shows how a simple importance sampler reduces the
  number of draws required for 10 percent relative precision from about 1e11 to
  fewer than 1e3. The discussion covers stratification, conditional Monte Carlo,
  weight degeneracy, finite-variance conditions, effective sample size, MCMC
  dependence and the limits of the square-root law.
keywords:
- Monte Carlo integration
- variance reduction
- control variates
- antithetic variates
- importance sampling
- rare event simulation
why_this_exists: >-
  Monte Carlo is often presented as a brute-force method whose accuracy is
  controlled almost entirely by the number of simulations. That view hides the
  main design problem. Two estimators can target exactly the same expectation and
  have radically different variances, making estimator construction more
  important than raw sample count.
evidence: >-
  Exact variance calculations for standard Monte Carlo, control variates and
  antithetic sampling; analytic rare-event calculations for Gaussian importance
  sampling; and classical Monte Carlo variance-reduction theory.
methodology: >-
  Start from the independent Monte Carlo estimator and its central limit theorem,
  then compare alternative unbiased estimators at equal computational cost.
  Derive exact variance ratios where possible and use a Gaussian six-sigma event
  to quantify rare-event efficiency.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-mathematics-voronoi.jpg
  og_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_image: /assets/images/headers/photo-mathematics-voronoi.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-voronoi.jpg
  twitter_image: /assets/images/headers/photo-mathematics-voronoi.jpg
---

<!--
Development contract
Question: Why can two unbiased Monte Carlo estimators of the same quantity have radically different computational value?
Claim: Monte Carlo accuracy is determined by estimator variance and computational cost, not sample size alone. Variance reduction can change the cost of a target precision by orders of magnitude without changing the estimand.
Counterclaim: More sophisticated estimators are not automatically better. Control variates can be weak, antithetic coupling can increase variance for unsuitable integrands, and importance sampling can fail catastrophically when weights have large or infinite variance.
Evidence object: Exact control-variate variance reduction for E[X^4], exact antithetic variance reduction for E[exp(U)], and analytic importance sampling for P(Z>6).
Failure case: Reporting the number of simulations without Monte Carlo standard error, using an importance proposal with insufficient support or unstable weights, or comparing methods without accounting for computational cost.
Reader payoff: Treat Monte Carlo as an estimator-design problem and know when to use control variates, antithetic coupling, stratification, conditional expectation or importance sampling rather than simply increasing N.
Exclusions: A survey of MCMC algorithms, a catalogue of quasi-Monte-Carlo constructions, and repetition of the existing standalone importance-sampling article.
-->

Monte Carlo methods are often introduced with a reassuringly simple recipe: generate random draws, evaluate a function, average the results, and increase the sample size until the answer stabilises. The mathematical justification is equally familiar. If independent observations have finite variance, the central limit theorem gives an error that decreases at the rate $N^{-1/2}$. This rate is dimension-independent in a narrow but important sense, and it explains why Monte Carlo remains useful when deterministic quadrature becomes difficult.

The same presentation can encourage a misleading computational habit. If the estimate is noisy, run more simulations. If the desired standard error is ten times smaller, run one hundred times as many. That conclusion follows only after the estimator itself has been fixed. Monte Carlo methods are not merely about how many random numbers are generated. They are about constructing a random estimator of a deterministic quantity, and two unbiased estimators of exactly the same target can have variances that differ by factors of ten, one thousand, or much more.

Variance is therefore a computational resource. Reducing it changes the amount of simulation required for a given precision. Control variates, antithetic sampling, stratification, conditional expectation and importance sampling all exploit known structure to change variance without changing the target quantity. In difficult rare-event problems, estimator design can be the difference between a calculation that finishes in seconds and one that is effectively impossible.

This article concerns classical Monte Carlo integration rather than Markov chain Monte Carlo. The blog already has separate material on [MCMC](/mathematics/Monte_Carlo/) and [importance sampling](/mathematics/Importance_Sampling/). The purpose here is to develop the common mathematical principle behind variance reduction and to quantify what the gains actually mean.

## The square-root law is a statement about one estimator

Suppose the target is

$$
\mu
=
\mathbb E_p[f(X)]
=
\int f(x)p(x)\,dx.
$$

If

$$
X_1,\ldots,X_N
$$

are independent draws from $p$, the ordinary Monte Carlo estimator is

$$
\hat\mu_N
=
\frac{1}{N}
\sum_{i=1}^N
f(X_i).
$$

Provided

$$
\sigma_f^2
=
\operatorname{Var}_p[f(X)]
<
\infty,
$$

the estimator is unbiased,

$$
\mathbb E[\hat\mu_N]
=
\mu,
$$

with variance

$$
\operatorname{Var}(\hat\mu_N)
=
\frac{\sigma_f^2}{N}.
$$

The central limit theorem gives

$$
\sqrt N
\frac{
\hat\mu_N-\mu
}{
\sigma_f
}
\Rightarrow
N(0,1).
$$

A natural Monte Carlo standard error is therefore

$$
\operatorname{MCSE}
\approx
\frac{s_f}{\sqrt N},
$$

where $s_f^2$ is the sample variance of the simulated values $f(X_i)$.

The square-root rate is both powerful and unforgiving. If the standard error must be divided by two, the sample size must be multiplied by four. Dividing error by ten requires roughly one hundred times as many draws. Dividing it by one hundred requires roughly ten thousand times as many draws. Brute-force Monte Carlo becomes expensive quickly when each function evaluation involves a PDE solve, a stochastic simulator, a financial portfolio revaluation, a reliability model or another expensive forward computation.

The rate also hides the constant

$$
\sigma_f.
$$

Two estimators can both converge at rate

$$
N^{-1/2}
$$

while one has a variance one hundred times smaller. At equal cost, its standard error is ten times smaller. Equivalently, the high-variance estimator needs one hundred times as many effective evaluations to match it.

For this reason, the useful comparison is not simply asymptotic rate. It is variance at a fixed computational budget, or more generally a cost-normalised measure such as

$$
\text{efficiency}
\propto
\frac{1}{
\text{cost}\times\operatorname{Var}(\hat\mu)
}.
$$

A variance-reduction method that halves variance while doubling evaluation cost has achieved nothing at equal budget. A method that reduces variance by a factor of thirty while using the same number of function evaluations has changed the computational problem substantially.

## Control variates use a quantity whose expectation is already known

The cleanest variance-reduction argument begins with correlation. Suppose $Y=f(X)$ has unknown mean $\mu$, while another random variable $Z$, generated from the same draw, has known expectation

$$
\mathbb E[Z]
=
m_Z.
$$

For any constant $c$,

$$
\hat\mu_c
=
\bar Y
-
c(\bar Z-m_Z)
$$

remains unbiased because

$$
\mathbb E[\bar Z-m_Z]
=
0.
$$

Its single-draw variance is

$$
\operatorname{Var}
\left[
Y-c(Z-m_Z)
\right]
=
\operatorname{Var}(Y)
+
c^2\operatorname{Var}(Z)
-
2c\operatorname{Cov}(Y,Z).
$$

Minimising with respect to $c$ gives

$$
c^\star
=
\frac{
\operatorname{Cov}(Y,Z)
}{
\operatorname{Var}(Z)
}.
$$

At this value,

$$
\operatorname{Var}(Y-c^\star Z)
=
\operatorname{Var}(Y)
\left(
1-\rho_{YZ}^2
\right),
$$

where $\rho_{YZ}$ is the correlation between $Y$ and $Z$. A control variate therefore converts correlation with a known quantity directly into variance reduction.

An exact Gaussian example makes the gain visible. Let

$$
X\sim N(0,1)
$$

and suppose the target is

$$
\mu
=
\mathbb E[X^4]
=
3.
$$

The naive Monte Carlo variable is

$$
Y=X^4.
$$

Using Gaussian moments,

$$
\mathbb E[X^4]
=
3,
\qquad
\mathbb E[X^8]
=
105,
$$

so

$$
\operatorname{Var}(Y)
=
105-3^2
=
96.
$$

A natural control variate is

$$
Z=X^2,
$$

because

$$
\mathbb E[Z]
=
1
$$

is known exactly. Its variance is

$$
\operatorname{Var}(Z)
=
\mathbb E[X^4]-1
=
2,
$$

and

$$
\operatorname{Cov}(X^4,X^2)
=
\mathbb E[X^6]
-
\mathbb E[X^4]\mathbb E[X^2]
=
15-3
=
12.
$$

The optimal coefficient is therefore

$$
c^\star
=
\frac{12}{2}
=
6.
$$

The controlled variable is

$$
X^4
-
6(X^2-1),
$$

and its variance is

$$
96
-
\frac{12^2}{2}
=
24.
$$

The estimator has exactly the same expectation as the naive estimator and one quarter of its variance. With $N$ draws,

$$
\operatorname{Var}(\hat\mu_{\text{naive}})
=
\frac{96}{N},
$$

while

$$
\operatorname{Var}(\hat\mu_{\text{control}})
=
\frac{24}{N}.
$$

The naive method therefore needs four times as many samples to obtain the same variance.

This example is modest compared with many practical applications because the correlation is not extremely close to one. In simulation models, useful control variates often come from simplified analytic approximations, conservation laws, lower-fidelity simulators or quantities whose expectation is available by symmetry. When a control tracks most of the variation in the expensive output, the gain can be dramatic.

The central principle is that randomness in a Monte Carlo estimator is not automatically useful. If part of that randomness is shared with a quantity whose mean is already known, it can be subtracted rather than averaged away slowly through larger $N$.

## Deliberate dependence can reduce variance

Independent sampling is convenient, not sacred. Negative dependence can make an average more stable.

Let

$$
U\sim U(0,1)
$$

and consider

$$
\mu
=
\mathbb E[e^U]
=
e-1.
$$

One ordinary estimator evaluates $e^U$ at independent uniforms. Antithetic sampling uses each draw together with its reflection,

$$
1-U.
$$

The pair estimator is

$$
A(U)
=
\frac{
e^U+e^{1-U}
}{2}.
$$

Because both $U$ and $1-U$ are uniformly distributed,

$$
\mathbb E[A(U)]
=
e-1.
$$

The two function evaluations are negatively correlated. When $U$ is large, $1-U$ is small, and vice versa.

For one ordinary evaluation,

$$
\operatorname{Var}(e^U)
=
\frac{e^2-1}{2}
-
(e-1)^2
=
-\frac{e^2}{2}
-\frac32
+
2e
\approx
0.24204.
$$

For one antithetic pair,

$$
\operatorname{Var}[A(U)]
=
-\frac{3e^2}{4}
-\frac54
+
\frac{5e}{2}
\approx
0.0039125.
$$

A fair comparison should use the same number of function evaluations. With $2N$ independent evaluations, ordinary Monte Carlo has variance

$$
\frac{
0.24204
}{
2N
}.
$$

With $N$ antithetic pairs, also requiring $2N$ evaluations, the variance is

$$
\frac{
0.0039125
}{
N
}.
$$

The variance ratio is therefore approximately

$$
\frac{
0.24204/(2N)
}{
0.0039125/N
}
\approx
30.9.
$$

At equal evaluation cost, antithetic coupling is more than thirty times as efficient in this example.

There is no universal guarantee that antithetic sampling helps. The construction works when the two evaluations tend to move in opposite directions around the target. For monotone functions of a scalar uniform variable, the reflection $U\mapsto1-U$ often induces useful negative correlation. For irregular or non-monotone functions, the covariance may be weak or even positive. The right diagnostic is the covariance of the paired evaluations, not the fact that the method is called antithetic.

Stratified sampling exploits a related idea by forcing the sample to cover parts of the domain deliberately rather than trusting random allocation. If the integration domain is partitioned into strata $S_h$ with probabilities $p_h$, then

$$
\mu
=
\sum_h
p_h
\mathbb E[
f(X)\mid X\in S_h
].
$$

Estimating each conditional expectation separately prevents random overrepresentation of some strata and underrepresentation of others. With suitable allocation, variance can fall substantially when outcomes are relatively homogeneous within strata.

Conditional Monte Carlo goes further. If $Y$ is a noisy estimator and $Z$ is a useful conditioning variable, replace $Y$ by

$$
\mathbb E[Y\mid Z]
$$

whenever that conditional expectation can be computed. The law of total variance gives

$$
\operatorname{Var}(Y)
=
\mathbb E[
\operatorname{Var}(Y\mid Z)
]
+
\operatorname{Var}
\left(
\mathbb E[Y\mid Z]
\right).
$$

Therefore,

$$
\operatorname{Var}
\left(
\mathbb E[Y\mid Z]
\right)
\le
\operatorname{Var}(Y).
$$

This is the variance-reduction version of Rao-Blackwellisation. Randomness that can be integrated out analytically should not be simulated merely because a simulator can generate it.

## Rare events expose the limits of brute-force simulation

Variance reduction becomes essential when the target itself is rare. Consider

$$
Z\sim N(0,1)
$$

and the probability

$$
p
=
P(Z>6).
$$

The exact value is approximately

$$
p
=
9.8659\times10^{-10}.
$$

A naive Monte Carlo estimator is

$$
\hat p_N
=
\frac1N
\sum_{i=1}^N
\mathbf 1\{Z_i>6\}.
$$

This is simply a Bernoulli average. Its variance is

$$
\operatorname{Var}(\hat p_N)
=
\frac{
p(1-p)
}{
N
}.
$$

For rare events, absolute standard error is less informative than relative error. The relative standard error is approximately

$$
\frac{
\sqrt{
\operatorname{Var}(\hat p_N)
}
}{
p
}
\approx
\frac{
1
}{
\sqrt{Np}
}.
$$

To obtain relative standard error near 10%, we need roughly

$$
Np
\approx
100,
$$

so

$$
N
\approx
\frac{
100
}{
9.8659\times10^{-10}
}
\approx
1.01\times10^{11}.
$$

That is about one hundred billion independent standard-normal draws for an estimator whose relative uncertainty is still around ten percent.

The problem is not that normal random numbers are expensive. The problem is that almost every simulated observation contributes exactly zero. At

$$
N=10^8,
$$

the expected number of six-sigma exceedances is only about

$$
0.099.
$$

The probability of observing no exceedance at all is approximately

$$
(1-p)^N
\approx
e^{-Np}
\approx
0.906.
$$

A simulation containing one hundred million draws will therefore return the estimate zero about ninety percent of the time.

Importance sampling changes the distribution from which simulations are drawn. Let the target density be the standard-normal density $\phi(x)$, but simulate instead from

$$
q(x)
=
\phi(x-6),
$$

the density of

$$
X\sim N(6,1).
$$

The probability can be written exactly as

$$
p
=
\mathbb E_q
\left[
\mathbf 1\{X>6\}
\frac{
\phi(X)
}{
q(X)
}
\right].
$$

The likelihood ratio is

$$
\frac{
\phi(x)
}{
\phi(x-6)
}
=
\exp(-6x+18).
$$

The importance-sampling estimator is therefore

$$
\hat p_N^{\mathrm{IS}}
=
\frac1N
\sum_{i=1}^N
\mathbf 1\{X_i>6\}
\exp(-6X_i+18),
$$

with

$$
X_i\sim N(6,1).
$$

The estimator remains unbiased, but the event now occurs with probability one half under the proposal rather than probability $10^{-9}$.

Its second moment can be calculated analytically:

$$
\mathbb E_q
\left[
\mathbf 1\{X>6\}
\left(
\frac{p(X)}{q(X)}
\right)^2
\right]
=
e^{36}
\bar\Phi(12)
\approx
7.6588\times10^{-18}.
$$

Since

$$
p^2
\approx
9.7335\times10^{-19},
$$

the relative variance of one importance-sampling draw is

$$
\frac{
7.6588\times10^{-18}
}{
p^2
}
-
1
\approx
6.8685.
$$

The relative standard error of the sample mean is therefore

$$
\operatorname{RSE}_{\mathrm{IS}}
\approx
\sqrt{
\frac{
6.8685
}{
N
}
}
=
\frac{
2.621
}{
\sqrt N
}.
$$

For 10% relative error, this requires roughly

$$
N
\approx
\left(
\frac{2.621}{0.1}
\right)^2
\approx
687
$$

draws.

The contrast is not subtle.

| Method | Approximate draws for 10% relative standard error |
| --- | ---: |
| Naive Monte Carlo | $1.01\times10^{11}$ |
| Shifted importance sampling $N(6,1)$ | $687$ |

Both estimators target exactly the same probability. Both are unbiased. Both ultimately obey a square-root law. The difference lies in the variance constant.

This is why statements such as "Monte Carlo converges at $N^{-1/2}$" are mathematically correct and computationally incomplete. The exponent of $N$ does not tell us whether the constant multiplying it is $10^{-3}$, $1$, or $10^9$.

The zero-variance importance distribution would sample directly from the target distribution conditional on the rare event,

$$
q^\star(x)
=
p(x\mid Z>6).
$$

If this proposal were available, every weighted sample would equal the desired probability and the estimator would have zero variance. Of course, constructing the exact conditional law usually requires knowledge as difficult as the original problem. The theoretical optimum is still useful because it reveals what a good practical proposal should imitate: it should concentrate probability where the integrand contributes most.

## Importance sampling can also be much worse than naive Monte Carlo

Changing measure does not automatically improve an estimator. For

$$
\mu
=
\int f(x)p(x)\,dx,
$$

importance sampling from $q$ uses

$$
Y
=
f(X)
\frac{
p(X)
}{
q(X)
},
\qquad
X\sim q.
$$

Its variance is finite only if

$$
\int
f(x)^2
\frac{
p(x)^2
}{
q(x)
}
\,dx
<
\infty.
$$

This condition explains why proposal tails matter. If $q(x)$ becomes too small in regions where $f(x)p(x)$ remains important, the importance weights

$$
w(x)
=
\frac{
p(x)
}{
q(x)
}
$$

can become enormous. A few observations then dominate the estimate. The method can have much higher variance than naive Monte Carlo and, in extreme cases, infinite variance even though the target expectation itself is finite.

Support is an even more basic requirement. If

$$
q(x)=0
$$

on a region where

$$
f(x)p(x)\ne0,
$$

then the proposal never visits part of the integral. No reweighting can recover a region that is assigned zero sampling probability.

This is why stable importance sampling should be assessed through its weights as well as its final estimate. A common descriptive diagnostic is the effective sample size,

$$
N_{\mathrm{eff}}
=
\frac{
\left(
\sum_i w_i
\right)^2
}{
\sum_i w_i^2
},
$$

or, for normalised weights $\tilde w_i$,

$$
N_{\mathrm{eff}}
=
\frac{
1
}{
\sum_i\tilde w_i^2
}.
$$

This quantity is only a heuristic summary and should not be confused with a universal variance identity. It is nevertheless useful for exposing weight concentration. Ten thousand simulated points can carry the information of only a few observations when almost all total weight is assigned to a tiny subset.

Self-normalised importance sampling,

$$
\hat\mu_{\mathrm{SN}}
=
\frac{
\sum_i
w_i f(X_i)
}{
\sum_i w_i
},
$$

is useful when the target density is known only up to a normalising constant. Unlike the basic importance-sampling estimator, it is generally biased at finite $N$, although it can be consistent under appropriate conditions. The distinction illustrates a broader principle: Monte Carlo design is not limited to choosing between unbiased estimators. A small controlled bias can sometimes be worthwhile if it reduces variance or makes the computation possible, but the trade-off should be stated explicitly.

## Variance reduction should be evaluated at equal cost

The previous examples counted function evaluations because that is the dominant cost in many simulations. Real Monte Carlo algorithms have more complicated cost structures. A control variate may require an additional model evaluation. A stratified design may require preprocessing. Importance sampling may need optimisation to build the proposal. Conditional Monte Carlo may replace simulation with an expensive numerical integral.

The relevant comparison is therefore not variance per draw but variance per unit of computation. If method $A$ costs $c_A$ per independent replication and has single-replication variance $v_A$, while method $B$ has cost $c_B$ and variance $v_B$, then at a total budget $C$,

$$
\operatorname{Var}(\hat\mu_A)
\approx
\frac{
c_Av_A
}{
C
},
$$

and similarly for $B$. The product

$$
c_Av_A
$$

is a natural first-order measure of inefficiency.

This matters in multi-fidelity simulation. Suppose a high-fidelity model is expensive and a lower-fidelity approximation is cheap but highly correlated with it. The cheap model can be used as a control variate. A large number of low-fidelity evaluations can then reduce the variance of a much smaller number of high-fidelity simulations. The optimal allocation depends on both correlations and relative costs.

The same logic appears in nested Monte Carlo, where an outer simulation calls an inner simulation. Increasing inner accuracy uniformly can be spectacularly wasteful if many outer states contribute little to the final quantity. Adaptive allocation can devote computation where conditional variance is large or where the decision boundary is uncertain.

Monte Carlo computation should therefore be budgeted around variance contribution, not around a fixed number of samples at every stage.

## Standard errors are part of the result

A simulation estimate without a Monte Carlo error assessment is incomplete. For independent finite-variance draws, the estimated MCSE

$$
\frac{s_f}{\sqrt N}
$$

is straightforward. A simulation can stop when the MCSE falls below a scientifically meaningful absolute or relative tolerance rather than after an arbitrary round number of iterations.

The stopping criterion should be connected to the scale of the decision. If the estimate is approximately $1000$, an MCSE of $0.01$ may be pointless if model uncertainty is $50$. Conversely, a simulation standard error of $10$ may be unacceptable if the difference between two policies is expected to be $5$.

Monte Carlo uncertainty is not the same as uncertainty in the underlying scientific model. If a posterior mean is estimated by simulation, the posterior uncertainty describes uncertainty about the model parameter conditional on the model and data, while the MCSE describes numerical uncertainty from approximating that posterior expectation with finite simulation. These two layers should not be conflated.

Variance estimation also becomes more difficult when draws are dependent. In Markov chain Monte Carlo,

$$
X_1,X_2,\ldots
$$

are intentionally correlated. For a stationary chain, the asymptotic variance of the sample mean can often be written as

$$
\operatorname{Var}(\bar f_N)
\approx
\frac{
\sigma_f^2
}{
N
}
\tau_{\mathrm{int}},
$$

where

$$
\tau_{\mathrm{int}}
=
1
+
2
\sum_{k=1}^{\infty}
\rho_k
$$

is an integrated autocorrelation time. Positive serial correlation reduces the effective information in $N$ iterations. This motivates the familiar approximation

$$
N_{\mathrm{eff}}
\approx
\frac{
N
}{
\tau_{\mathrm{int}}
}.
$$

The dependent case reinforces the main theme rather than changing it. Raw iteration count is not a measure of accuracy. Variance is.

## The square-root law has assumptions

The classical central limit theorem requires finite second moments and appropriate independence or weak-dependence conditions. Monte Carlo estimators can violate these assumptions.

If

$$
\operatorname{Var}[f(X)]
=
\infty,
$$

the usual

$$
N^{-1/2}
$$

standard error need not apply. Heavy-tailed simulation outputs can produce estimates dominated by rare enormous contributions, with sample variance itself unstable. Running longer may create the appearance of convergence for a while and then change the estimate dramatically when a previously unseen tail event occurs.

This is especially relevant in importance sampling because poor weight distributions can create infinite second moments even when the original integrand has finite variance. Weight diagnostics, tail analysis and proposal sensitivity are therefore part of numerical reliability, not optional extras.

High dimension creates a different limitation. Plain Monte Carlo's nominal convergence rate does not contain dimension explicitly, which is often contrasted favourably with deterministic grids. The variance constant can still deteriorate badly with dimension, and importance proposals can become mismatched in exponentially many directions. A dimension-free exponent does not imply dimension-free computational difficulty.

Quasi-Monte Carlo methods attack integration error differently by replacing independent random points with low-discrepancy sequences. Under smoothness and effective-dimension conditions, they can outperform the classical square-root rate. Randomised quasi-Monte Carlo restores a form of error assessment while retaining much of the low-discrepancy advantage. This is a distinct subject deserving its own treatment; it is enough here to note that $N^{-1/2}$ is not a universal computational lower bound for numerical integration.

## Monte Carlo is an estimator-design problem

The examples lead to a consistent conclusion. Ordinary Monte Carlo estimated

$$
\mathbb E[X^4]
$$

with single-draw variance $96$. A control variate reduced that variance to $24$ without changing the expectation. Antithetic sampling estimated

$$
\mathbb E[e^U]
$$

more than thirty times as efficiently as independent sampling at equal evaluation cost. For the six-sigma Gaussian tail, naive simulation required about $10^{11}$ draws for 10% relative precision, while a simple shifted importance sampler required fewer than $10^3$.

None of these improvements changed the quantity being estimated. They changed the random variable used to estimate it.

That is the most useful way to think about Monte Carlo methods. The problem is not merely to draw random numbers from the most obvious distribution and average them. The problem is to construct an estimator whose expectation is correct, whose variance is controlled, whose computational cost is acceptable, and whose error can be measured.

Sometimes the best method is indeed ordinary sampling. It is simple, parallel, auditable and surprisingly robust. When function evaluations are cheap and the variance is moderate, elaborate variance reduction can add complexity without practical benefit.

When simulation is expensive or the target is rare, the estimator itself becomes the main numerical object. Known expectations can become control variates. Symmetry can create antithetic pairs. Conditioning can remove unnecessary randomness. Stratification can prevent inefficient allocation. A change of measure can move computation into a rare region that naive simulation almost never visits.

The $N^{-1/2}$ law remains true for a large class of these estimators. It simply does not tell us how large $N$ needs to be.

That number is controlled by variance.

## References

Bucklew, J. A. (2004). *Introduction to Rare Event Simulation*. Springer.

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

Hammersley, J. M., & Handscomb, D. C. (1964). *Monte Carlo Methods*. Methuen.

Kahn, H., & Marshall, A. W. (1953). Methods of reducing sample size in Monte Carlo computations. *Journal of the Operations Research Society of America*, 1(5), 263–278.

Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer.

Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*.

Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.

Rubinstein, R. Y., & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method* (3rd ed.). Wiley.
