---
author_profile: false
categories:
- Statistics
classes: wide
title: 'Power Is a Surface, Not a Number'
excerpt: A worked power calculation shows how effect size, variance, sample size, and stopping rules change what a study can detect.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- statistical power
- experimental design
- minimum detectable effect
- sequential testing
- sensitivity analysis
seo_title: 'Power Is a Surface, Not a Number'
seo_description: 'Calculate power across effect sizes and variances, distinguish detectable from worthwhile effects, and check the consequences of interim testing.'
seo_type: article
summary: 'An exact Gaussian calculation and a reproducible stopping-rule simulation turn a single power claim into an auditable design analysis.'
tags:
- Statistical Power
- Experimental Design
- Simulation
- Statistical Inference
why_this_exists: 'A single sample-size answer hides the assumptions that make it useful. This article exposes those assumptions with calculations readers can rerun.'
evidence: 'Original Gaussian power tables and 100,000 simulated null experiments with four interim looks; all data are synthetic.'
methodology: 'Derive the two-sided known-variance power function, vary effect and noise separately, and compare fixed, unadjusted sequential, and Bonferroni decision rules.'
reviewed_at: 2026-09-18
---

<!--
Development contract
Question: What must be specified before an 80 percent power claim is meaningful?
Claim: Power is conditional on an effect, nuisance parameters, design, and complete decision rule.
Counterclaim: A single planning value can be adequate when its assumptions are explicit and credible.
Evidence object: Exact Gaussian derivation, numerical sensitivity tables, and a null simulation.
Failure case: Known-variance independent Gaussian calculations do not certify clustered, adaptive, or misspecified analyses.
Reader payoff: Specify a range of operating characteristics before committing to a sample size.
Exclusions: A catalogue of sample-size formulas, clinical design guidance, and implementation of advanced sequential boundaries.
-->

An experiment has 200 observations in each arm. Its planning document says that it has more than 80 percent power.

For a mean difference of 0.3 and a common standard deviation of 1, a conventional Gaussian calculation supports that statement: power is approximately 85 percent. If the standard deviation is 1.5, power falls to 52 percent. If the effect is 0.1, it falls to 17 percent.

The sample size did not change. The question being asked of the design did.

“80 percent power” is an incomplete sentence until we specify the alternative, the variability, the sampling design, and the rule that will declare a result. A useful planning calculation shows how the answer moves when those assumptions move.

## Define the repeated experiment

Let $\theta$ denote the effect of interest, $\eta$ the remaining parameters of the data-generating process, $\mathcal D$ the design, and $\mathcal A$ the analysis and decision rule. The rejection probability is

$$
\pi(\theta,\eta,n,\alpha,\mathcal D,\mathcal A)
=P_{\theta,\eta,\mathcal D}\{\mathcal A(\text{data})\text{ rejects }H_0\}.
$$

Under an alternative, this is power. Under the null, it is the false-positive probability, which a valid level-$\alpha$ procedure must control over the relevant null parameter space.

The probability refers to repeated datasets generated under the stated conditions. It is not the probability that the alternative is true, the probability that a particular estimate is accurate, or a guarantee that the effect matters.

The decision rule includes the details that change the event inside the probability: one or two tails, adjustment for multiple comparisons, interim looks, exclusion rules, and any selection of the reported analysis.

## A calculation we can audit exactly

Suppose two independent arms each contain $n$ independent observations:

$$
X_i\sim N(\mu_X,\sigma^2),\qquad
Y_i\sim N(\mu_Y,\sigma^2).
$$

Assume that $\sigma$ is known. Define $\Delta=\mu_Y-\mu_X$ and test $H_0:\Delta=0$ with

$$
Z=\frac{\bar Y-\bar X}{\sigma\sqrt{2/n}}.
$$

Under the alternative, $Z\sim N(\lambda,1)$, where

$$
\lambda=\frac{\Delta}{\sigma}\sqrt{\frac n2}.
$$

For a two-sided level-$\alpha$ test, reject when $|Z|>c$, with $c=\Phi^{-1}(1-\alpha/2)$. Therefore

$$
\pi(\Delta,n,\sigma,\alpha)
=\Phi(-c-\lambda)+\Phi(\lambda-c).
$$

Both tails appear. At $\Delta=0$, the expression returns $\alpha$; changing the sign of $\Delta$ leaves power unchanged. These are useful checks on an implementation.

The formula is exact for this known-variance Gaussian model. If the actual analysis estimates variance and uses a t statistic, its finite-sample power is different. A clustered experiment or a heavy-tailed outcome may require a substantially different calculation.

## Read a surface instead of one cell

At $\alpha=0.05$ and $\sigma=1$, the formula gives:

| Observations per arm | $\Delta=0.1$ | $\Delta=0.2$ | $\Delta=0.3$ | $\Delta=0.5$ |
| --- | --- | --- | --- | --- |
| 50 | 7.9% | 17.0% | 32.3% | 70.5% |
| 100 | 10.9% | 29.3% | 56.4% | 94.2% |
| 200 | 17.0% | 51.6% | 85.1% | 99.9% |
| 400 | 29.3% | 80.7% | 98.9% | >99.9% |

Each column traces a sample-size curve. Each row traces an effect-size curve. A design that reliably detects a large effect can still tell us very little about a small one.

Now hold the effect at $\Delta=0.3$ and the sample size at 200 per arm:

| Assumed standard deviation | Power |
| --- | --- |
| 0.75 | 97.9% |
| 1.00 | 85.1% |
| 1.50 | 51.6% |

These are different planning scenarios in the outcome's original units. Writing only the standardized effect $d=\Delta/\sigma$ would conceal whether an optimistic assumption came from a large effect or a small noise level.

The following standard-library Python code reproduces both tables. The smallest integer sample size is found by evaluating the full two-tail expression.

```python
from math import ceil, sqrt
from statistics import NormalDist

normal = NormalDist()

def power(delta, n, sigma=1.0, alpha=0.05):
    noncentrality = delta / (sigma * sqrt(2 / n))
    critical = normal.inv_cdf(1 - alpha / 2)
    return (normal.cdf(-critical - noncentrality)
            + normal.cdf(noncentrality - critical))

for n in (50, 100, 200, 400):
    print(n, [f"{power(d, n):.3f}" for d in (0.1, 0.2, 0.3, 0.5)])

for sigma in (0.75, 1.0, 1.5):
    print("sigma", sigma, "power", f"{power(0.3, 200, sigma):.3f}")

target = 0.80
delta = 0.30
approx_n = ceil(2 * (normal.inv_cdf(0.975)
                     + normal.inv_cdf(target))**2 / delta**2)
exact_n = next(n for n in range(1, approx_n + 1)
               if power(delta, n) >= target)
print("observations per arm", exact_n)
print("power with alpha=0.01", f"{power(0.3, 200, alpha=0.01):.3f}")
```

The calculation gives 175 observations per arm for the specified 0.3 effect. That is a conditional design answer, not a general recommendation to collect 350 observations.

## Detectable and worthwhile are separate judgments

For a positive effect, ignoring the small opposite-tail rejection probability gives the familiar approximation

$$
\Delta_{\mathrm{MDE}}
\approx
\left(z_{1-\alpha/2}+z_{1-\beta}\right)
\sigma\sqrt{\frac2n}.
$$

At 200 observations per arm, $\sigma=1$, $\alpha=0.05$, and target power 0.8, this gives approximately 0.280.

The minimum detectable effect is the effect at which the chosen design reaches the target rejection probability. It is not a sharp boundary below which detection is impossible. The first table contains nonzero power below that value.

Nor does the calculation determine the smallest effect worth acting on. That threshold comes from the application: the cost of changing a process, the scale of the outcome, and the consequences of a wrong decision.

If an improvement of 0.1 matters but the feasible design only reaches 80 percent power at 0.28, the gap belongs in the planning decision. Increasing the assumed effect until the sample-size calculator returns an affordable number does not resolve it.

## The same number of rows can carry different information

The independent-arm calculation uses

$$
\operatorname{Var}(\bar Y-\bar X)=\frac{2\sigma^2}{n}.
$$

With $n$ genuine pairs, equal marginal variances, and within-pair correlation $\rho$, the variance of the average paired difference is instead

$$
\operatorname{Var}(\bar D)=\frac{2\sigma^2(1-\rho)}n.
$$

Positive correlation can make a paired design more informative at the same number of measurements. It also introduces another planning assumption: the correlation must be plausible for the population and measurement protocol.

Clustering can move the calculation in the opposite direction. Multiple observations from one participant or site do not usually carry the same information as the same number of independent participants or sites. A power calculation must preserve the dependence that the final analysis will encounter.

The threshold matters too. In the worked example, changing $\alpha$ from 0.05 to 0.01 lowers power from 85.1 to 66.4 percent. If the final analysis uses a stricter threshold because it tests several endpoints, a calculation using 0.05 for every endpoint describes a different rule.

## A stopping rule changes the experiment

Consider a separate, deliberately simple simulation. Independent standardized differences satisfy $D_i\sim N(0,1)$ under the null. We inspect the cumulative mean after 100, 200, 300, and 400 observations, using the known variance.

Compare three rules:

1. Test only at observation 400 with a two-sided threshold of 0.05.
2. Stop and reject whenever any of the four ordinary p-values is below 0.05.
3. Stop and reject whenever any p-value is below $0.05/4$.

The third rule uses a Bonferroni allocation across the four prespecified looks. The union bound controls its probability of any false rejection at 0.05, regardless of dependence between looks. It is a simple conservative demonstration, rather than an efficient sequential design.

```python
import numpy as np
from statistics import NormalDist

rng = np.random.default_rng(20260918)
normal = NormalDist()
replications = 100_000
looks = np.array([100, 200, 300, 400])

# A block sum of 100 independent N(0, 1) values is N(0, 100).
increments = rng.normal(0, np.sqrt(100), size=(replications, 4))
z = np.cumsum(increments, axis=1) / np.sqrt(looks)

ordinary = normal.inv_cdf(1 - 0.05 / 2)
adjusted = normal.inv_cdf(1 - (0.05 / 4) / 2)
rules = {
    "fixed": np.abs(z[:, -1]) > ordinary,
    "four unadjusted looks": (np.abs(z) > ordinary).any(axis=1),
    "four Bonferroni looks": (np.abs(z) > adjusted).any(axis=1),
}

for name, rejected in rules.items():
    rate = rejected.mean()
    mcse = np.sqrt(rate * (1 - rate) / replications)
    print(name, f"rate={rate:.5f}", f"MCSE={mcse:.5f}")
```

With NumPy 2.3.5, the run gives:

| Decision rule | False-positive rate | Monte Carlo standard error |
| --- | --- | --- |
| Fixed final analysis | 4.924% | 0.068 percentage points |
| Four unadjusted looks | 12.549% | 0.105 percentage points |
| Four Bonferroni looks | 3.522% | 0.058 percentage points |

These are null rejection rates, not power estimates. They establish why a power comparison would be misleading before checking calibration. A rule that rejects more often because it allows more false positives has changed the error budget.

The simulation evaluates all four looks to determine whether a crossing ever occurs. Stopping at the first crossing gives the same rejection event. Expected sample size under stopping is a separate quantity and is not reported here.

More efficient procedures can account directly for repeated observation. Confidence sequences, for example, provide coverage simultaneously over time under their stated assumptions. That guarantee belongs to a specifically constructed procedure; it does not extend ordinary fixed-sample intervals automatically. [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*](https://arxiv.org/abs/1810.08240).

For a sequential design, evaluate null error, power across alternatives, expected sample size, and the distribution of stopping times under the actual stopping and analysis rules. The site's [worked optional-stopping example](/statistics/optional_stopping_sequential_experiment_tests/) develops that problem further.

## What to do when the inputs are uncertain

A sensitivity grid does not eliminate uncertainty about the effect or variance. It makes that uncertainty visible.

There are several defensible planning summaries. We can report performance at a central scenario and at plausible unfavorable scenarios. We can require target power throughout a specified parameter range. Or, given an explicitly justified planning distribution $p(\theta,\eta)$, we can average:

$$
\operatorname{Assurance}(n)
=\int \pi(\theta,\eta,n)\,p(\theta,\eta)\,d\theta\,d\eta.
$$

This average is often called assurance. It answers a different question from power at one fixed alternative and inherits the assumptions of the planning distribution. Calling uncertain inputs a distribution does not make them well supported.

The same caution applies to using a noisy pilot effect as if it were the population effect. External evidence and scientifically relevant effect ranges are usually more informative than one convenient point estimate.

After an experiment, plugging its observed effect into a power formula does not independently explain a nonsignificant result. Report the estimate and interval, and assess what effects the design could reasonably distinguish. Design analysis using external effect assumptions remains useful; Gelman and Carlin extend it to the probability of sign errors and exaggeration among significant estimates. [Gelman and Carlin, *Beyond Power Calculations*](https://journals.sagepub.com/doi/10.1177/1745691614551642).

## A design statement that someone else can review

A useful planning statement would say:

> Under independent Gaussian sampling with known standard deviation 1, 200 observations per arm give 85.1 percent power to detect a mean difference of 0.3 using one two-sided test at 0.05. Power falls to 51.6 percent if the standard deviation is 1.5. The analysis has no interim rejection rule.

That statement can be challenged. Someone can dispute the effect, the noise assumption, the independence, or the stopping rule and immediately see which calculation needs to change.

For an actual study, add the smallest worthwhile effect, the source of the planning inputs, the proposed treatment of missing data, and the precision required for the final estimate. If the scientific goal is estimation or an operational decision, rejection probability may be only one design criterion.

A single power value can be an adequate summary once those choices are settled. The surface is what lets us decide whether the summary is trustworthy.
