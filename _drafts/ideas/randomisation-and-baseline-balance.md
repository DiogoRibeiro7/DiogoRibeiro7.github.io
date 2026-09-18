---
title: 'What Randomisation Promises About Baseline Balance'
permalink: /research/randomisation_and_baseline_balance/
author_profile: false
classes: wide
categories:
- Research
tags:
- Experimental Design
- Randomisation
- Covariate Adjustment
- Statistical Inference
excerpt: 'An exact twenty-unit experiment separates chance imbalance, biased assignment, and the precision gained from a useful baseline covariate.'
seo_title: 'What Randomisation Promises About Baseline Balance'
seo_description: 'Enumerate baseline imbalance in a randomised experiment and calculate when prespecified covariate adjustment improves precision.'
seo_type: article
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
why_this_exists: 'A baseline table can look unbalanced even when allocation is correct. Exact enumeration makes that distinction visible and connects it to an analysis decision.'
evidence: 'Original finite-population calculations for twenty synthetic units, an adjustment sensitivity analysis, and CONSORT reporting guidance.'
methodology: 'Count every allocation by its binary-covariate composition, then derive the randomisation variance of an estimator with a fixed adjustment coefficient.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: What should an analyst infer from unequal baseline characteristics after random allocation?
Claim: Randomisation supplies a known assignment mechanism; exact balance and high precision require additional design or analysis choices.
Counterclaim: A serious imbalance can matter for precision and can motivate checking the allocation process, especially alongside operational evidence.
Evidence object: Exact hypergeometric probabilities and a finite-population variance calculation with an externally fixed adjustment coefficient.
Failure case: The argument assumes the recorded assignment mechanism was followed and does not repair attrition, interference, or manipulated allocation.
Reader payoff: Read a baseline table without confusing chance imbalance with bias, and specify adjustment before seeing favourable results.
Exclusions: Clinical trial design advice, a catalogue of regression estimators, and a universal significance threshold for auditing randomisation.
-->

A small experiment assigns ten units to an intervention and ten to a control. Seven intervention units have a high value of a baseline characteristic, compared with three control units. The baseline table contains a forty percentage point difference.

One reader says that randomisation failed. Another says that randomisation makes the difference irrelevant. Both conclusions go beyond what the design establishes.

We can make the issue concrete without relying on a large simulation or an unexplained p-value. Suppose the experiment contains exactly ten units with the characteristic and ten without it. Select the ten intervention units uniformly from all possible groups of ten. The chance of a baseline difference of at least forty percentage points, in either direction, is about **17.9 percent**.

That difference is compatible with correctly implemented randomisation. Whether it matters for the outcome depends on how strongly the characteristic predicts that outcome. The allocation mechanism and the scientific importance of an observed imbalance are separate questions.

## Specify what was randomised

Label the twenty units before assignment. Write $X_i=1$ for the ten units with the baseline characteristic and $X_i=0$ for the other ten. These values are fixed throughout the calculation.

The random object is the assignment vector: exactly ten units receive treatment, and every possible set of ten is equally likely. This is complete randomisation with fixed arm sizes. There are

$$
\binom{20}{10}=184{,}756
$$

possible allocations.

This definition excludes several superficially similar designs. Independent coin flips do not guarantee ten units in each arm. Randomisation within strata restricts which allocations can occur. Randomising whole sites creates dependence among units at the same site. Each mechanism produces its own reference distribution.

Keeping the mechanism explicit prevents a common analytical mistake: treating any shuffled dataset as a valid representation of the experiment. A valid randomisation calculation reproduces the assignment rule actually used, including its restrictions.

Now let $K$ be the number of units with $X=1$ assigned to treatment. There are $\binom{10}{K}$ ways to choose those units and $\binom{10}{10-K}$ ways to fill the remaining treatment positions from the other group. Therefore

$$
P(K=k)=\frac{\binom{10}{k}\binom{10}{10-k}}{\binom{20}{10}},
\qquad k=0,\ldots,10.
$$

The treatment proportion is $K/10$, and the control proportion is $(10-K)/10$. Their difference is

$$
D_X=\bar X_T-\bar X_C=\frac{K-5}{5}.
$$

Every number in the opening example follows from this finite counting problem.

## Balance in expectation leaves room for imbalance

The distribution is symmetric around $K=5$, so $E[D_X]=0$. Across repeated assignments, neither arm systematically receives more units with the characteristic.

Exact balance requires $K=5$. Its probability is

$$
P(K=5)=\frac{\binom{10}{5}^2}{\binom{20}{10}}
\approx0.3437.
$$

Consequently, **65.6 percent of allocations have some imbalance** on this characteristic. A promise of balance in expectation is quite different from a promise that the realised table will contain matching columns.

The opening allocation has $K=7$, giving $D_X=0.4$. Counting both tails gives

$$
P(|D_X|\ge0.4)=P(K\le3)+P(K\ge7)\approx0.1789.
$$

This is an exact probability under the stated assignment mechanism. It is neither the probability that the researchers followed the protocol nor the probability that the intervention has no effect. Those are different propositions and require different information.

![Exact baseline-balance probabilities for twenty units and the standard deviation of a treatment-effect estimator as a fixed adjustment coefficient varies.](/assets/images/figures/research_randomisation_balance.png){: width="1785" height="665" loading="lazy"}

*Original synthetic calculation. The left panel enumerates complete randomisation. The right panel uses the fixed population and adjustment rule defined below; it does not fit a regression coefficient to each assignment.*

Small samples make the distinction especially visible, but increasing the sample size does not turn randomisation into a guarantee of identical groups. It generally reduces the scale of chance differences in averages. With enough measured characteristics, some table entries will still look unusually different.

## Connect imbalance to the outcome

A baseline difference matters differently when its variable is strongly prognostic, weakly prognostic, or unrelated to the outcome.

Consider a deliberately transparent potential-outcome model:

$$
Y_i(0)=10+4X_i+e_i,\qquad Y_i(1)=Y_i(0)+1.
$$

The intervention adds one outcome unit for every experimental unit. The finite-population average treatment effect is therefore exactly one. The baseline characteristic contributes four units, and $e_i$ represents additional fixed outcome variation.

These are synthetic outcome units, not a claim about a particular treatment or clinical measurement. The construction is useful because it keeps the causal effect visible while changing the composition of the observed groups.

The observed difference in arm means can be written as

$$
\widehat\tau=\bar Y_T-\bar Y_C
=1+4D_X+D_e.
$$

For the seven-versus-three allocation, the covariate contribution is $4(0.4)=1.6$. If the residual means happen to match, the observed treatment difference is 2.6 even though the effect on every unit is one.

This is sampling variation generated by random assignment. Over the full allocation distribution, both $D_X$ and $D_e$ have expectation zero, so $E[\widehat\tau]=1$. An unbiased estimator can still be quite far from its target in the single experiment we observe.

Change the prognostic coefficient from four to zero, and the same baseline table contributes nothing through $X$. Change it to eight, and the contribution doubles. Reading the baseline table without asking about outcome relevance misses the part that determines its practical consequence.

## Calculate what adjustment buys

Suppose an analyst specifies an adjustment coefficient $b$ using external information, before assignment and without adapting it to the observed results. Define

$$
\widehat\tau_b=(\bar Y_T-\bar Y_C)-b(\bar X_T-\bar X_C).
$$

Because $E[D_X]=0$, this estimator remains unbiased for the constant effect for any fixed $b$. Its precision depends on the remaining variation in $Y_i(0)-bX_i$.

For a fixed population of size $N$ and complete randomisation with arm sizes $n_T$ and $n_C$, the constant-effect construction gives

$$
\operatorname{Var}(\widehat\tau_b)
=\left(\frac1{n_T}+\frac1{n_C}\right)S_{Y(0)-bX}^2,
$$

where $S^2$ uses the finite-population denominator $N-1$. The expression already accounts for sampling without replacement: the two groups partition the same fixed population.

Complete the example by giving each ten-unit $X$ group the residual sequence

$$
(-2,-1,0,1,2,-2,-1,0,1,2).
$$

Each group has residual mean zero. Across all twenty units, the finite-population covariance between $X$ and $e$ is zero, $S_X^2=5/19$, and $S_e^2=40/19$. Hence

$$
\operatorname{Var}(\widehat\tau_b)
=0.2\left[(4-b)^2\frac5{19}+\frac{40}{19}\right].
$$

| Fixed coefficient $b$ | Standard deviation of estimated effect | Interpretation |
| --- | ---: | --- |
| 0 | 1.124 | Unadjusted comparison |
| 2 | 0.795 | Part of the prognostic contribution removed |
| 4 | 0.649 | Entire $X$ contribution removed in this construction |
| 8 | 1.124 | Overcorrection gives back the variance reduction |

The best coefficient in this particular population is four. Choosing two still helps. Choosing eight gives the same variance as ignoring the covariate, and choosing a sufficiently extreme coefficient makes precision worse.

The calculation makes a useful criterion explicit: adjustment helps when it reduces residual outcome variation. It does not help merely because a variable appears in a baseline table or has a small baseline-comparison p-value.

## Keep the inferential target visible

The uncertainty just calculated comes from assigning these twenty fixed units. It does not include uncertainty from recruiting a different set of units from a larger population. A claim about how the intervention would perform elsewhere needs an account of that additional step: who was eligible, who participated, and which features might change the response.

This distinction is easy to miss because both questions can be expressed using a treatment-effect average. An average for the enrolled units and an average for a target population may differ even when the allocation was flawless. Random allocation helps compare outcomes within the experiment; it does not make the enrolled sample representative of every setting to which a reader might want to generalise.

The constant effect in our example also removes a source of complexity. If effects vary across units, the unadjusted difference in means remains unbiased for the enrolled units' average effect under this complete-randomisation design. Its randomisation variance then depends on both potential-outcome distributions and their unit-level relationship. The simple residual-variance expression above used the stronger constant-effect construction.

Keeping these boundaries visible lets the example do useful work without turning it into an all-purpose guarantee. We have quantified one consequence of a specified assignment mechanism and a specified estimator. External validity and heterogeneous responses require further assumptions or evidence.

## A fitted coefficient is a different estimator

In most applications, the prognostic coefficient is unknown. Fitting a regression to the trial data replaces the fixed $b$ above with an estimated quantity that depends on the observed assignment and outcomes.

We cannot insert that random estimate into the fixed-coefficient variance formula and declare the uncertainty problem solved. Estimation of the coefficient, model form, interactions, treatment-effect heterogeneity, and the small-sample behaviour of the reported standard error all become relevant.

The example is an illustration of why useful adjustment can improve precision, not a proof that every adjusted regression has exact finite-sample unbiasedness or correct confidence intervals. Its clean algebra comes partly from the constant treatment effect and from treating the adjustment coefficient as fixed.

A credible analysis plan therefore specifies the covariates and how they enter the model, the effect being estimated, and the method for uncertainty estimation. It also states how missing baseline measurements will be handled. Otherwise, a nominally prespecified list can still leave substantial freedom in the final analysis.

This matters particularly when a fitted model contains many terms relative to the number of observations. A richly adjusted model may spend considerable information estimating nuisance relationships. The decision should follow the scientific role of the covariates and an appropriate precision analysis, rather than an assumption that more adjustment is automatically better.

## Why a baseline significance screen is unhelpful

A common workflow tests each baseline characteristic, adjusts for the ones with small p-values, and leaves the others out. This asks whether an observed imbalance is unusual under a reference model, then uses that answer as a substitute for whether the variable predicts the outcome.

Our calculation shows why the substitution is poor. A strong prognostic covariate can be useful even when its realised difference is small. A weak prognostic covariate can have an unusually large difference while contributing little to outcome variation. Selecting the model after inspecting the assignment also makes the eventual estimator depend on a preliminary decision rule.

CONSORT 2025 recommends reporting baseline characteristics without significance tests and interpreting chance differences in light of their magnitude and prognostic relevance. That reporting recommendation is consistent with the distinction demonstrated here. It is not an instruction to conceal an inconvenient table. [CONSORT 2025 explanation and elaboration, item 25](https://www.bmj.com/content/389/bmj-2024-081124)

Report the actual values, including distributions when means obscure important features. Explain the prespecified adjustment and identify exploratory alternatives as exploratory. A reader can then understand both the realised sample and the analysis without treating a baseline p-value as a certificate of successful randomisation.

## Design can constrain the imbalance directly

If balance on $X$ is important enough to enforce, change the allocation mechanism before the experiment starts. In this population, randomising five of the ten $X=1$ units and five of the ten $X=0$ units to treatment makes $D_X=0$ for every permitted allocation.

There are now

$$
\binom{10}{5}\binom{10}{5}=63{,}504
$$

permitted assignments. The design has deliberately excluded the other assignments. An analysis based on re-randomising labels should respect that restriction instead of recreating the original complete-randomisation distribution.

This guarantee applies to the chosen binary characteristic. It does not force residual outcome balance or equality on every unmeasured variable. Trying to enforce many constraints can also make the design difficult to operate or sharply reduce the set of allowable assignments.

The timing of the rule matters. A documented balance restriction applied before outcomes are observed defines a design. Discarding an inconvenient realised allocation after looking at outcomes introduces a selection process that the original probability calculation does not describe.

## The table cannot audit the entire experiment

An ordinary-looking baseline table does not prove that allocation was concealed or that the recorded assignments were implemented faithfully. Conversely, an unusual table does not, by itself, establish misconduct or a broken random-number generator.

If there are operational reasons for concern, inspect the assignment records, enrolment order, implementation logs, and any deviations from the protocol. The probability calculation can contextualise a specific anomaly, but the evidence needed to investigate the process extends beyond the table.

Random assignment also does not automatically resolve problems that arise later. Differential attrition can change who contributes outcomes. Interference can make one unit's outcome depend on other units' assignments. Selective reporting can hide the analyses that were attempted. None of these problems is repaired by showing that baseline averages are close.

The useful separation is between assignment, estimation, and observation. Specify the assignment mechanism; choose an estimator whose operating characteristics match it; and describe how the outcomes actually became available. A baseline table occupies only one part of that account.

## Reproduce and adapt the example

The accompanying [calculation and figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_coverage_draft_figures.py) prints the probabilities and variance comparisons with `--dry-run`. The probability function supports unequal group sizes and different covariate prevalences. Numerical tests compare its results with explicit enumeration in smaller populations, and compare the variance formula with the variance across every assignment.

The core probability needs only Python's standard library:

```python
from math import comb

denominator = comb(20, 10)
probabilities = [comb(10, k) * comb(10, 10-k) / denominator
                 for k in range(11)]
print(probabilities[5])  # 0.3437182013: exact balance
print(sum(p for k, p in enumerate(probabilities) if abs(k-5) >= 2))
# 0.1788954080: at least a 40 percentage point difference
```

For a new study, first replace the assignment mechanism, covariate distribution, and outcome relationship with plausible choices for that study. Then examine the distribution of the estimator you actually intend to report. The important output is an account of what the design and analysis make likely, including the uncomfortable allocations that a single reassuring average can hide.
