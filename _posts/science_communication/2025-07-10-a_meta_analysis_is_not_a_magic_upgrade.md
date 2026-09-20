---
permalink: '/science-communication/a_meta_analysis_is_not_a_magic_upgrade/'
title: 'A Meta-Analysis Is Not a Magic Upgrade'
date: '2025-07-10'
categories:
- Science Communication
tags:
- Meta Analysis
- Systematic Reviews
- Evidence Synthesis
- Heterogeneity
- Research Methods
author_profile: false
classes: wide
seo_title: 'A Meta-Analysis Is Not a Magic Upgrade'
seo_description: 'Meta-analysis can improve precision, but it cannot repair biased studies, incompatible estimands or selective evidence. Heterogeneity, prediction intervals and risk of bias determine what the pooled result means.'
seo_type: article
excerpt: >-
  A pooled estimate can be very precise and still be scientifically weak. The
  quality of a meta-analysis depends on the studies that entered it, the question
  they actually share, the heterogeneity between them and the assumptions used
  to combine their results.
summary: >-
  This article develops meta-analysis as a statistical synthesis rather than a
  universal top layer of evidence. A worked five-study example produces a
  fixed-effect pooled estimate of 0.43 with a narrow 95% interval, while severe
  heterogeneity gives I-squared above 93%, a much wider random-effects interval
  and a prediction interval spanning negative to strongly positive effects. The
  article then examines systematic review methods, risk of bias, effect
  harmonisation, publication bias, small-study effects, random-effects models,
  prediction intervals, meta-regression, sensitivity analysis and current PRISMA
  and Cochrane guidance.
keywords:
- meta analysis
- systematic review
- heterogeneity
- prediction interval
- publication bias
- evidence synthesis
why_this_exists: >-
  Meta-analysis is frequently treated in public discussion as though pooling
  automatically upgrades weak or inconsistent evidence into a definitive answer.
  This article shows what pooling actually does mathematically and why study
  selection, estimand compatibility, bias and heterogeneity remain part of the
  conclusion.
evidence: >-
  An original five-study numerical example, current Cochrane guidance on
  heterogeneity and meta-analysis, PRISMA 2020 reporting guidance, methodological
  work on random-effects models, prediction intervals, publication bias and
  small-study effects.
methodology: >-
  Construct a synthetic meta-analysis with five equal-precision studies and
  deliberately heterogeneous effects. Compute the fixed-effect estimator,
  Cochran's Q, I-squared, DerSimonian-Laird between-study variance, the
  random-effects estimator and a t-based prediction interval. Use the example to
  distinguish precision of the pooled mean from heterogeneity of underlying
  effects and from bias in the evidence base.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  og_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  overlay_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-overlapping-cis.jpg
  twitter_image: /assets/images/headers/photo-statistics-overlapping-cis.jpg
---

<!--
Development contract
Question: What does a meta-analysis establish beyond the individual studies it combines?
Claim: Meta-analysis can improve precision and organise quantitative evidence, but it cannot repair bias, incompatible estimands, selective publication or scientifically important heterogeneity. A pooled mean is only one feature of the evidence distribution.
Counterclaim: Meta-analysis is one of the most powerful tools in quantitative evidence synthesis when eligibility criteria, risk-of-bias assessment, effect measures and modelling assumptions are appropriate. The problem is not pooling; it is treating pooling as automatic validation.
Evidence object: Five synthetic studies with effects 0.10, 0.15, 0.20, 0.80 and 0.90, common standard error 0.10, fixed-effect estimate 0.43, Q=59.8, I-squared about 93.3%, DerSimonian-Laird tau-squared 0.1395, random-effects interval about [0.09, 0.77], and prediction interval about [-0.88, 1.74].
Failure case: Interpreting I-squared mechanically, treating random effects as a cure for incompatible studies, using funnel-plot asymmetry as proof of publication bias, or assuming a systematic review is weak because it declines to pool.
Reader payoff: Read a meta-analysis as a structured argument about comparability, bias and uncertainty rather than as one pooled number.
Exclusions: Ranking named journals or review groups, claiming one estimator is universally best, and treating meta-analysis as appropriate when the primary studies do not share a meaningful estimand.
-->

A meta-analysis can produce one of the most persuasive objects in scientific communication: a pooled estimate with a narrow confidence interval.

Several studies point in roughly the same direction. Their results are combined. The uncertainty appears to shrink. A diamond at the bottom of a forest plot summarises the evidence. The visual impression is that many imperfect studies have been converted into one strong answer.

Sometimes that impression is justified.

Sometimes the pooled estimate is precise because the studies genuinely estimate a common or coherently distributed effect and the synthesis has been designed carefully.

Sometimes the precision is the least interesting feature of the evidence.

A meta-analysis is a statistical operation. It does not automatically repair weak design, poor measurement, incompatible outcomes, confounding, selective publication or heterogeneous interventions.

The pooled result inherits the scientific structure of the studies that entered it.

## A systematic review and a meta-analysis are different objects

A systematic review is a research process for identifying, selecting, appraising and synthesising a body of literature using explicit methods.

A meta-analysis is the statistical combination of quantitative results from separate studies.

A systematic review can contain a meta-analysis.

It does not have to.

If the included studies are too different to support a coherent quantitative comparison, declining to pool can be the correct methodological decision.

The distinction matters because public discussion often treats the terms as interchangeable:

> A meta-analysis reviewed twenty studies.

More precisely, a systematic review identified and evaluated the studies, and the authors may then have meta-analysed some subset of sufficiently comparable results.

The search and selection process determines which evidence enters the statistical model.

The meta-analysis begins after that decision.

## The pooled estimate answers a specific mathematical question

Suppose study $i$ estimates an effect

$$
\hat\theta_i
$$

with standard error

$$
s_i.
$$

Under a simple common-effect model, inverse-variance weights are

$$
w_i
=
\frac{1}{s_i^2}.
$$

The pooled estimator is

$$
\hat\theta_{\mathrm{FE}}
=
\frac{
\sum_{i=1}^{k}
w_i\hat\theta_i
}{
\sum_{i=1}^{k}
w_i
}.
$$

Its standard error is

$$
SE(\hat\theta_{\mathrm{FE}})
=
\sqrt{
\frac{1}{
\sum_i w_i
}
}.
$$

This estimator is sometimes called fixed-effect, although terminology varies and the underlying scientific assumption should be stated explicitly.

If all studies are estimating the same underlying effect and differ only through sampling error, inverse-variance weighting is natural.

More precise studies receive more weight.

Pooling can then estimate the common effect much more precisely than any individual study.

The difficulty begins when the phrase "same underlying effect" is not credible.

## Five studies can produce a precise pooled mean and an incoherent story

Consider five synthetic studies.

Each reports the same standard error,

$$
s_i=0.10.
$$

Their effect estimates are:

| Study | Effect estimate | Standard error |
| --- | ---: | ---: |
| 1 | 0.10 | 0.10 |
| 2 | 0.15 | 0.10 |
| 3 | 0.20 | 0.10 |
| 4 | 0.80 | 0.10 |
| 5 | 0.90 | 0.10 |

Because all standard errors are equal, all fixed-effect weights are equal.

The pooled estimate is simply the mean:

$$
\hat\theta_{\mathrm{FE}}
=
\frac{
0.10+0.15+0.20+0.80+0.90
}{5}
=
0.43.
$$

Each inverse-variance weight is

$$
w_i
=
\frac{1}{0.10^2}
=
100.
$$

Total weight is

$$
500.
$$

Therefore,

$$
SE(\hat\theta_{\mathrm{FE}})
=
\sqrt{
\frac{1}{500}
}
\approx
0.0447.
$$

A conventional 95% confidence interval is approximately

$$
0.43
\pm
1.96(0.0447),
$$

or

$$
[0.34,\ 0.52].
$$

That interval looks impressively precise.

The problem is visible before any heterogeneity statistic is calculated.

Three studies estimate effects near 0.15.

Two estimate effects near 0.85.

The pooled value

$$
0.43
$$

is not close to either cluster.

It may describe the arithmetic centre of the studies while describing no study particularly well.

## Cochran's Q asks whether dispersion exceeds sampling variation

One classical heterogeneity statistic is

$$
Q
=
\sum_i
w_i
(\hat\theta_i-\hat\theta_{\mathrm{FE}})^2.
$$

For the synthetic example,

$$
Q=59.8.
$$

Under a common-effect model and regularity conditions, $Q$ is compared with a chi-squared distribution with

$$
k-1=4
$$

degrees of freedom.

A value near 60 is far larger than would be expected from sampling error alone when each study has standard error 0.10.

The studies are not merely noisy versions of one common effect under this model.

Their differences contain additional structure.

## I-squared describes relative inconsistency

A common transformation of $Q$ is

$$
I^2
=
\max
\left(
0,
\frac{
Q-(k-1)
}{
Q
}
\right)
\times100\%.
$$

For the example,

$$
I^2
\approx
93.3\%.
$$

This indicates that, under the assumptions behind the statistic, a large fraction of the observed dispersion is inconsistent with ordinary within-study sampling variation alone.

The number is useful.

It is also frequently overinterpreted.

$I^2$ is not the percentage of studies that are wrong.

It is not the probability that the effects differ.

It is not a direct measure of clinical importance.

It depends on both between-study variation and within-study precision.

With extremely precise studies, even modest differences can produce a large $I^2$.

With imprecise studies, meaningful heterogeneity can produce a smaller $I^2$.

The effect estimates and their scientific context remain necessary.

## Random effects changes the model, not the history

A random-effects model allows underlying study effects to differ.

One common formulation is

$$
\theta_i
\sim
\mathcal N(\mu,\tau^2),
$$

where $\mu$ is the mean of the study-effect distribution and $\tau^2$ is between-study variance.

Observed estimates satisfy

$$
\hat\theta_i
\sim
\mathcal N(\theta_i,s_i^2).
$$

A common random-effects weight is

$$
w_i^\ast
=
\frac{
1
}{
s_i^2+\hat\tau^2
}.
$$

The model therefore recognises two sources of variation:

$$
\text{within-study uncertainty}
+
\text{between-study heterogeneity}.
$$

That is scientifically different from assuming one identical effect.

It does not make heterogeneous studies automatically comparable.

The intervention, population and outcome still need to justify a meaningful common synthesis.

## Estimating between-study variance changes the uncertainty substantially

For the synthetic example, the DerSimonian-Laird estimate of between-study variance is approximately

$$
\hat\tau^2
=
0.1395.
$$

Because the individual standard errors are all equal, the random-effects pooled mean remains

$$
\hat\mu
=
0.43.
$$

The standard error of the pooled mean becomes approximately

$$
0.173.
$$

A conventional normal-based 95% interval is therefore about

$$
[0.09,\ 0.77].
$$

The same five point estimates that generated the narrow fixed-effect interval

$$
[0.34,\ 0.52]
$$

now produce a much wider interval when between-study variation enters the model.

The point estimate did not change.

The scientific uncertainty did.

## The confidence interval around the mean is still not the range of future effects

The random-effects confidence interval estimates uncertainty about the mean of the distribution of true study effects.

It does not directly answer:

> What effect might we see in another comparable setting?

For that question, a prediction interval is more informative.

A simple prediction interval has the form

$$
\hat\mu
\pm
t
\sqrt{
\hat\tau^2
+
SE(\hat\mu)^2
}.
$$

Using a t critical value with three degrees of freedom in this small synthetic example gives a prediction interval of approximately

$$
[-0.88,\ 1.74].
$$

That interval is extremely wide.

The pooled mean is positive.

The predicted effect in another comparable setting can plausibly range from negative to strongly positive under the model.

This is the distinction that a forest plot can hide when attention moves immediately to the pooled diamond.

A mean effect can be estimated while the effect in a new setting remains highly uncertain.

Current Cochrane guidance explicitly recommends considering prediction intervals in random-effects meta-analysis because they express heterogeneity on the same scale as the effect measure and can be more interpretable than $I^2$ or $\tau^2$ alone.

## A pooled average can answer the wrong scientific question

Suppose the first three studies involve low-dose intervention $A$ and the final two involve a substantially larger dose.

Pooling all five may estimate an average across doses.

That can be mathematically valid.

It may not answer any useful causal question.

If the real question is:

> What is the effect of the low dose?

then the two high-dose studies belong to a different estimand.

If the question is:

> What is the average effect across this mixture of dose regimes?

then the pooled estimate may be appropriate, although the weighting scheme still determines what "average" means.

The decision to combine studies is therefore partly statistical and partly scientific.

Software cannot determine estimand compatibility from the numbers alone.

## Harmonising effect measures can hide incompatible outcomes

Suppose several studies report depression outcomes.

One uses a validated clinical scale.

Another uses a three-item questionnaire.

Another reports diagnosis.

Another reports symptom-free days.

Researchers may convert continuous outcomes to standardized mean differences or transform different effect measures onto a common scale.

That can enable synthesis.

It does not make the outcomes scientifically identical.

A standardized mean difference,

$$
d
=
\frac{
\bar X_1-\bar X_0
}{
s_{\mathrm{pooled}}
},
$$

removes the original measurement units.

It does not remove differences in construct validity, measurement reliability or clinical meaning.

Effect harmonisation is a modelling decision.

## Random effects is not permission to pool anything

A common mistake is:

> The studies are heterogeneous, so use random effects.

Random effects models variation.

They do not explain it.

If studies estimate genuinely different constructs, interventions or populations with no coherent target distribution, adding $\tau^2$ does not create one.

Suppose one study estimates short-term biomarker change in healthy adults and another estimates ten-year mortality in patients with severe disease.

The statistical model can produce a number if the effects are transformed onto some common scale.

The scientific interpretation may remain meaningless.

Heterogeneity models cannot repair incompatible estimands.

## Risk of bias survives pooling

Consider ten randomised trials that all have substantial differential loss to follow-up favouring treatment.

If each estimate is biased upward, pooling reduces the sampling variance around the biased centre.

Symbolically, suppose

$$
\hat\theta_i
=
\theta
+
b
+
\varepsilon_i,
$$

where $b$ is a common bias.

Averaging gives

$$
\bar{\hat\theta}
=
\theta+b+\bar\varepsilon.
$$

As the number of studies increases,

$$
\bar\varepsilon
\rightarrow0,
$$

but

$$
b
$$

does not disappear.

Meta-analysis can therefore make a shared bias look more certain.

This is why risk-of-bias assessment belongs inside evidence synthesis rather than after it.

## Weighting by precision does not weight by validity

Inverse-variance meta-analysis gives more weight to studies with smaller standard errors.

That usually means larger or more precise studies contribute more.

The weight is not a direct measure of methodological quality.

A very large biased study can dominate a meta-analysis because its standard error is tiny.

A smaller rigorous study can receive less statistical weight even when its causal design is stronger.

Some review methods incorporate risk-of-bias judgements through exclusion, sensitivity analysis or other modelling choices.

There is no universal transformation from "quality" to an inverse-variance weight.

Statistical precision and scientific validity are different dimensions.

## A systematic review constrains selective citation

One major strength of systematic review is not the pooled estimator.

It is the search process.

If a communicator selects three favourable studies from a literature containing thirty, the conclusion depends on source selection.

A systematic review specifies eligibility criteria before or independently of the study outcomes, searches multiple information sources and documents why studies were included or excluded.

PRISMA 2020 focuses strongly on transparent reporting of this process.

The review can still miss evidence.

Its selection procedure is at least auditable.

## Search strategy is part of the result

Suppose Review A searches:

- one database
- English-language papers
- published journal articles

Review B searches:

- several bibliographic databases
- trial registries
- preprints
- reference lists
- conference proceedings
- regulatory records

The evidence sets can differ even when eligibility criteria are nominally similar.

Search decisions can therefore change the meta-analysis.

A pooled result should not be read independently of how the literature was located.

## Publication bias creates a selected literature

Suppose studies with significant results are more likely to be published.

Then the observed literature satisfies

$$
P(\text{observed}\mid p<0.05)
>
P(\text{observed}\mid p\ge0.05).
$$

Even a perfectly executed meta-analysis of the available published studies is then analysing a selected sample.

This is not a failure of the pooling formula.

It is a failure of the evidence set to represent all completed research.

Trial registration and prospective protocols make missing results more visible.

## Funnel plots do not prove publication bias

A funnel plot displays study effect estimates against a measure of study size or precision.

Under a simple model, smaller studies should scatter more widely.

Asymmetry can be consistent with publication bias.

It can also arise from:

- true effect heterogeneity
- different methods in small studies
- poorer quality among small studies
- chance
- outcome-dependent standard errors

The plot is a diagnostic.

It is not a causal detector of publication bias.

Statistical tests of funnel asymmetry have similar limitations, especially with few studies.

## Small-study effects are a broader concept

Suppose smaller studies tend to report larger effects.

This pattern is called a small-study effect.

Publication bias is one possible explanation.

Others include:

- stronger interventions in small trials
- selected populations
- lower methodological quality
- different outcome measurement
- genuine effect modification

The descriptive pattern should therefore be separated from its explanation.

## Leave-one-out analysis asks whether one study controls the conclusion

Suppose pooled effect is

$$
\hat\theta.
$$

For each study $j$, recompute the synthesis without that study:

$$
\hat\theta_{(-j)}.
$$

If omitting one study changes the conclusion dramatically, the meta-analysis is sensitive to that study.

This does not mean the influential study should be removed.

It means the result depends heavily on it.

Influence is information.

## Sensitivity analysis should vary plausible decisions

A meta-analysis contains many decisions:

- eligibility criteria
- effect measure
- fixed versus random effects
- estimator of $\tau^2$
- treatment of zero-event studies
- inclusion of high-risk-of-bias studies
- handling of multiple outcomes
- choice of follow-up time
- duplicate populations

A robust conclusion should survive reasonable alternatives that do not change the scientific question.

Cochrane explicitly recommends sensitivity analysis for potentially influential decisions.

The goal is not to search until a preferred answer appears.

It is to show which conclusions depend on uncertain methodological choices.

## Meta-regression can explore heterogeneity

Suppose study effect depends on study-level characteristic $Z_i$:

$$
\theta_i
=
\beta_0+\beta_1Z_i+u_i.
$$

A meta-regression can estimate whether effect sizes vary systematically with $Z_i$.

Possible moderators include:

- dose
- mean age
- follow-up duration
- intervention intensity
- risk of bias
- baseline severity

This is useful for hypothesis generation and sometimes explanation.

It has important limitations.

The unit of analysis is the study.

A study-level association can differ from an individual-level association.

With few studies, meta-regression is unstable.

Multiple moderator searches can generate spurious patterns.

## Ecological bias exists inside meta-regression too

Suppose studies with older average participants show larger effects.

It does not follow that older individuals benefit more.

The meta-regression uses study averages.

The relationship can be produced by another study-level difference correlated with average age.

This is an ecological inference problem.

Individual participant data meta-analysis can sometimes address such questions more directly.

## Individual participant data meta-analysis changes what can be estimated

In an individual participant data meta-analysis, researchers obtain participant-level data from the original studies rather than relying only on published aggregate estimates.

This can allow:

- consistent outcome definitions
- common covariate adjustment
- participant-level interactions
- improved missing-data analysis
- more flexible time-to-event models

It is often considered a particularly rich form of synthesis.

It is also expensive, time-consuming and still dependent on which studies provide data.

Unavailable studies can create another selection problem.

## Prediction intervals deserve more attention

A random-effects pooled confidence interval answers uncertainty about the mean effect.

A prediction interval answers a different question:

> What range of true effects is plausible in another setting drawn from the same effect distribution?

When heterogeneity is substantial, the difference can be dramatic.

In the synthetic example:

$$
\hat\mu=0.43.
$$

The random-effects confidence interval is about

$$
[0.09,\ 0.77].
$$

The prediction interval is about

$$
[-0.88,\ 1.74].
$$

Reporting only the pooled confidence interval can make the evidence look much more homogeneous than the model itself implies.

IntHout and colleagues argued for routine use of prediction intervals for exactly this reason.

## Prediction intervals are themselves uncertain

With only a few studies, $\tau^2$ is estimated imprecisely.

A prediction interval based on $\hat\tau^2$ can therefore also be unstable.

A very wide interval is not automatically a perfect summary of future effects.

It is a reminder that the review contains limited information about heterogeneity.

Small meta-analyses should not create false confidence merely because several estimates have been placed into one model.

## A forest plot should be read from top to bottom

Many readers look immediately at the pooled diamond.

A better sequence is:

1. What populations were studied?
2. What interventions and comparators were used?
3. What outcomes were measured?
4. How precise is each study?
5. Do the point estimates agree?
6. Are confidence intervals broadly compatible?
7. Is there a pattern by design, dose or population?
8. What is the risk of bias?
9. What does the pooled estimate represent?
10. What does the prediction interval say?

The individual studies are not clutter surrounding the pooled result.

They are the evidence from which the pooled result was constructed.

## Non-overlapping confidence intervals are not the definition of heterogeneity

Two studies can have overlapping confidence intervals and still differ statistically.

Two studies can have non-overlapping intervals because they are both extremely precise even when the difference is not practically important.

Formal heterogeneity analysis uses the estimates and variances jointly.

Visual inspection is useful.

It should not substitute for the statistical and scientific comparison.

## Statistical heterogeneity and clinical heterogeneity are different

Statistical heterogeneity concerns variation in numerical effects beyond sampling error.

Clinical heterogeneity concerns differences in populations, interventions, outcomes or settings that may matter scientifically.

Methodological heterogeneity concerns differences in design and risk of bias.

A meta-analysis can show low statistical heterogeneity while containing important clinical differences if the studies are imprecise.

It can show high statistical heterogeneity around effects that are all clinically beneficial if studies are extremely precise.

The type of heterogeneity should be named.

## Zero heterogeneity estimates do not prove identical effects

Suppose a random-effects model estimates

$$
\hat\tau^2=0.
$$

With few or imprecise studies, that may reflect lack of information rather than true equality of effects.

Between-study variance is difficult to estimate accurately from small meta-analyses.

A zero estimate is not logical proof that all settings share one identical effect.

## Fixed-effect and random-effects models answer different questions

A common-effect analysis asks about one effect assumed to be shared by the included studies.

A random-effects analysis asks about a distribution of effects and often targets its mean.

Neither model is selected solely because a heterogeneity test crosses a p-value threshold.

The model should follow the scientific understanding of the studies.

If effects could reasonably vary by setting, a distributional model may be appropriate even when a heterogeneity test has low power.

If the studies are genuinely designed to estimate one common parameter, a common-effect model may be coherent.

Model choice should not be delegated to one preliminary test.

## A significant pooled effect is not the same as consistent benefit

Suppose a random-effects meta-analysis gives

$$
\hat\mu>0
$$

with a confidence interval excluding zero.

This supports a positive mean effect under the model.

It does not imply

$$
\theta_i>0
$$

for every setting.

If the prediction interval crosses zero, some comparable settings may plausibly have negligible or negative effects.

This distinction matters for policy and clinical recommendations.

A positive average can coexist with real effect heterogeneity.

## A non-significant pooled effect can still hide important subgroup effects

The reverse can occur.

Suppose half the studies show meaningful benefit and half show meaningful harm in systematically different populations.

The pooled mean may be near zero.

Concluding "no effect" would erase the heterogeneity.

The scientific question becomes:

> Which settings generate which effects?

A zero pooled mean can be an average of strong opposing effects.

The average is not automatically the conclusion.

## Combining p-values is not the same as combining effects

Some evidence-synthesis methods combine significance information across studies.

That can answer whether there is evidence against a global null under specific assumptions.

It does not estimate effect magnitude.

A meta-analysis intended to inform decisions usually needs an effect scale.

Knowing that several studies collectively reject zero does not tell us whether the effect is large enough to matter.

## Relative and absolute effects can tell different stories

Suppose several trials report a common relative risk near

$$
RR=0.8.
$$

Baseline risk differs across studies.

If one population has control risk

$$
0.05,
$$

the treated risk is approximately

$$
0.04,
$$

an absolute reduction of

$$
0.01.
$$

If another has control risk

$$
0.50,
$$

treated risk is approximately

$$
0.40,
$$

an absolute reduction of

$$
0.10.
$$

A meta-analysis of relative effects can be stable while the practical benefit differs substantially across settings.

The effect measure chosen for synthesis matters.

## Time points should not be pooled casually

An intervention can have:

- short-term benefit
- medium-term attenuation
- long-term harm

Combining outcomes measured at six weeks, six months and five years into one pooled effect can obscure the time structure.

Follow-up time is part of the estimand.

A meta-analysis should define whether it is estimating an effect at a particular time or across a meaningful time window.

## Duplicate populations can double-count evidence

Several papers can analyse the same underlying cohort.

If all are included as independent studies, participants can be counted multiple times.

The meta-analysis then appears to contain more independent evidence than it does.

Systematic reviews should therefore track study cohorts, trial registrations and overlapping datasets.

The unit of evidence is not always the publication.

## Outcome switching can enter the meta-analysis downstream

Suppose a trial measures ten outcomes but publishes only the favourable ones.

A systematic reviewer sees the published subset.

The bias occurred before the meta-analysis.

Comparing publications with protocols and trial registrations can reveal selective outcome reporting.

This is one reason prospective registration improves evidence synthesis even for researchers who never conduct the original trials.

## Risk-of-bias tools should change interpretation, not decorate a table

A review may include a coloured risk-of-bias figure.

If the pooled conclusion is then discussed as though all studies were equally credible, the assessment has not influenced inference.

Risk of bias should inform:

- sensitivity analyses
- certainty assessments
- interpretation
- sometimes eligibility

The correct approach depends on the review question and framework.

The important point is that bias assessment should have consequences.

## GRADE asks a different question from meta-analysis

GRADE is a framework for rating certainty in a body of evidence.

It considers domains such as:

- risk of bias
- inconsistency
- indirectness
- imprecision
- publication bias

A meta-analysis can produce a precise pooled estimate while certainty remains low because the evidence is indirect or biased.

The pooled standard error and the certainty of the evidence are different quantities.

## Protocols reduce review-level flexibility

Systematic reviews also face researcher degrees of freedom.

Authors choose:

- databases
- search dates
- eligibility rules
- outcomes
- effect measures
- subgroup definitions
- synthesis models

A prespecified review protocol makes those decisions visible before the results are known.

PROSPERO and other registries provide mechanisms for registering many systematic reviews prospectively.

Registration is not proof of quality.

It creates an audit trail.

## Updating a review can change the conclusion without invalidating the original review

Scientific evidence accumulates.

Suppose an early meta-analysis contains four small studies and estimates a large effect.

Several later, larger studies produce smaller effects.

An updated review may shrink the pooled estimate.

That does not mean the first review was fraudulent or useless.

It means the available evidence changed.

Evidence synthesis is a time-indexed statement:

$$
P(\text{conclusion}\mid\text{evidence available at time }t).
$$

A review should be interpreted relative to its search date.

## Living systematic reviews make that time dependence explicit

For rapidly changing fields, a living systematic review is updated as new evidence becomes available.

The method is especially useful when:

- evidence is emerging quickly
- decisions are time-sensitive
- new studies are likely to change conclusions

The review becomes an evolving evidence object rather than a static publication.

The methodology must still control repeated screening, analysis and versioning transparently.

## A meta-analysis should make disagreement harder to hide

The best meta-analysis is not the one that produces the narrowest pooled interval.

It is the one that makes the structure of the evidence visible.

That includes:

- which studies were found
- which were excluded
- which populations differ
- which outcomes differ
- which studies are biased
- how effects vary
- which assumptions determine the pooled result
- what happens under reasonable alternative analyses

Pooling is useful because it forces quantitative comparison.

It becomes misleading when the pooled number hides the comparison it was meant to organise.

## The synthetic example contains the central warning

The five studies were:

$$
0.10,\quad
0.15,\quad
0.20,\quad
0.80,\quad
0.90
$$

with common standard error

$$
0.10.
$$

A common-effect analysis produced

$$
\hat\theta_{\mathrm{FE}}=0.43
$$

with 95% confidence interval approximately

$$
[0.34,\ 0.52].
$$

That appears precise.

Yet

$$
Q=59.8
$$

and

$$
I^2\approx93.3\%.
$$

A random-effects analysis estimated

$$
\hat\tau^2\approx0.1395,
$$

giving a much wider confidence interval around the mean of roughly

$$
[0.09,\ 0.77].
$$

The prediction interval was approximately

$$
[-0.88,\ 1.74].
$$

One dataset therefore supports three very different statements:

> The arithmetic centre of the studies is estimated precisely under a common-effect model.

> The mean of a heterogeneous effect distribution is positive but uncertain.

> The effect in a new comparable setting could plausibly differ enormously and may even have the opposite sign.

The numbers did not contradict one another.

They answered different questions.

## Meta-analysis should be read as an argument, not a badge

A good meta-analysis is powerful because it makes evidence accumulation explicit.

It can improve precision.

It can reveal heterogeneity.

It can identify gaps.

It can test whether findings are robust across studies.

It can expose small-study patterns.

It can estimate effects that no individual study estimates well.

None of that makes the pooled result independent of the evidence that produced it.

The right questions are therefore not:

> Is this a meta-analysis?

or

> How many studies were pooled?

The better questions are:

What studies were eligible?

What was missed?

What estimand did they share?

How were outcomes harmonised?

What was the risk of bias?

How heterogeneous were the effects?

What does the prediction interval show?

Do sensitivity analyses change the conclusion?

Is the pooled mean actually the quantity needed for the decision?

Meta-analysis is not a magic upgrade.

It is a model for combining evidence.

Its strength comes from how well the evidence and the model fit the scientific question.

## References

Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). *Introduction to Meta-Analysis*. Wiley.

Deeks, J. J., Higgins, J. P. T., Altman, D. G., McKenzie, J. E., & Veroniki, A. A. (2024). Analysing data and undertaking meta-analyses. In *Cochrane Handbook for Systematic Reviews of Interventions*, version 6.5.

DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188. https://doi.org/10.1016/0197-2456(86)90046-2

Egger, M., Davey Smith, G., Schneider, M., & Minder, C. (1997). Bias in meta-analysis detected by a simple, graphical test. *BMJ*, 315, 629–634. https://doi.org/10.1136/bmj.315.7109.629

Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539–1558. https://doi.org/10.1002/sim.1186

IntHout, J., Ioannidis, J. P. A., Rovers, M. M., & Goeman, J. J. (2016). Plea for routinely presenting prediction intervals in meta-analysis. *BMJ Open*, 6, e010247. https://doi.org/10.1136/bmjopen-2015-010247

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71. https://doi.org/10.1136/bmj.n71

Sterne, J. A. C., Sutton, A. J., Ioannidis, J. P. A., Terrin, N., Jones, D. R., Lau, J., et al. (2011). Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials. *BMJ*, 343, d4002. https://doi.org/10.1136/bmj.d4002
