---
permalink: '/science-communication/replication_is_more_than_getting_the_same_p_value_twice/'
title: 'Replication Is More Than Getting the Same p-Value Twice'
date: '2025-06-05'
categories:
- Science Communication
tags:
- Replication
- Reproducibility
- Statistical Power
- Effect Sizes
- Scientific Reasoning
author_profile: false
classes: wide
seo_title: 'Replication Is More Than Repeating Statistical Significance'
seo_description: 'Two studies can estimate the same underlying effect while producing different p-values. Replication requires comparing effect sizes, uncertainty, design, measurement, and the scientific claim.'
seo_type: article
excerpt: >-
  Repeating p < 0.05 is a poor definition of replication. Under modest power,
  an exact repeat of a real effect may often fail to cross the same threshold,
  while selection on the first significant result can make the replication
  estimate look smaller even when the underlying effect is unchanged.
summary: >-
  This article distinguishes computational reproducibility from empirical
  replication and develops a quantitative account of why repeated statistical
  significance is an inadequate replication criterion. A normal sampling model
  shows how low power limits the probability of repeating significance and how
  conditioning the original study on significance inflates its expected effect
  estimate. The discussion then considers effect-size agreement, uncertainty,
  measurement, design fidelity, conceptual replication, prediction, meta-analysis,
  and informative disagreement between studies.
keywords:
- replication
- reproducibility
- p values
- statistical power
- effect sizes
- winner's curse
why_this_exists: >-
  Public discussion often reduces replication to whether a second study also
  obtains p < 0.05. That criterion ignores power, selection, effect magnitude,
  measurement, and the actual scientific proposition being tested. This article
  shows mathematically why the binary rule can misclassify both successful and
  unsuccessful replications.
evidence: >-
  An original normal-sampling calculation for repeated significance and
  significance-conditioned effect inflation, National Academies definitions of
  reproducibility and replicability, the Open Science Collaboration's large
  replication project, and methodological work on replication, statistical
  power, and interpretation of repeated studies.
methodology: >-
  Model independent estimates as normal around a common true effect. Derive
  the probability that an exact replication again crosses a two-sided 0.05
  threshold, then condition the original estimate on significance to quantify
  selection-induced inflation. Compare binary significance with effect-size,
  interval, predictive, and meta-analytic approaches to replication assessment.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  og_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  twitter_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
---

<!--
Development contract
Question: What does it mean for a scientific result to replicate?
Claim: Replication concerns consistency of evidence about a scientific claim across independently collected data. Repeating a significance threshold is neither necessary nor sufficient for that consistency.
Counterclaim: Binary criteria can be useful operational summaries when specified in advance, and large well-powered replications may legitimately use hypothesis tests. The problem is treating one threshold as a complete definition of replication.
Evidence object: Normal sampling model with a true effect of 0.30 and standard error 0.20, exact calculation of 32.3% repeat-significance probability, conditional inflation of the original significant estimate to about 0.52, and comparisons based on effect estimates and uncertainty.
Failure case: Treating every difference between studies as harmless sampling variation, declaring a failed replication whenever p exceeds 0.05, or using broad conceptual similarity to excuse a replication that no longer tests the original proposition.
Reader payoff: Distinguish reproducibility from replication, understand why p-values vary across repeated studies, recognise winner's-curse behaviour, and evaluate whether two studies support materially compatible scientific claims.
Exclusions: Ranking disciplines by replicability, reducing the replication literature to a single crisis statistic, and claiming that a failed replication proves misconduct or that an original study was false.
-->

A replication study is sometimes summarised with a question that sounds objective: did the second study also obtain (p<0.05)?

The attraction is obvious. The rule is simple, familiar, and easy to apply. If the original result was significant and the replication is significant in the same direction, the finding appears to have replicated. If the replication is not significant, it appears not to have replicated.

The rule is also mathematically inadequate.

Two independent studies can estimate the same underlying effect and produce different p-values because p-values depend on the realised estimate and its standard error. A low-powered original study that reaches significance has also passed a selection filter that tends to favour unusually large estimates. An exact replication can therefore estimate a smaller effect and fail to cross the same threshold even when nothing about the underlying causal effect has changed.

The opposite problem is possible as well. Two studies can both produce (p<0.05) while estimating effects that differ materially in magnitude, arise from different measurement artefacts, or support different substantive interpretations.

Replication is therefore not a property of one p-value.

It is a comparison between scientific claims and the evidence generated by independent data.

## Reproducibility and replication are not the same task

Terminology varies across disciplines, which is one reason discussions of replication can become confused before the statistical questions even begin.

The National Academies of Sciences, Engineering, and Medicine uses **reproducibility** to mean computational reproducibility: obtaining consistent computational results from the same data, code, methods, and conditions of analysis.

Under that definition, another researcher who receives the original dataset and analysis code should be able to regenerate the reported tables, estimates, and figures.

A **replication** uses newly collected data to investigate the same or a closely related scientific question.

These are distinct tests.

A result can be computationally reproducible and empirically fail to replicate. The code may be flawless while the estimated effect does not persist in new data.

A result can also be empirically replicable while the original analysis is poorly documented. Another study may independently find a similar effect even though nobody can reconstruct exactly how the first paper produced its numbers.

Both properties matter, but they diagnose different parts of the scientific process.

## The p-value is a random variable across replications

Suppose two independent studies estimate the same true effect $\delta$.

Let

$$
\hat\delta_j
\sim
\mathcal N(\delta,\sigma^2),
$$

for studies (j=1,2), and assume for simplicity that both have the same known standard error $\sigma$.

The corresponding z statistic is

$$
Z_j
=
\frac{\hat\delta_j}{\sigma}.
$$

Its distribution is

$$
Z_j
\sim
\mathcal N
\left(
\frac{\delta}{\sigma},
1
\right).
$$

Even though $\delta$ is fixed, $Z_j$ changes from sample to sample because $\hat\delta_j$ changes.

The two-sided p-value

$$
p_j
=
2\left[1-\Phi(|Z_j|)\right]
$$

is therefore also random.

There is no reason to expect two exact replications to produce the same p-value.

The relevant probability is not

$$
\Pr(p_2=p_1),
$$

which is effectively zero for continuous statistics.

It is the probability that the replication falls into whatever evidential region was defined in advance.

If that region is simply

$$
p_2<0.05,
$$

the probability of "successful replication" under the rule is just the statistical power of the replication design against the true effect.

That observation already causes trouble for the binary criterion.

## A real effect can fail to repeat significance most of the time

Take a true effect of

$$
\delta=0.30
$$

and standard error

$$
\sigma=0.20.
$$

The expected z statistic is

$$
\frac{\delta}{\sigma}
=
1.5.
$$

For a two-sided 0.05 test, statistical significance requires approximately

$$
|Z|>1.96.
$$

Under the stated true effect,

$$
Z
\sim
\mathcal N(1.5,1).
$$

The probability of significance is therefore

$$
\Pr(|Z|>1.96)
=
\Pr(Z>1.96)
+
\Pr(Z<-1.96).
$$

Numerically,

$$
\Pr(|Z|>1.96)
\approx
0.323.
$$

So an exact replication of a real effect under this design has only about a 32.3% chance of producing (p<0.05).

Equivalently, it has about a 67.7% chance of being labelled "non-significant".

Nothing in this calculation represents a failed mechanism, changing population, poor laboratory practice, data fabrication, or analytical mistake.

It is ordinary sampling variation under inadequate power.

A criterion that defines replication as repeated significance would therefore classify most exact repetitions of this true effect as failures.

That is not a property of replication.

It is a property of the chosen decision threshold and design.

## The original significant estimate is selected for being unusually large

The problem becomes sharper when the original study enters scientific attention partly because it was significant.

Continue with

$$
\delta=0.30
$$

and

$$
\sigma=0.20.
$$

For a positive observed estimate to be significant at the two-sided 0.05 level, it must exceed

$$
1.96(0.20)=0.392.
$$

The original estimate follows

$$
\hat\delta
\sim
\mathcal N(0.30,0.20^2).
$$

Condition on the event

$$
\hat\delta>0.392.
$$

For a normally distributed variable truncated below at a threshold $a$, the conditional mean is

$$
\mathbb E[\hat\delta\mid\hat\delta>a]
=
\delta
+
\sigma
\frac{\phi(\alpha)}
{1-\Phi(\alpha)},
$$

where

$$
\alpha
=
\frac{a-\delta}{\sigma}.
$$

Here,

$$
\alpha
=
\frac{0.392-0.30}{0.20}
=
0.46.
$$

Substituting gives

$$
\mathbb E[
\hat\delta
\mid
\hat\delta>0.392
]
\approx
0.522.
$$

The true effect is 0.30.

Among original studies that happen to clear the significance threshold in the positive direction, the expected reported estimate under this simplified model is about 0.52.

That is not bias in the estimator before selection. The unconditional estimator is centred correctly at 0.30.

The inflation appears because we selected studies conditional on an unusually favourable statistic.

This is a version of the winner's curse.

## A smaller replication estimate may be exactly what the model predicts

Suppose the original significant study reports

$$
\hat\delta_1=0.52.
$$

An exact replication then reports

$$
\hat\delta_2=0.28.
$$

The second estimate is nearly half the first.

That difference may look like evidence that the effect weakened.

Under the model just developed, it is unsurprising.

The original study was selected from the upper tail of its sampling distribution. The replication estimate is not conditioned on reproducing the same selection accident. It is centred on the underlying effect of 0.30.

The expected regression from the selected original estimate toward the underlying value is therefore a statistical consequence of the selection rule.

This does not mean that every smaller replication estimate should be dismissed as winner's curse. Real effect heterogeneity, differences in implementation, measurement, population, and bias can also produce smaller effects.

It means that "the replication estimate is smaller" is not itself a diagnosis.

We need to compare the observed difference with the difference expected under a coherent model of sampling variation and selection.

## Comparing significance statuses throws away most of the evidence

Suppose the original study reports

$$
\hat\delta_1=0.45
$$

with standard error

$$
SE_1=0.18.
$$

Its z statistic is

$$
Z_1=2.50,
$$

so

$$
p_1\approx0.012.
$$

Now suppose the replication reports

$$
\hat\delta_2=0.35
$$

with

$$
SE_2=0.22.
$$

Then

$$
Z_2\approx1.59,
$$

and

$$
p_2\approx0.11.
$$

A binary comparison says:

$$
\text{significant}
\quad\text{versus}\quad
\text{not significant}.
$$

That wording creates the impression of contradiction.

The effect estimates differ by only

$$
0.45-0.35=0.10.
$$

Under independence, the standard error of their difference is approximately

$$
SE_{\Delta}
=
\sqrt{SE_1^2+SE_2^2}
$$

so

$$
SE_{\Delta}
=
\sqrt{0.18^2+0.22^2}
\approx
0.284.
$$

The standardized difference is

$$
\frac{0.10}{0.284}
\approx
0.35.
$$

The two estimated effects are highly compatible with a common underlying value.

The apparently dramatic difference came from comparing each estimate separately with zero rather than comparing the estimates with each other.

This is the same logical mistake behind the statement that "significant in one group but not significant in another" proves that the groups differ.

Difference in significance is not significance of the difference.

## Confidence intervals expose what the p-value hides

The same example is clearer if we examine uncertainty intervals.

The original 95% interval is approximately

$$
0.45
\pm
1.96(0.18),
$$

or

$$
[0.10, 0.80].
$$

The replication interval is approximately

$$
0.35
\pm
1.96(0.22),
$$

or

$$
[-0.08, 0.78].
$$

The replication interval contains zero.

It also contains much of the original interval and includes effect sizes that would be scientifically close to the original estimate.

Calling the replication a failure because one interval crosses zero discards this information.

Intervals do not solve every replication problem. Their interpretation still depends on the statistical model, sampling design, selection process, and estimand.

They do force us to confront magnitude and precision.

A replication estimate of 0.35 with a wide interval means something different from an estimate of 0.01 with a narrow interval, even if both produce (p>0.05).

The first may be inconclusive but broadly compatible with the original effect.

The second may provide strong evidence that the original magnitude does not persist.

A binary p-value rule treats them alike.

## A successful replication needs a target claim

Before deciding whether a study replicated, we need to know what proposition is being tested.

Consider three possible original claims:

$$
C_1:
\delta>0,
$$

$$
C_2:
\delta\ge0.50,
$$

and

$$
C_3:
\delta
\text{ is large enough to change a practical decision}.
$$

A replication estimate of

$$
0.20
$$

may support $C_1$, contradict $C_2$, and be irrelevant to $C_3$ until a practical threshold is specified.

The replication status therefore depends on the scientific claim.

This is why repeating the original p-value threshold is conceptually weak. The null hypothesis

$$
H_0:\delta=0
$$

may never have been the most important scientific proposition.

If the substantive claim concerns a large effect, testing only whether the replication differs from zero sets the bar too low.

If the claim concerns the existence of any nonzero effect, requiring the replication point estimate to equal the original estimate sets the bar too high.

Replication should be evaluated against the proposition that justified scientific interest in the first place.

## Effect-size agreement should not mean numerical equality

Two independent estimates are almost never exactly equal.

If

$$
\hat\delta_1
\sim
\mathcal N(\delta,\sigma_1^2)
$$

and

$$
\hat\delta_2
\sim
\mathcal N(\delta,\sigma_2^2),
$$

then their difference satisfies

$$
\hat\delta_1-\hat\delta_2
\sim
\mathcal N
\left(
0,
\sigma_1^2+\sigma_2^2
\right)
$$

under a common-effect model.

Some disagreement is therefore expected even when the studies are exact replications.

The right question is whether the disagreement is large relative to expected variation and relative to the scientific tolerance that matters.

A difference of 0.10 may be trivial when standard errors are 0.30.

The same difference may be highly informative when standard errors are 0.01.

Statistical consistency depends on uncertainty.

Scientific consistency also depends on what magnitude of disagreement would materially alter the interpretation.

## Meta-analysis treats studies as evidence, not verdicts

Once several studies estimate related effects, a natural next step is to combine their information.

Under a simple fixed-effect model,

$$
\hat\delta_j
\sim
\mathcal N(\delta,s_j^2),
$$

and an inverse-variance weighted estimate is

$$
\hat\delta_{\text{FE}}
=
\frac{
\sum_j w_j\hat\delta_j
}{
\sum_j w_j
},
$$

where

$$
w_j=\frac{1}{s_j^2}.
$$

This formulation treats each study as a noisy estimate of a common effect rather than as a binary success or failure.

If effects differ across populations or implementations, a random-effects model introduces between-study variation:

$$
\delta_j
\sim
\mathcal N(\mu,\tau^2).
$$

Now replication is no longer framed as demanding identical effects. The model asks whether observed variation is compatible with a distribution of effects and attempts to estimate its centre and heterogeneity.

These models require assumptions and can themselves be misused. A pooled estimate does not rescue biased studies, and a heterogeneity parameter does not explain why studies differ.

The conceptual advantage is that evidence accumulates continuously.

Scientific knowledge is not reset to zero every time one study crosses a threshold and another does not.

## Direct replication tests stability under closely matched conditions

A direct replication attempts to reproduce the important features of the original design as closely as practical while collecting new data.

Its strength is diagnostic.

If the same manipulation, measurement process, target population, and analysis repeatedly produce similar effects, the original result gains credibility within that domain.

If the result does not persist, the disagreement narrows the search for what mattered.

Perhaps the original estimate was inflated.

Perhaps the replication changed a detail that was scientifically important.

Perhaps the measurement was unstable.

Perhaps the effect depends on a contextual variable that neither study originally recognised.

A direct replication is therefore not merely ceremonial repetition.

It is an experiment on the stability of the original result.

## Conceptual replication asks a harder question

A conceptual replication changes aspects of the operationalisation while testing a related theoretical claim.

Suppose a theory predicts that mechanism $M$ should affect outcome $Y$.

The original study manipulates $M$ using procedure $A$.

A conceptual replication uses a different procedure $B$.

If both produce the predicted consequence, confidence can increase that the finding is not an artefact unique to procedure $A$.

This is scientifically valuable.

It is also easier to interpret flexibly after the fact.

If procedure $B$ fails, researchers can argue that $B$ did not manipulate the construct correctly.

If $B$ succeeds differently, the theory can sometimes be modified to accommodate the result.

The more a replication changes, the more scientific judgement enters the claim that it tested the same proposition.

Direct and conceptual replications therefore answer related but distinct questions.

One tests stability under close repetition.

The other tests generality across operationalisations.

Neither should be allowed to substitute for the other automatically.

## Measurement replication matters as much as outcome replication

Suppose two studies use a variable labelled "stress".

The first uses a questionnaire total.

The second uses salivary cortisol.

Even if both are described with the same word, they do not necessarily measure the same construct.

A replication that changes measurement can fail because the original phenomenon does not generalise.

It can also fail because the operationalisations are not equivalent.

The problem is the same one developed in measurement theory: the observed variable is not identical to the construct named by the variable.

A serious replication therefore needs to document not only sample size and statistical analysis but also whether the measurement process preserves the interpretation required by the theory.

This is especially important when instruments differ across language, population, device version, laboratory assay, or time.

Replication depends on the stability of the measurement function as well as the stability of the effect.

## A larger replication is not automatically a fairer replication

Increasing sample size reduces sampling uncertainty.

That is usually valuable.

It does not repair a changed estimand, biased measurement, altered intervention, poor adherence, or systematic selection.

Suppose the original study estimates

$$
\delta_A
$$

under population $A$, while the replication estimates

$$
\delta_B
$$

under population $B$.

If treatment effects are heterogeneous,

$$
\delta_A

eq
\delta_B
$$

can be true even when both studies are internally valid.

A much larger replication in population $B$ can estimate $\delta_B$ with great precision while saying less about $\delta_A$ than its sample size suggests.

Precision is not transportability.

This is why replication design requires more than "use a bigger sample."

The target population and conditions belong in the scientific claim.

## Prediction offers another replication target

Some scientific claims imply quantitative predictions for future observations.

Suppose the original model estimates

$$
\hat\theta
$$

and predicts that a replication estimate should follow

$$
\hat\theta_{\text{rep}}
\sim
\mathcal N
\left(
\hat\theta,
V_{\text{pred}}
\right),
$$

where (V_{\text{pred}}) includes uncertainty in the original estimate and sampling variation in the replication.

The replication can then be assessed against a predictive distribution rather than a significance threshold.

If the new result falls in a region the original model considered extremely improbable, the model has encountered a genuine challenge.

If the replication is non-significant but lies comfortably inside the predictive distribution, calling it a contradiction would be difficult to defend.

Prediction therefore makes replication more demanding in one sense and less arbitrary in another.

The model must specify what range of new results it actually expected.

## Replication can fail because the original estimate was too precise

The usual story imagines that replication failure means the replication estimate moved away from the truth.

Sometimes the problem is the original uncertainty estimate.

Suppose an analysis ignores clustering.

The reported standard error may be

$$
SE_{\text{naive}}=0.10,
$$

while a correct cluster-robust analysis gives

$$
SE_{\text{correct}}=0.25.
$$

The point estimate does not change.

The p-value may change dramatically.

A later study that handles dependence correctly can look like a failure to reproduce the result even though the difference arises from inferential assumptions rather than the underlying effect.

Other examples include optional stopping, unmodelled repeated measures, multiple testing, data-dependent subgroup selection, and underestimated measurement uncertainty.

Replication therefore evaluates a whole inferential procedure, not only a numerical effect.

## One failed replication does not identify the cause of disagreement

Suppose an original study and replication disagree substantially.

At least several explanations remain possible.

The original study may have overestimated the effect through sampling variation or selective analysis.

The replication may have low power or implementation problems.

The true effect may differ across populations.

The measurement process may have changed.

The intervention may interact with an unmeasured contextual factor.

One study may contain bias that the other avoids.

Or the original scientific theory may simply be wrong.

The disagreement is evidence.

It does not, by itself, tell us which explanation generated the disagreement.

The National Academies explicitly notes that non-replicability can sometimes expose previously unrecognised variation or complexity rather than merely signalling poor research practice.

That is one reason replication is scientifically useful even when it fails.

A disagreement that is investigated carefully can reveal the boundary conditions of a phenomenon.

## Large replication projects are evidence about systems as well as individual findings

The Open Science Collaboration's 2015 project attempted independent replications of 100 experimental and correlational studies from psychology.

Its importance extends beyond any single replication percentage.

The project demonstrated that a large body of published findings could be examined using coordinated protocols, independent data collection, direct comparison of effect estimates, and multiple criteria for replication.

It also made visible a central statistical fact: replication results form distributions.

Some estimates remain close to the originals.

Some shrink.

Some change sign.

Some remain uncertain.

Reducing such a project to one success rate loses much of the information that made it scientifically useful.

Replication evidence is inherently multivariate. It concerns effect magnitude, precision, direction, design fidelity, heterogeneity, and the interpretation of the underlying claim.

## A replication criterion should be specified before seeing the result

Replication becomes vulnerable to hindsight when the success criterion changes after the new data are known.

If the replication is significant, significance becomes the criterion.

If it is not significant but the effect has the same sign, direction becomes the criterion.

If the effect is smaller, overlap of confidence intervals becomes the criterion.

If the effect reverses, methodological differences become the explanation.

Any one of those arguments can sometimes be scientifically legitimate.

The problem is selecting among them after the outcome is visible.

A stronger design states in advance what would count as meaningful consistency.

That criterion might concern:

- a minimum effect size
- an equivalence region around the original effect
- a prediction interval
- a directional hypothesis
- a meta-analytic update
- a prespecified combination of several measures

The choice should follow from the scientific claim.

Pre-specification does not remove judgement.

It prevents the judgement from being completely determined by the result.

## Replication should change belief continuously

The most useful way to think about replication is as evidence accumulation.

Suppose an original study supports an effect near

$$
\delta=0.50
$$

but with substantial uncertainty.

A precise replication near

$$
0.45
$$

should increase confidence that an effect of roughly that magnitude is real.

A precise replication near

$$
0.05
$$

should reduce confidence in the original magnitude.

A very imprecise replication near

$$
0.05
$$

should change confidence less because it provides little information.

A replication near

$$
-0.50
$$

with small uncertainty creates a much deeper conflict requiring explanation.

These cases should not receive the same binary label merely because some have (p>0.05).

The amount by which evidence changes should depend on what was observed and how informative the observation was.

## The scientific claim is the unit that should replicate

Return to the original low-power example.

The true effect is

$$
\delta=0.30.
$$

The design has standard error

$$
0.20.
$$

An exact replication has only about a 32.3% chance of producing (p<0.05).

Among original studies selected for positive significance, the expected estimate is about

$$
0.52,
$$

even though the underlying effect remains

$$
0.30.
$$

A replication estimate near 0.30 can therefore be smaller and non-significant while being more representative of the true effect than the original significant estimate.

That is enough to reject repeated statistical significance as a general definition of replication.

A serious replication asks whether independently collected evidence supports a materially compatible scientific proposition.

That comparison may involve effect sizes, uncertainty intervals, predictive distributions, design fidelity, measurement validity, population differences, and theory.

Sometimes the appropriate conclusion is that the result replicated closely.

Sometimes the effect exists but is smaller than originally believed.

Sometimes the studies are too imprecise to decide.

Sometimes the disagreement identifies genuine heterogeneity.

Sometimes the original claim does not survive.

The p-value can contribute evidence to those conclusions.

It cannot define them by itself.

## References

Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S and Type M errors. *Perspectives on Psychological Science*, 9(6), 641–651. https://doi.org/10.1177/1745691614551642

Goodman, S. N., Fanelli, D., & Ioannidis, J. P. A. (2016). What does research reproducibility mean? *Science Translational Medicine*, 8(341), 341ps12. https://doi.org/10.1126/scitranslmed.aaf5027

Hedges, L. V., & Schauer, J. M. (2019). More than one replication study is needed for unambiguous tests of replication. *Journal of Educational and Behavioral Statistics*, 44(5), 543–570. https://doi.org/10.3102/1076998619852953

National Academies of Sciences, Engineering, and Medicine. (2019). *Reproducibility and Replicability in Science*. Washington, DC: The National Academies Press. https://doi.org/10.17226/25303

Nosek, B. A., & Errington, T. M. (2020). What is replication? *PLoS Biology*, 18(3), e3000691. https://doi.org/10.1371/journal.pbio.3000691

Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251), aac4716. https://doi.org/10.1126/science.aac4716

Patil, P., Peng, R. D., & Leek, J. T. (2016). A statistical definition for reproducibility and replicability. *bioRxiv*. https://doi.org/10.1101/066803

Simonsohn, U. (2015). Small telescopes: Detectability and the evaluation of replication results. *Psychological Science*, 26(5), 559–569. https://doi.org/10.1177/0956797614567341
