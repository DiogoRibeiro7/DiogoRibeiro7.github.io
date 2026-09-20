---
permalink: '/science-communication/absence_of_evidence_is_not_always_evidence_of_absence/'
title: 'Absence of Evidence Is Not Always Evidence of Absence'
date: '2024-05-23'
categories:
- Science Communication
tags:
- Statistical Power
- Equivalence Testing
- Confidence Intervals
- Hypothesis Testing
- Scientific Reasoning
author_profile: false
classes: wide
seo_title: 'When a Null Result Does and Does Not Support No Meaningful Effect'
seo_description: 'A non-significant result does not by itself establish no effect. Confidence intervals, statistical power, and equivalence tests show when data are merely inconclusive and when they can rule out effects of practical importance.'
seo_type: article
excerpt: >-
  Two studies can report the same estimated effect and the same non-significant
  result while providing radically different evidence. The difference lies in
  what their uncertainty allows the data to rule out.
summary: >-
  This article distinguishes failure to reject a point null hypothesis from
  evidence that an effect is small enough to be practically negligible. Two
  studies with the same point estimate are compared analytically: one is too
  imprecise to distinguish large benefit from large harm, while the other
  supports equivalence within a pre-specified region of negligible effects.
  The discussion then treats statistical power, confidence intervals,
  equivalence testing, post-hoc power, Bayesian evidence, and the role of
  scientifically meaningful effect thresholds.
keywords:
- absence of evidence
- statistical power
- equivalence testing
- null result
- confidence intervals
- minimum effect of interest
why_this_exists: >-
  Scientific reports routinely translate a non-significant result into claims
  that there is no difference, no association, or no effect. The inferential
  error is not corrected by repeating that absence of evidence is never
  evidence of absence. With sufficiently informative data and an explicit
  definition of a negligible effect, observations can provide genuine evidence
  against effects large enough to matter.
evidence: >-
  Exact normal-theory calculations comparing two equally estimated but
  differently precise effects, the two one-sided tests procedure for
  equivalence, classical work by Altman and Bland, modern equivalence-testing
  guidance, and methodological work on power and post-hoc power.
methodology: >-
  Model an estimator as normally distributed around an unknown effect. Compare
  conventional point-null testing with interval estimation and equivalence
  testing under a pre-specified smallest effect of interest. Derive design
  power for a two-sample comparison and distinguish prospective power from
  retrospective calculations based on the observed effect.
reviewed_at: '2024-05-23'
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
Question: What can be inferred when a study fails to find a statistically significant effect?
Claim: Failure to reject a point null hypothesis does not establish a negligible effect. Evidence for practical absence requires data precise enough to exclude effects that would matter under a scientifically justified criterion.
Counterclaim: A null result is not forever uninformative. A sufficiently precise estimate, an explicit equivalence region, or another inferential framework can provide evidence against effects of scientifically important magnitude.
Evidence object: Two normal-theory studies with identical point estimates but very different standard errors, exact 95% and 90% confidence intervals, two one-sided equivalence tests, and prospective power calculations for a standardised two-sample effect.
Failure case: Treating every non-significant result as inconclusive regardless of precision, choosing equivalence bounds after seeing the data, using observed post-hoc power to reinterpret a completed study, or equating a statistical equivalence margin with literal equality.
Reader payoff: Distinguish non-significance from evidence of negligible effect, use confidence intervals to identify what remains compatible with the data, understand what power describes, and recognise when equivalence testing addresses the scientific question more directly than a point-null test.
Exclusions: Replacing domain judgement with a universal equivalence threshold, interpreting confidence intervals as posterior probability intervals, and claiming that frequentist equivalence testing is the only valid route to evidence for absence.
-->

A scientific study reports no statistically significant difference between two groups. The conclusion then appears almost automatically: there was no effect.

The statistical calculation does not support that conclusion by itself.

A conventional significance test asks whether the data are sufficiently incompatible with a specified null hypothesis, often an exact effect of zero. Failure to reject that hypothesis means that the data did not cross the chosen rejection threshold under the test that was performed. It does not establish that the true effect equals zero, and it does not establish that the study has ruled out effects large enough to matter.

The familiar phrase "absence of evidence is not evidence of absence" was used by Altman and Bland to describe this error. The phrase remains useful, but taken literally as a universal rule it creates a second problem. Data can provide evidence that a scientifically important effect is absent when the study is sufficiently informative to exclude effects of that magnitude.

The issue is therefore not whether the result is labelled significant or non-significant.

The issue is what effects remain compatible with the data.

## A point null and practical absence are different hypotheses

Let (	heta) denote an unknown effect. For a difference in means, (	heta) could represent the population mean difference between two conditions. Suppose an estimator (hat	heta) is approximately normally distributed:

[
hat	heta
sim
mathcal N(	heta, s^2),
]

where (s) is the standard error.

The conventional two-sided null hypothesis is

[
H_0:	heta=0.
]

A corresponding test statistic is

[
Z
=
rac{hat	heta}{s}.
]

At significance level (alpha=0.05), the null is rejected in a large-sample normal test when

[
|Z|>1.96.
]

This procedure distinguishes data that cross a particular incompatibility threshold under (	heta=0) from data that do not. It does not directly test whether the effect is too small to matter.

That scientific question requires a different hypothesis.

Suppose effects between (-Delta) and (Delta) would be considered practically negligible:

[
-Delta<	heta<Delta.
]

The value (Delta) must come from substantive considerations. It may represent the smallest effect that would alter a decision, a clinically meaningful difference, an engineering tolerance, an economically relevant change, or another domain-specific criterion. It should not be chosen merely because it makes the observed result convenient.

Under this formulation, the question is no longer whether (	heta) is exactly zero. The question is whether effects of meaningful magnitude can be excluded.

Those are different inferential tasks.

## The same estimate can support very different conclusions

Consider two studies that produce the same point estimate:

[
hat	heta=0.10.
]

Suppose both use the conventional two-sided test of (H_0:	heta=0).

In Study A, the standard error is

[
s_A=0.60.
]

The test statistic is

[
Z_A
=
rac{0.10}{0.60}
approx
0.167,
]

which gives a two-sided (p)-value of approximately

[
p_Aapprox0.868.
]

The 95% confidence interval is

[
0.10
pm
1.96(0.60),
]

or approximately

[
[-1.08, 1.28].
]

Now consider Study B, with the same point estimate but a much smaller standard error:

[
s_B=0.12.
]

Its test statistic is

[
Z_B
=
rac{0.10}{0.12}
approx
0.833,
]

giving

[
p_Bapprox0.405.
]

This result is also conventionally non-significant.

Its 95% confidence interval, however, is

[
0.10
pm
1.96(0.12),
]

or approximately

[
[-0.14, 0.34].
]

If the results are reduced to "both studies found no statistically significant effect", the distinction between them disappears.

That distinction is the main scientific information.

Study A remains compatible with effects around (-1) and (+1). Depending on the scale, those may represent substantial benefit and substantial harm. The study has not distinguished either possibility from zero with much precision.

Study B is different. Its uncertainty interval is narrow. It has not established that the true effect is exactly zero, but it has excluded a much larger range of effects.

The two studies therefore provide different evidence even though their point estimates are identical and both (p)-values exceed 0.05.

## Confidence intervals show what non-significance hides

A confidence interval does not assign probabilities to parameter values in the way a Bayesian credible interval does. Under repeated sampling, the interval procedure has a stated coverage property. For a 95% confidence procedure, 95% of intervals generated under repeated use of the procedure cover the true parameter under the model assumptions.

That formal interpretation matters. So does the practical information contained in the interval.

Study A produced

[
[-1.08, 1.28].
]

Study B produced

[
[-0.14, 0.34].
]

Both contain zero. Only the first contains effects of large magnitude.

A statement such as "there was no significant difference" therefore compresses two distinct questions into one binary label:

[
	ext{Does the interval contain zero?}
]

and

[
	ext{What nonzero effects does the interval also contain?}
]

The first question determines the conventional 0.05 significance result for a corresponding two-sided test. The second determines whether the study has learned enough to exclude scientifically important alternatives.

This is the core reason that non-significance cannot be interpreted without precision.

A wide interval crossing zero often means uncertainty.

A narrow interval crossing zero can mean that any remaining effect is small.

Those are not the same result.

## Equivalence testing reverses the burden of proof

Suppose effects within

[
[-0.50, 0.50]
]

are considered negligible for the scientific problem at hand. Here,

[
Delta=0.50.
]

A conventional test evaluates

[
H_0:	heta=0.
]

An equivalence test instead evaluates whether the data provide enough evidence to reject effects outside the negligible region.

Using the two one-sided tests procedure, the composite null hypothesis is

[
H_0:
	hetaleq-0.50
quad	ext{or}quad
	hetageq0.50.
]

The alternative is

[
H_1:
-0.50<	heta<0.50.
]

At level (alpha=0.05), equivalence is concluded only if both one-sided null components can be rejected.

There is an equivalent confidence-interval interpretation. For the ordinary symmetric setting considered here, the 90% confidence interval must lie completely inside the equivalence bounds.

For Study A,

[
0.10
pm
1.645(0.60)
]

gives the 90% interval

[
[-0.89, 1.09].
]

This interval extends far outside

[
[-0.50, 0.50].
]

Study A therefore supports neither a conventional difference from zero nor equivalence within the chosen bounds.

It is inconclusive with respect to both questions.

For Study B,

[
0.10
pm
1.645(0.12)
]

gives

[
[-0.10, 0.30].
]

That interval lies entirely inside

[
[-0.50, 0.50].
]

Study B therefore supports equivalence under the pre-specified criterion.

The two studies had the same estimated effect.

They had the same qualitative significance label.

One could not exclude large effects.

The other could.

This is what it means for absence of evidence to become evidence against effects large enough to matter.

## Equivalence is not equality

A conclusion of equivalence does not establish

[
	heta=0.
]

It establishes something weaker and usually more useful:

[
|	heta|<Delta
]

is supported relative to the chosen inferential procedure and assumptions.

The distinction matters because exact equality is rarely the relevant scientific question. In many physical, biological, economic, and behavioural systems, an effect is unlikely to be exactly zero in a literal mathematical sense. With enough measurement precision, extremely small departures from zero may eventually be detectable.

Scientific decisions usually concern scale.

A manufacturing process may tolerate deviations smaller than a technical threshold. A medical comparison may treat sufficiently small differences as clinically negligible. A forecasting system may regard a change as irrelevant if it does not alter operational decisions.

The equivalence margin makes that substantive criterion explicit.

It also creates an opportunity for misuse.

If the bound (Delta) is chosen after seeing the confidence interval, the test no longer represents an independent scientific standard. A researcher can always make equivalence easier to establish by widening the region declared negligible.

The scientific work therefore begins before the test is run.

What magnitude would matter, and why?

## Statistical power belongs to the design

The same problem can be expressed prospectively through statistical power.

Power is the probability that a statistical procedure will reject the null hypothesis under a specified alternative, assuming the design and model used in the calculation are correct.

For a simple two-group comparison with equal group sizes (n), common standard deviation (sigma), and a difference in population means (delta), the standard error of the difference is approximately

[
s
=
sigma
sqrt{rac{2}{n}}.
]

The noncentrality parameter for a normal approximation is

[
lambda
=
rac{delta}{s}
=
rac{delta}{sigma}
sqrt{rac{n}{2}}.
]

For a two-sided test at level (alpha=0.05), approximate power is

[
Prleft(
|Z+lambda|>1.96
ight),
]

where (Zsimmathcal N(0,1)).

Suppose the scientifically important standardised effect is

[
rac{delta}{sigma}=0.50.
]

With

[
n=16
]

per group,

[
lambda
=
0.50
sqrt{rac{16}{2}}
approx
1.414.
]

The resulting power is only about 0.29.

A study with this design will therefore fail to reject the zero null in most repeated experiments even when the true standardised effect is 0.50.

A non-significant result from such a design would not be surprising under a scientifically meaningful effect.

Now increase the design to

[
n=64
]

per group. Then

[
lambda
=
0.50
sqrt{rac{64}{2}}
approx
2.828.
]

Power rises to approximately 0.81.

The second design has a much better chance of distinguishing the target effect from zero.

Power therefore matters before the data are observed because it describes how informative a planned design is expected to be under specified alternatives.

It does not tell us the probability that the null hypothesis is true after a non-significant result.

## Low power creates a mixture of silence and exaggeration

Low power has another consequence that is sometimes overlooked.

If a study only crosses the significance threshold when random variation pushes the estimate far enough from zero, the significant estimates that survive selection can be exaggerated in magnitude.

Suppose the true effect is positive but modest. In an imprecise study, estimates will vary widely around that value. Small and moderate observed effects often fail to reach the significance threshold and disappear into the "non-significant" category. More extreme estimates are preferentially labelled discoveries.

The resulting literature can therefore contain two apparently contradictory messages:

non-significant small studies interpreted as evidence of no effect, and significant small studies with effect estimates larger than the underlying effect.

Both patterns can arise from the same weak design combined with threshold-based selection.

This is another reason to treat the estimated effect and its uncertainty as primary information rather than dividing studies into significant and non-significant categories.

## Post-hoc power does not rescue a completed null result

After observing a non-significant result, researchers sometimes calculate power using the observed effect estimate and report that the study had low "observed power".

That calculation adds little information.

For a fixed significance level and test structure, observed power calculated from the observed effect is largely determined by the same test statistic that generated the (p)-value. A small observed effect produces a large (p)-value and, mechanically, a low observed-power calculation.

Hoenig and Heisey described this practice as a pervasive misuse of power. Once the data have been observed, the confidence interval or likelihood-based information about the effect is more direct than a power calculation using the observed estimate as if it were the design alternative.

Prospective power asks:

> If the true effect were a specified value, how often would this design detect it?

Observed post-hoc power asks a circular question:

> If the effect were equal to the noisy estimate we just observed, how often would a design like this reject zero?

Those are not equivalent scientific tasks.

After a study is complete, the uncertainty interval shows more directly what effect magnitudes the study has and has not constrained.

## A large (p)-value is not positive evidence for the null

A (p)-value is calculated under the test hypothesis. It measures how incompatible the observed data and more extreme outcomes are with that hypothesis under the assumptions of the model and test.

It is not

[
Pr(H_0mid	ext{data}).
]

A value such as

[
p=0.80
]

does not mean there is an 80% probability that the null hypothesis is true.

It can occur because the true effect is very small and the study is precise.

It can also occur because the study is extremely noisy.

Those situations can have similar (p)-values while providing very different information.

Greenland and colleagues catalogue this and related interpretations in their discussion of common errors involving (p)-values, confidence intervals, and power. The difficulty is not that these quantities are useless. It is that a binary significance label discards the scale and precision needed for the scientific conclusion.

The comparison between Study A and Study B makes that loss visible.

## Evidence for absence requires an alternative worth excluding

The phrase "no meaningful effect" is incomplete until meaningful has been defined.

Suppose an experiment estimates a treatment difference of

[
0.02
]

with a narrow confidence interval. Whether that supports absence depends on the scale.

If an effect of 0.10 would materially change a clinical decision, the interval may or may not be narrow enough.

If only effects larger than 5 units matter operationally, the same estimate may provide overwhelming evidence that the intervention is irrelevant for the decision.

A threshold such as (Delta) therefore cannot be supplied by statistics alone.

It may come from a cost-benefit calculation, an established clinical threshold, a measurement resolution, an engineering tolerance, a policy criterion, or a theoretically justified scale. The justification should be visible because different values of (Delta) can produce different conclusions from the same data.

This is not a defect of equivalence testing.

It is an explicit version of a judgement that conventional null-hypothesis testing often leaves hidden.

A significance test with (H_0:	heta=0) asks whether the data can distinguish the effect from exact zero.

Most scientific decisions ask whether the effect is large enough to matter.

Those are different thresholds.

## Evidence of absence can also be expressed in Bayesian terms

Equivalence testing is not the only framework in which data can support a null or near-null conclusion.

A Bayesian comparison can evaluate the relative predictive support of two hypotheses or models. For example, let

[
H_0:	heta=0
]

and let (H_1) assign a prior distribution to plausible nonzero effects.

The Bayes factor comparing the models is

[
BF_{01}
=
rac{p(ymid H_0)}
{p(ymid H_1)}.
]

If the observed data are much more probable under the null model than under the specified alternative model, then

[
BF_{01}>1
]

favours (H_0) relative to that alternative.

The result depends on the alternative prior. A very broad prior and a tightly concentrated prior make different predictions and can produce different Bayes factors. This is not an incidental technicality. Evidence is always evidence relative to competing explanations.

The same conceptual point appears in equivalence testing. Evidence that effects larger than 0.50 are absent is not evidence that effects larger than 0.05 are absent.

Every claim of absence needs a description of what alternatives the data were capable of excluding.

## Repeated null results are not automatically cumulative evidence

Suppose five small studies all report (p>0.05).

It is tempting to count them as five pieces of evidence for no effect.

That conclusion does not follow from the labels alone.

If each study is imprecise, each may remain compatible with substantial positive and negative effects. Five unresolved studies do not become decisive merely because all failed to reject zero.

Combining their quantitative information may help. A meta-analysis can increase precision when the studies estimate sufficiently comparable effects and the synthesis model is appropriate. But heterogeneity, publication processes, outcome definitions, measurement differences, and study quality still matter.

The correct unit of evidence is not the number of times the phrase "not significant" appears.

It is the information supplied by the estimates and their uncertainty under the scientific model.

## Extremely precise evidence can reject both zero and importance

Another useful case occurs when an enormous study detects an effect that is statistically different from zero but too small to matter.

Suppose

[
hat	heta=0.08
]

with

[
s=0.02.
]

The test statistic is

[
Z=4,
]

so the conventional two-sided (p)-value is very small.

The 95% interval is approximately

[
[0.04, 0.12].
]

If the smallest meaningful effect is

[
Delta=0.50,
]

the result provides strong evidence that the effect is nonzero and also strong evidence that it lies within a range considered practically negligible.

There is no contradiction.

"Different from zero" and "large enough to matter" are not opposites on the same hypothesis test.

This case exposes why the language of effect present versus effect absent is often too crude. A parameter can be detectably nonzero and practically irrelevant at the same time.

The scientific conclusion should describe magnitude.

## The design should match the claim

If the intended conclusion is that an intervention is better than a comparator, the study should be designed to estimate or test superiority with adequate precision.

If the intended conclusion is that two procedures perform similarly within a meaningful tolerance, the study should be designed for equivalence.

If the intended conclusion is that a new intervention is not worse than an established one by more than a specified margin, the relevant design is non-inferiority.

These designs differ because the burden of evidence differs.

A failed superiority test cannot simply be relabelled as evidence of equivalence.

That switch changes the hypothesis after the data are known.

Schuirmann's work on the two one-sided tests procedure in bioequivalence was developed precisely because failing to detect a difference is not an adequate method for establishing equivalence. The hypotheses must be constructed around equivalence from the beginning.

This distinction is methodological, but it expresses a broader scientific principle.

An experiment should be capable of answering the question that the conclusion claims it answered.

## When is absence of evidence evidence of absence?

The answer depends on what absence means.

If absence means the parameter is exactly zero, ordinary empirical data rarely establish that proposition with certainty.

If absence means that effects larger than a scientifically justified magnitude are inconsistent with sufficiently informative observations, then evidence of absence is entirely possible.

Study A and Study B show the difference.

Both estimated

[
hat	heta=0.10.
]

Both failed to reject

[
H_0:	heta=0.
]

Study A produced a wide interval:

[
[-1.08, 1.28].
]

It did not provide evidence that important effects were absent.

Study B produced a narrow interval:

[
[-0.14, 0.34].
]

Under pre-specified equivalence bounds of

[
[-0.50, 0.50],
]

it did.

The difference is not the (p)-value category.

It is the range of scientifically relevant possibilities the data have excluded.

A null result becomes informative when the study was capable of finding effects that would have mattered and the resulting estimate is precise enough to rule those effects out. Confidence intervals, equivalence tests, likelihoods, Bayesian model comparisons, and other methods can express that information in different ways.

What they cannot do is rescue an undefined scientific question.

Before asking whether the evidence supports absence, we need to say absence of what magnitude.

Once that quantity is explicit, the statistical problem becomes much clearer.

## References

Altman, D. G., & Bland, J. M. (1995). Absence of evidence is not evidence of absence. *BMJ*, 311(7003), 485. https://doi.org/10.1136/bmj.311.7003.485

Greenland, S., Senn, S. J., Rothman, K. J., Carlin, J. B., Poole, C., Goodman, S. N., & Altman, D. G. (2016). Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations. *European Journal of Epidemiology*, 31, 337–350. https://doi.org/10.1007/s10654-016-0149-3

Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power: The pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19–24. https://doi.org/10.1198/000313001300339897

Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362. https://doi.org/10.1177/1948550617697177

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680. https://doi.org/10.1007/BF01068419
