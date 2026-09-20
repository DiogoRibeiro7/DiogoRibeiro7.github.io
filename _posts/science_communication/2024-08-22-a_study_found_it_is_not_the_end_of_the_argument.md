---
permalink: '/science-communication/a_study_found_it_is_not_the_end_of_the_argument/'
title: 'A Study Found It Is Not the End of the Argument'
date: '2024-08-22'
categories:
- Science Communication
tags:
- Scientific Literacy
- Statistical Power
- External Validity
- Replication
- Evidence
author_profile: false
classes: wide
seo_title: 'A Study Found It Is Not the End of the Argument'
seo_description: 'A small study can be real, significant and useful while supporting a much narrower claim than the public generalisation made from it. Power, uncertainty, population and replication all matter.'
seo_type: article
excerpt: >-
  "A study found" is the beginning of an evidence assessment, not the end.
  Sample size, uncertainty, population, outcome, design, replication and the
  surrounding literature determine how far the conclusion can legitimately travel.
summary: >-
  This article examines how a scientifically valid study can be stretched into
  an invalid public claim. A worked two-group example with twenty participants
  per arm has only about 15.8% power for a true effect of 0.3 standard deviations.
  Conditional on obtaining a positive significant result, the expected reported
  effect rises to about 0.79 standard deviations. The article then separates
  statistical significance from magnitude, internal validity from external
  validity, exploratory from confirmatory evidence, and one study from the
  accumulated literature.
keywords:
- small studies
- statistical power
- external validity
- generalisation
- replication
- scientific communication
why_this_exists: >-
  Public scientific claims are often defended by pointing to one paper that
  technically supports part of the statement. The important question is not
  whether a citation exists, but how much inferential distance lies between the
  study that was conducted and the conclusion being communicated.
evidence: >-
  An original low-power two-group calculation, significance-conditioned
  magnitude inflation, methodological work on Type S and Type M errors, published
  analyses of low statistical power, external-validity literature and research
  on false-positive risk and selective scientific evidence.
methodology: >-
  Construct a two-group normal model with twenty observations per group,
  unit variance and a true standardized effect of 0.3. Derive approximate
  two-sided power, the significance threshold and the expected effect estimate
  conditional on positive significance. Use the example to separate estimation,
  testing, generalisation and evidence synthesis.
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
Question: What can one statistically significant study legitimately establish?
Claim: A study can provide valid evidence while still supporting only a narrow conclusion. Sample size, uncertainty, population, endpoint, design, multiplicity, replication and the surrounding literature determine how far the result can be generalised.
Counterclaim: Small studies are not inherently bad. Carefully designed small experiments can be highly informative when effects are large, measurements are precise, interventions are controlled, populations are rare or the scientific question is mechanistic.
Evidence object: Two-group normal experiment with n=20 per arm, true standardized effect 0.3, approximate power of 15.8%, significance-conditioned expected estimate near 0.79, confidence-interval calculations and a sequence of scope transformations from study result to public claim.
Failure case: Treating all small studies as unreliable, assuming large studies are automatically valid, or using evidence hierarchies mechanically without considering the causal question and measurement design.
Reader payoff: Ask what was actually estimated, how uncertain it is, who was studied, what endpoint was measured and whether the public claim stays inside those boundaries.
Exclusions: Evaluating named influencers, inferring intent, and claiming that one specific sample size is too small in every research setting.
-->

"A study found" is one of the most efficient sentences in public scientific communication.

It gives a claim an immediate empirical source. It distinguishes the statement from pure opinion. It can also stop the argument too early.

The fact that a study exists does not determine how strong the evidence is, how precise the estimate is, whether the measured outcome is the outcome being discussed, whether the sampled population resembles the population receiving the advice, or whether the result survives comparison with the rest of the literature.

A study can be completely genuine and still support a much narrower conclusion than the sentence built around it.

That distinction matters because poor scientific communication does not require fabricated evidence. It can be produced by real evidence that has been stretched beyond its design.

## Small is not a synonym for bad

A small study can be excellent science.

A tightly controlled laboratory experiment may need few participants to detect a large and repeatable physical or physiological effect. Rare diseases can make large samples impossible. Expensive imaging or invasive measurements can constrain sample size. Early-stage mechanistic studies may be designed to establish whether a process occurs at all rather than estimate a population effect with high precision.

The problem is therefore not

[
n	ext{ is small}.
]

The problem is the relationship among

[
n,
]

effect size,

measurement variance,

design,

and the strength of the conclusion.

A study with twenty participants can be highly informative about one question and almost uninformative about another.

Scientific communication becomes weak when the sample size is discussed only as a number rather than as part of an uncertainty calculation.

## A low-power study can produce a large significant estimate

Consider a simplified two-group experiment.

There are

[
n=20
]

participants in each group.

Assume the outcome has standard deviation

[
sigma=1
]

in both groups.

Let the true standardized mean difference be

[
delta=0.30.
]

Under a normal approximation, the standard error of the difference in group means is

[
SE
=
sqrt{
rac{1}{20}
+
rac{1}{20}
}
=
sqrt{0.1}
approx
0.316.
]

The expected z statistic under the true effect is therefore

[
rac{delta}{SE}
=
rac{0.30}{0.316}
approx
0.949.
]

For a two-sided test at level

[
alpha=0.05,
]

statistical significance requires approximately

[
|Z|>1.96.
]

The probability of significance under the true effect is only about

[
15.8%.
]

The study is not incapable of producing a significant result.

It produces one only when sampling variation pushes the estimate unusually far from zero.

That selection has consequences for the magnitude that gets reported.

## Significance filters the observed effect

A positive significant estimate must satisfy

[
hatdelta
>
1.96(SE).
]

With

[
SEapprox0.316,
]

the threshold is approximately

[
0.620.
]

The true effect is only

[
0.30.
]

A positive result therefore needs to land more than twice as far from zero as the true effect before it receives the label "statistically significant".

Conditional on crossing that threshold, the expected estimate is not 0.30.

For

[
hatdelta
sim
mathcal N(0.30,0.316^2),
]

the conditional expectation above the significance boundary is

[
mathbb E[
hatdelta
mid
hatdelta>0.620
]
=
delta
+
SE
rac{
phi(a)
}{
1-Phi(a)
},
]

where

[
a
=
rac{0.620-0.30}{0.316}
approx
1.011.
]

This gives approximately

[
0.785.
]

The underlying effect is

[
0.30.
]

Among the positive significant results generated by this design, the expected reported estimate is about

[
0.79.
]

The study can therefore be unbiased before selection while the subset of studies that become exciting enough to circulate is strongly inflated.

Gelman and Carlin call this a Type M problem, where M refers to magnitude.

## The significant estimate can be more than twice the true effect

The exaggeration ratio in the example is approximately

[
rac{0.785}{0.30}
approx
2.62.
]

A communicator reading only the significant study sees an effect around 0.8 standard deviations.

The underlying data-generating process contains an effect of 0.3.

Nothing dishonest occurred in the hypothetical experiment.

The difference comes from low power combined with selection on statistical significance.

This is one reason small significant studies can look unusually impressive.

Button and colleagues emphasised the same broader problem in their review of low-powered neuroscience. Low power reduces the chance of detecting a true effect and, when significant results are selected, can contribute to exaggerated effect estimates and poor reproducibility.

The lesson is not "ignore small studies".

It is "interpret the magnitude through the design that generated it".

## A confidence interval makes the uncertainty visible

Suppose the observed estimate is

[
hatdelta=0.78.
]

Using the approximate standard error

[
0.316,
]

a 95% confidence interval is

[
0.78
pm
1.96(0.316).
]

This gives approximately

[
[0.16, 1.40].
]

The point estimate looks large.

The compatible range is broad.

An article that reports only

[
hatdelta=0.78
]

and the p-value hides most of the inferential uncertainty.

A good public summary would need to preserve the fact that the data remain compatible with effects much smaller than the headline estimate.

## Statistical significance does not identify practical importance

Suppose a study reports

[
p=0.03.
]

That tells us something about the incompatibility of the observed statistic with a specified null model under the assumptions of the test.

It does not tell us whether the effect is:

- large
- clinically important
- economically relevant
- precisely estimated
- replicable
- generalisable

These are separate questions.

The public phrase

> the effect was significant

often silently becomes

> the effect was important.

That transformation has no mathematical justification.

## The endpoint matters as much as the sample size

Suppose a small study measures a biomarker.

The intervention changes the biomarker.

That can be an important result.

It does not automatically establish a change in disease, symptoms, mortality or long-term health.

Let

[
M
]

be the measured biomarker and

[
Y
]

the outcome that matters to the public claim.

The study estimates

[
Delta M.
]

The public statement is about

[
Delta Y.
]

An additional causal argument is required.

The strength of that argument depends on whether (M) is a validated surrogate, whether interventions that change (M) reliably change (Y), and whether other pathways offset the effect.

A small mechanistic study can be perfectly designed for (Delta M) while being incapable of answering the question about (Delta Y).

## The population defines the scope of the evidence

Suppose a study recruits forty healthy men aged 18 to 25 who already perform resistance training.

The result concerns that sample and the population to which the design can reasonably generalise.

A public claim about

[
	ext{all adults}
]

requires additional assumptions.

A claim about:

- older adults
- women
- people with cardiovascular disease
- people taking medication
- sedentary populations

moves progressively further from the observed population.

Rothwell's work on external validity makes this point directly. Internal validity asks whether the study estimated the effect correctly for the participants and conditions studied. External validity asks whether that effect applies elsewhere.

The first does not guarantee the second.

## Generalisation is a model

Suppose the treatment effect depends on age (A):

[
	au(A)
=
eta_0+eta_1A.
]

A study restricted to ages

[
18le Ale25
]

contains little direct information about

[
	au(70).
]

Applying the result at age 70 requires extrapolation of the treatment-effect model.

If

[
eta_1=0,
]

the effect is constant across age.

If not, it changes.

The small study cannot identify this relationship outside its observed age range without external evidence.

Generalisation is therefore not a rhetorical act.

It is an inferential model.

## A study on trained participants is not automatically a study on beginners

Training status, baseline risk, disease severity and previous exposure can all modify effects.

Suppose treatment effect depends on baseline state (X):

[
	au(X)
=
	au_0+gamma X.
]

A sample concentrated in one part of the (X) distribution can estimate the local effect well while providing weak information elsewhere.

This is why statements such as

> this works in humans

are often too broad.

Humans are not one homogeneous treatment-effect stratum.

## Study design determines which causal claim is available

Different designs answer different questions.

A randomized trial can identify causal effects under its design assumptions because treatment assignment is controlled.

An observational cohort can estimate associations and may support causal inference if confounding is adequately addressed.

A cross-sectional study measures variables at one time.

A case series documents observations without a comparison group.

An in vitro experiment establishes behaviour under laboratory conditions.

An animal study provides evidence in another organism.

These are not ranked by one universal hierarchy independent of the question.

A mechanistic animal experiment can be more informative than a huge observational database for one biological pathway.

A randomized clinical trial can be more informative for treatment efficacy in humans.

The correct question is:

> Does this design identify the claim being made?

## Exploratory evidence should remain exploratory

Small studies often contain many measurements.

Suppose researchers test:

[
20
]

biomarkers,

[
5
]

subgroups,

and

[
3
]

time points.

The space of possible comparisons becomes large.

An unexpected association can be scientifically valuable.

It can generate a hypothesis.

The problem comes when the exploratory result is communicated as though it were the prespecified primary finding.

A result discovered after looking at the data has had more opportunities to appear by chance.

Confirmatory evidence should therefore come from new data or from a design that accounts for the selection process.

## Multiple outcomes change the probability of finding something

Suppose twenty independent null hypotheses are tested at

[
alpha=0.05.
]

The probability that none is significant is

[
0.95^{20}.
]

Therefore the probability of at least one false positive is

[
1-0.95^{20}
approx
0.642.
]

That is about

[
64.2%.
]

The calculation is simplified because real outcomes are often correlated.

The principle remains.

A paper that tests many outcomes and highlights the one significant result is not equivalent to a study that prespecified one primary outcome and found the same p-value.

The public citation often removes that distinction.

## One study belongs inside a literature

Suppose one small study finds a large effect.

Several larger studies find effects near zero.

A communicator cannot resolve the conflict by saying:

> There is a study showing it works.

That sentence is technically true and scientifically incomplete.

The relevant evidence object is now the collection of studies.

We need to ask:

- Which designs are stronger?
- Which measurements are closer to the outcome?
- Are the populations comparable?
- Are effects heterogeneous?
- Is there publication bias?
- Was one result exploratory?
- Are there systematic reviews?
- Are the large studies precise enough to exclude the small study's magnitude?

A citation establishes existence.

Evidence synthesis establishes context.

## Large studies are not automatically better

The reverse simplification is also wrong.

A dataset with one million observations can provide extremely precise estimates of a biased quantity.

If treatment assignment is confounded,

[
nightarrowinfty
]

does not remove the confounding.

If measurement is systematically wrong, more observations estimate the wrong measurement process more precisely.

If the target population differs from the sampled population, sample size does not automatically repair transportability.

The advantage of a larger study is reduced sampling uncertainty, all else equal.

"All else equal" does substantial work.

## A systematic review can still inherit weak evidence

Evidence synthesis is not magic.

A meta-analysis of biased studies can produce a precise biased pooled estimate.

Heterogeneous definitions can make studies difficult to combine.

Publication bias can make the observed literature unrepresentative of completed studies.

A review can also be methodologically poor.

The reason to prefer synthesis over cherry-picking is not that reviews are infallible.

It is that a good review makes the selection of evidence explicit and attempts to evaluate the complete relevant literature rather than choosing one convenient result.

## The claim should preserve the study's nouns

One practical communication rule is to preserve the important nouns from the study.

If the paper studied:

> resistance-trained men aged 18 to 25

the public statement should not silently become:

> people.

If it measured:

> fasting biomarker X

the public statement should not silently become:

> health.

If it tested:

> 500 mg daily for eight weeks

the public statement should not silently become:

> taking this supplement.

If it found:

> an association

the public statement should not silently become:

> a cause.

Every noun that disappears can widen the claim.

## The claim should preserve the study's numbers

A second rule is to preserve magnitude.

Suppose the study estimates an improvement of

[
2.1%
]

with a confidence interval from

[
0.2%
]

to

[
4.0%.
]

The public version:

> dramatically improves performance

contains more information than the study provided.

Words such as:

- dramatically
- strongly
- substantially
- dangerous
- protective
- transformative

are quantitative claims disguised as adjectives.

If the magnitude matters, report the magnitude.

## The claim should preserve uncertainty

Suppose two studies estimate:

[
0.80pm0.60
]

and

[
0.35pm0.08.
]

The first has a larger point estimate.

The second is much more precise.

A communicator selecting only the largest estimate is selecting noise as well as signal.

Scientific communication should therefore retain enough uncertainty that the audience can distinguish an impressive point estimate from an impressive body of evidence.

## Repetition changes the status of a result

A single small experiment can establish an interesting observation.

If independent groups reproduce the effect with similar estimates under similar conditions, confidence changes.

If the effect appears across different populations and measurement methods, generality changes.

If larger studies shrink the estimate, the magnitude changes even if the direction remains.

Replication does not turn uncertainty into certainty.

It changes the evidential structure.

A claim should evolve with that structure.

## A real result can support a false generalisation

Suppose the following study-level statement is true:

> In forty trained young men, intervention X increased biomarker Y over six weeks.

Now consider the statement:

> X improves health.

The first statement can be entirely correct while the second is unsupported.

The problem is not fake science.

It is inferential distance.

A public claim can fail because it crosses too many untested bridges:

[
	ext{sample}
ightarrow
	ext{population},
]

[
	ext{biomarker}
ightarrow
	ext{clinical outcome},
]

[
	ext{short term}
ightarrow
	ext{long term},
]

[
	ext{association}
ightarrow
	ext{causation},
]

[
	ext{one study}
ightarrow
	ext{established knowledge}.
]

The more bridges crossed, the more evidence is required.

## Confidence should decrease as inferential distance increases

Suppose the study directly supports claim (C_0).

A communicator wants to make broader claim (C_1).

Then broader claim (C_2).

Each transformation requires assumptions.

A useful conceptual model is:

[
E
ightarrow
C_0
ightarrow
C_1
ightarrow
C_2.
]

If every arrow introduces uncertainty, confidence should not remain constant across the chain.

This is where poor communication often becomes visible.

The wording becomes broader.

The confidence stays the same.

Scientific reasoning should do the opposite.

## The strongest evidence is the evidence that could have failed

A study becomes more informative when its design creates a serious opportunity for the preferred explanation to be wrong.

Randomisation challenges confounding.

Blinding challenges expectation effects.

Prespecification challenges outcome selection.

Adequate sample size challenges random exaggeration.

Independent replication challenges study-specific accidents.

External validation challenges population-specific explanations.

Converging measurement methods challenge instrument-specific artefacts.

The best question is not:

> Can I find a study that supports the claim?

It is:

> What evidence would have contradicted the claim, and did the research expose it to that test?

## "A study found" should trigger questions, not end them

When a scientific claim is defended with one paper, the next questions are straightforward.

What was the sample size?

What effect was estimated?

What was the uncertainty interval?

What population was studied?

What was the comparator?

Was the endpoint direct or surrogate?

Was the analysis prespecified?

How many outcomes were tested?

Was the finding replicated?

What does the rest of the literature show?

How far is the public claim from the actual study conclusion?

Those questions are not pedantry.

They are the scientific method applied to communication.

## The toy experiment shows why the distinction matters

The hypothetical experiment had:

[
20
]

participants per group,

unit outcome variance,

and a true standardized effect of

[
0.30.
]

Its approximate power was only

[
15.8%.
]

A positive significant result had to exceed roughly

[
0.62.
]

Conditional on positive significance, the expected reported effect was approximately

[
0.79.
]

The design can therefore generate a published estimate more than twice the underlying effect without misconduct and without a calculation error.

That is why "the study was significant" is not enough.

And even if the estimate were perfectly accurate for the sample, additional questions would remain about population, outcome, duration and generalisation.

A study is evidence.

It is not the entire argument.

## References

Button, K. S., Ioannidis, J. P. A., Mokrysz, C., Nosek, B. A., Flint, J., Robinson, E. S. J., & Munafò, M. R. (2013). Power failure: why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience*, 14, 365–376. https://doi.org/10.1038/nrn3475

Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S and Type M errors. *Perspectives on Psychological Science*, 9(6), 641–651. https://doi.org/10.1177/1745691614551642

Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124. https://doi.org/10.1371/journal.pmed.0020124

Rothwell, P. M. (2005). External validity of randomised controlled trials: To whom do the results of this trial apply? *The Lancet*, 365(9453), 82–93. https://doi.org/10.1016/S0140-6736(04)17670-8

Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129–133. https://doi.org/10.1080/00031305.2016.1154108
