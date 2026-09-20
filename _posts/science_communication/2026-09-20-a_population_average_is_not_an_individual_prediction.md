---
permalink: '/science-communication/a_population_average_is_not_an_individual_prediction/'
title: 'A Population Average Is Not an Individual Prediction'
date: '2026-09-20'
categories:
- Science Communication
tags:
- Causal Inference
- Treatment Effect Heterogeneity
- Potential Outcomes
- Precision Medicine
- Scientific Reasoning
author_profile: false
classes: wide
seo_title: 'Average Treatment Effects Do Not Identify Individual Treatment Effects'
seo_description: 'Randomized trials can identify average treatment effects while leaving individual benefit and harm only partially identified. A worked potential-outcomes example shows why.'
seo_type: article
excerpt: >-
  An average treatment effect is a property of a population comparison, not a
  prediction for every member of that population. Even a perfectly randomized
  trial does not reveal both potential outcomes for the same person.
summary: >-
  This article develops the distinction between population average treatment
  effects, subgroup effects, and individual treatment effects. A binary
  potential-outcomes example constructs two populations with identical
  randomized-trial arm results but radically different fractions of people who
  benefit or are harmed. The discussion then examines partial identification,
  baseline risk, absolute versus relative effects, subgroup analyses,
  individualized prediction, and the legitimate role of population averages.
keywords:
- average treatment effect
- individual treatment effect
- treatment effect heterogeneity
- potential outcomes
- subgroup analysis
- precision medicine
why_this_exists: >-
  Trial results are frequently translated from statements about groups into
  statements about individuals without acknowledging the additional assumptions
  required. This article shows mathematically what randomization identifies,
  what it leaves unidentified, and how baseline information can narrow the gap.
evidence: >-
  The Rubin potential-outcomes framework, Holland's fundamental problem of
  causal inference, an original binary counterexample with sharp elementary
  bounds on benefit and harm, methodological work on treatment-effect
  heterogeneity, and guidance on predictive approaches to heterogeneous
  treatment effects.
methodology: >-
  Define individual, average, and conditional average treatment effects using
  potential outcomes. Construct distinct joint distributions of potential
  outcomes with identical treatment and control marginals. Derive bounds on the
  fraction who benefit and are harmed, then examine how baseline risk and
  effect modifiers alter absolute treatment effects.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-statistics-scatter-correlation.jpg
  og_image: /assets/images/headers/photo-statistics-scatter-correlation.jpg
  overlay_image: /assets/images/headers/photo-statistics-scatter-correlation.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-scatter-correlation.jpg
  twitter_image: /assets/images/headers/photo-statistics-scatter-correlation.jpg
---

<!--
Development contract
Question: What does a population average treatment effect tell us about the effect of treatment on a particular individual?
Claim: Randomization can identify population average potential outcomes and their difference, but it does not generally identify the joint distribution of both potential outcomes within the same individuals.
Counterclaim: Population averages are not defective statistics. They answer important population-level questions, and conditional averages can become informative for decisions when effect modifiers and baseline risks are measured and validated.
Evidence object: Potential-outcomes notation, two binary populations with identical randomized-trial marginals but different individual benefit and harm patterns, sharp elementary bounds on those patterns, and a baseline-risk example showing scale-dependent heterogeneity.
Failure case: Treating all individual responses as unknowable, interpreting exploratory subgroups as established effect modifiers, or assuming that an individualized prediction model observes a person's true counterfactual treatment effect.
Reader payoff: Distinguish ATE, CATE, and ITE; understand why randomization does not reveal both outcomes for one person; recognise when absolute effects differ despite a common relative effect; and interpret personalised treatment claims more carefully.
Exclusions: Giving medical treatment advice, claiming that subgroup or prediction models cannot improve decisions, and treating heterogeneity observed on one effect scale as automatically present on every other scale.
-->

Randomized trials are designed to answer a counterfactual question. What would happen, on average, if a population received one treatment rather than another?

That is already a demanding scientific problem. Randomization is powerful because it allows the outcomes observed in different groups to estimate quantities that cannot be observed simultaneously in the same group.

The resulting answer is usually an average.

Problems begin when the average is silently converted into a statement about a particular person.

If a trial reports that an intervention improves an outcome by 20 percentage points on average, it does not follow that every participant improves by 20 percentage points. For a binary outcome, such a statement would not even make sense at the individual level. A person either experiences the event or does not under each treatment condition.

The deeper difficulty is that the individual causal effect contains two outcomes, while the individual can reveal only one of them.

This is not a limitation of sample size. It is built into the structure of the causal question.

## Two outcomes exist in the model, but only one can be observed

Let

[
Y_i(1)
]

denote the outcome person (i) would experience under treatment, and let

[
Y_i(0)
]

denote the outcome the same person would experience under control.

The individual treatment effect is

[
	au_i
=
Y_i(1)-Y_i(0).
]

For a continuous outcome, (	au_i) is an ordinary difference. For a binary outcome with (Y=1) representing success, (	au_i) can take three values:

[
	au_i=
egin{cases}
1, & 	ext{treatment changes failure into success},\
0, & 	ext{the outcome is the same under both conditions},\
-1, & 	ext{treatment changes success into failure}.
end{cases}
]

The problem is immediate. If person (i) receives treatment, we observe (Y_i(1)), but (Y_i(0)) is counterfactual. If the person receives control, the reverse is true.

We therefore never observe

[
igl(Y_i(1),Y_i(0)igr)
]

as a pair for the same person at the same moment.

Holland described this as the fundamental problem of causal inference. Rubin's potential-outcomes framework makes the same structure explicit.

Randomization does not remove this missing potential outcome. It solves a different problem.

## What randomization identifies

Suppose treatment assignment is random. Then, apart from sampling variation and complications such as non-adherence or missing data, the treatment group represents the population under treatment and the control group represents the population under control.

This allows us to estimate

[
mathbb E[Y(1)]
]

and

[
mathbb E[Y(0)].
]

Their difference is the average treatment effect,

[
operatorname{ATE}
=
mathbb E[Y(1)-Y(0)]
=
mathbb E[Y(1)]-mathbb E[Y(0)].
]

The equality follows from linearity of expectation. It does not require us to observe both potential outcomes for the same individual.

That is precisely why averages are identifiable even when individual effects are not.

The trial identifies the marginal distribution of outcomes under treatment and the marginal distribution under control. It does not automatically identify how those two potential outcomes are paired inside individuals.

That missing pairing can matter enormously.

## Two populations can produce the same trial result

Consider a binary outcome where (1) means success.

Suppose a very large randomized trial finds

[
Pr(Y(1)=1)=0.60
]

and

[
Pr(Y(0)=1)=0.40.
]

The average treatment effect on the risk-difference scale is therefore

[
operatorname{ATE}
=
0.60-0.40
=
0.20.
]

Treatment increases the probability of success by 20 percentage points on average.

Now consider the four possible joint potential-outcome types.

| Type | (Y(1)) | (Y(0)) | Individual effect |
| --- | ---: | ---: | ---: |
| Always succeeds | 1 | 1 | 0 |
| Benefits | 1 | 0 | 1 |
| Harmed | 0 | 1 | -1 |
| Never succeeds | 0 | 0 | 0 |

A randomized parallel-group trial does not observe these four categories directly because it never observes both columns for the same person.

Now construct **Population A**:

| Type | Proportion |
| --- | ---: |
| Always succeeds | 0.40 |
| Benefits | 0.20 |
| Harmed | 0.00 |
| Never succeeds | 0.40 |

Under treatment, success occurs among the always-successful and those who benefit:

[
0.40+0.20=0.60.
]

Under control, only the always-successful group succeeds:

[
0.40.
]

The trial therefore observes a 20 percentage point average benefit.

In this population, 20% benefit and nobody is harmed.

Now construct **Population B**:

| Type | Proportion |
| --- | ---: |
| Always succeeds | 0.10 |
| Benefits | 0.50 |
| Harmed | 0.30 |
| Never succeeds | 0.10 |

Under treatment,

[
0.10+0.50=0.60.
]

Under control,

[
0.10+0.30=0.40.
]

The randomized trial produces exactly the same treatment and control success rates:

[
60%
quad	ext{versus}quad
40%.
]

The average treatment effect is again

[
0.20.
]

But the individual response structure is radically different. Half the population benefits and 30% is harmed.

The observed trial marginals cannot distinguish these two worlds.

The average is identical. The distribution of individual effects is not.

## The fraction who benefit is only partially identified

The same example can be written algebraically.

Let

[
a=Pr(Y(1)=1,Y(0)=1),
]

[
b=Pr(Y(1)=1,Y(0)=0),
]

[
c=Pr(Y(1)=0,Y(0)=1),
]

and

[
d=Pr(Y(1)=0,Y(0)=0).
]

The treatment arm tells us

[
a+b=0.60,
]

while the control arm tells us

[
a+c=0.40.
]

Subtracting gives

[
b-c=0.20.
]

This identifies the average treatment effect because the proportion who benefit minus the proportion who are harmed is 0.20.

It does not identify (b) and (c) separately.

Because all probabilities must be non-negative,

[
0le ale0.40.
]

Since

[
b=0.60-a,
]

the fraction who benefit can range from

[
0.20
]

to

[
0.60.
]

Similarly,

[
c=0.40-a,
]

so the fraction harmed can range from

[
0
]

to

[
0.40.
]

The trial therefore establishes

[
0.20le
Pr(	ext{benefit})
le0.60
]

and

[
0le
Pr(	ext{harm})
le0.40
]

without additional assumptions or information.

These are not confidence intervals caused by a finite sample. Even an infinitely large randomized trial with the same marginal probabilities would retain this ambiguity.

The uncertainty comes from non-identification of the joint potential-outcome distribution.

Work on probabilities of causation and treatment-benefit inequality develops this problem formally. Huang and colleagues, for example, study bounds on the fraction of a population that benefits from treatment when randomized data identify outcome distributions but not the person-level pairing of potential outcomes.

## An assumption can close the gap, but then the assumption is doing work

Suppose we impose the condition that treatment can never convert a success under control into a failure under treatment.

In the binary example, this is

[
Y_i(1)ge Y_i(0)
]

for every individual.

This is a monotonicity assumption.

It rules out the harmed category, so

[
c=0.
]

Since

[
b-c=0.20,
]

we then obtain

[
b=0.20.
]

Under monotonicity, exactly 20% benefit.

But the randomized trial did not establish monotonicity. The assumption supplied the information needed to select one joint distribution from several compatible with the trial data.

That does not make the assumption illegitimate. Scientific knowledge often constrains otherwise unidentified quantities.

It does mean that the conclusion has changed form.

"Twenty per cent benefit because the randomized trial showed it" is different from "twenty per cent benefit under the randomized evidence plus an assumption that nobody is harmed."

The second statement makes the inferential structure visible.

## Subgroup averages are still averages

Measured baseline characteristics can make treatment-effect estimates more relevant to a particular person.

Let (X) denote baseline information such as age, disease severity, a biomarker, previous exposure, or another characteristic measured before treatment.

A conditional average treatment effect is

[
	au(x)
=
mathbb E[Y(1)-Y(0)mid X=x].
]

If (	au(x)) differs across values of (x), treatment effects are heterogeneous with respect to that characteristic.

This can be scientifically and practically important.

But (	au(x)) remains an average among people who share the specified covariate pattern. It is not generally the unobservable individual effect

[
Y_i(1)-Y_i(0).
]

This distinction matters because the word "individualized" is often used for models that actually estimate conditional average risks.

For a binary outcome, an individualized treatment-effect model may estimate

[
Pr(Y(1)=1mid X=x)
]

and

[
Pr(Y(0)=1mid X=x),
]

then report their difference,

[
delta(x)
=
Pr(Y(1)=1mid X=x)
-
Pr(Y(0)=1mid X=x).
]

This can be an excellent decision quantity. It estimates how treatment changes outcome probability among individuals with covariates like (x).

It still does not reveal both potential outcomes of the actual person standing in front of us.

Hoogland and colleagues make this distinction explicit in their tutorial on individualized treatment-effect prediction: what is called individualized prediction is typically a conditional expected treatment effect for a covariate profile rather than an observed person-specific causal contrast.

## Baseline risk alone can create important differences in absolute benefit

Treatment-effect heterogeneity does not always require a biological interaction that changes the relative effect of treatment.

Suppose an intervention reduces the relative risk of an adverse outcome by 20% across all baseline-risk groups:

[
RR=0.80.
]

Consider a low-risk group with untreated risk

[
p_0=0.05.
]

Under treatment,

[
p_1=0.80(0.05)=0.04.
]

The absolute risk reduction is

[
ARR=0.05-0.04=0.01.
]

Now consider a higher-risk group with untreated risk

[
p_0=0.50.
]

Under the same relative effect,

[
p_1=0.80(0.50)=0.40,
]

giving

[
ARR=0.50-0.40=0.10.
]

The relative effect is identical.

The absolute benefit differs by a factor of ten.

If one chooses to express the result as a number needed to treat,

[
NNT=rac{1}{ARR},
]

the corresponding values are

[
NNT=100
]

and

[
NNT=10.
]

These are population-level quantities under the stated risks. They do not mean that exactly one identifiable person among every ten will benefit.

The example instead shows why baseline prognosis can be central to treatment decisions even when there is little evidence that treatment responsiveness itself differs biologically across groups.

Dahabreh, Hayward, and Kent emphasise this distinction between person-level heterogeneity and group-level heterogeneity, and the importance of absolute risk when translating trial evidence toward individual decisions.

## Heterogeneity depends on the effect scale

The previous example reveals another difficulty.

A treatment can have a constant relative effect and a varying absolute effect.

Conversely, a constant absolute effect generally implies a changing relative effect when baseline risks differ.

Suppose two groups have control risks (p_{0a}) and (p_{0b}). If treatment subtracts a constant risk difference (d),

[
p_1=p_0-d,
]

then the relative risk is

[
RR
=
rac{p_0-d}{p_0}
=
1-rac{d}{p_0}.
]

The relative effect therefore depends on baseline risk.

Claims that "the treatment effect is the same in every subgroup" are incomplete unless the effect measure is specified.

Risk difference, relative risk, odds ratio, hazard ratio, and differences on transformed continuous scales can produce different patterns of apparent heterogeneity.

This is not merely a technical inconvenience. Different scales answer different questions.

For an individual decision, absolute outcome probabilities under each option are often more interpretable than a single relative effect detached from baseline risk.

## Subgroup discovery can manufacture heterogeneity

Recognising that average effects can hide heterogeneity does not justify searching every possible subgroup until different effects appear.

Suppose a trial examines treatment interactions with age, sex, baseline severity, five biomarkers, region, previous treatment, smoking status, body mass, and several arbitrary cut points within each continuous variable.

Enough comparisons will eventually generate unusual subgroup estimates by chance.

Small subgroups also have larger sampling error. An apparently dramatic treatment effect in one subgroup and a modest effect in another does not establish that the true effects differ.

The relevant hypothesis concerns interaction.

If a model is

[
Y
=
eta_0
+
eta_1A
+
eta_2X
+
eta_3AX
+
arepsilon,
]

then heterogeneity with respect to (X) is represented by the interaction term

[
eta_3.
]

Testing treatment separately within each subgroup and observing significance in one subgroup but not another is not equivalent to testing whether the subgroup effects differ.

Rothwell's discussion of subgroup analyses emphasises prespecification, biological or clinical justification, limited numbers of important subgroup questions, and appropriate interaction testing. Multiplicity and low power make post hoc subgroup claims particularly fragile.

The lesson is not that subgroup effects are unimportant.

It is that evidence of heterogeneity requires its own statistical design.

## More flexible models do not remove the counterfactual problem

Modern causal machine learning can estimate conditional treatment effects using many predictors and flexible functional forms.

Methods such as causal forests, meta-learners, Bayesian models, and other heterogeneous-treatment-effect estimators can be useful when simple one-variable subgroup analyses are inadequate.

They do not observe the missing potential outcome.

Their target is usually something like

[
	au(x)
=
mathbb E[Y(1)-Y(0)mid X=x].
]

To estimate this quantity reliably, the model must learn treatment-outcome relationships across covariate space. That creates familiar statistical problems.

Flexible models can overfit apparent treatment interactions. Rare covariate profiles can have little effective sample size. A model developed in one population may not transport to another. Calibration of outcome risks under each treatment matters. Model selection performed on the same data used for evaluation can exaggerate heterogeneity.

The PATH statement developed by Kent and colleagues distinguishes risk modelling from effect modelling and provides guidance for predictive analyses of heterogeneous treatment effects.

The central point is conceptual before it is computational.

A more complicated prediction algorithm can use more information about (X). It does not transform an unobserved counterfactual into an observed individual response.

## The best treatment rule is a different estimand from the individual effect

Decision making does not always require identification of the complete distribution of individual causal effects.

Suppose a model estimates

[
mu_1(x)
=
mathbb E[Y(1)mid X=x]
]

and

[
mu_0(x)
=
mathbb E[Y(0)mid X=x].
]

A treatment rule may choose treatment when

[
mu_1(x)>mu_0(x)
]

for an outcome where larger values are desirable.

This rule can improve average outcomes even though the actual pair

[
igl(Y_i(1),Y_i(0)igr)
]

remains unknown for every person.

That distinction is important.

The scientific question "what is this person's true individual treatment effect?" can remain unidentified while the decision question "which option has the better expected outcome for people with this information?" is answerable under additional modelling assumptions.

Decision theory operates on expected consequences. It does not require clairvoyance about an individual's counterfactual future.

This is one reason personalised prediction can be useful even though literal person-specific causal effects remain unobservable.

## Repeated treatment within the same person can sometimes provide more information

The impossibility of observing two potential outcomes at the same moment does not mean that within-person evidence is always unavailable.

In some settings, treatments can be started, stopped, and repeated while the relevant condition remains sufficiently stable. N-of-1 trials can alternate treatment periods within one participant and compare repeated outcomes.

Such designs can provide unusually direct evidence about treatment response for that person.

They require strong conditions.

The treatment effect must be reversible or sufficiently short-lived. Carryover between periods must be controlled. The underlying condition cannot change so rapidly that period effects become inseparable from treatment effects. Outcomes must be measurable repeatedly, and treatment order should preferably be randomised.

N-of-1 evidence is therefore powerful for some chronic and reversible treatment questions and unsuitable for many others, including irreversible interventions and outcomes that occur only once.

The broader point remains: identifying more individualised causal information requires additional design or assumptions. It does not emerge automatically from a population average.

## The average is not the enemy

It would be easy to turn all of this into an argument against average treatment effects.

That would be a mistake.

The ATE answers a coherent and often important question:

[
mathbb E[Y(1)-Y(0)].
]

If a health system, company, school, or government must choose one intervention for an entire target population, the population average can be directly relevant to the decision.

Average effects also provide a stable first description of causal impact. More granular estimates require more data and are usually less precise. Attempting to estimate dozens of interactions when the sample barely supports one overall comparison can produce worse inference, not better personalisation.

The error is not averaging.

The error is forgetting what was averaged.

A positive ATE tells us that benefits exceed harms on the chosen outcome scale when averaged over the target population. It does not tell us how those benefits and harms are distributed across people.

Population A and Population B demonstrate this exactly. Both have the same average effect of 0.20. One has no harmed individuals. The other has 30% harmed.

The trial marginals alone cannot determine which structure generated the average.

## From trial result to individual decision

A scientifically careful translation from population evidence to an individual decision therefore requires several layers.

The first is the population causal effect established by the study.

The second is the person's baseline outcome risk under the available alternatives.

The third is evidence about measured variables that modify treatment effects.

The fourth is uncertainty in those conditional estimates, which is usually larger than uncertainty in the overall effect.

The fifth is the balance between desirable and undesirable outcomes, because a treatment can have several effects with different importance to the individual.

The final decision may also depend on preferences, costs, reversibility, and uncertainty tolerance. Those quantities are not treatment effects, but they affect which treatment effect matters.

This is why the task of estimating the effect of an intervention in a population and the task of selecting an intervention for an individual are connected but not identical.

## What the randomized trial really gave us

Return to the original example.

The randomized trial establishes

[
Pr(Y(1)=1)=0.60
]

and

[
Pr(Y(0)=1)=0.40.
]

It therefore identifies

[
operatorname{ATE}=0.20.
]

That is substantial information.

What it does not identify is whether the population contains

[
20%	ext{ benefiting and }0%	ext{ harmed},
]

or

[
50%	ext{ benefiting and }30%	ext{ harmed},
]

or another compatible mixture.

To distinguish those possibilities, we need additional information: baseline predictors, structural assumptions, repeated observations, mechanistic knowledge, alternative experimental designs, or some combination of them.

Randomization protects the comparison of groups from confounding by treatment assignment.

It does not reveal the unobserved life each participant would have experienced under the treatment they did not receive.

That missing outcome is the reason an average causal effect can be identified while a literal individual causal effect remains hidden.

The average is therefore neither a fiction nor a personal forecast.

It is a population estimand.

Using it well requires remembering the difference.

## References

Dahabreh, I. J., Hayward, R., & Kent, D. M. (2016). Using group data to treat individuals: understanding heterogeneous treatment effects in the age of precision medicine and patient-centred evidence. *International Journal of Epidemiology*, 45(6), 2184–2193. https://doi.org/10.1093/ije/dyw125

Holland, P. W. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945–960.

Hoogland, J., IntHout, J., Belias, M., et al. (2021). A tutorial on individualized treatment effect prediction from randomized trials with a binary endpoint. *Statistics in Medicine*, 40(26), 5961–5981. https://doi.org/10.1002/sim.9154

Huang, E. J., Fang, E. X., Hanley, D. F., & Rosenblum, M. (2017). Inequality in treatment benefits: Can we determine if a new treatment benefits the many or the few? *Biostatistics*, 18(2), 308–324. https://doi.org/10.1093/biostatistics/kxw049

Kent, D. M., Paulus, J. K., van Klaveren, D., et al. (2020). The Predictive Approaches to Treatment effect Heterogeneity (PATH) Statement. *Annals of Internal Medicine*, 172(1), 35–45. https://doi.org/10.7326/M18-3667

Kent, D. M., Nelson, J., Dahabreh, I. J., Rothwell, P. M., Altman, D. G., & Hayward, R. A. (2016). Risk and treatment effect heterogeneity: re-analysis of individual participant data from 32 large clinical trials. *International Journal of Epidemiology*, 45(6), 2075–2088. https://doi.org/10.1093/ije/dyw118

Rothwell, P. M. (2005). Subgroup analysis in randomised controlled trials: importance, indications, and interpretation. *The Lancet*, 365(9454), 176–186. https://doi.org/10.1016/S0140-6736(05)17709-5

Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Tian, J., & Pearl, J. (2000). Probabilities of causation: Bounds and identification. *Annals of Mathematics and Artificial Intelligence*, 28, 287–313.
