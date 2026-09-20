---
permalink: '/science-communication/not_all_studies_answer_the_same_question/'
title: 'Not All Studies Answer the Same Question'
date: '2025-05-22'
categories:
- Science Communication
tags:
- Scientific Method
- Study Design
- Systematic Reviews
- Meta Analysis
- Research Methods
author_profile: false
classes: wide
seo_title: 'Not All Studies Answer the Same Question'
seo_description: 'Randomised trials, cohorts, case-control studies, mechanistic experiments and meta-analyses answer different questions. Good science begins by matching the design to the estimand.'
seo_type: article
excerpt: >-
  There is no universal ladder on which every study can be ranked. A randomised
  trial, cohort, diagnostic study, mechanistic experiment and meta-analysis can
  each be the right design for a different scientific question.
summary: >-
  This article develops the scientific method as a sequence from question and
  estimand to design, measurement, analysis, criticism, replication and evidence
  synthesis. It maps the main study designs to the questions they can answer,
  explains the difference between systematic reviews and meta-analysis, and
  describes current good practices including protocols, preregistration,
  prespecified outcomes, power and precision, randomisation, allocation
  concealment, blinding, missing-data plans, effect sizes, uncertainty,
  sensitivity analysis, open science and design-specific reporting standards.
keywords:
- scientific method
- study designs
- randomised controlled trials
- cohort studies
- case control studies
- systematic reviews
- meta analysis
- research best practices
why_this_exists: >-
  Public discussion often treats research designs as a simple hierarchy and
  treats meta-analysis as automatically conclusive. That obscures the more
  important question: what scientific quantity was the study designed to
  identify, under which assumptions, and with which sources of bias?
evidence: >-
  Current reporting standards including CONSORT 2025, SPIRIT 2025, STROBE and
  PRISMA 2020; the Cochrane Handbook for evidence synthesis; causal-inference
  literature; and established methodological work on bias, randomisation,
  observational studies and transparent reporting.
methodology: >-
  Organise research designs by the estimand they target rather than by a single
  evidence pyramid. For each major design, state the characteristic data
  structure, inferential strength, principal failure modes and appropriate
  reporting framework. Derive the basic fixed and random effects meta-analytic
  estimators to show how synthesis depends on both within-study uncertainty and
  between-study heterogeneity.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/study-design-question.jpg
  og_image: /assets/images/headers/study-design-question.jpg
  overlay_image: /assets/images/headers/study-design-question.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/study-design-question.jpg
  twitter_image: /assets/images/headers/study-design-question.jpg
---

<!--
Development contract
Question: How should a reader decide what a study can establish, and what constitutes good scientific practice across different research designs?
Claim: Study quality cannot be inferred from a universal design ranking alone. The design must be matched to the estimand, conducted in a way that controls the relevant biases, analysed transparently and interpreted within the limits of the population, measurement and design.
Counterclaim: Design hierarchies are still useful shorthand for some questions, especially intervention effects, because randomisation can remove important sources of confounding. The mistake is extending one hierarchy to every scientific question.
Evidence object: Design-by-question map, potential-outcome formulation for causal effects, prevalence and diagnostic estimands, fixed and random effects meta-analysis equations, and current reporting standards.
Failure case: Calling every observational study weak, treating every randomised trial as definitive, assuming a meta-analysis automatically upgrades weak studies, or confusing reporting guidelines with guarantees of good conduct.
Reader payoff: Identify which study design fits which question, recognise the main biases of each design, distinguish systematic review from meta-analysis and use a practical checklist for scientific quality before accepting a public claim.
Exclusions: Ranking named journals, universities or researchers, claiming that one methodological framework is sufficient for all disciplines, and reducing qualitative or mechanistic research to lower quality simply because it answers a different class of question.
-->

The scientific method is often presented as a short sequence: ask a question, state a hypothesis, run an experiment, and draw a conclusion. That version is useful for teaching, but it hides most of the work that determines whether a scientific claim is actually defensible.

Real research is iterative. Measurements reveal problems with the original question. Experiments expose assumptions that were previously invisible. Replications shrink some effects and strengthen others. Observational data generate causal hypotheses that later become experimental questions. Mechanistic work explains why the same intervention behaves differently across populations. Systematic reviews show that apparently contradictory studies were often estimating different quantities.

A better abstraction is

$$
Q
\rightarrow
E
\rightarrow
D
\rightarrow
M
\rightarrow
A
\rightarrow
I
\rightarrow
R,
$$

where $Q$ is the scientific question, $E$ the estimand, $D$ the design, $M$ the measurement process, $A$ the analysis, $I$ the interpretation, and $R$ replication, criticism and revision.

The order matters.

If the estimand is vague, no statistical method can repair the question later. If the measurement does not represent the target construct, a large sample only estimates the wrong quantity more precisely. If the design does not identify the causal contrast being claimed, sophisticated modelling can hide rather than remove the problem.

## Start with the estimand

Before asking whether a study is a randomised trial, cohort or meta-analysis, ask what quantity the investigators wanted to know.

Examples include prevalence,

$$
\pi=P(Y=1),
$$

conditional prediction,

$$
\mathbb E[Y\mid X=x],
$$

diagnostic probability,

$$
P(Y=1\mid T=1),
$$

average causal effect,

$$
\mathbb E[Y(1)-Y(0)],
$$

or a time-to-event distribution,

$$
F_T(t)=P(T\le t).
$$

These are different estimands.

A study design that is excellent for one may be inappropriate for another. This is why a universal evidence pyramid can mislead. It compresses a design decision into one ranking even though the scientific target changes from question to question.

## Descriptive studies estimate what exists

Suppose the question is:

> How common is condition $Y$ in population $P$?

A well-designed cross-sectional survey can estimate prevalence directly.

Randomising people to treatment and control would not improve the prevalence estimate. It would answer another question.

For descriptive inference, the central problems include representativeness, non-response, measurement quality, sampling frame and weighting. A huge convenience sample can therefore be less informative than a smaller probability sample.

The important design property is whether the observed sample supports inference about the target population.

## Cross-sectional studies measure variables at one period

A cross-sectional study measures exposure and outcome at roughly the same time.

It can estimate prevalence and associations such as

$$
P(Y=1\mid X=x).
$$

Its major causal limitation is temporality.

If physical activity and depression are negatively associated in a cross-sectional survey, several structures remain compatible:

$$
\text{activity}\rightarrow\text{depression},
$$

$$
\text{depression}\rightarrow\text{activity},
$$

or

$$
U\rightarrow\{\text{activity},\text{depression}\},
$$

where $U$ represents common causes.

The study can document the association without identifying its direction.

That is not a defective result. It becomes defective only when the interpretation changes from association to causation.

## Ecological studies use group-level observations

An ecological study analyses units such as countries, regions, schools, hospitals, firms or time periods.

A relationship between average income and mortality across countries is a group-level relationship. It does not imply that the same relationship holds among individuals inside each country.

The ecological fallacy occurs when a relationship at one level is transferred to another without justification.

Ecological studies remain useful for population policy, environmental exposure, health-system comparisons, macroeconomic variation and contextual effects. The unit of analysis must remain visible in the conclusion.

## Case reports and case series detect unusual phenomena

A case report documents one observation. A case series documents several.

These designs generally cannot estimate comparative treatment efficacy because there is no counterfactual comparison group.

They can still be scientifically important.

Rare adverse events, unexpected disease presentations and novel treatment responses are often first detected this way.

The appropriate claim is usually:

> This event occurred.

Not:

> This intervention causes this event at a known rate.

Case reports are strong for existence and signal generation. They are weak for estimating frequency and causal effects.

## Case-control studies begin with the outcome

A case-control study selects people with the outcome and a suitable comparison group without the outcome, then compares previous exposures.

A common estimand is the odds ratio,

$$
OR=
\frac{
\text{odds of exposure among cases}
}{
\text{odds of exposure among controls}
}.
$$

The design is efficient for rare outcomes because researchers do not need to follow an enormous cohort waiting for rare events to occur.

Its characteristic vulnerabilities include control selection, recall bias, exposure misclassification and confounding.

The control group is not simply "people without disease". It should represent the exposure distribution in the population that produced the cases.

## Cohort studies begin with exposure and follow outcomes

A cohort study compares outcome incidence among people with different exposures.

A prospective cohort follows participants forward. A retrospective cohort reconstructs histories from existing records.

Cohorts can estimate risk,

$$
P(Y=1\mid X=x),
$$

risk ratios,

$$
RR=
\frac{P(Y=1\mid X=1)}
{P(Y=1\mid X=0)},
$$

risk differences, incidence rates and time-to-event quantities.

Temporal ordering is clearer than in a cross-sectional study.

Confounding remains a central problem.

If

$$
X\leftarrow U\rightarrow Y,
$$

then an association between $X$ and $Y$ can exist even without a causal effect of $X$.

Adjustment can help when the relevant confounders are measured appropriately. It cannot adjust for important confounders that were never observed or were measured badly.

## Randomisation targets causal comparability

For an intervention, define potential outcomes

$$
Y_i(1)
$$

and

$$
Y_i(0).
$$

The average treatment effect is

$$
ATE
=
\mathbb E[Y(1)-Y(0)].
$$

We never observe both potential outcomes for the same person.

Random assignment creates treatment groups that are comparable in expectation before treatment. Under proper implementation, it breaks systematic dependence between baseline causes of the outcome and treatment assignment.

This makes randomised controlled trials particularly powerful for intervention effects.

But randomisation is not a magic label.

A good trial also needs appropriate allocation concealment, adherence monitoring, blinding where feasible, valid outcome measurement, missing-data handling, prespecified endpoints and transparent analysis.

A trial with poor measurement can estimate the wrong outcome very precisely. A trial with severe differential attrition can lose much of the protection created by randomisation.

## Allocation concealment and blinding solve different problems

Allocation concealment protects the randomisation process before assignment.

The person enrolling a participant should not be able to predict the next treatment allocation.

Blinding occurs after assignment.

Participants, clinicians, outcome assessors or analysts may be unaware of treatment allocation to reduce behavioural or measurement effects.

A trial can be randomised without being blinded. A trial can be blinded badly even when allocation was concealed perfectly.

Each procedure targets a different bias.

## Not every causal question can be randomised

Researchers cannot ethically randomise many long-term harmful exposures, childhood socioeconomic conditions, environmental disasters or genetic variants.

Some policy interventions are also implemented before randomisation is feasible.

Observational causal inference is therefore not optional.

Useful designs can exploit natural experiments, interrupted time series, regression discontinuity, instrumental variables, difference in differences and target-trial emulation.

The relevant question is not simply:

> Is this observational?

It is:

> What source of variation identifies the causal contrast, and what assumptions make that comparison credible?

## Quasi-experiments use structure in treatment assignment

Suppose programme eligibility changes sharply at threshold $c$.

A regression discontinuity design compares units immediately around that threshold.

If potential outcomes vary smoothly around $c$, the discontinuity can identify a local causal effect:

$$
\tau_{RD}
=
\lim_{x\downarrow c}\mathbb E[Y\mid X=x]
-
\lim_{x\uparrow c}\mathbb E[Y\mid X=x].
$$

The estimate can be internally credible while remaining local to observations near the threshold.

Internal identification and broad external validity are separate properties.

## Diagnostic studies answer a classification question

Suppose diagnostic test $T$ is compared with a reference standard for disease $D$.

Sensitivity is

$$
P(T=1\mid D=1),
$$

specificity is

$$
P(T=0\mid D=0),
$$

and positive predictive value is

$$
P(D=1\mid T=1).
$$

Predictive values depend on prevalence.

A diagnostic accuracy study can therefore be excellent science even though it says nothing about whether treating patients improves outcomes.

STARD is the relevant reporting family for diagnostic accuracy studies.

The design follows the question.

## Prognostic studies predict future outcomes

A prognostic model asks:

> Given information available now, what is the probability of outcome $Y$ later?

The target may be

$$
P(Y=1\mid X).
$$

Good practice includes calibration, discrimination, overfitting control, external validation, predictor measurement and transportability.

A model can predict well without identifying causes.

Prediction and intervention are different scientific tasks.

TRIPOD and its extensions provide reporting guidance for prediction models.

## Mechanistic studies explain processes

Mechanistic experiments may operate at molecular, cellular, organ, animal or computational levels.

They can establish that a pathway exists, identify intermediate processes and constrain causal explanations.

Their major inferential problem is translation.

A pathway observed in vitro does not automatically establish a meaningful clinical effect in vivo.

Dose, tissue exposure, compensatory mechanisms and competing pathways matter.

Mechanistic work can therefore be decisive for mechanism while remaining insufficient for estimating population-level outcome magnitude.

## Animal studies require their own standards

Preclinical animal research has design requirements analogous to human research: appropriate controls, randomisation where feasible, blinding, sample-size justification, prespecified outcomes, transparent exclusions and complete reporting.

ARRIVE provides reporting guidance for animal research.

A well-conducted mouse study is still a mouse study. The quality of the experiment and the validity of translation to humans are separate questions.

## Qualitative studies answer questions numbers may not

Some scientific questions concern experience, implementation, meaning, barriers, decision processes and organisational behaviour.

Interviews, focus groups, ethnography and related qualitative designs can answer questions that cannot be reduced to an average treatment effect.

Sampling strategy, reflexivity, coding, saturation and context become central.

COREQ and SRQR provide reporting frameworks for qualitative research.

Qualitative and quantitative evidence can complement one another rather than compete in one universal hierarchy.

## N-of-1 trials can answer individual questions

A conventional RCT estimates an average effect across participants.

An N-of-1 trial repeatedly alternates treatments within one person when the condition and intervention make that possible.

The design can be powerful when treatment effects are reversible, outcomes can be measured repeatedly, carryover can be controlled and the condition is sufficiently stable.

It is inappropriate for irreversible treatments and many one-time outcomes.

Again, the appropriate design follows the estimand.

## One useful map is question by design

| Scientific question | Typical useful designs | Main threat |
| --- | --- | --- |
| How common is it? | Cross-sectional survey, registry | Selection and measurement bias |
| What predicts it? | Cohort, prediction study | Overfitting, transportability |
| What caused it? | RCT, quasi-experiment, causal observational design | Confounding, missing data |
| What preceded a rare outcome? | Case-control | Control selection, recall bias |
| Does a mechanism exist? | Laboratory, mechanistic, animal study | Translation and dose relevance |
| How accurate is a test? | Diagnostic accuracy study | Spectrum bias, reference standard |
| What is the experience or implementation problem? | Qualitative study | Sampling, reflexivity, context |
| What does the literature show? | Systematic review, evidence synthesis | Search, bias, heterogeneity |

The table is not a ranking.

It is a matching problem.

## A systematic review is a research design for the literature

A systematic review begins with a defined question and an explicit protocol for finding, selecting and evaluating relevant studies.

A simplified workflow is

$$
\text{question}
\rightarrow
\text{eligibility criteria}
\rightarrow
\text{search}
\rightarrow
\text{screening}
\rightarrow
\text{data extraction}
\rightarrow
\text{risk of bias}
\rightarrow
\text{synthesis}.
$$

The word *systematic* refers to making this process explicit and reproducible.

A narrative review can be excellent scholarship. Its source selection is usually less formalised, making omission harder to audit.

PRISMA 2020 provides reporting guidance for systematic reviews.

A PRISMA flow diagram does not make a review methodologically strong. It makes the selection process visible.

## A systematic review does not require meta-analysis

Suppose included studies differ fundamentally in intervention, population, outcome, design, follow-up or effect measure.

Combining them numerically may produce a precise answer to a question the studies do not actually share.

A systematic review can therefore conclude that statistical pooling is inappropriate.

That is not a failed review.

It may be the most scientifically defensible synthesis.

## Meta-analysis is a statistical operation

Meta-analysis is the statistical combination of results from two or more separate studies.

Suppose study $i$ estimates effect

$$
\hat\theta_i
$$

with standard error

$$
s_i.
$$

Under a simple fixed-effect model, inverse-variance weight is

$$
w_i=\frac{1}{s_i^2}.
$$

The pooled estimate is

$$
\hat\theta_{\mathrm{FE}}
=
\frac{
\sum_i w_i\hat\theta_i
}{
\sum_i w_i
}.
$$

More precise studies receive more weight.

This can improve precision if the model and study set are appropriate.

It does not remove bias inside the component studies.

## Random effects allow underlying effects to differ

Suppose the true study-specific effects differ:

$$
\theta_i
\sim
\mathcal N(\mu,\tau^2).
$$

Here

$$
\tau^2
$$

represents between-study heterogeneity.

Random-effects weights often take the form

$$
w_i^\ast
=
\frac{1}{
s_i^2+\tau^2
}.
$$

The pooled estimate becomes

$$
\hat\mu
=
\frac{
\sum_i w_i^\ast\hat\theta_i
}{
\sum_i w_i^\ast
}.
$$

The model now asks about the mean of a distribution of study effects rather than one identical effect shared by every study.

Choosing fixed or random effects is therefore not a cosmetic software option.

It changes the model.

## Heterogeneity is information

Suppose a meta-analysis contains effects near

$$
0.1,\quad0.1,\quad0.1,\quad1.2,\quad1.3.
$$

A pooled mean can be calculated.

The more important scientific question may be why the studies differ.

Possible sources include population, dose, intervention implementation, outcome measurement, follow-up, bias and genuine effect modification.

The distribution of study results is part of the evidence.

A forest plot should be read as more than the diamond at the bottom.

## A meta-analysis cannot rescue weak inputs

Suppose ten studies are biased in the same direction.

A meta-analysis may produce a narrow confidence interval around their common bias.

Precision increased.

Validity did not.

The Cochrane Handbook explicitly warns that meta-analysis can mislead when within-study bias, study design differences, heterogeneity and reporting bias are not addressed.

"Meta-analysis" therefore does not mean "highest level of evidence" independently of what was combined.

A meta-analysis of strong studies can be strong.

A meta-analysis of weak studies can be precisely weak.

## Publication bias changes the evidence set

Suppose publication probability depends on the result:

$$
P(\text{published}\mid p<0.05)
>
P(\text{published}\mid p\ge0.05).
$$

The published literature is then a selected sample of completed research.

A systematic review can search the published literature perfectly and still miss studies that were never made public.

Trial registration, protocols, preprints and regulatory records help reveal this missingness.

They do not eliminate it automatically.

## Network meta-analysis adds indirect comparisons

Suppose treatments $A$ and $B$ have never been directly compared, but both have been compared with $C$.

A network meta-analysis can use the $A$ versus $C$ and $B$ versus $C$ evidence to infer an indirect comparison between $A$ and $B$.

This requires assumptions such as transitivity: the study sets must be sufficiently comparable for the indirect path to represent the target comparison.

Network meta-analysis is powerful.

It obtains additional comparisons by adding additional structure.

## Umbrella and scoping reviews answer still different questions

An umbrella review synthesises systematic reviews or meta-analyses. It can be useful for broad fields but must handle overlap of primary studies, inconsistent eligibility criteria and differences in review quality.

A scoping review maps a field: concepts, methods, terminology and evidence gaps. It may not attempt to estimate one pooled effect.

Calling a scoping review weak because it does not produce a pooled number misunderstands its purpose.

Again, the design follows the question.

## Good science begins before data collection

Many research failures are locked in before the first observation is recorded.

Good practice starts with a protocol that specifies the research question, population, intervention or exposure, comparator, outcomes, sample-size rationale, inclusion criteria, analysis plan, missing-data plan and subgroup strategy.

For randomised trial protocols, SPIRIT 2025 provides current reporting guidance.

The purpose is not bureaucracy.

It is to make the planned experiment distinguishable from decisions made after seeing the data.

## Preregistration separates prediction from discovery

Preregistration records key hypotheses and analyses before outcome data are examined.

It helps distinguish confirmatory analysis from exploratory analysis.

Exploration is essential to science.

The problem is presenting a discovery found after inspecting the data as if it had been predicted before inspection.

Preregistration does not guarantee good science. A poor plan can be preregistered perfectly.

Its value is provenance.

## Sample size should follow the question

A useful sample-size argument is not simply:

> We need thirty participants.

For estimation, one may target a confidence interval width.

For hypothesis testing, power can be defined against a scientifically meaningful effect.

For prevalence, precision depends on expected prevalence and sample size.

For cluster designs, intracluster correlation affects effective information.

For repeated measures, within-person correlation matters.

The sample-size argument belongs to the design.

It should not be reconstructed after the result is known.

## Power is prospective

Before data collection, power asks

$$
P(\text{reject }H_0\mid\theta=\theta_1).
$$

It helps determine whether the planned study can detect a scientifically meaningful effect.

After the study, uncertainty is better described by the estimate and its interval.

Observed or post-hoc power mostly repackages the observed p-value.

Good practice uses power to design an informative study rather than to excuse an uninformative result afterward.

## Outcomes should be prespecified

Suppose a study records twenty outcomes.

If the primary outcome is selected after inspection, the nominal error rate no longer describes the selection process.

Prespecification identifies which endpoint was intended to answer the main question.

Secondary and exploratory outcomes remain useful.

They should be labelled honestly.

CONSORT 2025 now includes updated reporting requirements and an explicit open-science section covering trial registration, protocol and statistical-analysis-plan access, data sharing, funding and competing interests.

## Multiplicity should be visible

Multiplicity can arise from multiple outcomes, treatment arms, time points, subgroups, models and interim analyses.

There is no single correction that fits every scientific context.

The first requirement is transparency.

How many opportunities existed to find the reported result?

A single p-value cannot answer that question.

## Measurement validity belongs in the design

Suppose target construct $T$ is observed through

$$
X=T+B+\varepsilon,
$$

where $B$ is nuisance variation.

A huge RCT can estimate the intervention effect on $X$ very precisely.

If $X$ poorly represents $T$, the scientific claim about $T$ remains weak.

Measurement validity, calibration and invariance are therefore part of design quality, not optional decorations.

## Missing data need a plan

Suppose outcomes are missing more often among participants doing poorly.

Complete-case analysis conditions on being observed.

That can change the comparison.

Missingness should therefore be prevented where possible, documented and analysed under explicit assumptions.

Sensitivity analyses can examine conclusions under alternative missing-data mechanisms.

SPIRIT guidance requires protocols to specify analysis populations and planned handling of missing data rather than improvising those decisions after trial completion.

## Report effect sizes and uncertainty

A study that reports only

$$
p<0.05
$$

has hidden the magnitude.

Scientific interpretation usually needs an estimate

$$
\hat\theta
$$

and uncertainty around it.

For binary outcomes, risk difference, risk ratio and odds ratio answer different questions.

For survival outcomes, relative measures should be interpreted together with absolute event risks and time.

The effect measure should match the scientific decision.

## Sensitivity analysis asks whether the result depends on arbitrary choices

Suppose the conclusion changes when one influential observation is removed, a confounder definition changes, a different missing-data assumption is used or one statistical model is replaced.

That instability belongs in the result.

A robust conclusion should survive plausible alternatives that ought not determine the science.

Sensitivity analysis is not trying every possible analysis.

It targets uncertain assumptions that could materially change the conclusion.

## Reporting guidelines improve transparency, not truth

Current reporting frameworks include CONSORT 2025 for randomised trials, SPIRIT 2025 for trial protocols, STROBE for major observational designs, PRISMA 2020 for systematic reviews, STARD for diagnostic accuracy, TRIPOD for prediction models, ARRIVE for animal studies and COREQ or SRQR for qualitative research.

The EQUATOR Network maintains a large library of design-specific reporting guidance.

These frameworks specify information readers need to judge research.

They do not certify validity.

STROBE explicitly states that it is a reporting guideline rather than a quality-assessment instrument.

CONSORT 2025 makes the same distinction.

A badly designed study can be reported transparently.

Transparency makes the weakness visible.

That is progress.

## Open science should expose the research object

Useful transparency can include protocol, preregistration, statistical analysis plan, code, data dictionaries, de-identified data where ethical and legal constraints permit, deviations from protocol, funding and conflicts of interest.

The goal is auditability.

Another researcher should be able to understand what was planned, what changed and how the result was produced.

## Replication is part of the method

A scientific result is not weakened by being tested again.

Direct replication tests stability under similar conditions.

Conceptual replication tests whether the idea survives different operationalisations.

A failed replication does not automatically prove misconduct.

A successful replication does not prove universality.

Both update the evidential structure.

## Evidence should accumulate across designs

Suppose a causal hypothesis receives support from a cohort study, a randomised trial, mechanistic work, a natural experiment and independent replication.

These designs have different failure modes.

Agreement across them can be stronger than several copies of the same design.

The logic is not simply "more studies".

It is

$$
\text{different assumptions}
+
\text{convergent result}.
$$

A common bias is less likely to explain evidence generated through genuinely different methods.

## Scientific criticism should identify the failure mechanism

Weak criticism says:

> It is observational, so ignore it.

Better criticism says:

> Exposure is self-selected, important confounders are poorly measured and the analysis does not establish comparability between groups.

Weak criticism says:

> It is an RCT, so it is definitive.

Better criticism says:

> Randomisation was appropriate, but attrition differed strongly between groups and the primary outcome changed after registration.

Weak criticism says:

> It is a meta-analysis, so this settles the question.

Better criticism says:

> The pooled estimate is precise, but most included studies are at high risk of bias and heterogeneity is substantial.

Methodological criticism should identify how error could enter the result.

## A practical reading sequence

When reading a scientific claim, ask:

1. What is the scientific question?
2. What is the estimand?
3. What population does it concern?
4. What was measured?
5. What design generated the comparison?
6. What assumptions make the estimate valid?
7. How large is the effect?
8. How uncertain is it?
9. What are the plausible biases?
10. Was the analysis prespecified or exploratory?
11. Has the result been replicated?
12. How does it fit the wider literature?

The sequence prevents one impressive feature from dominating the appraisal.

A large sample cannot erase confounding.

Randomisation cannot repair invalid measurement.

A systematic review cannot repair biased primary studies.

A meta-analysis cannot create comparability among studies that estimate different things.

## The scientific method is a system of constraints

Good scientific practice is not one ritual.

It is a set of constraints designed to make error visible.

Randomisation constrains confounding.

Allocation concealment constrains selection during enrolment.

Blinding constrains behavioural and measurement effects.

Preregistration constrains undisclosed analytic flexibility.

Power and precision calculations constrain uninformative design.

Validation constrains measurement error.

Sensitivity analysis constrains dependence on modelling choices.

Replication constrains study-specific accidents.

Systematic review constrains selective citation.

Meta-analysis constrains informal weighting of quantitative evidence.

Open methods constrain unverifiable analysis.

None of these guarantees truth.

Together they make claims harder to protect from contradiction.

That is one of the defining strengths of science.

## The design should match the question

A case report can be the right design for documenting a previously unknown event.

A cross-sectional survey can be the right design for prevalence.

A cohort can be the right design for prognosis.

A case-control study can be efficient for rare outcomes.

A randomised trial can be the strongest feasible design for many intervention effects.

A natural experiment can identify effects that cannot ethically be randomised.

A mechanistic experiment can answer why an effect occurs.

A qualitative study can explain implementation and experience.

A systematic review can organise the full evidence base.

A meta-analysis can quantify evidence when studies are sufficiently comparable.

The scientific method does not ask which label sits highest on a pyramid.

It asks whether the design, measurement and analysis make the intended conclusion defensible.

## References

Chan, A.-W., Boutron, I., Hopewell, S., Moher, D., Schulz, K. F., Collins, G. S., et al. (2025). SPIRIT 2025 statement: updated guideline for protocols of randomised trials. *BMJ*, 389, e081477.

Deeks, J. J., Higgins, J. P. T., Altman, D. G., McKenzie, J. E., & Veroniki, A. A. (2024). Analysing data and undertaking meta-analyses. In *Cochrane Handbook for Systematic Reviews of Interventions*, version 6.5.

Greenland, S., Pearl, J., & Robins, J. M. (1999). Causal diagrams for epidemiologic research. *Epidemiology*, 10(1), 37–48.

Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.

Hopewell, S., Chan, A.-W., Collins, G. S., Hróbjartsson, A., Moher, D., Schulz, K. F., et al. (2025). CONSORT 2025 statement: updated guideline for reporting randomised trials. *BMJ*, 389, e081123. https://doi.org/10.1136/bmj-2024-081123

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71. https://doi.org/10.1136/bmj.n71

Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology* (3rd ed.). Lippincott Williams & Wilkins.

von Elm, E., Altman, D. G., Egger, M., Pocock, S. J., Gøtzsche, P. C., & Vandenbroucke, J. P. (2007). The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. *The Lancet*, 370(9596), 1453–1457. https://doi.org/10.1016/S0140-6736(07)61602-X
