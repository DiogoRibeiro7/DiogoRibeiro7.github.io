---
permalink: '/science-communication/preregistration_protocols_confirmation_and_discovery/'
title: 'Preregistration, Protocols, and the Separation of Confirmation from Discovery'
date: '2026-06-25'
categories:
- Science Communication
tags:
- Scientific Method
- Preregistration
- Open Science
- Research Methods
- Reproducibility
author_profile: false
classes: wide
seo_title: 'Preregistration, Protocols, and the Separation of Confirmation from Discovery'
seo_description: 'Preregistration improves scientific auditability by separating planned analyses from data-dependent decisions, but it does not rescue weak design, invalid measurement or poor causal identification.'
seo_type: article
excerpt: >-
  Preregistration is useful because it records what researchers intended to do
  before the results were known. Its value lies in auditability, not in turning
  a weak design into a strong one or converting exploratory work into truth.
summary: >-
  This article examines preregistration as a control on researcher degrees of
  freedom. It distinguishes protocols, registrations, statistical analysis plans
  and Registered Reports, develops a quantitative example with 48 possible
  analysis paths, discusses HARKing and outcome switching, and explains why
  transparent deviations are scientifically preferable to rigid adherence to a
  bad plan. Recent evidence on outcome switching in registered cohort studies is
  used to show that registration without sufficiently specific prespecification
  can still leave substantial ambiguity.
keywords:
- preregistration
- research protocol
- registered reports
- outcome switching
- HARKing
- scientific method
why_this_exists: >-
  Discussions of open science often reduce preregistration to a badge of
  credibility. The scientifically important issue is narrower: whether the
  record is specific enough to distinguish decisions made before seeing the
  results from decisions made after seeing them.
evidence: >-
  Methodological literature on researcher degrees of freedom, HARKing,
  preregistration and Registered Reports, together with recent meta-research on
  outcome switching in prospectively registered cohort studies of interventions.
methodology: >-
  Formalise analytic flexibility as a finite set of plausible analysis paths,
  compute the probability of at least one nominally significant result under an
  independence toy model, and use that calculation to motivate preregistration
  as an audit mechanism rather than a guarantee of correctness.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/preregistration-protocol.jpg
  og_image: /assets/images/headers/preregistration-protocol.jpg
  overlay_image: /assets/images/headers/preregistration-protocol.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/preregistration-protocol.jpg
  twitter_image: /assets/images/headers/preregistration-protocol.jpg
---

<!--
Development contract
Question: What problem does preregistration solve, and what does it leave untouched?
Claim: Preregistration improves scientific auditability by separating decisions made before observing the results from those made afterwards. It reduces ambiguity around confirmatory claims but does not guarantee good design, valid measurement, adequate power, correct causal identification or honest interpretation.
Counterclaim: Scientific work often requires adaptation because data quality, recruitment, measurement and model assumptions cannot always be anticipated. Rigid adherence to an unsuitable plan can be worse than a transparent deviation.
Evidence object: A toy analysis space with 48 plausible analysis paths, empirical evidence on outcome switching, the distinction among protocols, registrations, statistical analysis plans and Registered Reports, and examples of legitimate and illegitimate deviations.
Failure case: Treating preregistration as a quality certificate, preregistering vague intentions that leave all consequential choices open, or hiding sensible deviations because the protocol is treated as immutable.
Reader payoff: Understand what must be specified in advance, how to report deviations, how exploratory analysis can coexist with confirmatory analysis, and why Registered Reports address incentives that ordinary preregistration does not.
Exclusions: Claiming that all exploratory research should be preregistered, treating unregistered research as automatically invalid, or reducing research quality to compliance with an open-science checklist.
-->

Preregistration is sometimes described as though it were a certificate attached to a study before the analysis begins. The study is preregistered, therefore it is rigorous. The study is not preregistered, therefore it is suspect. That interpretation is too simple and obscures the problem preregistration was designed to address. Its main scientific value is not that it guarantees a good hypothesis, a representative sample, a valid measurement model, adequate statistical power or correct causal identification. Its value is that it creates a time-stamped record of what researchers intended to do before the results were known, allowing readers to distinguish planned decisions from decisions that were influenced by the observed data.

That distinction matters because a research dataset rarely determines one unique analysis. Researchers choose outcomes, transformations, exclusion rules, covariates, model families, interaction terms, subgroup definitions, missing-data procedures and stopping rules. Many of those choices can be scientifically reasonable. The problem is not the existence of alternatives. The problem arises when a large space of plausible analyses is explored, one favourable result is selected, and the final paper is written as though that analysis had been specified from the beginning. In that case the nominal uncertainty reported by the selected model no longer represents the full process that produced the claim.

Preregistration is therefore best understood as an audit mechanism. It records part of the decision process before the result is observed. A good registration does not prevent researchers from learning from the data, changing their minds or conducting exploratory analyses. It allows readers to see which claims were tested under a prespecified plan and which emerged after inspection of the data. That separation is central to the difference between confirmation and discovery.

## Researcher degrees of freedom are a design problem

Suppose a study contains four plausible primary outcomes. For each outcome the researchers could reasonably choose one of three covariate specifications, two transformations and two exclusion rules. The analysis space then contains

$$
4\times3\times2\times2=48
$$

plausible paths.

This does not imply that researchers actually run all forty-eight analyses, nor that the analyses are independent. It simply illustrates how quickly flexibility accumulates when several defensible choices are combined. If, as a deliberately simplified null model, we imagine forty-eight independent tests at level $\alpha=0.05$, the probability that at least one result is nominally significant is

$$
1-(1-0.05)^{48}
=
1-0.95^{48}
\approx0.915.
$$

Under the same toy assumptions, the probability that at least one p-value falls below $0.01$ is

$$
1-0.99^{48}
\approx0.383.
$$

The expected minimum of forty-eight independent p-values uniformly distributed on $[0,1]$ is

$$
\mathbb E[p_{\min}]
=
\frac{1}{49}
\approx0.0204.
$$

Real analysis paths are usually correlated, so these calculations should not be interpreted as direct estimates of false-positive rates in an actual study. Their purpose is to make the structural problem visible. Once many data-dependent routes are available, the final p-value is not generated by one analysis chosen independently of the result. It is generated by an analysis-selection procedure, and the properties of that procedure matter.

Simmons, Nelson and Simonsohn described this problem as researcher degrees of freedom, showing how apparently reasonable undisclosed flexibility can substantially increase false-positive findings. Kerr had earlier described HARKing, hypothesising after the results are known, as a related problem in the narrative stage of research. A hypothesis discovered in the data can be scientifically interesting. The distortion occurs when it is reported as though it had been predicted before the data were observed.

The same distinction applies beyond p-values. A researcher can examine several model specifications and report the one with the largest effect size, the narrowest interval, the most favourable subgroup or the cleanest visual pattern. Bayesian analyses can also contain data-dependent flexibility through prior changes, model selection and stopping rules. Machine-learning studies can overfit benchmark choices, preprocessing, random seeds or evaluation datasets. Preregistration addresses the provenance of decisions rather than one particular statistical philosophy.

## A protocol, a registration and an analysis plan are not the same thing

The language around preregistration often collapses several different documents into one. A protocol describes the scientific design of the study before or during its conduct. Depending on the field, it can include the research question, target population, recruitment procedure, intervention, comparator, measurement schedule, outcomes, sample-size rationale and general analytical strategy. A registry entry may contain a more compact public record of the study. A statistical analysis plan specifies the analysis in greater detail, including estimands, model equations, covariates, missing-data procedures, transformations, multiplicity adjustments, subgroup definitions and sensitivity analyses.

The distinction is useful because a registration can exist without being specific enough to constrain important analytical choices. Writing that the primary outcome is "depression", for example, leaves open the instrument, time point, aggregation rule and analysis metric. A reader cannot determine whether the outcome reported later is the one originally intended. Prespecification only creates an audit trail when the planned quantity is defined precisely enough to be compared with what was ultimately analysed.

Recent empirical work makes that limitation concrete. Song and colleagues examined 124 prospectively registered cohort studies of interventions and compared the registration records with the eventual publications. Only 30 studies, 24%, completely prespecified their primary outcomes according to four components: measurement variable, analysis metric, aggregation method and time point. Outcome switching occurred in 60 studies, 48%, while only two publications explained the change. Among assessable cases in which outcomes were introduced, upgraded or downgraded, most of the changes favoured statistically significant results. The study does not establish that every discrepancy was opportunistic. Some changes may have been scientifically justified. It does show why registration alone is insufficient when the record is too vague to distinguish planned from post hoc decisions.

The lesson is not that every study must predict every implementation detail before recruitment begins. Some decisions cannot be known in advance. The lesson is that consequential choices should either be specified before the results are observed or disclosed later as deviations. A reader should be able to reconstruct which parts of the analysis were planned and which parts were responses to information encountered during the study.

## Confirmation and exploration should coexist rather than compete

A common objection to preregistration is that science is exploratory. That objection is correct about science and incorrect about what careful preregistration requires. Discovery often depends on looking at unexpected patterns, testing alternative specifications, noticing subgroups and changing models when assumptions fail. A system that prohibited those activities would damage research.

The relevant distinction is not between allowed and forbidden analyses. It is between analyses whose inferential status was established before the data were examined and analyses generated by the data themselves. Suppose a preregistered primary analysis finds no clear effect, but exploratory analysis reveals a strong interaction with age. The scientifically responsible report is not to suppress the interaction because it was not preregistered. It is to describe it as exploratory, quantify the uncertainty, explain why it is interesting and test it prospectively in new data if the claim is important.

Exploration becomes problematic when the chronology is erased. If the age interaction is written into the introduction as a strong prior hypothesis and presented with the same evidential status as the prespecified primary test, the publication gives the reader a false account of how the claim arose. HARKing is therefore not objectionable because post hoc hypotheses are worthless. Many important hypotheses originate in unexpected data. The objection is that a hypothesis generated by one dataset should not be presented as though that same dataset provided an independent confirmatory test.

This can be expressed with a simple separation. Let $D_1$ be the data used to generate a hypothesis $H$. If

$$
H=g(D_1),
$$

then testing $H$ on the same $D_1$ does not reproduce the evidential situation in which $H$ was fixed independently of the data. A cleaner confirmatory design evaluates the hypothesis on new information,

$$
D_2\perp D_1
$$

under the relevant sampling structure, so that the claim generated from $D_1$ is exposed to an independent opportunity to fail.

Preregistration approximates this separation by fixing parts of the hypothesis and analysis before the outcome data are known. It does not create independence where none exists, but it records chronology. That chronology is essential for interpreting whether an analysis was confirmatory, exploratory or somewhere between the two.

## Deviations are not failures if they are visible

No serious methodology should require researchers to execute an obviously inappropriate analysis merely because it was written down months earlier. Recruitment may fail. A measurement instrument may malfunction. A planned model may not converge. A distribution may be far more skewed than expected. An external event may change the target population. Data may reveal coding errors or structural missingness that could not reasonably have been anticipated.

A rigid interpretation of preregistration would force researchers to choose between methodological sense and procedural compliance. That is not the objective. The better principle is that deviations should be explicit, justified and separated from the original plan. A paper can report the preregistered analysis, explain why it became unsuitable, present the revised analysis and show whether the substantive conclusion changes.

The importance of this distinction can be seen in missing-data analysis. Suppose a protocol prespecifies complete-case analysis because little missingness was expected. During the study, attrition reaches 25% and is strongly related to baseline severity. Following the original plan without comment may now be less defensible than using a principled missing-data model. The scientifically relevant questions are why the plan changed, whether the change was influenced by the direction of the treatment effect, and how sensitive the conclusion is to alternative assumptions.

The same logic applies to transformations and model assumptions. If a planned Gaussian model is clearly incompatible with the observed outcome distribution, the researcher should not preserve it merely to claim perfect preregistration compliance. The deviation should be documented and the revised model justified. A preregistration that forces bad analysis has been misunderstood. Its purpose is to expose decision chronology, not to prevent methodological correction.

Versioning matters for this reason. If a protocol changes before outcome data are available, the new version can be time-stamped and the reason recorded. If the change occurs after investigators have seen relevant results, that fact should also be visible. The scientific importance lies in when the decision became data-dependent.

## Registered Reports address incentives that preregistration alone does not

Ordinary preregistration documents a plan, but the study may still be submitted to a journal only after the results are known. Publication incentives therefore remain. Researchers may conduct a well-preregistered study, obtain a null result and decide not to write it up. Journals may still prefer surprising positive findings. The registered plan improves the audit trail, but it does not by itself guarantee that the study enters the published literature.

Registered Reports alter this sequence by moving part of peer review before results are known. The research question, rationale and methods are reviewed first. If the design is judged sufficiently rigorous, the journal can offer in-principle acceptance before the outcome is available. Publication then depends primarily on following the approved protocol and interpreting the results appropriately rather than on obtaining a statistically significant or novel finding.

This changes incentives in a way ordinary preregistration does not. The design can be improved before data collection, and the probability of publication becomes less dependent on the eventual direction of the result. Registered Reports therefore address both analytic flexibility and publication selection, although they cannot eliminate every source of bias. A poorly chosen research question can still be studied rigorously. Measurement can still fail. Recruitment can still differ from expectations. Reviewers can still approve a design containing mistaken assumptions.

The important methodological point is that open-science practices solve different problems. Preregistration records chronology. Protocols define design. Statistical analysis plans specify analytical decisions. Registered Reports reduce result-dependent publication incentives. Data and code sharing improve reproducibility and auditability. None of these should be collapsed into one score of research quality.

## What good preregistration needs to specify

The level of detail required depends on the scientific question, but a confirmatory registration should be precise about the quantity to be estimated. The target population, exposure or intervention, comparator, outcome, measurement time, analysis metric and primary model should be identifiable. Inclusion and exclusion rules should be stated in a form that can be applied without knowing which observations favour the hypothesis. If covariates are part of the primary analysis, they should be named. If subgroup analyses are confirmatory, the subgroup definitions and interaction tests should be specified. The treatment of missing data and multiplicity should be described at a level that determines what the main inferential procedure will be.

This does not require writing a script that can run without modification months later, although executable analysis plans can be valuable in some settings. It requires removing enough flexibility that the eventual confirmatory result can be compared with an actual prior plan rather than with a vague statement of intent.

Contingency rules can also be preregistered. A study might specify that if a variable exceeds a stated missingness threshold, multiple imputation will replace the primary complete-case analysis, or that if proportional hazards assumptions fail, a restricted mean survival time analysis will be reported as a planned alternative. These rules are useful because they anticipate foreseeable problems without pretending that all contingencies can be known.

There is also a point at which excessive detail becomes performative. Hundreds of pages of preregistration can make the document difficult to audit and may hide consequential decisions among irrelevant implementation details. The goal is not maximal volume. It is specificity around the choices that could materially change the scientific conclusion.

## Preregistration should make interpretation more modest, not more automatic

A preregistered result can still be wrong. Randomisation can be implemented badly. A sample can be unrepresentative. A study can be underpowered. A questionnaire can fail to measure the intended construct. An instrumental variable can violate exclusion restrictions. A model can be misspecified. Data can contain systematic measurement error. A statistically significant prespecified effect can be clinically negligible.

Conversely, an unregistered result can be important and correct. Historical datasets, natural experiments and exploratory scientific work often involve questions that were not or could not have been registered before the relevant data existed. The absence of preregistration changes the information available about analytical chronology. It does not logically determine whether the claim is true.

This is why preregistration should not become another authority signal used in place of methodological reading. The appropriate conclusion is narrower. A sufficiently detailed, prospectively time-stamped plan reduces uncertainty about which decisions were made before the results were known. That information helps readers assess selective analysis, outcome switching and the distinction between confirmation and exploration. It strengthens the provenance of the analysis.

The strongest practice is therefore not rigid adherence to a document, but a transparent chain from protocol to registration to analysis to publication. The reader should be able to see what was planned, what changed, when it changed and why. Exploratory work should remain visible as exploratory work, and unexpected results should be treated as sources of new hypotheses rather than retroactively rewritten predictions.

The scientific method depends on revision, but revision becomes more informative when its chronology is preserved. Preregistration contributes to that record. It does not certify truth. It makes parts of the path to a claim auditable.

## References

Chambers, C. D., & Tzavella, L. (2022). The past, present and future of Registered Reports. *Nature Human Behaviour*, 6, 29–42. https://doi.org/10.1038/s41562-021-01193-7

Kerr, N. L. (1998). HARKing: Hypothesizing after the results are known. *Personality and Social Psychology Review*, 2(3), 196–217. https://doi.org/10.1207/s15327957pspr0203_4

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114

Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science*, 22(11), 1359–1366. https://doi.org/10.1177/0956797611417632

Song, Z., Jespersen, C., Hróbjartsson, A., Kim, S. J., Fowler, R., Austin, P. C., & Chan, A.-W. (2026). Outcome switching in cohort studies of interventions: meta-epidemiological study. *BMJ*, 393, e087975. https://doi.org/10.1136/bmj-2025-087975

van 't Veer, A. E., & Giner-Sorolla, R. (2016). Pre-registration in social psychology: A discussion and suggested template. *Journal of Experimental Social Psychology*, 67, 2–12. https://doi.org/10.1016/j.jesp.2016.03.004
