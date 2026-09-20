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
  This article treats meta-analysis as a statistical model for evidence synthesis,
  not as a label that automatically upgrades a literature. A worked five-study
  example shows how a narrow common-effect confidence interval can coexist with
  severe heterogeneity, a much wider random-effects interval and a prediction
  interval that spans effects in both directions. The discussion then turns to
  estimand compatibility, risk of bias, publication bias, small-study effects,
  meta-regression, individual-participant-data meta-analysis, sensitivity
  analysis, certainty assessment and the interpretation of forest plots.
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
  deliberately heterogeneous effects. Compute the common-effect estimator,
  Cochran's Q, I-squared, DerSimonian-Laird between-study variance, the
  random-effects estimator and a t-based prediction interval. Use the example to
  separate precision of the pooled mean from heterogeneity of underlying effects
  and from bias in the evidence base.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/meta-analysis-pooling.jpg
  og_image: /assets/images/headers/meta-analysis-pooling.jpg
  overlay_image: /assets/images/headers/meta-analysis-pooling.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/meta-analysis-pooling.jpg
  twitter_image: /assets/images/headers/meta-analysis-pooling.jpg
---

<!--
Development contract
Question: What does a meta-analysis establish beyond the individual studies it combines?
Claim: Meta-analysis can improve precision and organise quantitative evidence, but it cannot repair bias, incompatible estimands, selective publication or scientifically important heterogeneity. A pooled mean is only one feature of the evidence distribution.
Counterclaim: Meta-analysis is one of the most powerful tools in quantitative evidence synthesis when eligibility criteria, risk-of-bias assessment, effect measures and modelling assumptions are appropriate. The problem is not pooling, but treating pooling as automatic validation.
Evidence object: Five synthetic studies with effects 0.10, 0.15, 0.20, 0.80 and 0.90, common standard error 0.10, pooled common-effect estimate 0.43, Q=59.8, I-squared about 93.3%, DerSimonian-Laird tau-squared 0.1395, random-effects interval about [0.09, 0.77], and prediction interval about [-0.88, 1.74].
Failure case: Interpreting I-squared mechanically, treating random effects as a cure for incompatible studies, using funnel-plot asymmetry as proof of publication bias, or assuming a systematic review is weak because it declines to pool.
Reader payoff: Read a meta-analysis as a structured argument about comparability, bias and uncertainty rather than as one pooled number.
Exclusions: Ranking named journals or review groups, claiming one estimator is universally best, and treating meta-analysis as appropriate when the primary studies do not share a meaningful estimand.
-->

A meta-analysis can produce one of the most persuasive objects in scientific communication: a pooled estimate with a narrow confidence interval. Several studies are assembled, each estimate is assigned a weight, and a single number appears at the bottom of a forest plot. The presentation suggests that uncertainty has been reduced and that several imperfect studies have converged on one answer. In well designed evidence synthesis that interpretation can be justified, but the pooled estimate is only the final stage of a much longer argument about what was searched, which studies were eligible, whether they measured sufficiently comparable quantities, how biased they may be, and what statistical model is appropriate for the differences among them.

This distinction matters because meta-analysis is often discussed as if it were a study design that automatically sits above the studies it combines. It is not. A systematic review is the research process used to identify, select, appraise and synthesise a body of literature using explicit methods. A meta-analysis is the statistical combination of quantitative results from separate studies. A systematic review may contain one or several meta-analyses, or it may conclude that statistical pooling would be misleading. The scientific strength of the synthesis therefore depends first on the review process and the compatibility of the underlying evidence, not on the presence of a pooled diamond.

## What the pooled estimate actually estimates

Suppose study $i$ reports an effect estimate $\hat\theta_i$ with standard error $s_i$. Under a simple common-effect model, each study is assumed to estimate the same underlying effect $\theta$, with differences among observed estimates attributed to sampling variation. Inverse-variance weighting assigns

$$
w_i=\frac{1}{s_i^2},
$$

and the pooled estimator is

$$
\hat\theta_{\mathrm{FE}}
=
\frac{
\sum_{i=1}^{k}w_i\hat\theta_i
}{
\sum_{i=1}^{k}w_i
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

The calculation is straightforward, but its interpretation depends on the common-effect assumption. If the studies really are estimating the same underlying quantity, and if their standard errors adequately describe their sampling uncertainty, then pooling can improve precision substantially. The mathematical operation reduces uncertainty because several independent estimates contribute information about the same parameter. The operation does not test whether the parameter should have been common in the first place. That is a scientific and statistical judgement about the studies.

Consider five synthetic studies, all with standard error $0.10$, but with effect estimates equal to $0.10$, $0.15$, $0.20$, $0.80$ and $0.90$. Equal standard errors imply equal inverse-variance weights, so the pooled common-effect estimate is simply the arithmetic mean,

$$
\hat\theta_{\mathrm{FE}}
=
\frac{
0.10+0.15+0.20+0.80+0.90
}{5}
=
0.43.
$$

Each study has weight $100$, so the total weight is $500$ and

$$
SE(\hat\theta_{\mathrm{FE}})
=
\sqrt{\frac{1}{500}}
\approx0.0447.
$$

A conventional 95% confidence interval is therefore approximately

$$
0.43\pm1.96(0.0447),
$$

which gives

$$
[0.34,\ 0.52].
$$

If only the pooled estimate and its confidence interval were reported, the evidence would appear remarkably precise. Yet the individual studies tell a visibly different story. Three estimates lie between $0.10$ and $0.20$, while two lie near $0.85$. The pooled value of $0.43$ sits between two clusters and is not especially representative of either one. Precision around an arithmetic centre is not the same thing as scientific agreement.

## Heterogeneity is part of the result

Cochran's $Q$ statistic measures the weighted dispersion of study estimates around the pooled common-effect estimate,

$$
Q
=
\sum_i
w_i
(\hat\theta_i-\hat\theta_{\mathrm{FE}})^2.
$$

For the five-study example,

$$
Q=59.8.
$$

Under a common-effect model, $Q$ is compared with a chi-squared distribution with $k-1=4$ degrees of freedom. A value close to $60$ is far larger than would be expected from sampling error alone when every study has standard error $0.10$. The numerical disagreement among the studies therefore contains much more structure than the common-effect model allows.

A frequently reported transformation of $Q$ is

$$
I^2
=
\max\left(
0,
\frac{
Q-(k-1)
}{
Q
}
\right)\times100\%.
$$

For this example,

$$
I^2\approx93.3\%.
$$

That value is evidence of substantial inconsistency relative to the within-study precision, but it should not be read mechanically. $I^2$ is not the percentage of studies that are wrong, nor the probability that the effects differ, nor a direct measure of clinical importance. It depends on the size of between-study differences relative to within-study uncertainty. Extremely precise studies can produce a high $I^2$ for differences that are scientifically modest, while imprecise studies can hide meaningful heterogeneity inside wide sampling error.

The most important object is therefore not the heterogeneity statistic in isolation, but the pattern of estimates and the scientific reasons that might generate it. Differences in population, dose, implementation, follow-up, outcome definition, measurement reliability, study quality or true effect modification can all produce heterogeneity. A serious synthesis should ask whether those explanations are plausible before treating heterogeneity as a nuisance term that can simply be absorbed by a different estimator.

A random-effects model makes this distinction explicit by allowing the underlying study effects to vary,

$$
\theta_i
\sim
\mathcal N(\mu,\tau^2),
$$

with observed estimates satisfying

$$
\hat\theta_i
\sim
\mathcal N(\theta_i,s_i^2).
$$

Here $\mu$ is the mean of the distribution of study-specific effects and $\tau^2$ is the between-study variance. A common random-effects weighting rule is

$$
w_i^\ast
=
\frac{1}{
s_i^2+\hat\tau^2
}.
$$

The model now acknowledges two sources of uncertainty, the sampling uncertainty within studies and the variation in true effects across studies. That is scientifically different from the common-effect model. It changes the target from one universal effect to a distribution of effects with a mean.

Using the DerSimonian-Laird estimator for the synthetic example gives

$$
\hat\tau^2\approx0.1395.
$$

Because all five studies have the same within-study standard error, the random-effects mean remains $0.43$. Its uncertainty changes dramatically. The standard error rises to approximately $0.173$, producing a 95% interval of about

$$
[0.09,\ 0.77].
$$

The data have not changed. The question posed by the model has changed. The common-effect analysis asks about one effect assumed to be shared by all studies, while the random-effects analysis asks about the mean of a heterogeneous effect distribution. A much wider interval is therefore not a statistical failure. It is the price of acknowledging variation that the first model treated as impossible.

## The pooled mean is not the effect in a new setting

Even the random-effects confidence interval answers only part of the problem. It describes uncertainty about the mean effect $\mu$ under the specified model. It does not directly answer what effect might occur in another comparable setting. For that purpose, a prediction interval is more informative because it incorporates both uncertainty about the mean and the estimated between-study variance.

A simple prediction interval has the form

$$
\hat\mu
\pm
t
\sqrt{
\hat\tau^2+SE(\hat\mu)^2
}.
$$

Using a $t$ critical value with three degrees of freedom in the five-study example gives approximately

$$
[-0.88,\ 1.74].
$$

This interval is enormously wider than either confidence interval around the pooled mean. The mean effect is positive, but the model still allows an effect in a new comparable setting to be negative or strongly positive. That contrast is exactly why prediction intervals deserve more attention in evidence synthesis. The confidence interval around the mean answers how precisely the average effect distribution is estimated. The prediction interval addresses how variable the underlying effects themselves may be.

The distinction is particularly important when a meta-analysis is used to justify recommendations. A positive pooled mean can coexist with settings in which the effect is negligible or reversed. Conversely, a pooled mean near zero can conceal systematic benefit in one class of populations and harm in another. In either case, the question is no longer whether the average differs from zero, but why effects vary and whether the variables causing that variation can be identified.

Prediction intervals are not infallible. With few studies, $\tau^2$ itself is estimated imprecisely, so the prediction interval can be unstable. The correct response is not to ignore it and return to the pooled mean. It is to report that the evidence contains limited information about between-study variability.

## Pooling does not repair incompatible estimands or biased studies

The arithmetic of meta-analysis can be applied whenever numerical estimates and standard errors exist. Scientific comparability is harder. Suppose the first three studies in the example evaluate a low dose while the last two evaluate a much larger dose. A pooled estimate across all five may still be mathematically well defined, but its meaning depends on the question. If the estimand is the effect of the low dose, the high-dose studies do not estimate that quantity. If the estimand is an average effect across a specified distribution of doses, then pooling may be appropriate, but the weighting scheme and the composition of the evidence set become part of the estimand.

The same problem arises when outcomes are nominally related but scientifically different. A set of studies may all use the label "depression" while one measures diagnosis, another uses a validated symptom scale, another measures days without symptoms and another reports a short unvalidated questionnaire. Converting the results to a standardized mean difference can put them on a common numerical scale,

$$
d
=
\frac{
\bar X_1-\bar X_0
}{
s_{\mathrm{pooled}}
},
$$

but it does not make the constructs identical. Statistical harmonisation cannot remove differences in measurement validity, clinical meaning or timing.

This is why random effects should not be used as permission to pool anything. A random-effects model allows effect sizes to vary around a distribution. It does not establish that the studies belong to one scientifically meaningful distribution. If studies estimate fundamentally different interventions or outcomes, adding $\tau^2$ does not create comparability. Heterogeneity models describe variation after the synthesis question has been justified. They do not justify the synthesis question.

Bias creates a related problem because pooling generally reduces random error, not systematic error. If each study estimate can be written as

$$
\hat\theta_i
=
\theta+b+\varepsilon_i,
$$

where $b$ is a shared bias and $\varepsilon_i$ is sampling error, then averaging across studies reduces the contribution of $\varepsilon_i$ but leaves $b$ untouched. As the number of studies grows,

$$
\bar\varepsilon\rightarrow0,
$$

while the pooled estimate approaches

$$
\theta+b.
$$

The result can become increasingly precise around the wrong value. This is one reason risk-of-bias assessment is part of evidence synthesis rather than an optional appendix to it.

Inverse-variance weighting should also not be confused with weighting by scientific validity. A large study with a tiny standard error can dominate a meta-analysis even if its measurement or causal design is weak. A smaller rigorous study can receive less statistical weight because it is less precise. Statistical precision and methodological credibility are different dimensions. A review can address this through eligibility restrictions, stratified analyses, sensitivity analyses or other modelling choices, but there is no universal formula that converts "study quality" into a correct inverse-variance weight.

## The review process determines which evidence can be pooled

The strongest part of a systematic review is often not the final pooled estimator, but the discipline imposed on source selection. A review protocol defines the question, eligibility criteria, search strategy, outcomes and planned synthesis before the results are fully known. That procedure constrains selective citation and makes omissions easier to detect. PRISMA 2020 is valuable for this reason. It makes the flow from search to inclusion auditable, but it does not guarantee that the search was scientifically appropriate or that the included studies are unbiased.

Search strategy itself can change the evidence base. A review that searches one bibliographic database, restricts to English-language journal articles and ignores trial registries may retrieve a different set of studies from a review that searches several databases, reference lists, conference proceedings, preprints and regulatory records. The pooled effect is conditional on that evidence set. A precise meta-analysis can therefore be sensitive to decisions made before any effect estimate is combined.

Publication bias is one important example. If the probability that a study becomes visible depends on its result,

$$
P(\text{observed}\mid p<0.05)
>
P(\text{observed}\mid p\ge0.05),
$$

then the published literature is a selected sample of completed research. A meta-analysis can combine the visible studies perfectly and still estimate a distorted effect. Trial registration, prospective protocols and regulatory records help reveal missing studies and unreported outcomes, but none of those mechanisms eliminates selective reporting automatically.

Funnel plots and formal asymmetry tests are useful diagnostics in this context, but they should not be treated as detectors with one interpretation. Smaller studies may show larger effects because of publication bias, but also because they use stronger interventions, more selected populations, different outcome definitions or weaker methods. The observed pattern is better described as a small-study effect until the mechanism has been investigated. Publication bias is one possible cause rather than the definition of the pattern.

Selective outcome reporting can enter the literature even when every trial itself is published. A study may measure several outcomes but report only those that are favourable. A later systematic review cannot reconstruct the missing results from the paper alone. Comparing publications with protocols, registrations and statistical analysis plans therefore improves evidence synthesis by revealing which outcomes were planned and which appeared only after the data were known.

## Sensitivity analyses are part of the substantive result

A meta-analysis contains many analytic decisions, including eligibility rules, effect measure, fixed or random effects, estimator of $\tau^2$, treatment of zero-event studies, handling of multiple outcomes, selection of follow-up time, exclusion of studies at high risk of bias and methods for dealing with dependent estimates. These choices are not merely technical settings. When plausible alternatives produce materially different conclusions, that instability is part of the evidence.

Leave-one-out analysis provides a simple example. If $\hat\theta$ is the pooled estimate, one can recompute the synthesis after omitting study $j$,

$$
\hat\theta_{(-j)}.
$$

A large change after removing one study does not mean that the study should automatically be discarded. It shows that the conclusion depends heavily on that observation and should therefore be interpreted with that dependence visible.

Sensitivity analysis extends the same logic to model choices. If the pooled effect remains similar after excluding high-risk-of-bias studies, changing the between-study variance estimator and using alternative outcome definitions, confidence in the broad conclusion increases. If the sign or practical magnitude changes under reasonable choices, the review should report that fragility rather than selecting one preferred specification.

Meta-regression can be useful when heterogeneity has plausible study-level explanations. A model such as

$$
\theta_i
=
\beta_0+\beta_1Z_i+u_i
$$

can examine whether effect size varies with dose, average age, intervention intensity, follow-up duration or another study characteristic. Interpretation requires caution because the unit of analysis is the study. If studies with older average participants show larger effects, that does not imply that older individuals benefit more. Study-level relationships are vulnerable to ecological bias, and meta-regression is often unstable when only a small number of studies are available.

Individual participant data meta-analysis can address some of these limitations by obtaining participant-level data from the original studies. This can allow common outcome definitions, consistent covariate adjustment, participant-level interaction analyses and more flexible handling of missing data. It is a richer form of synthesis, but it remains conditional on which trials provide data and on the quality of the original studies. Additional data access expands what can be estimated. It does not remove design bias from the source studies.

## How to read a meta-analysis

A forest plot should be read from the individual studies toward the pooled estimate rather than from the pooled diamond backward. The first questions are whether populations, interventions, comparators, outcomes and follow-up times are sufficiently similar to justify synthesis. The next questions concern risk of bias and the precision of each study. Only after that does the pooled estimate acquire a stable interpretation.

A useful reading sequence is to inspect the direction and magnitude of each study estimate, compare their uncertainty intervals, look for clusters or outliers, examine whether differences align with known design features, and then consider the pooled model. If a random-effects model is used, $\tau^2$ and a prediction interval are often more informative about generalisability than $I^2$ alone. If the review contains major clinical or methodological heterogeneity, the pooled mean should be interpreted as a summary of a heterogeneous distribution rather than a universal treatment effect.

Effect scale matters as well. A relative risk can be stable across studies while absolute benefit differs greatly because baseline risk differs. Suppose several studies estimate

$$
RR=0.8.
$$

If control risk is $0.05$, treated risk is approximately $0.04$, giving an absolute reduction of $0.01$. If control risk is $0.50$, treated risk is approximately $0.40$, giving an absolute reduction of $0.10$. A meta-analysis of relative effects can therefore be statistically coherent while the practical consequences differ by a factor of ten.

Time should also remain part of the estimand. Pooling effects measured at six weeks, six months and five years can erase important treatment dynamics. An intervention may produce short-term benefit followed by attenuation or long-term harm. A synthesis should therefore specify the time horizon rather than treating every follow-up as another estimate of the same quantity.

The unit of evidence also requires care. Several publications can analyse the same cohort or trial. If they are treated as independent studies, participants are counted more than once and the apparent evidence base becomes larger than it really is. Systematic reviews therefore need to identify overlapping populations, trial registrations and duplicate analyses before deciding which estimates enter a meta-analysis.

## Precision, consistency and certainty are different quantities

A pooled confidence interval describes uncertainty in a statistical parameter under a specified model. It does not by itself describe the overall certainty of the evidence. Frameworks such as GRADE consider additional dimensions including risk of bias, inconsistency, indirectness, imprecision and publication bias. A narrow pooled interval can coexist with low certainty if the studies are biased or if the evidence is indirect relative to the target question.

This distinction becomes especially important when evidence is used in public communication. A statement such as "a meta-analysis of twenty studies found" can sound definitive even when the twenty studies are small, heterogeneous, indirect or at high risk of bias. The number of studies and the narrowness of the pooled interval are not sufficient summaries. The relevant scientific question is whether the evidence base collectively identifies the claim being communicated.

Systematic review protocols help by making review-level decisions auditable. Search sources, eligibility criteria, outcome definitions and synthesis methods can all be influenced by knowledge of the emerging literature. Prospective registration through platforms such as PROSPERO creates a record of what the review intended to do before the final results were known. Registration is not a guarantee of quality, but it makes unreported changes easier to detect.

The time at which a review was conducted also matters. An early meta-analysis of four small studies can legitimately produce a different conclusion from an updated synthesis that includes several later, larger studies. The first review was a statement about the evidence available at that time. Scientific synthesis is therefore conditional on the information set,

$$
P(\text{conclusion}\mid\text{evidence available at time }t).
$$

Living systematic reviews make this dependence explicit by updating the synthesis as new evidence appears. Their advantage is not that they remove uncertainty, but that they reduce the lag between the evolving literature and the formal summary.

## The five-study example in context

The synthetic example is useful because every calculation is transparent. Five equally precise studies reported effects of

$$
0.10,\quad0.15,\quad0.20,\quad0.80,\quad0.90.
$$

A common-effect model produced

$$
\hat\theta_{\mathrm{FE}}=0.43
$$

with a narrow 95% interval of approximately

$$
[0.34,\ 0.52].
$$

The same data produced

$$
Q=59.8
$$

and

$$
I^2\approx93.3\%.
$$

Allowing a distribution of underlying effects yielded

$$
\hat\tau^2\approx0.1395,
$$

with a random-effects confidence interval around the mean of approximately

$$
[0.09,\ 0.77].
$$

The prediction interval widened further to approximately

$$
[-0.88,\ 1.74].
$$

These summaries do not contradict one another. They answer different questions. The narrow common-effect interval describes the arithmetic centre under a model that assumes one shared effect. The random-effects interval describes uncertainty about the mean of a heterogeneous effect distribution. The prediction interval describes the range in which a true effect from another comparable setting might plausibly fall under that model.

The scientific conclusion should therefore not be reduced to the sentence "the pooled effect is 0.43". The evidence is better described as strongly heterogeneous, with a positive average under the random-effects model but substantial uncertainty about how effects vary across settings. If the heterogeneity reflects identifiable differences in population, dose or intervention implementation, those differences may be more important than the pooled mean itself.

A good meta-analysis is valuable precisely because it can make this structure visible. Its purpose is not to turn disagreement into one authoritative number. It is to organise evidence in a way that clarifies where studies agree, where they differ, how much uncertainty remains and which assumptions are required to combine them.

Meta-analysis is therefore not a magic upgrade. It is a statistical model embedded inside a systematic argument about evidence. Its strength depends on the quality of the studies, the transparency of the review, the coherence of the estimand and the adequacy of the model used to synthesise the results.

## References

Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). *Introduction to Meta-Analysis*. Wiley.

Deeks, J. J., Higgins, J. P. T., Altman, D. G., McKenzie, J. E., & Veroniki, A. A. (2024). Analysing data and undertaking meta-analyses. In *Cochrane Handbook for Systematic Reviews of Interventions*, version 6.5.

DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188. https://doi.org/10.1016/0197-2456(86)90046-2

Egger, M., Davey Smith, G., Schneider, M., & Minder, C. (1997). Bias in meta-analysis detected by a simple, graphical test. *BMJ*, 315, 629–634. https://doi.org/10.1136/bmj.315.7109.629

Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539–1558. https://doi.org/10.1002/sim.1186

IntHout, J., Ioannidis, J. P. A., Rovers, M. M., & Goeman, J. J. (2016). Plea for routinely presenting prediction intervals in meta-analysis. *BMJ Open*, 6, e010247. https://doi.org/10.1136/bmjopen-2015-010247

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71. https://doi.org/10.1136/bmj.n71

Sterne, J. A. C., Sutton, A. J., Ioannidis, J. P. A., Terrin, N., Jones, D. R., Lau, J., et al. (2011). Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials. *BMJ*, 343, d4002. https://doi.org/10.1136/bmj.d4002
