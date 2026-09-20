---
permalink: '/science-communication/detection_is_not_evidence_of_danger/'
title: 'Detection Is Not Evidence of Danger'
date: '2025-08-14'
categories:
- Science Communication
tags:
- Analytical Chemistry
- Toxicology
- Exposure Assessment
- Risk Assessment
- Scientific Reasoning
author_profile: false
classes: wide
seo_title: 'Detection, Dose, Hazard, and Risk Are Different Scientific Claims'
seo_description: 'Finding a substance establishes evidence of presence under a measurement method. It does not by itself establish a harmful dose, while non-detection does not establish absence.'
seo_type: article
excerpt: >-
  Analytical detection answers whether a method can distinguish a signal from
  its background. Toxicological risk requires additional information about
  concentration, exposure, dose, response, route, duration, and uncertainty.
summary: >-
  This article separates analytical detection from toxicological inference.
  A measurement model shows why detection depends on the sensitivity of the
  analytical method. A synthetic exposure calculation then converts a measured
  concentration into an estimated dose and demonstrates why presence alone
  cannot determine risk. The discussion also treats hazard versus risk,
  non-detects, dose-response assumptions, repeated exposure, susceptible
  populations, mixtures, and the limits of simple exposure ratios.
keywords:
- limit of detection
- dose response
- hazard versus risk
- exposure assessment
- analytical chemistry
- toxicology
why_this_exists: >-
  Modern instruments can detect substances at concentrations that would have
  been analytically invisible to earlier methods. Public discussion often turns
  this improvement in measurement into a biological conclusion. This article
  identifies the quantities that must be added before detection can support a
  claim about danger.
evidence: >-
  An original analytical detection model, an original synthetic exposure and
  dose calculation, foundational work on detection limits, the National
  Research Council risk assessment framework, current EPA exposure assessment
  guidance, and the IARC distinction between hazard identification and risk
  assessment.
methodology: >-
  Separate the true concentration from measurement error and an operational
  detection criterion. Derive detection probability near the analytical limit.
  Convert a synthetic environmental concentration into an external dose using
  intake and body mass, then show how different dose-response models map the
  same dose to different risk conclusions.
reviewed_at: '2025-08-14'
header:
  image: /assets/images/headers/photo-microscope.jpg
  og_image: /assets/images/headers/photo-microscope.jpg
  overlay_image: /assets/images/headers/photo-microscope.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-microscope.jpg
  twitter_image: /assets/images/headers/photo-microscope.jpg
---

<!--
Development contract
Question: What can be inferred when an analytical method detects a substance in food, water, blood, air, or another sample?
Claim: Detection supports a statement about presence under a specified measurement procedure. A claim about danger additionally requires information about exposure, dose, dose-response, route, duration, population, and uncertainty.
Counterclaim: Very low concentrations are not automatically harmless. Some agents can be biologically important at low doses, cumulative exposure can matter, susceptible populations can differ, and for some hazards regulators use models without a presumed safe threshold.
Evidence object: Gaussian measurement model around a detection criterion, a thousand-fold change in analytical sensitivity applied to an unchanged sample, a synthetic concentration-to-dose calculation, and alternative dose-response functions.
Failure case: Treating any detected amount as harmless because it is small, treating a reference value as a biological boundary between safe and unsafe, or interpreting a non-detect as proof that the substance is absent.
Reader payoff: Distinguish presence from exposure and risk, interpret non-detects correctly, translate concentration into an exposure quantity, and identify the assumptions required before a toxicological conclusion follows.
Exclusions: Assessing the safety of a named food, product, chemical, medication, or personal exposure, and using the synthetic numerical example as a real toxicological benchmark.
-->

Modern analytical chemistry can identify extraordinarily small quantities of material. That is a scientific achievement. It is also a source of recurring confusion because an instrument's ability to detect a substance and a substance's ability to cause harm are questions from different parts of science.

A laboratory may report that a compound is present in drinking water, food, blood, soil, or air. The result can be completely correct. The compound may have been identified with a validated analytical procedure, appropriate controls, and a signal that clearly exceeds the method's detection criterion. None of that yet specifies whether the measured amount produces a meaningful biological effect.

The missing step is not semantic. Detection concerns the relationship between a signal and an analytical background. Toxicological risk concerns the relationship between an exposure and an adverse outcome.

Those relationships can be connected, but they are not identical.

The same distinction works in the opposite direction. A report of "not detected" does not ordinarily mean that the true concentration is exactly zero. It means that the measurement did not satisfy the method's criterion for reliable detection. A more sensitive method can therefore turn a non-detect into a detection without anything in the sample having changed.

Scientific interpretation begins by keeping these questions separate.

## Detection is a property of a measurement procedure

Let (C) represent the true concentration of an analyte in a sample. Suppose an analytical procedure returns

[
X=C+arepsilon,
]

where (arepsilon) represents measurement variation. For illustration, assume

[
arepsilonsimmathcal N(0,sigma^2).
]

Now suppose the laboratory uses a signal criterion (L) and records a detection when

[
X>L.
]

The probability of detection is then

[
Pr(	ext{detect}mid C)
=
Pr(C+arepsilon>L)
=
1-Phileft(rac{L-C}{sigma}ight),
]

where (Phi) is the standard normal cumulative distribution function.

The important feature of this expression is not the normality assumption. Real analytical methods use more detailed validation procedures, and definitions of detection limits are operational rather than universal. The important point is that detection depends jointly on the concentration and the performance of the measurement system.

When (C) is far above the relevant analytical limit, detection is highly probable. When (C) is near that limit, repeated measurements can sometimes satisfy the criterion and sometimes fail it. When (C) is below the limit, the substance may still be physically present even though the method cannot reliably distinguish its signal from the blank or background.

This is the logic behind the limit of detection. NIST describes it in terms of the lowest concentration or amount that can be reliably distinguished according to the criteria of the analytical method. Currie's foundational treatment of detection limits and later work distinguishing the limit of blank, limit of detection, and limit of quantitation make the same broader point: the boundary is defined by an analytical decision problem.

It is not a toxicological threshold.

## Better instruments can create more detections without creating more exposure

Consider a sample with a true concentration

[
C=0.10 mu	ext{g/L}.
]

Suppose an older analytical method has a detection limit of approximately

[
L_{	ext{old}}=1.0 mu	ext{g/L}.
]

The sample may be reported as not detected.

A newer method has a detection limit of

[
L_{	ext{new}}=0.001 mu	ext{g/L}.
]

The same concentration is now well within the method's analytical capability and is reported as detected.

The true concentration did not increase. Exposure did not increase. Toxicity did not change. Only the measurement system improved.

The analytical sensitivity improved by a factor of one thousand:

[
rac{L_{	ext{old}}}{L_{	ext{new}}}
=
1000.
]

A comparison of historical surveillance records can therefore show an apparent increase in the frequency of detection simply because laboratories became capable of seeing concentrations that had previously been censored below the analytical limit. Before interpreting such a pattern as environmental deterioration or increased contamination, the measurement methods need to be comparable.

The reverse mistake is equally serious. A non-detect under the older method would not justify replacing (C) with zero. The information is censored by the measurement process. Exactly how a non-detect should be handled statistically depends on the design, detection procedure, distributional assumptions, and purpose of the analysis, but the logical point is simpler: below the reporting boundary does not mean nonexistent.

## Concentration is still not dose

Once a substance has been quantified, another inferential step remains. Concentration in a sample is not generally the same quantity as dose received by a person.

For ingestion through a single medium, a deliberately simplified external dose calculation can be written as

[
D
=
rac{CI}{W},
]

where (C) is concentration, (I) is intake of the medium per unit time, and (W) is body mass. More elaborate exposure assessments include frequency, duration, absorption, multiple sources, route-specific factors, population distributions, and other variables. The simplified expression is useful because it shows what a concentration alone cannot tell us.

Consider a fictional compound (Q) measured in water at

[
C=5 	ext{ng/L}.
]

Assume, purely for the calculation, that a person consumes

[
I=2 	ext{L/day}
]

from that source and has body mass

[
W=70 	ext{kg}.
]

The estimated external dose is

[
D
=
rac{(5 	ext{ng/L})(2 	ext{L/day})}
{70 	ext{kg}}
]

or

[
D
approx
0.143 	ext{ng/(kg day)}.
]

Expressed in micrograms,

[
D
approx
1.43	imes10^{-4}
 mu	ext{g/(kg day)}.
]

The analytical statement "compound (Q) was detected at 5 ng/L" and the exposure statement "under these assumptions, the external ingestion dose is approximately (1.43	imes10^{-4} mu	ext{g/(kg day)})" are therefore different claims. The second requires information that is absent from the first.

Even the calculated external dose is not the final biological quantity. Absorption may be incomplete. Metabolism can change the active form. Distribution can concentrate an agent in particular tissues. Elimination can be rapid or slow. Repeated exposures can accumulate when uptake exceeds clearance.

Exposure assessment exists because the path from concentration in a sampled medium to dose at a biological target is not automatic.

## A detected dose still needs a dose-response model

Suppose, continuing the fictional example, that an experiment provides a reference point

[
D_ast
=
10 mu	ext{g/(kg day)}.
]

This number is invented for the article. It is not a safety standard for any real compound.

The ratio between the estimated exposure and the reference point is

[
R_D
=
rac{D}{D_ast}
=
rac{1.43	imes10^{-4}}{10}
approx
1.43	imes10^{-5}.
]

The reference point is about seventy thousand times larger than the estimated external dose:

[
rac{D_ast}{D}
approx
70,000.
]

That comparison is informative. It is still not, by itself, a proof of safety.

Its interpretation depends on what (D_ast) represents, how it was obtained, which biological endpoint was studied, whether the exposure routes are comparable, how duration differs, what uncertainty factors or extrapolations are appropriate, and what dose-response relationship is assumed.

The distinction matters because different response models can agree at observed experimental doses and disagree when extrapolated elsewhere.

Under a simple threshold model,

[
Delta R(D)
=
egin{cases}
0, & D<D_0,\
f(D), & Dge D_0,
end{cases}
]

there is no additional modeled risk below (D_0).

Under a linear model without a threshold,

[
Delta R(D)=kD,
]

every positive dose corresponds to a positive modeled increment, although the magnitude becomes smaller with dose.

A nonlinear model could instead take a form such as

[
Delta R(D)
=
R_{max}
rac{D^n}{K^n+D^n}.
]

These models make different biological and regulatory assumptions. Merely knowing that (D>0) does not select among them.

This is why the slogan that "the dose makes the poison" is useful but incomplete. Dose is indispensable to toxicological reasoning, but dose alone does not determine the response without a dose-response relationship and the relevant biological context.

## Hazard and risk answer different questions

The distinction between hazard and risk is frequently blurred because both concern harmful outcomes.

Hazard identification asks whether an agent can cause a particular adverse effect under some relevant circumstances. Risk assessment asks about the probability or magnitude of harm under specified conditions of exposure.

The National Research Council's 1983 framework separated risk assessment into hazard identification, dose-response assessment, exposure assessment, and risk characterization. EPA continues to use this conceptual structure. In that framework, risk characterization integrates evidence about what an agent can do, how response changes with dose, and how much exposure actually occurs.

IARC makes the same distinction explicit for its Monographs programme. Its classifications concern carcinogenic hazard identification. They assess the strength of evidence that an agent can cause cancer. The classification itself does not report the cancer risk associated with a particular real-world exposure level.

A hazardous property can be scientifically important even when ordinary exposure produces little risk. Conversely, a modest per-unit risk can become important when exposure is widespread or repeated across a large population.

Neither hazard nor exposure should therefore be discarded. They answer different parts of the causal question.

A statement such as "this substance can cause harm" is incomplete as an estimate of risk.

A statement such as "the measured concentration is very small" is also incomplete if the agent has important effects at small doses, accumulates over time, or reaches susceptible populations through multiple routes.

The scientific task is to connect the two.

## A binary detection variable destroys information

Suppose the measured concentration is (C), but public discussion reduces it to

[
Z=
egin{cases}
1, & C 	ext{ detected},\
0, & C 	ext{ not detected}.
end{cases}
]

This transformation discards nearly all quantitative exposure information.

A sample at (0.002 mu	ext{g/L}) and another at (200 mu	ext{g/L}) can both receive (Z=1). Their concentrations differ by a factor of one hundred thousand.

At the same time, a concentration immediately below a method's detection criterion and one that is truly absent can both receive (Z=0).

The binary variable is useful for some analytical and surveillance purposes. It is a poor substitute for dose when the scientific question concerns biological effect.

This loss of information explains why headlines built around the phrase "scientists found chemical (X)" can sound more informative than they are. Presence is scientifically relevant. Without concentration, uncertainty, exposure conditions, and toxicological context, it does not specify the scale of the problem.

The more sensitive analytical chemistry becomes, the more important this distinction becomes. An instrument that can identify one molecule among vastly more background molecules increases our knowledge of composition. It does not lower the biological dose required for an effect.

## Small does not mean irrelevant

Rejecting a detection-only argument should not be replaced by the opposite slogan that trace concentrations never matter.

Some biological systems respond at very low concentrations. Potency differs by many orders of magnitude across agents. Endogenous signalling molecules can act at small concentrations. Persistent compounds can accumulate. Developmental timing can alter susceptibility. An acute dose and a chronic dose with the same daily average need not have the same consequences. Route can matter because inhalation, ingestion, dermal contact, and direct tissue exposure have different kinetics.

Population heterogeneity matters as well. Children, pregnant people, individuals with particular diseases, workers with high occupational exposures, or people with genetic differences in metabolism may not share the same dose-response relationship.

Mixtures add another complication. An exposure assessment conducted one compound at a time may not capture additive, antagonistic, or synergistic effects when several agents act through related pathways.

These possibilities are not reasons to abandon quantitative reasoning. They are reasons to improve it.

If an argument is that an unusually low concentration remains biologically important, that is an empirical claim. It can be supported by pharmacokinetics, toxicodynamics, experimental evidence, epidemiology, mechanistic evidence, or an appropriate risk model. The mere fact of detection does not supply those missing data.

## A reference value is not a wall in nature

Risk communication can create another binary mistake by treating regulatory or health-based reference values as though molecules become harmless immediately below a line and dangerous immediately above it.

That is usually not what the value means.

Reference doses, tolerable intakes, occupational limits, drinking water standards, benchmark doses, and related quantities are constructed for different purposes and under different legal and scientific frameworks. Some incorporate uncertainty factors. Some are based on particular endpoints. Some are risk-management standards rather than direct estimates of a biological threshold. Some carcinogenic risk assessments instead estimate risk under low-dose extrapolation models.

Crossing a reference value can therefore be important without implying that an adverse outcome suddenly begins at that exact number. Remaining below a reference value can be reassuring under the assumptions behind the value without proving that risk is mathematically zero.

The same caution applies to the synthetic ratio calculated earlier. The factor of seventy thousand is descriptive of two doses. Its scientific interpretation depends on the provenance of the reference point.

Numbers do not remove the need to understand what the numbers represent.

## Duration changes the exposure question

A concentration measured once describes a sample at one time and place. Chronic risk questions often concern a distribution of exposures over months, years, or decades.

Let (D_t) denote daily dose on day (t). A simple average daily dose over (T) days is

[
ar D
=
rac{1}{T}
sum_{t=1}^{T}D_t.
]

Two people can have the same (ar D) while having very different exposure patterns. One may receive a nearly constant dose. Another may receive short, high peaks separated by long periods of little exposure.

If biological response depends nonlinearly on peak concentration, those patterns need not be equivalent.

A simple toxicokinetic model makes another dimension visible. Let (B_t) denote body burden, let (u_t) represent uptake, and let (k) represent an elimination rate:

[
rac{dB_t}{dt}
=
u_t-kB_t.
]

When elimination is rapid, body burden can fall quickly after exposure stops. When (k) is small, repeated low exposures can accumulate toward a higher steady state.

The concentration in one external sample therefore cannot answer every chronic exposure question. Frequency, duration, timing, uptake, and clearance may all matter.

EPA's exposure assessment guidance explicitly treats magnitude, frequency, and duration as core elements of exposure characterization for this reason.

## Measurement uncertainty remains after detection

Once a substance is detected, the reported concentration is itself an estimate.

Suppose the laboratory reports

[
hat C = 5.0 	ext{ng/L}
]

with measurement standard uncertainty

[
u=0.8 	ext{ng/L}.
]

The scientifically relevant object is not simply the printed value 5.0. The measurement procedure supports a range of plausible values, subject to its calibration model, matrix effects, sample handling, recovery, blank correction, and other sources of uncertainty.

Near the detection limit, relative uncertainty can be substantial. This is one reason analytical chemistry distinguishes detecting an analyte from quantifying it with acceptable precision.

Armbruster and Pry's discussion of the limit of blank, limit of detection, and limit of quantitation makes this distinction explicit. A procedure may produce sufficient evidence that an analyte is present before it can estimate the concentration with the precision required for a particular quantitative use.

That distinction matters if a risk estimate is sensitive to small changes in concentration. Analytical uncertainty should propagate through the exposure calculation rather than disappearing when a laboratory result enters a spreadsheet.

If

[
D=rac{CI}{W},
]

uncertainty in (C), (I), and (W) all contribute to uncertainty in (D). More elaborate risk models add uncertainty in toxicokinetics, dose-response parameters, population variability, and model form.

A single detected concentration is therefore the beginning of a quantitative argument, not its conclusion.

## Detection can be important without proving danger

None of this makes detection scientifically unimportant.

A new detection can identify an unexpected exposure pathway. It can reveal failures in manufacturing or environmental control. It can establish that a compound reaches a tissue previously assumed to be unexposed. It can motivate targeted toxicological experiments or improved surveillance. It can show that a regulatory assumption about absence was false.

Analytical detection is evidence.

The question is evidence of what.

If the claim is that a substance is present in a sample, validated detection may answer the question directly. If the claim is that people are exposed, additional information about contact and route is required. If the claim concerns internal dose, pharmacokinetic information becomes relevant. If the claim is that a particular exposure is dangerous, dose-response and risk information are needed.

Scientific reasoning becomes weaker when all of these propositions are compressed into a single word such as "toxic".

## The missing quantities are the argument

Suppose a social media post, news report, or product warning states that a laboratory detected a chemical in a commonly consumed item.

The correct response is not to dismiss the finding because the concentration sounds small. It is also not to infer harm because the chemical name sounds unfamiliar.

The first useful quantity is the measured concentration, together with the analytical method and uncertainty. The next question is how the relevant population encounters the substance and at what frequency. That allows concentration to be translated into an exposure estimate. Toxicokinetics may then be needed to estimate internal or target-tissue dose. Only after that can an appropriate dose-response relationship begin to connect exposure to an adverse outcome.

The order matters because each step answers a different scientific question.

The sequence can be represented schematically as

[
	ext{analytical signal}
ightarrow
	ext{concentration}
ightarrow
	ext{exposure}
ightarrow
	ext{dose}
ightarrow
	ext{biological response}
ightarrow
	ext{risk}.
]

Evidence can enter at every stage, and uncertainty can enter at every stage.

A detection establishes something near the beginning of this chain. Treating it as though it already established the final term skips most of the scientific work.

The opposite shortcut fails for the same reason. A non-detect does not prove that every earlier term is zero. It says that the available analytical procedure did not establish presence above its operational criterion.

Better instruments will continue to detect more of the chemical world around us. That should increase what we know, not increase what we fear by definition.

Detection is a measurement result.

Danger is a causal and quantitative claim.

Between them lie exposure, dose, response, and uncertainty.

## References

Armbruster, D. A., & Pry, T. (2008). Limit of blank, limit of detection and limit of quantitation. *The Clinical Biochemist Reviews*, 29(Suppl 1), S49–S52.

Currie, L. A. (1968). Limits for qualitative detection and quantitative determination: Application to radiochemistry. *Analytical Chemistry*, 40(3), 586–593. https://doi.org/10.1021/ac60259a007

International Agency for Research on Cancer. (2019). *Preamble to the IARC Monographs on the Identification of Carcinogenic Hazards to Humans*. Lyon: IARC.

National Research Council. (1983). *Risk Assessment in the Federal Government: Managing the Process*. Washington, DC: National Academies Press. https://doi.org/10.17226/366

National Institute of Standards and Technology. (2026). *Limit of Detection (LOD)*. OSAC Lexicon.

U.S. Environmental Protection Agency. (2019). *Guidelines for Human Exposure Assessment*. EPA/100/B-19/001. Washington, DC: Risk Assessment Forum.
