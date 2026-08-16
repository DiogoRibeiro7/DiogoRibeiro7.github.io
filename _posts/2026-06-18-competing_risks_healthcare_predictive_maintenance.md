---
permalink: '/statistics/competing_risks_healthcare_predictive_maintenance/'
redirect_from:
- '/statistics/data science/healthcare/competing_risks_healthcare_predictive_maintenance/'
title: "Competing Risks in Healthcare and Predictive Maintenance"
categories:
- Statistics
- Data Science
- Healthcare
tags:
- Competing Risks
- Survival Analysis
- Predictive Maintenance
- Healthcare Analytics
- Cumulative Incidence
- Risk Modeling
author_profile: false
seo_title: "Competing Risks in Healthcare and Predictive Maintenance"
seo_description: 'Competing risks in healthcare and predictive maintenance: cumulative incidence, cause-specific hazards, Fine-Gray models, and censoring.'
excerpt: "Competing risks occur when more than one event can happen, and one event changes or prevents the chance of observing another."
summary: "This article explains competing risks as a practical statistical framework for healthcare and predictive maintenance. It covers why ordinary survival analysis can mislead when event types compete, how cumulative incidence differs from Kaplan-Meier estimates, when to use cause-specific hazards and Fine-Gray models, and how these ideas support clinical and maintenance decisions."
keywords:
- "competing risks"
- "survival analysis"
- "cause-specific hazard"
- "Fine-Gray model"
- "cumulative incidence"
- "predictive maintenance statistics"
classes: wide
date: '2026-06-18'
header:
  image: /assets/images/statistics_teaser.jpg
  og_image: /assets/images/statistics_teaser.jpg
  overlay_image: /assets/images/statistics_teaser.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/statistics_teaser.jpg
  twitter_image: /assets/images/statistics_teaser.jpg
---

Many real-world prediction problems are not about whether an event happens. They are about which event happens first.

In healthcare, a patient may experience disease recurrence, treatment toxicity, discharge, readmission, transplant, or death. In predictive maintenance, a machine may fail because of bearing wear, overheating, corrosion, software faults, operator misuse, or scheduled replacement. Once one event occurs, it may prevent or fundamentally change the chance of observing the others.

This is the setting of competing risks.

Competing risks matter because ordinary survival analysis can answer the wrong question when multiple event types are possible. A model that treats every non-target event as ordinary censoring may estimate a probability that cannot occur in the real world. It may say that the chance of one failure mode is high while ignoring that another failure mode removes many assets from risk earlier. It may overstate recurrence risk in patients who die before recurrence can be observed. It may overstate component failure risk when many components are replaced preventively before failure.

The statistical issue is not subtle. If events compete, the way we define risk changes.

## What Is a Competing Risk?

A competing risk is an event that prevents the event of interest from occurring or changes the probability that it can be observed.

Suppose the event of interest is hospital readmission within 30 days. Death before readmission is a competing event because a patient who dies cannot later be readmitted in the usual sense. Treating death as simple censoring assumes that the patient remains comparable to someone who was merely lost to follow-up. That assumption is usually wrong.

Suppose the event of interest is bearing failure in an industrial motor. Scheduled replacement before bearing failure is a competing event. Once the bearing is replaced, the original bearing can no longer fail. Treating replacement as ordinary censoring can exaggerate the probability of bearing failure under actual maintenance policy.

The central structure is:

$$
T = \text{time to first event}
$$

and

$$
J = \text{type of first event}
$$

The pair \( (T, J) \) tells us both when something happened and what happened.

## Censoring Is Not the Same as Competing

Competing risks are often confused with censoring.

Censoring occurs when follow-up ends before the event is observed. A patient moves away. A study ends. A sensor stops reporting. A machine leaves the fleet. The event may still occur later, but we do not observe it.

A competing event is different. It is observed, and it changes the event process. A patient dies. A component is replaced. A machine is decommissioned. A loan prepays before default. A customer cancels before upgrade.

Ordinary right-censoring methods assume that censored cases would continue to have the same event process as comparable uncensored cases. That may be plausible for administrative study end. It is not plausible when the "censoring" event is actually death, replacement, decommissioning, or another absorbing outcome.

Calling a competing event censoring does not make it harmless. It changes the estimand.

## The Problem with Kaplan-Meier in Competing Risks

The Kaplan-Meier estimator estimates survival for a single event type under censoring assumptions. If we use Kaplan-Meier for one event type and censor all competing events, we estimate a hypothetical world in which competing events do not occur.

That may be useful for some scientific questions, but it is not the same as the real-world probability of the event.

For example, imagine 100 high-risk patients after surgery:

- 15 experience recurrence
- 20 die before recurrence
- 65 remain alive without recurrence during follow-up

If deaths are censored, Kaplan-Meier estimates recurrence risk among patients treated as if death did not remove them from the risk set. The resulting recurrence probability can be higher than the actual probability that a patient experiences recurrence before death.

In maintenance, suppose 100 pumps are monitored:

- 10 fail from seal degradation
- 30 are replaced preventively
- 60 continue operating

If preventive replacement is censored, a Kaplan-Meier estimate of seal failure may describe what might happen if replacements did not occur. That is not the same as the observed probability of seal failure under the maintenance policy.

The practical question is usually not "what would happen if other events disappeared?" It is "what is the probability that this event happens before the others?"

That question is answered by cumulative incidence.

## Cumulative Incidence

The cumulative incidence function for event type \( k \) is:

$$
F_k(t) = P(T \leq t, J = k)
$$

It is the probability that event type \( k \) has occurred by time \( t \) before any competing event.

If the event is disease recurrence, cumulative incidence estimates the probability of recurrence by time \( t \) in the presence of death and other events. If the event is bearing failure, it estimates the probability of bearing failure by time \( t \) in the presence of replacement, other failure modes, and decommissioning.

The cumulative incidence functions across event types add up to the probability that any event has occurred:

$$
\sum_k F_k(t) = P(T \leq t)
$$

This is a major advantage. It keeps the probabilities grounded in the real event process. The risk of recurrence, death without recurrence, and other outcomes can be shown together rather than pretending each event exists in isolation.

## Cause-Specific Hazards

The cause-specific hazard for event type \( k \) is the instantaneous rate of event \( k \) among subjects who have not yet experienced any event:

$$
\lambda_k(t) =
\lim_{\Delta t \to 0}
\frac{P(t \leq T < t+\Delta t, J=k \mid T \geq t)}{\Delta t}
$$

This asks: among those still event-free at time \( t \), how quickly are they experiencing event \( k \)?

Cause-specific hazard models are useful when the scientific or engineering question is about mechanisms. In healthcare, we may ask which covariates increase the instantaneous rate of recurrence among patients who are still alive and recurrence-free. In maintenance, we may ask which operating conditions increase the instantaneous rate of overheating failure among machines that have not yet failed or been replaced.

A common approach is to fit one Cox model per event type, treating other event types as censored for that cause-specific analysis.

This is valid for estimating cause-specific hazard relationships. But the coefficients should not be interpreted directly as effects on cumulative incidence without care. A variable can increase the hazard of one event and also increase the hazard of a competing event, producing non-obvious effects on real-world event probabilities.

## Fine-Gray Models

Fine-Gray models target the subdistribution hazard, which is connected more directly to the cumulative incidence function.

Where cause-specific hazards focus on event rates among those still event-free, Fine-Gray models keep people or assets who experienced competing events in a modified risk set. This makes the model useful for estimating how covariates affect cumulative incidence of a specific event.

The practical distinction is:

- Cause-specific hazard models are often better for etiological or mechanism questions.
- Fine-Gray models are often better for direct prediction of event probability in the presence of competing risks.

In healthcare, a cause-specific model may help study whether a biomarker is associated with recurrence among patients still alive. A Fine-Gray model may help estimate a patient's probability of recurrence by five years while accounting for death as a competing risk.

In maintenance, a cause-specific model may help identify conditions that accelerate seal failure. A Fine-Gray model may help estimate the probability that seal failure is the first event before preventive replacement or another failure type.

Both models can be useful. They answer different questions.

## Healthcare Example

Consider patients treated for cancer. The event of interest is recurrence within five years. Death before recurrence is a competing event.

A naive analysis might censor deaths and use Kaplan-Meier to estimate recurrence. This estimates recurrence in a hypothetical setting where death does not preclude recurrence. For clinical decision-making, that can be misleading because patients and clinicians need absolute risk in the real world.

A competing-risks analysis would estimate:

- Cumulative incidence of recurrence
- Cumulative incidence of death without recurrence
- Effects of covariates on recurrence hazard
- Effects of covariates on death hazard
- Predicted recurrence probability by time horizon

This can change clinical interpretation. A treatment may reduce recurrence but increase treatment-related mortality. A frailty score may be strongly associated with death before recurrence, lowering observed recurrence incidence while worsening overall prognosis. Age may affect recurrence and death in different directions.

Without competing-risks thinking, these patterns can be misread.

## Maintenance Example

Consider a fleet of industrial pumps. The event of interest is mechanical seal failure. Competing events include motor failure, scheduled replacement, decommissioning, and failure from another component.

A naive analysis may censor all non-seal events and estimate seal-failure probability with Kaplan-Meier. This may answer a narrow engineering question: what is the latent seal-failure process if other events are ignored? But a maintenance planner often needs a different answer: how likely is seal failure to occur before replacement or another failure mode?

A competing-risks analysis can estimate:

- Probability of seal failure by operating age
- Probability of motor failure before seal failure
- Probability of preventive replacement before failure
- Effect of temperature, vibration, load, and fluid properties on each event type
- Which assets are likely to fail from which cause first

This helps maintenance teams prioritize inspections, spare parts, and intervention timing. It also prevents over-investing in a failure mode that appears severe only because competing events were incorrectly censored.

## Competing Risks and Decision-Making

Competing-risks models are useful because decisions often depend on absolute probabilities.

A clinician may need to compare recurrence risk with treatment toxicity and mortality. A maintenance planner may need to compare failure risk with replacement cost and downtime. An insurer may need to compare claim types. A hospital may need to compare discharge, readmission, and death.

The decision is rarely based on one hazard in isolation.

For example, suppose two machines have the same cause-specific hazard of bearing failure. Machine A has a high probability of being replaced soon because it is scheduled for upgrade. Machine B is expected to remain in service. The cumulative incidence of bearing failure may be much lower for Machine A because replacement competes with failure.

The same logic appears in health. Two patients may have similar recurrence hazard, but different mortality risk. Their cumulative probability of recurrence before death can differ.

Good decisions require the probability of events in the world where competing outcomes exist.

## Covariates Can Behave Differently by Event Type

A key advantage of competing-risks analysis is that covariates can have different relationships with different event types.

In healthcare:

- Age may increase death risk more than recurrence risk.
- A treatment may reduce recurrence but increase toxicity-related discontinuation.
- A biomarker may predict disease progression but not non-disease mortality.
- Comorbidities may dominate competing mortality risk.

In maintenance:

- Temperature may increase overheating failures.
- Vibration may predict bearing failures.
- Corrosive environments may increase seal degradation.
- Asset age may increase failure risk but also trigger preventive replacement.

Lumping all events into a single "failure" outcome can hide these differences. Sometimes that is acceptable if any event has the same operational consequence. Often it is not.

If the intervention differs by event type, the model should usually distinguish event types.

## Data Requirements

Competing-risks analysis needs carefully structured data.

At minimum, each subject or asset should have:

- A start time
- An end time
- An event indicator
- An event type for observed events
- Covariates measured before or during follow-up
- A clear definition of censoring

For healthcare, the start time might be surgery, diagnosis, treatment initiation, discharge, or enrollment. For maintenance, it might be installation, last overhaul, commissioning, or start of monitoring.

The start time matters. Mixing different origins can produce misleading risk estimates.

Event definitions also matter. In maintenance logs, "failure" may be recorded inconsistently across technicians or sites. In healthcare records, recurrence, progression, readmission, and death may come from different systems with different delays. Statistical sophistication cannot rescue poorly defined event types.

## Time-Varying Covariates

Many covariates change over time.

In healthcare, lab values, medication exposure, disease status, and care plans evolve. In maintenance, vibration, temperature, load, operating hours, inspection results, and lubricant quality change continuously.

Using future covariate values creates leakage. A model predicting recurrence risk at discharge should not use lab values measured after discharge. A maintenance model predicting failure risk at the start of a week should not use sensor summaries from later in that week.

Time-varying covariates must be aligned with prediction time. This often requires a counting-process format or repeated risk snapshots.

The rule is simple: the model should only use information available at the time the risk prediction would be made.

## Visualization

Competing risks should be visualized, not only modeled.

Useful plots include:

- Cumulative incidence curves by event type
- Stacked event probability curves
- Cause-specific hazard curves
- Event-type distributions by subgroup
- Predicted risk curves for representative profiles
- Calibration plots for event-specific risk predictions

In healthcare, cumulative incidence curves can show recurrence and death without recurrence side by side. In maintenance, they can show seal failure, motor failure, replacement, and decommissioning as competing first events.

These plots help stakeholders understand that risks share probability mass. If one event becomes more likely, another may become less likely simply because fewer subjects remain able to experience it first.

## Common Mistakes

The first mistake is censoring competing events and interpreting Kaplan-Meier estimates as real-world probabilities.

The second mistake is merging distinct event types into one composite outcome when different actions depend on the cause.

The third mistake is interpreting cause-specific hazard ratios as direct effects on cumulative incidence.

The fourth mistake is ignoring informative censoring. If assets leave observation because of risk-related reasons, or patients are lost to follow-up because of health status, estimates may be biased.

The fifth mistake is using covariates measured after the prediction time.

The sixth mistake is reporting model coefficients without absolute risks. Decision-makers usually need probabilities by time horizon, not only relative effects.

## Practical Workflow

A practical competing-risks workflow can be organized as follows.

Define the time origin. Be explicit about when follow-up begins.

Define all event types. Separate true censoring from competing events.

Estimate cumulative incidence. Start with nonparametric cumulative incidence curves before fitting regression models.

Fit cause-specific models if the goal is mechanism, etiology, or event-process understanding.

Fit Fine-Gray or other cumulative-incidence models if the goal is direct prediction of absolute event probability.

Check calibration by event type and time horizon.

Perform sensitivity analysis for censoring assumptions, event definitions, and delayed event recording.

Translate results into decisions. Show event probabilities, not only hazard ratios.

This workflow keeps the statistical question connected to the operational question.

## Conclusion

Competing risks appear whenever several mutually exclusive events can occur and the first event changes the chance of observing the others. That makes them central to healthcare, reliability engineering, maintenance planning, finance, insurance, and many other applied domains.

The main lesson is that event probabilities must be modeled in the world where competing events exist. Kaplan-Meier estimates that censor competing events may answer a hypothetical question, but they often overstate real-world risk.

Cumulative incidence estimates absolute probability. Cause-specific hazards explain event processes. Fine-Gray models connect covariates to cumulative incidence. Together, these tools help analysts distinguish mechanism from prediction and relative effects from decision-relevant risk.

In healthcare, this can clarify recurrence, mortality, readmission, and treatment risk. In predictive maintenance, it can separate failure modes, replacement policies, and decommissioning. In both domains, competing-risks analysis prevents a common statistical error: acting as if only one kind of event can happen.

## References

- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220.
