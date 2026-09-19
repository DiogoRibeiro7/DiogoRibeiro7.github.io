---
author_profile: false
categories:
- Healthcare
classes: wide
title: 'Passive Sensing Is a Measurement Model Before It Is a Prediction Model'
excerpt: RSSI, wearables, and smart-home signals do not observe behaviour directly. Geometry, calibration, missingness, device placement, and noise determine what can be inferred before machine learning begins.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- passive sensing
- RSSI
- wearables
- smart home
- measurement error
- Gaussian processes
seo_title: 'Passive Sensing Is a Measurement Model Before It Is a Prediction Model'
seo_description: 'Why passive health sensing should begin with geometry, calibration, observation noise, missingness, and measurement validity before choosing an ML model.'
seo_type: article
summary: 'A sensing system is first a measurement instrument. Prediction quality is bounded by what the sensors observe, how the environment distorts those observations, and whether the measurement process remains stable.'
tags:
- Passive Sensing
- Measurement Error
- Gaussian Processes
- Digital Health
why_this_exists: 'Passive-sensing projects often jump directly from raw signals to classification. The harder problem is usually establishing what physical or behavioural quantity the signal actually measures.'
evidence: 'RSSI localisation examples, wearable sampling examples, calibration drift, geometry effects, and measurement-error reasoning.'
methodology: 'Write an explicit observation model before the predictive model, then examine how calibration, missingness, and environmental changes propagate into inference.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: What can a passive sensor actually observe before ML is applied?
Claim: Passive sensing is first a measurement problem; prediction cannot recover distinctions absent from or confounded in the sensor process.
Counterclaim: Flexible models can sometimes learn around complex measurement effects if training data cover the deployment environment.
Evidence object: RSSI geometry example, wearable missingness example, and environmental calibration shift.
Failure case: Explicit physical models can be too simple for complex indoor environments.
Reader payoff: Diagnose observability limits before spending effort on model architecture.
Exclusions: A survey of wearable devices.
-->

A smart-home classifier says that someone is in the kitchen.

A wearable says that activity increased.

An RSSI system says that a device moved toward the bedroom.

Each statement sounds like a prediction problem.

Before it is a prediction problem, it is a measurement problem.

The sensor never observes “kitchen”, “activity”, or “bedroom” directly.

It observes voltage, signal strength, acceleration, packet timing, or another physical quantity from which the behavioural variable is inferred.

## Write the observation model first

Let (z_t) denote the latent behavioural state.

Let (y_t) denote the sensor observation.

The measurement problem is

$$
y_t sim p(y_t mid z_t, c_t),
$$

where (c_t) represents context: geometry, device placement, obstruction, calibration, temperature, battery, or other conditions.

The prediction problem begins only after this mapping is understood.

If (c_t) changes, the same behaviour can produce different sensor values.

## RSSI is an environmental measurement

Received signal strength depends on distance, but also on walls, orientation, multipath propagation, antenna placement, interference, and device hardware.

A simple path-loss model is

$$
	ext{RSSI}(d) = A - 10nlog_{10}(d/d_0) + epsilon.
$$

The parameters are environmental.

Treating RSSI as a direct distance sensor therefore embeds assumptions about the space.

A Gaussian process can be useful because it allows spatial structure and uncertainty to be learned without pretending that one deterministic path-loss curve is exact.

But the GP does not remove the measurement problem.

It models it.

## Geometry determines identifiability

If two rooms produce nearly identical signal patterns, no classifier can reliably distinguish them from RSSI alone.

This is not underfitting.

It is lack of information.

Adding a deeper network cannot create a separation absent from the sensor geometry.

The right response may be to move a receiver, add another sensor, or redefine the target.

That is system design, not hyperparameter tuning.

## Missingness can be behavioural

Wearables are often missing data when the device is removed.

Removal may be correlated with sleep, charging, bathing, discomfort, illness, or non-adherence.

The missingness pattern can therefore contain information about behaviour.

Imputing it as if the device had continued sampling can erase that signal.

Again, observation process comes first.

## Calibration drift can masquerade as behaviour change

A device firmware update changes its signal scale.

A new router changes RSSI distribution.

A wearable band is tightened differently.

The model sees a shift.

Nothing about the person changed.

This is why passive-sensing systems need calibration monitoring alongside prediction monitoring.

Otherwise the model can convert instrument drift into behavioural inference.

## Population models and personal baselines answer different questions

A population model asks whether the current pattern resembles patterns seen across many people.

A personal baseline asks whether the current pattern differs from this person's own history.

Both can be useful.

But neither is meaningful unless the measurement process is stable enough that observed change plausibly reflects behavioural change.

A personal model can adapt to sensor drift and accidentally normalise it.

A population model can mistake between-home geometry for between-person behaviour.

The measurement layer mediates both.

## More sensors do not automatically solve observability

Adding sensors can help if they contribute independent information.

It can also create redundant noise.

Sensor fusion should therefore be thought of as uncertainty reduction rather than feature accumulation.

The useful question is:

> Which ambiguity does this additional sensor resolve?

If the answer is unclear, the extra channel may add complexity without identifiability.

## Validation should include perturbations of the measurement system

A sensing model should be tested under changes such as:

- receiver relocation;
- device orientation;
- furniture changes;
- missing packets;
- battery state;
- sensor replacement;
- different homes.

This is the equivalent of stress-testing the observation model.

Random train/test splits inside one stable environment can badly overestimate deployment robustness.

## Prediction cannot rescue an invalid label either

The target variable also belongs to measurement.

If “sleep” is derived from another imperfect device, or “activity” is labelled from sparse self-report, the model is trained against a noisy operational definition.

High predictive performance may then mean agreement with the label-generating instrument rather than agreement with the underlying clinical construct.

Measurement error can exist on both sides.

## The right model may be simpler than expected

Once the observation structure is understood, a transparent model may be enough.

A GP, state-space model, calibrated logistic model, or probabilistic classifier can outperform a more complex architecture in usefulness because it exposes uncertainty and environmental dependence.

The modelling stance should follow the measurement problem.

Not the other way around.

## Conclusion

Passive sensing does not begin with machine learning.

It begins with the question of what the sensor actually observes.

Geometry, calibration, environmental context, missingness, and label construction determine what information reaches the model.

A better classifier cannot recover information that the measurement system never captured.

The first architecture is therefore the measurement architecture.

## References

- Rasmussen CE, Williams CKI. *Gaussian Processes for Machine Learning*. MIT Press.
- Särkkä S. *Bayesian Filtering and Smoothing*. Cambridge University Press.
- Hightower J, Borriello G. Location systems for ubiquitous computing. *Computer*. 2001.
- Honkavirta V, Perälä T, Ali-Löytty S, Piché R. A comparative survey of WLAN location fingerprinting methods. *WPNC*. 2009.
