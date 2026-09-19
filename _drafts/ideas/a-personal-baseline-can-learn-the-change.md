---
title: 'A Personal Baseline Can Learn the Change You Wanted to Detect'
permalink: /healthcare/a_personal_baseline_can_learn_the_change/
author_profile: false
classes: wide
categories:
- Healthcare
tags:
- Digital Health
- Wearable Technology
- Anomaly Detection
- Time Series
- Personalisation
excerpt: 'An adaptive reference can stop flagging a persistent change because the reference moved. A worked monitoring example separates adaptation, recovery, and missing observations.'
seo_title: 'A Personal Baseline Can Learn the Change You Wanted to Detect'
seo_description: 'Calculate how an adaptive health-monitoring baseline absorbs a persistent shift, and examine calibration uncertainty, missingness, and evaluation design.'
seo_type: article
header:
  image: /assets/images/headers/photo-statistics-scatter-correlation.jpg
  teaser: /assets/images/headers/photo-statistics-scatter-correlation.jpg
why_this_exists: 'Personalisation changes the reference against which a deviation is measured. The article makes one consequence explicit with an auditable time-series counterexample.'
evidence: 'Original synthetic step-response calculations, a Gaussian baseline-estimation example, NIST monitoring definitions, and FDA guidance on digital measurement validation.'
methodology: 'Score observations before an exponentially weighted baseline update, compare learning rates and missing-data handling, and distinguish signal detection from clinical interpretation.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: When can a personalised adaptive reference erase the deviation a monitoring system was intended to retain?
Claim: A falling deviation score can reflect adaptation of the reference while the measured change persists; update policy is part of the detection specification.
Counterclaim: Adaptation is useful when the task is to follow a changing normal state, and fixed references can accumulate irrelevant alerts.
Evidence object: Exact geometric step response, missing-observation counterexample, and baseline-estimation uncertainty calculation.
Failure case: A noiseless synthetic score cannot establish sensitivity, specificity, clinical importance, or an appropriate patient alert threshold.
Reader payoff: Specify the reference, update timing, missingness policy, and event-level evaluation before calling a monitor personalised or reliable.
Exclusions: Diagnosis, treatment advice, product comparisons, and clinical validation of the illustrative threshold.
-->

A monitoring dashboard initially flags a change. A few days later, its deviation score returns to the accepted range. The measured value has not returned to its earlier level. The algorithm has updated its idea of normal.

This behaviour can be desirable when a system is supposed to follow a new routine. It can be undesirable when the purpose is to preserve evidence of a persistent departure from a previously established state. The same update rule can serve one objective and frustrate the other.

Personalisation therefore needs a more precise description than “the model learns your baseline.” Which observations define that baseline? How quickly may it change? Can observations that triggered concern subsequently become the reference used to dismiss the concern?

We can answer those questions for a simple algorithm without pretending to validate a medical device. All values below are synthetic, expressed in arbitrary measurement units. The flag threshold illustrates software behaviour and has no clinical interpretation.

## Separate the person from the measurement process

A useful starting model for a repeated measurement is

$$
x_{it}=\mu_i+s_{it}+c_{it}+\epsilon_{it}.
$$

Here $\mu_i$ is a person's reference level, $s_{it}$ a change of interest, $c_{it}$ a contribution from measured or unmeasured context, and $\epsilon_{it}$ measurement noise. The decomposition is a modelling choice; the sensor does not label these components for us.

A pooled population reference can confuse stable differences between people with changes within a person. An individual reference can reduce that problem by asking how a person's recent values compare with their own earlier values under comparable conditions.

But subtracting a personal average does not isolate $s_{it}$ automatically. A new measurement procedure, altered daily routine, or changed context can also move the recorded series. The interpretation depends on how the data were collected and which changes the system is designed to recognise.

There is a second limitation. A stable personal reference can itself reflect a state that matters clinically. Statistical familiarity is not a definition of health. A deviation detector answers a question about departure from a reference, while an assessment of clinical significance requires additional evidence.

That distinction shapes the entire evaluation. A system that forecasts the next observation, one that recognises a sustained shift, and one that estimates an outcome risk may all consume the same sensor stream. They still have different targets and different reasons to update their references.

## Specify the order: score, then update

Let $b_{t-1}$ be the baseline available immediately before observation $x_t$ arrives. Define the deviation score

$$
r_t=x_t-b_{t-1}.
$$

After calculating the score, update the baseline with an exponentially weighted rule:

$$
b_t=(1-\alpha)b_{t-1}+\alpha x_t,
\qquad0\le\alpha\le1.
$$

An $\alpha$ of zero freezes the reference. A larger value makes it follow recent measurements faster. The baseline is a state of the algorithm: future scores depend on the entire history of its updates.

The ordering is consequential. If software updates first and then calculates $x_t-b_t$, the displayed deviation is

$$
x_t-b_t=(1-\alpha)(x_t-b_{t-1})=(1-\alpha)r_t.
$$

The current observation has partly moved its own comparator. With $\alpha=0.25$, that alternative ordering immediately reduces every displayed deviation by 25 percent. An implementation can therefore change the effective threshold without changing the threshold value in its configuration.

For the rest of this article, scoring always precedes updating. This is an explicit definition, not a claim that one ordering is correct for every possible forecasting or monitoring task.

## A change that never recovers

Start with $b_0=0$. From day one onward, let every observed value equal four:

$$
x_t=4,\qquad t\ge1.
$$

There is no noise, no recovery, no further drift, and no missing data. This deliberately removes competing explanations. Solving the recursion gives

$$
b_{t-1}=4\left[1-(1-\alpha)^{t-1}\right],
\qquad
r_t=4(1-\alpha)^{t-1}.
$$

For every $0<\alpha\le1$, the deviation eventually reaches zero or approaches it. The recorded level remains four. The baseline moves towards it.

Set an illustrative flag rule $r_t>2.5$. We use a strict inequality so the behaviour at the threshold is unambiguous. The results over thirty observed days are:

| Update weight | Flagged observed days | First unflagged observed day | What the measurement did |
| --- | ---: | ---: | --- |
| 0, frozen | All 30 | None within the window | Stayed at 4 |
| 0.02 | Days 1–24 | 25 | Stayed at 4 |
| 0.10 | Days 1–5 | 6 | Stayed at 4 |
| 0.25 | Days 1–2 | 3 | Stayed at 4 |

For $\alpha=0.1$, the scores begin $4,3.6,3.24,2.916,2.6244,2.36196$. On day six the flag disappears, despite a perfectly unchanged measurement since day one.

![A persistent synthetic measurement stays at four while adaptive baselines rise towards it and deviation scores fall below an illustrative threshold; the frozen reference retains the full deviation.](/assets/images/figures/healthcare_adaptive_baseline.png){: width="1785" height="827" loading="lazy"}

*Original deterministic counterexample. The threshold is arbitrary. The figure demonstrates the baseline update mechanism, not a measured false-negative rate or a clinical alert policy.*

For a positive shift $\Delta$ and threshold $h<\Delta$, the condition is

$$
\Delta(1-\alpha)^{t-1}>h.
$$

When $0<\alpha<1$, taking logarithms gives

$$
t-1<\frac{\log(h/\Delta)}{\log(1-\alpha)}.
$$

This identifies how many observed updates can occur before the flag fades in the noiseless example. Increasing the learning rate shortens that period. A parameter presented as a responsiveness setting also controls how quickly the reference absorbs a persistent departure.

## An adaptive reference and an EWMA control chart differ

The recursion is familiar from exponential smoothing, but the role of the smoothed quantity matters. In the construction above, the moving baseline is subtracted from each new observation, so it follows a sustained shift and reduces the residual.

An EWMA control chart can instead compare the smoothed statistic with limits tied to an established reference process. In that arrangement, smoothing helps reveal a persistent change rather than automatically redefining it as normal. NIST describes the EWMA statistic and its use in detecting smaller or gradual process shifts. [NIST: EWMA Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc314.htm)

The conclusion is therefore specific to what we score and what we allow to move. It would be incorrect to infer that exponentially weighted methods inherently fail at change detection. The same mathematical building block can be used in systems with different reference states and decision rules.

An evaluation should name the score explicitly. “We use an EWMA” does not tell a reviewer whether the system monitors a smoothed level against a fixed target, monitors innovations relative to a changing predictor, or applies some additional rule to retain a previously detected event.

## Adaptation has a legitimate job

A frozen reference also has failure modes. If the measurement process changes for a known and acceptable reason, a permanently fixed comparator can continue producing irrelevant deviations. Repeated nuisance flags can burden whatever review process receives them.

The design problem is to specify which changes should update the reference and which should remain visible. That choice cannot be recovered from prediction accuracy alone. A fast learner may forecast a new level extremely well precisely because it has stopped treating the level as a departure.

Possible designs include a separate long-term reference and short-term predictor, an explicit review before resetting the reference, or a persistent event state that is not cleared solely because the numerical residual shrank. Each introduces a policy that must be evaluated.

For example, freezing updates whenever a score exceeds a threshold prevents the particular absorption shown here. It also selects which observations may define normal. A nuisance excursion can leave a stale reference in place, and a series of small shifts can still be admitted. Gating is a different algorithm with its own operating characteristics, not a universal repair.

Similarly, maintaining two baselines creates another question: which disagreement matters, over what duration, and who or what resolves it? Adding a second curve is easy. Establishing the meaning of its decisions requires a stated task and relevant data.

## A short calibration period adds uncertainty

So far the starting reference has been known exactly. In practice it is often estimated from a finite calibration window.

To isolate the statistical effect, suppose the calibration observations are independent $N(\mu,\sigma^2)$ measurements, with known $\sigma$, and the baseline is their mean $\bar X_n$. Assume a future unchanged observation is independent of the calibration sample. Then

$$
X_{\mathrm{new}}-\bar X_n
\sim N\left(0,\sigma^2\left(1+\frac1n\right)\right).
$$

The deviation contains uncertainty from both the new observation and the estimated reference. Dividing only by $\sigma$ ignores the latter.

For a two-sided rule $|X_{\mathrm{new}}-\bar X_n|>3\sigma$, the marginal flag probability under this unchanged Gaussian model is

$$
2\left[1-\Phi\left(\frac3{\sqrt{1+1/n}}\right)\right].
$$

It is about 0.501 percent for seven calibration observations and 0.320 percent for twenty-eight. With a known mean, the corresponding three-standard-deviation probability is about 0.270 percent.

These values concern one future measurement under the model. They are not daily medical false-alert rates. Estimating $\sigma$, serial dependence, changing context, and a calibration period that already contains a shift all alter the calculation.

Repeated scores also share the same estimated reference. Even when future raw observations are independent, subtracting that common random estimate induces dependence between their deviations. We should not take the one-measurement probability and blindly apply an independence formula to obtain the chance of any flag over a year.

The practical lesson is that “personal” does not mean “known precisely.” A baseline needs a measure of how it was established and how uncertain it remains.

## Missing time and observed time are different clocks

The step-response table counts observed updates. Suppose data arrive only every second day and the system updates only when a valid observation is available. Its sixth observed value occurs on calendar day eleven, not day six.

The same learning rate now has a different calendar-time meaning. A time-aware rule can define its weighting using elapsed time, but that is a new specification. In particular, more elapsed time does not itself provide a measurement of the state during the gap.

In the accompanying implementation, a missing observation is `None`. It produces no score and leaves the baseline unchanged. A sequence `[4, None, None, 4]` therefore makes two updates. It does not make four updates containing two normal or zero-valued measurements.

Replacing missing values with zero would drag the baseline towards zero even if zero has no relevant measurement meaning. Carrying the last value forward and updating repeatedly would give one actual observation extra influence. Both choices can change the future detection behaviour while making the dataset look more complete.

Missingness can also depend on the situation being monitored. A device may be worn differently during particular activities, or valid observations may be less available in contexts important to the target. A dataset containing only successful recordings cannot establish performance during those gaps.

The December 2023 FDA guidance on digital technologies for remote data acquisition discusses validation for the proposed use and population, as well as plans for missing data and data quality. It provides relevant context for evaluating a measurement pipeline; it does not validate this synthetic detector or prescribe its threshold. [FDA guidance, sections IV.C and IV.E](https://www.fda.gov/media/155022/download)

## Evaluate episodes and preserve the state history

A useful evaluation should reproduce the chronology of deployment. Estimate the initial reference from an earlier period, process subsequent observations in order, and apply only information available at each time. Randomly mixing future and past records can let the reference learn from the very changes it is supposed to detect.

For personalisation, distinguish performance for people with an established calibration history from performance for new people. Those are different information settings. Combining them into a single score can hide the uncertainty of the initial period.

The endpoint should also match the task. Detecting the onset of an episode, retaining a flag while a condition persists, and correctly clearing a resolved episode are different behaviours. In our step example, every adaptive configuration detects the first observation. Their difference concerns how long the deviation remains visible, so an onset-only metric would miss it entirely.

Useful measurements may include time to first detection, the fraction of a labelled episode covered by a flag, unresolved-event duration, nuisance events per person-time, and the proportion of time with enough data to evaluate. Appropriate labels and denominators must be defined before these can be interpreted.

Retain the state needed to explain decisions: the incoming observation, its quality status, the baseline before scoring, the score, whether an update occurred, the baseline afterwards, and the algorithm version. Without that record, a falling score may be impossible to distinguish from a changed measurement or a changed reference.

## Reproduce the counterexample

The [calculation and figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_coverage_draft_figures.py) prints all thirty steps for each learning rate with `--dry-run`. Tests compare the recursive implementation with the exact geometric solution and verify that missing observations neither generate scores nor update the reference.

The essential calculation is short enough to inspect directly:

```python
baseline = 0.0
alpha = 0.1
for day in range(1, 8):
    observation = 4.0
    deviation = observation - baseline
    flagged = deviation > 2.5
    print(day, round(deviation, 6), flagged)
    baseline += alpha * deviation
```

The sixth line reports an unflagged observation. Nothing in the input has recovered. A meaningful explanation of that result must include the moving reference and the purpose it was intended to serve.
