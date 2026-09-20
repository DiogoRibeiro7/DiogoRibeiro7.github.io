---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-03-29'
excerpt: Real-time epidemiological surveillance is a streaming inference problem with delayed, revised, incomplete, and privacy-sensitive data. Low latency is useful only when statistical calibration survives those data constraints.
header:
  image: /assets/images/headers/photo-data-streaming.jpg
  og_image: /assets/images/headers/photo-data-streaming.jpg
  overlay_image: /assets/images/headers/photo-data-streaming.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-streaming.jpg
  twitter_image: /assets/images/headers/photo-data-streaming.jpg
keywords:
- epidemiological surveillance
- streaming data
- event time
- Apache Flink
- outbreak detection
seo_description: Real-time epidemiological surveillance explained through event-time processing, delayed reports, revisions, statistical alerting, data quality, and privacy.
seo_title: 'Real-Time Epidemiological Surveillance: Streaming Data Is Not Enough'
seo_type: article
summary: A systems-and-statistics view of real-time surveillance that separates stream processing from outbreak inference and explains event time, late data, revision, alert calibration, privacy, and reproducibility.
tags:
- Data Engineering
- Epidemiology
- Streaming
title: 'Real-Time Epidemiological Surveillance: Streaming Data Is Not Enough'
---

Real-time epidemiological surveillance is often framed as a data-engineering problem.

Ingest events quickly.

Process them in a streaming engine.

Update a dashboard.

That is necessary infrastructure.

It is not the surveillance method.

A useful system has two coupled layers:

$$
\boxed{
\text{stream processing}
+
\text{statistical inference}.
}
$$

Low-latency processing cannot compensate for delayed diagnoses, incomplete reports, duplicate records, or a badly calibrated outbreak detector.

## Event time and processing time

Streaming systems distinguish the time an event occurred from the time the platform received it.

Let

$$
t_e
$$

be event time and

$$
t_p
$$

processing time.

In public-health data,

$$
t_p-t_e
$$

can vary substantially because of:

- laboratory turnaround;
- reporting practices;
- weekends;
- batching;
- data-entry delay;
- network failure.

A system that groups only by processing time can create artificial spikes when delayed records arrive together.

## Late data

Modern streaming engines use concepts such as watermarks to decide when an event-time window is sufficiently complete for computation.

But epidemiological reporting has no magical point at which late data cease to exist.

The engineering policy must be connected to the epidemiological reporting-delay model.

A one-hour lateness threshold may be reasonable for telemetry and absurd for laboratory surveillance.

## Revisions are part of the data model

Case counts can be revised because records are:

- deduplicated;
- reclassified;
- assigned to a different onset date;
- corrected after laboratory confirmation;
- removed after quality review.

A real-time system must therefore support updates, not only append-only events.

The current estimate for historical date $t$ is a versioned object:

$$
Y_t^{(v)},
$$

where $v$ is the data vintage.

This is essential for honest retrospective evaluation.

## Nowcasting

If the recent count is incomplete, the system can estimate the final count.

Let

$$
N_{t,d}
$$

be cases occurring at time $t$ and reported with delay $d$.

A nowcast estimates

$$
Y_t
=
\sum_dN_{t,d}
$$

before all delays have arrived.

The uncertainty in that estimate should propagate into any alerting layer.

Treating the partial count as if it were final biases recent trends downward.

## Streaming architecture

A generic architecture can include:

1. event ingestion;
2. schema validation;
3. deduplication;
4. event-time assignment;
5. late-data handling;
6. aggregation;
7. statistical model update;
8. alert generation;
9. audit logging.

The streaming platform may be Apache Flink, Kafka Streams, Spark Structured Streaming, or another system.

The statistical requirements should not depend on one vendor.

## Why Flink can be useful

Flink supports stateful event-time processing and watermarks.

Those capabilities fit surveillance streams where records arrive out of order.

For example, a keyed stream can maintain counts by:

- region;
- syndrome;
- age group;
- laboratory;
- time window.

Stateful processing makes it possible to update derived quantities as new reports arrive.

The epidemiological validity still depends on how those quantities are defined.

## Idempotence and duplicate reports

Health records may be resent.

If a repeated message increments the count twice, the streaming system creates a false outbreak.

Events should have stable identifiers when possible.

The update operation should be idempotent:

$$
f(f(S,e),e)
=
f(S,e).
$$

Applying the same event twice should not change the state after the first application.

This is a data-quality requirement with direct statistical consequences.

## Alerting is a sequential-testing problem

Suppose each region is evaluated every hour.

A naive rule

$$
p_t<0.05
$$

at each time point produces repeated opportunities for false alarms.

Long-run false-alert behavior depends on the entire sequential procedure.

Useful performance measures include:

- alerts per month under baseline;
- average run length;
- detection delay;
- probability of detection;
- calibration across regions.

One threshold from a static hypothesis test is not enough.

## Baselines drift

Healthcare usage changes over time.

So do:

- population size;
- diagnostic access;
- coding practice;
- vaccination coverage;
- circulating pathogens;
- reporting systems.

An outbreak detector needs a baseline model that can adapt without immediately absorbing the outbreak it is supposed to detect.

That creates a bias-variance trade-off in baseline updating.

## Privacy and access control

Real-time health streams can contain highly sensitive information.

System design should minimize unnecessary identifiers and enforce access control.

Useful principles include:

- data minimization;
- role-based access;
- encryption;
- audit trails;
- retention policy;
- aggregation when individual-level detail is unnecessary.

Privacy is not a downstream dashboard setting.

It belongs in the data architecture.

## Reproducibility

A dashboard value should be traceable to:

- source records;
- schema version;
- transformation code;
- model version;
- configuration;
- data vintage.

Without lineage, a real-time number cannot be reproduced after the stream has changed.

This is especially important when public-health decisions depend on an alert.

## Failure modes

A surveillance platform can fail while every server remains healthy.

Examples include:

- silent underreporting;
- a laboratory feed disappearing;
- duplicated messages;
- changed coding standards;
- delayed batch arrival;
- demographic fields becoming missing;
- geographic reassignment.

Monitoring should therefore include data-quality metrics, not only CPU and latency.

## Machine learning is optional

An outbreak detector does not need a neural network.

Simple count models, state-space models, CUSUM procedures, or Bayesian surveillance models can be easier to calibrate and audit.

Machine learning may help with:

- text classification;
- syndrome extraction;
- anomaly scoring;
- missing-data prediction.

The model should enter because it solves a defined problem, not because the architecture is “real time.”

## Conclusion

Real-time surveillance is not just fast data.

The pipeline is

$$
\boxed{
\text{event time}
\rightarrow
\text{quality control}
\rightarrow
\text{revision-aware aggregation}
\rightarrow
\text{nowcasting}
\rightarrow
\text{sequential detection}
\rightarrow
\text{auditable alert}.
}
$$

A streaming framework can implement the pipeline.

It cannot define the epidemiological model for us.

## References

- Höhle, M. (2007). Surveillance: An R package for the monitoring of infectious diseases. *Computational Statistics*, 22, 571–582.
- Salmon, M., Schumacher, D., & Höhle, M. (2016). Monitoring count time series in R: Aberration detection in public health surveillance. *Journal of Statistical Software*, 70(10).
- Apache Flink documentation. Event time and watermarks.
