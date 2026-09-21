---
permalink: '/programming/data_engineering_introduction/'
author_profile: false
categories:
- Programming
classes: wide
date: '2023-12-30'
excerpt: "Data engineering is the design of reliable data systems: ingestion, storage, contracts, transformations, orchestration, lineage, quality, and serving."
header:
  image: /assets/images/headers/photo-factory.jpg
  og_image: /assets/images/headers/photo-factory.jpg
  overlay_image: /assets/images/headers/photo-factory.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-factory.jpg
  twitter_image: /assets/images/headers/photo-factory.jpg
keywords:
- Data engineering
- Data contracts
- ETL
- ELT
- Streaming
- Data quality
- Lineage
redirect_from:
- '/data engineering/data_engineering_introduction/'
seo_description: "A modern introduction to data engineering covering contracts, batch and streaming semantics, ETL/ELT, orchestration, idempotence, lineage, quality, and serving."
seo_title: "Data Engineering: Reliable Data Systems"
seo_type: article
tags:
- Data Engineering
- MLOps
title: "Data Engineering: Reliable Data Systems"
---

Data engineering is the discipline of building systems that make data trustworthy and usable over time. The important properties are not merely volume and speed. A good system preserves meaning, handles failure, records provenance, and gives downstream users stable contracts.

## The data product begins with a contract

A table or event stream should have a defined contract:

- schema
- field semantics
- units
- allowed nullability
- primary or business keys
- event-time meaning
- update semantics
- retention
- ownership

Without these, pipelines can be technically successful while silently changing the meaning of downstream analysis.

## ETL and ELT are execution choices

ETL transforms data before loading it into the analytical destination.

ELT loads source data first and performs transformations in the destination system.

The choice is not simply "ETL for structured data, ELT for big data." It depends on governance, compute economics, privacy, source limitations, latency, and where transformation logic is easiest to test and maintain.

Modern warehouses and lakehouse systems often make ELT attractive because storage is cheap and compute is elastic. Sensitive data may still require transformation or filtering before landing.

## Batch semantics

A batch pipeline should ideally be idempotent: rerunning the same logical input should not duplicate or corrupt results.

If a daily partition is rebuilt, the operation should have predictable overwrite or merge semantics.

A pipeline should also distinguish:

$$
\text{processing date}
\neq
\text{event date}.
$$

Backfills, late-arriving records, and corrections make that distinction unavoidable.

## Streaming semantics

Streaming systems introduce event time, processing time, lateness, ordering, and replay.

An event generated at time $t_e$ may be processed later at $t_p$.

If aggregation is based on event time, late records may update previously emitted windows.

Watermarks provide a policy for how long the system waits for late data. They are not a statement that later events are impossible.

## Exactly-once is an end-to-end property

Messaging systems often advertise exactly-once features, but business-level exactly-once behavior depends on the complete pipeline.

If a consumer processes an event twice and writes twice to an external database without idempotent keys, the business result is duplicated even if the broker has strong delivery guarantees.

Deduplication keys, transactional boundaries, and replay behavior must be designed across components.

## Data quality should be tested as code

Useful quality checks include:

- schema conformance
- uniqueness
- accepted ranges
- referential integrity
- freshness
- volume anomalies
- distribution shifts
- reconciliation with source totals

A passing pipeline only proves that code ran. It does not prove the data are correct.

## Lineage

Lineage records how a downstream dataset depends on upstream sources and transformations.

This matters when:

- a source column changes
- a metric definition is revised
- an error must be traced
- a model must be reproduced
- regulated outputs require provenance

Lineage is most useful when it is generated from actual pipeline metadata rather than maintained manually in a diagram that drifts from reality.

## Orchestration

Workflow orchestrators manage dependencies, retries, scheduling, and observability.

A DAG such as

$$
A\rightarrow B\rightarrow C
$$

states dependency, not necessarily data correctness.

Retries should be safe. A task that sends emails, charges customers, or appends rows may need explicit idempotence controls before automatic retry is enabled.

## Storage design

Warehouses, object stores, transactional databases, and streaming logs solve different problems.

Columnar analytical storage is efficient for scans and aggregation. Row-oriented transactional stores are optimized for point reads and updates. Object storage is durable and inexpensive but often relies on table formats and metadata layers for transactional semantics.

The architecture should follow access patterns and consistency requirements rather than a fashion label such as "data lake."

## Data modeling

A good analytical model makes grain explicit.

For example, a fact table may have one row per

$$
\text{order line}
$$

rather than one row per customer or order.

Without a declared grain, joins can multiply rows and silently inflate metrics.

Dimensional modeling, normalized schemas, data vaults, and wide analytical tables each have contexts where they are useful.

## Serving machine learning

ML pipelines introduce additional contracts:

- training-serving feature consistency
- point-in-time correctness
- label availability
- feature freshness
- versioned transformations

Feature leakage often originates in data engineering rather than model code.

A feature computed from data that arrived after the prediction timestamp is invalid even if the join succeeds perfectly.

## Cost is an engineering constraint

Cloud systems make it easy to scale inefficient queries.

Partition pruning, clustering, incremental models, compaction, caching, and workload isolation are not only performance optimizations. They determine whether a platform remains economically sustainable.

Cost observability should be part of platform monitoring.

## Conclusion

Data engineering is not a catalogue of tools. Kafka, Spark, Flink, Airflow, dbt, warehouses, and lakehouse formats are implementation choices.

The core discipline is preserving trustworthy semantics through time:

$$
\text{source}
\rightarrow
\text{contract}
\rightarrow
\text{transformation}
\rightarrow
\text{quality}
\rightarrow
\text{lineage}
\rightarrow
\text{serving}.
$$

A reliable data platform makes failures visible and meaning stable.

## References

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*.
- Reis, J., & Housley, M. (2022). *Fundamentals of Data Engineering*.
