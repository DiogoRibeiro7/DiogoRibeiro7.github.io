---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-12-01'
excerpt: "Data science and engineering both contain uncertainty. The useful management distinction is between discovery risk, implementation risk, and operational risk rather than a false opposition between experimental science and predictable engineering."
header:
  image: /assets/images/headers/photo-data-science-theater.jpg
  og_image: /assets/images/headers/photo-data-science-theater.jpg
  overlay_image: /assets/images/headers/photo-data-science-theater.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-theater.jpg
  twitter_image: /assets/images/headers/photo-data-science-theater.jpg
keywords:
- Data science management
- Experimentation
- Engineering
- Model development
- ML delivery
permalink: '/data-science/managing_data_science/'
redirect_from:
- '/data science/managing_data_science/'
seo_description: "How to manage data science work by separating discovery risk, model risk, implementation risk, and operational delivery."
seo_title: "Managing Data Science Under Uncertainty"
seo_type: article
summary: "Data science should not be managed as an endless research project, nor should engineering be treated as fully predictable. Good delivery separates exploration from production and defines explicit evidence gates."
tags:
- Data Science
- MLOps
title: "Managing Data Science Under Uncertainty"
---

Data science projects are often described as fundamentally different from engineering projects because data science is "unknown" while engineering is "known." That contrast is too sharp. Software engineering contains discovery, integration risk, performance uncertainty, and changing requirements. Data science also contains routine engineering once the statistical problem is understood.

The useful distinction is between different kinds of uncertainty.

## Discovery risk

Discovery risk asks whether a useful signal exists at all.

Examples include:

- whether the target can be measured reliably
- whether predictors available at decision time contain enough information
- whether the effect size is operationally meaningful
- whether the model transports across sites or future periods

This phase should be managed as an experiment. The goal is evidence, not production code.

## Statistical risk

Even when signal exists, model performance is uncertain. Validation estimates have sampling error. Class prevalence can shift. Calibration can fail. Labels can be noisy. Performance may differ by subgroup.

A project should therefore define success criteria before extensive model tuning. For example:

$$
\text{deploy only if }
\operatorname{AUCPR}>c_1,
\quad
\text{calibration error}<c_2,
$$

or, more usefully, if expected operational utility exceeds a baseline policy.

The point is to stop treating "better model metric" as an open-ended research objective.

## Engineering risk

Once the model is scientifically defensible, familiar engineering concerns dominate:

- latency
- reliability
- data contracts
- observability
- versioning
- security
- deployment
- rollback
- cost

At that stage, milestones can be estimated much more like ordinary software delivery.

## Operational risk

A technically correct model can still fail in use. Predictions may arrive too late, users may ignore them, interventions may be capacity constrained, or model outputs may alter the future data-generating process.

The deployed unit is therefore not the model. It is the decision system.

## Separate discovery from production

A useful project structure has explicit phases.

### 1. Problem definition

Specify the decision, population, target, prediction horizon, available information, and error costs.

### 2. Feasibility

Build a simple baseline and test whether the data can support the target.

A baseline might be logistic regression, a persistence forecast, a historical average, or an existing business rule. If a complex model cannot beat a relevant baseline, more tuning may not be justified.

### 3. Validation

Use a split that reproduces deployment. Temporal, grouped, site-based, or external validation may be required.

### 4. Productionization

Only after evidence is strong enough should the team invest heavily in APIs, feature stores, monitoring, orchestration, and scaling.

### 5. Post-deployment monitoring

Monitor both inputs and outcomes. Feature drift without performance deterioration may be harmless; stable input distributions do not guarantee stable calibration.

## Time-box exploration, not conclusions

Uncertainty does not imply that timelines are meaningless. Exploration can be time-boxed.

For example, a team might allocate three weeks to determine whether a signal exceeds a predefined baseline. At the end, the decision can be:

- proceed
- revise the target
- collect better data
- stop

The deadline applies to the experiment, not to forcing a positive result.

## Manage hypotheses, not model catalogs

Testing ten algorithms is not the same as scientific exploration. A productive experiment changes one assumption at a time.

Examples:

- Does recency carry more signal than long-term history?
- Does adding sensor frequency information improve discrimination?
- Is site variation large enough to require hierarchical structure?
- Does calibration deteriorate after a device replacement?

This makes progress cumulative because failed experiments still eliminate hypotheses.

## Track uncertainty explicitly

A roadmap can distinguish between known implementation tasks and uncertain research tasks.

| Work item | Type | Evidence required |
|---|---|---|
| define target | design | stakeholder agreement |
| build baseline | discovery | reproducible validation |
| compare feature family | experiment | out-of-sample improvement |
| production API | engineering | latency/reliability tests |
| monitor calibration | operations | observed outcomes |

This is more informative than pretending every ticket has the same uncertainty.

## Avoid premature infrastructure

One common failure mode is building production-grade infrastructure around an unvalidated idea. Another is the reverse: leaving successful models in notebooks with no reproducible pipeline.

The right sequence is progressive hardening. Prototype cheaply, validate rigorously, then engineer the parts that have earned permanence.

## Research debt and technical debt

Data science systems accumulate more than software technical debt. They also accumulate research debt:

- undocumented target definitions
- unclear label provenance
- unexplained feature choices
- validation splits no one can reproduce
- unrecorded negative experiments
- thresholds selected after looking at the test set

These make future model changes scientifically difficult even if the codebase is clean.

## Conclusion

Data science should not be managed as mysterious work that cannot be planned. Nor should it be forced into a production schedule before feasibility is established.

The practical model is staged uncertainty reduction:

$$
\text{problem}
\rightarrow
\text{feasibility}
\rightarrow
\text{validation}
\rightarrow
\text{production}
\rightarrow
\text{monitoring}.
$$

Each stage has different risks, evidence, and deliverables. Managing those differences explicitly produces better science and more predictable engineering.

## References

- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems.
- Breck, E., et al. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.
- Amershi, S., et al. (2019). Software Engineering for Machine Learning: A Case Study.
