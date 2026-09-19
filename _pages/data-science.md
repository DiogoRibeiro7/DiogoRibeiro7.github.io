---
layout: page
title: "Data Science"
permalink: /data-science/
author_profile: true
seo_title: "Data Science Articles"
seo_description: "Data science articles on exploratory analysis, statistical modelling in practice, evaluating decisions and interventions, models in production, and applied work in healthcare and predictive maintenance."
---

This hub is for the work that sits between a question and a model: exploring a dataset, choosing what to measure, deciding whether an intervention worked, and keeping an analysis honest once it runs in production. It is the second-largest subject on the site, and many of its articles are applied case studies.

## Start here

- [Exploratory Data Analysis: A Beginner's Guide](/data-science/exploratory_data_analysis_intro/)
- [Understanding Statistical Significance in Data Analysis](/data-science/understanding_statistical_significance_data_analysis/)
- [Exploring Kernel Density Estimation](/data-science/exploring_kernel_density_estimation_powerful_tool_data_analysis/)
- [Synthetic Control: Evaluating an Intervention on One Unit](/data-science/synthetic_control_single_unit_interventions/)

## Analysis in Python

Exploratory analysis with pandas, density estimation, entropy, outlier and anomaly detection, and worked examples in the scientific Python stack.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Python" limit=8 %}

## Statistical modelling in practice

Generalised linear models, latent class analysis, count models, missing data in clinical research, and choosing between tests.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Statistical Modeling" limit=8 %}

## Decisions and interventions

Did the change work, and for whom? Synthetic control, uplift modelling and counterfactual evaluation of decision policies.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Causal Inference" limit=6 %}

## Models and data in production

Silent data-quality failures, drift detection, validating anomaly detectors, monitoring with wearables and IoT sensors, and what to check before deployment.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Model Monitoring" limit=8 %}

## Healthcare and ageing

Readmission risk, fall prediction, remote monitoring, chronic disease, clinical text and the ethics of AI in elderly care.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Healthcare" limit=6 %}

## Predictive maintenance

Measuring whether a maintenance programme pays for itself, dashboards, cloud and edge analytics, maintenance text, and the effect on operations.

{% include_relative _partials/hub-post-list.html category="Data Science" tag="Predictive Maintenance" limit=6 %}

## Latest in Data Science

{% include_relative _partials/hub-post-list.html category="Data Science" limit=9 %}

Every article in this subject is listed under [Data Science in the category index]({{ '/categories/' | relative_url }}#data-science). For neighbouring subjects see [Statistics](/statistics/), [Machine Learning](/machine-learning/) and [Research Methods](/research-methods/).
