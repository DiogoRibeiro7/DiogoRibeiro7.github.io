---
layout: page
title: "sensor-modeling"
permalink: /packages/sensor-modeling/
author_profile: true
seo_title: "sensor-modeling Python Package"
seo_description: "Project page for sensor-modeling, a research toolkit for interpretable, probabilistic, privacy-preserving analysis of behavioural and ambient sensor data."
---

`sensor-modeling` is a research toolkit for behavioural sensor streams in ambient assisted living, digital health and smart-home studies. It runs an end-to-end pipeline from heterogeneous sensor observations to explained alerts, on top of a modelling core of Bernoulli autoregressive models, hidden Markov models, change-point detection and non-homogeneous Poisson processes. The methods are chosen for the awkward properties of ambient data: irregular sampling, frequent missingness, binary activations, and the need for models a clinician can read.

It is a research toolkit, not a medical device. Nothing it produces is a diagnosis, and the quantitative results in its documentation come from the bundled simulator rather than validated real-world data.

## Install

```bash
pip install sensor-modeling
```

## Project Links

- **PyPI:** [sensor-modeling](https://pypi.org/project/sensor-modeling/)
- **Documentation:** [sensor-modeling.readthedocs.io](https://sensor-modeling.readthedocs.io)
- **Source:** [github.com/DiogoRibeiro7/behavioral-sensing-research](https://github.com/DiogoRibeiro7/behavioral-sensing-research)
- **Issues:** [github.com/DiogoRibeiro7/behavioral-sensing-research/issues](https://github.com/DiogoRibeiro7/behavioral-sensing-research/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/behavioral-sensing-research/blob/main/CHANGELOG.md)
- **Citation:** [CITATION.cff](https://github.com/DiogoRibeiro7/behavioral-sensing-research/blob/main/CITATION.cff), archived at [doi.org/10.5281/zenodo.17070041](https://doi.org/10.5281/zenodo.17070041)

## Package Metadata

- **Current release:** `0.2.0`
- **Requires Python:** `>=3.10,<3.13`
- **Status:** beta

## Where It Fits

Use it for reproducible analysis of activity and presence sensors where interpretability matters more than raw accuracy: routine modelling, anomaly and change-point detection in daily patterns, and privacy-preserving multimodal fusion. It is the software companion of the behavioural-sensing research listed on the [papers page](/papers/).
