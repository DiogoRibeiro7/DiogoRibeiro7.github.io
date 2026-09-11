---
layout: page
title: Projects & Packages
permalink: /packages/
author_profile: true
header:
  image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_1.jpg
  teaser: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  twitter_image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_1.jpg
seo_type: article
seo_title: "Open Source Projects and Packages"
seo_description: "Open-source Python and R packages for survival simulation, heavy-tailed distributions, design of experiments, physics-informed neural networks, QCA, anomaly detection, imbalanced-learning diagnostics and time-series representations."
---

## Open Source Projects & Packages

This page collects software projects that are more useful as technical assets than as generic blog posts: installable packages, documentation, source repositories, examples and research tooling.

The Python projects below are published on PyPI under [DiogoRibeiro7](https://pypi.org/user/DiogoRibeiro7/). They are grouped here by purpose so readers can see what each package is for before jumping into package documentation.

## PyPI Packages

| Package | Current release | Install | Purpose |
| --- | --- | --- | --- |
| [wifi-activity-recognition](/packages/wifi-activity-recognition/) | 0.2.0 | `pip install wifi-activity-recognition` | Human activity recognition using WiFi CSI and computer vision workflows. |
| [tscv-vision](/packages/tscv-vision/) | 0.4.0 | `pip install tscv-vision` | NumPy-first structured representation engineering for time series. |
| [pinnlab](/packages/pinnlab/) | 0.6.1 | `pip install pinnlab` | Physics-informed neural network implementations. |
| [setqca](/packages/setqca/) | 0.2.0 | `pip install setqca` | Native Python toolkit for crisp-set and fuzzy-set qualitative comparative analysis. |
| [gen-surv](/packages/gensurvpy/) | 3.1.2 | `pip install gen-surv` | Survival-data simulation with a known truth: twelve models from proportional hazards to multi-state processes. |
| [pinn-rk](/packages/pinn-rk/) | 0.6.0 | `pip install pinn-rk` | Runge-Kutta physics-informed neural networks with time-discrete losses in PyTorch. |
| [heavytails](/packages/heavytails/) | 0.6.3 | `pip install heavytails` | Heavy-tailed distributions, tail index estimators and extreme value diagnostics, vectorised over NumPy. |
| [industrialstats](/packages/industrialstats/) | 0.2.0 | `pip install industrialstats` | Industrial statistics and design of experiments: design generators, ANOVA, diagnostics, power and response surfaces. |
| [sensor-modeling](/packages/sensor-modeling/) | 0.2.0 | `pip install sensor-modeling` | Interpretable, probabilistic, privacy-preserving analysis of behavioural and ambient sensor data. |
| [cfad](/packages/cfad/) | 0.2.2 | `pip install cfad` | Characteristic-function detection of distributional-shape changes in financial time series. |
| [oversampleqa](/packages/oversampleqa/) | 0.8.0 | `pip install oversampleqa` | Validation, audit and benchmarking of oversampling methods for imbalanced classification. |
| [anomalybench](/packages/anomalybench/) | 0.6.1 | `pip install anomalybench` | Benchmarking suite for anomaly detection algorithms with dataset loaders and a CLI. |
| [DataExcept](/packages/dataexcept/) | 1.6.0 | `pip install DataExcept` | Structured, hierarchical exception classes for data science and machine learning pipelines. |

### Scientific Machine Learning

#### [pinnlab](/packages/pinnlab/)

Physics-informed neural network implementations for experiments where differential equations, boundary conditions and neural approximators need to live in the same workflow.

- **Project page:** [pinnlab](/packages/pinnlab/)
- **PyPI:** [pinnlab](https://pypi.org/project/pinnlab/)
- **Source:** [github.com/DiogoRibeiro7/pinn](https://github.com/DiogoRibeiro7/pinn)
- **Requires Python:** `>=3.10`

```bash
pip install pinnlab
```

#### [pinn-rk](/packages/pinn-rk/)

Runge-Kutta PINNs for time-discrete physics-informed learning, including Gauss, Radau and Lobatto style losses in PyTorch.

- **Project page:** [pinn-rk](/packages/pinn-rk/)
- **PyPI:** [pinn-rk](https://pypi.org/project/pinn-rk/)
- **Source:** [github.com/DiogoRibeiro7/pinn-rk](https://github.com/DiogoRibeiro7/pinn-rk)
- **Requires Python:** `>=3.10,<3.13`

```bash
pip install pinn-rk
```

### Statistics, Survival Analysis and Research Methods

#### [gen-surv](/packages/gensurvpy/)

Simulate survival data with a known truth. Version 3 generates time-to-event datasets from twelve models spanning proportional hazards, accelerated failure time, competing risks, cure fractions, piecewise hazards, recurrent events and two illness-death processes, so an estimator can be tested against parameters you chose yourself. It started as a Python port of the R package genSurv and now goes well past the original's four models; it ships `py.typed`, and only the two scikit-survival conversion helpers need an optional extra.

- **PyPI:** [gen-surv](https://pypi.org/project/gen-surv/)
- **Documentation on this site:** [genSurvPy](/packages/gensurvpy/)
- **External documentation:** [diogoribeiro7.github.io/genSurvPy](https://diogoribeiro7.github.io/genSurvPy/)
- **Source:** [github.com/DiogoRibeiro7/genSurvPy](https://github.com/DiogoRibeiro7/genSurvPy)
- **Requires Python:** `>=3.11,<3.14`

```bash
pip install gen-surv
```

#### [setqca](/packages/setqca/)

A native Python toolkit for crisp-set and fuzzy-set Qualitative Comparative Analysis. It belongs with the research-methods part of the site because it helps encode configurational arguments, not just fit predictive models.

- **Project page:** [setqca](/packages/setqca/)
- **PyPI:** [setqca](https://pypi.org/project/setqca/)
- **Source:** [github.com/DiogoRibeiro7/setqca-python](https://github.com/DiogoRibeiro7/setqca-python)
- **Requires Python:** `>=3.11,<4.0`

```bash
pip install setqca
```

#### [heavytails](/packages/heavytails/)

Heavy-tailed distributions, tail index estimators and extreme value diagnostics with NumPy-backed vectorised evaluation. Densities, quantiles and samplers are derived from first principles, survival functions are computed directly so they hold far into the tail, and the applied layer covers peaks-over-threshold analysis, return levels, tail-risk measures, copulas and GARCH fitting.

- **Project page:** [heavytails](/packages/heavytails/)
- **PyPI:** [heavytails](https://pypi.org/project/heavytails/)
- **Source:** [github.com/DiogoRibeiro7/heavytails](https://github.com/DiogoRibeiro7/heavytails)
- **Requires Python:** `>=3.10,<3.14`

```bash
pip install heavytails
```

#### [industrialstats](/packages/industrialstats/)

Industrial statistics and design of experiments: reproducible design generators, ANOVA with Type I, II and III sums of squares, effect sizes, multiple comparisons, contrasts, mixed-effects models, diagnostics, power and sample-size calculations and response-surface optimisation, validated against textbook results and reference software. Pre-1.0, with provisional methods labelled as such.

- **Project page:** [industrialstats](/packages/industrialstats/)
- **PyPI:** [industrialstats](https://pypi.org/project/industrialstats/)
- **Source:** [github.com/DiogoRibeiro7/industrialstats](https://github.com/DiogoRibeiro7/industrialstats)
- **Requires Python:** `>=3.11,<3.15`

```bash
pip install industrialstats
```

### Time Series, Signals and Activity Recognition

#### [tscv-vision](/packages/tscv-vision/)

Structured representation engineering for time series with a NumPy-first API. This package is a better fit for reusable transformations and experiments than one-off notebook code.

- **Project page:** [tscv-vision](/packages/tscv-vision/)
- **PyPI:** [tscv-vision](https://pypi.org/project/tscv-vision/)
- **Source:** [github.com/DiogoRibeiro7/tscv-vision](https://github.com/DiogoRibeiro7/tscv-vision)
- **Requires Python:** `>=3.10,<3.13`

```bash
pip install tscv-vision
```

#### [wifi-activity-recognition](/packages/wifi-activity-recognition/)

A package for human activity recognition using WiFi channel-state information and computer vision workflows. It sits at the intersection of sensing, signal processing and applied machine learning.

- **Project page:** [wifi-activity-recognition](/packages/wifi-activity-recognition/)
- **PyPI:** [wifi-activity-recognition](https://pypi.org/project/wifi-activity-recognition/)
- **Source:** [github.com/diogoribeiro7/wifi-csi-activity-recognition](https://github.com/diogoribeiro7/wifi-csi-activity-recognition)
- **Requires Python:** `>=3.10`

```bash
pip install wifi-activity-recognition
```

#### [sensor-modeling](/packages/sensor-modeling/)

A research toolkit for behavioural and ambient sensor data in assisted living, digital health and smart-home studies: an end-to-end pipeline from heterogeneous sensor observations to explained alerts, built on Bernoulli autoregressive models, hidden Markov models, change-point detection and non-homogeneous Poisson processes. Research software, not a medical device.

- **Project page:** [sensor-modeling](/packages/sensor-modeling/)
- **PyPI:** [sensor-modeling](https://pypi.org/project/sensor-modeling/)
- **Source:** [github.com/DiogoRibeiro7/behavioral-sensing-research](https://github.com/DiogoRibeiro7/behavioral-sensing-research)
- **Requires Python:** `>=3.10,<3.13`

```bash
pip install sensor-modeling
```

#### [cfad](/packages/cfad/)

Characteristic-function anomaly detection for financial returns. Each rolling window's empirical characteristic function is compared with the Gaussian one fitted to that window's mean and variance, so the score reacts to tail and skewness changes rather than to level or volatility, and a two-sided Page-CUSUM turns scores into sequential alarms.

- **Project page:** [cfad](/packages/cfad/)
- **PyPI:** [cfad](https://pypi.org/project/cfad/)
- **Source:** [github.com/DiogoRibeiro7/cfad](https://github.com/DiogoRibeiro7/cfad)
- **Requires Python:** `>=3.10`

```bash
pip install cfad
```

### Machine Learning Diagnostics and Engineering

#### [oversampleqa](/packages/oversampleqa/)

A diagnostic toolkit for oversampling in imbalanced classification. It hides part of the majority class and scores each synthetic sample by its nearest-neighbour distance to the hidden majority and to the real minority, so the hidden-majority error rate says how often an oversampler manufactures majority-like points. Benchmarks, a CLI and a plugin system for custom metrics come with it.

- **Project page:** [oversampleqa](/packages/oversampleqa/)
- **PyPI:** [oversampleqa](https://pypi.org/project/oversampleqa/)
- **Source:** [github.com/diogoribeiro7/OversampleQA](https://github.com/diogoribeiro7/OversampleQA)
- **Requires Python:** `>=3.10`

```bash
pip install oversampleqa
```

#### [anomalybench](/packages/anomalybench/)

A benchmarking suite for anomaly detection: detectors, loaders for tabular, image, time-series and graph benchmark datasets, and a command line that compares detectors under one protocol. Optional extras add deep-learning, streaming and Prophet-based detectors. Python 3.12 only for now.

- **Project page:** [anomalybench](/packages/anomalybench/)
- **PyPI:** [anomalybench](https://pypi.org/project/anomalybench/)
- **Source:** [github.com/DiogoRibeiro7/anomalybench](https://github.com/DiogoRibeiro7/anomalybench)
- **Requires Python:** `>=3.12,<3.13`

```bash
pip install anomalybench
```

#### [DataExcept](/packages/dataexcept/)

Structured, hierarchical exception classes for data science, machine learning and data engineering pipelines: over a hundred specific, catchable failure types with context, JSON export against a versioned schema, pickling across process boundaries and logging helpers. The exception layer that `industrialstats` and other packages here standardise on.

- **Project page:** [DataExcept](/packages/dataexcept/)
- **PyPI:** [DataExcept](https://pypi.org/project/DataExcept/)
- **Source:** [github.com/DiogoRibeiro7/DataExcept](https://github.com/DiogoRibeiro7/DataExcept)
- **Requires Python:** `>=3.10,<3.15`

```bash
pip install DataExcept
```

## R Packages

### [myrpackage](/packages/myrpackage/)

A multilingual greeting and farewell package that serves as an example of proper R package structure.

**Features:**

- Multilingual support (English, Spanish, French, Portuguese, German, Italian)
- Comprehensive documentation with roxygen2
- Full test coverage with testthat
- Continuous integration with GitHub Actions
- Proper package structure following R standards

[View Documentation →](/packages/myrpackage/)

--------------------------------------------------------------------------------

### [unconfoundedr](/packages/unconfoundedr/)

Test (un)confoundedness by comparing an effect from an RCT-like dataset to the same estimand from an observational dataset. Includes robust estimators, inference, and transportability tools.

**Features:**

- IPW and AIPW (doubly robust) estimators for the marginal ATE
- Bootstrap confidence intervals and Wald test
- Transport modes: `none`, `rct_to_obs`, and `auto` (KS/energy shift detection)
- Diagnostics for propensity overlap, stabilized weights, trimming, and transport ESS

[View Documentation →](/packages/unconfoundedr/)

--------------------------------------------------------------------------------

For source repositories, issues and development history, visit my [GitHub profile](https://github.com/DiogoRibeiro7).

