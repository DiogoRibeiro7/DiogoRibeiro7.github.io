---
layout: single
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
seo_description: "Open-source Python and R packages by Diogo Ribeiro, including PyPI packages for survival simulation, physics-informed neural networks, QCA, time-series representations and WiFi activity recognition."
---

## Open Source Projects & Packages

This page collects software projects that are more useful as technical assets than as generic blog posts: installable packages, documentation, source repositories, examples and research tooling.

The Python projects below are published on PyPI under [DiogoRibeiro7](https://pypi.org/user/DiogoRibeiro7/). They are grouped here by purpose so readers can see what each package is for before jumping into package documentation.

## PyPI Packages

| Package | Current release | Install | Purpose |
| --- | --- | --- | --- |
| [wifi-activity-recognition](https://pypi.org/project/wifi-activity-recognition/) | 0.2.0 | `pip install wifi-activity-recognition` | Human activity recognition using WiFi CSI and computer vision workflows. |
| [tscv-vision](https://pypi.org/project/tscv-vision/) | 0.4.0 | `pip install tscv-vision` | NumPy-first structured representation engineering for time series. |
| [pinnlab](https://pypi.org/project/pinnlab/) | 0.6.1 | `pip install pinnlab` | Physics-informed neural network implementations. |
| [setqca](https://pypi.org/project/setqca/) | 0.2.0 | `pip install setqca` | Native Python toolkit for crisp-set and fuzzy-set qualitative comparative analysis. |
| [gen-surv](https://pypi.org/project/gen-surv/) | 2.0.0 | `pip install gen-surv` | Survival-data simulation and visualization for statistical research and benchmarking. |
| [pinn-rk](https://pypi.org/project/pinn-rk/) | 0.6.0 | `pip install pinn-rk` | Runge-Kutta physics-informed neural networks with time-discrete losses in PyTorch. |

### Scientific Machine Learning

#### [pinnlab](https://pypi.org/project/pinnlab/)

Physics-informed neural network implementations for experiments where differential equations, boundary conditions and neural approximators need to live in the same workflow.

- **PyPI:** [pinnlab](https://pypi.org/project/pinnlab/)
- **Documentation:** [DiogoRibeiro7.github.io/pinn](https://DiogoRibeiro7.github.io/pinn)
- **Source:** [github.com/DiogoRibeiro7/pinn](https://github.com/DiogoRibeiro7/pinn)
- **Requires Python:** `>=3.10`

```bash
pip install pinnlab
```

#### [pinn-rk](https://pypi.org/project/pinn-rk/)

Runge-Kutta PINNs for time-discrete physics-informed learning, including Gauss, Radau and Lobatto style losses in PyTorch.

- **PyPI:** [pinn-rk](https://pypi.org/project/pinn-rk/)
- **Documentation:** [diogoribeiro7.github.io/pinn-rk](https://diogoribeiro7.github.io/pinn-rk)
- **Source:** [github.com/DiogoRibeiro7/pinn-rk](https://github.com/DiogoRibeiro7/pinn-rk)
- **Requires Python:** `>=3.10,<3.13`

```bash
pip install pinn-rk
```

### Statistics, Survival Analysis and Research Methods

#### [gen-surv](/packages/gensurvpy/)

A Python package for simulating survival data and producing visualizations under Cox proportional hazards, accelerated failure time, multi-state, time-dependent covariate, hidden Markov, competing risks, mixture cure and piecewise exponential models.

- **PyPI:** [gen-surv](https://pypi.org/project/gen-surv/)
- **Documentation on this site:** [genSurvPy](/packages/gensurvpy/)
- **External documentation:** [gensurvpy.readthedocs.io](https://gensurvpy.readthedocs.io/en/latest/)
- **Source:** [github.com/DiogoRibeiro7/genSurvPy](https://github.com/DiogoRibeiro7/genSurvPy)
- **Requires Python:** `>=3.11,<3.14`

```bash
pip install gen-surv
```

#### [setqca](https://pypi.org/project/setqca/)

A native Python toolkit for crisp-set and fuzzy-set Qualitative Comparative Analysis. It belongs with the research-methods part of the site because it helps encode configurational arguments, not just fit predictive models.

- **PyPI:** [setqca](https://pypi.org/project/setqca/)
- **Documentation:** [diogoribeiro7.github.io/setqca-python](https://diogoribeiro7.github.io/setqca-python/)
- **Source:** [github.com/DiogoRibeiro7/setqca-python](https://github.com/DiogoRibeiro7/setqca-python)
- **Requires Python:** `>=3.11,<4.0`

```bash
pip install setqca
```

### Time Series, Signals and Activity Recognition

#### [tscv-vision](https://pypi.org/project/tscv-vision/)

Structured representation engineering for time series with a NumPy-first API. This package is a better fit for reusable transformations and experiments than one-off notebook code.

- **PyPI:** [tscv-vision](https://pypi.org/project/tscv-vision/)
- **Documentation:** [GitHub README](https://github.com/DiogoRibeiro7/tscv-vision#readme)
- **Source:** [github.com/DiogoRibeiro7/tscv-vision](https://github.com/DiogoRibeiro7/tscv-vision)
- **Requires Python:** `>=3.10,<3.13`

```bash
pip install tscv-vision
```

#### [wifi-activity-recognition](https://pypi.org/project/wifi-activity-recognition/)

A package for human activity recognition using WiFi channel-state information and computer vision workflows. It sits at the intersection of sensing, signal processing and applied machine learning.

- **PyPI:** [wifi-activity-recognition](https://pypi.org/project/wifi-activity-recognition/)
- **Documentation:** [wifi-activity-recognition.readthedocs.io](https://wifi-activity-recognition.readthedocs.io/)
- **Source:** [github.com/diogoribeiro7/wifi-csi-activity-recognition](https://github.com/diogoribeiro7/wifi-csi-activity-recognition)
- **Requires Python:** `>=3.10`

```bash
pip install wifi-activity-recognition
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

