---
layout: single
title: "pinn-rk"
permalink: /packages/pinn-rk/
author_profile: true
seo_title: "pinn-rk Python Package"
seo_description: "Project page for pinn-rk, a PyTorch package for Runge-Kutta physics-informed neural networks."
---

`pinn-rk` provides Runge-Kutta physics-informed neural networks with time-discrete losses in PyTorch. It focuses on Gauss, Radau and Lobatto style formulations for scientific machine learning problems where temporal discretization is part of the modelling choice.

## Install

```bash
pip install pinn-rk
```

## Project Links

- **PyPI:** [pinn-rk](https://pypi.org/project/pinn-rk/)
- **Documentation:** [diogoribeiro7.github.io/pinn-rk](https://diogoribeiro7.github.io/pinn-rk)
- **Source:** [github.com/DiogoRibeiro7/pinn-rk](https://github.com/DiogoRibeiro7/pinn-rk)
- **Issues:** [github.com/DiogoRibeiro7/pinn-rk/issues](https://github.com/DiogoRibeiro7/pinn-rk/issues)
- **Discussions:** [github.com/DiogoRibeiro7/pinn-rk/discussions](https://github.com/DiogoRibeiro7/pinn-rk/discussions)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/pinn-rk/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.6.0`
- **Requires Python:** `>=3.10,<3.13`
- **License:** MIT

## Where It Fits

Use this package when the learning problem depends on time-stepping choices, numerical integration structure or a physics-informed loss that should be explicit and inspectable.
