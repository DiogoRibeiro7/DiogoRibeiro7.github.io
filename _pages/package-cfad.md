---
layout: page
title: "cfad"
permalink: /packages/cfad/
author_profile: true
seo_title: "cfad Python Package"
seo_description: "Project page for cfad, a characteristic-function anomaly detector for changes in the distributional shape of financial returns."
---

`cfad` (Characteristic Function Anomaly Detector) watches for changes in the *shape* of a return distribution rather than in its level or volatility. For each rolling window it compares the empirical characteristic function with the Gaussian characteristic function fitted to that window's mean and variance; the normalised discrepancy over a band of frequencies is the anomaly score. Because location and scale are fitted inside every window, the score responds to higher-order changes such as heavier tails or skewness. A two-sided Page-CUSUM turns the score sequence into sequential alarms.

The package is research software. Its README is explicit that the finite-sample empirical characteristic function is entire, so no branch cuts or poles are inferred from empirical contour residues; complex contour integration remains available only as a diagnostic for parametric characteristic functions.

## Install

```bash
pip install cfad
```

## Project Links

- **PyPI:** [cfad](https://pypi.org/project/cfad/)
- **Documentation:** [diogoribeiro7.github.io/cfad](https://diogoribeiro7.github.io/cfad/)
- **Source:** [github.com/DiogoRibeiro7/cfad](https://github.com/DiogoRibeiro7/cfad)
- **Issues:** [github.com/DiogoRibeiro7/cfad/issues](https://github.com/DiogoRibeiro7/cfad/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/cfad/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.2.2`
- **Requires Python:** `>=3.10`
- **Status:** alpha

## Where It Fits

Use it on financial or other heavy-tailed series when a volatility model already explains the second moment and the question is whether the tails or the asymmetry have moved. It complements `heavytails`, which supplies the distributions and tail estimators, and `anomalybench`, which compares general-purpose detectors.
