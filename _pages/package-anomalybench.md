---
layout: page
title: "anomalybench"
permalink: /packages/anomalybench/
author_profile: true
seo_title: "anomalybench Python Package"
seo_description: "Project page for anomalybench, a benchmarking suite with anomaly detection algorithms, dataset loaders and a command line for comparing detectors."
---

`anomalybench` bundles anomaly detection algorithms, loaders for benchmark datasets and a command-line runner so that detectors can be compared on the same data under the same protocol. The base install covers classical detectors, ARIMA forecasting, graph detectors and the benchmark workflows; optional extras add deep-learning detectors (PyTorch and TensorFlow), streaming detectors (River) and Prophet-based forecasting. The bundled datasets span tabular, image, time-series and graph data.

## Install

```bash
pip install anomalybench
```

The project targets Python 3.12 only for now; 3.13 is blocked until the runtime and dependency stack are validated there. Extras: `deep`, `streaming`, `forecasting` and `all-detectors`.

## Project Links

- **PyPI:** [anomalybench](https://pypi.org/project/anomalybench/)
- **Source:** [github.com/DiogoRibeiro7/anomalybench](https://github.com/DiogoRibeiro7/anomalybench)
- **Issues:** [github.com/DiogoRibeiro7/anomalybench/issues](https://github.com/DiogoRibeiro7/anomalybench/issues)

## Package Metadata

- **Current release:** `0.6.1`
- **Requires Python:** `>=3.12,<3.13`
- **Status:** alpha

## Where It Fits

Use it when the question is "which detector, on which kind of data, under which budget" rather than "how do I run one detector". It pairs with the anomaly-detection articles on this site, which lean on the same distinction between point, contextual and collective anomalies, and with `cfad` for the distribution-shape end of the problem.
