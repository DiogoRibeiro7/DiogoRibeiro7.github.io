---
layout: page
title: "oversampleqa"
permalink: /packages/oversampleqa/
author_profile: true
seo_title: "oversampleqa Python Package"
seo_description: "Project page for oversampleqa, a diagnostic toolkit that validates, audits and benchmarks oversampling methods for imbalanced classification."
---

`oversampleqa` answers a question that SMOTE-style oversamplers never answer for themselves: do the synthetic minority points actually look like the minority class, or have they drifted into majority territory? It hides part of the majority class, scores every synthetic sample by its nearest-neighbour distance to the hidden majority and to the real minority under a chosen metric, and reports the share of synthetic points that sit closer to the majority. That hidden-majority error rate works for binary and multiclass problems, where it becomes a confusion-style matrix across classes.

## Install

```bash
pip install oversampleqa
```

The optional performance helpers come with `pip install "oversampleqa[performance]"`.

## Project Links

- **PyPI:** [oversampleqa](https://pypi.org/project/oversampleqa/)
- **Documentation:** [diogoribeiro7.github.io/OversampleQA](https://diogoribeiro7.github.io/OversampleQA/)
- **Source:** [github.com/diogoribeiro7/OversampleQA](https://github.com/diogoribeiro7/OversampleQA)
- **Issues:** [github.com/diogoribeiro7/OversampleQA/issues](https://github.com/diogoribeiro7/OversampleQA/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/diogoribeiro7/OversampleQA/blob/main/CHANGELOG.md)
- **Citation:** [CITATION.cff](https://github.com/diogoribeiro7/OversampleQA/blob/main/CITATION.cff)

## Package Metadata

- **Current release:** `0.8.0`
- **Requires Python:** `>=3.10`
- **License:** MIT

## Where It Fits

Reach for it when an imbalanced-classification pipeline uses oversampling and you need evidence, not faith, that the generated points are minority-like: comparing oversamplers on a dataset, choosing a distance metric (Hassanat, Euclidean, Mahalanobis and others are built in), or running the benchmark suite and CLI across several datasets with exportable results. A plugin system takes custom metrics and validators.
