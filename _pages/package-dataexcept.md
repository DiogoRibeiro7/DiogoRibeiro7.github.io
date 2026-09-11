---
layout: page
title: "DataExcept"
permalink: /packages/dataexcept/
author_profile: true
seo_title: "DataExcept Python Package"
seo_description: "Project page for DataExcept, a library of structured, hierarchical exception classes for data science, machine learning and data engineering pipelines."
---

`DataExcept` replaces the generic `ValueError` and `RuntimeError` that data pipelines usually raise with a hierarchy of specific, catchable exception types: data loading and validation, missing columns, feature engineering, model training and convergence, inference, and the other operational boundaries of a data science or machine learning system. Each exception carries context (the field, the model, the dataset), exports to JSON against a published, versioned schema so non-Python consumers can validate it, and pickles cleanly so it survives a process boundary.

## Install

```bash
pip install DataExcept
```

## Project Links

- **PyPI:** [DataExcept](https://pypi.org/project/DataExcept/)
- **Documentation:** [diogoribeiro7.github.io/DataExcept](https://diogoribeiro7.github.io/DataExcept/)
- **Source:** [github.com/DiogoRibeiro7/DataExcept](https://github.com/DiogoRibeiro7/DataExcept)
- **Issues:** [github.com/DiogoRibeiro7/DataExcept/issues](https://github.com/DiogoRibeiro7/DataExcept/issues)
- **Changelog:** [GitHub releases](https://github.com/DiogoRibeiro7/DataExcept/releases)

## Package Metadata

- **Current release:** `1.6.0`
- **Requires Python:** `>=3.10,<3.15`

## Where It Fits

Use it in pipelines and libraries where a caller needs to distinguish "the input was wrong" from "the model did not converge" from "inference ran out of memory" without parsing message strings. It is the exception layer that `industrialstats` and other packages on this site standardise on, and it is a good fit for services that log or forward failures to systems written in other languages.
