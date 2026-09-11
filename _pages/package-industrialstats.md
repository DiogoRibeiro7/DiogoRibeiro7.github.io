---
layout: page
title: "industrialstats"
permalink: /packages/industrialstats/
author_profile: true
seo_title: "industrialstats Python Package"
seo_description: "Project page for industrialstats, industrial statistics and design of experiments for Python: design generators, ANOVA, diagnostics, power and response-surface optimisation."
---

`industrialstats` brings design of experiments and industrial statistics to Python with an emphasis on statistical correctness before breadth. It provides reproducible experimental-design generators with seedable randomisation and inspectable design matrices, and an analysis layer with ANOVA (Type I, II and III sums of squares), effect sizes, multiple comparisons, contrasts, mixed-effects models, factorial main-effect and interaction analysis, residual, leverage, influence and assumption diagnostics, power and sample-size calculations, stepwise and hierarchical fitting, response-surface optimisation, and design-efficiency and prediction-variance utilities. Plots cover the design space, effects, diagnostics, response surfaces, contours and prediction variance.

The project is pre-1.0. Methods are validated against textbook results and reference software such as `statsmodels`, and anything provisional is labelled as such. Operational failures are raised through `DataExcept`.

## Install

```bash
pip install industrialstats
```

## Project Links

- **PyPI:** [industrialstats](https://pypi.org/project/industrialstats/)
- **Documentation:** [diogoribeiro7.github.io/industrialstats](https://diogoribeiro7.github.io/industrialstats/)
- **Source:** [github.com/DiogoRibeiro7/industrialstats](https://github.com/DiogoRibeiro7/industrialstats)
- **Issues:** [github.com/DiogoRibeiro7/industrialstats/issues](https://github.com/DiogoRibeiro7/industrialstats/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/industrialstats/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.2.0`
- **Requires Python:** `>=3.11,<3.15`
- **License:** MIT
- **Status:** alpha

## Where It Fits

Use it for designed experiments in manufacturing, engineering and research where the words effect, block, alias, resolution and optimality criterion need their precise meanings, and where you want the design matrix and the sums of squares in front of you rather than behind an abstraction. It pairs with the experimental-design articles under [Research Methods](/research-methods/).
