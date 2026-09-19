---
layout: page
title: "subspaceknn"
permalink: /packages/subspaceknn/
author_profile: true
seo_title: "subspaceknn Python Package"
seo_description: "Project page for subspaceknn, interpretable k-nearest-neighbour classification by complementary selection of low-dimensional feature subspaces."
---

`subspaceknn` fits a k-nearest-neighbour model on every small subset of features, one, two or three at a time, and builds a small ensemble of them that votes on new samples. Every member lives in a space that can be drawn, so a prediction is explained by a handful of pictures: which subspaces agreed, which dissented, and where the sample sits among its neighbours in each.

What sets the method apart is how the ensemble is chosen. Keeping the subspaces that score best on their own tends to produce near-copies of each other, so **complementary selection** adds subspaces one vote at a time, each time the one that most improves the ensemble's out-of-fold predictions. Exact leave-one-out predictions from a single neighbour query per subspace make it cheap to consider up to a thousand candidates. On fourteen benchmark datasets this gains two to three points of macro-F1 over ranking subspaces individually, and with subspaces of up to three features it beats plain kNN in the full feature space on average. The idea of an ensemble of drawable kNN models comes from Brett Kennedy's interpretable kNN (ikNN), whose ranked scheme remains available as `selection="ranked"`.

`SubspaceKNNClassifier` follows the scikit-learn contract, so it works inside `Pipeline`, `GridSearchCV` and `cross_val_score`; `explain` returns the per-subspace votes behind a prediction and `plot_subspaces` draws them.

## Install

```bash
pip install subspaceknn
```

The `plot` extra adds matplotlib for `plot_subspaces`. The package needs scikit-learn 1.6 or newer.

## Project Links

- **PyPI:** [subspaceknn](https://pypi.org/project/subspaceknn/)
- **Documentation:** [repository documentation](https://github.com/DiogoRibeiro7/subspaceknn/tree/main/docs)
- **Method:** [method documentation](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/docs/method.md)
- **Benchmark:** [benchmark documentation](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/docs/benchmark.md)
- **Source:** [github.com/DiogoRibeiro7/subspaceknn](https://github.com/DiogoRibeiro7/subspaceknn)
- **Issues:** [github.com/DiogoRibeiro7/subspaceknn/issues](https://github.com/DiogoRibeiro7/subspaceknn/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.2.0`
- **Requires Python:** `>=3.10`
- **License:** MIT
- **Status:** alpha

## Where It Fits

Use it for classification problems where the explanation has to be something a person can look at, and where a point or two of accuracy is a fair price for that. Features must be numeric and scaled, candidate subspaces grow combinatorially with the feature count, and the voting weights are vote shares rather than calibrated probabilities, so treat `predict_proba` as a ranking unless you calibrate it.
