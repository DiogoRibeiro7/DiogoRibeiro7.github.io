---
layout: page
title: "pygeostats"
permalink: /packages/pygeostats/
author_profile: true
seo_title: "pygeostats Python Package"
seo_description: "Project page for pygeostats, geostatistics for Python with a Rust-accelerated core: variograms, kriging, point patterns and spatial autocorrelation."
---

`pygeostats` covers the geostatistical workflow in one package: estimate an empirical variogram, fit a model, krige with prediction variance and cross-validate the result, behind a `fit` / `predict` API that accepts NumPy arrays, pandas DataFrames and GeoPandas GeoDataFrames. Distances, empirical variograms, model fitting and kriging run in a compiled Rust core built with PyO3 and maturin, and one stable-ABI wheel per platform covers Python 3.11 to 3.14, so installing it needs no Rust toolchain.

Beyond kriging (ordinary, simple, universal and anisotropic), it includes point-pattern statistics (Ripley's K and L, G, F and pair correlation functions, kernel density, Gi* hot spots, Poisson, Cox and marked process simulation), spatial autocorrelation (global and local Moran's I and Geary's C, Getis-Ord statistics, several weight schemes) and validation by leave-one-out, spatial k-fold and block cross-validation. When the data does not constrain a variogram model, `fit()` reports it through `converged_` and `warnings_` instead of returning a range as though it were reliable.

## Install

```bash
pip install pygeostats
```

While no stable release exists, pip installs the pre-release. Extras: `plotting` (plotly), `progress` (tqdm) and `approx` (annoy, for approximate neighbour search). Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64); elsewhere pip builds from source and needs a Rust toolchain.

## Project Links

- **PyPI:** [pygeostats](https://pypi.org/project/pygeostats/)
- **Documentation:** [repository documentation](https://github.com/DiogoRibeiro7/pygeostats/tree/main/docs)
- **Known limitations:** [project documentation](https://github.com/DiogoRibeiro7/pygeostats/blob/main/docs/index.md)
- **Source:** [github.com/DiogoRibeiro7/pygeostats](https://github.com/DiogoRibeiro7/pygeostats)
- **Issues:** [github.com/DiogoRibeiro7/pygeostats/issues](https://github.com/DiogoRibeiro7/pygeostats/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CHANGELOG.md)

## Package Metadata

- **Current release:** `0.1.0a2` (pre-release)
- **Requires Python:** `>=3.11`
- **License:** MIT
- **Status:** alpha

## Where It Fits

Use it for spatial interpolation and spatial pattern analysis when you want the variogram, the fit diagnostics and the kriging variance in front of you. It is alpha software: fitting an anisotropic model from directional variograms is not implemented, anisotropy estimates are rough, and Matérn models cannot be fitted yet, so read the known limitations before relying on anisotropy analysis. The project was previously called `pyspatialstats` and was renamed because that name belongs to an unrelated package on PyPI.
