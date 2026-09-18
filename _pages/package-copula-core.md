---
layout: page
title: "copula-core"
permalink: /packages/copula-core/
author_profile: true
seo_title: "copula-core Rust Crate"
seo_description: "Project page for copula-core, a Rust crate for copula modelling, simulation and dependence analysis, with property-based tests of the copula axioms."
---

`copula-core` is a Rust crate for copula modelling, simulation and dependence analysis. By Sklar's theorem a joint distribution with continuous marginals factors into its marginals and a copula, so an implementation has to preserve mathematical constraints and not only return finite numbers: values in the unit interval, uniform margins, the Fréchet-Hoeffding bounds, non-negative densities and valid parameter domains. Property-based tests check those axioms and numerical invariants for the main families.

The core surface, the most mature part of the crate, covers Gaussian and Student-t copulas, the Clayton, Gumbel, Frank, Joe and Ali-Mikhail-Haq families, Marshall-Olkin and empirical copulas, CDF and PDF evaluation, random sampling, tail-dependence coefficients, pseudo-observations, Kendall's tau and Spearman's rho, goodness-of-fit statistics (Cramér-von Mises, Kolmogorov-Smirnov, Anderson-Darling) and AIC and BIC. Parameter estimation by canonical maximum likelihood and inversion of Kendall's tau, with k-fold cross-validation for model selection, sits behind the `estimation` feature and is still evolving. Extreme-value, factor and vine copulas and low-discrepancy sampling are experimental modules outside any stability contract.

## Install

```sh
cargo add copula-core
cargo add rand@0.10   # sampling takes a `rand` 0.10 RNG
```

No Cargo features are enabled by default. `estimation` enables `FittableCopula` and the estimation and model-selection modules, `serde` adds validated `Serialize` and `Deserialize` for the core copula types, and `full` enables both.

## Project Links

- **crates.io:** [copula-core](https://crates.io/crates/copula-core)
- **Documentation:** [docs.rs/copula-core](https://docs.rs/copula-core)
- **Source:** [github.com/DiogoRibeiro7/copula-core](https://github.com/DiogoRibeiro7/copula-core)
- **Issues:** [github.com/DiogoRibeiro7/copula-core/issues](https://github.com/DiogoRibeiro7/copula-core/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/copula-core/blob/main/CHANGELOG.md)
- **Roadmap:** [ROADMAP.md](https://github.com/DiogoRibeiro7/copula-core/blob/main/ROADMAP.md)

## Package Metadata

- **Current release:** `0.2.0`
- **Minimum supported Rust version:** `1.89`
- **License:** MIT OR Apache-2.0
- **Status:** experimental, pre-1.0

## Where It Fits

Use it for dependence modelling and simulation in risk, finance and insurance work that lives in Rust, or as a fast, checkable reference when a copula is one component of a larger simulation. The API changes between minor releases and parts of the numerical surface are not yet validated for inferential work; the project's stated priority is numerical validation of what exists before adding families. For copulas inside a Python extreme-value workflow, see [heavytails](/packages/heavytails/).
