---
layout: page
title: "uncertain-numerics"
permalink: /packages/uncertain-numerics/
author_profile: true
seo_title: "uncertain-numerics Rust Crate"
seo_description: "Project page for uncertain-numerics, a Rust crate of probabilistic numerical methods: Bayesian quadrature and probabilistic linear solvers that return calibrated Gaussian posteriors."
---

`uncertain-numerics` treats numerical computation as an inference problem. Instead of returning only a point estimate of an integral or of the solution of a linear system, every method returns a validated Gaussian posterior: its mean is the estimate and its variance states how much the computation still does not know. Each statistical and numerical assumption behind that posterior is explicit, tested against analytic references and checked for calibration.

It provides Bayesian quadrature in one dimension with a Gaussian-process prior (RBF kernel, Gaussian integration measure) and closed-form kernel integrals; active Bayesian quadrature with a posterior-variance-reduction acquisition rule, a sequential evaluation loop and explicit stopping rules; and probabilistic linear solvers for dense symmetric positive-definite systems, which condition a Gaussian belief over the solution on exact projections under residual-driven, A-conjugate and covariance-greedy search policies. The numerics are auditable: Cholesky solves instead of explicit inverses, jitter as a visible parameter that is never escalated silently, and typed errors instead of `NaN` for invalid input. The integration tests are written as scientific studies of coverage, calibration, misspecification and numerical stability. The crate has one dependency, `nalgebra`, and forbids `unsafe`.

## Install

```sh
cargo add uncertain-numerics
```

## Project Links

- **crates.io:** [uncertain-numerics](https://crates.io/crates/uncertain-numerics)
- **Documentation:** [docs.rs/uncertain-numerics](https://docs.rs/uncertain-numerics)
- **Source:** [github.com/DiogoRibeiro7/uncertain-numerics](https://github.com/DiogoRibeiro7/uncertain-numerics)
- **Issues:** [github.com/DiogoRibeiro7/uncertain-numerics/issues](https://github.com/DiogoRibeiro7/uncertain-numerics/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/CHANGELOG.md)
- **Roadmap:** [ROADMAP.md](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/ROADMAP.md)

## Package Metadata

- **Current release:** `0.1.0`
- **Minimum supported Rust version:** `1.85` (edition 2024)
- **License:** MIT OR Apache-2.0
- **Status:** pre-1.0

## Where It Fits

Use it when the uncertainty of a computation matters as much as its result: a quadrature budget of a handful of evaluations, or a linear solve stopped early, where the posterior variance tells you what the remaining error could be. It is not a fast solver. With the identity prior the projection solvers are Craig's method in disguise, and at equal matrix-vector budgets classical conjugate gradients is far more accurate, so use them for the posterior covariance and pick the covariance-greedy policy when calibration matters. Quadrature is limited to one dimension, and the linear solvers to dense systems of moderate size.
