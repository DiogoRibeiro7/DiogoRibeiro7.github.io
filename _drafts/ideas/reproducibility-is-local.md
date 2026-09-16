---
author_profile: false
categories:
- Research
classes: wide
excerpt: Reproducibility in a long-lived research repository should be enforced at the project boundary, not by pretending every historical notebook belongs to one global environment.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- reproducibility
- research software
- monorepos
- scientific computing
- software engineering
seo_description: Why reproducibility in research monorepos should be scoped to projects rather than forced into one synthetic repository-wide environment.
seo_title: Reproducibility Is Local
seo_type: article
summary: A practical argument for project-scoped environments, tests, provenance, and maintenance guarantees in long-lived research repositories.
tags:
- Research Software
- Reproducibility
- Engineering
title: 'Reproducibility Is Local'
---

A long-lived research repository is rarely one application.

It may contain notebooks from different years, papers with frozen environments, one-off analyses, teaching material, small packages, generated figures, and active projects that still evolve.

Trying to make all of that obey one dependency graph often makes the repository less honest, not more reproducible.

## Reproducibility needs a boundary

A reproducibility claim should answer a simple question:

> Reproducible **what**, under **which environment**, from **which inputs**?

For a maintained project, that boundary may include:

- a local dependency specification;
- a documented execution command;
- public or controlled data provenance;
- deterministic seeds where relevant;
- tests or validation checks;
- generated outputs tied to code.

Those guarantees should apply to the project that defines them.

## One repository does not imply one runtime

A monorepo is a storage and collaboration choice. It does not imply that every directory shares the same scientific lifecycle.

Forcing an old notebook and a modern package into one Python environment can create a false promise: the repository root appears reproducible while individual artifacts still depend on undocumented assumptions.

A better model is

$$
\boxed{
\text{repository} = \sum \text{projects with explicit local contracts}
}
$$

rather than one synthetic global application.

## Historical material should be classified, not rewritten

Old code can remain valuable even when it no longer runs unchanged.

The useful distinction is between states such as:

- maintained and verified;
- reproducible historical;
- preserved historical;
- unclassified.

This is more informative than silently modernizing everything until the original computational context disappears.

## Reproducibility is more than dependencies

A lockfile alone does not reproduce a scientific result.

We also need to know:

- where the data came from;
- which preprocessing was applied;
- which random seeds matter;
- which outputs are generated;
- which claims those outputs support;
- which external services or secrets are required.

The execution environment is one part of provenance, not the whole of it.

## CI should follow the same boundary

Project-scoped CI is usually more meaningful than a repository-wide workflow that attempts to execute every historical artifact.

A maintained subproject can own its tests and reproducibility checks. Repository-level automation can remain lightweight: link validation, metadata checks, catalogue consistency, or secret scanning.

The result is a system where automation matches the guarantees actually being made.

## The practical rule

Reproducibility becomes easier to reason about when its scope is explicit.

Do not ask whether an entire research archive is reproducible as though that were a single binary property.

Ask instead:

$$
\boxed{
\text{Which project, which result, which inputs, which environment, and which guarantee?}
}
$$

That produces smaller claims, but they are claims we can actually defend.
