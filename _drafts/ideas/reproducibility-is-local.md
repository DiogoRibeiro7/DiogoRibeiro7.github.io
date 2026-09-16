---
author_profile: false
categories:
- Research
classes: wide
excerpt: Reproducibility in a long-lived research repository should be attached to explicit research objects and project contracts, not inferred from the repository root.
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
- computational reproducibility
- research software
- research compendia
- monorepos
- scientific computing
- provenance
seo_description: Why reproducibility in research repositories should be scoped to explicit projects and results rather than inferred from one synthetic global environment.
seo_title: Reproducibility Is Local
seo_type: article
summary: A practical argument for project-scoped environments, provenance, tests, execution contracts, and honest maintenance guarantees in long-lived research repositories.
tags:
- Research Software
- Reproducibility
- Engineering
title: 'Reproducibility Is Local'
---

A long-lived research repository is rarely one application.

It may contain notebooks from different years, code written for several papers, teaching material, one-off analyses, small packages, generated figures, frozen publication artifacts, and active projects that still evolve.

Putting all of those files in one Git repository is an organizational choice. It does not imply that they share one scientific lifecycle, one dependency graph, or one reproducibility guarantee.

That is the central argument of this article:

$$
\boxed{
\text{Reproducibility should be attached to an explicit research object, not inferred from repository membership.}
}
$$

Here, **local** does not mean “works only on my laptop.” It means **local to the claim being made**: a project, analysis, result, figure, table, package, or research compendium with an explicit execution contract.

## First, define what we mean by reproducibility

The vocabulary in this area is famously inconsistent across disciplines. I will use the distinction adopted by the US National Academies:

- **reproducibility** means obtaining consistent computational results using the same input data, computational steps, methods, code, and conditions of analysis;
- **replicability** means obtaining consistent results in a new study addressing the same scientific question with newly obtained data.

Under that definition, this article is mainly about **computational reproducibility**.

The distinction matters because a perfectly reproducible analysis can still fail to replicate scientifically, and a scientifically robust result can be difficult to reproduce computationally if the original software environment or data provenance was not recorded.

Reproducibility is therefore a property of an execution claim, not a synonym for scientific truth.

## Reproducibility needs a boundary

A useful reproducibility claim should answer at least five questions:

$$
\mathcal R
=
(\text{result},\text{inputs},\text{code},\text{environment},\text{execution path}).
$$

For example:

> Figure 3 can be regenerated from versioned public data using commit `abc123`, the project-local environment specification, and `make figure-3`.

That is a much stronger statement than:

> This repository is reproducible.

The second claim is ambiguous. Which result? Which data? Which directory? Which runtime? Which version? Does the guarantee include ten-year-old notebooks? Does it include external APIs? Does it include private data that cannot legally be distributed?

A precise reproducibility claim is smaller, but it is testable.

## Repository, project, and result are different objects

This distinction is easy to lose in a monorepo.

Consider a repository with this shape:

```text
research-archive/
├── paper-a/
├── paper-b/
├── teaching-notebooks/
├── old-experiment-2018/
├── maintained-package/
└── figures-from-blog-posts/
```

The repository is one version-control unit.

But the natural reproducibility units may be:

```text
paper-a/
maintained-package/
```

while `old-experiment-2018/` may be only a preserved historical artifact and `teaching-notebooks/` may not make any formal reproducibility claim at all.

The conceptual model is therefore closer to

$$
\text{repository}
=
\sum_{k=1}^{K}\text{research objects}_k
$$

where each research object can have its own status, environment, provenance, and maintenance contract.

## One repository does not imply one runtime

A common modernization instinct is to place a single environment file at the repository root and make every directory conform to it.

Sometimes that is correct.

If the repository genuinely contains one application, one dependency graph, one release cycle, and components that are expected to evolve together, then a global environment can be exactly the right abstraction.

But for a research archive spanning several years, a global environment can manufacture a false unity.

Imagine two projects:

```text
project-a: Python 3.8 + pandas 1.x + an old scientific API
project-b: Python 3.13 + pandas 3.x + a modern package stack
```

Forcing both into one environment may require rewriting the older analysis. That rewrite can improve maintainability while simultaneously weakening historical fidelity.

The result may run today, but it is no longer a faithful description of the computational object that produced the original result.

This is why “modernized” and “reproducible” are not synonyms.

## A local environment is an execution contract

For an actively maintained research project, I want the project boundary to make its own requirements explicit.

A small project might contain:

```text
analysis-project/
├── README.md
├── pyproject.toml
├── lockfile
├── data/
├── src/
├── tests/
├── figures/
└── Makefile
```

The exact file names are not important. The contract is.

At minimum, the project should make it possible to answer:

- Which interpreter or runtime is required?
- Which packages and versions matter?
- Which command produces the result?
- Which data are inputs rather than outputs?
- Which artifacts are generated?
- Which randomness must be controlled?
- Which external services are required?
- Which validation checks establish that the reproduction succeeded?

This is close to the idea of a **research compendium**: code, data, documentation, environment information, and outputs packaged around one research object rather than around an arbitrary repository root.

## A lockfile is necessary surprisingly often, and sufficient almost never

Dependency capture is important. It is also only one layer.

A fully pinned Python environment will not reproduce a result if:

- the original data were overwritten;
- a preprocessing step happened manually in Excel;
- an API now returns different data;
- a model download changed upstream;
- a random seed was not recorded;
- GPU kernels are nondeterministic;
- a required environment variable is undocumented;
- a database query depends on mutable production tables;
- the plotting code reads a cached intermediate file whose provenance is unknown.

So I prefer to think of reproducibility as a provenance chain:

$$
\text{claim}
\leftarrow
\text{output}
\leftarrow
\text{code}
\leftarrow
\text{processed data}
\leftarrow
\text{raw inputs}
\leftarrow
\text{source provenance}.
$$

The environment surrounds that chain, but it does not replace it.

Sandve and colleagues make essentially this point in operational form: keep track of how each result was produced, avoid manual manipulation, archive software versions, record random seeds, preserve underlying data, and connect textual claims to the computations that support them.

## Reproducible today is not the same as reconstructible historically

There are at least two useful goals that often get mixed together.

### Goal 1: active reproducibility

A maintained project should run now under a documented environment and produce validated outputs.

### Goal 2: historical reconstruction

An older project should preserve enough information to understand how the original result was produced, even if running it today requires an obsolete interpreter, an archived container, unavailable hardware, or data that cannot be redistributed.

These are different maintenance problems.

Trying to turn every historical artifact into an actively maintained modern project can consume enormous effort while erasing the original context.

For long-lived repositories, it is often better to classify projects honestly.

For example:

| Status | Meaning |
| --- | --- |
| **Maintained and verified** | Current environment, documented execution, validation, active maintenance. |
| **Reproducible historical** | Environment and execution path are documented well enough to rerun or reconstruct the main result, but the project is not continuously modernized. |
| **Preserved historical** | Code, data, and context are retained for provenance, but execution is not guaranteed. |
| **Unclassified** | Reproducibility status has not yet been audited. |

This vocabulary is more informative than a repository-wide badge saying “reproducible.”

## Continuous integration should test the guarantee you actually make

CI is most useful when it enforces a real contract.

For a maintained subproject, that may mean:

```text
install locked environment
→ validate data schema
→ run tests
→ execute analysis
→ compare key outputs or invariants
```

That is meaningful because the workflow corresponds to a specific reproducibility promise.

By contrast, a repository-wide workflow that attempts to execute every notebook accumulated over ten years may create two bad outcomes.

The first is permanent red CI that everybody learns to ignore.

The second is continual rewriting of historical material merely to keep the root check green.

Neither improves scientific trust.

Repository-level CI can still be valuable, but its responsibilities should be different:

- validate metadata;
- check catalogue consistency;
- verify links;
- scan for secrets;
- enforce conventions for **new** maintained work;
- confirm that declared project paths and local environment files exist.

In other words:

$$
\boxed{
\text{CI scope should match guarantee scope.}
}
$$

## Reproducibility also needs a success criterion

“Command completed without error” is not always enough.

Suppose a simulation runs successfully but produces materially different conclusions because an upstream numerical library changed.

The pipeline executed. The scientific result did not reproduce.

A stronger project contract should define what successful reproduction means.

Depending on the project, that might be:

- exact file hashes;
- identical deterministic tables;
- numerical tolerances;
- invariant model coefficients within expected precision;
- regenerated figures from the same underlying data;
- statistical acceptance criteria for stochastic outputs;
- semantic checks on the conclusions derived from an analysis.

The appropriate criterion depends on the computation.

Exact bitwise identity is useful when achievable, but it is not the only legitimate form of computational reproducibility.

## External services create moving dependencies

Modern research code increasingly depends on objects outside the repository:

```text
APIs
cloud buckets
databases
model registries
package indexes
remote files
hosted notebooks
LLM endpoints
```

These are dependencies too.

If an analysis uses a public API, recording the URL is weaker than recording the retrieved snapshot and retrieval date when licensing permits it.

If an analysis queries a mutable database, the SQL statement is not sufficient provenance unless the underlying data state can also be reconstructed.

If a project depends on a hosted model, the provider name alone may be insufficient if model revisions can change while the name remains stable.

Reproducibility contracts should therefore distinguish:

$$
\text{versioned local dependency}
\neq
\text{mutable external dependency}.
$$

The latter often needs snapshots, checksums, release identifiers, or explicit statements that exact reproduction is impossible after the external state changes.

## FAIR and reproducible are related, but not identical

The FAIR principles ask whether digital research objects are **Findable, Accessible, Interoperable, and Reusable**.

Those qualities strongly support reproducibility, particularly through metadata, persistent identifiers, accessibility, and reuse.

But FAIRness does not by itself mean that pressing a button will regenerate a paper.

Likewise, a perfectly reproducible private pipeline may be poorly findable or inaccessible to other researchers.

It is useful to keep the concepts separate:

$$
\text{FAIRness}
\quad\text{supports}\quad
\text{reusability and discovery}
$$

while

$$
\text{computational reproducibility}
\quad\text{concerns}\quad
\text{reconstructing the computation and result}.
$$

They overlap, but they answer different questions.

## Project-local does not mean isolated

There is one possible misunderstanding worth avoiding.

Project-scoped environments do not require every project to duplicate everything.

A repository can still share:

- CI templates;
- linting configuration;
- documentation conventions;
- common libraries;
- data schemas;
- infrastructure modules;
- publication tooling.

The key is that shared infrastructure should not silently create a stronger reproducibility claim than the projects can support.

A useful design is:

$$
\text{shared engineering infrastructure}
+
\text{local scientific contracts}.
$$

That gives consistency without pretending every artifact has the same dependency history.

## A practical reproducibility contract

For each maintained research object, I would like to be able to write something like this:

```text
Object:
  Analysis supporting Table 2.

Inputs:
  data/raw/cohort.csv, SHA-256 ...

Code:
  commit 91c4e2f

Environment:
  Python 3.13 + project lockfile

Command:
  make table-2

Randomness:
  seed 20260916

External dependencies:
  none

Expected output:
  results/table_2.csv

Validation:
  all rows present;
  coefficients within declared numerical tolerance;
  manuscript values generated from this artifact.
```

That is not bureaucratic metadata for its own sake.

It is a compact answer to the question:

> What exactly would another person need to do to check this result?

## The right boundary can change over time

A research object may begin as one notebook and later become a package with several analyses.

Two projects may eventually converge into one maintained system.

A historical folder may be promoted from “preserved” to “reproducible historical” after its environment is reconstructed.

So project boundaries are not sacred.

What matters is that the boundary and the guarantee move together.

If two projects genuinely share one lifecycle, combining their environment may simplify reproducibility.

If two artifacts only happen to live in the same repository, forcing them into one runtime usually does the opposite.

## The practical rule

Do not ask:

> Is this entire repository reproducible?

Ask instead:

$$
\boxed{
\begin{aligned}
&\text{Which result?}\\
&\text{Which inputs?}\\
&\text{Which code version?}\\
&\text{Which environment?}\\
&\text{Which execution path?}\\
&\text{Which success criterion?}
\end{aligned}
}
$$

That produces narrower claims.

But narrower claims are often the stronger scientific claims, because somebody can actually test them.

Reproducibility is not made more rigorous by making its scope larger.

It is made more rigorous by making its scope explicit.

## References

- National Academies of Sciences, Engineering, and Medicine. *Reproducibility and Replicability in Science*. National Academies Press, 2019. DOI: [10.17226/25303](https://doi.org/10.17226/25303).
- Sandve GK, Nekrutenko A, Taylor J, Hovig E. *Ten Simple Rules for Reproducible Computational Research*. PLOS Computational Biology. 2013;9(10):e1003285. DOI: [10.1371/journal.pcbi.1003285](https://doi.org/10.1371/journal.pcbi.1003285).
- Wilson G, Bryan J, Cranston K, Kitzes J, Nederbragt L, Teal TK. *Good enough practices in scientific computing*. PLOS Computational Biology. 2017;13(6):e1005510. DOI: [10.1371/journal.pcbi.1005510](https://doi.org/10.1371/journal.pcbi.1005510).
- Wilkinson MD, Dumontier M, Aalbersberg IJ, et al. *The FAIR Guiding Principles for scientific data management and stewardship*. Scientific Data. 2016;3:160018. DOI: [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18).
- The Turing Way Community. *The Turing Way: A handbook for reproducible, ethical and collaborative data science*. See the chapters on reproducible environments and research compendia.
