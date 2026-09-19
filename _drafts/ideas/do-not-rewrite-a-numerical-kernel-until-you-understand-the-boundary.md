---
author_profile: false
categories:
- Programming
classes: wide
title: 'Do Not Rewrite a Numerical Kernel Until You Understand the Boundary'
excerpt: Rewriting Fortran or C in Python, Rust, or another language can increase maintenance cost without improving the bottleneck. The right boundary often matters more than the source language.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- Fortran
- C
- Python
- interoperability
- f2py
- ISO_C_BINDING
- numerical computing
seo_title: 'Do Not Rewrite a Numerical Kernel Until You Understand the Boundary'
seo_description: 'Why interoperability can be better than rewriting numerical kernels, and why algorithm, data movement, vectorization, and interface stability matter more than language fashion.'
seo_type: article
summary: 'A numerical kernel should be rewritten only after the algorithmic and memory bottlenecks are understood. Stable language boundaries can preserve validated kernels while exposing modern interfaces.'
tags:
- Scientific Computing
- Fortran
- Performance Engineering
why_this_exists: 'Legacy numerical code is often treated as a language problem. In practice the difficult questions are algorithmic complexity, memory layout, validation, and interface design.'
evidence: 'Small dense-kernel benchmark design, interface examples using f2py and ISO_C_BINDING, and maintenance trade-off analysis.'
methodology: 'Separate algorithmic cost, memory traffic, interpreter overhead, library calls, and interface overhead before deciding whether a rewrite is justified.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: When is interoperability better than rewriting a numerical kernel?
Claim: Stable, verified kernels should often be preserved behind a narrow interface until measurement shows that the implementation boundary is the actual bottleneck.
Counterclaim: Rewrites can be justified when the original code is unverifiable, unmaintainable, or blocks required architecture.
Evidence object: Matrix and stencil kernels, boundary-cost decomposition, and interop examples.
Failure case: Interoperability can preserve technical debt if the kernel itself is poorly specified or untested.
Reader payoff: Measure before rewriting and design the boundary around stable mathematical operations.
Exclusions: Language advocacy.
-->

A scientific codebase contains a Fortran kernel written fifteen years ago.

The instinct is familiar:

> We should rewrite this in a modern language.

Sometimes that is correct.

Sometimes it replaces validated numerical software with a slower, less mature implementation while leaving the real bottleneck untouched.

The first question should not be which language is fashionable.

It should be where the computational boundary belongs.

## Separate algorithm from language

Suppose a routine has complexity

$$
O(n^3).
$$

Rewriting it in another language does not change that complexity.

A poor algorithm implemented in C can lose to a good algorithm called from Python.

Likewise, a Python wrapper around BLAS may outperform hand-written native code because the heavy work is executed in highly tuned compiled libraries.

Source language is only one layer.

## Interpreter overhead matters mainly at the wrong granularity

Python function calls are expensive relative to arithmetic.

That matters when the hot loop runs in Python.

It matters far less when Python calls one compiled routine that performs millions of floating-point operations.

This suggests a natural architecture:

high-level orchestration outside;
dense numerical kernel inside.

The goal is not to eliminate language boundaries.

It is to make them coarse enough that boundary cost is negligible.

## Data movement can dominate arithmetic

Modern numerical performance is often constrained by memory bandwidth rather than floating-point throughput.

A kernel that repeatedly copies arrays between layouts or runtimes can lose most of its performance before arithmetic begins.

This is why interface design matters.

A rewrite that preserves the same unnecessary copies may not help.

An interop layer that passes contiguous memory directly may be better.

## Stable mathematical kernels are good boundaries

Some operations have clear contracts:

- solve this linear system;
- advance this PDE state one step;
- evaluate this likelihood;
- compute this quadrature;
- apply this transform.

These are natural language boundaries.

They can be tested against analytic solutions, conservation laws, or reference implementations.

A narrow interface also isolates the rest of the application from implementation language.

## f2py is useful when the kernel is already Fortran

Fortran numerical routines can often be exposed to Python with f2py.

The attractive case is a kernel that already has:

- explicit array inputs;
- deterministic outputs;
- stable dimensions;
- minimal global state.

The wrapper can then expose a Pythonic interface while preserving the validated numerical implementation.

The hard work is usually not the wrapper.

It is cleaning the kernel boundary.

## ISO_C_BINDING makes the interface explicit

Modern Fortran provides ISO_C_BINDING for interoperable interfaces.

This is valuable even if Python is the final caller.

A C-compatible boundary can be reused by Python, Rust, Julia, or other hosts.

That reduces coupling.

The numerical kernel becomes a library rather than a language-specific application.

## Rewriting can silently change numerics

A rewrite may alter:

- floating-point order;
- convergence criteria;
- random-number generation;
- linear-algebra backend;
- memory precision;
- boundary conditions.

The new code can look cleaner while producing subtly different results.

This is why numerical verification should precede migration.

A rewrite without a verification harness is not modernisation.

It is an uncontrolled experiment.

## Performance claims require a benchmark design

A useful benchmark should control:

- input size;
- warm-up;
- compiler optimisation;
- BLAS implementation;
- thread count;
- memory layout;
- number of repetitions.

Otherwise the comparison measures environment differences rather than language effects.

A single wall-clock number is not enough.

## Maintenance can justify a rewrite

Performance is not the only objective.

A rewrite may be justified when:

- nobody can build the original code;
- tests are absent;
- global state makes reuse impossible;
- the interface cannot represent required data;
- platform support has become unacceptable.

But those are engineering reasons.

They should be stated explicitly.

## Interoperability can preserve validation capital

Scientific software accumulates validation over time.

Researchers compare outputs with experiments, publications, and earlier implementations.

That history has value.

Wrapping a stable kernel preserves that validation capital.

A rewrite resets part of it.

This cost is easy to ignore because it does not appear in the profiler.

## The correct unit of modernisation is the boundary

A useful modernisation plan is often:

1. define the mathematical contract;
2. add verification tests;
3. profile the implementation;
4. isolate the kernel;
5. expose a stable interface;
6. rewrite only if evidence still supports it.

This sequence avoids treating language identity as the problem.

## Conclusion

A numerical kernel should not be rewritten simply because its language is old.

The relevant questions are algorithm, memory movement, verification, interface stability, and maintenance cost.

A narrow interoperable boundary often buys most of the architectural benefit without discarding a validated kernel.

Modernisation begins with understanding the boundary.

Not with deleting the old file extension.

## References

- NumPy Developers. *F2PY user guide*.
- Metcalf M, Reid J, Cohen M. *Modern Fortran Explained*. Oxford University Press.
- ISO/IEC 1539-1. *Programming languages — Fortran*.
- Hennessy JL, Patterson DA. *Computer Architecture: A Quantitative Approach*. Morgan Kaufmann.
