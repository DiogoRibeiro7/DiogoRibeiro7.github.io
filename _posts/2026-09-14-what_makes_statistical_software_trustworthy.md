---
permalink: '/statistics/what_makes_statistical_software_trustworthy/'
title: 'What Makes Statistical Software Trustworthy?'
categories:
- Statistics
- Software Engineering
tags:
- Statistical Software
- Testing
- Numerical Analysis
- Reproducibility
- R Packages
author_profile: false
seo_title: 'What Makes Statistical Software Trustworthy? Invariants, Numerical Tests and Failure Modes'
seo_description: 'Passing unit tests is not enough for statistical software. Trust comes from testing mathematical invariants, numerical stability, state behaviour, malformed inputs, calibration and edge cases that can otherwise return plausible wrong answers.'
excerpt: >-
  Statistical software can pass ordinary unit tests and still fail scientifically.
  The most dangerous bugs often return plausible numbers. Trust comes from testing
  mathematical invariants, numerical stability, state behaviour, calibration and
  adversarial edge cases rather than only checking a few expected outputs.
summary: >-
  A practical framework for testing statistical software as executable mathematics.
  The article uses concrete failure patterns including translation invariance,
  integer overflow, malformed dimensioned input, RNG-state locality, fast-path
  equivalence, calibration and computational boundaries to show why trustworthy
  packages need stronger contracts than conventional unit tests.
keywords:
  - statistical software testing
  - numerical stability
  - invariant testing
  - R package quality
  - reproducibility
  - floating point
classes: wide
date: '2026-09-14'
why_this_exists: >-
  Statistical packages can produce outputs that look reasonable even when a hidden
  numerical, inferential or state-management assumption has failed. The relevant
  question is therefore not whether the code runs, but whether the implementation
  preserves the mathematical properties the method is supposed to have.
evidence: >-
  The discussion is motivated by several real classes of statistical-software
  failure: silent loss of translation invariance from catastrophic cancellation,
  integer overflow at large sample sizes, matrices accepted where vectors were
  documented, global RNG-state leakage, approximate optimisations checked against
  simple reference implementations, and inferential guarantees validated by
  calibration rather than by example outputs alone.
methodology: >-
  Organises testing around explicit mathematical and software invariants. Each
  section turns a theoretical property into an executable regression test and
  explains what class of bug that test can reveal.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-statistics-mahalanobis.jpg
  og_image: /assets/images/headers/photo-statistics-mahalanobis.jpg
  overlay_image: /assets/images/headers/photo-statistics-mahalanobis.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-mahalanobis.jpg
  twitter_image: /assets/images/headers/photo-statistics-mahalanobis.jpg
---

A statistical function can pass all of its obvious tests and still be wrong.

That sounds dramatic until you look at the kinds of failures numerical software actually produces.

A function returns a breakpoint. The value is plausible.

A confidence set has the right class and the right fields.

A Monte Carlo p-value lies between zero and one.

A bootstrap is reproducible under a fixed seed.

None of those facts establishes that the implementation is trustworthy.

The most dangerous statistical bugs are often not crashes. They are silent changes in the mathematical problem being solved.

The result still looks statistical. It may even look reasonable.

That is why I increasingly think of good statistical software as **executable mathematics**.

The tests should not merely ask whether the code returns expected numbers for a few examples. They should encode the properties that must remain true if the implementation still represents the mathematics we claim it does.

## Unit Tests Are Necessary, but They Are Not the Contract

A conventional unit test often looks like this:

```r
x <- c(rep(0, 20), rep(2, 20))
fit <- detect_change(x)

expect_equal(fit$breakpoint, 20)
```

That is useful.

It verifies one known case.

But it leaves many questions unanswered.

Should the result change if I add one billion to every observation?

Should permuting irrelevant input rows change the inference?

What happens when the sample grows from 10,000 to 100,000?

Does passing a matrix where a vector was documented trigger an error or silently flatten the data?

Does a seeded bootstrap alter the user's random-number stream?

Does an optimized implementation still agree with the simple specification it replaced?

Those questions are much closer to the real contract of statistical software.

## Start From Invariants

An invariant is a property that should remain true under a transformation that does not change the statistical problem.

If the theory says

$$
T(x+c)=T(x),
$$

then translation invariance should be a test.

If a procedure is invariant to scale,

$$
T(ax+b)=T(x),
\qquad a>0,
$$

that should be a test too.

If row order has no mathematical meaning, reordering rows should not change the inferential object.

If a confidence interval is the inversion of a test, then test and interval decisions should agree exactly at the matching significance level.

These are stronger tests than examples because they cover entire classes of inputs.

## 1. Translation Invariance Can Expose Catastrophic Cancellation

Suppose a procedure compares residual sums of squares across candidate change points.

A mathematically valid within-segment sum of squares is invariant to adding a constant:

$$
\sum_i (y_i-\bar y)^2
=
\sum_i \{(y_i+c)-(\bar y+c)\}^2.
$$

So if a breakpoint confidence set changes after replacing

$$
y_i
$$

with

$$
y_i+10^8,
$$

something is wrong numerically.

A common one-pass formula is

$$
\sum_i y_i^2
-
\frac{(\sum_i y_i)^2}{m}.
$$

In exact arithmetic that identity is fine.

In floating-point arithmetic the two terms can both be enormous while their difference is small. Subtracting nearly equal large numbers destroys precision.

The bug is not that the formula is algebraically wrong.

The bug is that the chosen computation does not preserve the algebra numerically.

A very useful regression test is therefore:

```r
fit1 <- location_confset(y)
fit2 <- location_confset(y + 1e8)

expect_identical(fit1$set, fit2$set)
```

The test is expressing mathematics directly.

No expected confidence set needs to be hard-coded.

## 2. Integer Overflow Is a Statistical Bug When It Changes the Answer

Consider a contrast weight

$$
w_\tau
=
\sqrt{\frac{\tau(n-\tau)}{n}}.
$$

In R, integer multiplication can overflow before division converts anything to double precision.

The product

$$
\tau(n-\tau)
$$

is largest near \(n^2/4\).

Once that exceeds `.Machine$integer.max`, the result can become `NA` with a warning.

If downstream code uses something like `which.max()`, those missing central candidates may simply disappear from consideration.

The final result can still be a perfectly ordinary integer breakpoint.

That is a terrible failure mode because the output looks plausible.

A boundary test should hit the arithmetic threshold deliberately:

```r
expect_warning(
  unguarded_weight(tau, n),
  "integer overflow"
)

expect_true(is.finite(safe_weight(tau, n)))
```

Then add an end-to-end regression at a sample size beyond the threshold.

A helper-level test proves the arithmetic fix.

The end-to-end test proves that the scientific result is protected.

## 3. Validate the Documented Type, Not Something Close to It

Suppose a function documents its input as a numeric vector.

This guard is not enough:

```r
is.numeric(x)
```

A numeric matrix is numeric too.

If the implementation then uses cumulative sums or vector operations, R may flatten the matrix column-major and proceed without complaint.

The result can be completely nonsensical while looking well formed.

For example, two unrelated time series stored in adjacent columns can create an apparent change point exactly at the column boundary after flattening.

The safer contract is closer to:

```r
is.numeric(x) && is.null(dim(x))
```

when a plain numeric vector is genuinely required.

The corresponding test should not merely check `character` or `NA` inputs. It should check structurally plausible wrong types:

```r
expect_error(
  detect_change(matrix(rnorm(40), ncol = 2)),
  "numeric vector"
)
```

The broader lesson is simple:

$$
\boxed{
\text{type validation is part of statistical correctness.}
}
$$

If the implementation silently answers a different statistical question from the one documented, that is not just an API issue.

## 4. Randomness Has State, Not Just Seeds

A function that uses simulation can be reproducible and still interfere with the caller's experiment.

If it calls

```r
set.seed(42)
```

internally and leaves the resulting `.Random.seed` behind, every stochastic operation after the function sees a different stream.

A stronger RNG contract tests four cases:

$$
\begin{array}{ll}
\text{seed supplied, state existed} & \rightarrow \text{restore it exactly},\\
\text{seed supplied, no state existed} & \rightarrow \text{leave none},\\
\text{seed omitted} & \rightarrow \text{advance the caller stream},\\
\text{same explicit seed} & \rightarrow \text{same result}.
\end{array}
$$

Those are behavioural invariants.

They are more informative than checking reproducibility alone.

```r
set.seed(123)
before <- .Random.seed

invisible(my_bootstrap(seed = 7))

expect_identical(.Random.seed, before)
```

State behaviour belongs in the test suite because random state is part of the software's observable effect.

## 5. Keep a Slow Specification When Optimizing a Fast Path

Optimization is one of the easiest ways to break statistical code while making it look better.

Suppose a straightforward implementation builds a large matrix \(A\) and computes

$$
Av.
$$

Later you realize each row has a special structure and derive the same result from cumulative sums in linear memory.

The optimized implementation is much faster.

Do not delete the simple version immediately.

The slow implementation can become a **specification**.

For moderate inputs, test

$$
\|f_{fast}(x)-f_{reference}(x)\|
$$

against a tight numerical tolerance.

```r
expected <- reference_method(x)
observed <- fast_method(x)

expect_equal(observed, expected, tolerance = 1e-12)
```

This is particularly valuable when the optimization reassociates floating-point arithmetic and bitwise equality is neither realistic nor mathematically required.

A readable reference implementation is often worth keeping precisely because nobody wants to run it in production.

## 6. Test Relationships Between Public Functions

Statistical APIs often contain mathematical dualities.

If a two-sided test and a confidence interval are exact inversions of one another, then

$$
p<\alpha
\iff
0\notin CI_{1-\alpha}.
$$

That should be an executable test.

```r
p <- selective_test(fit)$p.value
ci <- selective_confint(fit, level = 0.95)$conf.int

expect_identical(
  p < 0.05,
  !(ci[1] <= 0 && 0 <= ci[2])
)
```

This kind of cross-function invariant catches inconsistencies that isolated unit tests cannot.

Each function may individually look reasonable while the package as a mathematical system is incoherent.

## 7. Calibration Claims Need Calibration Tests

Suppose a method claims an exact or calibrated p-value under a null model.

Testing a few p-values for membership in \([0,1]\) says almost nothing.

If under the null

$$
P\sim U(0,1),
$$

then the test suite should probe that property directly.

Useful checks include empirical rejection rates:

$$
\widehat{P}(P\le\alpha)
$$

for several \(\alpha\), and distribution-level checks such as a Kolmogorov-Smirnov diagnostic when computationally affordable.

The tolerance must reflect Monte Carlo uncertainty.

At level \(\alpha\) with \(R\) independent simulation replications,

$$
SE\{\widehat p\}
\approx
\sqrt{\frac{\alpha(1-\alpha)}{R}}.
$$

A test that insists a simulated rejection rate equal exactly 0.05 is itself statistically wrong.

A test that accepts anything between 0.01 and 0.20 is nearly useless.

Testing statistical software means applying statistics to the tests too.

## 8. Coverage Needs Its Own Test

A calibrated p-value does not imply a separately implemented confidence-interval routine is correct.

If the interval inverts a pivot through a different numerical path, that path needs direct coverage testing.

For a 95% confidence interval with true parameter \(\theta_0\), simulate many datasets and estimate

$$
\widehat C
=
\frac1R\sum_{r=1}^{R}
\mathbf 1\{\theta_0\in CI_r\}.
$$

Then assess whether \(\widehat C\) is consistent with 0.95 at the Monte Carlo precision of the experiment.

This is a good example of a test that sits between ordinary unit testing and a full research simulation study.

Its purpose is not to publish a result.

Its purpose is to prevent a regression from invalidating a documented inferential guarantee.

## 9. Extreme but Correct Output Should Be Documented, Not “Fixed”

Stress tests sometimes reveal something that looks like a bug but is actually mathematics.

Suppose an exact selective interval can become extremely wide when the selected signal is weak.

A width of thousands of standard deviations can look absurd.

But if the selection event places the observed statistic near a truncation boundary, the inverted conditional pivot may genuinely contain very little information.

In that case the correct response is not to clip the interval until it looks reasonable.

It is to document the behaviour and test that the numerical routine remains stable.

Trustworthy software distinguishes

$$
\text{unexpected output}
$$

from

$$
\text{incorrect output}.
$$

Those are not synonyms.

## 10. Computational Complexity Is Part of the Statistical Interface

A statistically correct method that requires quadratic memory can still be unusable at the sample sizes users reasonably expect.

Complexity should therefore be treated as part of the method's practical contract.

Suppose an algorithm materializes an \(n\times n\) matrix:

$$
M(n)=O(n^2).
$$

At small \(n\), ordinary tests pass.

At large \(n\), the procedure cannot even reach the arithmetic regimes where other bugs may live.

This can hide correctness problems.

Improving the memory complexity can expose new failures, such as integer overflow that was previously unreachable in practice.

Performance work and correctness work are therefore not always separate tracks.

Sometimes removing a computational ceiling is what allows deeper validation.

## 11. Measure Complexity Instead of Guessing It

If a procedure appears slow, benchmark the actual shipped function over a controlled sequence of input sizes.

Do not infer complexity from one timing.

For example, if the work is proportional to candidate count times bootstrap count, and candidate count itself grows with \(n\), a method can naturally approach quadratic runtime.

A useful benchmark table may be more informative than another micro-optimization:

| \(n\) | candidates | rescans | measured time |
| ---: | ---: | ---: | ---: |
| 60 | 51 | 50,949 | 1.2 s |
| 120 | 111 | 110,889 | 3.5 s |
| 240 | 231 | 230,769 | 10.8 s |
| 480 | 471 | 470,529 | 33.4 s |

If the inner statistical operation dominates, there may be no meaningful implementation overhead left to remove.

The honest engineering outcome can be documentation rather than optimization.

That too is part of trust.

## 12. Invariants Beat Snapshot Testing for Scientific Code

Snapshot tests are useful for print methods, serialized objects and stable presentation contracts.

They are usually weak scientific tests.

A snapshot says

> this output did not change.

An invariant says

> this property must not change.

The second is more robust to legitimate implementation changes.

For numerical code I would rather test

$$
T(x+c)=T(x)
$$

than freeze a giant output object produced from one arbitrary \(x\).

I would rather test row-order invariance than snapshot the order of internal bookkeeping rows.

I would rather test coverage than snapshot one interval.

The goal is to protect the mathematics, not fossilize the implementation.

## 13. Negative Tests Should Target Plausible Wrong Inputs

Many validation suites focus on obviously invalid inputs:

- `NULL`,
- character strings,
- negative counts,
- missing values.

Those are necessary.

The more dangerous inputs are often *almost valid*:

- a one-column matrix instead of a vector,
- duplicated grid rows,
- unsorted but otherwise valid support points,
- integer values large enough to overflow a product,
- adjacent representable doubles where no strict midpoint exists,
- a stochastic fixture that is still a lazy promise.

These are the inputs likely to produce plausible wrong output rather than immediate failure.

A mature test suite gradually accumulates exactly these cases because they correspond to ways users and computers actually fail.

## 14. Numerical Edge Cases Deserve Named Contracts

Suppose an iterative refinement algorithm bisects an interval.

Normally,

$$
m=\frac{a+b}{2}
$$

lies strictly between \(a\) and \(b\).

In floating-point arithmetic, if \(a\) and \(b\) are adjacent representable numbers, there may be no representable number strictly between them.

Then the algorithm has stalled even though the mathematical interval width is positive.

That condition deserves an explicit state such as

```text
stalled = TRUE
converged = FALSE
```

rather than pretending the tolerance criterion was satisfied.

This is another example of why numerical algorithms need semantic states, not only returned numbers.

## 15. Row-Order Invariance Is Often a Scientific Property

Suppose a confidence-set procedure evaluates a user-supplied grid of candidate parameter values.

If the grid is a set mathematically, permuting its rows should not alter the inferred set.

You can test

$$
C(G)=C(\pi G)
$$

for a random permutation \(\pi\), while allowing purely representational metadata such as witness row numbers to change.

That distinction matters.

The *inference* should be invariant.

The *bookkeeping index* need not be.

Good tests know which parts of the object are mathematical and which parts are incidental representation.

## 16. Statistical Guarantees Should Appear in Documentation Tests Too

A package can have correct code and misleading documentation.

That is still a scientific defect.

Suppose one function is finite-sample exact, two others are approximate plug-in bootstraps, and another is exact only when a nuisance parameter is known.

If the README calls all of them “exact inference,” the package is not trustworthy even if every function is implemented correctly.

Claims should therefore be synchronized across:

- function help,
- README,
- vignettes,
- package description,
- citation metadata,
- release notes.

This is less glamorous than numerical analysis, but users experience the method through the claims attached to it.

Trust includes claim discipline.

## 17. A Green Check Is Evidence, Not Proof

Continuous integration answers a narrow question:

> Did the checks we chose pass in the environments we ran them in?

It does not answer:

> Is the statistical method correct?

A package can have 100% code coverage and weak scientific tests.

Conversely, a package with lower statement coverage can have extremely strong invariant tests around its critical mathematics.

Coverage is useful for finding unexercised code.

It is not a validity certificate.

The more useful question is:

$$
\boxed{
\text{Which mathematical claims would fail if this implementation were wrong?}
}
$$

Then write tests that make those claims executable.

## A Practical Hierarchy of Statistical Tests

I find it useful to think in layers.

### Layer 1: ordinary examples

Known inputs with known outputs.

These catch regressions and gross implementation errors.

### Layer 2: validation contracts

Malformed, dimensioned, non-finite, duplicated or structurally inconsistent inputs.

These stop the function from silently solving the wrong problem.

### Layer 3: mathematical invariants

Translation, scale, sign, permutation, duality and conservation identities.

These protect the theory.

### Layer 4: numerical equivalence

Fast path versus reference path, alternate stable formulas, convergence under refinement.

These protect the computation.

### Layer 5: stochastic behaviour

RNG locality, reproducibility, empirical calibration and coverage.

These protect simulation-based guarantees.

### Layer 6: computational boundary

Memory growth, runtime scaling and large-input arithmetic.

These protect the domain where the method is claimed to be usable.

A trustworthy package usually needs evidence across several layers.

## The Most Valuable Test Often Comes From a Bug

There is a natural tendency to view regression tests as scars from past mistakes.

I think they are better understood as discoveries about the actual contract.

A cancellation bug reveals that translation invariance matters numerically.

An integer overflow reveals that sample-size boundaries matter.

A flattened matrix reveals that input shape matters scientifically.

An RNG leak reveals that random state is part of the API.

A failed lazy fixture reveals that evaluation order matters to the test harness itself.

Each bug teaches you one more property that should never be allowed to drift again.

That is how a statistical test suite becomes more than a collection of examples.

It becomes a machine-readable statement of the method.

## The Standard I Would Use

I would not call statistical software trustworthy because it has many tests.

I would call it trustworthy when the tests make important wrong behaviours difficult to hide.

That means checking things like:

$$
\boxed{
\begin{aligned}
&\text{mathematical invariance},\\
&\text{numerical stability},\\
&\text{inferential calibration},\\
&\text{state locality},\\
&\text{input semantics},\\
&\text{reference equivalence},\\
&\text{edge-case honesty},\\
&\text{claim discipline}.
\end{aligned}
}
$$

The central idea is simple:

> Statistical software should be tested against the mathematics it claims to implement, not only against the outputs it happened to produce yesterday.

That is a higher standard than “the package builds.”

For scientific software, it is the standard that matters.
