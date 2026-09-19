---
permalink: '/statistics/reproducible_randomness_is_more_than_calling_set_seed/'
title: 'Reproducible Randomness Is More Than Calling set.seed()'
categories:
- Statistics
tags:
- R
- Reproducibility
- Random Number Generation
- Statistical Computing
- R Packages
- Monte Carlo
author_profile: false
seo_title: 'Reproducible Randomness in R Without Breaking the Caller RNG State'
seo_description: 'Calling set.seed() inside a statistical function can make its result reproducible while silently changing every random result that follows. A better RNG contract preserves the caller state when a seed is supplied and advances it normally when no seed is supplied.'
excerpt: >-
  A function can be reproducible and still behave badly. Calling set.seed()
  internally fixes its own Monte Carlo result but also replaces the caller's
  random-number stream. Statistical software should guarantee both reproducibility
  and RNG-state locality.
summary: >-
  A practical RNG contract for R functions that use simulation: identical seeds
  reproduce identical results, supplied seeds restore the caller's state exactly,
  callers with no prior RNG state are left with none, and unseeded calls consume
  the caller's stream normally. Includes regression tests, error-path checks and an
  R lazy-evaluation trap that can make an RNG test test itself instead of the code.
keywords:
  - R random seed
  - reproducibility
  - random number generator state
  - Monte Carlo simulation
  - statistical software
  - set.seed
classes: wide
date: '2026-09-07'
why_this_exists: >-
  Statistical functions frequently expose a seed argument for reproducibility, but
  reproducibility is only half of the API contract. A seeded function that leaves
  .Random.seed changed can silently alter simulations, bootstrap samples and tests
  executed later in the user's session.
evidence: >-
  The article develops four executable invariants for seeded statistical functions:
  exact restoration when a caller already has RNG state, cleanup when the caller
  had no RNG state, normal stream advancement when no seed is supplied, and exact
  replay under the same explicit seed.
methodology: >-
  Uses a minimal Monte Carlo function to contrast naive internal set.seed() calls
  with a local-seed wrapper that snapshots RNG kind and state, restores them with
  on.exit(), and deliberately leaves unseeded calls connected to the caller's
  stream. Regression tests exercise normal, absent-state and error paths.
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

Calling `set.seed()` is usually the first thing we learn about reproducible simulation in R.

It is necessary often enough that it becomes almost synonymous with reproducibility:

```r
set.seed(42)
rnorm(5)
```

Run the code again with the same R version, RNG configuration and seed, and the same pseudo-random sequence is generated.

That is useful.

It is not yet a good random-number contract for a statistical function.

Consider a package function that runs a bootstrap internally:

```r
bootstrap_mean <- function(x, B = 999L, seed = 1L) {
  set.seed(seed)

  estimates <- replicate(
    B,
    mean(sample(x, replace = TRUE))
  )

  quantile(estimates, c(0.025, 0.975))
}
```

The function is reproducible.

It also silently replaces the caller's random-number stream.

That second fact is easy to miss because the function returns the right object. The damage appears later, in code that has nothing visibly to do with the function call.

For statistical software, I want a stronger guarantee:

$$
\boxed{
\text{reproducibility}
+
\text{RNG-state locality}
}
$$

The function should control its own randomness when explicitly asked to do so, without controlling the caller's future randomness as a side effect.

## The Hidden Global Variable

R's ordinary pseudo-random-number generator is stateful.

After random numbers have been generated, its state is represented by the object

```r
.Random.seed
```

in the global environment.

A call such as

```r
runif(1)
```

uses the current state and replaces it with the next state in the stream.

Schematically,

$$
S_t
\xrightarrow{\text{draw}}
(X_t,S_{t+1}),
$$

where $S_t$ is the generator state and $X_t$ the generated value.

`set.seed(42)` does something different. It replaces the current state with the state associated with seed 42:

$$
S_t
\xrightarrow{\texttt{set.seed(42)}}
S_{42}.
$$

That is exactly what we want at the top of a script when we intentionally define the random experiment.

Inside a reusable function it is a global side effect.

## A Reproducible Function Can Break a Reproducible Script

Here is a smaller example.

```r
naive_random_mean <- function(seed) {
  set.seed(seed)
  mean(rnorm(100))
}
```

Now compare a caller's intended random stream with the stream after that function is inserted between two draws.

```r
set.seed(100)
expected <- c(runif(1), runif(1))

set.seed(100)
first <- runif(1)

invisible(naive_random_mean(42))

second <- runif(1)

identical(c(first, second), expected)
# FALSE
```

The caller seeded the experiment correctly.

The function changed it anyway.

The failure is compositional. Each piece of code looks reproducible in isolation, but combining them changes the experiment.

This is why I do not think the contract

> same seed, same result

is sufficient for a statistical package.

A reusable function also needs to say what happens to the surrounding random stream.

## The Four Invariants I Want

For a statistical function with an optional `seed` argument, I would test four behaviours.

### 1. Explicit seed, existing caller state

If the caller already has an RNG state and supplies a function-local seed, the caller state should be byte-for-byte identical after the call:

$$
S_{\text{after}}=S_{\text{before}}.
$$

### 2. Explicit seed, no existing caller state

If `.Random.seed` did not exist before the call, it should not suddenly exist afterwards.

The function should not leave random-generator state behind merely because its internal implementation used simulation.

### 3. No explicit seed

If `seed = NULL`, the function should normally consume the caller's current random stream.

Then

$$
S_{\text{after}}\neq S_{\text{before}}
$$

when the function actually draws randomness.

This preserves the usual R semantics. An unseeded stochastic call belongs to the caller's experiment.

### 4. Same explicit seed, same result

Two calls with the same inputs and explicit seed should reproduce the same stochastic result.

This is the familiar part, but it belongs together with the other three.

The full contract is therefore:

$$
\boxed{
\begin{array}{ll}
\text{seed supplied, state exists} & \rightarrow \text{restore it},\\
\text{seed supplied, no state exists} & \rightarrow \text{leave none},\\
\text{seed omitted} & \rightarrow \text{advance caller stream},\\
\text{same seed} & \rightarrow \text{same result}.
\end{array}
}
$$

## A Local Seed Wrapper in Base R

A small wrapper can implement that contract.

I prefer a callback here because it makes the evaluation boundary explicit.

```r
with_local_seed <- function(seed, fn) {
  stopifnot(is.function(fn))

  if (is.null(seed)) {
    return(fn())
  }

  if (
    length(seed) != 1L ||
    is.na(seed) ||
    seed != as.integer(seed)
  ) {
    stop("`seed` must be NULL or a single integer.", call. = FALSE)
  }

  old_kind <- RNGkind()
  had_seed <- exists(
    ".Random.seed",
    envir = .GlobalEnv,
    inherits = FALSE
  )

  if (had_seed) {
    old_seed <- get(
      ".Random.seed",
      envir = .GlobalEnv,
      inherits = FALSE
    )
  }

  on.exit({
    # Restore the generator configuration before restoring its exact state.
    do.call(RNGkind, as.list(old_kind))

    if (had_seed) {
      assign(
        ".Random.seed",
        old_seed,
        envir = .GlobalEnv
      )
    } else if (
      exists(
        ".Random.seed",
        envir = .GlobalEnv,
        inherits = FALSE
      )
    ) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(as.integer(seed))
  fn()
}
```

The `on.exit()` is the important part.

The state must be restored not only after a successful result but also after an error.

The wrapper also records `RNGkind()`. A function that only calls `set.seed()` does not change the RNG kind, so this is slightly more defensive than strictly necessary for the minimal example. It becomes necessary if code inside the boundary changes the generator, normal generator or sampling algorithm.

## A Better Monte Carlo Function

Now the stochastic function can make its seed semantics explicit.

```r
random_mean <- function(n = 100L, seed = NULL) {
  if (
    length(n) != 1L ||
    is.na(n) ||
    n < 1L ||
    n != as.integer(n)
  ) {
    stop("`n` must be a positive integer.", call. = FALSE)
  }

  with_local_seed(seed, function() {
    mean(rnorm(as.integer(n)))
  })
}
```

A seeded call is now local:

```r
set.seed(100)
expected <- c(runif(1), runif(1))

set.seed(100)
first <- runif(1)

invisible(random_mean(seed = 42L))

second <- runif(1)

identical(c(first, second), expected)
# TRUE
```

The function gets deterministic internal randomness while the caller keeps the stream it already had.

That is much closer to how I expect a library function to behave.

## Reproducibility Without State Leakage

The simplest regression checks are direct.

### Existing state is restored exactly

```r
set.seed(123L)
before <- .Random.seed

invisible(random_mean(seed = 7L))

after <- .Random.seed

identical(after, before)
# TRUE
```

I would use `identical()`, not a numeric tolerance.

The RNG state is discrete program state. If the contract says it is restored, approximate equality is not the claim.

### No state is left behind

This branch deserves its own test.

```r
if (
  exists(
    ".Random.seed",
    envir = .GlobalEnv,
    inherits = FALSE
  )
) {
  rm(".Random.seed", envir = .GlobalEnv)
}

invisible(random_mean(seed = 7L))

exists(
  ".Random.seed",
  envir = .GlobalEnv,
  inherits = FALSE
)
# FALSE
```

This case is easy to forget because most interactive R sessions already have a `.Random.seed` by the time tests are inspected manually.

But it is a different branch of the state-management logic.

If it is wrong, a function that claims to be locally seeded can leak a new global state into a previously clean session.

### No seed means normal stream consumption

The opposite behaviour matters too.

```r
set.seed(321L)
before <- .Random.seed

invisible(random_mean(seed = NULL))

after <- .Random.seed

identical(after, before)
# FALSE
```

A wrapper that *always* restores the caller state would be wrong here.

It would make an apparently random unseeded function repeat the same draws whenever called from the same point in the stream.

Locality should be opt-in through the explicit seed, not imposed on ordinary stochastic execution.

### Same explicit seed means exact replay

```r
first <- random_mean(n = 1_000L, seed = 42L)
second <- random_mean(n = 1_000L, seed = 42L)

identical(first, second)
# TRUE
```

This is the usual reproducibility assertion.

It is now one quarter of the contract rather than the whole contract.

## Error Paths Are Part of the RNG Contract

State restoration that works only when the simulation succeeds is not state restoration.

Suppose the random work throws an error after drawing some values:

```r
set.seed(12L)
before <- .Random.seed

try(
  with_local_seed(99L, function() {
    rnorm(10)
    stop("simulation failed")
  }),
  silent = TRUE
)

after <- .Random.seed

identical(after, before)
# TRUE
```

This is why `on.exit()` is a better primitive than manually restoring the state on the final line of a function.

Statistical code has many ways to exit early:

- failed optimisers,
- singular matrix decompositions,
- non-finite simulated statistics,
- interrupted loops,
- explicit validation errors after stochastic preprocessing.

The global RNG state should not depend on which exit path happened to fire.

## Why `set.seed()` at the Top of Every Function Is a Bad Pattern

It is sometimes defended as a way to make package behaviour stable:

```r
my_bootstrap <- function(x) {
  set.seed(123)
  # ...
}
```

This creates two problems.

First, every call gets the same random sequence whether or not the caller asked for it.

Second, every call resets the global stream.

A stochastic algorithm with hidden fixed seeding is not really random from the caller's perspective. Repeated calls can reuse the same resamples, Monte Carlo draws or initialisations in ways that are not visible in the API.

A seed is part of the experimental design. It should not be an undocumented constant buried inside a function.

A better interface is

```r
my_bootstrap(x, seed = NULL)
```

with documented behaviour for both `NULL` and an explicit integer.

## Why Saving Only the Integer Seed Is Not Enough

Another tempting pattern is:

```r
set.seed(seed)
# do work
set.seed(old_seed)
```

There is no `old_seed` integer that generally represents the caller's current point in the random stream.

The full state is `.Random.seed`, an integer vector encoding both generator information and its current state.

The caller may have executed thousands of random draws since the last explicit `set.seed()` call.

What must be restored is the state itself:

```r
old_seed <- .Random.seed
```

not merely some earlier seed value.

This is an important distinction:

$$
\text{seed}
\neq
\text{current RNG state}.
$$

The seed initializes a stream. The state identifies where we currently are in it.

## The Test Can Be Wrong Too

R adds one more subtlety: arguments are evaluated lazily.

That matters when the object used by an RNG test is itself created by code that changes the seed.

Consider this helper:

```r
make_input <- function() {
  set.seed(1L)
  rnorm(20)
}
```

and a function that accepts an input as a lazy argument:

```r
make_runner <- function(x) {
  function() mean(x)
}
```

This looks harmless:

```r
runner <- make_runner(make_input())
```

But `make_input()` may still be an unevaluated promise. It can be forced later, when `runner()` first needs `x`.

An RNG-state test can therefore do this accidentally:

1. construct a lazy fixture that will call `set.seed()` later,
2. record `.Random.seed`,
3. call the function under test,
4. force the fixture inside that call,
5. observe that the RNG state changed,
6. blame the function under test.

The test has measured its own fixture.

The fix is simple when a helper is meant to capture a fully constructed object:

```r
make_runner <- function(x) {
  force(x)

  function() mean(x)
}
```

The broader lesson is better:

$$
\boxed{
\text{RNG tests must control randomness in the test harness too.}
}
$$

Testing random-state side effects requires thinking about evaluation order, not just values.

## Test the State, Not Only the Result

Many stochastic tests look like this:

```r
set.seed(42)
a <- my_function()

set.seed(42)
b <- my_function()

expect_equal(a, b)
```

That verifies replay.

It says nothing about whether `my_function()` damaged the surrounding stream.

For package-quality testing, I would add explicit state assertions.

Using `testthat`, the four core cases look like this:

```r
test_that("explicit seed restores existing RNG state", {
  set.seed(123L)
  before <- .Random.seed

  invisible(random_mean(seed = 7L))

  expect_identical(.Random.seed, before)
})


test_that("explicit seed leaves no state when none existed", {
  if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    rm(".Random.seed", envir = .GlobalEnv)
  }

  invisible(random_mean(seed = 7L))

  expect_false(
    exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  )
})


test_that("NULL seed advances caller stream", {
  set.seed(321L)
  before <- .Random.seed

  invisible(random_mean(seed = NULL))

  expect_false(identical(.Random.seed, before))
})


test_that("same explicit seed reproduces result", {
  expect_identical(
    random_mean(1_000L, seed = 42L),
    random_mean(1_000L, seed = 42L)
  )
})
```

I would add an error-path test as a fifth assertion whenever the local-seed helper is part of package infrastructure.

## Monte Carlo Replicates Are Part of Reproducibility Too

A seed alone does not define a Monte Carlo procedure.

Suppose a p-value is estimated as

$$
\widehat p
=
\frac{1+K}{B+1},
$$

where $K$ is the number of simulated statistics at least as extreme as the observed statistic.

Then both

$$
\texttt{seed}
$$

and

$$
B
$$

are part of the computational experiment.

Changing $B$ changes the set of possible p-values:

$$
\left\{
\frac{1}{B+1},
\frac{2}{B+1},
\ldots,
1
\right\}.
$$

For a bootstrap confidence interval, the replicate count similarly affects Monte Carlo variability in the estimated quantiles.

So reproducibility metadata should retain at least:

- the seed,
- the number of replicates,
- algorithm settings that affect random draws,
- the software version when exact replay matters.

`set.seed(42)` by itself is not a complete computational specification.

## Seeds Should Be Inputs, Not Provenance Lost in a Script

If stochastic output matters scientifically, I prefer the seed to appear explicitly in the result or experiment record.

For example:

```r
result <- list(
  estimate = estimate,
  interval = interval,
  B = B,
  seed = seed
)
```

That does not mean every end-user print method needs to display the seed prominently.

It means the experiment can be reconstructed without guessing which line in a script happened to initialize the generator three pages earlier.

A random seed is small provenance with unusually high value.

## Parallel Randomness Is a Separate Problem

Everything above concerns one ordinary R process.

Parallel simulation adds another layer.

Naively sending the same seed to several workers can create identical streams. Letting workers inherit state can make results depend on scheduling or process-launch details.

For parallel Monte Carlo, independent reproducible streams should be designed explicitly. In base R that usually points toward the `L'Ecuyer-CMRG` generator family and its stream/substream machinery. Higher-level parallel frameworks often provide their own reproducible seeding contracts.

The principle remains the same:

$$
\text{randomness should be explicit experimental state, not accidental global state.}
$$

A local-seed wrapper solves scope inside one process. It is not a substitute for a parallel RNG design.

## Randomness Is Part of the API

When a statistical function contains a bootstrap, permutation test, Monte Carlo integral, random initialization or simulation step, its RNG behaviour is part of its public interface whether the documentation acknowledges it or not.

Users need to know:

- whether repeated calls are stochastic,
- whether an explicit seed can reproduce a result,
- whether that seed is local to the function,
- whether an unseeded call consumes the current caller stream,
- whether parallel execution changes the guarantee.

Those are API semantics, not implementation trivia.

A function that returns numerically correct output but invisibly resets the global RNG can make the *next* analysis numerically wrong relative to the experiment the user intended to run.

## The Practical Rule

For scripts, notebooks and one-off experiments, this remains perfectly sensible:

```r
set.seed(42)
```

Set the seed at the experiment boundary and make the whole sequence reproducible.

For reusable statistical functions, I would use a different rule:

$$
\boxed{
\text{If a seed is supplied, make it local.}
}
$$

That means snapshotting the caller's state, running the stochastic computation under the requested seed, and restoring the state even if the computation fails.

If no seed is supplied, let the function participate normally in the caller's random stream.

That gives us the property we actually want:

$$
\boxed{
\text{same explicit seed}
\Rightarrow
\text{same result},
\qquad
\text{without changing what happens next.}
}
$$

That is a stronger form of reproducibility than merely calling `set.seed()`.
