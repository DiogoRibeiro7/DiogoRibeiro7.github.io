---
permalink: '/statistics/writing_statistical_software_as_executable_mathematics/'
title: 'Writing Statistical Software as Executable Mathematics'
categories:
- Statistics
- Programming
tags:
- Software Engineering
- Statistical Computing
- Testing
- Mathematical Invariants
- R
- Reproducibility
author_profile: false
seo_title: 'Writing Statistical Software as Executable Mathematics'
seo_description: 'Statistical software should test the identities that connect its procedures, not only example outputs. Write the mathematical contract first, implement it second, and turn the relation into an executable regression test.'
excerpt: >-
  In statistical software, many of the strongest tests are not input-output examples.
  They are equations: test inversion must agree with pointwise decisions, projections
  must reduce to simpler procedures in special cases, and equivalent parameter-grid
  orderings must produce the same inferential object.
summary: >-
  A practical approach to statistical software in which mathematical identities,
  reduction rules, invariances, tie conventions and finite-Monte-Carlo decision
  rules become executable contracts. The goal is to make the test suite verify that
  the implementation still represents the same mathematics after refactoring.
keywords:
- statistical software
- mathematical invariants
- regression testing
- Monte Carlo inference
- test inversion
- reproducibility
classes: wide
date: '2026-09-15'
why_this_exists: >-
  Conventional unit tests often lock isolated examples. Statistical methods usually
  contain stronger relationships between tests, confidence sets, projections and
  transformed parameterizations. Encoding those relationships directly can catch
  errors that every individual function appears to pass in isolation.
evidence: >-
  Motivated by recent statistical-package work in which finite-Monte-Carlo decision
  duality, test inversion, projection reductions, row-order invariance, deterministic
  witness ties and numerical refinement semantics are expressed as explicit
  mathematical/software contracts.
methodology: >-
  Develops generic examples of relational testing: derive identities independently
  from the implementation, construct fixtures that expose equality and boundary
  cases exactly, and require related APIs to commute under reductions and
  transformations rather than comparing only against hard-coded snapshots.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
---

Most software tests have the same basic shape.

Give a function an input.

Check that the output equals the expected value.

For example:

```r
expect_equal(mean(c(1, 2, 3)), 2)
```

There is nothing wrong with that style of test. Statistical software needs ordinary unit tests too.

But many of the strongest tests available to us are not examples of the form

$$
 x \mapsto y.
$$

They are mathematical relations between different parts of the software.

A test produces a p-value and a rejection decision. A confidence set is constructed by inverting those decisions. A projection aggregates pointwise evidence over nuisance values. A scalar transformation should reduce to a coordinate projection when the transformation is the identity. A fast implementation should agree with a simple reference definition. Reordering an arbitrary grid should not change the inferential object.

Those are not isolated outputs.

They are equations.

My preferred approach to statistical software is therefore:

$$
\boxed{
\text{write the mathematical identity first, implement it second, test the identity third.}
}
$$

The test suite then becomes more than a collection of remembered answers.

It becomes executable mathematics.

## An Example Output Is Weaker Than an Identity

Suppose a test statistic is \(T\), a Monte Carlo p-value is \(p\), and the test rejects at level \(\alpha\).

A conventional test might use one fixture and require

```r
expect_equal(result$p_value, 0.04)
expect_true(result$reject)
```

That checks one output.

But if the procedure is defined so that

$$
\text{reject}
\iff
p\le\alpha,
$$

then we possess a stronger statement.

For every valid fixture,

$$
\boxed{
\texttt{reject}
=
\mathbf 1\{p\le\alpha\}.
}
$$

That is the contract I want the test suite to protect.

Why is it stronger?

Because a refactor can change both the p-value and the rejection decision in a mutually inconsistent way while still accidentally preserving the expected result for a handful of stored examples.

A relational test checks whether the pieces still fit together mathematically.

## Decision Duality Should Survive Every Layer

Suppose a confidence set is obtained by inverting pointwise tests.

At parameter value \(\theta\), define

$$
C_{1-\alpha}
=
\{\theta:\text{do not reject }H_0(\theta)\}.
$$

If rejection is defined by

$$
p(\theta)\le\alpha,
$$

then set membership is not an independent design choice.

It follows immediately that

$$
\boxed{
\theta\in C_{1-\alpha}
\iff
p(\theta)>\alpha.
}
$$

Notice the strict inequality.

If the test rejects when

$$
p\le\alpha,
$$

then a point with exactly

$$
p=\alpha
$$

is rejected and therefore excluded from the inverted set.

This is a tiny boundary detail that can easily be lost when two different functions are written months apart.

The code for the test may use

```r
reject <- p_value <= alpha
```

while the code for inversion later uses

```r
accepted <- p_value >= alpha
```

Every ordinary example away from the boundary can pass.

The mathematics is still wrong.

A relational regression test should state the whole chain:

```r
expect_identical(result$reject, result$p_value <= alpha)
expect_identical(result$accepted, !result$reject)
expect_identical(result$accepted, result$p_value > alpha)
```

This is not duplicated testing.

It is testing a theorem connecting three representations of the same decision.

## Finite Monte Carlo Makes Boundary Logic More Important

Monte Carlo p-values often use the plus-one form

$$
\widehat p
=
\frac{1+K}{B+1},
$$

where \(K\) is the number of simulated statistics at least as extreme as the observed statistic and \(B\) is the number of simulated replicates.

Then the attainable p-values lie on the grid

$$
\left\{
\frac{1}{B+1},
\frac{2}{B+1},
\ldots,
1
\right\}.
$$

The rejection rule

$$
\widehat p\le\alpha
$$

may also be represented by a simulated critical value \(c_\alpha\).

If the software reports both quantities, they should satisfy

$$
\boxed{
\widehat p\le\alpha
\iff
T>c_\alpha.
}
$$

The exact convention around ties matters.

A generic empirical quantile does not automatically produce a critical value decision-equivalent to the plus-one p-value for every \((B,\alpha)\).

This is a perfect candidate for executable mathematics because the identity defines what the two outputs mean together.

A test can generate tie-heavy simulated statistics deliberately and verify the relation across many valid replication counts.

## Some Significance Levels May Be Unattainable

Finite simulation produces another useful contract.

The smallest possible plus-one p-value is

$$
\frac{1}{B+1}.
$$

Therefore if

$$
\alpha<\frac{1}{B+1},
$$

rejection is impossible no matter how extreme the observed statistic is.

A well-designed implementation should not hide that fact behind an arbitrary empirical quantile.

The mathematical implication is

$$
\boxed{
\alpha<\frac{1}{B+1}
\Longrightarrow
\text{reject}=\text{FALSE for every }T.
}
$$

That should become a test.

The interesting cases in statistical software are often exactly these edge cases where the mathematics gives a discrete boundary and generic numerical utilities give something merely plausible.

## Confidence-Set Construction Is an Equation

Suppose we evaluate pointwise tests on a finite grid

$$
\mathcal G
=
\{\theta_1,\ldots,\theta_K\}.
$$

The represented inverted set is

$$
\widehat C
=
\{\theta_k\in\mathcal G:\widehat p_k>\alpha\}.
$$

Then the returned accepted values should equal exactly

$$
\{\theta_k:\widehat p_k>\alpha\}.
$$

Again, this sounds obvious.

That is precisely why it should be executable.

The set-building layer can be tested independently of any particular expected set:

```r
expected <- grid[p_values > alpha]
expect_identical(accepted_values, expected)
```

This is more robust than storing one expected vector from one fixture.

If the pointwise p-values legitimately change after a numerical improvement, the relational test remains correct.

It protects the inferential construction rather than yesterday's floating-point output.

## Projection Gives Us Another Contract

Now suppose the parameter is multidimensional,

$$
\theta=(\psi,\lambda),
$$

where \(\psi\) is the target and \(\lambda\) is nuisance.

On a represented finite grid, define a profile p-value

$$
\widehat p_{\mathrm{prof}}(\psi)
=
\max_{r:\psi_r=\psi}\widehat p_r.
$$

The projected set is then

$$
\widehat C_\psi
=
\{\psi:\widehat p_{\mathrm{prof}}(\psi)>\alpha\}.
$$

There are two identities to test:

$$
\boxed{
\widehat p_{\mathrm{prof}}(\psi)
=
\max_{r:\psi_r=\psi}\widehat p_r
}
$$

and

$$
\boxed{
\psi\in\widehat C_\psi
\iff
\widehat p_{\mathrm{prof}}(\psi)>\alpha.
}
$$

Those equations are a much better specification than a prose sentence saying that the function "projects a confidence set."

They tell us exactly what every returned value must mean.

## Reduction Identities Are Extremely Powerful

A complicated API often contains special cases that should reduce exactly to simpler APIs.

Those reductions are some of the best tests we can write because they compare independent code paths.

Suppose a general coordinate-projection function operates on a multidimensional parameter grid.

If the model has only one parameter, there is no nuisance dimension.

Therefore projection should reduce to ordinary inversion:

$$
\boxed{
\operatorname{Project}_{\theta}
\{\widehat C(\theta)\}
=
\widehat C(\theta).
}
$$

The software should satisfy the same identity.

Likewise, suppose a scalar projection accepts a transformation

$$
g(\theta).
$$

If we choose

$$
g(\theta)=\theta_j,
$$

the arbitrary scalar projection should reduce to coordinate projection onto parameter \(j\).

So we obtain another contract:

$$
\boxed{
\operatorname{ScalarProject}_{g(\theta)=\theta_j}
=
\operatorname{CoordinateProject}_j.
}
$$

A regression test can compare:

- represented target values,
- profile p-values,
- decisions,
- accepted values,
- connected components,
- witness semantics.

No hard-coded numerical answer is necessary.

Two independently implemented APIs become mutual checks on each other.

## Think in Commutative Diagrams

There is a useful mathematical way to think about this.

Suppose two paths through the software are supposed to represent the same operation.

Then we want the diagram to commute.

For example,

$$
\begin{array}{ccc}
\Theta & \xrightarrow{\text{pointwise tests}} & \{p_r\}\\
\downarrow g & & \downarrow \max_{g(\theta_r)=\psi}\\
\Psi & \xrightarrow{\text{projection}} & \{p_{\mathrm{prof}}(\psi)\}
\end{array}
$$

The two paths should lead to the same result.

Software engineering usually talks about unit tests and integration tests.

For mathematical software, I also want **commutativity tests**.

If two mathematically equivalent routes through the API disagree, at least one route is wrong.

## Invariance Tests Encode What Must Not Matter

Some mathematical statements say that a transformation should leave the answer unchanged.

These are equally valuable.

Suppose the rows of a finite parameter grid are merely a representation of a set.

Then permuting those rows should not change the inferential object.

If \(\pi\) is a row permutation,

$$
\mathcal G
=
\{\theta_1,\ldots,\theta_K\}
$$

and

$$
\pi(\mathcal G)
=
\{\theta_{\pi(1)},\ldots,\theta_{\pi(K)}\},
$$

we should have

$$
\boxed{
\widehat C(\mathcal G)
=
\widehat C\{\pi(\mathcal G)\}
}
$$

after canonical ordering of the result.

The same applies to projected p-values and accepted target values.

This tests something fundamentally different from an ordinary fixture.

It asks whether the implementation depends accidentally on representation order.

## But Not Everything Must Be Invariant

A good invariant test also requires knowing what is allowed to change.

Suppose the projection object reports a witness row attaining

$$
\max_{r:\psi_r=\psi}p_r.
$$

If the input grid is permuted, the numeric row index of that witness can change even though the inferential result does not.

So the correct contract is not

$$
\text{witness row number is invariant}.
$$

It is

$$
\boxed{
\text{the witness belongs to the right group and attains the group maximum.}
}
$$

This distinction matters.

Overly rigid tests can be just as misleading as weak tests. They freeze incidental representation choices instead of protecting the mathematics.

## Deterministic Tie-Breaking Can Be Part of the API

Sometimes the mathematical solution is not unique.

Suppose two nuisance rows satisfy

$$
p_1=p_2
=
\max_r p_r.
$$

The profile p-value is unambiguous.

The witness is not.

If the software exposes a single witness row, it needs a deterministic convention.

For example:

$$
\boxed{
\text{choose the first maximizing row in supplied grid order.}
}
$$

That rule is not a theorem about statistics.

It is a theorem about the software contract.

Once documented, it should be tested with an **exact tie fixture** rather than hoping two floating-point calculations happen to tie.

A good fixture can deliberately make the nuisance coordinate irrelevant so several rows produce mathematically identical tests.

Then the tie is structural, not accidental.

## Construct Fixtures From the Mathematics

This is a general principle.

If we want to test equality, build a fixture where equality holds exactly by construction.

If we want to test sign symmetry, generate one dataset and its negative.

If we want to test translation invariance, add a common constant.

If we want to test permutation invariance, permute the same represented set.

If we want to test a reduction identity, choose the transformation that makes the general case algebraically identical to the simple case.

If we want to test an empty selected set, construct moments that are all comfortably slack.

The mathematics should design the test data.

Random fixtures are useful for stress testing, but exact structural fixtures tell us why a failure occurred.

## Numerical Refinement Needs Its Own Mathematics

Even numerical routines have useful executable contracts.

Suppose a confidence-set boundary is refined by bisection inside a bracket

$$
[a,b]
$$

whose endpoints have opposite decisions.

If convergence is defined by

$$
b-a\le\varepsilon,
$$

then the software should report convergence only when that condition is actually true.

Now consider adjacent representable floating-point numbers.

There may be no machine number strictly between them.

Then the mathematical midpoint exists in \(\mathbb R\), but the numerical midpoint satisfies

$$
\operatorname{fl}\left(\frac{a+b}{2}\right)
\in\{a,b\}.
$$

The algorithm cannot subdivide the bracket further.

That is not convergence to tolerance.

It is a numerical stall.

The contract should distinguish

$$
\boxed{
\text{stalled}
\neq
\text{converged}.
}
$$

This is another example where precise vocabulary becomes executable behaviour.

## A Reference Implementation Can Be a Mathematical Specification

Optimized statistical code is often much harder to inspect than the formula it implements.

Suppose a direct method computes a quantity in \(O(n^2)\) memory while an optimized derivation computes it in \(O(n)\).

The direct version may be too slow for production but excellent as a reference.

Then test

$$
\boxed{
F_{\mathrm{fast}}(x)
\approx
F_{\mathrm{reference}}(x)
}
$$

across carefully chosen inputs.

I like retaining such simple reference implementations when they are small enough.

They are executable definitions.

The production code can be aggressively optimized while the test suite continues to ask whether the optimization represents the same mathematics.

## Exact Equality and Numerical Equivalence Are Different Contracts

Not every identity should use the same comparison operator.

Some relations are discrete and should be exact:

- rejection decisions,
- accepted grid membership,
- names,
- dimensions,
- deterministic ordering,
- RNG-state restoration.

Others involve reassociated floating-point arithmetic and should be tested numerically:

$$
|x_{\mathrm{fast}}-x_{\mathrm{ref}}|
\le
\tau.
$$

The tolerance \(\tau\) should come from numerical reasoning, not convenience.

A test that uses `all.equal()` everywhere can hide meaningful errors.

A test that demands bit identity after a stable algebraic optimization can reject perfectly valid code.

The mathematical contract tells us which one is appropriate.

## Documentation and Tests Should Share the Same Equation

One practice I find especially useful is to write the defining identity in the documentation and then mirror that identity in the regression test.

For example, documentation says

$$
\text{accepted}
\iff
p>\alpha.
$$

The test says

```r
expect_identical(accepted, p_value > alpha)
```

Documentation says

$$
p_{\mathrm{prof}}(\psi)
=
\max_{r:\psi_r=\psi}p_r.
$$

The test recomputes that maximum independently.

Documentation says identity scalar projection reduces to coordinate projection.

The test calls both APIs and compares the inferential fields.

This creates a useful triangle:

$$
\boxed{
\text{theory}
\leftrightarrow
\text{documentation}
\leftrightarrow
\text{tests}
}
$$

If one corner changes, the mismatch becomes visible.

## Do Not Test Only the Happy Interior

Mathematical contracts often become most informative at boundaries.

Test

$$
p=\alpha.
$$

Test a tied simulated critical value.

Test one represented grid point.

Test an empty set.

Test all points accepted.

Test no nuisance coordinates.

Test exact duplicate maximizers.

Test adjacent floating-point numbers.

Test a significance level unattainable at the chosen Monte Carlo replication count.

Those cases are not pathological distractions.

They are where conventions become observable.

## A Statistical API Is a Network of Identities

As a package grows, it is tempting to think of it as a list of exported functions.

I find another mental model more useful.

Think of it as a graph.

The nodes are procedures:

- statistic construction,
- pointwise testing,
- p-value calculation,
- critical values,
- inversion,
- refinement,
- projection,
- transformations,
- summaries.

The edges are mathematical relationships:

$$
\text{p-value}
\leftrightarrow
\text{decision},
$$

$$
\text{decision}
\leftrightarrow
\text{set membership},
$$

$$
\text{joint set}
\leftrightarrow
\text{projection},
$$

$$
\text{general API}
\leftrightarrow
\text{special-case API},
$$

$$
\text{reference formula}
\leftrightarrow
\text{optimized computation}.
$$

Bugs often appear on the edges rather than inside the nodes.

Each individual function may look locally reasonable while two functions disagree about what the same mathematical quantity means.

Relational tests target those edges directly.

## Why This Matters More in Statistical Software

Ordinary application software can often be specified by examples and business rules.

Statistical software makes claims about mathematical objects.

A function may claim to return a level-\(1-\alpha\) inverted set. Another may claim to project it. Another may report the p-value supporting the projected decision.

If those objects do not satisfy their defining relations, the problem is not merely an implementation bug.

The software is representing a different statistical procedure from the one its documentation describes.

That is why I want equations in the tests.

## The Workflow I Prefer

When implementing a statistical method, I now like to proceed in roughly this order.

First, write the mathematical object.

For example,

$$
C_{1-\alpha}
=
\{\theta:p(\theta)>\alpha\}.
$$

Second, identify relations to objects already implemented.

For example,

$$
\theta\in C_{1-\alpha}
\iff
\neg\operatorname{reject}(\theta).
$$

Third, identify reductions and invariances.

For example,

$$
\operatorname{Project}_{\theta}(C)=C
$$

in one dimension.

Fourth, decide which conventions are needed where mathematics is non-unique.

For example, first-max witness selection under ties.

Fifth, implement the algorithm.

Sixth, write regression tests that reconstruct the relations independently.

That order matters because it makes the test derive from the method rather than from whatever the first implementation happened to do.

## Tests Should Not Merely Ratify the Code

A weak testing workflow is:

1. write the implementation,
2. run it once,
3. copy the output into the test,
4. declare regression coverage.

That can be useful for detecting changes, but it does not tell us whether the original output was correct.

A stronger workflow is:

1. derive a relation independently,
2. construct a fixture where the relation is informative,
3. compute each side through different code paths,
4. require them to agree.

The test then has an epistemic source outside the implementation itself.

## The Best Regression Test Often Looks Like a Theorem

Some of my favorite statistical tests can be written almost directly as mathematics:

$$
\boxed{
\operatorname{reject}
\iff
p\le\alpha
\iff
T>c_\alpha
}
$$

$$
\boxed{
\theta\in C_{1-\alpha}
\iff
p(\theta)>\alpha
}
$$

$$
\boxed{
p_{\mathrm{prof}}(\psi)
=
\max_{r:\psi_r=\psi}p_r
}
$$

$$
\boxed{
\operatorname{ScalarProject}_{g(\theta)=\theta_j}
=
\operatorname{CoordinateProject}_j
}
$$

$$
\boxed{
F(\mathcal G)
=
F\{\pi(\mathcal G)\}
}
$$

for every permutation \(\pi\) when row order is irrelevant.

These equations are concise because the mathematics has already done most of the specification work.

## The Deeper Point

Statistical software is not just code that happens to perform mathematics.

At its best, the architecture can reflect the mathematics itself.

Definitions become functions.

Theorems become relations between functions.

Invariances become metamorphic tests.

Special cases become reduction tests.

Non-uniqueness becomes an explicit deterministic convention.

Numerical limits become separate states rather than silently changing inferential meaning.

Then a refactor does not merely have to reproduce a collection of old numbers.

It has to preserve the mathematical structure of the method.

That is the standard I want from statistical software:

$$
\boxed{
\text{the implementation should not merely run the mathematics; the test suite should enforce it.}
}
