---
permalink: '/mathematics/preconditioning_changes_the_problem_your_iterative_solver_sees/'
title: 'Preconditioning Changes the Problem Your Iterative Solver Sees'
date: '2026-05-28'
categories:
- Mathematics
tags:
- Numerical Linear Algebra
- Krylov Methods
- Preconditioning
- Sparse Matrices
- Iterative Solvers
author_profile: false
classes: wide
seo_title: 'Preconditioning Changes the Problem Your Iterative Solver Sees'
seo_description: 'Iterative solvers depend on spectral geometry, not only on the equation Ax=b. Preconditioning can transform that geometry and reduce iteration counts dramatically without changing the exact solution.'
seo_type: article
excerpt: >-
  Large sparse linear systems are often limited less by arithmetic than by
  conditioning and spectral geometry. A preconditioner changes the system seen
  by the iterative method so that the same exact solution can become far easier
  to compute.
summary: >-
  This article develops preconditioning from the geometry of Krylov methods.
  Conjugate gradients are analysed through their polynomial error representation
  and classical condition-number bound, with the one-dimensional Poisson matrix
  showing how mesh refinement drives the condition number like O(n^2). A second
  exact example shows that Jacobi scaling can leave conditioning unchanged, while
  an appropriate approximate inverse can transform the spectrum completely. The
  article then treats GMRES, nonnormality, left versus right preconditioning,
  residual-versus-error stopping criteria, incomplete factorizations, multigrid,
  and the distinction between improving iterative convergence and changing the
  conditioning of the original scientific problem.
keywords:
- preconditioning
- conjugate gradient
- GMRES
- Krylov subspace methods
- sparse linear systems
- condition number
why_this_exists: >-
  Discussions of iterative solvers often focus on algorithm names while treating
  preconditioning as an implementation detail. In large scientific systems the
  preconditioner is frequently the main determinant of computational viability
  because it changes the spectrum and geometry on which the Krylov iteration
  operates.
evidence: >-
  Exact eigenvalues of the finite-difference Poisson matrix, classical
  conjugate-gradient convergence bounds, an exact nonnormal GMRES
  counterexample, and standard theory for incomplete factorization and multigrid
  preconditioning.
methodology: >-
  Begin with the polynomial interpretation of Krylov methods, derive the
  condition-number bound for conjugate gradients, then analyse mesh refinement
  for the discrete Poisson operator. Introduce preconditioning as an approximate
  inverse, distinguish useful from useless scaling, and use a two-by-two Jordan
  example to show why eigenvalues alone do not determine GMRES convergence.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/kernel_math.webp
  og_image: /assets/images/kernel_math.webp
  overlay_image: /assets/images/kernel_math.webp
  show_overlay_excerpt: false
  teaser: /assets/images/kernel_math.webp
  twitter_image: /assets/images/kernel_math.webp
---

<!--
Development contract
Question: Why can two algebraically equivalent linear systems require radically different computational effort?
Claim: Krylov convergence depends on the geometry of the operator seen by the iteration, including eigenvalue distribution, conditioning and nonnormality. Preconditioning can transform that geometry while leaving the exact solution unchanged.
Counterclaim: Condition number alone is not a complete predictor of convergence, especially for nonsymmetric or nonnormal matrices, and a theoretically strong preconditioner can be computationally useless if setup, memory or application cost dominates.
Evidence object: Exact one-dimensional Poisson spectrum, conjugate-gradient convergence bound, diagonal-scaling counterexample, exact preconditioned diagonal system, and a nonnormal two-by-two GMRES example with repeated eigenvalue one.
Failure case: Comparing solvers only by wall-clock time on one matrix, treating residual reduction as equivalent to error reduction, or assuming that a preconditioner which makes the algebraic solve easy has removed ill-posedness from the underlying inverse problem.
Reader payoff: Diagnose iterative solvers from matrix structure, understand what a preconditioner is actually changing, and judge preconditioners by total cost rather than by iteration count alone.
Exclusions: A complete numerical-linear-algebra textbook, a solver API tutorial, and a repetition of regularization theory for inverse problems.
-->

A linear system

$$
Ax=b
$$

looks finished once the matrix and right-hand side have been written down. In exact arithmetic, if $A$ is nonsingular, there is one solution

$$
x=A^{-1}b.
$$

From that algebraic perspective, solving the system is merely a matter of carrying out enough arithmetic. Large scientific computations reveal a different picture. The same exact solution can be trivial to obtain in one representation and painfully slow in another, not because the mathematics of the solution changed, but because the numerical algorithm sees different spectral geometry.

This is the setting in which preconditioning becomes central. A preconditioner does not normally change the physical model or the exact solution one wants. It changes the operator presented to the iterative method. The aim is to replace a system whose error components decay at wildly different rates by one in which those components are better balanced, clustered or otherwise easier for the Krylov space to approximate.

That distinction matters because large sparse systems rarely permit us to form an inverse explicitly. The cost and storage of a dense factorization can be prohibitive, while the original sparse matrix may support cheap matrix-vector products. Krylov methods exploit this structure by building approximations from repeated applications of $A$. Their success depends on how effectively low-degree polynomials can approximate the inverse action over the relevant spectrum.

The preconditioner is therefore not a cosmetic acceleration layer. In many PDE, optimization, least-squares and inverse problems, it is the part of the algorithm that determines whether the computation is feasible at all.

## Krylov methods approximate the inverse by polynomials

Start with an initial guess $x_0$ and residual

$$
r_0=b-Ax_0.
$$

The $k$th Krylov subspace is

$$
\mathcal K_k(A,r_0)
=
\operatorname{span}
\{
r_0,
Ar_0,
A^2r_0,
\ldots,
A^{k-1}r_0
\}.
$$

An iterative Krylov method searches for an approximation of the form

$$
x_k
=
x_0
+
v_k,
\qquad
v_k\in\mathcal K_k(A,r_0).
$$

Equivalently, the error can often be written as

$$
e_k
=
x_\star-x_k
=
p_k(A)e_0,
$$

where $x_\star$ is the exact solution and $p_k$ is a polynomial satisfying a normalization such as

$$
p_k(0)=1.
$$

The solver is therefore constructing a polynomial that is small where the matrix acts most strongly on the current error.

For a symmetric positive-definite matrix, conjugate gradients chooses the approximation that minimizes the error in the energy norm

$$
\|e\|_A
=
\sqrt{
e^\top A e
}.
$$

A classical bound is

$$
\frac{
\|e_k\|_A
}{
\|e_0\|_A
}
\le
2
\left(
\frac{
\sqrt\kappa-1
}{
\sqrt\kappa+1
}
\right)^k,
$$

where

$$
\kappa
=
\kappa_2(A)
=
\frac{
\lambda_{\max}(A)
}{
\lambda_{\min}(A)
}
$$

for symmetric positive-definite $A$.

The bound explains the usual statement that conjugate-gradient convergence worsens as the condition number grows. If

$$
\kappa
\approx1,
$$

the contraction factor is small. If

$$
\kappa
$$

is enormous, the factor approaches one and the worst-case bound deteriorates.

The bound is useful and deliberately pessimistic. CG can converge much faster when eigenvalues are clustered, when the initial error has little component in difficult eigendirections, or when a small number of outlying eigenvalues can be represented accurately by a low-degree polynomial. Two matrices with the same condition number can therefore produce different convergence histories.

This already hints at the real objective of preconditioning. Reducing the condition number can help, but the deeper aim is to transform the spectrum and invariant geometry into a form that the Krylov polynomial can approximate efficiently.

## Mesh refinement can make the algebraic problem progressively harder

Consider the one-dimensional Poisson equation

$$
-u''(x)
=
f(x),
\qquad
0<x<1,
$$

with homogeneous Dirichlet conditions

$$
u(0)=u(1)=0.
$$

Using $n$ interior grid points with spacing

$$
h
=
\frac{
1
}{
n+1
},
$$

the standard second-order finite-difference discretization produces

$$
A_n
=
\frac{
1
}{
h^2
}
\begin{pmatrix}
2 & -1 \\
-1 & 2 & -1 \\
& \ddots & \ddots & \ddots \\
&& -1 & 2 & -1\\
&&& -1 & 2
\end{pmatrix}.
$$

The eigenvalues are known exactly:

$$
\lambda_j
=
\frac{
4
}{
h^2
}
\sin^2
\left(
\frac{
j\pi
}{
2(n+1)
}
\right),
\qquad
j=1,\ldots,n.
$$

Therefore,

$$
\lambda_{\min}
=
\frac{
4
}{
h^2
}
\sin^2
\left(
\frac{
\pi
}{
2(n+1)
}
\right),
$$

while

$$
\lambda_{\max}
=
\frac{
4
}{
h^2
}
\cos^2
\left(
\frac{
\pi
}{
2(n+1)
}
\right).
$$

The condition number is

$$
\kappa_2(A_n)
=
\cot^2
\left(
\frac{
\pi
}{
2(n+1)
}
\right).
$$

For large $n$,

$$
\cot z
\sim
\frac1z,
$$

so

$$
\kappa_2(A_n)
\sim
\frac{
4(n+1)^2
}{
\pi^2
}.
$$

The matrix becomes increasingly ill-conditioned as the grid is refined, even though the continuous differential equation itself has not changed.

A few exact values show the scale:

| Interior points $n$ | $\kappa_2(A_n)$ |
| ---: | ---: |
| 20 | 178.1 |
| 50 | 1053.5 |
| 100 | 4133.6 |
| 200 | 16373.2 |
| 500 | 101726.2 |

The generic CG bound therefore deteriorates with mesh size. For

$$
n=100,
$$

we have approximately

$$
\kappa
=
4133.6
$$

and contraction factor

$$
\frac{
\sqrt\kappa-1
}{
\sqrt\kappa+1
}
\approx
0.96937.
$$

To make the bound itself smaller than

$$
10^{-6},
$$

one would need roughly

$$
467
$$

iterations according to the inequality.

That number exceeds the matrix dimension. In exact arithmetic, CG terminates in at most $n=100$ steps because there are at most one hundred independent eigencomponents to eliminate. The contradiction is only apparent. The condition-number formula is an upper bound valid for every SPD matrix with that condition number; it is not a prediction of the exact iteration count for this particular spectrum.

This is a useful warning against treating

$$
\kappa(A)
$$

as a complete performance metric. It captures worst-case spectral spread, but not the distribution of eigenvalues within that interval.

For the two-dimensional Poisson problem on an $n\times n$ grid, the dimension is of order

$$
n^2,
$$

while the condition number still grows like

$$
O(n^2).
$$

Unpreconditioned iteration counts therefore grow with spatial resolution even though each sparse matrix-vector multiplication remains relatively cheap. The discretization becomes more accurate and the algebraic solve becomes harder at the same time.

A good preconditioner tries to break this coupling.

## A preconditioner should be easy to invert and close enough to the operator

For left preconditioning, replace

$$
Ax=b
$$

by

$$
M^{-1}Ax
=
M^{-1}b.
$$

The exact solution is unchanged as long as $M$ is nonsingular. The iterative method, however, now sees

$$
M^{-1}A
$$

instead of $A$.

One should almost never form

$$
M^{-1}
$$

explicitly. Applying the preconditioner means solving

$$
Mz=r
$$

for $z$ whenever the algorithm needs the action of $M^{-1}$ on a vector $r$.

An idealized choice would be

$$
M=A.
$$

Then

$$
M^{-1}A
=
I,
$$

and the preconditioned system is solved immediately. This is useless if solving with $M=A$ costs exactly as much as solving the original system.

The preconditioning problem is therefore a trade-off:

$$
M
\approx
A
$$

well enough to improve the iteration, while systems with $M$ remain much cheaper than systems with $A$.

A trivial diagonal example makes the spectral effect exact. Let

$$
A
=
\begin{pmatrix}
1 & 0 & 0\\
0 & 10^2 & 0\\
0 & 0 & 10^4
\end{pmatrix}.
$$

Then

$$
\kappa_2(A)
=
10^4.
$$

Take

$$
M
=
\operatorname{diag}(A)
=
A.
$$

The preconditioned operator is

$$
M^{-1}A
=
I,
$$

so

$$
\kappa_2(M^{-1}A)
=
1.
$$

The system has not acquired a new solution. The coordinate directions have simply been rescaled so that the iterative method sees equal stiffness in every direction.

This is why diagonal scaling can be effective for matrices whose difficulty is dominated by inconsistent units or large differences in coefficient magnitude.

The same method can also do almost nothing. For the finite-difference Poisson matrix,

$$
\operatorname{diag}(A_n)
=
\frac{2}{h^2}I.
$$

Jacobi preconditioning therefore gives

$$
M^{-1}A_n
=
\frac12
\begin{pmatrix}
2 & -1\\
-1 & 2 & -1\\
& \ddots & \ddots & \ddots\\
&&-1&2&-1\\
&&&-1&2
\end{pmatrix}.
$$

Every eigenvalue is scaled by the same constant. The condition number is unchanged:

$$
\kappa_2(M^{-1}A_n)
=
\kappa_2(A_n).
$$

The example is valuable because it removes a common misconception. A method called a preconditioner is not useful merely because it is cheap or standard. It must transform the difficult directions of the actual matrix.

## Incomplete factorizations approximate the solve rather than only the scaling

If

$$
A
$$

is symmetric positive definite, a complete Cholesky factorization gives

$$
A
=
LL^\top.
$$

Using this exact factorization as a preconditioner would again solve the system essentially directly. The problem is fill-in: even when $A$ is sparse, the factor $L$ can contain many nonzeros absent from the original matrix.

Incomplete Cholesky restricts the factorization,

$$
A
\approx
\tilde L\tilde L^\top,
$$

by dropping selected fill entries or using a prescribed sparsity pattern. The preconditioner becomes

$$
M
=
\tilde L\tilde L^\top.
$$

Application requires two sparse triangular solves rather than a full solve with $A$.

For nonsymmetric matrices, incomplete LU plays an analogous role,

$$
A
\approx
\tilde L\tilde U.
$$

The central design choice is how much fill to retain. More fill generally gives a better approximation to $A$ and fewer Krylov iterations, but increases setup cost, memory, and the cost of every preconditioner application.

The best preconditioner is therefore rarely the one with the smallest iteration count.

Suppose preconditioner $M_1$ requires 200 iterations at cost 1 unit each, while $M_2$ requires 40 iterations at cost 8 units each after a large setup. The second method has a much better iteration count and may be slower overall.

This becomes even more important in repeated solves. If thousands of right-hand sides share the same matrix, a costly preconditioner setup can be amortized. If the matrix changes every timestep, the setup may dominate.

The unit of comparison is total solution cost for the workload, not one convergence plot.

## Multigrid attacks the error at the scale where it lives

Poisson-type systems reveal a limitation of purely local relaxation methods. Jacobi or Gauss-Seidel smoothing often removes high-frequency error components quickly while leaving low-frequency components to decay slowly.

Multigrid methods exploit the observation that an error that is smooth on a fine grid looks less smooth on a coarser grid. A typical cycle therefore combines:

1. smoothing on the fine grid,
2. restriction of the residual to a coarser grid,
3. solution or approximation of the coarse-grid error,
4. interpolation back to the fine grid,
5. additional smoothing.

The result is a hierarchy that attacks different error frequencies at the scales where they are easiest to represent.

For elliptic PDEs, an effective multigrid preconditioner can produce a preconditioned condition number that remains bounded or grows only mildly as the mesh is refined. This changes the computational scaling qualitatively. Instead of requiring more and more Krylov iterations on finer grids, iteration counts can become approximately mesh independent.

This is one reason multigrid is so important in large PDE calculations. The goal is not merely to shave a constant factor from a solver. It is to prevent refinement from making the linear solve asymptotically harder.

Geometric multigrid uses an explicit hierarchy of meshes. Algebraic multigrid constructs coarse spaces from the matrix itself, which is useful when the underlying geometry is irregular or unavailable.

Domain decomposition uses a related structural idea. Split the problem into subdomains, solve local problems approximately or exactly, and combine them through additive or multiplicative corrections. Schwarz methods, block preconditioners and substructuring approaches can exploit physical or algebraic locality while supporting parallel computation.

A strong preconditioner often understands the operator at least as well as the outer Krylov method does.

## GMRES exposes why eigenvalues are not enough

For nonsymmetric systems, GMRES chooses

$$
x_k
\in
x_0+\mathcal K_k(A,r_0)
$$

to minimize the Euclidean residual norm

$$
\|b-Ax_k\|_2.
$$

When $A$ is normal, spectral information gives substantial insight because the eigenvectors form an orthogonal basis. For nonnormal matrices, eigenvectors can be highly nonorthogonal and eigenvalues alone may say very little about transient behaviour.

A two-by-two example makes this exact.

First consider

$$
A_1
=
I.
$$

All eigenvalues are one. Starting from

$$
x_0=0
$$

with right-hand side

$$
b
=
\begin{pmatrix}
0\\
1
\end{pmatrix},
$$

GMRES solves the system in one step because

$$
A_1b=b.
$$

Now consider

$$
A_K
=
\begin{pmatrix}
1 & K\\
0 & 1
\end{pmatrix}.
$$

This matrix has the same eigenvalues:

$$
\lambda_1
=
\lambda_2
=
1.
$$

It is strongly nonnormal when

$$
K
$$

is large.

After one GMRES step, the approximation lies in

$$
\operatorname{span}\{b\},
$$

so write

$$
x_1
=
\alpha b
=
\begin{pmatrix}
0\\
\alpha
\end{pmatrix}.
$$

The residual is

$$
r_1
=
b-A_Kx_1
=
\begin{pmatrix}
-K\alpha\\
1-\alpha
\end{pmatrix}.
$$

GMRES chooses $\alpha$ to minimize

$$
K^2\alpha^2
+
(1-\alpha)^2.
$$

Differentiating gives

$$
\alpha^\star
=
\frac{
1
}{
K^2+1
}.
$$

The residual norm becomes

$$
\|r_1\|_2
=
\frac{
K
}{
\sqrt{
K^2+1
}
}.
$$

For

$$
K=100,
$$

this is approximately

$$
0.99995.
$$

One GMRES step has made almost no progress even though every eigenvalue is exactly one.

For

$$
A_1=I,
$$

the same eigenvalues produced exact convergence in one step.

Both two-dimensional systems converge in at most two GMRES steps in exact arithmetic, so the example is not intended as a difficult large-scale problem. Its purpose is sharper: eigenvalue location alone does not determine GMRES convergence for nonnormal matrices.

Pseudospectra, field-of-values information, eigenvector conditioning and polynomial approximation on nonnormal operators become relevant. This is why a preconditioner that clusters eigenvalues beautifully can still perform poorly on a highly nonnormal system.

The condition number story that works cleanly for SPD conjugate gradients becomes only one part of the nonsymmetric problem.

## Left and right preconditioning change what residual is minimized

Left preconditioning solves

$$
M^{-1}Ax
=
M^{-1}b.
$$

The residual of the transformed system is

$$
M^{-1}(b-Ax).
$$

If GMRES is applied directly to this system, it minimizes the norm of the preconditioned residual,

$$
\|M^{-1}r\|_2,
$$

not necessarily the original residual norm

$$
\|r\|_2.
$$

Right preconditioning writes

$$
AM^{-1}y
=
b,
$$

then recovers

$$
x
=
M^{-1}y.
$$

GMRES then minimizes the original residual

$$
b-AM^{-1}y
=
b-Ax.
$$

The two formulations have the same exact solution for $x$ under the usual nonsingularity assumptions, but their finite-iteration residual histories and Krylov spaces differ.

This distinction matters when stopping criteria are based on the physical residual. A small preconditioned residual need not imply that

$$
\|b-Ax\|_2
$$

is small if

$$
M^{-1}
$$

distorts the norm strongly.

Symmetric positive-definite preconditioning for CG requires additional structure. One typically works with an SPD preconditioner and an equivalent symmetrically transformed system such as

$$
M^{-1/2}AM^{-1/2}y
=
M^{-1/2}b,
$$

with

$$
x
=
M^{-1/2}y.
$$

Implementations avoid forming square roots explicitly, but the symmetric transformation explains why the preconditioned operator remains suitable for CG theory.

A preconditioner must therefore be compatible with the outer solver. An arbitrary nonsymmetric approximate inverse can destroy the symmetry or definiteness required by conjugate gradients.

## Residual is not error

Iterative solvers usually monitor the residual

$$
r_k
=
b-Ax_k
$$

because it is cheap to compute. The true error is

$$
e_k
=
x_\star-x_k.
$$

They are connected by

$$
Ae_k
=
r_k,
$$

so

$$
e_k
=
A^{-1}r_k.
$$

A small residual does not automatically imply a small solution error in every norm.

For a nonsingular matrix,

$$
\|e_k\|
\le
\|A^{-1}\|
\|r_k\|.
$$

If

$$
\|A^{-1}\|
$$

is large, a residual that looks numerically tiny can still correspond to substantial solution error.

Relative perturbation bounds make the same point. Roughly, for a well-scaled linear system,

$$
\frac{
\|\delta x\|
}{
\|x\|
}
\lesssim
\kappa(A)
\frac{
\|\delta b\|
}{
\|b\|
}.
$$

Conditioning describes sensitivity of the mathematical problem to perturbations. Convergence describes how efficiently an algorithm approaches the solution of the finite system.

The two interact but are not the same concept.

A preconditioner can make

$$
M^{-1}A
$$

easy for a Krylov method while the original map

$$
b\mapsto x
$$

remains sensitive. The preconditioner has accelerated the algebraic solve. It has not created information that was absent from the original problem.

This is particularly important in inverse problems. If a discretized inverse operator is ill-conditioned because the scientific inverse is unstable, an algebraic preconditioner may let us solve the discretized normal equations rapidly and accurately. It does not make the recovered parameter scientifically stable with respect to measurement noise.

Regularization changes the inferential problem. Preconditioning changes the numerical representation used to solve it.

Confusing the two can lead to a dangerously fast computation of a statistically unstable answer.

## Stopping criteria should be tied to the problem scale

A common stopping rule is

$$
\frac{
\|r_k\|_2
}{
\|b\|_2
}
<
\varepsilon.
$$

This is useful but not universal.

If the equations contain quantities with very different physical units or scales, the Euclidean residual may overemphasize some equations. If the matrix is badly scaled, normwise residual reduction may not correspond to small componentwise backward error.

Backward error asks a different question: how much would the data have to change for the computed solution to be exact? For the right-hand side alone,

$$
(A)(x_k)
=
b-r_k,
$$

so the residual itself is the perturbation in $b$ that makes $x_k$ exact.

A relative backward-error measure can take a form such as

$$
\eta_k
=
\frac{
\|r_k\|
}{
\|A\|\|x_k\|+\|b\|
}.
$$

A small backward error says the algorithm has solved a nearby linear system accurately. Whether the solution is close to the desired $x_\star$ still depends on conditioning.

For PDE solvers, the linear algebra tolerance should also be compared with discretization error. Solving the linear system to machine precision is wasteful if the finite-difference or finite-element approximation itself is only accurate to

$$
10^{-4}.
$$

For nonlinear Newton-Krylov methods, oversolving early linearizations can be similarly wasteful. Inexact Newton methods deliberately relax the inner linear tolerance when the outer nonlinear iterate is still far from convergence.

The correct tolerance is part of the overall error budget, not an arbitrary constant copied from a solver example.

## A good preconditioner preserves scalability, not merely a small benchmark time

Performance comparisons can be misleading when they report only one matrix size.

Suppose unpreconditioned CG takes 100 iterations on one grid and a preconditioned method takes 20. That is useful information. It does not tell us what happens after the mesh is refined by a factor of ten.

The stronger question is how iteration count scales with problem size.

For a sequence of discretizations

$$
A_hx_h=b_h,
$$

an effective PDE preconditioner aims to keep the spectrum of

$$
M_h^{-1}A_h
$$

uniformly well behaved as

$$
h\to0.
$$

If the number of Krylov iterations remains roughly constant under refinement, the method has achieved mesh-independent convergence.

Total complexity also depends on matrix-vector cost, preconditioner setup, memory traffic, communication and parallel synchronization. On modern hardware, sparse triangular solves can be harder to parallelize efficiently than sparse matrix-vector multiplication. A preconditioner that is ideal in flop count can scale poorly across distributed memory.

This is one reason multigrid, block-Jacobi, additive Schwarz and approximate sparse inverse methods are often evaluated partly through parallel structure rather than only through iteration counts.

The matrix structure matters too. PDE saddle-point systems benefit from block preconditioners derived from Schur complements. Optimization KKT systems require different approximations from scalar elliptic operators. Graph Laplacians, covariance matrices and nonsymmetric transport problems each bring different spectral and sparsity patterns.

There is no universal preconditioner because there is no universal matrix geometry.

## Preconditioning is mathematical modelling for the solver

The best preconditioners often come from understanding what the matrix means.

For a PDE, one may approximate the differential operator at another resolution. For a block multiphysics system, one may approximate a Schur complement. For a least-squares problem, one may exploit row or column scaling. For a graph Laplacian, one may use low-stretch spanning trees or multilevel graph coarsening. For a sequence of related systems, one may recycle subspaces learned from previous solves.

This is why preconditioning sits at the boundary between numerical analysis and modelling. The outer Krylov method can be generic. The preconditioner often encodes domain-specific structure.

The one-dimensional Poisson example showed the problem clearly. Mesh refinement drives

$$
\kappa_2(A_n)
$$

like

$$
O(n^2).
$$

Simple Jacobi scaling leaves that condition number unchanged because all diagonal entries are equal. A solver that ignores the operator's multiscale structure therefore becomes progressively less attractive as the mesh is refined.

The GMRES example added a second lesson. Even perfect-looking eigenvalues do not guarantee rapid convergence for a nonnormal matrix. The matrix

$$
\begin{pmatrix}
1 & 100\\
0 & 1
\end{pmatrix}
$$

has only the eigenvalue one, yet one GMRES step leaves a residual norm near one for a simple right-hand side. Spectral clustering is useful, but nonnormal geometry can matter just as much.

A preconditioner should therefore be judged by what it does to the operator that the actual iterative method sees.

The practical workflow is not complicated in principle. Inspect symmetry, definiteness, scaling, sparsity and block structure. Estimate or probe conditioning and spectral behaviour. Choose a Krylov method compatible with those properties. Construct a preconditioner that approximates the difficult part of the operator while remaining cheap enough to apply. Measure total time, memory and scaling, not only iteration count. Monitor residuals in a norm that matches the actual stopping requirement.

The exact solution of

$$
Ax=b
$$

does not depend on how we write the iteration.

The cost of reaching that solution can depend on it completely.

## References

Benzi, M. (2002). Preconditioning techniques for large linear systems: A survey. *Journal of Computational Physics*, 182(2), 418–477. https://doi.org/10.1006/jcph.2002.7176

Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial* (2nd ed.). SIAM.

Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). *Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics* (2nd ed.). Oxford University Press.

Greenbaum, A. (1997). *Iterative Methods for Solving Linear Systems*. SIAM.

Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 49, 409–436.

Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856–869. https://doi.org/10.1137/0907058

Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.

Van der Vorst, H. A. (2003). *Iterative Krylov Methods for Large Linear Systems*. Cambridge University Press.
