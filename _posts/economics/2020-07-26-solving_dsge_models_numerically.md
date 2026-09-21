---
permalink: '/economics/solving_dsge_models_numerically/'
author_profile: false
categories:
- Economics
classes: wide
date: '2020-07-26'
excerpt: DSGE models are solved by approximating policy functions or value functions, not by applying finite differences to arbitrary equations. This article separates local perturbation from global numerical methods.
header:
  image: /assets/images/headers/photo-statistics-regression-errors.jpg
  og_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_image: /assets/images/headers/photo-statistics-regression-errors.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-regression-errors.jpg
  twitter_image: /assets/images/headers/photo-statistics-regression-errors.jpg
keywords:
- DSGE models
- Perturbation methods
- Value function iteration
- Numerical methods
- Dynamic programming
- Economics
- Python
redirect_from:
- '/mathematical economics/solving_dsge_models_numerically/'
seo_description: A rigorous guide to numerical DSGE solution methods, distinguishing local perturbation from global dynamic-programming and projection approaches.
seo_title: Solving DSGE Models Numerically
seo_type: article
summary: A mathematical guide to DSGE solution methods with a reproducible stochastic growth-model example and a clear distinction between perturbation, projection, and finite-difference approximations.
tags:
- Economics
- Numerical Methods
- Mathematical Modeling
- Python
title: 'Solving DSGE Models Numerically: Perturbation and Global Methods'
---

A dynamic stochastic general equilibrium model is not "solved" by evaluating its equations at a few future values. The numerical task is to recover functions that map the state of the economy into decisions. If the state is

$$
s_t,
$$

a solution typically consists of policy functions such as

$$
c_t=g_c(s_t),
\qquad
k_{t+1}=g_k(s_t),
$$

together with laws of motion for the exogenous states. That distinction matters because a piece of code can satisfy one Euler equation at one pair of points without solving the dynamic model.

The earlier version of this article made exactly that mistake. It called a root finder on a static expression "first-order perturbation" and called a forward numerical derivative "finite-difference solution." Neither operation computes the policy functions of a DSGE model.

This revision starts from the actual numerical problem.

## A benchmark stochastic growth model

Consider a planner with preferences

$$
E_0
\sum_{t=0}^{\infty}
\beta^t\log c_t
$$

subject to

$$
c_t+k_{t+1}
=
z_t k_t^\alpha
+
(1-\delta)k_t,
$$

with productivity following a Markov process

$$
P(z_{t+1}=z_j\mid z_t=z_i)
=
\Pi_{ij}.
$$

The state is

$$
s_t=(k_t,z_t).
$$

The control can be written as next-period capital $k_{t+1}$, because consumption is determined by the resource constraint:

$$
c_t
=
z_t k_t^\alpha
+
(1-\delta)k_t
-
k_{t+1}.
$$

A solution is a policy

$$
k_{t+1}=g(k_t,z_t)
$$

that satisfies optimality for every relevant state, not merely along one simulated path.

## The Bellman equation

The recursive problem is

$$
V(k,z)
=
\max_{k'>0}
\left\{
\log c
+
\beta
E\left[
V(k',z')
\mid z
\right]
\right\},
$$

where

$$
c
=
zk^\alpha
+
(1-\delta)k
-k'.
$$

The expectation is

$$
E[V(k',z')\mid z_i]
=
\sum_j
\Pi_{ij}
V(k',z_j).
$$

This equation gives us one route to a global numerical solution: value-function iteration.

## The Euler equation

For an interior optimum, the policy also satisfies

$$
\frac{1}{c_t}
=
\beta
E_t
\left[
\frac{1}{c_{t+1}}
\left(
\alpha z_{t+1}k_{t+1}^{\alpha-1}
+
1-\delta
\right)
\right].
$$

A correct numerical solution should make the Euler-equation residual small over the region where the policy will be used. That residual is a diagnostic. It is not, by itself, a solution algorithm.

## Deterministic steady state

At

$$
z=1
$$

and a non-stochastic steady state,

$$
k_{t+1}=k_t=k^\ast,
$$

so the Euler equation becomes

$$
1
=
\beta
\left[
\alpha(k^\ast)^{\alpha-1}
+
1-\delta
\right].
$$

Therefore,

$$
k^\ast
=
\left[
\frac{\alpha}
{\beta^{-1}-1+\delta}
\right]^{1/(1-\alpha)}.
$$

Consumption is

$$
c^\ast
=
(k^\ast)^\alpha
-
\delta k^\ast.
$$

The steady state is important for both local and global algorithms. Perturbation expands the solution around it. A global method often uses it to define a sensible computational domain.

## What perturbation actually means

Let

$$
F(s_t,s_{t+1},\varepsilon_{t+1};\sigma)=0
$$

collect the equilibrium conditions, where $\sigma$ scales shock size. A perturbation method treats the equilibrium policy function as an unknown smooth function of the state and shock scale and computes derivatives of that function around the deterministic steady state,

$$
(s^\ast,\sigma=0).
$$

A first-order approximation has the form

$$
\hat s_{t+1}
=
A\hat s_t
+
B\varepsilon_{t+1},
$$

where hats denote deviations from steady state, often in logs. The matrices $A$ and $B$ are not obtained by calling a generic scalar root finder on the Euler equation. They come from differentiating the complete equilibrium system and solving the resulting linear rational-expectations problem. At first order, certainty equivalence commonly appears: shock variances do not change the mean policy rule.

At second order, curvature introduces terms involving variances and interactions, allowing uncertainty to affect expected decisions and welfare. That is one reason second-order perturbation is used for risk premia and welfare calculations.

## Local accuracy is the main trade-off

Perturbation is attractive because it is fast. For large macroeconomic models with many state and control variables, a first- or second-order local solution can be dramatically cheaper than constructing a high-dimensional global grid. But the approximation is local. If the economy moves far from the expansion point, or if occasionally binding constraints matter, the truncated Taylor expansion can become inaccurate or even imply impossible decisions.

This is not a defect in Taylor series. It is the consequence of asking a local approximation to describe a global nonlinear problem.

## Finite differences are not a competing DSGE solution method by themselves

A finite difference such as

$$
f'(x)
\approx
\frac{f(x+h)-f(x)}{h}
$$

approximates a derivative. That tool can appear inside many numerical algorithms. For example, finite differences can approximate derivatives in a Hamilton-Jacobi-Bellman equation, compute Jacobians for Newton methods, or discretize a continuous-state problem. But evaluating finite differences of the production function does not solve the DSGE model.

The numerical method is defined by the equation being discretized and the policy or value function being recovered. So the meaningful comparison is not

$$
\text{perturbation versus finite differences}.
$$

It is closer to

$$
\boxed{
\text{local perturbation}
\quad\text{versus}\quad
\text{global approximation methods}
}
$$

with finite differences being one possible numerical ingredient.

## A global alternative: value-function iteration

For the growth model above, discretize capital on a grid

$$
k_1,\ldots,k_M
$$

and productivity on states

$$
z_1,\ldots,z_S.
$$

For each current state $(k_i,z_s)$ and each candidate $k_j'$, compute feasible consumption

$$
c_{isj}
=
z_s k_i^\alpha
+
(1-\delta)k_i
-
k_j'.
$$

The Bellman update is

$$
V^{new}(k_i,z_s)
=
\max_j
\left[
\log c_{isj}
+
\beta
\sum_{r=1}^{S}
\Pi_{sr}
V(k_j',z_r)
\right].
$$

Repeat until

$$
\|V^{new}-V\|_\infty
<
\varepsilon.
$$

The maximizing index at each state is the discrete policy function.

## Reproducible Python implementation

The following code solves a two-state stochastic growth model by value-function iteration.

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

beta: float = 0.96
alpha: float = 0.36
delta: float = 0.08

productivity: FloatArray = np.exp(
    np.array([-0.02, 0.02], dtype=float)
)

transition: FloatArray = np.array(
    [
        [0.95, 0.05],
        [0.05, 0.95],
    ],
    dtype=float,
)

if transition.shape != (2, 2):
    raise ValueError("Transition matrix must be 2 x 2.")

if not np.allclose(
    transition.sum(axis=1),
    1.0,
):
    raise ValueError(
        "Each transition row must sum to one."
    )

k_ss: float = (
    alpha
    / (1.0 / beta - 1.0 + delta)
) ** (1.0 / (1.0 - alpha))

capital_grid: FloatArray = np.linspace(
    0.5 * k_ss,
    1.5 * k_ss,
    250,
)

value: FloatArray = np.zeros(
    (
        productivity.size,
        capital_grid.size,
    ),
    dtype=float,
)

policy_index: IntArray = np.zeros(
    value.shape,
    dtype=np.int64,
)

tolerance: float = 1e-8
max_iterations: int = 1_000

for iteration in range(max_iterations):
    value_new = np.empty_like(value)
    policy_new = np.empty_like(policy_index)

for z_index, z_value in enumerate(
        productivity
    ):
        expected_value: FloatArray = (
            transition[z_index] @ value
        )

for k_index, capital in enumerate(
            capital_grid
        ):
            resources: float = (
                z_value * capital**alpha
                + (1.0 - delta) * capital
            )

consumption: FloatArray = (
                resources - capital_grid
            )

objective: FloatArray = np.full(
                capital_grid.shape,
                -np.inf,
                dtype=float,
            )

feasible: NDArray[np.bool_] = (
                consumption > 0.0
            )

objective[feasible] = (
                np.log(consumption[feasible])
                + beta
                * expected_value[feasible]
            )

best_index: int = int(
                np.argmax(objective)
            )

value_new[
                z_index,
                k_index,
            ] = objective[best_index]

policy_new[
                z_index,
                k_index,
            ] = best_index

sup_norm: float = float(
        np.max(np.abs(value_new - value))
    )

value = value_new
    policy_index = policy_new

if sup_norm < tolerance:
        break
else:
    raise RuntimeError(
        "Value iteration did not converge."
    )

policy_capital: FloatArray = (
    capital_grid[policy_index]
)

print(
    "iterations:",
    iteration + 1,
)

print(
    "sup-norm:",
    f"{sup_norm:.3e}",
)

print(
    "steady-state capital:",
    f"{k_ss:.6f}",
)
~~~

With the parameters above, the code converges on the specified grid. The policy moves next-period capital upward in the high-productivity state and downward in the low-productivity state around the deterministic steady state, which is the direction economic intuition predicts.

## Grid error is different from model error

A value-function solution on a finite grid contains discretization error. If the true optimum lies between $k_j$ and $k_{j+1}$, a discrete policy must choose one grid point. Increasing the grid density reduces this source of error but increases computational cost. Interpolation can improve the approximation without making the grid prohibitively dense.

The distinction is worth making explicit:

$$
\text{economic approximation error}
\neq
\text{numerical discretization error}.
$$

Perturbation mainly introduces truncation error from a local Taylor expansion. Grid methods introduce discretization and interpolation error. Both need diagnostics.

## Euler-equation errors are a useful common diagnostic

After obtaining a policy $g(k,z)$, compute consumption from the resource constraint and evaluate the Euler residual,

$$
\mathcal E(k,z)
=
1
-
\beta
E
\left[
\frac{u'(c')}
{u'(c)}
\left(
\alpha z' (k')^{\alpha-1}
+
1-\delta
\right)
\right].
$$

A small residual over the relevant state space indicates that the approximate policy nearly satisfies the first-order condition. This allows different numerical methods to be compared on a common economic equation rather than on implementation-specific convergence criteria alone.

## Projection and collocation methods

Value-function iteration is not the only global method. Projection methods approximate an unknown policy or value function by basis functions,

$$
g(s)
\approx
\sum_{m=1}^{M}
a_m\phi_m(s),
$$

then choose the coefficients $a_m$ so that equilibrium residuals are small at selected collocation points or in a weighted integral sense. Chebyshev polynomials are a common basis because they have good approximation properties over bounded intervals. Projection can be much faster than dense grids in smooth low-dimensional problems. The curse of dimensionality remains important.

## Occasionally binding constraints change the method choice

Consider a borrowing constraint,

$$
b_{t+1}\ge\underline b,
$$

or a policy rate constrained by

$$
i_t\ge0.
$$

Near a point where the constraint never binds, a local perturbation can completely miss the kink created when it becomes active. Piecewise-linear methods, occasionally binding constraint algorithms, endogenous-grid methods, projection, or other global approaches can be more appropriate. The numerical method should follow the economic structure.

## Choosing a method

A useful summary is:

| Feature | Perturbation | Global grid / projection |
|---|---|---|
| Approximation | Local | Broader state region |
| Speed | Usually high | Usually lower |
| Large models | Often feasible | Curse of dimensionality |
| Strong nonlinearities | Higher orders help locally | Can represent them globally |
| Occasionally binding constraints | Difficult for plain perturbation | Often better suited |
| Main error | Taylor truncation | Grid / basis approximation |
| Diagnostics | Euler errors, simulation moments | Euler errors, Bellman residuals |

There is no universally best solver. There is a model, a region of the state space that matters, and an accuracy requirement.

## Conclusion

A DSGE solution is a set of decision rules satisfying equilibrium conditions over the relevant states. Perturbation obtains local derivatives of those rules around a steady state. Global methods approximate the functions over a larger region. Finite differences can help approximate derivatives inside such methods, but a finite-difference formula is not itself a DSGE solution.

That distinction is the difference between numerical analysis and code that merely produces numbers.

## References

- Schmitt-Grohé, S., & Uribe, M. (2004). Solving dynamic general equilibrium models using a second-order approximation to the policy function. *Journal of Economic Dynamics and Control*, 28(4), 755–775.
- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.
- Miranda, M. J., & Fackler, P. L. (2002). *Applied Computational Economics and Finance*. MIT Press.
- Heer, B., & Maussner, A. (2009). *Dynamic General Equilibrium Modeling* (2nd ed.). Springer.
