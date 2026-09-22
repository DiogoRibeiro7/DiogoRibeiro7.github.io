---
permalink: '/mathematics/Markov_Chain/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-17'
header:
  image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  og_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  twitter_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
keywords:
- Markov chains
- Transition matrix
- Stationary distribution
- Recurrence
- Mixing time
- Hidden Markov models
redirect_from:
- '/mathematics/statistics/data science/machine learning/Markov_Chain/'
seo_description: "A rigorous introduction to Markov chains covering state representation, transition structure, recurrence, stationarity, reversibility, convergence, mixing, hitting times, absorbing chains, and Hidden Markov Models."
seo_title: "Markov Chains: State, Recurrence, Stationarity, and Mixing"
seo_type: article
subtitle: "From transition kernels to long-run behaviour"
tags:
- Stochastic Processes
- Probability
- Markov Chains
title: "Markov Chains: State, Recurrence, Stationarity, and Mixing"
---

A Markov chain is a stochastic process whose future evolution is conditionally independent of the distant past once the present state is known. That sentence is often shortened to the slogan that a Markov chain is “memoryless,” but the slogan is easy to misunderstand. The process may have a rich history, and that history may have shaped the current state in a complicated way. The Markov property says only that, **conditional on the present state**, the earlier path contains no additional information about the next transition.

For a discrete-time process $(X_t)_{t\ge0}$ on a state space $\mathcal S$, the Markov property is

$$
P(
X_{t+1}=j
\mid
X_t=i,X_{t-1},\ldots,X_0
)
=
P(
X_{t+1}=j
\mid
X_t=i
).
$$

This is a conditional-independence statement, not a claim that the process literally forgets the past. Whether the assumption is plausible depends strongly on what has been chosen as the state. If tomorrow’s demand depends on both today’s and yesterday’s demand, then the scalar process $X_t$ is not first-order Markov, but the augmented state

$$
Z_t=(X_t,X_{t-1})
$$

may be. In many applications, the question “is the system Markov?” is therefore partly a question about whether the state representation contains enough information.

## Transition structure and finite-state chains

For a time-homogeneous finite-state Markov chain, the transition probabilities

$$
P_{ij}
=
P(X_{t+1}=j\mid X_t=i)
$$

do not depend on $t$. Collecting them gives the transition matrix

$$
P=
\begin{bmatrix}
P_{11} & \cdots & P_{1m}\\
\vdots & \ddots & \vdots\\
P_{m1} & \cdots & P_{mm}
\end{bmatrix},
$$

where every entry is non-negative and every row sums to one. If $\pi_t$ is the row vector of state probabilities at time $t$, then

$$
\pi_{t+1}
=
\pi_tP,
$$

and after $n$ steps,

$$
\pi_{t+n}
=
\pi_tP^n.
$$

The powers of the transition matrix therefore encode all finite-horizon transition probabilities. In particular,

$$
(P^n)_{ij}
=
P(X_{t+n}=j\mid X_t=i).
$$

This algebraic representation makes finite-state Markov chains unusually tractable. Questions about accessibility, equilibrium, return times, absorption, and convergence can often be translated into matrix calculations. The danger is to confuse matrix convenience with modeling validity. A transition matrix estimated from historical data is meaningful only if the state definition and the assumed time-homogeneity are appropriate for the process.

If the transition law changes with time, season, intervention, or an exogenous covariate, then a single matrix $P$ is not enough. One may instead have

$$
P_t
$$

at each time step, producing

$$
\pi_{t+n}
=
\pi_tP_tP_{t+1}\cdots P_{t+n-1}.
$$

A parking-occupancy process, for example, is unlikely to have the same transition law at 08:00 and 03:00. Treating it as homogeneous without including time-of-day in the state silently imposes a false assumption.

## Communicating classes, recurrence, and transience

Long-run behavior depends first on which states can reach one another. State $j$ is **accessible** from state $i$ if there exists some $n\ge0$ such that

$$
(P^n)_{ij}>0.
$$

States $i$ and $j$ **communicate** if each is accessible from the other. Communication partitions the state space into communicating classes. A chain is **irreducible** when all states belong to one class.

Irreducibility matters because a chain with several closed classes can have fundamentally different limiting behavior depending on where it starts. If one subset of states can never be left once entered, while another subset behaves differently, then there is no single global equilibrium describing all trajectories.

A second distinction concerns returns. Let

$$
\tau_i^+
=
\inf\{t\ge1:X_t=i\}
$$

be the first return time to state $i$. The state is **recurrent** if

$$
P_i(\tau_i^+<\infty)=1,
$$

and **transient** otherwise. Recurrence says that a return eventually occurs with probability one. In a finite irreducible chain, every state is positive recurrent. In countably infinite chains, recurrence alone is not enough to guarantee a stationary probability distribution; positive recurrence is required.

This distinction is one of the places where finite-state intuition can become misleading. A symmetric random walk on the integers is recurrent, yet it has no normalizable stationary distribution over $\mathbb Z$. The chain returns to every state with probability one, but it does not spend a stable positive fraction of time at each state in a way that sums to one over the infinite state space.

## Stationary distributions are not the same as convergence

A probability vector $\pi$ is stationary if

$$
\pi=\pi P.
$$

If the chain starts with $X_0\sim\pi$, then

$$
X_t\sim\pi
$$

for every $t$. Stationarity is therefore an invariance property of the distribution under one step of the Markov dynamics.

For a finite irreducible chain, a unique stationary distribution exists. This fact alone does **not** imply that the distribution of the chain converges to $\pi$ from every starting state. Periodicity can prevent convergence.

The period of state $i$ is

$$
d(i)
=
\gcd
\{
n\ge1:(P^n)_{ii}>0
\}.
$$

In an irreducible chain all states have the same period. A chain with period one is aperiodic. Consider the deterministic two-state chain

$$
P=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}.
$$

Its stationary distribution is

$$
\pi=
\left(
\frac12,\frac12
\right),
$$

but if the chain starts from state 1, the distribution alternates forever between the two states. It never converges pointwise to $\pi$.

For a finite irreducible and aperiodic chain,

$$
P^n(i,\cdot)
\longrightarrow
\pi
$$

as $n\to\infty$ for every starting state $i$. These are the conditions behind the standard finite-state ergodic theorem. The convergence claim is stronger than the existence of a stationary distribution and depends on both irreducibility and aperiodicity.

Even when ordinary convergence fails because of periodicity, time averages can still converge. Under suitable recurrence assumptions, ergodic averages of the form

$$
\frac1n
\sum_{t=0}^{n-1}
f(X_t)
$$

converge to the stationary expectation

$$
E_\pi[f(X)].
$$

This is why stationary distributions remain useful for long-run average behavior even when the one-step distribution oscillates.

## Reversibility and detailed balance

A stationary distribution $\pi$ satisfies **detailed balance** with transition matrix $P$ if

$$
\pi_iP_{ij}
=
\pi_jP_{ji}
$$

for every pair of states $i,j$. Summing over $i$ immediately gives

$$
\sum_i
\pi_iP_{ij}
=
\pi_j
\sum_iP_{ji}^{\text{reverse}},
$$

and more directly,

$$
\sum_i
\pi_iP_{ij}
=
\sum_i
\pi_jP_{ji}
=
\pi_j,
$$

so detailed balance implies stationarity.

The converse is false. A chain can be stationary without being reversible. Detailed balance is therefore a sufficient condition, not the definition of stationarity.

Reversibility is especially important in Markov chain Monte Carlo because it gives a convenient way to construct a chain with a desired invariant distribution. Metropolis-Hastings, for example, chooses transition probabilities so that the target distribution satisfies detailed balance. The resulting chain can then be simulated to approximate expectations under the target. The price of this convenience is that reversibility can restrict the dynamics; non-reversible samplers can sometimes mix faster.

## Mixing and convergence rate

Knowing that a chain converges does not tell us how long convergence takes. For simulation and applications, the rate can be more important than the existence theorem. One standard measure is total variation distance,

$$
\|
P^n(i,\cdot)-\pi
\|_{\mathrm{TV}}
=
\frac12
\sum_j
\left|
P^n(i,j)-\pi_j
\right|.
$$

The mixing time at tolerance $\varepsilon$ can be defined as

$$
t_{\mathrm{mix}}(\varepsilon)
=
\min
\left\{
n:
\max_i
\|
P^n(i,\cdot)-\pi
\|_{\mathrm{TV}}
\le\varepsilon
\right\}.
$$

Two chains may share exactly the same stationary distribution yet have radically different mixing times. This is a central issue in MCMC. A sampler that has the correct invariant distribution but moves only slowly between modes can produce highly autocorrelated samples and poor finite-run estimates.

For reversible finite chains, spectral properties of $P$ help characterize mixing. If

$$
1=\lambda_1>\lambda_2\ge\cdots\ge\lambda_m\ge-1
$$

are eigenvalues in the appropriate ordering, then the gap between 1 and the largest nontrivial eigenvalue in magnitude influences the convergence rate. A small spectral gap corresponds to slow relaxation toward equilibrium.

This connection between probability and linear algebra is one of the reasons Markov chains are so useful: questions about stochastic dynamics can often be related to eigenvalues, eigenvectors, conductance, and graph structure.

## Hitting times and absorbing behavior

Many applications are concerned not with equilibrium but with the time needed to reach a particular state or set of states. For a target set $A\subseteq\mathcal S$, define the hitting time

$$
\tau_A
=
\inf\{t\ge0:X_t\in A\}.
$$

Expected hitting times often satisfy linear recursions. If

$$
h_i
=
E_i[\tau_A],
$$

then for $i\notin A$,

$$
h_i
=
1
+
\sum_j
P_{ij}h_j,
$$

with boundary condition

$$
h_i=0
\qquad
\text{for }i\in A.
$$

This converts a probabilistic first-passage problem into a system of linear equations.

An absorbing state satisfies

$$
P_{ii}=1.
$$

If a finite chain is ordered so that transient states come first and absorbing states last, its transition matrix can be written in canonical form,

$$
P=
\begin{bmatrix}
Q & R\\
0 & I
\end{bmatrix}.
$$

The matrix

$$
N
=
(I-Q)^{-1}
$$

is called the fundamental matrix. Entry $N_{ij}$ gives the expected number of visits to transient state $j$ before absorption when starting from transient state $i$. Summing across a row gives the expected time to absorption.

These results are directly useful in reliability models, disease progression, credit default, queueing, customer lifecycle analysis, and any setting in which some states represent terminal outcomes.

## Hidden Markov models

A Hidden Markov Model separates the latent state process from the observed data. Let $Z_t$ denote an unobserved Markov chain and $Y_t$ an observation generated conditionally on the hidden state. A standard HMM factorization is

$$
P(z_{1:T},y_{1:T})
=
P(z_1)
\prod_{t=2}^T
P(z_t\mid z_{t-1})
\prod_{t=1}^T
P(y_t\mid z_t).
$$

The Markov property applies to the latent states, while observations are assumed conditionally independent given those states.

This structure supports several distinct inferential tasks. **Filtering** computes

$$
P(Z_t\mid Y_{1:t}),
$$

the current latent-state distribution given observations up to the present. **Smoothing** computes

$$
P(Z_t\mid Y_{1:T}),
$$

using future observations as well. **Decoding**, often through the Viterbi algorithm, seeks the most probable latent-state sequence. Parameter estimation can be carried out through maximum likelihood, commonly using the Baum-Welch algorithm, or through Bayesian methods.

HMMs are useful when observed sequences are generated by regimes that are not directly visible: speech phonemes, market regimes, machine operating states, biological sequence structure, or human activity modes. Their usefulness depends on whether the latent-state abstraction is adequate. Adding more hidden states can always increase flexibility, but the resulting states may become difficult to identify or interpret.

## State design is the modeling problem

The Markov property is not purely a property of the physical world; it is also a property of the chosen representation. Suppose machine failure risk depends on the accumulated stress history. If the state records only the current temperature, the process may be strongly non-Markov. If the state includes cumulative load, age, and recent stress history, the Markov approximation may become much more plausible.

This observation links Markov modeling to the broader concept of sufficient state. The state should contain enough information that the conditional distribution of the future is approximately determined by the present representation. If important information is omitted, apparent long memory can remain. If too much information is included, the state space becomes enormous and transition estimates become statistically sparse.

There is therefore a trade-off between state sufficiency and statistical estimability. A highly aggregated state space may violate the Markov assumption; an excessively detailed state space may make reliable estimation impossible. Good state design uses scientific structure to find a representation that is both predictive of the future and estimable from available data.

The same principle applies to reinforcement learning, where the Markov Decision Process assumption requires the environment state to contain the information needed for the transition and reward distributions. If observations are only partial, the problem is more naturally treated as a partially observable Markov decision process.

## Applications require stronger assumptions than the mathematics alone

It is easy to write down a transition matrix and call a process Markov. The difficult part is deciding whether the estimated transition law is stable enough to support the intended inference. Historical transition frequencies can change under seasonality, interventions, policy changes, population drift, or feedback from decisions based on the model.

A customer-lifecycle model estimated from one pricing regime may fail after pricing changes. A hospital-state transition model may change after a treatment protocol is introduced. A credit migration matrix may behave differently across macroeconomic regimes. In such cases, a time-homogeneous Markov chain can still be a useful local approximation, but its stationary distribution should not be interpreted as an immutable long-run equilibrium of the real system.

The same caution applies to simulation. A chain can be mathematically ergodic and still require far more steps to mix than are computationally feasible. In MCMC, stationarity is an asymptotic property; finite-run validity depends on autocorrelation, initialization, mode exploration, and convergence diagnostics. The mere existence of a target stationary distribution is not enough.

## Conclusion

Markov chains provide a compact language for stochastic systems whose present state summarizes the information needed for future evolution. Their apparent simplicity comes from the Markov property, but the theory becomes rich as soon as one asks the questions that matter in applications: which states communicate, which are recurrent, whether a stationary distribution exists, whether the chain converges to it, how quickly that convergence occurs, how long it takes to hit important states, and whether the chosen state representation is actually sufficient.

These concepts are related but not interchangeable. Stationarity is an invariance property; irreducibility concerns communication; recurrence concerns returns; aperiodicity concerns cyclic structure; mixing concerns convergence rate; reversibility is a symmetry property that can simplify construction and analysis. Confusing them leads to incorrect claims about equilibrium and long-run behavior.

The most important modeling decision is often the state itself. A poorly chosen state can make a process look non-Markov, while a richer representation can restore the conditional-independence structure at the cost of a larger state space. That trade-off between sufficiency and estimability is central to practical Markov modeling.

Used carefully, Markov chains connect probability, linear algebra, stochastic processes, simulation, and statistical inference in an unusually elegant way. Used mechanically, they reduce complex dynamics to a transition matrix without asking whether the assumptions behind that matrix are defensible. The mathematics is powerful precisely because it makes those assumptions explicit.

## References

- Levin, D. A., Peres, Y., & Wilmer, E. L. (2017). *Markov Chains and Mixing Times* (2nd ed.). American Mathematical Society.
- Norris, J. R. (1997). *Markov Chains*. Cambridge University Press.
- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257-286.
