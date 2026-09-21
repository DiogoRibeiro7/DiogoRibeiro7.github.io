---
permalink: '/mathematics/entropy_information_theory/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2022-09-27'
excerpt: Entropy appears in information theory, statistical mechanics, and quantum theory through related mathematical forms, but the meanings depend on the probability model, state space, and physical interpretation.
header:
  image: /assets/images/headers/photo-mathematics-information.jpg
  og_image: /assets/images/headers/photo-mathematics-information.jpg
  overlay_image: /assets/images/headers/photo-mathematics-information.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-information.jpg
  twitter_image: /assets/images/headers/photo-mathematics-information.jpg
keywords:
- Shannon entropy
- information theory
- statistical mechanics
- Kullback-Leibler divergence
- mutual information
- von Neumann entropy
seo_description: A mathematically precise introduction to Shannon entropy, relative entropy, mutual information, thermodynamic entropy, and von Neumann entropy, with emphasis on what is shared and what is not.
seo_title: 'Entropy: Information, Probability, and Physical State Counting'
seo_type: article
summary: A rigorous guide to entropy across information theory, statistical mechanics, and quantum theory, separating the shared mathematics from domain-specific interpretation.
tags:
- Information Theory
- Probability
- Statistical Mechanics
title: 'Entropy: Information, Probability, and Physical State Counting'
---

The word **entropy** appears in information theory, thermodynamics, statistical mechanics, and quantum theory.

The formulas are related.

The interpretations are not interchangeable.

A useful discussion therefore begins with the probability model.

## Shannon entropy

For a discrete random variable $X$ with probability mass function

$$
p(x)=P(X=x),
$$

Shannon entropy is

$$
H(X)
=
-\sum_x
p(x)
\log p(x).
$$

If the logarithm is base 2, the unit is bits.

If the logarithm is natural, the unit is nats.

Entropy is a functional of the probability distribution.

It does not belong to one realized observation.

## Why the logarithm appears

If two independent events have probabilities $p$ and $q$, their joint probability is

$$
pq.
$$

An additive information measure therefore satisfies

$$
I(pq)
=
I(p)+I(q).
$$

The logarithm converts multiplication into addition:

$$
-\log(pq)
=
-\log p
-
\log q.
$$

This motivates the self-information

$$
I(x)
=
-\log p(x),
$$

and Shannon entropy is its expectation:

$$
H(X)
=
E[-\log p(X)].
$$

## Uniform distributions maximize discrete entropy

If $X$ has $k$ possible outcomes, then

$$
H(X)
\le
\log k.
$$

Equality holds for

$$
p(x)=\frac1k.
$$

This is a precise statement about a fixed finite support.

It should not be generalized to “more uncertainty always means more entropy” without specifying the probability space and constraints.

## Binary entropy

For a Bernoulli variable with

$$
P(X=1)=p,
$$

the entropy is

$$
H(p)
=
-p\log p
-
(1-p)\log(1-p).
$$

It is maximal at

$$
p=\frac12
$$

and tends to zero as $p$ approaches 0 or 1.

A deterministic Bernoulli variable therefore has zero Shannon entropy.

## Joint and conditional entropy

For variables $X$ and $Y$,

$$
H(X,Y)
=
-\sum_{x,y}
p(x,y)\log p(x,y).
$$

Conditional entropy is

$$
H(X\mid Y)
=
H(X,Y)-H(Y).
$$

The chain rule gives

$$
H(X,Y)
=
H(Y)
+
H(X\mid Y).
$$

If $Y$ tells us something about $X$, then uncertainty about $X$ can decrease after conditioning.

## Mutual information

Mutual information is

$$
I(X;Y)
=
H(X)
-
H(X\mid Y).
$$

Equivalently,

$$
I(X;Y)
=
\sum_{x,y}
p(x,y)
\log
\frac{
p(x,y)
}{
p(x)p(y)
}.
$$

This is the Kullback-Leibler divergence between the joint distribution and the product of marginals:

$$
I(X;Y)
=
D_{\mathrm{KL}}
\left(
p(x,y)
\;\|\;
p(x)p(y)
\right).
$$

Therefore

$$
I(X;Y)\ge0.
$$

It is zero if and only if $X$ and $Y$ are independent, under the usual regularity conditions.

Mutual information detects arbitrary statistical dependence, not only linear correlation.

It does not establish causation.

## Kullback-Leibler divergence

For distributions $P$ and $Q$,

$$
D_{\mathrm{KL}}(P\|Q)
=
\sum_x
p(x)
\log
\frac{
p(x)
}{
q(x)
}.
$$

It satisfies

$$
D_{\mathrm{KL}}(P\|Q)\ge0,
$$

but it is not a metric:

$$
D_{\mathrm{KL}}(P\|Q)
\ne
D_{\mathrm{KL}}(Q\|P)
$$

in general.

KL divergence appears throughout statistics and machine learning because expected log-likelihood differences can often be written in KL form.

Maximum likelihood under model misspecification, variational inference, coding theory, and information geometry all use this structure.

## Cross-entropy

For a true distribution $P$ and model distribution $Q$,

$$
H(P,Q)
=
-
E_P[\log q(X)].
$$

The decomposition

$$
H(P,Q)
=
H(P)
+
D_{\mathrm{KL}}(P\|Q)
$$

shows why minimizing cross-entropy is equivalent to minimizing KL divergence from the data-generating distribution when $H(P)$ does not depend on the model.

In classification, cross-entropy loss is therefore not an arbitrary heuristic.

It is a negative conditional log-likelihood under a categorical model.

## Differential entropy

For a continuous variable with density $f(x)$,

$$
h(X)
=
-
\int
f(x)
\log f(x)\,dx.
$$

Differential entropy behaves differently from discrete entropy.

It can be negative.

It is not invariant under a change of units or coordinates.

For

$$
Y=aX,
$$

we have

$$
h(Y)
=
h(X)+\log|a|.
$$

This is why differential entropy should not be interpreted as an absolute amount of uncertainty in exactly the same way as discrete Shannon entropy.

Mutual information and KL divergence have more stable coordinate interpretations.

## Maximum entropy under constraints

Maximum-entropy modeling chooses the distribution with largest entropy among those satisfying specified constraints.

For example, among continuous distributions on the real line with fixed mean and variance, the Gaussian distribution maximizes differential entropy.

The statement is conditional on those constraints.

It does not mean the Gaussian is the “most random” distribution in every sense.

For a nonnegative variable with fixed mean, the exponential distribution is the maximum-entropy distribution.

Different constraints produce different solutions.

## Statistical mechanics

In statistical mechanics, a macrostate corresponds to many microscopic configurations.

Boltzmann entropy is

$$
S
=
k_B\log\Omega,
$$

where $\Omega$ is the number of accessible microstates compatible with the macrostate.

For a probability distribution over microstates,

$$
S
=
-k_B
\sum_i
p_i\log p_i
$$

is the Gibbs entropy.

The formal similarity to Shannon entropy is exact up to the factor $k_B$.

The physical interpretation is different because the probability distribution is tied to a thermodynamic model and physical state space.

## Canonical distribution

For a system in thermal equilibrium with a heat bath,

$$
p_i
=
\frac{
e^{-\beta E_i}
}{
Z
},
$$

where

$$
\beta
=
\frac{
1
}{
k_BT
}
$$

and

$$
Z
=
\sum_i
e^{-\beta E_i}
$$

is the partition function.

This distribution can be obtained by maximizing Gibbs entropy subject to normalization and a fixed expected energy.

The Lagrange multiplier associated with the energy constraint becomes $\beta$.

This is one of the cleanest mathematical connections between constrained entropy maximization and statistical mechanics.

## Thermodynamic entropy is not merely “disorder”

The word **disorder** is a pedagogical analogy, not a definition.

Thermodynamic entropy is a state function.

In statistical mechanics, it is linked to the number or distribution of microscopic states compatible with macroscopic constraints.

“More disorder” can be misleading because macroscopic ordering, mixing, phase transitions, and constraints can behave in ways that do not match an intuitive visual notion of disorder.

## The second law

For an isolated macroscopic system, thermodynamic entropy does not decrease:

$$
\Delta S
\ge0.
$$

This is a statistical-mechanical statement about overwhelmingly typical macroscopic evolution under the physical model.

It is not a theorem that every subsystem, organism, company, market, or algorithm must become “more disordered.”

Local entropy can decrease when entropy is exported to the environment.

## Von Neumann entropy

For a quantum state with density operator $\rho$,

$$
S(\rho)
=
-\operatorname{Tr}
\left(
\rho\log\rho
\right).
$$

If $\rho$ has eigenvalues

$$
\lambda_1,\ldots,\lambda_k,
$$

then

$$
S(\rho)
=
-\sum_i
\lambda_i\log\lambda_i.
$$

The formula therefore reduces to Shannon entropy applied to the eigenvalue distribution of the density operator.

For a pure state,

$$
\rho^2=\rho,
$$

so its eigenvalues are one 1 and the rest 0, giving

$$
S(\rho)=0.
$$

Mixed states generally have positive entropy.

## Entanglement entropy

For a bipartite pure state

$$
|\psi\rangle_{AB},
$$

the reduced state of subsystem $A$ is

$$
\rho_A
=
\operatorname{Tr}_B
\left(
|\psi\rangle\langle\psi|
\right).
$$

Its von Neumann entropy

$$
S(\rho_A)
$$

is the entanglement entropy of the pure bipartite state.

For a product state, this entropy is zero.

For an entangled pure state, the reduced subsystem can have positive entropy even though the joint state is pure.

This is a specifically quantum phenomenon.

## Entropy in machine learning

Entropy enters machine learning in several precise ways:

- categorical cross-entropy as negative log-likelihood;
- decision-tree split criteria;
- mutual information for dependence or feature screening;
- KL divergence in variational inference;
- maximum-entropy models;
- information bottleneck methods.

These uses share mathematical structure.

They should not be interpreted as thermodynamic entropy unless a physical derivation actually connects the model to thermodynamics.

## Decision trees

For class proportions

$$
p_1,\ldots,p_K
$$

inside a tree node, Shannon entropy is

$$
H
=
-
\sum_{k=1}^{K}
p_k\log p_k.
$$

A split can be scored by information gain:

$$
IG
=
H(\text{parent})
-
\sum_j
w_j
H(\text{child}_j).
$$

This is a data-partition criterion.

It does not imply that a decision tree is carrying out thermodynamic inference.

## Entropy and causality

Mutual information can reveal dependence:

$$
I(X;Y)>0.
$$

That does not identify whether

$$
X\rightarrow Y,
$$

$$
Y\rightarrow X,
$$

or a common cause explains the association.

Information measures can be useful inside causal discovery methods, but causal interpretation requires additional assumptions.

## Entropy rate

For a stationary stochastic process

$$
X_1,X_2,\ldots,
$$

the entropy rate is

$$
h
=
\lim_{n\to\infty}
\frac{
1
}{
n
}
H(
X_1,\ldots,X_n
),
$$

when the limit exists.

It measures information generated per time step.

For an IID process,

$$
h=H(X_1).
$$

For a dependent process, predictability reduces the entropy rate relative to the marginal entropy.

This is more appropriate for time-series information content than applying single-variable entropy independently at each time.

## Source coding

Shannon's source coding theorem connects entropy with lossless compression.

For an IID source with entropy $H(X)$, no lossless code can have expected code length below $H(X)$ bits per symbol in the ideal asymptotic sense, while codes can approach that bound arbitrarily closely over long blocks.

The theorem does not say every practical compressor reaches entropy exactly.

Finite blocks, source dependence, model mismatch, and coding constraints matter.

## What entropy does not mean

Entropy is not automatically:

- randomness in every colloquial sense;
- causal complexity;
- intelligence;
- economic inefficiency;
- biological disorder;
- model uncertainty in a complete Bayesian sense;
- algorithmic complexity.

Related concepts exist in those domains, but the definitions differ.

Using the same word does not make the quantities equivalent.

## Conclusion

Entropy is useful precisely because the mathematics recurs across several fields.

The cleanest relationships are:

$$
\boxed{
\text{Shannon entropy}
\rightarrow
\text{expected information}
}
$$

$$
\boxed{
\text{KL divergence}
\rightarrow
\text{distributional discrepancy}
}
$$

$$
\boxed{
\text{mutual information}
\rightarrow
\text{statistical dependence}
}
$$

$$
\boxed{
\text{Gibbs/Boltzmann entropy}
\rightarrow
\text{physical state counting and probabilities}
}
$$

$$
\boxed{
\text{von Neumann entropy}
\rightarrow
\text{quantum state uncertainty}
}
$$

The formulas are connected.

The scientific interpretation still belongs to the model in which each formula is used.

## References

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379–423, 623–656.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620–630.
- Pathria, R. K., & Beale, P. D. (2011). *Statistical Mechanics* (3rd ed.). Elsevier.
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information* (10th anniversary ed.). Cambridge University Press.
