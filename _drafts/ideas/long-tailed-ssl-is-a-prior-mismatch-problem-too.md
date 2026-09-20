---
author_profile: false
categories:
- Machine Learning
- Statistics
classes: wide
title: 'Long-Tailed Semi-Supervised Learning Is a Prior-Mismatch Problem Too'
excerpt: 'A critical reading of Meta-Expert for long-tailed SSL with labelled-unlabelled distribution mismatch, where different experts specialize in head, medium and tail regions.'
keywords:
- long-tailed learning
- semi-supervised learning
- distribution mismatch
- mixture of experts
- class imbalance
- ICML 2025
tags:
- Semi-Supervised Learning
- Long-Tailed Learning
- Distribution Shift
- Mixture of Experts
- ICML 2025
seo_title: 'Meta-Expert for Long-Tailed Semi-Supervised Learning'
seo_description: 'Technical reading of Hou and Jia (ICML 2025), focusing on labelled-unlabelled class-prior mismatch, expert assignment, feature-depth bias and generalization.'
summary: 'A paper-driven article on Meta-Expert for LTSSL, treating the problem as class-prior mismatch plus representation bias rather than only imbalance, and examining whether dynamic expert assignment solves the right statistical problem.'
why_this_exists: 'Many SSL methods quietly assume similar class proportions between labelled and unlabelled data. Long-tailed settings make that assumption visibly false. Meta-Expert provides a modern test case for how pseudo-labeling behaves under prior mismatch.'
evidence: 'Hou and Jia, A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning, ICML 2025, PMLR 267:23960-23975.'
methodology: 'Formalize labelled and unlabelled class priors separately, inspect the dynamic expert assignment and multi-depth feature fusion, and distinguish class-prior shift from conditional distribution shift.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: What changes in SSL when labelled and unlabelled class distributions are both long-tailed and mismatched?
Claim: Different experts can be useful in different head/medium/tail regimes, and dynamic assignment can exploit that specialization.
Counterclaim: Head/medium/tail membership is itself estimated; assignment errors can route examples to the wrong expert and reinforce pseudo-label bias.
Evidence object: ICML 2025 Meta-Expert paper, generalization analysis and CIFAR-10-LT/STL-10-LT/SVHN-LT experiments.
Failure case: Class-prior mismatch is not the same as covariate or conditional shift; methods may fail when several forms of shift coexist.
Reader payoff: A precise distinction between imbalance, prior shift, pseudo-label bias and expert specialization.
Exclusions: Do not write a generic class-imbalance article.
-->

## Paper

Yaxin Hou and Yuheng Jia.  
**A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning.**  
ICML 2025, PMLR 267:23960-23975.  
https://proceedings.mlr.press/v267/hou25d.html

## Core statistical setup

Let the labelled class prior be

$$
\pi_L(y)
$$

and the unlabelled class prior be

$$
\pi_U(y).
$$

Long-tailed SSL with distribution mismatch allows

$$
\pi_L(y)
\neq
\pi_U(y).
$$

Even if

$$
p_L(x\mid y)
=
p_U(x\mid y),
$$

a pseudo-labeler calibrated to $pi_L$ can be systematically misaligned with the unlabelled population.

The article should keep prior shift separate from other forms of distribution shift.

## Method angle

Meta-Expert uses experts associated with different parts of the class-frequency spectrum.

A routing function estimates whether an example belongs to a head, medium or tail class, then selects a suitable expert.

Conceptually:

$$
r(x)
\in
\{H,M,T\},
$$

$$
\hat y
=
f_{r(x)}(x).
$$

This is a mixture-of-experts problem with an estimated router.

The critical question is how router error propagates into pseudo-label error.

## Feature-depth claim

The paper reports that:
- deeper features are more discriminative but more head-biased;
- shallower features are less biased but less discriminative.

The multi-depth fusion module tries to trade those properties.

This is a good opportunity to discuss representation bias as a function of network depth rather than as one scalar property.

## What to inspect

- How is head/medium/tail membership estimated?
- Does the router use predicted class or representation statistics?
- How large is the router error?
- What happens when the unlabelled prior is uniform but labelled data are long-tailed?
- What happens in the reverse mismatch?
- Are gains concentrated in tail recall?
- Does overall accuracy hide head-tail trade-offs?
- How is the generalization bound connected to the actual routing procedure?
- Are conditional distributions assumed stable?

## Proposed article structure

1. Imbalance is not the same as mismatch
2. Separate labelled and unlabelled priors
3. Why one pseudo-labeler can be biased
4. Expert specialization
5. Dynamic routing
6. Representation depth and head bias
7. Generalization argument
8. Tail metrics versus overall accuracy
9. Failure under combined prior and covariate shift
10. Practical diagnostics

## Reproducibility plan

Create a synthetic multiclass problem with fixed class-conditionals and controllable priors.

Vary

$$
\pi_L
$$

and

$$
\pi_U
$$

independently, then compare:
- one classifier;
- prior correction;
- simple class-specific experts;
- an oracle router;
- an estimated router.

This separates the value of expertise from the cost of routing error.
