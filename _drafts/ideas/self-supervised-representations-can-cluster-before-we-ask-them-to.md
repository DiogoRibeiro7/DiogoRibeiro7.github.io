---
author_profile: false
categories:
- Machine Learning
classes: wide
title: 'Self-Supervised Representations Can Cluster Before We Ask Them To'
excerpt: 'A technical reading of ReSA and the claim that joint-embedding self-supervised representations develop stable clustering structure that can be fed back into representation learning.'
keywords:
- self-supervised learning
- clustering
- joint embedding
- representation learning
- ReSA
- ICML 2025
tags:
- Self-Supervised Learning
- Clustering
- Representation Learning
- ICML 2025
seo_title: 'Clustering Properties of Self-Supervised Learning'
seo_description: 'Critical reading of Weng et al. (ICML 2025) and Representation Self-Assignment, with emphasis on whether clustering structure is measured independently or reinforced by the training objective.'
summary: 'A paper-driven article on the clustering properties of joint-embedding SSL and the ReSA positive-feedback mechanism, focusing on representation geometry, self-assignment, stability, and the risk of validating structure with the same machinery used to create it.'
why_this_exists: 'Modern SSL often produces semantically organized representations without explicit class labels. ReSA turns that empirical property into a training signal, creating a useful case study in positive feedback between representation and clustering.'
evidence: 'Weng et al., Clustering Properties of Self-Supervised Learning, ICML 2025, PMLR 267:66597-66616.'
methodology: 'Reconstruct the clustering metrics used across representation components, formalize the self-assignment feedback loop, inspect ablations, and separate emergence of structure from reinforcement of structure.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: When self-supervised representations appear clusterable, is that structure an emergent property, an evaluation artifact, or a signal that can improve training?
Claim: ReSA uses representation self-assignment to turn existing cluster structure into a positive-feedback learning signal.
Counterclaim: Once clustering is inserted into the objective, later improvements in clustering metrics are not independent confirmation of semantic structure.
Evidence object: ICML 2025 ReSA paper and its clustering metrics/ablations.
Failure case: Strong benchmark alignment may depend on datasets whose semantic classes already match augmentation and pretraining biases.
Reader payoff: Understand when clustering is a property of representation learning rather than a separate downstream algorithm.
Exclusions: Do not write another generic SSL overview.
-->

## Paper

Xi Weng, Jianing An, Xudong Ma, Binhang Qi, Jie Luo, Xi Yang, Jin Song Dong, and Lei Huang.  
**Clustering Properties of Self-Supervised Learning.**  
ICML 2025, PMLR 267:66597-66616.  
https://proceedings.mlr.press/v267/weng25a.html

## Core question

Joint-embedding SSL learns a representation

$$
z=f_\theta(x)
$$

without semantic labels.

Yet downstream points often become class-clustered.

The paper asks whether that structure can be measured during pretraining and then used as an additional self-guidance signal.

The article should separate:

$$
\text{clusterability emerges}
$$

from

$$
\text{clusterability is explicitly reinforced}.
$$

## Mathematical angle

Represent self-assignment as a mapping

$$
z_i \mapsto q_i
$$

where $q_i$ is a soft or hard assignment derived from the representation.

Then the positive-feedback loop is

$$
\theta_t
\rightarrow
z_t
\rightarrow
q_t
\rightarrow
L_{assign}
\rightarrow
\theta_{t+1}.
$$

This resembles pseudo-labelling, except the pseudo-target is a representation cluster rather than a task class.

The critical question is whether the feedback sharpens meaningful structure or merely whatever partition happened to appear early.

## What to inspect

- Which network component has the strongest reported clustering properties?
- Which metrics define "better clustering"?
- Are labels used only for evaluation?
- How stable are assignments across seeds and augmentations?
- Does ReSA help linear probing, k-NN, clustering accuracy, or all three?
- What happens at fine-grained versus coarse-grained label levels?
- Are gains robust to changing cluster count?
- What happens when data have continuous rather than discrete semantic structure?

## Critical extension

Build a continuous latent-variable negative control.

If self-assignment creates increasingly discrete clusters on data generated from a continuum, then better internal clustering is not enough to establish semantic categories.

This would connect directly to the existing "Stability Is Not Truth" article.

## Proposed article structure

1. Why SSL representations cluster at all
2. Which representation layer is measured
3. ReSA as self-assignment feedback
4. The clustering metrics
5. Fine versus coarse semantics
6. Positive feedback and confirmation bias
7. Continuous negative control
8. What the paper establishes
9. What "semantic structure" should mean
10. When self-assignment is worth using

## Reproducibility plan

Use a small SSL encoder on:
- a labelled benchmark for external evaluation;
- a synthetic continuous manifold with no discrete classes.

Track clustering metrics over epochs in both cases.
