---
author_profile: false
categories:
- Machine Learning
classes: wide
title: 'Diffusion Models Can Be Clusterers Too'
excerpt: 'A paper-driven article on CLUDI, which combines pretrained Vision Transformer features with self-supervised diffusion to produce stochastic cluster assignments.'
keywords:
- diffusion models
- clustering
- self-supervised learning
- CLUDI
- vision transformers
- ICML 2025
tags:
- Diffusion Models
- Clustering
- Self-Supervised Learning
- ICML 2025
seo_title: 'Clustering via Self-Supervised Diffusion'
seo_description: 'Technical reading of CLUDI from ICML 2025: what diffusion contributes to clustering, how stochastic assignments are learned, and whether the generative machinery earns its complexity.'
summary: 'A critical reading of Clustering via Self-Supervised Diffusion, focusing on the role of pretrained ViT features, diffusion-based stochastic cluster assignment, objective design, computational cost, and what should be compared against simpler clustering baselines.'
why_this_exists: 'Diffusion models are usually discussed as generators. CLUDI uses diffusion dynamics as part of an unsupervised clustering framework, making it a useful example of modern generative machinery being repurposed for structure discovery.'
evidence: 'Uziel, Chelly, Freifeld, and Pakman, Clustering via Self-Supervised Diffusion, ICML 2025, PMLR 267:60711-60726.'
methodology: 'Dissect the CLUDI pipeline into pretrained representation, diffusion process, assignment mechanism and clustering objective; then compare each contribution against simpler alternatives.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: What does a diffusion model add to unsupervised clustering beyond a strong pretrained representation and a simpler assignment model?
Claim: CLUDI uses diffusion to model stochastic cluster assignments and reports strong clustering accuracy with pretrained ViT features.
Counterclaim: The pretrained feature extractor may already contain most of the semantic structure; the incremental value of diffusion needs careful ablation.
Evidence object: ICML 2025 CLUDI paper, method ablations and benchmark comparisons.
Failure case: Improvement may shrink when compared against equally tuned clustering on the same pretrained features.
Reader payoff: Understand a modern diffusion-based clustering pipeline without treating diffusion as magic.
Exclusions: Do not explain diffusion models from first principles beyond what the clustering argument needs.
-->

## Paper

Roy Uziel, Irit Chelly, Oren Freifeld, and Ari Pakman.  
**Clustering via Self-Supervised Diffusion.**  
ICML 2025, PMLR 267:60711-60726.  
https://proceedings.mlr.press/v267/uziel25a.html

## Core question

The useful decomposition is

$$
x
\xrightarrow{\text{pretrained ViT}}
z
\xrightarrow{\text{diffusion clustering}}
\hat c.
$$

The article should ask how much information about $hat c$ comes from the pretrained representation and how much is added by the diffusion mechanism.

## Mathematical angle

Focus on the stochastic assignment distribution rather than the entire image-generation literature.

Questions to derive from the paper:
- What is diffused?
- What is the forward corruption process?
- What does the reverse model estimate?
- How are cluster assignments represented?
- Which loss prevents trivial assignment collapse?
- Is the final clustering a MAP assignment, expectation, sampling consensus, or another quantity?

## Critical ablation lens

A strong evaluation should compare:

1. K-means on the same ViT features;
2. Gaussian mixtures on the same features;
3. spectral clustering on the same affinity;
4. CLUDI without key self-supervised terms;
5. CLUDI full model.

The article should report whether the paper makes those comparisons cleanly.

## Compute question

Modern clustering quality should not be reported without computational context.

Track:
- pretrained feature extraction cost;
- diffusion training cost;
- inference/sampling cost;
- number of clustering runs or seeds;
- memory cost.

A one-point clustering gain can mean something very different depending on the compute multiplier.

## Proposed article structure

1. Diffusion is not only generation
2. The CLUDI pipeline
3. Stochastic cluster assignment
4. What the pretrained representation already gives
5. Why diffusion could help
6. Collapse and identifiability
7. Baselines and ablations
8. Compute versus gain
9. Failure cases
10. Where diffusion clustering may actually be useful

## Reproducibility plan

Start with public pretrained features from a small benchmark.

Compare simple clustering against a lightweight approximation to the paper's stochastic assignment idea before attempting the full diffusion pipeline.
