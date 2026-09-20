---
author_profile: false
categories:
- Machine Learning
classes: wide
title: 'Does Classical Semi-Supervised Learning Still Help Once You Have a Foundation Model?'
excerpt: 'A critical reading of recent evidence that labelled-only parameter-efficient fine-tuning can match classical semi-supervised learning when strong vision foundation models provide the representation.'
keywords:
- foundation models
- semi-supervised learning
- parameter-efficient fine-tuning
- self-training
- pseudo-labels
- vision foundation models
tags:
- Semi-Supervised Learning
- Foundation Models
- PEFT
- Self-Training
seo_title: 'Revisiting Semi-Supervised Learning in the Era of Foundation Models'
seo_description: 'A technical reading of Zhang et al. on whether unlabeled data still adds value once strong pretrained vision representations are available.'
summary: 'A paper-driven article asking whether modern pretrained representations absorb part of the role traditionally played by unlabeled data in SSL, and when self-training still adds information beyond labelled-only PEFT.'
why_this_exists: 'The classical SSL question changes when the representation has already been learned from enormous external datasets. The marginal value of local unlabeled data should be measured against a strong labelled-only foundation-model baseline.'
evidence: 'Ping Zhang, Zheda Mai, Quang-Huy Nguyen, and Wei-Lun Chao, Revisiting semi-supervised learning in the era of foundation models, arXiv:2503.09707.'
methodology: 'Separate representation learning from task adaptation, compare labelled-only PEFT with SSL, inspect benchmark construction and pseudo-label ensembling, and formulate the incremental-information question explicitly.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: Does local unlabeled data still materially help once a strong foundation model supplies the representation?
Claim: The correct baseline for modern SSL is not training from scratch; it is strong labelled-only adaptation of a pretrained model.
Counterclaim: Foundation pretraining itself used vast unlabeled data, so saying "SSL is unnecessary" can simply move the unlabeled-learning stage upstream.
Evidence object: Zhang et al. 2025 benchmarks and PEFT/self-training comparisons.
Failure case: Results may depend strongly on backbone, pretraining corpus, domain shift, and label budget.
Reader payoff: A modern definition of what incremental SSL value means in the foundation-model era.
Exclusions: Do not turn this into a general foundation-model survey.
-->

## Paper

Ping Zhang, Zheda Mai, Quang-Huy Nguyen, and Wei-Lun Chao.  
**Revisiting semi-supervised learning in the era of foundation models.**  
arXiv:2503.09707, 2025.  
https://arxiv.org/abs/2503.09707

## Core question

Classical SSL asks whether

$$
\mathcal D_L + \mathcal D_U
$$

outperforms learning from

$$
\mathcal D_L
$$

alone.

With a foundation model, that comparison is incomplete because the representation already contains information learned from a much larger external corpus.

The relevant comparison becomes

$$
\text{pretrained representation} + \mathcal D_L
$$

versus

$$
\text{pretrained representation} + \mathcal D_L + \mathcal D_U.
$$

The article should measure the *incremental* value of the local unlabeled set.

## Mathematical angle

Write prediction as

$$
f_{\phi,\theta}(x),
$$

where $\phi$ is a pretrained representation and $\theta$ is task adaptation.

Classical SSL often uses $mathcal D_U$ to improve both representation and boundary estimation.

With frozen or lightly adapted $phi$, much of the geometry may already be fixed.

The question becomes whether

$$
I(Y;\mathcal D_U\mid \phi,\mathcal D_L)
$$

is large enough to justify the extra machinery.

This is not a directly estimable mutual information quantity in practice, but it gives the article a clean conceptual frame.

## What to inspect

- How are the new benchmarks chosen?
- Why do frozen VFMs underperform on them?
- Which PEFT methods are used?
- How strong is the labelled-only baseline?
- When does SSL still improve?
- Does pseudo-label ensembling work because of backbone diversity, adapter diversity, or both?
- Are the unlabeled examples from the same distribution as the labelled set?
- How does performance change with label count?
- Is the computation cost of multiple backbones included in the comparison?

## Critical angle

A result such as

> labelled-only PEFT matches SSL

does not imply that unlabeled representation learning has become irrelevant.

The foundation model itself is largely the product of large-scale unlabeled or weakly labelled pretraining.

The more precise conclusion may be:

> once a strong externally learned representation is available, the marginal value of *additional in-domain unlabeled data* can shrink.

That is a much better statement.

## Proposed article structure

1. SSL before foundation models
2. The baseline has changed
3. Representation versus adaptation
4. What the paper benchmarks
5. Why labelled-only PEFT is surprisingly strong
6. Self-training with model ensembles
7. Where local unlabeled data still helps
8. Cost and compute accounting
9. External pretraining as hidden unlabeled data
10. What modern SSL should benchmark against

## Reproducibility plan

Use one public vision foundation model and a small labelled benchmark. Compare:
- linear probe;
- LoRA or another PEFT method;
- simple self-training;
- ensemble pseudo-labels if feasible.

Report gain per additional unlabeled example and gain per unit of additional compute.
