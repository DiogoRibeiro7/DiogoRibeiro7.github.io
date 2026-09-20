---
author_profile: false
categories:
- Machine Learning
classes: wide
title: 'Non-Stationary Pseudo-Labels May Carry More Boundary Information Than Stable Ones'
excerpt: 'A critical reading of an ICML 2025 paper arguing that predictions which switch class during training can be useful pseudo-label candidates rather than noise to discard.'
keywords:
- pseudo-labels
- training dynamics
- semi-supervised learning
- decision boundaries
- two-phase labels
- ICML 2025
tags:
- Semi-Supervised Learning
- Pseudo-Labels
- Training Dynamics
- ICML 2025
seo_title: 'When Non-Stationary Pseudo-Labels Are Informative'
seo_description: 'Technical reading of Pei et al. (ICML 2025) on two-phase pseudo-labels, training-dynamics metrics and the idea that class-switching predictions can identify informative boundary regions.'
summary: 'A paper-driven article on two-phase pseudo-label dynamics, asking whether prediction switches reveal useful decision-boundary information or simply identify unstable examples.'
why_this_exists: 'Most pseudo-label methods prefer high-confidence, temporally stable predictions. Pei et al. deliberately study a class of predictions that violates both preferences and report gains from retaining them.'
evidence: 'Pei et al., Non-Stationary Predictions May Be More Informative: Exploring Pseudo-Labels with a Two-Phase Pattern of Training Dynamics, ICML 2025, PMLR 267:48662-48678.'
methodology: 'Model prediction trajectories over epochs, define stationarity and switching events, inspect the proposed 2-phasic metric, and test whether class switches correspond to boundary information or generic optimization instability.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: Can changes in predicted class during training be informative rather than merely signs of unreliable pseudo-labels?
Claim: Two-phase prediction trajectories can identify examples carrying useful decision-boundary information.
Counterclaim: Prediction switches can also be caused by optimizer noise, learning-rate schedules, label noise, representation drift or poor initialization.
Evidence object: ICML 2025 paper, 2-phasic metric, eight-dataset experiments and reported gains on image and graph tasks.
Failure case: If switching frequency depends heavily on training schedule, the metric may describe the optimizer as much as the data.
Reader payoff: Treat the full training trajectory as information rather than reducing pseudo-label quality to one confidence score.
Exclusions: Do not repeat the existing confidence-calibration article.
-->

## Paper

Hongbin Pei, Jingxin Hai, Yu Li, Huiqi Deng, Denghao Ma, Jie Ma, Pinghui Wang, Jing Tao, and Xiaohong Guan.  
**Non-Stationary Predictions May Be More Informative: Exploring Pseudo-Labels with a Two-Phase Pattern of Training Dynamics.**  
ICML 2025, PMLR 267:48662-48678.  
https://proceedings.mlr.press/v267/pei25a.html

## Core question

Most pseudo-label selection collapses an example's history to one current score:

$$
p_t(y\mid x).
$$

This paper treats the trajectory

$$
\{p_t(y\mid x)\}_{t=1}^{T}
$$

as the object of interest.

A two-phase example initially supports one class and later switches to another.

The paper argues that such examples can carry useful information about the evolving decision boundary.

## Mathematical angle

Define predicted class trajectory

$$
c_t(x)
=
\arg\max_y p_t(y\mid x).
$$

A stable pseudo-label has approximately

$$
c_1(x)=\cdots=c_T(x).
$$

A two-phase trajectory has a change point

$$
\tau
$$

such that

$$
c_t(x)=a
\quad t<\tau,
$$

and

$$
c_t(x)=b
\quad t\ge\tau,
$$

with $a\neq b$.

This immediately connects the paper to change-point detection and temporal stability metrics.

## Critical question

Is the switch an intrinsic property of the example or a property of the training procedure?

Test dependence on:
- optimizer;
- learning rate;
- seed;
- batch order;
- augmentation strength;
- architecture;
- early stopping.

If the same example switches under one training schedule and stays stable under another, "two-phase" is not solely a data property.

## Reported result to examine

The paper reports that adding two-phase labels boosts existing pseudo-label methods, with average gains of about:
- 1.73 percentage points on image datasets;
- 1.92 percentage points on graph datasets.

The article should inspect variance across datasets rather than quote only the average.

## Proposed article structure

1. Why stable pseudo-labels became the default
2. Prediction trajectories contain more information
3. Defining two-phase behavior
4. The 2-phasic metric
5. Boundary examples versus optimizer instability
6. The tailored learning objective
7. Image and graph evidence
8. Sensitivity to training schedule
9. Relation to active learning
10. When unstable predictions are worth keeping

## Reproducibility plan

Train a small classifier on a two-dimensional synthetic problem.

Store

$$
p_t(y\mid x)
$$

for every unlabelled point across epochs.

Map:
- stable-correct;
- stable-wrong;
- two-phase-correct;
- multi-switch unstable.

Then plot those categories relative to the true decision boundary.

Repeat across several optimizers and seeds to test whether the two-phase identity is stable.
