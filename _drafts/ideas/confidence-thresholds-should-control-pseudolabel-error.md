---
author_profile: false
categories:
- Machine Learning
- Statistics
classes: wide
title: 'Confidence Thresholds Should Control Pseudo-Label Error, Not Just Confidence'
excerpt: 'A paper-driven article on replacing heuristic pseudo-label thresholds with explicit error control and examining what that guarantee means under miscalibration and distribution shift.'
keywords:
- semi-supervised learning
- pseudo-labels
- confidence thresholds
- calibration
- selective risk
- ICML 2025
tags:
- Semi-Supervised Learning
- Pseudo-Labels
- Calibration
- ICML 2025
seo_title: 'Rethinking Confidence Thresholds in Pseudo-Label SSL'
seo_description: 'Technical reading of Vishwakarma et al. (ICML 2025) on principled pseudo-label scores and thresholds with explicit error control.'
summary: 'A critical reading of Rethinking Confidence Scores and Thresholds in Pseudolabeling-based SSL, focusing on selective risk, quality-quantity trade-offs, calibration, and whether explicit error control survives distribution shift.'
why_this_exists: 'Heuristic confidence thresholds such as 0.95 are common in SSL, but their numerical value has no fixed reliability interpretation without calibration. This paper proposes an explicit error-control knob and deserves a mathematical treatment.'
evidence: 'Vishwakarma et al., ICML 2025, PMLR 267:61582-61600.'
methodology: 'Derive the score-threshold selection problem, distinguish confidence from conditional pseudo-label error, inspect the paper’s guarantees and experiments, and build a small calibration-shift counterexample.'
reviewed_at: '2026-09-20'
---

<!--
Development contract
Question: Can pseudo-label acceptance be formulated around explicit error control rather than heuristic confidence thresholds?
Claim: The useful object is selective pseudo-label risk at a chosen coverage, not the raw model confidence.
Counterclaim: Error control estimated on one distribution may not transfer under calibration shift or subgroup mismatch.
Evidence object: ICML 2025 paper, theoretical score/threshold construction, reported accuracy and training-efficiency experiments.
Failure case: Strong guarantees may depend on score quality and validation assumptions that fail under distribution shift.
Reader payoff: A mathematical framework for choosing pseudo-label thresholds by risk rather than habit.
Exclusions: Do not turn the article into a generic calibration tutorial.
-->

## Paper

Harit Vishwakarma, Yi Chen, Satya Sai Srinath Namburi Gnvv, Sui Jiet Tay, Ramya Korlakai Vinayak, and Frederic Sala.  
**Rethinking Confidence Scores and Thresholds in Pseudolabeling-based SSL.**  
ICML 2025, PMLR 267:61582-61600.  
https://proceedings.mlr.press/v267/vishwakarma25a.html

## Core question

What should a pseudo-label threshold actually control?

The usual rule

$$
max_k p_	heta(k\mid x) \ge \tau
$$

controls only the model's reported confidence.

The quantity we really care about is closer to

$$
R_{PL}(\tau)
=
P\{\hat Y\neq Y\mid S(X)\ge\tau\},
$$

where $S$ is the acceptance score.

The article should examine how the paper turns this into an explicit quality-versus-quantity problem.

## Mathematical angle

Develop the coverage-risk curve

$$
\Gamma(\tau)=P\{S(X)\ge\tau\},
$$

$$
R_{PL}(\tau)=P\{\hat Y\neq Y\mid S(X)\ge\tau\}.
$$

Then separate three objects:

1. raw confidence;
2. score calibration or ranking quality;
3. selective pseudo-label error.

A useful critical extension is to ask what happens if

$$
P_{train}(Y\mid S)
\neq
P_{unlabelled}(Y\mid S).
$$

That gives a direct distribution-shift stress test.

## What to inspect in the paper

- How is the error-control parameter defined?
- Is the control finite-sample, asymptotic, empirical, or probabilistic?
- What data are needed to learn the score and threshold?
- Does the method spend scarce trusted labels to estimate pseudo-label quality?
- How does coverage change as the target error becomes stricter?
- Which SSL baselines receive the largest gain?
- Are improvements primarily due to better pseudo-label quality, faster training, or both?
- How sensitive are results to class imbalance and calibration drift?

## Proposed article structure

1. Why 0.95 is not a reliability guarantee
2. Selective risk is the right object
3. The paper's score-and-threshold framework
4. Quality versus quantity of pseudo-labels
5. Where calibration enters
6. A shift counterexample
7. Experimental evidence
8. What the paper establishes
9. What it does not establish
10. Practical threshold-selection protocol

## Reproducibility plan

Build a binary synthetic experiment with:
- calibrated scores;
- overconfident scores;
- covariate shift that preserves ranking but changes calibration;
- equal target threshold across settings.

Plot pseudo-label error versus accepted coverage.

The figure should make visible that the same numerical confidence threshold can correspond to different empirical error rates.
