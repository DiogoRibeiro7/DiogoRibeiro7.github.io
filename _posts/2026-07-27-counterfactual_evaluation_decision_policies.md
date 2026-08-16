---
permalink: '/data-science/counterfactual_evaluation_decision_policies/'
title: Counterfactual Evaluation for Decision Policies
categories:
- Data Science
tags:
- Causal Inference
- Policy Evaluation
- Machine Learning
author_profile: false
seo_title: Counterfactual Evaluation for Decision Policies
seo_description: How to evaluate new targeting, triage, pricing, or recommendation policies using logged observational data.
excerpt: Counterfactual evaluation helps teams estimate how a new decision policy might perform before deploying it to users, patients, customers, or operations.
summary: This article explains logged bandit data, propensity scores, inverse probability weighting, doubly robust estimators, and practical risks in policy evaluation.
keywords:
- counterfactual evaluation
- off-policy evaluation
- propensity scores
- decision policies
- causal inference
classes: wide
date: '2026-07-27'
header:
  image: /assets/images/data_science_16.jpg
  og_image: /assets/images/data_science_16.jpg
  overlay_image: /assets/images/data_science_16.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_16.jpg
  twitter_image: /assets/images/data_science_16.jpg
---

Many machine learning systems recommend actions: who should receive a discount, which ticket should be escalated, which patient should be contacted, which machine should be inspected. Offline accuracy does not answer the most important question: what would happen if we used a different decision policy?

Counterfactual evaluation estimates the performance of a new policy using data collected under an old policy. It is essential when direct experimentation is expensive, risky, or slow.

## The Core Problem

For each historical case, we observe the action that was taken and the outcome that followed. We do not observe what would have happened under the other possible actions. That missing potential outcome is the counterfactual.

If the old policy mostly treated high-risk cases, a naive comparison of treated and untreated cases will be biased. The groups were different before treatment.

## Logged Decision Data

A useful decision log should include:

- context available at decision time;
- action taken;
- probability of taking that action under the logging policy;
- outcome;
- timestamp;
- any constraints or eligibility rules.

The action probability is often missing in legacy systems. Without it, reliable off-policy evaluation becomes much harder.

## Inverse Probability Weighting

Inverse probability weighting estimates how the new policy would perform by upweighting historical cases that match the new policy but were unlikely under the old policy.

The intuition is simple: if a rare historical action matches the new policy, it represents many similar cases the old policy almost never chose.

The weakness is variance. Very small propensities create very large weights, making estimates unstable.

## Doubly Robust Estimation

Doubly robust methods combine:

- a model for expected outcomes;
- a weighting correction based on propensities.

If either the outcome model or the propensity model is correct, the estimator can remain consistent under standard assumptions. In practice, doubly robust estimation is attractive because it can reduce variance while preserving some protection against model misspecification.

## Practical Failure Modes

Counterfactual evaluation depends on assumptions:

| Risk | Meaning | Mitigation |
|------|---------|------------|
| Unobserved confounding | important decision factors are missing | improve logging, use experiments where possible |
| Poor overlap | new policy chooses actions rarely seen before | restrict policy class or collect exploration data |
| Bad propensities | action probabilities are wrong | audit logging code and deterministic rules |
| Delayed outcomes | outcome window differs across actions | define comparable attribution windows |
| Policy interference | one decision affects other cases | use cluster-aware designs |

No estimator can recover information about actions the system never tried.

## Deployment Strategy

Use counterfactual evaluation as a gate, not as final proof.

1. Reject policies that look worse offline.
2. Stress-test policies by subgroup and time period.
3. Check overlap and effective sample size.
4. Deploy promising policies through a controlled experiment or staged rollout.
5. Continue monitoring after the policy changes the data-generating process.

## Conclusion

Counterfactual evaluation lets teams reason about action policies before exposing the system to full operational risk. It is most valuable when combined with careful logging, causal thinking, and incremental experimentation.

The discipline is to remember that a predictive model is not the same as a decision policy. Once a model changes actions, evaluation must ask what would happen under those actions.

## References

- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Dudik, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and learning. *ICML*.
- Bottou, L., et al. (2013). Counterfactual reasoning and learning systems: The example of computational advertising. *Journal of Machine Learning Research*, 14, 3207-3260.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
