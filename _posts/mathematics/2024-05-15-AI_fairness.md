---
permalink: '/mathematics/AI_fairness/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-15'
header:
  image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  og_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  overlay_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-polyhedra.jpg
  twitter_image: /assets/images/headers/photo-mathematics-polyhedra.jpg
redirect_from:
- '/mathematics/statistics/data science/machine learning/ethics research/AI_fairness/'
seo_description: "A rigorous treatment of fairness in machine learning, covering incompatible group metrics, calibration, measurement and label bias, individual and counterfactual fairness, uncertainty, threshold policy, and feedback effects."
seo_title: "AI Fairness: Metrics, Trade-offs, Causal Assumptions, and Governance"
seo_type: article
subtitle: "Why fairness is a property of decision systems, not a single score"
tags:
- Ethics
- Machine Learning
- Fairness
- Statistical Decision Theory
title: "AI Fairness: Metrics, Trade-offs, Causal Assumptions, and Governance"
---

Fairness in machine learning is often presented as though it were a technical property that can be measured once the model has been trained. That framing is too narrow. A predictive model sits inside a larger decision system containing a population, a measurement process, a target definition, a prediction rule, a threshold or allocation policy, and a set of downstream consequences. A fairness analysis that begins only with the final predictions can therefore miss disparities that were created earlier, for example through who was observed, how labels were defined, which outcomes were recorded, or how resources were distributed before the model ever saw the data.

The mathematics of fairness reflects this broader problem. There is no single scalar quantity called fairness. Instead, different formal criteria encode different notions of what should be equal across groups: selection rates, true-positive rates, false-positive rates, calibration, treatment of similar individuals, or counterfactual invariance. These criteria are not interchangeable, and in many realistic settings they cannot all be satisfied simultaneously. The central statistical question is therefore not whether a model is “fair” in the abstract, but which disparity is considered harmful, which population is affected, what causal or normative assumptions justify the chosen criterion, and what consequences follow from optimizing it.

## Group fairness criteria encode different objectives

Let $A$ denote a protected or otherwise policy-relevant group attribute, $Y$ the observed target, $\widehat Y$ a binary decision, and $\widehat p$ a predicted probability. **Demographic parity** requires the positive decision rate to be equal across groups,

$$
P(\widehat Y=1\mid A=a)
=
P(\widehat Y=1\mid A=b).
$$

This criterion is concerned with allocation. If a decision corresponds to access to a scarce opportunity, parity can be a relevant policy objective because it constrains how often different groups receive that opportunity. It says nothing, however, about whether the selected individuals have the same outcome distribution or whether error rates are equal. A model can satisfy demographic parity while producing very different false-positive and false-negative rates across groups.

**Equal opportunity** instead constrains the true-positive rate,

$$
P(\widehat Y=1\mid Y=1,A=a)
=
P(\widehat Y=1\mid Y=1,A=b).
$$

The criterion asks whether individuals who truly satisfy the positive condition are equally likely to receive the positive decision. In a setting where false negatives represent denial of an opportunity to qualified individuals, this can be more relevant than demographic parity. Equal opportunity does not constrain false positives, so two groups can satisfy the criterion while still experiencing different rates of inappropriate positive decisions.

**Equalized odds** is stronger. It requires the decision to be conditionally independent of group membership given the observed outcome,

$$
\widehat Y
\perp
A
\mid
Y.
$$

Equivalently, both true-positive and false-positive rates should agree across groups. This criterion equalizes the confusion-matrix error structure, but it still depends on the meaning and quality of the observed label $Y$. If the label is systematically biased or reflects unequal historical processes, equalizing errors relative to that label does not necessarily equalize error relative to the underlying construct that matters scientifically.

That dependence on labels is central. Fairness metrics are frequently computed as though $Y$ were an objective ground truth. In many applications it is not. Arrest is not the same construct as offending, recorded default is not the same as financial capacity, diagnosis is not the same as disease burden, and promotion is not the same as employee contribution. If the observed label is affected by differential surveillance, access, documentation, or prior decisions, then every error-rate metric conditions on a measurement process that may already encode disparity.

## Calibration measures something different

Calibration concerns the interpretation of predicted probabilities rather than equality of decisions. A risk model is calibrated within group $A=a$ if, among individuals assigned risk approximately $r$, the observed event frequency is also approximately $r$:

$$
P(Y=1\mid \widehat p=r,A=a)
\approx
r.
$$

If two groups are both calibrated, a predicted probability of 0.2 has the same empirical meaning in each group. This is a desirable property when predictions are used as quantitative risks rather than only as rankings.

Calibration can conflict with equalized error rates when outcome prevalences differ across groups and the predictor is imperfect. This incompatibility is not a failure of optimization software. It is a mathematical consequence of imposing several distinct conditional-independence requirements on the same joint distribution. Kleinberg, Mullainathan, and Raghavan, and independently Chouldechova, formalized versions of this conflict for risk scores: when base rates differ, one generally cannot have calibration within groups together with equal false-positive and false-negative behavior unless prediction is essentially perfect or the base rates coincide.

This result is important because it prevents the fairness problem from being reduced to “choose all the good metrics.” Improving one criterion can worsen another. A decision-maker therefore has to state which form of disparity is relevant to the application and why. The choice is normative and contextual, although the consequences of the choice can and should be analyzed statistically.

## Measurement, labels, and the data-generating process

A fairness audit can start too late if it begins with model outputs. The observed dataset is generated by a chain of mechanisms,

$$
\text{population}
\rightarrow
\text{selection}
\rightarrow
\text{measurement}
\rightarrow
\text{label}
\rightarrow
\text{model}
\rightarrow
\text{decision}.
$$

Disparity can enter at every stage. Some groups may be underrepresented because access to the service being studied is unequal. Features may be measured with different error distributions. Labels may be more likely to be recorded for one group because monitoring intensity differs. Historical decisions may determine who appears in the training set. Even a statistically perfect learner will reproduce the conditional structure of the data it receives unless the modeling strategy explicitly changes the target.

This observation also explains why **fairness through unawareness** is weak. Removing $A$ from the feature set does not imply that the model is independent of group membership. Other variables can act as proxies, especially when geography, education, occupation, language, income, network structure, or prior institutional decisions are correlated with $A$. Moreover, omitting the protected attribute can make auditing harder because group-specific performance cannot be estimated directly.

Protected attributes can therefore play different roles. They may be excluded from the final decision rule for legal, policy, or scientific reasons while still being required during validation to quantify disparities. Whether this is appropriate depends on the application and governing rules, but mathematically there is no contradiction: a variable need not be used for prediction in order to be necessary for evaluation.

## Thresholds are policy choices, not purely statistical parameters

Many deployed systems transform a score into a binary decision through a threshold,

$$
\widehat Y
=
\mathbf 1\{\widehat p\ge\tau\}.
$$

Changing $\tau$ changes true-positive and false-positive rates simultaneously. Group-specific thresholds can therefore be used to satisfy criteria such as equal opportunity or equalized odds. The procedure is mathematically straightforward, but the decision to apply different thresholds across groups is not merely a hyperparameter choice. It changes who receives benefits, burdens, investigation, credit, treatment, or other interventions.

The relevant analysis should therefore go beyond whether a metric can be equalized. It should examine the resulting confusion matrices, expected losses, capacity constraints, calibration after thresholding, and downstream effects. A threshold policy that improves one fairness metric may increase another form of harm. For example, equalizing false-negative rates can require accepting substantially different false-positive rates when score distributions differ.

Cost-sensitive decision theory provides a useful language here. If false positives and false negatives have costs $c_{FP}$ and $c_{FN}$, then an optimal threshold under a calibrated model depends on those costs and on the decision objective. Fairness constraints effectively add further restrictions to the optimization problem. The result is no longer simply a predictive model; it is a constrained decision rule.

## Individual fairness requires a defensible notion of similarity

Group metrics operate on aggregate rates and can conceal arbitrary treatment of individuals within each group. The principle of **individual fairness** proposes that similar individuals should be treated similarly. In abstract form, one seeks a decision function $f$ satisfying a Lipschitz-type condition,

$$
d_{\mathcal Y}(f(x_i),f(x_j))
\le
L\,d_{\mathcal X}(x_i,x_j),
$$

where $d_{\mathcal X}$ measures similarity between individuals and $d_{\mathcal Y}$ measures similarity between outcomes.

The difficulty is concentrated in the metric $d_{\mathcal X}$. Declaring two individuals similar requires substantive judgment about which differences are relevant to the decision. In lending, should employment history, postcode, family wealth, or educational opportunity contribute to similarity? In hiring, which career-path differences should count as legitimate? The mathematics can enforce consistency relative to a metric, but it cannot determine whether the metric itself is ethically or scientifically justified.

This is a recurring theme in fairness research: formal guarantees are conditional on the representation of the problem. A mathematically exact guarantee can be normatively empty if the distance function, labels, or causal assumptions have been chosen poorly.

## Counterfactual fairness is a causal concept

Counterfactual fairness attempts to express fairness through interventions in a structural causal model. The basic idea is that a prediction for an individual should remain invariant under a counterfactual intervention on a protected attribute, while preserving the causal structure of the remaining variables. If $A$ is the protected attribute and $\widehat Y$ the predictor, one asks whether the distribution of the counterfactual prediction would change under interventions such as $do(A=a)$ versus $do(A=a')$ for the same latent individual.

This cannot be implemented correctly by simply flipping the value of $A$ in a dataset while holding every other variable fixed. If $A$ causally affects education, income, health access, or other measured variables, then an intervention on $A$ propagates through the causal graph. Counterfactual fairness therefore depends on assumptions about causal structure, latent variables, and which causal pathways are considered permissible.

Those assumptions are often difficult to identify from observational data. Two analysts can agree on the observed distribution and disagree about the causal graph, producing different counterfactual fairness conclusions. The method is powerful because it makes causal assumptions explicit, but it does not eliminate the need for scientific and normative judgment.

## Fairness estimates have uncertainty

Fairness dashboards often report group metrics as though they were known exactly. In reality they are estimated from finite samples, sometimes from small subgroups. If one group contains very few positive outcomes, a true-positive rate can have substantial sampling uncertainty. Comparing point estimates alone can therefore exaggerate apparent disparities or hide important ones.

For a group-specific rate $\widehat p_g$, uncertainty can be represented through confidence intervals, Bayesian posterior intervals, bootstrap distributions, or hierarchical models when many related groups are analyzed. Intersectional analysis makes this especially important. Splitting by age, sex, geography, disability, ethnicity, or other characteristics can produce cells with very small counts. The desire to detect heterogeneous harm then collides with the variance of the estimators.

Multiple comparisons arise as well. Monitoring dozens of groups across multiple metrics and thresholds can produce apparent disparities by chance. Formal multiplicity adjustment is not always the only relevant solution because fairness auditing is partly exploratory, but uncertainty and selection should be reported rather than hidden.

## Feedback loops and dynamic effects

A static fairness metric measures the current relationship between predictions, outcomes, and groups. Deployment can change that relationship. Lending decisions affect future credit histories, recommendation systems affect exposure and consumption, hiring systems affect who gains experience, and fraud detection affects which transactions are investigated. The model therefore changes the data-generating process from which future models will be trained.

This creates feedback loops. Suppose one group receives fewer positive decisions because its initial scores are lower. If the positive decision itself creates opportunities to improve the future outcome, then the score gap can widen over time even if the original model satisfies a chosen static fairness criterion. Conversely, an intervention designed to equalize short-term decisions may have unexpected long-term effects if behavior adapts.

Dynamic fairness therefore requires more than repeated calculation of the same confusion-matrix metric. It may require longitudinal models, policy simulation, causal inference, or reinforcement-learning formulations that account for how decisions alter future states. The relevant horizon should be stated explicitly because a policy can appear fair in the short term and harmful over a longer period.

## Fairness cannot be separated from the decision system

The practical implication is that fairness analysis should begin with the decision and the harm, not with a software library. Toolkits such as Fairlearn and AI Fairness 360 are useful for computing metrics, exploring thresholds, and implementing some mitigation procedures, but they cannot determine which fairness definition is appropriate. That decision requires knowledge of the target population, the institutional process, the consequences of errors, the reliability of labels, and the legal or policy context.

A defensible workflow begins by defining the decision being made and the population to which it applies. The next step is to examine selection, measurement, and label construction before evaluating model performance. Relevant fairness criteria should then be selected because they correspond to specific harms, not because they are available in a dashboard. Group metrics should be reported with uncertainty, calibration should be checked separately from discrimination, and threshold policies should be evaluated on the decision scale rather than only through statistical scores.

After deployment, monitoring should include not only the fairness metrics originally optimized but also changes in population composition, label processes, calibration, subgroup sample size, and downstream outcomes. A system that remains numerically “fair” according to one frozen metric can become substantively unfair if the population or decision context changes.

## Conclusion

Fairness in machine learning is not a property that can be attached to a model independently of the system in which the model operates. Formal metrics are valuable because they force particular notions of equality to be stated mathematically, but they do not resolve the normative question of which equality matters. Demographic parity, equal opportunity, equalized odds, calibration, individual fairness, and counterfactual fairness each encode different assumptions and different notions of harm. Some of these criteria are mathematically incompatible when base rates differ, so there is no universal procedure that can satisfy every reasonable definition simultaneously.

The more useful question is therefore not whether an algorithm is fair, but whether the complete decision process treats relevant people in a defensible way under a clearly stated criterion. Answering that question requires attention to measurement, labels, uncertainty, causal structure, threshold policy, resource constraints, and feedback effects. Mathematics can expose trade-offs and quantify consequences; it cannot choose the social objective on its own.

A rigorous fairness analysis should make those choices visible. It should state what is being equalized, why that quantity is relevant, which assumptions make the metric meaningful, what uncertainty surrounds the estimate, and what other harms may increase when the chosen criterion is enforced. That is a much stronger standard than placing several fairness scores beside a model and declaring the problem solved.

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.
- Chouldechova, A. (2017). Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments. *Big Data*, 5(2), 153-163.
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness Through Awareness. *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference*, 214-226.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. *Advances in Neural Information Processing Systems*, 29.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent Trade-Offs in the Fair Determination of Risk Scores. *Proceedings of Innovations in Theoretical Computer Science*.
- Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual Fairness. *Advances in Neural Information Processing Systems*, 30.
