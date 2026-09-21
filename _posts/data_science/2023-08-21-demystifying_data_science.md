---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-08-21'
excerpt: Data science is not a catalogue of algorithms. It is a discipline for turning imperfect observations into defensible descriptions, predictions, and decisions under uncertainty.
header:
  image: /assets/images/headers/photo-data-science-air-quality.jpg
  og_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_image: /assets/images/headers/photo-data-science-air-quality.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-air-quality.jpg
  twitter_image: /assets/images/headers/photo-data-science-air-quality.jpg
keywords:
- Data science
- Statistical modeling
- Machine learning
- Causal inference
- Decision science
- Data quality
- Model validation
- Predictive analytics
- Experimental design
- Data-generating process
permalink: '/data-science/demystifying_data_science/'
redirect_from:
- '/data science/demystifying_data_science/'
seo_description: A rigorous introduction to data science as a discipline of measurement, statistical reasoning, prediction, causal inference, validation, and decision-making under uncertainty.
seo_title: 'What Data Science Actually Does'
seo_type: article
subtitle: From Measurement to Decisions Under Uncertainty
summary: Data science combines measurement, statistics, computation, domain knowledge, and decision analysis. Its value comes not from using fashionable algorithms but from defining the problem correctly, understanding how the data were generated, validating claims under realistic conditions, and connecting model outputs to decisions.
tags:
- Data Science
- Statistics
- Machine Learning
- Causal Inference
title: Demystifying Data Science
---

![Data has better idea - Demystifying Data Science](/assets/images/data_has_better_idea.jpg){: width="4032" height="3024" loading="lazy"}

Data science is often introduced as a mixture of statistics, computer science, and domain knowledge. That description is broadly correct, but it says little about the intellectual work that makes a data-science project succeed or fail. A project can contain sophisticated software, large datasets, and modern machine-learning models and still answer the wrong question. Conversely, a carefully designed analysis with a simple model can be decisive when the target, assumptions, measurements, and decision context are clear. The useful unit of analysis is therefore not the algorithm. It is the chain that connects a real-world question to observations, a statistical or computational representation, an uncertainty statement, and an action.

That distinction matters because data do not arrive as neutral facts. They are produced by measurement systems, business processes, experiments, sensors, surveys, transactions, and human decisions. Each mechanism determines what can be learned. A customer database records customers who interacted with the company, not an abstract population of all possible customers. A hospital database reflects admission criteria, clinical workflows, missingness, and treatment decisions. A machine sensor observes a physical process through calibration error, sampling frequency, drift, and failure modes. Before choosing a model, one should understand that observation process.

## Start with the question, not the model

Most data-science work can be organised around a small number of inferential targets. A descriptive question asks what happened in the observed data. A predictive question asks how well an outcome can be forecast for new observations drawn from a relevant future population. A causal question asks how an outcome would change under an intervention. A decision problem asks which action should be taken once uncertainty, costs, constraints, and competing objectives are considered. These targets overlap, but they are not interchangeable.

Suppose sales fall after a price increase. A descriptive analysis can quantify the change. A predictive model can estimate future sales from price and other covariates. Neither result, by itself, identifies the causal effect of the price change, because competitors, promotions, seasonality, and customer composition may also have changed. A causal analysis requires a design or assumptions that make the relevant counterfactual comparison credible. Even after a causal effect is estimated, the business decision is not automatic: margin, inventory, long-run customer value, and strategic constraints still determine whether the price change is desirable.

This is why "data-driven decision-making" should not mean replacing judgment with a model score. A defensible workflow makes the judgment explicit. It states the target, the information available at decision time, the losses associated with different errors, and the conditions under which historical evidence is relevant to the future decision.

## Data quality is part of the model

Cleaning data is sometimes presented as a preliminary engineering task that occurs before the "real" analysis. In serious work, data quality is inseparable from inference. Missing values may arise because a sensor failed, because a clinician chose not to order a test, because a customer stopped interacting with a service, or because a survey respondent declined to answer. Those mechanisms have different consequences. Treating them all as blank cells to be imputed ignores information about how the dataset was generated.

The same applies to labels. A fraud label may mean confirmed fraud, investigated fraud, chargeback fraud, or a rule-based proxy. A churn label may depend on an arbitrary inactivity window. A failure label in predictive maintenance may reflect the maintenance policy rather than the latent degradation process. If the operational definition changes, the statistical target changes with it. The first task is therefore to define the observation unit, outcome, predictors, time origin, censoring rules, and provenance of each variable.

Measurement error also matters. If a predictor is noisy, the effect is not limited to a small reduction in model accuracy. Classical measurement error can attenuate regression coefficients, misclassification can distort associations, and sensor replacement can create distribution shifts that look like changes in the underlying process. The measurement system should be documented as carefully as the model.

## Statistical models encode assumptions

A model is not a machine that extracts truth from data. It is a set of assumptions that defines which patterns are treated as signal, which variation is treated as noise, and how information is pooled. Linear regression assumes a particular conditional mean structure. A Gaussian process encodes beliefs about smoothness and covariance. A random forest partitions the feature space through an ensemble of trees. A neural network represents a flexible function class whose practical behaviour depends on architecture, regularisation, optimisation, and training data.

The relevant question is therefore not whether a model is "advanced". It is whether its assumptions and inductive biases are suitable for the target. In many scientific or operational settings, interpretability, uncertainty quantification, sample size, extrapolation behaviour, calibration, and computational constraints matter as much as raw predictive accuracy. A transparent parametric model can be preferable when it captures the structure of the problem and permits direct scrutiny of its assumptions.

Model complexity should be earned by out-of-sample evidence. A flexible method has more ways to fit accidental structure. Cross-validation, rolling-origin evaluation, external validation, or a genuinely held-out test set should reproduce the way the model will encounter future data. Randomly shuffling observations is inappropriate when future observations differ systematically from past observations or when multiple rows belong to the same patient, customer, machine, or site.

## Prediction, probability, and decisions are different layers

Many deployed systems ultimately produce a probability or score. That number is useful only if its interpretation is understood. A classifier can rank cases well while producing badly calibrated probabilities. A model can be calibrated in the population but perform poorly for an operationally important subgroup. A threshold that maximises the F1 score may be inappropriate when false negatives and false positives have very different consequences.

For a probabilistic classifier with estimated risk $\hat p(x)$, a decision threshold should ideally arise from the loss of the available actions. If action $a_1$ has a false-positive cost $C_{FP}$ and failing to act has a false-negative cost $C_{FN}$, the optimal threshold under a simple two-action model depends on those costs, not on an arbitrary convention such as 0.5. Real decisions are often more complicated because capacity, fairness constraints, delayed outcomes, and downstream interventions also matter.

Forecasting has the same structure. A point forecast is not a full representation of uncertainty. Inventory, staffing, finance, and reliability decisions often require predictive distributions because the cost of under-prediction differs from the cost of over-prediction. Optimising a model for mean squared error and then using the point forecast inside a nonlinear decision rule can be inferior to modelling the distribution needed by the decision itself.

## Experiments and causal inference

When the question is causal, prediction is not enough. Randomised experiments remain powerful because treatment assignment breaks systematic links between treatment and pre-treatment confounders in expectation. Observational data require stronger assumptions and a clear causal structure. Regression adjustment, propensity scores, inverse-probability weighting, instrumental variables, regression discontinuity, and difference-in-differences solve different identification problems; they are not interchangeable entries in a modelling menu.

The estimand should be stated before the estimator. Are we interested in an average treatment effect, an effect among treated units, a conditional effect for a subgroup, a policy value, or the effect of a dynamic treatment regime? The answer determines what data and assumptions are needed. Without that discipline, it is easy to report a statistically precise estimate of a quantity that does not correspond to the actual decision.

Experiments also require attention to interference, attrition, non-compliance, repeated exposure, novelty effects, and multiple outcomes. Randomisation protects against some biases, but it does not eliminate poor measurement, missing data, or a badly chosen endpoint.

## Deployment changes the statistical problem

A model is not finished when validation metrics look good. Once deployed, it enters a system. Predictions can change human behaviour, which changes future data. Fraud models alter which transactions are investigated. Recommendation systems alter what users see and therefore what they click. Predictive-maintenance models change when machines are inspected or replaced, modifying the failure data used for future training. These feedback loops mean that the post-deployment data-generating process may differ from the training process.

Monitoring should therefore include more than feature drift. Teams need to track outcome definitions, calibration, residual structure, data latency, missingness, schema changes, subgroup behaviour, operational thresholds, and whether the decision policy is still delivering the intended utility. Retraining on a schedule is not a substitute for diagnosing why performance changed.

Reproducibility is equally important. A defensible analysis records data versions, transformations, random seeds where relevant, package versions, model specifications, validation splits, and the exact code that generated reported numbers. Reproducibility does not guarantee correctness, but it makes claims inspectable and allows errors to be found.

## What data science contributes

The distinctive contribution of data science is not that it automates every decision or that it discovers hidden truth in large datasets. It provides a disciplined way to reason with imperfect observations at scale. That discipline combines measurement, statistical modelling, computation, experimental design, domain knowledge, uncertainty quantification, and decision analysis.

The best projects usually ask a sequence of increasingly specific questions. What exactly is being measured? Which population and time period do the data represent? What quantity do we want to estimate or predict? Which assumptions identify that quantity? How will the model be validated under deployment-like conditions? What uncertainty remains? Which action will use the output, and what are the costs of being wrong? What could change after deployment?

Once those questions are answered, the choice between regression, tree ensembles, Bayesian models, Gaussian processes, neural networks, or no predictive model at all becomes much easier. Data science is most useful when the modelling method follows from the problem rather than when the problem is reshaped to justify the method.

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199-231.
- Cleveland, W. S. (2001). Data science: An action plan for expanding the technical areas of the field of statistics. *International Statistical Review*, 69(1), 21-26.
- Donoho, D. (2017). 50 years of data science. *Journal of Computational and Graphical Statistics*, 26(4), 745-766.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
