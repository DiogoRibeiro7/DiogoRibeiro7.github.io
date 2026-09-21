---
permalink: '/statistics/sample_size/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2023-09-27'
excerpt: "Sample size should be derived from the estimand, design, effect size, uncertainty target, and error rates, not from a universal rule that more data are always better."
header:
  image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  og_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-logistic-pdf.jpg
  twitter_image: /assets/images/headers/photo-statistics-logistic-pdf.jpg
keywords:
- Sample size
- Statistical power
- Effect size
- Precision
- Experimental design
- Confidence intervals
- Clustered data
seo_description: "How to reason about sample size from power, effect size, precision, design effects, clustering, multiplicity, and practical constraints."
seo_title: 'Sample Size: Power, Precision, and Design'
seo_type: article
subtitle: "Power, Precision, and Design"
tags:
- Data Analysis
- Sample Size
- Experimental Design
title: "Sample Size: Power, Precision, and Design"
---

![Sample 1 - Sample Size](/assets/images/sample_1.png){: width="320" height="126" loading="lazy"}

Sample size is not a universal measure of study quality. More observations can reduce sampling variability, but they do not repair biased measurement, selection bias, confounding, leakage, or a badly defined estimand. A very large sample can estimate the wrong quantity with great precision.

The useful question is therefore not "How large should the sample be?" in isolation. It is "How much information is required for this estimand under this design and decision criterion?"

## Precision and the square-root law

For many simple estimators, standard error decreases approximately as

$$
\operatorname{SE}\propto \frac{1}{\sqrt n}.
$$

This means diminishing returns. To cut a standard error in half, one often needs roughly four times as many independent observations.

For a mean with standard deviation $\sigma$,

$$
\operatorname{SE}(\bar X)=\frac{\sigma}{\sqrt n}.
$$

If the goal is an interval with half-width $h$, a normal approximation gives roughly

$$
n\approx \left(\frac{z_{1-\alpha/2}\sigma}{h}\right)^2.
$$

This is a precision calculation, not a power calculation.

## Power depends on an alternative

Power is the probability of rejecting the null under a specified alternative:

$$
\pi(\theta)=P_\theta(\text{reject }H_0).
$$

A sample-size calculation therefore needs an effect size or alternative hypothesis. Asking for 80% or 90% power without specifying the effect to be detected is incomplete.

For a simple two-group mean comparison with common variance, the required sample size depends on the standardized effect

$$
d=\frac{\mu_1-\mu_0}{\sigma},
$$

along with alpha, desired power, allocation ratio, and test specification.

## Minimum detectable effect

Sometimes the design starts from a fixed budget or feasible sample size. Then the more honest question is the minimum detectable effect rather than pretending the sample can be chosen freely.

A minimum detectable effect should be interpreted substantively. A study can have enough power to detect a tiny effect that has no practical relevance, or too little power to detect the smallest effect that would actually change a decision.

## Larger samples make trivial effects significant

With sufficiently large n, very small deviations from a null model can produce small p-values. Statistical significance therefore becomes easier as sample size increases.

This is not a defect of hypothesis testing; it is a reminder to report effect sizes and uncertainty. Practical importance is a scientific or decision question, not a p-value threshold.

## Dependence reduces effective information

One thousand independent observations and one thousand highly correlated observations do not contain the same information.

For clustered data, a common approximation uses the design effect

$$
DE=1+(m-1)\rho,
$$

where m is average cluster size and $\rho$ the intraclass correlation. The effective sample size is then roughly

$$
n_{\mathrm{eff}}\approx \frac{n}{DE}.
$$

Repeated measures, households, hospitals, schools, devices, and spatial units all require attention to dependence.

## Attrition and missing data

Planned sample size should account for expected loss to follow-up, but merely inflating n does not solve informative missingness.

If dropout depends on prognosis, treatment response, or unobserved outcomes, the missingness mechanism can bias estimates. The design should include retention strategies and a principled analysis plan, not only an attrition multiplier.

## Multiple outcomes and subgroup analyses

If a study has several primary outcomes, many treatment arms, or planned subgroup analyses, the sample-size problem changes. Multiplicity can reduce power after error-rate control, while interaction effects often require much larger samples than main effects.

A design powered for an overall treatment effect may be severely underpowered for heterogeneity of treatment effect.

## Prediction models have different requirements

Sample-size planning for prediction is not the same as planning a two-sample test. Relevant quantities include outcome prevalence, number of candidate parameters, expected signal strength, shrinkage, calibration, and the intended validation strategy.

Rules such as "ten events per variable" are too crude to serve as universal design criteria. Modern prediction-model planning should be based on expected model complexity and target performance rather than a single heuristic.

## External validity is not bought with n

A huge convenience sample can be less informative about a target population than a smaller probability sample. Representativeness depends on sampling design and transportability assumptions.

If the sample excludes important subpopulations, increasing n within the same biased frame only estimates the restricted population more precisely.

## Simulation is often the right tool

When the design includes clustering, censoring, nonlinearity, adaptive rules, complex estimators, or several endpoints, analytic formulas can become unreliable or unavailable.

A simulation-based design can generate data under plausible parameter values, apply the planned analysis, and estimate operating characteristics such as power, bias, interval width, and coverage.

## Conclusion

Sample size should follow from the inferential target. The central ingredients are effect size, precision, alpha, power, dependence, missingness, model complexity, multiplicity, and the intended decision.

More data are useful when they add relevant information. They are not a substitute for good measurement or good design.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Lakens, D. (2022). Sample Size Justification.
- Riley, R. D., et al. (2020). Calculating the sample size required for developing a clinical prediction model.
- van Belle, G. (2008). *Statistical Rules of Thumb*.
