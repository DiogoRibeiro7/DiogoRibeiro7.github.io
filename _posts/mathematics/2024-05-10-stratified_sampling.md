---
permalink: '/mathematics/stratified_sampling/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-05-10'
header:
  image: /assets/images/headers/photo-mathematics-klein-quartic.jpg
  og_image: /assets/images/headers/photo-mathematics-klein-quartic.jpg
  overlay_image: /assets/images/headers/photo-mathematics-klein-quartic.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-klein-quartic.jpg
  twitter_image: /assets/images/headers/photo-mathematics-klein-quartic.jpg
redirect_from:
- '/mathematics/statistics/data science/stratified_sampling/'
seo_description: How stratified sampling improves representativeness and accuracy by dividing a population into subgroups before sampling.
seo_title: 'Stratified Sampling: Methods and Applications'
seo_type: article
subtitle: A Key to Representative Research
tags:
- Statistical Modeling
- Model Evaluation
- Sample Size
title: Stratified Sampling
---

## Abstract

Stratified sampling divides a finite population into non-overlapping strata and samples independently within each stratum. Its main statistical advantage is precision: when units within strata are relatively homogeneous for the outcome of interest, a stratified estimator can have substantially lower variance than a simple random sample of the same total size. Stratification does not automatically remove bias, and irrelevant stratification usually costs efficiency rather than creating bias when the estimator uses the correct design weights.

## Setup and Estimator

Suppose a population of size $N$ is partitioned into $H$ strata. Stratum $h$ contains $N_h$ units, with

$$
N = \sum_{h=1}^{H} N_h.
$$

Draw a simple random sample without replacement of size $n_h$ from each stratum. Define the stratum weight

$$
W_h = \frac{N_h}{N}
$$

and let $\bar{y}_h$ be the sample mean in stratum $h$. The standard stratified estimator of the population mean is

$$
\bar{y}_{st} = \sum_{h=1}^{H} W_h \bar{y}_h.
$$

This estimator is design-unbiased under the sampling design, whether the strata are highly informative or almost irrelevant. Good stratification improves precision. Poor stratification can waste sample size, but it does not by itself bias the correctly weighted estimator.

For a population total,

$$
\hat{Y}_{st} = \sum_{h=1}^{H} N_h \bar{y}_h.
$$

## Variance

Let $S_h^2$ denote the finite-population variance within stratum $h$ and let

$$
f_h = \frac{n_h}{N_h}
$$

be the sampling fraction. Under independent simple random sampling within strata,

$$
\operatorname{Var}(\bar{y}_{st})
=
\sum_{h=1}^{H}
W_h^2
\left(1-f_h\right)
\frac{S_h^2}{n_h}.
$$

An estimator of this variance replaces $S_h^2$ by the sample variance $s_h^2$.

This formula shows where the gain comes from. Stratification is useful when the partition produces small within-stratum variances. If the strata do not separate units with different outcome behavior, the variance reduction may be negligible.

## Allocation of the Sample

### Proportional allocation

A simple choice is

$$
n_h = n \frac{N_h}{N},
$$

so each stratum receives a sample proportional to its population size. With equal sampling fractions, the resulting design is self-weighting.

### Neyman allocation

If sampling costs are approximately equal and the within-stratum standard deviations are known or can be estimated, Neyman allocation minimizes the variance of the stratified mean for a fixed total sample size:

$$
n_h
=
n
\frac{N_h S_h}
{\sum_{j=1}^{H} N_j S_j}.
$$

Larger and more variable strata receive more observations.

If per-unit sampling costs $c_h$ differ by stratum, the cost-sensitive optimum is proportional to

$$
\frac{N_h S_h}{\sqrt{c_h}}.
$$

## Worked Example

Consider a population with two strata:

| Stratum | $N_h$ | $S_h$ |
|---|---:|---:|
| 1 | 800 | 10 |
| 2 | 200 | 30 |

Suppose the total sample size is $n=100$.

Under proportional allocation,

$$
n_1=80, \qquad n_2=20.
$$

Under Neyman allocation,

$$
n_1
=
100\frac{800(10)}{800(10)+200(30)}
\approx 57,
$$

and

$$
n_2
=
100\frac{200(30)}{800(10)+200(30)}
\approx 43.
$$

Although the second stratum is much smaller, it receives nearly half the sample because it is much more variable. This is the practical point of optimal allocation: sample size follows both population size and information content.

## Stratified Sampling and Cluster Sampling

Strata and clusters are often confused because both partition the population, but the design logic is almost opposite.

In stratified sampling, units are sampled from every stratum. We usually want units within each stratum to be relatively similar with respect to the outcome, because this reduces within-stratum variance.

In one-stage cluster sampling, only some clusters are selected and the sampled clusters contribute many or all of their units. For efficiency, clusters would ideally resemble small versions of the full population. In practice, units within a geographic or organizational cluster are often positively correlated, which increases variance through the design effect.

Stratification is therefore primarily a precision and representation device. Cluster sampling is often a cost and logistics device.

## Choosing Strata

Useful stratification variables are known for the population before sampling, define mutually exclusive and exhaustive groups, and are associated with the outcome or with important analytic domains.

Examples include age bands in a health survey, school type in an education study, region in a national household survey, or customer segment in market research.

If the stratification variable is weakly related to the outcome, the design may provide little precision gain. This is not the same as inducing bias. Bias arises from problems such as incorrect weights, frame errors, nonresponse, measurement error, or a sampling mechanism that is not implemented as designed.

## Disproportionate Sampling and Weights

Researchers often oversample small or policy-important strata to obtain adequate subgroup precision. In that case, the raw sample is not proportional to the population.

The design weight for a sampled unit in stratum $h$ is the inverse inclusion probability:

$$
d_h = \frac{N_h}{n_h}.
$$

These weights must be used when estimating population quantities. Ignoring them can bias population-level estimates when sampling fractions differ across strata.

## Applications

In health research, stratification can guarantee adequate representation of age, sex, region, or risk groups. In education, it can ensure that school types or socioeconomic groups are represented. In market research, it can support precise estimates for customer segments that would be too sparse under simple random sampling.

The method is especially useful when subgroup estimates are part of the research question. It guarantees planned sample sizes within those domains rather than hoping that a simple random sample happens to contain enough observations.

## Limitations

Stratified sampling requires a sampling frame containing the stratification variables before selection. Poor classification, stale frame information, or operational mistakes can undermine the intended design. Many strata can also make fieldwork and variance estimation more complicated.

Optimal allocation requires advance information about $S_h$, which may be unavailable. Pilot data, historical studies, or conservative approximations are often used instead.

## Conclusion

The mathematics of stratified sampling is simple but important. The population mean is reconstructed as a weighted sum of stratum means, and the variance is a weighted sum of within-stratum variances. Good strata reduce those within-stratum variances. Allocation then decides where additional observations are most valuable.

The key distinction is worth keeping explicit: stratification is not a generic cure for bias. With correct design weights, its main statistical benefit is improved precision and guaranteed representation of chosen subgroups.

## References

- Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Lohr, S. L. (2010). *Sampling: Design and Analysis* (2nd ed.). Brooks/Cole.
- Neyman, J. (1934). On the two different aspects of the representative method: The method of stratified sampling and the method of purposive selection. *Journal of the Royal Statistical Society*, 97(4), 558–625.
- Särndal, C.-E., Swensson, B., & Wretman, J. (1992). *Model Assisted Survey Sampling*. Springer.
- Bethel, J. (1989). Sample allocation in multivariate surveys. *Survey Methodology*, 15(1), 47–57.
