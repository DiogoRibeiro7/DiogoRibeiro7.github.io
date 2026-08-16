---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-11-25'
excerpt: See how hypothesis testing helps draw meaningful conclusions from data in
  practical scenarios.
header:
  image: /assets/images/data_science_13.jpg
  og_image: /assets/images/data_science_13.jpg
  overlay_image: /assets/images/data_science_13.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_13.jpg
  twitter_image: /assets/images/data_science_13.jpg
keywords:
- Hypothesis testing
- P-values
- Significance
- Data science
seo_description: Learn how to apply hypothesis tests in real-world analyses and avoid
  common pitfalls when interpreting p-values and confidence levels.
seo_title: Applying Hypothesis Testing in the Real World
seo_type: article
summary: This post walks through frequentist hypothesis testing, showing how to formulate
  null and alternative hypotheses and interpret the results in practical data science
  tasks.
tags:
- Hypothesis Testing
- Statistics
- Experimental Design
title: Applying Hypothesis Testing in the Real World
---

Hypothesis testing allows data scientists to objectively assess whether an observed pattern is likely due to chance or reflects a genuine effect.

## Null vs. Alternative Hypotheses

Every test starts with a **null hypothesis**, representing the status quo, and an **alternative hypothesis**, representing a potential effect. By choosing a significance level and calculating a p-value, we can decide whether to reject the null hypothesis.

The logic is indirect, and worth stating carefully. We assume $H_0$ is true, work out how surprising the observed data would be under that assumption, and reject $H_0$ only if the data is surprising enough. Formally, the p-value is

$$
p = P\big(T(\text{data}) \ge t_{\text{obs}} \;\big|\; H_0 \text{ true}\big),
$$

where $T$ is the test statistic. Read that conditioning bar closely: the p-value is the probability of the data given the hypothesis, never the probability of the hypothesis given the data. Those two quantities can differ by orders of magnitude, and conflating them is the single most common error in applied statistics.

Failing to reject $H_0$ is not evidence that $H_0$ is true. It means the data was insufficient to rule it out, which is a statement about your sample size as much as about the world. If you want to argue for the absence of an effect, you need an equivalence test or a Bayesian analysis, not a non-significant p-value.

## The Two Ways to Be Wrong

Every decision rule trades off two errors:

|  | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error (rate $\alpha$) | Correct (power $= 1-\beta$) |
| **Fail to reject** | Correct | Type II error (rate $\beta$) |

Setting $\alpha = 0.05$ means accepting a 5% false-positive rate when the null is true. Lowering $\alpha$ reduces false positives but costs power. The only way to improve both at once is to collect more data or reduce measurement noise.

Power deserves more attention than it usually receives. An underpowered study is not merely inconclusive; it is actively misleading, because the effects that do reach significance in a small sample must be large, and are therefore systematically overestimated. This inflation is sometimes called the winner's curse, and it is a major driver of results that fail to replicate.

## Choosing a Test

The test follows from the structure of the question and the data:

- **Comparing two independent group means:** two-sample $t$-test, or Mann-Whitney U if the data is badly skewed or ordinal.
- **Comparing paired measurements:** paired $t$-test, or Wilcoxon signed-rank as the non-parametric counterpart.
- **Comparing three or more group means:** ANOVA, or Kruskal-Wallis without the normality assumption.
- **Testing association between categorical variables:** chi-square test of independence, or Fisher's exact test when expected counts are small.
- **Testing a regression coefficient:** $t$-test on the coefficient, with an $F$-test for groups of coefficients.

Each carries assumptions. The $t$-test assumes independent observations and approximate normality of the sampling distribution of the mean, and Welch's version should be the default because it does not additionally assume equal variances. The chi-square test relies on expected cell counts being large enough, conventionally at least 5.

## A Concrete Example

Suppose a checkout redesign is tested against the existing flow:

```python
import numpy as np
from scipy import stats

control   = np.array([  # seconds to complete checkout
    42, 51, 38, 47, 55, 44, 49, 41, 53, 46,
    50, 39, 45, 52, 48, 43, 56, 40, 47, 51])
treatment = np.array([
    38, 44, 35, 41, 47, 39, 43, 36, 45, 40,
    42, 34, 40, 46, 41, 37, 48, 35, 42, 44])

t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)

diff = control.mean() - treatment.mean()
pooled_sd = np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2)

print(f"difference in means: {diff:.2f} seconds")
print(f"t = {t_stat:.3f}, p = {p_value:.5f}")
print(f"Cohen's d          : {diff / pooled_sd:.3f}")
```

The p-value tells you the difference is unlikely under the null. It says nothing about whether a few seconds of checkout time matters to the business. That judgement requires the effect size and its confidence interval, which is why reporting a p-value alone is never sufficient.

## Common Pitfalls

Misinterpreting p-values or failing to consider effect sizes can lead to misguided conclusions. Always pair statistical significance with domain context to ensure results are meaningful. Several specific traps recur often enough to name.

**Multiple comparisons.** Testing 20 hypotheses at $\alpha = 0.05$ gives roughly a 64% chance of at least one false positive, since $1 - 0.95^{20} \approx 0.64$. Bonferroni correction divides $\alpha$ by the number of tests and is simple but conservative; controlling the false discovery rate via Benjamini-Hochberg is usually the better choice when many tests are genuinely exploratory.

**Optional stopping.** Checking results as data accumulates and stopping when $p < 0.05$ inflates the false-positive rate dramatically, because you are effectively running many tests. Fixing the sample size in advance, or using a sequential design built for repeated looks, is the fix.

**Dichotomising a continuum.** There is no meaningful difference between $p = 0.049$ and $p = 0.051$, yet the 0.05 threshold treats them as opposites. Report the actual value alongside an interval estimate.

**Confusing statistical and practical significance.** With a large enough sample, a trivially small effect becomes statistically detectable. Significance answers "is it distinguishable from zero", not "is it big enough to act on".

## Reporting Results Honestly

A useful report contains the effect size in the units people care about, a confidence interval showing the range of values compatible with the data, the sample size, the test used and why, and the p-value as one input among several. It also discloses how many analyses were run and whether the hypothesis was specified before or after seeing the data.

The American Statistical Association's 2016 statement on p-values makes the underlying point directly: scientific conclusions should not be based only on whether a p-value passes a threshold. Hypothesis testing is a tool for quantifying one specific kind of surprise, and it works well when used for exactly that and nothing more.

## References

- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129-133.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S and Type M errors. *Perspectives on Psychological Science*, 9(6), 641-651.
