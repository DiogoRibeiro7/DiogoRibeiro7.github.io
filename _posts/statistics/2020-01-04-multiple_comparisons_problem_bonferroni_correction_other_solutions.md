---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-04'
excerpt: Multiple testing is not one problem with one correction. Bonferroni and Holm control family-wise error, while Benjamini-Hochberg controls false discovery rate under different assumptions.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- multiple testing
- Bonferroni
- Holm
- false discovery rate
- Benjamini-Hochberg
permalink: '/statistics/multiple_comparisons_problem_bonferroni_correction_other_solutions/'
seo_description: A rigorous guide to family-wise error, Bonferroni, Holm, false discovery rate, and the Benjamini-Hochberg procedure, including dependence assumptions and reproducible Python.
seo_title: 'Multiple Testing: Bonferroni, Holm, and FDR'
seo_type: article
summary: Multiple testing methods target different error criteria. This article derives family-wise error and false discovery rate, explains when Bonferroni, Holm, and Benjamini-Hochberg apply, and fixes common implementation mistakes.
tags:
- Hypothesis Testing
- Multiple Testing
- Statistical Inference
title: 'Multiple Testing: Bonferroni, Holm, and False Discovery Rate'
---

The phrase “multiple comparisons problem” hides several different inferential problems. If we test many hypotheses, we need to decide what kind of error we want to control. Possible targets include:

- the probability of at least one false rejection;
- the expected proportion of false rejections among all rejections;
- the average false-positive rate per test;
- or a decision-theoretic loss that weighs different mistakes differently.

Bonferroni, Holm, and Benjamini-Hochberg do not solve the same problem with different levels of aggressiveness. They control different error criteria.

## Family-wise error rate

Suppose we test $m$ null hypotheses. Let $V$ be the number of false rejections. The family-wise error rate is

$$
\operatorname{FWER}
=
P(V\ge1).
$$

If all $m$ null hypotheses are true and the tests are independent with per-test Type I error $\alpha$, then

$$
\operatorname{FWER}
=
1-(1-\alpha)^m.
$$

For

$$
m=20,
\qquad
\alpha=0.05,
$$

this is

$$
1-0.95^{20}
\approx
0.642.
$$

That 64% calculation requires independence. Without independence, the exact FWER is different. The multiple-testing problem remains, but the simple formula is no longer exact.

## Bonferroni control

Bonferroni uses the union bound:

$$
P\left(
\bigcup_{i=1}^{m}
\{\text{false rejection of }H_i\}
\right)
\le
\sum_{i=1}^{m}
P(\text{false rejection of }H_i).
$$

If each test is performed at level

$$
\frac{\alpha}{m},
$$

then

$$
\operatorname{FWER}
\le
\alpha.
$$

No independence assumption is required for this bound. That robustness is the main strength of Bonferroni. Its price is conservatism when the number of tests is large or when the dependence structure could be exploited more efficiently.

## Adjusted p-values

Instead of comparing raw p-values with

$$
\alpha/m,
$$

Bonferroni-adjusted p-values can be written as

$$
p_i^{adj}
=
\min(1,mp_i).
$$

Reject when

$$
p_i^{adj}\le\alpha.
$$

The threshold and adjusted-p-value views are equivalent. Adjusted p-values are often easier to report because they preserve a common significance threshold.

## Holm's sequential procedure

Holm improves on ordinary Bonferroni while retaining strong FWER control. Sort the p-values:

$$
p_{(1)}
\le
p_{(2)}
\le
\cdots
\le
p_{(m)}.
$$

Compare sequentially:

$$
p_{(1)}
\le
\frac{\alpha}{m},
$$

then

$$
p_{(2)}
\le
\frac{\alpha}{m-1},
$$

and so on. Stop at the first non-rejection. All later hypotheses remain unrejected. Holm is uniformly at least as powerful as single-step Bonferroni while making no stronger dependence assumption for FWER control. So when strong FWER control is required, Holm is often a better default than plain Bonferroni.

## False discovery rate

FWER asks whether **any** false discovery occurs. In high-dimensional exploratory problems, that can be too stringent. Let

$$
R
$$

be the total number of rejected hypotheses and

$$
V
$$

the number of false rejections. The false discovery proportion is

$$
\operatorname{FDP}
=
\frac{V}{\max(R,1)}.
$$

The false discovery rate is

$$
\operatorname{FDR}
=
E[\operatorname{FDP}].
$$

This is not the same as the probability that an individual rejected hypothesis is false. It is an expectation over the random set of rejections produced by the whole procedure.

## Benjamini-Hochberg

Let the ordered p-values be

$$
p_{(1)}
\le
\cdots
\le
p_{(m)}.
$$

For target FDR level $q$, find the largest index

$$
k
=
\max
\left\{
i:
p_{(i)}
\le
\frac{i}{m}q
\right\}.
$$

If such a $k$ exists, reject

$$
H_{(1)},\ldots,H_{(k)}.
$$

The phrase **largest index** is important. A common incorrect implementation tests each ordered p-value separately against its threshold and rejects only the positions satisfying the inequality. The correct step-up rule rejects every hypothesis up to the largest qualifying index.

## Dependence assumptions for BH

Under independent p-values for the true nulls, Benjamini-Hochberg controls FDR at the target level, with the familiar bound involving the proportion of true nulls. The procedure also has control under certain forms of positive dependence. Under arbitrary dependence, the original BH guarantee does not generally hold unchanged. The Benjamini-Yekutieli procedure modifies the thresholds using

$$
c_m
=
\sum_{i=1}^{m}\frac{1}{i}
$$

to obtain broader dependence robustness, at the cost of power. So “FDR correction works under any dependence” is too strong.

## FWER and FDR answer different scientific questions

Suppose a confirmatory clinical trial has one primary endpoint and several prespecified key secondary endpoints. A false positive anywhere in that confirmatory family may be costly. Strong FWER control can be appropriate. Now suppose a genomics experiment screens 20,000 genes to generate candidates for later validation. Allowing some false discoveries may be acceptable if the expected fraction is controlled.

FDR can be much more useful. The choice is not

> conservative method versus powerful method.

It is

> Which error criterion matches the inferential role of this family of hypotheses?

## The family must be defined

Multiplicity corrections apply to a **family** of hypotheses. That family is not determined automatically by software. If ten outcomes, four subgroups, three models, and several time points are all examined, which tests belong to the same inferential family depends on the scientific claims being made. Defining the family after seeing the p-values defeats much of the purpose of multiplicity control.

Confirmatory analyses should define it in advance when possible.

## Exploratory and confirmatory analyses should not be blurred

Exploratory analyses can tolerate a different error structure from confirmatory claims. The problem arises when a broad exploratory search is conducted and only the smallest p-value is presented as though it came from one prespecified test. Multiplicity is then hidden rather than controlled. A transparent analysis reports the search space, the correction strategy, and which claims are confirmatory versus exploratory.

## Reproducible Python

Use a tested implementation rather than rewriting these procedures casually.

~~~python
from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

p_values: np.ndarray = np.array(
    [0.001, 0.008, 0.012, 0.030, 0.20]
)

reject_bonf, p_bonf, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="bonferroni",
)

reject_holm, p_holm, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="holm",
)

reject_bh, p_bh, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="fdr_bh",
)

print("Bonferroni:", reject_bonf, p_bonf)
print("Holm:", reject_holm, p_holm)
print("BH:", reject_bh, p_bh)
~~~

The returned adjusted p-values and rejection decisions preserve the original hypothesis order. That detail is easy to get wrong in hand-written implementations after sorting.

## A transparent BH implementation

For teaching, the step-up logic can be implemented explicitly.

~~~python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

def benjamini_hochberg(
    p_values: FloatArray,
    q: float = 0.05,
) -> BoolArray:
    if p_values.ndim != 1:
        raise ValueError(
            "p_values must be one-dimensional."
        )

if np.any(
        (p_values < 0.0)
        | (p_values > 1.0)
    ):
        raise ValueError(
            "p-values must lie in [0, 1]."
        )

if not 0.0 < q < 1.0:
        raise ValueError(
            "q must lie strictly between 0 and 1."
        )

m: int = p_values.size

if m == 0:
        return np.zeros(0, dtype=bool)

order = np.argsort(p_values)
    sorted_p = p_values[order]

thresholds = (
        np.arange(1, m + 1)
        / m
        * q
    )

qualifying = np.flatnonzero(
        sorted_p <= thresholds
    )

reject_sorted = np.zeros(
        m,
        dtype=bool,
    )

if qualifying.size > 0:
        k: int = int(qualifying[-1])
        reject_sorted[: k + 1] = True

reject = np.zeros(
        m,
        dtype=bool,
    )

reject[order] = reject_sorted
    return reject
~~~

The implementation deliberately rejects all ordered hypotheses up to the last qualifying index.

## Power is not the only reason to prefer one correction

Multiplicity procedures can encode structure. Gatekeeping procedures test secondary hypotheses only after primary success. Hierarchical procedures exploit ordered families. Closed testing can provide strong FWER control with logical relationships among hypotheses. Weighted procedures can allocate more Type I error to more important hypotheses when weights are prespecified.

The number of tests alone does not determine the best correction.

## Conclusion

Multiple testing is a problem of defining an error criterion across a family of hypotheses. Bonferroni controls FWER through a union bound and requires no independence assumption for that guarantee. Holm improves power while retaining strong FWER control. Benjamini-Hochberg controls the expected false discovery proportion under its dependence conditions and is aimed at a different inferential objective.

The central sequence is

$$
\boxed{
\text{define the family}
\rightarrow
\text{choose the error criterion}
\rightarrow
\text{choose the procedure}
}
$$

Applying a correction without defining those first two pieces is only mechanical p-value processing.

## References

- Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8, 3–62.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188.
