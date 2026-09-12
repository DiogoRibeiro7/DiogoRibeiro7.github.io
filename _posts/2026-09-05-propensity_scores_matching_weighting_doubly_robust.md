---
permalink: '/statistics/propensity_scores_matching_weighting_doubly_robust/'
title: 'Propensity Scores: Matching, Weighting and the Estimator That Forgives One Mistake'
categories:
- Statistics
tags:
- Causal Inference
- Observational Data
- Statistical Modeling
- Statistics
author_profile: false
seo_title: 'Propensity Score Matching, Weighting and Doubly Robust Estimation'
seo_description: 'Matching and weighting need the treatment model to be right; regression adjustment needs the outcome model to be right. A simulation shows each failing on its own weak spot, the doubly robust estimator surviving either, and what poor overlap does to all of them.'
excerpt: >-
  Four estimators agree on the effect when both models are correct.
  Misspecify the outcome model and regression adjustment is off by 0.19;
  misspecify the treatment model and matching and weighting are off by
  0.24. The doubly robust estimator is right in both cases.
summary: >-
  What a propensity score is and why one number can stand in for many
  covariates, a simulated observational study with three confounders in
  which the naive comparison is off by 84 percent, a comparison of
  regression adjustment, nearest-neighbour matching, inverse probability
  weighting and the doubly robust estimator under a correct model and
  under each kind of misspecification, what strong confounding does to
  overlap and to the weights, what trimming fixes and what it changes,
  and the balance check that should be reported with every estimate.
keywords:
  - propensity score
  - matching
  - inverse probability weighting
  - doubly robust
  - AIPW
  - overlap
  - causal inference
classes: wide
date: '2026-09-05'
why_this_exists: >-
  Propensity score methods are the default for observational effect
  estimation, and the choice between matching, weighting and regression
  is usually made by habit. This post shows on a simulation exactly which
  assumption each one rests on, which failures it survives and which it
  does not, so the choice can be made from the risk rather than the
  convention.
evidence: >-
  Simulated observational studies of 4,000 units with three confounders,
  treatment assigned by a logistic model and a true effect of 2, under
  four conditions: both models correct, an outcome model missing cubic
  and interaction terms, a propensity model missing a confounder, and
  strong confounding with poor overlap; 400 replications per condition.
methodology: >-
  Compares the bias, spread and root mean squared error of the naive
  difference, regression adjustment, one-to-one propensity matching,
  inverse probability weighting and the augmented (doubly robust)
  estimator; measures the distribution of propensity scores and weights
  as confounding strengthens; and reports covariate balance before and
  after weighting.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-library.jpg
  og_image: /assets/images/headers/photo-library.jpg
  overlay_image: /assets/images/headers/photo-library.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-library.jpg
  twitter_image: /assets/images/headers/photo-library.jpg
---
The training programme was not randomised. Employees chose it, and the ones who chose it were already more senior, more engaged and better reviewed. Comparing participants with non-participants gives an effect of 3.67 on the outcome. The true effect in the simulation behind this post is 2.00, and the extra 1.67 is the head start the participants brought with them.

Every method for fixing this rests on the same hope: that the characteristics driving the choice were measured, so that participants and non-participants with the same measured characteristics are comparable. Where the methods differ is in what else they need to be true. Regression adjustment needs a correct model of the outcome. Matching and weighting need a correct model of the treatment. The doubly robust estimator needs only one of the two, and the simulation below shows it collecting on that promise.

## One Number Instead of Many

The propensity score is the probability of receiving the treatment given the covariates, $e(x) = P(D = 1 \mid X = x)$. Rosenbaum and Rubin's result is that if treatment is unconfounded given $X$, it is also unconfounded given $e(X)$ alone. A single number carries all the information that many covariates carried, which is what makes matching feasible: matching on five covariates is hard, matching on one score is easy.

The score is used in three ways. Matching pairs each treated unit with a control of similar score and compares within pairs. Weighting reweights each unit by the inverse of its probability of the treatment it received, building a pseudo-population in which treatment is unrelated to the covariates. Stratification splits the sample into score bands and averages within-band differences. All three depend on the score being right, which means the treatment model being right.

## A Simulation

Four thousand units, three covariates, treatment assigned by a logistic model in all three, and an outcome that depends on the covariates and on the treatment with a true effect of 2.

```python
import numpy as np
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)
TAU = 2.0

def draw(n=4000, strength=1.0, nonlinear_outcome=False, r=rng):
    """Three covariates; treatment assigned by a logistic in them; outcome depends on them and on treatment."""
    x = r.normal(0, 1, (n, 3))
    logit = strength * (0.9 * x[:, 0] + 0.6 * x[:, 1] - 0.5 * x[:, 2])
    d = r.random(n) < 1 / (1 + np.exp(-logit))
    base = 2.0 * x[:, 0] + 1.0 * x[:, 1] + 0.5 * x[:, 2]
    if nonlinear_outcome:
        base = base + 1.0 * x[:, 0] ** 3 - 1.0 * x[:, 0] * x[:, 1]
    y = 10 + base + TAU * d + r.normal(0, 2, n)
    return x, d.astype(int), y

def fit_logistic(X, d, iters=60):
    """Newton logistic regression with an intercept."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ b))
        W = p * (1 - p) + 1e-9
        b += np.linalg.solve((A * W[:, None]).T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ (d - p))
    return 1 / (1 + np.exp(-A @ b))
```

The four estimators, plus the naive difference for reference:

```python
def naive(x, d, y, ps=None):
    return y[d == 1].mean() - y[d == 0].mean()

def regression(x, d, y, ps=None):
    A = np.column_stack([np.ones(len(x)), d, x])
    return np.linalg.lstsq(A, y, rcond=None)[0][1]

def matching(x, d, y, ps):
    """1:1 nearest neighbour on the score, with replacement."""
    tr, co = np.where(d == 1)[0], np.where(d == 0)[0]
    _, j = cKDTree(ps[co][:, None]).query(ps[tr][:, None])
    return np.mean(y[tr] - y[co][j])

def ipw(x, d, y, ps, trim=0.0):
    """Stabilised inverse probability weights, optionally trimming extreme scores."""
    keep = (ps > trim) & (ps < 1 - trim)
    d_, y_, ps_ = d[keep], y[keep], ps[keep]
    w1, w0 = d_ / ps_, (1 - d_) / (1 - ps_)
    return np.sum(w1 * y_) / np.sum(w1) - np.sum(w0 * y_) / np.sum(w0)

def aipw(x, d, y, ps):
    """Doubly robust: regression predictions corrected by weighted residuals."""
    A = np.column_stack([np.ones(len(x)), x])
    b1 = np.linalg.lstsq(A[d == 1], y[d == 1], rcond=None)[0]
    b0 = np.linalg.lstsq(A[d == 0], y[d == 0], rcond=None)[0]
    m1, m0 = A @ b1, A @ b0
    return np.mean(m1 - m0 + d * (y - m1) / ps - (1 - d) * (y - m0) / (1 - ps))
```

Each condition is run 400 times and the estimators compared against the truth:

```python
ESTIMATORS = {"naive difference": naive, "regression adjustment": regression,
              "propensity matching": matching, "IPW": ipw, "doubly robust (AIPW)": aipw}

def run(label, **kw):
    wrong_ps = kw.pop("wrong_ps", False)
    out = {k: [] for k in ESTIMATORS}
    for _ in range(400):
        x, d, y = draw(**kw)
        feats = x[:, :2] if wrong_ps else x            # a propensity model missing a confounder
        ps = np.clip(fit_logistic(feats, d), 0.01, 0.99)
        for k, fn in ESTIMATORS.items():
            out[k].append(fn(x, d, y, ps))
    print(label)
    for k in ESTIMATORS:
        v = np.array(out[k])
        print(f"  {k:24} {v.mean():>6.2f}  bias {v.mean() - TAU:>+6.2f}  sd {v.std():.2f}  "
              f"rmse {np.sqrt(np.mean((v - TAU) ** 2)):.2f}")

run("both models correct")
run("outcome model misspecified", nonlinear_outcome=True)
run("propensity model misspecified", wrong_ps=True)
run("strong confounding, poor overlap", strength=2.5)
```

**Both models correct.** Everything works, and the differences are in precision rather than bias.

| Estimator | Mean estimate | Bias | Standard deviation | Root mean squared error |
| --- | --- | --- | --- | --- |
| Naive difference | 3.67 | +1.67 | 0.10 | 1.67 |
| Regression adjustment | 2.00 | -0.00 | 0.07 | 0.07 |
| Propensity matching | 2.00 | -0.00 | 0.13 | 0.13 |
| Inverse probability weighting | 1.99 | -0.01 | 0.10 | 0.10 |
| Doubly robust | 2.00 | -0.00 | 0.08 | 0.08 |

The naive comparison is off by 84 percent of the true effect, and every adjusted estimator recovers it. Matching is the least precise, because one-to-one matching discards most of the control sample; weighting keeps everyone and is tighter; regression is tightest because it uses the covariates directly.

## Each Method's Weak Spot

The interesting question is what happens when a model is wrong, which in practice is always.

**The outcome model is misspecified.** The true outcome contains a cubic term and an interaction; the analyst fits a linear model. The propensity model is still correct.

| Estimator | Mean estimate | Bias | Standard deviation |
| --- | --- | --- | --- |
| Naive difference | 5.63 | +3.63 | 0.18 |
| Regression adjustment | 1.81 | **-0.19** | 0.10 |
| Propensity matching | 2.06 | +0.06 | 0.31 |
| Inverse probability weighting | 2.01 | +0.01 | 0.29 |
| Doubly robust | 2.00 | +0.00 | 0.20 |

Regression adjustment is biased by 0.19, about a tenth of the effect, and its confidence interval is narrow enough to exclude the truth. Matching and weighting are untouched, because neither ever models the outcome: they balance the covariates and let the data supply the rest.

**The propensity model is misspecified.** A confounder is left out of the treatment model, while the outcome model is correct.

| Estimator | Mean estimate | Bias | Standard deviation |
| --- | --- | --- | --- |
| Naive difference | 3.67 | +1.67 | 0.09 |
| Regression adjustment | 2.00 | -0.00 | 0.07 |
| Propensity matching | 1.76 | **-0.24** | 0.11 |
| Inverse probability weighting | 1.77 | **-0.23** | 0.09 |
| Doubly robust | 2.00 | -0.00 | 0.08 |

Now the failure is on the other side. Matching and weighting are biased by about 0.24 because the score they balanced on was the wrong score, and regression adjustment, which never used the score, is fine.

The doubly robust estimator is unbiased in both tables. It combines an outcome model with a weighting correction so that the errors of one are absorbed by the other; it is consistent if *either* model is right, which is why it is worth the extra machinery. It is not magic: with both models wrong it is biased like everything else, and it pays a little precision for the insurance, 0.20 against regression's 0.10 in the second table.

![Absolute bias of each estimator under the four conditions: both models correct, outcome model misspecified, propensity model misspecified and strong confounding. Regression fails on the first, matching and weighting on the second, and the doubly robust estimator survives all but a joint failure.](/assets/images/figures/propensity_estimator_bias.png){: width="1152" height="672" loading="lazy"}

## Overlap Is the Real Constraint

Unconfoundedness is the assumption that gets discussed. Overlap, the requirement that every kind of unit could have received either treatment, is the one that fails visibly and is easier to check.

```python
for s in (0.5, 1.0, 2.0, 3.0):
    x, d, y = draw(strength=s)
    ps = fit_logistic(x, d)
    w = np.where(d == 1, 1 / ps, 1 / (1 - ps))
    print(f"strength {s}: scores [{ps.min():.3f}, {ps.max():.3f}], outside [0.1, 0.9] {np.mean((ps < .1) | (ps > .9)):.1%}, "
          f"largest weight {w.max():.1f}, top 1% of weights carry {np.sort(w)[-40:].sum() / w.sum():.0%}")
```

| Confounding strength | Score range | Units outside [0.1, 0.9] | Largest weight | Share of total weight in the top 1% |
| --- | --- | --- | --- | --- |
| 0.5 | 0.13 to 0.89 | 0.0% | 8.6 | 2% |
| 1.0 | 0.02 to 0.98 | 5.7% | 24.0 | 5% |
| 2.0 | 0.00 to 1.00 | 37.2% | 78.1 | 14% |
| 3.0 | 0.00 to 1.00 | 54.9% | 2,623 | 38% |

As the covariates predict treatment more strongly, scores pile up at the ends, weights explode, and the estimate comes to depend on a handful of units. At strength 3, one unit in a hundred carries 38 percent of the weight: the effective sample is tiny and the variance is enormous, which is exactly what the fourth condition showed, with weighting biased by 0.29 and matching's spread quadrupled.

Trimming units with extreme scores is the usual response.

```python
for trim in (0.0, 0.01, 0.05, 0.10):
    vals = []
    for _ in range(200):
        x, d, y = draw(strength=2.5)
        ps = np.clip(fit_logistic(x, d), 1e-6, 1 - 1e-6)
        vals.append(ipw(x, d, y, ps, trim=trim))
    print(f"trim at {trim:.0%}: estimate {np.mean(vals):.2f}, sd {np.std(vals):.2f}")
```

| Trim | Estimate | Standard deviation |
| --- | --- | --- |
| None | 2.08 | 0.46 |
| 1% | 1.99 | 0.18 |
| 5% | 1.98 | 0.11 |
| 10% | 1.98 | 0.11 |

Trimming works, and it changes the question. The units dropped are those who would almost never receive one of the treatments, and after dropping them the estimate applies to the remaining population, not the original one. That is often the right trade, because an effect for units that could plausibly go either way is more useful than a noisy effect for everyone, but it has to be stated, since the estimand has changed.

## Report the Balance, Not the Model

A propensity model is a means, not a result. Nobody should care about its coefficients or its area under the curve; a score that predicts treatment perfectly is a score with no overlap and no usable comparison. What matters is whether the covariates are balanced after adjustment.

```python
x, d, y = draw(strength=1.0, n=20000)
ps = fit_logistic(x, d)
for j in range(3):
    raw = (x[d == 1, j].mean() - x[d == 0, j].mean()) / x[:, j].std()
    w1, w0 = d / ps, (1 - d) / (1 - ps)
    wm1 = np.sum(w1 * x[:, j]) / np.sum(w1); wm0 = np.sum(w0 * x[:, j]) / np.sum(w0)
    print(f"x{j+1}: before {raw:+.2f}, after weighting {(wm1 - wm0) / x[:, j].std():+.2f}")
```

| Covariate | Standardised difference before | After weighting |
| --- | --- | --- |
| x1 | +0.69 | +0.01 |
| x2 | +0.46 | +0.01 |
| x3 | -0.41 | -0.00 |

Differences of 0.4 to 0.7 standard deviations become hundredths. The conventional threshold is 0.1, and a covariate still above it after adjustment means the score is not doing its job, whatever its fit statistics say. This table, and the distribution of scores in each arm, are what belongs in the report.

## What None of It Can Fix

Every method here assumes the confounders were measured. If employees chose the training partly because of ambition, and ambition is not in the data, no score built from the data can balance it, and all four estimators are biased by the same unmeasured amount. The propensity machinery makes the measured-confounding correction as good as it can be; it says nothing about what was not measured.

That is why an observational estimate should be accompanied by a sensitivity analysis: how strong would an unmeasured confounder have to be, in its association with both treatment and outcome, to move the estimate to zero? Rosenbaum bounds and the E-value answer that question in one number, and a result that would be overturned by a modest unmeasured confounder should be presented as such.

## What to Do

1. **Fit the treatment model on pre-treatment covariates only**, and judge it by covariate balance after adjustment, never by its predictive accuracy.
2. **Check overlap first**: plot the score distribution in both arms. Little overlap means the question cannot be answered for the whole population, whichever estimator is used.
3. **Prefer the doubly robust estimator** when both models are plausible but neither is certain; it is consistent if either is right, at a small cost in precision.
4. **Use weighting over one-to-one matching** when the control pool is scarce, since matching discards most of it, and report the effective sample size implied by the weights.
5. **Trim extreme scores and say so**, because trimming changes the population the estimate describes.
6. **Report a sensitivity analysis** for unmeasured confounding alongside the estimate; no propensity method addresses it.

## References

- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846-866.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. *Biometrics*, 61(4), 962-973.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
- King, G., & Nielsen, R. (2019). Why propensity scores should not be used for matching. *Political Analysis*, 27(4), 435-454.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
