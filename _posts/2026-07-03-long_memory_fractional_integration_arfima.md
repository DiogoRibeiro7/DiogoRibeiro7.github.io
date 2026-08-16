---
permalink: '/time-series/long_memory_fractional_integration_arfima/'
title: 'Long Memory and Fractional Integration in Time Series'
categories:
- Time Series
tags:
- Time Series
- Statistical Modeling
- Statistics
- Finance
author_profile: false
seo_title: 'Long Memory and Fractional Integration'
seo_description: 'Some series are neither stationary nor unit-root. Fractional differencing sits between the two and explains slowly decaying autocorrelation.'
excerpt: >-
  Standard practice offers two options: the series is stationary, or you
  difference it. Some series are genuinely in between, and forcing them either
  way loses information.
summary: >-
  What long memory means, how the fractional differencing parameter
  interpolates between stationarity and a unit root, how to estimate it, and
  why hyperbolic rather than exponential decay in the autocorrelation function
  is the signature to look for.
keywords:
  - long memory
  - fractional integration
  - ARFIMA
  - Hurst exponent
  - persistence
classes: wide
date: '2026-07-03'
header:
  image: /assets/images/data_science_3.jpg
  og_image: /assets/images/data_science_3.jpg
  overlay_image: /assets/images/data_science_3.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_3.jpg
  twitter_image: /assets/images/data_science_3.jpg
---
Standard time series practice offers a binary choice. Either the series is stationary and you model it directly, or it has a unit root and you difference it. The Dickey-Fuller test decides which.

Some series answer neither. Their autocorrelation decays too slowly for stationarity but too quickly for a random walk, and forcing them into either category discards real structure. Fractional integration describes the space in between.

## Two Kinds of Decay

The distinction is visible in how the autocorrelation function falls away.

A stationary ARMA process has autocorrelation decaying **exponentially**: $\rho_k \sim \phi^k$. After a few dozen lags it is indistinguishable from zero, and shocks die out quickly.

A long memory process has autocorrelation decaying **hyperbolically**: $\rho_k \sim k^{2d-1}$ for a parameter $d$. This is far slower. Autocorrelations remain measurably positive at lag 100, sometimes lag 1000, and the sum $\sum_k |\rho_k|$ diverges — which is the formal definition of long memory.

The practical signature is a correlogram that declines steadily without ever quite reaching zero, combined with a Dickey-Fuller test that is ambiguous or rejects weakly.

## Fractional Differencing

Ordinary differencing applies the operator $(1-L)$ where $L$ is the lag operator. Differencing twice applies $(1-L)^2$. The integration order $d$ is a whole number by construction.

Fractional integration allows $d$ to be any real value, defining $(1-L)^d$ through its binomial expansion:

$$
(1-L)^d = \sum_{k=0}^{\infty} \binom{d}{k}(-L)^k
= 1 - dL - \frac{d(1-d)}{2}L^2 - \frac{d(1-d)(2-d)}{6}L^3 - \cdots
$$

For integer $d$ the series terminates and recovers ordinary differencing. For fractional $d$ it does not terminate: every past observation receives a weight, decaying slowly. That infinite, slowly decaying weighting is precisely what "long memory" means operationally.

The parameter partitions the behaviour:

| $d$ | Behaviour |
|---|---|
| $d = 0$ | Short memory; standard ARMA |
| $d \in (0, 0.5)$ | Stationary with long memory |
| $d \in [0.5, 1)$ | Non-stationary but mean-reverting |
| $d = 1$ | Unit root; random walk |

The range $d \in (0, 0.5)$ is the interesting one: the series is stationary, so standard asymptotics apply, yet shocks persist far longer than any ARMA model would allow.

**ARFIMA($p,d,q$)** combines fractional differencing with ARMA terms, letting $d$ capture the long-run persistence while $p$ and $q$ handle short-run dynamics.

```python
import numpy as np

def frac_diff_weights(d, n):
    """Binomial expansion weights for (1-L)^d, truncated at n terms."""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w

for d in (0.0, 0.3, 0.5, 1.0):
    w = frac_diff_weights(d, 6)
    print(f"d={d}: weights {np.round(w, 4)}")

# the filter weights and the process autocorrelation decay at DIFFERENT rates
print("\nfilter weights |w_k| ~ k^(-d-1), at lag 100:")
for d in (0.1, 0.3, 0.45):
    print(f"  d={d}: {abs(frac_diff_weights(d, 200)[100]):.6f}")

print("process autocorrelation rho_k ~ k^(2d-1), at lag 100:")
for d in (0.1, 0.3, 0.45):
    print(f"  d={d}: {100.0 ** (2 * d - 1):.5f}")
```

At $d = 1$ the weights are exactly $[1, -1, 0, 0, \dots]$, which is ordinary first differencing. At $d = 0$ they are $[1, 0, 0, \dots]$, leaving the series untouched. At fractional $d$ the expansion never terminates: every past observation receives a non-zero weight, and that is the operational meaning of long memory.

Two decay rates are easy to confuse here, and the output separates them deliberately. The **filter** weights fall off as $k^{-d-1}$, so a larger $d$ makes them decay *faster*. The **process** autocorrelation falls off as $k^{2d-1}$, so a larger $d$ makes it decay *slower* — at lag 100, $d = 0.45$ still leaves a correlation around 0.63 while $d = 0.1$ has dropped to 0.03. It is the second quantity that carries the memory; the first is just the filter needed to remove it.

## Estimating d

Several approaches exist, differing in robustness and assumptions.

The **rescaled range** statistic, from Hurst's work on Nile flooding, estimates the Hurst exponent $H$, related to the differencing parameter by $d = H - 0.5$. It is intuitive and known to be biased in small samples and sensitive to short-run correlation.

The **GPH estimator** regresses the log periodogram on log frequency near zero, exploiting the fact that the spectral density of a long memory process diverges at the origin like $f(\lambda) \sim \lambda^{-2d}$. It is semiparametric, requiring no model for the short-run dynamics, at the cost of depending on how many frequencies you include.

**Exact maximum likelihood** on a full ARFIMA specification is efficient when the model is right and sensitive to misspecification when it is not.

A recurring difficulty deserves emphasis: **long memory and structural breaks are easy to confuse.** A stationary series with occasional level shifts produces slowly decaying sample autocorrelation that mimics long memory closely. Estimating $d$ from such a series yields a significant value that describes breaks rather than persistence. Testing for breaks before concluding long memory is not optional.

## Where It Appears

Long memory shows up consistently in a few domains.

Financial **volatility** is the strongest case. Returns themselves are close to unpredictable, but absolute or squared returns show autocorrelation persisting over hundreds of days — which motivated FIGARCH and related fractionally integrated volatility models.

**Hydrology** is where the phenomenon was first quantified. Hurst's analysis of Nile river levels found persistence that standard models could not reproduce, and the Hurst exponent carries his name for that reason.

**Network traffic**, **inflation**, and some **climate series** also display it.

In machine learning, fractional differencing has found a use worth noting: differencing a price series to stationarity destroys almost all of its memory, whereas fractional differencing with the smallest $d$ that achieves stationarity preserves considerably more information while satisfying the model's requirements.

## Practical Cautions

Do not conclude long memory from a slowly decaying correlogram alone — check for breaks and trends first, since both produce the same appearance.

Be careful with forecasts. Long memory implies shocks persist, which makes long-horizon forecasts more sensitive to the estimated $d$ than to anything else in the model. Small errors in $d$ compound over the horizon.

And weigh the cost. ARFIMA is harder to fit, harder to explain, and frequently no better than a well-specified short-memory model at short horizons. It earns its place when the persistence itself is the object of study — as in volatility modelling — rather than when it is a marginal accuracy improvement.

## References

- Granger, C. W. J., & Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15-29.
- Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*, 68(1), 165-176.
- Geweke, J., & Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis*, 4(4), 221-238.
- Baillie, R. T. (1996). Long memory processes and fractional integration in econometrics. *Journal of Econometrics*, 73(1), 5-59.
- Diebold, F. X., & Inoue, A. (2001). Long memory and regime switching. *Journal of Econometrics*, 105(1), 131-159.
