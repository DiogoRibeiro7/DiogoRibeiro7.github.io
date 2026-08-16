---
permalink: '/time-series/regime_switching_models_time_series/'
title: 'Regime-Switching Models for Time Series'
categories:
- Time Series
tags:
- Time Series
- Statistical Modeling
- Economics
- Stochastic Processes
author_profile: false
seo_title: 'Regime-Switching Models'
seo_description: 'Some series do not have one set of dynamics. Markov switching models let the parameters change with an unobserved state.'
excerpt: >-
  A single model fitted across a recession and an expansion describes neither.
  Regime-switching models allow the dynamics themselves to change, with the
  regime inferred rather than assumed.
summary: >-
  How Markov-switching models let parameters vary with an unobserved regime,
  how the filter infers regime probabilities from the data, the difference
  between a regime switch and a structural break, and the identification
  problems that make these models easy to overfit.
keywords:
  - regime switching
  - Markov switching
  - hidden state
  - structural break
  - nonlinear time series
classes: wide
date: '2026-07-17'
header:
  image: /assets/images/data_science_10.jpg
  og_image: /assets/images/data_science_10.jpg
  overlay_image: /assets/images/data_science_10.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_10.jpg
  twitter_image: /assets/images/data_science_10.jpg
---
Fit one model across a recession and an expansion and it describes neither. The mean growth rate, the volatility, and often the autocorrelation all differ between the two, and a single parameter set averages across regimes that behave differently.

Regime-switching models allow the parameters themselves to change, with the regime treated as an unobserved state inferred from the data rather than assumed from a calendar.

## The Markov-Switching Structure

Let $S_t \in \{1, \dots, K\}$ be an unobserved regime. Conditional on it, the observation follows a regime-specific model — in the simplest case a different mean and variance:

$$
y_t \mid S_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2).
$$

The regime evolves as a Markov chain with transition probabilities

$$
P(S_t = j \mid S_{t-1} = i) = p_{ij},
$$

collected in a transition matrix whose rows sum to one.

The diagonal entries carry most of the interpretation. A value of $p_{11} = 0.95$ means that, once in regime 1, the process stays there with 95% probability each period, implying an expected duration of $1/(1 - p_{11}) = 20$ periods. High diagonals produce persistent regimes; low ones produce rapid switching that is usually a sign of a misspecified model.

The crucial difference from a **structural break** is that regimes recur. A break is a permanent, one-off change in the data generating process. A regime is a state the series can return to, which is what makes recessions, high-volatility periods and machine operating modes natural applications.

```python
import numpy as np

def simulate_markov_switching(n, mus, sigmas, P, seed=0):
    rng = np.random.default_rng(seed)
    K = len(mus)
    s = np.zeros(n, dtype=int)
    y = np.zeros(n)
    s[0] = 0
    for t in range(n):
        if t > 0:
            s[t] = rng.choice(K, p=P[s[t - 1]])
        y[t] = rng.normal(mus[s[t]], sigmas[s[t]])
    return y, s

P = np.array([[0.97, 0.03],
              [0.10, 0.90]])          # regime 0 persistent, regime 1 less so
y, s = simulate_markov_switching(3000, mus=[0.8, -1.5],
                                 sigmas=[1.0, 2.5], P=P)

print(f"time in regime 0 : {(s == 0).mean():.1%}")
print(f"expected duration: {1/(1-P[0,0]):.0f} and {1/(1-P[1,1]):.0f} periods")
print(f"overall mean/sd  : {y.mean():.2f} / {y.std():.2f}")
for k in (0, 1):
    print(f"  regime {k}: mean {y[s==k].mean():+.2f}  sd {y[s==k].std():.2f}")
```

Note what the pooled statistics conceal. The overall mean of 0.26 sits between the regime means of +0.80 and −1.50 and describes neither. The overall standard deviation of 1.82 likewise falls between the regimes' 1.00 and 2.59, inflated above the calm regime by the variation *between* regimes rather than within them.

A single-regime model fitted here would report a volatility of 1.82: too high for the 76% of the time the series spends in the calm state, and far too low for the turbulent one. That is the specific failure regime-switching addresses — not that the average is wrong, but that no period actually looks like the average.

## Inferring the Regime

Since $S_t$ is unobserved, estimation uses a filter closely analogous to the Kalman filter, alternating prediction and update on the regime *probabilities* rather than on a continuous state.

At each step, the predicted regime distribution is propagated through the transition matrix, then updated by how likely the new observation is under each regime. Hamilton's filter does this recursively, and the likelihood accumulates as a by-product, which is what parameter estimation maximises. The parameters are usually fitted by EM or by direct numerical maximisation.

The output is a **smoothed probability** of being in each regime at each time — not a hard classification. That probabilistic output is more useful than a label: a period with 55% probability of recession is genuinely ambiguous, and reporting it as "recession" discards that.

## Where These Models Are Used

**Business cycles.** Hamilton's original application modelled US GNP with two regimes and recovered dates close to the official NBER recession chronology, from the data alone.

**Volatility regimes.** Financial returns switch between calm and turbulent periods, and a two-state model captures the clustering that a constant-variance model cannot.

**Condition monitoring.** Equipment operating in distinct modes — idle, normal load, high load — produces sensor data with mode-specific dynamics, and inferring the mode is often useful in itself.

**Energy demand.** Behaviour differs qualitatively between heating and cooling seasons in ways a smooth seasonal term represents poorly.

## The Ways These Models Fail

They are easy to fit and easy to over-fit, and three problems recur.

**Label switching.** The likelihood is invariant to permuting regime labels, so "regime 1" in one run may be "regime 2" in another. Identification requires imposing an ordering constraint — for example, that $\mu_1 < \mu_2$ — otherwise results are not comparable across runs.

**Choosing K.** Standard likelihood ratio tests do not apply, because under the null of fewer regimes the extra parameters are unidentified, which invalidates the usual asymptotics. Information criteria are the common fallback, and they favour more regimes than are interpretable. Two or three states is usually the practical limit before the regimes stop meaning anything.

**Local optima.** The likelihood surface is multimodal, so results depend on starting values. Multiple random starts are necessary, and reporting only the best fit without noting the spread across starts is misleading.

There is also a genuine identification concern raised in the literature: long memory and regime switching can generate similar autocorrelation patterns, so a series with occasional regime shifts can be mistaken for a long-memory process and vice versa. Distinguishing them from data alone is difficult, and the choice is often better made on substantive grounds — does the domain have recurring states? — than on statistical ones.

## Practical Guidance

Start with two regimes and a constant transition matrix. Add complexity only when the simple version demonstrably fails, since each addition multiplies the identification problems.

Check that the estimated regimes correspond to something recognisable. If regime 2 occurs in scattered single periods with no interpretation, the model is fitting outliers rather than states, and a heavy-tailed single-regime model is the better description.

Report smoothed probabilities rather than hard assignments, and be careful about forecasting: multi-step forecasts must average over the future regime path, so prediction intervals are wider — and often multimodal — compared with a single-regime model. That multimodality is informative and should not be summarised away.

## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Kim, C.-J., & Nelson, C. R. (1999). *State-Space Models with Regime Switching*. MIT Press.
- Diebold, F. X., & Inoue, A. (2001). Long memory and regime switching. *Journal of Econometrics*, 105(1), 131-159.
- Ang, A., & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313-337.
