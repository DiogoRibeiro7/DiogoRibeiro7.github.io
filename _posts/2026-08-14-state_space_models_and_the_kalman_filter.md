---
permalink: '/time-series/state_space_models_and_the_kalman_filter/'
title: 'State Space Models and the Kalman Filter'
categories:
- Time Series
tags:
- Time Series
- Statistical Modeling
- Bayesian Statistics
- Signal Processing
author_profile: false
seo_title: 'State Space Models and the Kalman Filter'
seo_description: 'A recursive way to estimate what you cannot observe, and the framework that unifies exponential smoothing, ARIMA and structural models.'
excerpt: >-
  The Kalman filter is usually introduced as a tracking algorithm for
  spacecraft. It is more useful understood as the general engine for
  estimating hidden state from noisy observation.
summary: >-
  An introduction to state space models: separating the unobserved state from
  the noisy measurement, how the Kalman filter alternates prediction and
  update, why the same framework expresses exponential smoothing and ARIMA,
  and what it offers that a purely autoregressive view does not.
keywords:
  - state space models
  - Kalman filter
  - structural time series
  - latent state
  - recursive estimation
classes: wide
date: '2026-08-14'
header:
  image: /assets/images/data_science_10.avif
  og_image: /assets/images/data_science_10.avif
  overlay_image: /assets/images/data_science_10.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_10.avif
  twitter_image: /assets/images/data_science_10.avif
---
The Kalman filter is usually introduced as the algorithm that guided Apollo spacecraft, which makes it sound like an aerospace curiosity. It is better understood as the general answer to a question that appears constantly: given noisy measurements of something you cannot observe directly, what is your best estimate of its current state?

## Separating State from Measurement

A state space model splits the problem into two equations.

The **state equation** describes how the unobserved state evolves:

$$
x_t = F x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q).
$$

The **observation equation** describes how measurements relate to that state:

$$
y_t = H x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R).
$$

The separation is the conceptual contribution. A thermometer reading is not the temperature; it is the temperature plus measurement error. A reported sales figure is not underlying demand; it is demand plus reporting noise, censored by stockouts. Conflating the two — which a purely autoregressive model does — means fitting the model to the noise as well as the signal.

Two covariance matrices carry the modelling assumptions. $Q$ says how much the true state can change between steps; $R$ says how noisy the measurements are. Their *ratio* determines the filter's behaviour: large $Q$ relative to $R$ means "trust the new measurement", and the estimate tracks the data closely; small $Q$ means "trust the model", and the estimate is smooth and slow to react.

## Predict, Then Update

The filter alternates two steps at every time point.

**Predict** — project the state forward using the model, and grow the uncertainty because time has passed:

$$
\hat{x}_{t|t-1} = F \hat{x}_{t-1}, \qquad
P_{t|t-1} = F P_{t-1} F^\top + Q .
$$

**Update** — having seen $y_t$, correct the prediction in proportion to how surprising it was:

$$
K_t = P_{t|t-1} H^\top \left(H P_{t|t-1} H^\top + R\right)^{-1},
$$
$$
\hat{x}_t = \hat{x}_{t|t-1} + K_t\left(y_t - H\hat{x}_{t|t-1}\right), \qquad
P_t = (I - K_t H)P_{t|t-1}.
$$

The **Kalman gain** $K_t$ is where the intuition lives. It is a weight between zero and one deciding how much to move toward the new observation. When measurement noise $R$ is large the gain is small and the filter mostly ignores the data; when the state uncertainty $P$ is large the gain is close to one and the filter mostly adopts the measurement.

The term $y_t - H\hat{x}_{t|t-1}$ is the **innovation** — the part of the observation the model did not anticipate. It is also the natural diagnostic: if the model is correct, innovations should be uncorrelated with mean zero.

```python
import numpy as np

def kalman_local_level(y, Q, R):
    """Local level model: state is a random walk, observed with noise."""
    n = len(y)
    x, P = y[0], 1.0
    filtered, gains = np.zeros(n), np.zeros(n)
    for t in range(n):
        # predict
        x_pred, P_pred = x, P + Q
        # update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (y[t] - x_pred)
        P = (1 - K) * P_pred
        filtered[t], gains[t] = x, K
    return filtered, gains

rng = np.random.default_rng(0)
n = 300
true_level = np.cumsum(rng.normal(0, 0.1, n)) + 20     # slowly drifting truth
observed = true_level + rng.normal(0, 1.0, n)          # noisy measurements

for label, (Q, R) in {"trusts data (Q/R high)": (1.0, 1.0),
                      "balanced": (0.01, 1.0),
                      "trusts model (Q/R low)": (0.0001, 1.0)}.items():
    est, K = kalman_local_level(observed, Q, R)
    rmse = np.sqrt(((est - true_level) ** 2).mean())
    print(f"{label:24} steady-state gain={K[-1]:.3f}  RMSE vs truth={rmse:.3f}")

print(f"\nRMSE of raw observations vs truth: "
      f"{np.sqrt(((observed - true_level) ** 2).mean()):.3f}")
```

The comparison against the raw observations is the point of the exercise: a well-tuned filter recovers the hidden level considerably more accurately than the measurements themselves, because it pools information across time instead of trusting each reading in isolation.

## Why This Framework Is Worth Knowing

**It unifies methods that look unrelated.** Simple exponential smoothing is exactly the local level model above, with the smoothing parameter $\alpha$ equal to the steady-state Kalman gain. Holt's linear trend method is the local *linear* trend model. ARIMA models can be written in state space form. What look like separate techniques are one family seen from different angles.

**It handles missing data natively.** When an observation is absent, run the predict step and skip the update. Uncertainty grows across the gap and contracts when data resumes, which is the behaviour you want and which imputation only imitates.

**It is naturally online.** The filter carries only the current state and its covariance, updating in constant time per observation. There is no need to refit on the full history, which suits streaming applications.

**Components are interpretable.** Structural time series models add explicit level, trend, seasonal and regression components to the state vector, so the fitted output decomposes into parts that mean something — unlike the coefficients of a high-order ARIMA.

## Filtering, Smoothing, Forecasting

Three related questions have three different answers, and confusing them is a common error.

**Filtering** estimates the state at time $t$ using data up to $t$. This is what the recursion above computes and what a real-time system needs.

**Smoothing** estimates the state at time $t$ using *all* data, including observations after $t$. It is strictly more accurate and only available retrospectively — appropriate for historical analysis, and unavailable in production at the moment of decision.

**Forecasting** runs the predict step forward without any update. Uncertainty grows at each step, which is where honest widening prediction intervals come from.

## Assumptions and Their Limits

The standard filter assumes linear dynamics and Gaussian noise. Under those conditions it is provably optimal in mean squared error.

When linearity fails, the extended Kalman filter linearises around the current estimate and the unscented filter propagates a set of sample points through the true non-linear function; the latter is generally more robust. When the noise is not Gaussian — or the state is discrete — particle filters approximate the distribution by simulation, at considerably greater computational cost.

The practical difficulty is usually not the algorithm but the covariances. $Q$ and $R$ are rarely known, and they are typically estimated by maximum likelihood, which the filter supplies naturally since it computes the likelihood as a by-product of the innovations. Setting them by hand is possible but amounts to choosing how much to smooth, so it should be done deliberately rather than by leaving a default in place.

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.
