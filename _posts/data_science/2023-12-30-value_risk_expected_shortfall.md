---
author_profile: false
categories:
- Data Science
classes: wide
date: '2023-12-30'
excerpt: "Value at Risk is a quantile of a loss distribution. Expected Shortfall averages the tail beyond that quantile. Both are model-dependent and require careful horizon, sign, and backtesting conventions."
header:
  image: /assets/images/headers/photo-microscope.jpg
  og_image: /assets/images/headers/photo-microscope.jpg
  overlay_image: /assets/images/headers/photo-microscope.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-microscope.jpg
  twitter_image: /assets/images/headers/photo-microscope.jpg
keywords:
- Value at Risk
- Expected Shortfall
- Tail risk
- Backtesting
- Financial risk
- Quantile risk
permalink: '/data-science/value_risk_expected_shortfall/'
redirect_from:
- '/data science/value_risk_expected_shortfall/'
seo_description: "A rigorous comparison of Value at Risk and Expected Shortfall, including definitions, coherence, horizon scaling, estimation, and backtesting."
seo_title: "Value at Risk and Expected Shortfall: Quantiles and Tail Risk"
seo_type: article
tags:
- Finance
- Risk Management
- Python
title: "Value at Risk and Expected Shortfall: Quantiles and Tail Risk"
---

Value at Risk and Expected Shortfall summarize different parts of a loss distribution. Their usefulness depends on consistent sign conventions, horizon definitions, estimation methods, and validation.

Let $L$ denote portfolio loss, with larger positive values meaning worse outcomes.

## Value at Risk

At confidence level $\alpha$, VaR is the $\alpha$-quantile of the loss distribution,

$$
\operatorname{VaR}_\alpha(L)
=
\inf\{\ell:P(L\le \ell)\ge \alpha\}.
$$

A 99% one-day VaR of EUR 1 million means that, under the model used to construct the loss distribution, losses exceed EUR 1 million on about 1% of days.

It does not mean EUR 1 million is the maximum possible loss.

## Expected Shortfall

Expected Shortfall describes losses in the tail beyond VaR.

For a continuous distribution,

$$
\operatorname{ES}_\alpha(L)
=
E[L\mid L\ge \operatorname{VaR}_\alpha(L)].
$$

A more general quantile-based definition is

$$
\operatorname{ES}_\alpha(L)
=
\frac{1}{1-\alpha}
\int_\alpha^1
\operatorname{VaR}_u(L)\,du.
$$

This remains well defined when the loss distribution has atoms or ties.

## Why ES is different

VaR identifies a threshold. It does not describe how bad losses are after the threshold is crossed.

Two portfolios can have the same 99% VaR and very different tail severity. ES distinguishes them by averaging deeper tail losses.

Expected Shortfall is also coherent under standard conditions, including subadditivity, whereas VaR need not be subadditive for arbitrary loss distributions.

That does not make ES assumption-free.

## Estimation methods

Common approaches include:

- historical simulation
- parametric models
- filtered historical simulation
- Monte Carlo simulation
- extreme-value methods for tails

Each produces a different estimated loss distribution.

Historical simulation uses observed historical returns directly but assumes the historical window is relevant to current risk.

Parametric methods can be efficient if the model is credible, but Gaussian assumptions can badly understate skewness and heavy tails.

## The old code problem: cumulative wealth is not one-day loss

A common implementation error is to generate daily returns, compound them into a cumulative portfolio path, and then calculate

$$
V_0-V_t
$$

for every date as if those were independent one-day losses.

They are not. Those values are cumulative path losses at different horizons.

For one-day historical VaR, the loss series should correspond to one-day portfolio P&L or returns.

## A correct simple historical example

~~~python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def var_es(
    returns: Sequence[float],
    portfolio_value: float,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Estimate historical one-period VaR and ES from return observations."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    r: NDArray[np.float64] = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be a one-dimensional sample")

    # Positive values represent losses.
    losses = -portfolio_value * r

    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    es = float(np.mean(tail))

    return var, es
~~~

The sign convention is explicit and the horizon of the loss data matches the reported metric.

## Horizon scaling

The square-root-of-time rule,

$$
\sigma_h\approx \sqrt h\,\sigma_1,
$$

comes from independent, identically distributed increments with finite variance.

Scaling VaR or ES by $\sqrt h$ is therefore not universally valid. Volatility clustering, serial dependence, nonlinear portfolios, jumps, and changing positions can all invalidate it.

A direct multi-day simulation is often preferable.

## Backtesting VaR

If a model reports 99% VaR, exceedances should occur roughly 1% of the time under correct unconditional calibration.

But counting exceedances is not enough. Exceedances should also not cluster systematically.

Coverage and independence tests therefore examine different failure modes.

## Backtesting ES

ES is harder to backtest because it concerns the magnitude of tail losses, not only threshold exceedances.

Modern regulatory frameworks use joint or related testing procedures because VaR identifies the tail region over which ES is interpreted.

## Stress testing is complementary

VaR and ES summarize a modeled distribution. Stress testing asks what happens under specified severe scenarios.

A portfolio can have acceptable model-based VaR and still be vulnerable to liquidity shocks, basis breakdowns, concentration, or structural market changes not represented in the estimated distribution.

Risk management therefore needs both statistical tail measures and scenario analysis.

## Regulatory context

The Basel market-risk framework moved from VaR toward Expected Shortfall for internal-model capital because ES better captures the severity of tail losses and has more desirable aggregation properties. That policy change does not imply that ES eliminates model risk. citeturn0search1

## Conclusion

VaR answers:

> Where does the tail begin?

Expected Shortfall answers:

> How severe are losses once we are in the tail?

Both depend on the loss model, time horizon, portfolio dynamics, and validation procedure. The most important implementation discipline is to define the loss variable clearly and keep the estimation horizon consistent with the risk measure being reported.

## References

- Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent Measures of Risk.
- Basel Committee on Banking Supervision. Fundamental Review of the Trading Book.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*.
