---
author_profile: false
categories:
- Data Science
classes: wide
date: '2024-02-17'
excerpt: "Climate financial risk is better treated as scenario-conditioned loss analysis than as a simple extension of short-horizon market VaR."
header:
  image: /assets/images/headers/photo-data-science-heatmap.jpg
  og_image: /assets/images/headers/photo-data-science-heatmap.jpg
  overlay_image: /assets/images/headers/photo-data-science-heatmap.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-heatmap.jpg
  twitter_image: /assets/images/headers/photo-data-science-heatmap.jpg
keywords:
- Climate risk
- Scenario analysis
- Physical risk
- Transition risk
- Climate finance
- Value at risk
permalink: '/data-science/climate_var/'
redirect_from:
- '/data science/climate_var/'
seo_description: "How to quantify climate-related financial risk using scenario-conditioned loss distributions, physical and transition risk models, and uncertainty analysis."
seo_title: "Climate Financial Risk Beyond Traditional VaR"
seo_type: article
tags:
- Climate and Environment
- Finance
- Data Science
title: "Climate Financial Risk Beyond Traditional VaR"
---

Climate-related financial risk is often compressed into the phrase **Climate VaR**. The label is convenient, but it can be misleading if it suggests that climate risk is just ordinary one-day market VaR with an extra climate variable added.

Traditional market VaR usually concerns a relatively short horizon and a loss distribution estimated from recent market dynamics. Climate risk can act over years or decades, depends on policy and technology pathways, and may involve structural changes that are not represented in historical return data.

That makes scenario analysis central.

## Physical and transition risk

Physical risk includes damage from heat, flood, wildfire, storms, sea-level rise, chronic temperature shifts, and other climate hazards.

Transition risk includes changes in regulation, carbon pricing, technology, demand, financing conditions, and asset values during decarbonization.

These channels can interact. A company may simultaneously face physical disruption and transition-related repricing.

## The first problem is exposure mapping

A climate model does not directly produce a financial loss.

The chain is closer to

$$
\text{hazard}
\rightarrow
\text{asset exposure}
\rightarrow
\text{vulnerability}
\rightarrow
\text{economic loss}
\rightarrow
\text{portfolio loss}.
$$

Each arrow contains assumptions.

A flood projection must be matched to asset location, floor height, defenses, insurance, replacement cost, downtime, supplier dependence, and recovery dynamics before it becomes a financial quantity.

## Scenario-conditioned losses

Let $S$ denote a climate-policy scenario and $L$ portfolio loss.

A useful object is

$$
P(L\mid S),
$$

the loss distribution conditional on a stated scenario.

One may then report a quantile

$$
q_\alpha(S)
=
F^{-1}_{L\mid S}(\alpha),
$$

or an expected shortfall within that scenario.

The scenario label matters. A 95% quantile under one transition pathway is not comparable with a 95% quantile under another unless their assumptions are explicit.

## Climate scenarios are not ordinary probability forecasts

Many climate-financial scenarios are exploratory pathways rather than mutually exclusive events with known probabilities.

Assigning precise probabilities to them can create false confidence.

It is often more honest to present a set of conditional outcomes:

- orderly transition
- delayed transition
- high physical-risk pathway
- sector-specific policy shock

and then discuss sensitivity across scenarios.

## Historical simulation is weak for structural risk

Historical returns cannot contain future carbon taxes, future sea-level exposure, or technologies that have not yet been deployed.

Historical data remain useful for calibration and market dynamics, but they are insufficient as the sole generator of climate scenarios.

Climate stress testing therefore combines physical climate models, sector models, macroeconomic assumptions, asset-level data, and financial transmission models.

## Uncertainty propagates through the chain

Climate models, damage functions, adaptation assumptions, policy pathways, discount rates, and financial response models all contribute uncertainty.

If a final number is reported as

> Climate VaR = EUR 83.4 million

without showing scenario and model uncertainty, the decimal precision is deceptive.

Sensitivity analysis is usually more informative than one point estimate.

## Dependence and cascading losses

Climate losses are not independent across assets.

A regional event can affect physical assets, suppliers, transport, energy systems, insurers, and credit conditions simultaneously.

Portfolio aggregation must therefore consider dependence rather than simply summing independent asset-level VaRs.

## Transition risk is not just a carbon-price multiplier

A simple transition model might write firm value as a function of emissions and carbon price.

Real repricing can also depend on substitution, technology adoption, regulation, financing costs, customer demand, and competitive responses.

Machine-learning models can help estimate some relationships, but extrapolating outside observed policy regimes remains difficult.

## A better computational example

A toy model should make its assumptions explicit instead of adding random 'climate sensitivity' noise to daily returns.

~~~python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClimateScenario:
    physical_loss_mean: float
    physical_loss_sd: float
    transition_loss_mean: float
    transition_loss_sd: float


def simulate_losses(
    scenario: ClimateScenario,
    simulations: int = 100_000,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate scenario-conditioned portfolio losses as fractions of value."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)

    physical = rng.normal(
        scenario.physical_loss_mean,
        scenario.physical_loss_sd,
        simulations,
    )
    transition = rng.normal(
        scenario.transition_loss_mean,
        scenario.transition_loss_sd,
        simulations,
    )

    return np.maximum(physical + transition, 0.0)
~~~

This is still a toy model. Its advantage is conceptual: the scenario assumptions are explicit and separate from the Monte Carlo engine.

## Backtesting is fundamentally limited

Traditional VaR can be backtested against repeated realized short-horizon losses.

Long-horizon climate scenarios cannot be backtested in the same way because we do not observe many independent realizations of 2050 under alternative policy pathways.

Validation must therefore rely more heavily on component models, hindcasts, physical consistency, sensitivity analysis, and comparison across plausible assumptions.

## Conclusion

Climate financial risk is not best understood as ordinary VaR with an extra covariate.

It is a scenario-conditioned loss problem with deep model uncertainty.

The strongest analysis makes the chain from hazard to portfolio loss explicit and reports sensitivity across scenarios rather than hiding structural uncertainty inside one apparently precise number.

## References

- Network for Greening the Financial System. Climate Scenarios for central banks and supervisors.
- Basel Committee on Banking Supervision. Principles for the effective management and supervision of climate-related financial risks.
- IPCC. Assessment reports and climate scenario framework.
