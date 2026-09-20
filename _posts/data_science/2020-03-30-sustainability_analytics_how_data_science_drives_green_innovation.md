---
author_profile: false
categories:
- Data Science
classes: wide
date: '2020-03-30'
excerpt: Sustainability analytics should quantify environmental outcomes, baselines, boundaries, uncertainty, and rebound effects. Optimization is not automatically sustainability.
header:
  image: /assets/images/headers/photo-sustainability-energy.jpg
  og_image: /assets/images/headers/photo-sustainability-energy.jpg
  overlay_image: /assets/images/headers/photo-sustainability-energy.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-sustainability-energy.jpg
  twitter_image: /assets/images/headers/photo-sustainability-energy.jpg
keywords:
- sustainability analytics
- life cycle assessment
- carbon accounting
- resource optimization
- rebound effects
seo_description: A rigorous framework for sustainability analytics covering system boundaries, baselines, life-cycle effects, carbon accounting, optimization, causal attribution, and uncertainty.
seo_title: 'Sustainability Analytics: Measure the Environmental Outcome'
seo_type: article
summary: Data science can support sustainability only when environmental outcomes are defined explicitly. This article connects analytics with system boundaries, counterfactual baselines, life-cycle assessment, emissions accounting, and decision optimization.
tags:
- Sustainability
- Data Science
- Operations Research
title: 'Sustainability Analytics: Measure the Environmental Outcome'
---

Data science can reduce energy use, optimize routes, detect leaks, and improve material planning.

None of those activities is automatically sustainable.

The environmental claim depends on a measurable outcome.

A useful sustainability analysis begins with

$$
\boxed{
\text{system boundary}
+
\text{baseline}
+
\text{environmental metric}
+
\text{counterfactual}.
}
$$

Without those pieces, “AI for sustainability” can become ordinary efficiency work with a green label.

## Define the boundary

Suppose a routing model reduces fuel consumption in one distribution center.

What belongs in the analysis?

Possible boundaries include:

- one vehicle;
- one warehouse;
- one company;
- the entire supply chain;
- the product life cycle.

A local efficiency gain can shift emissions elsewhere.

For example, faster delivery may require more packaging or air freight.

System boundaries determine which trade-offs are visible.

## Carbon accounting

A simplified emissions calculation is

$$
E
=
\sum_i
A_i
EF_i,
$$

where:

- $A_i$ is activity, such as kWh or liters of fuel;
- $EF_i$ is an emission factor.

The uncertainty in $E$ comes from both activity data and emission factors.

If electricity emission intensity changes hourly, using one annual average can misrepresent the consequence of load shifting.

Temporal and geographic resolution matter.

## Scope 1, 2, and 3

Corporate inventories often distinguish:

- **Scope 1:** direct emissions from owned or controlled sources;
- **Scope 2:** emissions associated with purchased energy;
- **Scope 3:** other value-chain emissions.

Optimization that reduces Scope 1 emissions can increase Scope 3.

The accounting category should not be confused with total climate impact.

A model needs the boundary relevant to the decision.

## Baselines

Suppose energy use falls by 10%.

Compared with what?

Possible baselines include:

- previous year;
- business-as-usual forecast;
- matched untreated sites;
- engineering simulation;
- weather-normalized demand.

A before-after comparison can be biased by production volume, weather, prices, or operational changes.

The baseline is a statistical model.

## Causal attribution

If a company installs a new control algorithm and energy consumption falls, the causal effect is

$$
E[Y(1)-Y(0)],
$$

where $Y(1)$ is energy use with the intervention and $Y(0)$ without it.

Only one trajectory is observed.

Randomized rollout, difference-in-differences, synthetic controls, interrupted time series, or engineering models may help construct the counterfactual.

A dashboard trend alone does not establish savings caused by the algorithm.

## Energy optimization

For operations indexed by time $t$, an objective might be

$$
\min_{x_t}
\sum_t
c_t x_t
$$

for cost.

A sustainability objective might instead use marginal emissions intensity

$$
\min_{x_t}
\sum_t
e_t x_t.
$$

These are not the same problem.

The cheapest hour is not necessarily the lowest-carbon hour.

Multi-objective optimization can include both.

## Water analytics

Water systems can be modeled through a balance:

$$
\text{input}
=
\text{use}
+
\text{reuse}
+
\text{loss}
+
\Delta\text{storage}.
$$

Sensor analytics can estimate leaks and abnormal flows.

But reducing withdrawal at one site may matter differently depending on local water scarcity.

A cubic meter of water is not environmentally equivalent everywhere.

Context matters.

## Waste and circularity

Waste reduction metrics should distinguish:

- material avoided;
- material reused;
- recycled material;
- downcycled material;
- landfill;
- incineration;
- hazardous waste.

A high recycling rate can coexist with increasing total material throughput.

Useful metrics therefore include both relative rates and absolute mass flows.

## Life-cycle assessment

A product can shift impact across life-cycle stages.

A simplified life cycle includes:

$$
\text{raw material}
\rightarrow
\text{production}
\rightarrow
\text{transport}
\rightarrow
\text{use}
\rightarrow
\text{end of life}.
$$

Electrification may increase manufacturing emissions while decreasing use-phase emissions.

A life-cycle perspective is needed to evaluate the net effect.

Data science can improve inventories and scenarios.

It does not remove the need for life-cycle accounting.

## Supply-chain transparency

Traceability systems can improve knowledge of material origin.

Blockchain is one possible database architecture.

It does not verify that an upstream claim is true.

An immutable ledger can preserve an incorrect input perfectly.

Verification, audit, certification, and measurement remain necessary.

Technology should not be confused with evidence.

## Logistics optimization

A vehicle-routing problem may minimize

$$
\sum_{(i,j)}
d_{ij}x_{ij},
$$

where $d_{ij}$ is distance.

Emissions depend additionally on:

- vehicle type;
- load;
- speed;
- congestion;
- fuel;
- refrigeration;
- empty returns.

Distance is a proxy.

If emissions are the objective, model emissions directly when feasible.

## Rebound effects

Efficiency can reduce the cost of using a resource.

That can increase demand.

If energy per unit falls by 20% but production grows by 30%, total energy use can rise.

Let

$$
I
=
\frac{E}{Q}
$$

be energy intensity.

Then total energy is

$$
E=I Q.
$$

A fall in intensity does not guarantee a fall in total impact.

Both intensity and absolute totals should be reported.

## Model footprint

The computational footprint of analytics itself can matter, especially for large repeated training runs.

But the correct comparison is consequential.

If a model consumes 1 MWh of electricity to train but reliably avoids 1,000 MWh of future consumption, focusing only on training energy misses the system effect.

Likewise, vague claims that AI savings automatically outweigh computation are unsupported.

Measure both when material.

## Uncertainty

Environmental metrics often combine uncertain inputs.

Monte Carlo propagation can estimate the distribution of an output

$$
Z=f(X_1,\ldots,X_p)
$$

by repeatedly sampling uncertain inputs.

This is preferable to reporting a single emissions number with several hidden uncertain factors.

Scenario uncertainty should be separated from measurement uncertainty where possible.

## Optimization versus sustainability

Optimization asks

$$
\min_x f(x).
$$

Sustainability asks whether $f$ represents the environmental and social outcome we actually care about.

A perfectly optimized proxy can produce the wrong result.

The sequence should be:

1. define the system;
2. define the environmental objective;
3. establish the baseline;
4. quantify uncertainty;
5. optimize;
6. verify realized effects.

## Conclusion

Sustainability analytics is not a collection of green use cases for machine learning.

It is measurement and decision analysis under explicit environmental boundaries.

The strongest workflow is

$$
\boxed{
\text{boundary}
\rightarrow
\text{inventory}
\rightarrow
\text{counterfactual}
\rightarrow
\text{impact}
\rightarrow
\text{optimization}
\rightarrow
\text{verification}.
}
$$

Data science is valuable inside that chain.

It should not replace the chain.

## References

- Greenhouse Gas Protocol. *Corporate Standard* and *Scope 3 Standard*.
- ISO 14040:2006. *Environmental management — Life cycle assessment — Principles and framework*.
- ISO 14044:2006. *Environmental management — Life cycle assessment — Requirements and guidelines*.
