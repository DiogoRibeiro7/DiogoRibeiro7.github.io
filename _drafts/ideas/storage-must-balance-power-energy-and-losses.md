---
title: 'Storage Must Balance Power, Energy, and Losses'
permalink: /environment/storage_must_balance_power_energy_and_losses/
author_profile: false
classes: wide
categories:
- Environment
tags:
- Energy Storage
- Renewable Energy
- Energy Systems
- Mathematical Modelling
excerpt: 'A six-hour solar-and-battery ledger shows why equal generation and demand totals can still require imports, and which constraint a larger battery actually relaxes.'
seo_title: 'Storage Must Balance Power, Energy, and Losses'
seo_description: 'Work through battery charging, discharge limits, efficiency losses, and terminal energy with a reproducible six-hour dispatch example.'
seo_type: article
header:
  image: /assets/images/headers/photo-control-room.jpg
  teaser: /assets/images/headers/photo-control-room.jpg
why_this_exists: 'Storage discussions often compare energy totals while hiding power limits and the order of events. A complete numerical ledger exposes those constraints.'
evidence: 'Original synthetic solar, demand, and storage trajectories with explicit energy-conservation checks; DOE background on storage quantities.'
methodology: 'Dispatch three storage configurations against the same hourly trace, account for all imported, curtailed, lost, and retained energy, and examine boundary assumptions.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: Why can a solar-and-storage system require imports even when total solar generation equals total demand?
Claim: Feasibility depends on the ordered power balance, storage state, conversion losses, and boundary conditions, not only on aggregate energy.
Counterclaim: Aggregate energy comparisons can provide useful first bounds when their omitted constraints are stated.
Evidence object: A reproducible six-hour dispatch ledger comparing three battery configurations and checking conservation independently.
Failure case: A deterministic short trace with a greedy controller cannot establish annual reliability, optimal investment, or avoided emissions.
Reader payoff: Identify whether an example is constrained by charging power, stored energy, losses, or the timing of supply before proposing more capacity.
Exclusions: Product recommendations, battery cost forecasts, a grid adequacy study, and claims about the best generation mix.
-->

A solar array generates 12 kilowatt-hours during a short operating window. Demand over the same window is also 12 kilowatt-hours. Add a battery, and it is tempting to conclude that the two totals can be made to match in practice.

In the example below, a 6 kWh battery with a 2 kW charging and discharge limit still requires **4.76 kWh of imported electricity**. Increasing the power limit to 4 kW reduces imports to 2.60 kWh without changing the battery's energy capacity. Increasing capacity to 10 kWh then reduces imports to 2.00 kWh, but does not eliminate them.

These numbers are not estimates for a household, a region, or a particular product. They are a small accounting experiment. Its purpose is to make four distinct constraints visible: how much energy can be stored, how quickly it can move, how much is lost, and when it becomes available.

## Give each quantity a unit and a boundary

Power is a rate, measured here in kilowatts. Energy is the integral of power over time, measured in kilowatt-hours. A constant 2 kW load consumes 2 kWh in one hour and 6 kWh in three hours.

Storage has both a power capacity and an energy capacity. These describe different restrictions: the rate of charging or delivery, and the quantity that can be held. The US Department of Energy makes the same distinction in its explanation of solar energy storage. [DOE: Solar Integration—Solar Energy and Storage Basics](https://www.energy.gov/cmei/systems/solar-integration-solar-energy-and-storage-basics)

For this example, $E_t$ is energy stored inside the battery at the end of an interval. Its maximum usable stored energy is $E_{\max}$. The power limit $P_{\max}$ applies to electricity entering or leaving the battery system on the AC side. This convention matters because conversion losses separate stored energy from electricity delivered to the load.

We choose charging efficiency $\eta_c=0.9$ and discharge efficiency $\eta_d=0.9$ purely for illustration. Taking in 1 kWh adds 0.9 kWh to the stored state. Later withdrawing all of that stored energy delivers 0.81 kWh. The implied round-trip efficiency is the product, 0.81, under this simplified model.

The parameters do not represent typical or guaranteed performance. Actual ratings may use different measurement boundaries, and performance can depend on operating conditions. Before comparing any real specifications, establish where their energy and power quantities are measured.

## Write the dispatch rule before calculating

Let $G_t$ be solar generation and $L_t$ demand, both constant within an interval of length $\Delta t$. Solar serves the simultaneous load first. Any surplus can charge the battery; any deficit can be served by discharge.

Charging power is

$$
C_t=\min\left\{(G_t-L_t)_+,\ P_{\max},\
\frac{E_{\max}-E_{t-1}}{\eta_c\Delta t}\right\}.
$$

Discharge delivered to the load is

$$
D_t=\min\left\{(L_t-G_t)_+,\ P_{\max},\
\frac{\eta_d E_{t-1}}{\Delta t}\right\}.
$$

The notation $(a)_+=\max(a,0)$ prevents negative surplus or deficit. Since only one is positive, this rule never charges and discharges simultaneously.

The stored state advances according to

$$
E_t=E_{t-1}+\eta_c C_t\Delta t-\frac{D_t\Delta t}{\eta_d}.
$$

Imports fill the remaining deficit. Surplus that cannot be charged is curtailed: it is available generation we do not use. We assume no exports, no charging from the grid, no standby consumption, no battery degradation, and no protected emergency reserve.

Those assumptions define a simple greedy controller. It uses available solar immediately and serves demand whenever it can. It does not optimise electricity prices or preserve charge for a later high-priority load. Changing that objective can change the chosen dispatch even with identical physical equipment.

## Follow six hours all the way through

Demand is 2 kW in each of six one-hour intervals. Solar power is

$$
(0,6,6,0,0,0)\ \text{kW}.
$$

Both energy totals are 12 kWh. The battery begins empty, has 6 kWh of usable stored-energy capacity, and can charge or discharge at up to 2 kW.

In hour one, no solar or stored energy is available, so the system imports 2 kWh. Future sunlight cannot be used retroactively.

In hour two, solar supplies the 2 kW load directly and leaves a 4 kW surplus. The charging limit accepts only 2 kW. Over the hour, 2 kWh enters the battery system, 1.8 kWh reaches its stored state, and 2 kWh of available solar is curtailed.

Hour three repeats that pattern. The battery now holds 3.6 kWh. It has unused energy capacity, but the narrow charging path prevented the system from filling it during the available window.

In hour four, delivering 2 kWh consumes $2/0.9=2.222\ldots$ kWh of stored energy. At the end of the hour, 1.378 kWh remains. That can deliver only 1.24 kWh in hour five, leaving a 0.76 kWh import requirement. The battery is empty in hour six, so that hour requires another 2 kWh of imports.

| Hour | Solar (kW) | Demand (kW) | Charge (kW) | Delivered discharge (kW) | Stored energy at end (kWh) | Imports (kWh) | Curtailed (kWh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 2 | 0 | 0 | 0 | 2.00 | 0 |
| 2 | 6 | 2 | 2 | 0 | 1.800 | 0 | 2 |
| 3 | 6 | 2 | 2 | 0 | 3.600 | 0 | 2 |
| 4 | 0 | 2 | 0 | 2.00 | 1.378 | 0 | 0 |
| 5 | 0 | 2 | 0 | 1.24 | 0 | 0.76 | 0 |
| 6 | 0 | 2 | 0 | 0 | 0 | 2.00 | 0 |

Imports total 4.76 kWh. Of the available solar, 4 kWh serves demand directly and 4 kWh enters storage. Storage later returns 3.24 kWh to the load. The remaining 4 kWh is curtailed, while conversion loses 0.76 kWh.

The model has accounted for every unit of energy. Equal generation and demand totals did not imply that generation was usable at the required times.

## Change one constraint at a time

Keep the same trace and efficiencies, but increase the power limit to 4 kW. During hour two, all 4 kW of surplus can now charge the battery, adding 3.6 kWh. During hour three, only 2.4 kWh of free stored capacity remains. Filling it requires $2.4/0.9=2.667$ kWh of charging input; the rest is curtailed.

The full battery can deliver $6(0.9)=5.4$ kWh across the final three hours. Their demand totals 6 kWh, so imports during that period fall to 0.6 kWh. Add the unavoidable first-hour import under the empty-start assumption, and total imports are 2.6 kWh.

Now increase stored-energy capacity to 10 kWh while retaining the 4 kW power limit. Both sunny hours accept their full surplus, storing 7.2 kWh in total. This is enough to supply all 6 kWh of demand in the last three hours. The battery finishes with 0.533 kWh still stored, and only the initial 2 kWh import remains.

| Usable stored capacity / power limit | Imports (kWh) | Curtailed (kWh) | Conversion loss (kWh) | Final stored energy (kWh) |
| --- | ---: | ---: | ---: | ---: |
| 6 kWh / 2 kW | 4.760 | 4.000 | 0.760 | 0 |
| 6 kWh / 4 kW | 2.600 | 1.333 | 1.267 | 0 |
| 10 kWh / 4 kW | 2.000 | 0 | 1.467 | 0.533 |

![Three panels show the solar and demand trace, stored-energy paths for three configurations, and imports of 4.76, 2.60, and 2.00 kilowatt-hours.](/assets/images/figures/environment_storage_constraints.png){: width="2105" height="665" loading="lazy"}

*Original synthetic dispatch. All configurations start empty and use 90 percent charging and discharge efficiencies. The final stored energy differs between configurations and must remain in the accounting.*

Losses increase in the configurations that use more storage. This is consistent with improved service: more electricity passes through a lossy conversion path instead of being curtailed. Looking only at the loss column would reward a system that seldom charges, even if it relies heavily on imports.

There is also no reason to expect an additional 4 kWh of capacity to have the same value as an additional 2 kW of power. Each relaxes a different constraint. In the first configuration, extra energy capacity alone would accomplish nothing over this trace because the battery never reaches its existing capacity.

## Check the full energy identity

For each interval, the AC-side power balance is

$$
G_t+I_t+D_t=L_t+C_t+U_t,
$$

where $I_t$ denotes imports and $U_t$ curtailment. Charging and discharging move energy through the storage boundary, while losses explain the difference between their electrical and stored-energy amounts.

Summing the electrical balance and storage-state equation gives

$$
E_0+\sum_t(G_t+I_t)\Delta t
=E_T+\sum_t(L_t+U_t)\Delta t+\sum_t\ell_t,
$$

with interval conversion loss

$$
\ell_t=\left[(1-\eta_c)C_t+
\left(\frac1{\eta_d}-1\right)D_t\right]\Delta t.
$$

For the 10 kWh / 4 kW case, the identity is

$$
0+12+2=0.533\ldots+12+0+1.466\ldots.
$$

Dropping final stored energy would make the accounting appear to lose an extra 0.533 kWh. Counting initial stored energy as free generation would create a similarly misleading advantage.

A conservation check is more useful here than verifying a single printed import total. It can expose an efficiency applied in the wrong direction, a missed time-step multiplier, or a controller that releases energy it never stored.

## An evening promise needs two bounds

Suppose the question concerns only three evening hours, with no solar and a constant 2 kW load. A battery starts with 6 kWh stored. Ignoring discharge losses, its energy label looks sufficient for 6 kWh of demand.

With 90 percent discharge efficiency, it can deliver at most 5.4 kWh. If its power limit is 2 kW, it serves the first two hours fully and part of the third, leaving 0.6 kWh to another source.

If its power limit is only 1 kW, it delivers 3 kWh over the three hours and leaves 3 kWh of demand to another source. It still contains 2.667 kWh at the end. Energy remains available inside the battery, but cannot pass through the discharge path quickly enough.

Under these constant-load assumptions, meeting the entire evening demand requires both

$$
P_{\max}\ge2\ \text{kW},\qquad
E_0\ge\frac{6}{0.9}=6.667\ldots\ \text{kWh}.
$$

The second bound concerns the actual initial state, not merely the maximum capacity. A sufficiently large empty battery cannot meet the promise. If charging starts from empty earlier in the day, the required electrical input is at least $6/(0.9\times0.9)=7.407\ldots$ kWh, and that input must also fit through the charging-power constraint during the available time.

These bounds are useful screening calculations. They become insufficient once the demand fluctuates, other services share the battery, or operating reserves restrict discharge.

## The horizon can decide the apparent result

Our six-hour example starts empty and ends wherever the controller leaves the state. This is appropriate for demonstrating a particular sequence, but it is not a neutral assumption for every comparison.

A repeating representative-day calculation often needs a condition such as $E_T=E_0$, so it cannot repeatedly begin with free stored energy or discard an inconvenient terminal deficit. A long simulation may instead use a warm-up period and report how its initial state affects the results. A specific outage scenario needs an explicit assumption about how much charge exists when the outage begins.

The time resolution also matters. An hourly average of 2 kW can conceal a short interval above the inverter's power limit. If serving that peak is part of the question, an hourly energy balance is too coarse even when it conserves energy exactly.

Likewise, one convenient solar trace does not reveal the consequence of several low-generation days in succession. Averaging those days with sunny days can remove the very persistence that stresses storage. The ordered sequence should be preserved whenever the stored state carries information from one interval into the next.

## What the calculation can support

This model supports a narrow conclusion: energy totals alone do not establish feasible dispatch, and the active constraint can change as capacity or power increases. It provides an inspectable starting point for a richer model.

It does not determine the cheapest configuration. That requires a stated objective and assumptions about equipment costs, maintenance, replacement, tariffs, and the value of unmet demand. It does not determine annual reliability, which requires suitable joint demand and generation scenarios, operational limits, and a reliability criterion.

It also does not calculate avoided emissions. Imported electricity and charging electricity need an emissions model tied to the question and timing; conversion losses alone cannot determine the answer. These are additional quantities to estimate rather than conclusions hidden inside a successful energy balance.

Before enlarging a storage proposal, inspect the time steps where it fails. Was the battery empty, full, charge-limited, discharge-limited, or reserved for another service? Those diagnoses point to different changes. More capacity helps only when the shortage it addresses is actually a capacity shortage.

## Reproduce the ledger

The [calculation and figure generator](https://github.com/DiogoRibeiro7/blog-reproducibility/blob/main/scripts/figures/engineering/storage_ledger.py) prints every interval with `--dry-run` and exports the figure without that flag. Its `storage_ledger` function accepts the generation and demand arrays, efficiencies, interval length, initial state, and capacity limits.

Independent tests verify both instantaneous balance and total energy conservation, including non-hourly intervals, zero capacity, zero power, and nonempty initial states. Another test reproduces the evening example where energy remains unused because discharge power is too low.

To adapt the ledger, keep its measurement boundaries visible and change one assumption at a time. The first useful result is a trace that explains where every kilowatt-hour went and why each unmet interval occurred. A larger optimisation model should preserve that ability to account for its answer.
