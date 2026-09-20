---
permalink: '/data-science/lead_time_is_a_distribution_not_a_number/'
title: 'Lead Time Is a Distribution, Not a Number'
date: '2024-10-24'
categories:
- Data Science
tags:
- Supply Chain
- Logistics
- Inventory
- Lead Time
- Uncertainty
author_profile: false
classes: wide
seo_title: 'Lead Time Is a Distribution, Not a Number'
seo_description: 'Using average lead time can severely understate inventory risk. A worked example shows how lead-time variability and distribution shape propagate into safety stock and service levels.'
seo_type: article
excerpt: >-
  Average lead time is not enough for inventory control. Two suppliers can have
  the same mean delivery time and radically different safety-stock requirements,
  service levels, and tail risk.
summary: >-
  This article develops lead time as a random variable rather than a fixed
  planning constant. A worked example compares deterministic five-day lead time
  with a supplier that delivers in either two or eight days with equal probability.
  Both have mean lead time five days, yet the inventory consequences are very
  different. The article derives lead-time demand variance, shows why a mean-only
  reorder point can collapse service from 95% to about 50%, and explains why even
  a normal approximation based on mean and variance can fail when the lead-time
  distribution is multimodal.
keywords:
- lead time variability
- safety stock
- lead time demand
- supply chain uncertainty
- stochastic inventory
- reorder point
why_this_exists: >-
  Supply-chain systems often store lead time as one scalar and then treat that
  value as though it fully described replenishment uncertainty. This article
  shows mathematically what is lost when variability and distribution shape are
  removed from the planning problem.
evidence: >-
  An original stochastic lead-time example, analytical variance decomposition
  for lead-time demand, exact mixture calculations for cycle service, and
  standard inventory-control results for reorder points and safety stock.
methodology: >-
  Model daily demand as independent with fixed mean and variance and lead time
  as an independent random variable. Derive the mean and variance of cumulative
  demand over random lead time using the laws of total expectation and total
  variance. Compare a deterministic lead time with a bimodal lead-time
  distribution having the same mean, then evaluate reorder points under exact
  and approximate methods.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-network-cables.jpg
  og_image: /assets/images/headers/photo-network-cables.jpg
  overlay_image: /assets/images/headers/photo-network-cables.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-network-cables.jpg
  twitter_image: /assets/images/headers/photo-network-cables.jpg
---

<!--
Development contract
Question: What is lost when a supply-chain system represents lead time with a single average?
Claim: The mean lead time does not determine the distribution of demand during replenishment. Variance, tail behaviour, multimodality, correlation, and state dependence can materially change safety stock and service performance.
Counterclaim: A fixed lead time can be an adequate approximation in stable systems with little variability, and mean lead time remains useful as a summary statistic. The problem is treating it as a complete stochastic description.
Evidence object: Two suppliers with identical mean lead time of five days, analytical lead-time demand moments, exact service calculations under a two-point lead-time mixture, and comparison with normal approximations.
Failure case: Assuming that larger lead-time variance always implies worse operations regardless of policy, treating a normal approximation as universally wrong, or using one historical lead-time distribution when the process is non-stationary.
Reader payoff: Derive how lead-time uncertainty enters inventory variance, recognise when mean-only planning fails, and design backtests and forecasting systems that retain the full replenishment-time distribution.
Exclusions: Recommending supplier contracts, estimating safety stock for a real firm without its service and cost assumptions, and claiming that all logistics lead times require highly complex probabilistic models.
-->

Lead time often enters planning systems as one number.

A supplier takes five days.

A lane takes three days.

A purchase order takes two weeks.

A warehouse replenishment cycle takes forty-eight hours.

The number is useful, but it is usually incomplete.

A mean lead time of five days does not tell us whether nearly every order arrives between 4.8 and 5.2 days, whether half arrive in two days and half in eight, or whether most arrive in four days while a small fraction takes several weeks. Those processes share a similar average and create very different inventory risk.

For inventory control, the relevant random quantity is not lead time alone. It is demand accumulated while replenishment is unavailable.

That quantity depends on both demand uncertainty and lead-time uncertainty.

## Lead time should be modelled as a random variable

Let

[
L
]

denote lead time in days.

Let daily demand be

[
D_1,D_2,ldots
]

with

[
mathbb E[D_t]=mu_D
]

and

[
operatorname{Var}(D_t)=sigma_D^2.
]

The demand that accumulates during lead time is

[
S_L
=
sum_{t=1}^{L}D_t.
]

If lead time were fixed at (L=ell), then under independence,

[
mathbb E[S_Lmid L=ell]
=
mu_Dell
]

and

[
operatorname{Var}(S_Lmid L=ell)
=
sigma_D^2ell.
]

Once (L) is random, we need to average over its distribution.

By the law of total expectation,

[
mathbb E[S_L]
=
mu_Dmathbb E[L].
]

The mean depends only on the mean lead time.

The variance does not.

Using the law of total variance,

[
operatorname{Var}(S_L)
=
mathbb E[
operatorname{Var}(S_Lmid L)
]
+
operatorname{Var}[
mathbb E(S_Lmid L)
].
]

Substituting the conditional moments gives

[
operatorname{Var}(S_L)
=
sigma_D^2mathbb E[L]
+
mu_D^2operatorname{Var}(L).
]

This formula contains the central result.

Lead-time uncertainty contributes through

[
mu_D^2operatorname{Var}(L).
]

When average daily demand is large, even moderate variability in lead time can dominate ordinary day-to-day demand variability.

## The mean can stay fixed while risk changes dramatically

Suppose daily demand has

[
mu_D=100
]

and

[
sigma_D=20.
]

Consider Supplier A.

Its lead time is essentially deterministic:

[
L_A=5.
]

Therefore,

[
mathbb E[L_A]=5
]

and

[
operatorname{Var}(L_A)=0.
]

Lead-time demand has mean

[
mathbb E[S_{L_A}]
=
100(5)
=
500
]

and variance

[
operatorname{Var}(S_{L_A})
=
20^2(5)
=
2000.
]

The standard deviation is

[
sqrt{2000}
approx
44.72.
]

Now consider Supplier B.

Half of orders arrive in two days and half arrive in eight days:

[
L_B
=
egin{cases}
2, & p=0.5,\
8, & p=0.5.
end{cases}
]

Its mean is still

[
mathbb E[L_B]
=
0.5(2)+0.5(8)
=
5.
]

The variance is

[
operatorname{Var}(L_B)
=
9.
]

The mean lead-time demand is therefore also

[
500.
]

But the variance becomes

[
operatorname{Var}(S_{L_B})
=
20^2(5)
+
100^2(9).
]

Thus,

[
operatorname{Var}(S_{L_B})
=
2000+90000
=
92000.
]

The standard deviation is approximately

[
303.32.
]

The two suppliers have the same mean lead time.

The uncertainty in demand accumulated during lead time differs by almost a factor of seven in standard deviation.

A planning system that stores only the value five cannot distinguish them.

## Mean lead time can produce the wrong reorder point

Suppose a planner wants a 95% cycle service level.

If lead time is incorrectly treated as fixed at five days and lead-time demand is approximated as normal, the reorder point is

[
r
=
mu_D L
+
z_{0.95}sigma_Dsqrt{L}.
]

Using

[
z_{0.95}approx1.645,
]

we obtain

[
r
=
500
+
1.645(44.72)
approx
573.56.
]

For Supplier A, this is reasonable under the stated assumptions.

Now apply exactly the same reorder point to Supplier B.

Conditional on a two-day lead time,

[
S_L
sim
mathcal N(200,20^2cdot2).
]

Conditional on an eight-day lead time,

[
S_L
sim
mathcal N(800,20^2cdot8).
]

The unconditional distribution is therefore a mixture:

[
F_{S_L}(x)
=
0.5F_2(x)+0.5F_8(x),
]

where (F_2) and (F_8) are the two conditional normal distributions.

At

[
r=573.56,
]

the probability of avoiding a stockout is approximately

[
0.50.
]

A reorder point intended to deliver 95% service produces only about 50% service.

The mean lead time was correct.

The stochastic model was wrong.

## The failure comes from ignoring a mixture distribution

Why does the service collapse so badly?

Because the process contains two operational regimes.

When the order arrives in two days, a reorder point near 574 units is extremely conservative.

When the order arrives in eight days, the same reorder point is far too low.

Averaging the two lead times into five days creates a scenario that rarely occurs.

The mean is mathematically valid.

Operationally, it can be fictitious.

This is a general issue with multimodal processes. Averages can describe the centre of a distribution that has little actual probability mass near that centre.

Examples in logistics include:

- air freight versus sea freight fallback
- normal customs clearance versus inspection
- regular supplier delivery versus capacity-constrained delay
- domestic sourcing versus emergency overseas sourcing
- ordinary warehouse processing versus congestion periods

A single lead-time mean can hide the existence of qualitatively different states.

## Mean and variance still may not be enough

Suppose we retain both

[
mathbb E[L]=5
]

and

[
operatorname{Var}(L)=9.
]

Using the variance formula, lead-time demand has mean

[
500
]

and standard deviation

[
303.32.
]

A normal approximation would then set a 95% reorder point at

[
r_{mathrm{normal}}
=
500
+
1.645(303.32)
approx
998.91.
]

This is far safer than the mean-only result.

It is also substantially above the exact 95th percentile of the mixture.

For Supplier B, the exact 95th percentile is approximately

[
872.50.
]

The moment-matched normal approximation therefore carries more stock than necessary for the target service in this example.

This is not because normal approximations are intrinsically poor.

It is because the true lead-time demand distribution is strongly bimodal.

Matching the first two moments cannot reproduce that shape.

The example creates two different lessons.

Ignoring lead-time variability can dramatically understate risk.

Summarising the distribution only through mean and variance can still distort tail decisions.

## Safety stock is a quantile problem

A reorder point for a target cycle service level (alpha) is fundamentally a quantile:

[
r_alpha
=
F_{S_L}^{-1}(alpha).
]

If the operational target is

[
alpha=0.95,
]

the planner needs the 95th percentile of demand accumulated during the replenishment period.

A formula such as

[
r
=
mu+z_alphasigma
]

is therefore not the definition of safety stock.

It is a consequence of assuming an approximately normal predictive distribution.

The actual problem is distributional.

This distinction matters whenever lead times are skewed, discrete, multimodal, censored, state dependent, or heavy tailed.

## Tail behaviour matters more than the mean for high service levels

Suppose two suppliers have the same mean lead time and similar variance.

Supplier C has a light-tailed distribution.

Supplier D usually delivers quickly but occasionally experiences very long delays.

For moderate service targets, the inventory implications may be similar.

At service levels such as

[
99%
]

or

[
99.9%,
]

the rare delays can dominate the required reorder point.

High-service inventory policy is a tail-estimation problem.

This is why replacing the empirical lead-time distribution with its average is especially dangerous for critical parts, medical supplies, production-constraining components, or items with severe stockout penalties.

The operational cost is often concentrated in events the average deliberately smooths away.

## Lead time and demand may not be independent

The previous derivation assumed that lead time and demand are independent.

That assumption can fail.

High market demand may simultaneously:

- increase customer orders
- congest production
- reduce supplier availability
- extend transportation times
- delay warehouse processing

In such a system, periods of high demand can coincide with longer lead times.

The risk is then worse than a model that estimates the two distributions separately and combines them under independence.

A more general representation is

[
P(D,L),
]

the joint distribution of demand and lead time.

Lead-time demand should then be evaluated under this joint structure.

The practical implication is important.

Demand forecasting and lead-time forecasting should not automatically be built as independent pipelines if the underlying processes interact.

## Calendar time is not always operational time

Lead times can also depend on how time is measured.

Five calendar days are not equivalent to five working days when weekends, holidays, port schedules, production calendars, or carrier cutoffs matter.

Suppose an order placed Friday afternoon is processed Monday morning.

Its measured lead time in hours may be much longer than the operational processing time.

If lead-time models ignore calendar structure, the estimated distribution can mix predictable scheduling effects with genuinely stochastic delays.

This can inflate variance while providing little useful information for decision making.

A better model may condition on:

[
	ext{day of week},
quad
	ext{holiday proximity},
quad
	ext{cutoff time},
quad
	ext{origin},
quad
	ext{destination},
quad
	ext{carrier},
quad
	ext{supplier state}.
]

The objective is not merely to forecast lead time accurately.

It is to forecast the conditional distribution relevant at the moment the replenishment decision is made.

## Late deliveries should not be reduced immediately to a binary label

Logistics analytics often converts lead time into

[
Y=
egin{cases}
1, & 	ext{late},\
0, & 	ext{on time}.
end{cases}
]

That classification can be useful for service reporting.

It discards much of the distribution.

A shipment one hour late and a shipment fourteen days late both receive the same label.

Inventory impact depends heavily on the magnitude of delay.

For replenishment, more useful targets can include:

- full lead-time distribution
- upper quantiles
- expected excess delay
- probability of exceeding several thresholds
- remaining time to arrival conditional on current status

A probabilistic ETA model can therefore provide substantially more operational information than a late/not-late classifier.

## Survival analysis is a natural model for incomplete lead times

Suppose some purchase orders are still open when the dataset is extracted.

Their final lead times are not observed.

Deleting them creates selection bias because long-running orders are exactly the observations most likely to remain incomplete.

This is right censoring.

Let (T) denote time to delivery.

The survival function is

[
S(t)
=
P(T>t).
]

For an order that has already been open for (t_0) days, the conditional probability that it remains open beyond (t_0+s) is

[
P(T>t_0+smid T>t_0)
=
rac{S(t_0+s)}{S(t_0)}.
]

That quantity is directly relevant to an operations team waiting for a shipment.

It answers a dynamic question:

Given that the order has already taken this long, how much longer might it take?

This is more informative than assigning the order the historical average lead time.

## Lead-time distributions can drift

Supplier lead time is rarely stationary forever.

Capacity changes.

Routes change.

Ports become congested.

A supplier opens another plant.

A contract changes carrier.

Customs procedures change.

Seasonality alters transit conditions.

A lead-time model estimated from two years of pooled history can therefore describe no current operating regime particularly well.

Let the distribution at time (t) be

[
F_{L,t}.
]

If

[
F_{L,t}

eq
F_{L,t-k},
]

historical observations should not necessarily receive equal weight.

Useful monitoring quantities include changes in:

[
mathbb E[L],
]

[
operatorname{Var}(L),
]

upper quantiles,

[
P(L>ell),
]

and calibration of probabilistic lead-time forecasts.

Change-point detection can be more useful than a generic anomaly detector when the process undergoes a persistent structural shift.

## Supplier comparisons should use distributions, not just averages

Suppose Supplier A has:

[
mathbb E[L_A]=5.0.
]

Supplier B has:

[
mathbb E[L_B]=4.7.
]

It is tempting to call B faster.

But suppose:

[
Q_{0.95}(L_A)=6
]

while

[
Q_{0.95}(L_B)=14.
]

For an inventory system designed around a high service target, Supplier A may be operationally easier to manage despite the slightly slower mean.

Supplier evaluation can therefore include:

- median lead time
- upper quantiles
- variance
- tail probabilities
- probability of missing contractual SLA
- conditional delay after a threshold
- temporal stability
- dependence on order size or demand state

The appropriate metric depends on how supplier uncertainty enters the downstream decision.

## Lead-time variability can dominate demand variability

Return to the variance decomposition:

[
operatorname{Var}(S_L)
=
sigma_D^2mathbb E[L]
+
mu_D^2operatorname{Var}(L).
]

The first term is demand variability accumulated over average lead time.

The second term is variability created by the randomness of lead time itself.

Using the numerical example,

[
sigma_D^2mathbb E[L]
=
20^2(5)
=
2000.
]

The lead-time contribution is

[
mu_D^2operatorname{Var}(L)
=
100^2(9)
=
90000.
]

Lead-time uncertainty contributes forty-five times as much variance as ordinary daily demand variation.

A forecasting team can spend months reducing demand RMSE while ignoring the much larger source of uncertainty entering through replenishment time.

This is precisely why supply-chain data science should evaluate the complete decision system rather than one predictive component in isolation.

## Lead-time forecasting and demand forecasting should meet inside the same predictive object

A natural computational approach is simulation.

At decision time (t):

1. sample a future lead time

[
L^{(m)}
sim
p(Lmid X_t),
]

2. sample future demand conditional on the relevant information,

[
D_{t+1}^{(m)},D_{t+2}^{(m)},ldots,
]

3. accumulate demand until the sampled replenishment arrival,

[
S^{(m)}
=
sum_{h=1}^{L^{(m)}}D_{t+h}^{(m)},
]

4. repeat for

[
m=1,ldots,M.
]

The resulting empirical distribution of

[
S^{(m)}
]

is a predictive distribution for lead-time demand.

Inventory quantities can then be extracted directly.

For example,

[
r_{0.95}
=
Q_{0.95}
left(
S^{(1)},ldots,S^{(M)}
ight).
]

This approach naturally handles non-normal lead times, nonlinear demand models, calendar effects, and interactions if those structures are represented in the simulations.

The inventory policy consumes the distribution it actually needs.

## Backtesting should replay the lead-time process

A lead-time model should not be evaluated only with MAE or RMSE.

Those metrics can be useful.

The downstream question is whether the predicted distribution supports better replenishment decisions.

A rolling historical simulation can proceed as follows.

At each order date:

- estimate the demand distribution using only information then available
- estimate the lead-time distribution using only information then available
- construct the lead-time demand distribution
- choose the reorder point or order quantity
- advance through the realised demand and realised delivery time
- record stockouts, inventory, backorders, and service

This produces an operational comparison between alternative lead-time models.

Two models can have similar MAE and very different upper-tail calibration.

The second difference can dominate inventory performance.

## Point accuracy can reward the wrong lead-time model

Suppose most deliveries take four days and a small minority take fifteen.

A model predicting five days for every order may achieve good MAE.

A second model may predict a distribution with a 95th percentile near fifteen days.

Its point forecast can look no better.

Its uncertainty model can be much more useful for safety-stock decisions.

This is the same distinction between point prediction and decision-relevant probabilistic prediction that appears in demand forecasting.

When the cost of rare delays is asymmetric, tail calibration matters.

## Quantile loss provides a natural evaluation for service targets

If the inventory policy requires a lead-time quantile, the corresponding forecast can be evaluated with quantile loss.

For quantile level (alpha),

[
L_alpha(y,q)
=
egin{cases}
alpha(y-q), & yge q,\
(1-alpha)(q-y), & y<q.
end{cases}
]

A model intended to estimate the 95th percentile should therefore be judged differently from a model intended to estimate the mean.

If the operational decision uses

[
Q_{0.95}(L),
]

optimising RMSE of point lead-time forecasts can target the wrong functional.

Forecasting should begin by specifying what downstream decision consumes the prediction.

## The historical mean can be correct and still operationally useless

The strongest lesson from the worked example is not that averages are bad.

Both suppliers truly have

[
mathbb E[L]=5.
]

The average is mathematically correct.

What fails is the inference that the same mean implies similar inventory risk.

Supplier A has essentially no lead-time variance.

Supplier B alternates between two and eight days.

A mean-based reorder point designed for 95% service produces only about 50% service for Supplier B.

The error comes from replacing a distribution with one statistic and then asking that statistic to support a tail decision.

Supply chains do not experience average lead times.

They experience realised lead times.

Inventory systems should therefore preserve the uncertainty that operations actually face.

## References

Hadley, G., & Whitin, T. M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.

Zipkin, P. H. (2000). *Foundations of Inventory Management*. McGraw-Hill.

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.

Prak, D., Teunter, R., & Syntetos, A. (2017). On the calculation of safety stocks when demand is forecasted. *European Journal of Operational Research*, 256(2), 454–461. https://doi.org/10.1016/j.ejor.2016.06.035

Prak, D., & Teunter, R. (2019). A general method for addressing forecasting uncertainty in inventory models. *International Journal of Forecasting*, 35(1), 224–238. https://doi.org/10.1016/j.ijforecast.2017.11.004
