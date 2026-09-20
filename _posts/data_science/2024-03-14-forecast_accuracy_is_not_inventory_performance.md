---
permalink: '/data-science/forecast_accuracy_is_not_inventory_performance/'
title: 'Forecast Accuracy Is Not Inventory Performance'
date: '2024-03-14'
categories:
- Data Science
tags:
- Supply Chain
- Logistics
- Forecasting
- Inventory Optimization
- Decision Theory
author_profile: false
classes: wide
seo_title: 'Forecast Accuracy Is Not Inventory Performance'
seo_description: 'A lower RMSE does not guarantee lower inventory cost. A worked newsvendor example shows why forecast evaluation must be aligned with replenishment decisions and asymmetric costs.'
seo_type: article
excerpt: >-
  Forecasting metrics evaluate predictions. Supply chains pay for decisions.
  A model can achieve a lower RMSE and still produce higher stockout and
  inventory costs once its forecasts are passed through a replenishment policy.
summary: >-
  This article separates predictive accuracy from downstream inventory
  performance. A normal-demand example compares the mean forecast, which
  minimizes squared error, with the cost-optimal demand quantile under asymmetric
  shortage and holding costs. The lower-RMSE forecast produces more than twice
  the expected inventory cost. The discussion then extends the argument to
  lead-time demand, safety stock, probabilistic forecasting, backtesting,
  service levels, model bias, and forecast value added.
keywords:
- forecast accuracy
- inventory performance
- supply chain data science
- inventory optimization
- newsvendor
- probabilistic forecasting
why_this_exists: >-
  Forecasting projects are often evaluated with generic accuracy metrics even
  when the forecasts exist only to support replenishment decisions. This
  article shows mathematically why forecast metrics and operational objectives
  can rank models differently and develops a decision-oriented evaluation
  framework for supply chain forecasting.
evidence: >-
  An original normal-demand inventory example, the newsvendor critical
  fractile, decision-consistent point forecasting results, established work on
  forecast accuracy metrics, and research linking forecasting uncertainty to
  safety stock and inventory cost.
methodology: >-
  Compare a mean forecast with a cost-optimal quantile under asymmetric
  shortage and holding costs. Derive expected overage, expected shortage,
  RMSE, and total expected inventory cost analytically under normal demand.
  Extend the result to lead-time demand, predictive distributions, rolling
  inventory simulation, and policy-level evaluation.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-factory.jpg
  og_image: /assets/images/headers/photo-factory.jpg
  overlay_image: /assets/images/headers/photo-factory.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-factory.jpg
  twitter_image: /assets/images/headers/photo-factory.jpg
---

<!--
Development contract
Question: Does a more accurate demand forecast necessarily produce better inventory performance?
Claim: Forecast accuracy and inventory performance optimise different objective functions. A forecast that scores better under RMSE or MAE can produce higher operational cost when replenishment decisions face asymmetric shortage and holding costs.
Counterclaim: Forecast accuracy remains useful. Poor forecasts usually make good decisions harder, and well-chosen forecast metrics can be informative when they are aligned with the downstream decision. The problem is treating generic accuracy as the final business objective.
Evidence object: Normal-demand newsvendor model with mean 100, standard deviation 15, holding cost 1, shortage cost 9, analytical RMSE calculations, analytical expected-cost calculations, and the critical-fractile solution.
Failure case: Concluding that forecast metrics do not matter, replacing every forecasting problem with a cost function without checking policy assumptions, or claiming that a lower inventory cost in one simulation proves a universally better forecasting model.
Reader payoff: Understand why model rankings can reverse after replenishment, derive the decision-relevant demand quantile, design inventory-aware backtests, and separate forecasting quality from policy quality.
Exclusions: Recommending a specific commercial forecasting platform, prescribing safety-stock values for a real company without its cost and service assumptions, and claiming that RMSE is intrinsically a bad forecasting metric.
-->

Forecasting is usually presented as a prediction problem. Supply chains experience it as a decision problem.

A model predicts next week's demand. A replenishment rule converts that prediction into an order quantity. The order arrives after some lead time. Demand then consumes inventory. Excess units incur holding cost. Missing units create backorders, lost sales, emergency shipments, service failures, or some combination of them.

The forecasting model never pays RMSE.

The supply chain pays for what the replenishment decision does with the forecast.

This distinction sounds obvious, yet forecasting projects are often compared almost entirely through predictive metrics such as RMSE, MAE, MAPE, WAPE, or MASE. Those metrics answer useful questions about forecast errors. They do not automatically answer the operational question that motivated the forecast.

A lower forecast error can reduce inventory cost. It often does.

It does not have to.

The reason is mathematical. A forecast accuracy metric defines one loss function, while an inventory policy can operate under a different and asymmetric loss function. When the two objectives differ, they can rank forecasts differently.

## Forecasting metrics encode a decision even when they look neutral

Let demand in period (t) be (D_t), and let the point forecast be (f_t).

Squared error is

[
L_{mathrm{SE}}(D_t,f_t)
=
(D_t-f_t)^2.
]

The corresponding population objective is

[
mathbb E[(D-f)^2].
]

The value of (f) that minimizes expected squared error is the conditional mean,

[
f^ast_{mathrm{SE}}
=
mathbb E[Dmid X].
]

Absolute error uses

[
L_{mathrm{AE}}(D_t,f_t)
=
|D_t-f_t|,
]

and its population-optimal point forecast is a conditional median.

These results are not merely properties of convenient metrics. They reveal that an error function implicitly asks the forecast to target a particular functional of the predictive distribution.

Gneiting formalised this relationship in his work on point forecasting and consistent scoring functions. A point forecast should be evaluated by a scoring rule that is appropriate for the statistical functional the forecaster was asked to provide.

Supply-chain decisions add another layer.

If the downstream cost of underpredicting demand differs from the cost of overpredicting it, neither squared error nor absolute error necessarily represents the decision problem.

## Inventory costs are usually asymmetric

Consider a single-period inventory decision.

We choose stock level (q) before demand (D) is observed.

If

[
q>D,
]

we have excess inventory.

If

[
D>q,
]

we have a shortage.

Let (c_h) denote the unit overage or holding cost, and let (c_u) denote the unit underage or shortage cost.

A simple operational loss is

[
L(D,q)
=
c_h(q-D)^+
+
c_u(D-q)^+,
]

where

[
x^+=max(x,0).
]

This loss is asymmetric whenever

[
c_h
eq c_u.
]

If a stockout is nine times more expensive than carrying one excess unit, then an error of minus ten units and an error of plus ten units should not receive the same operational penalty.

RMSE does give them the same magnitude contribution.

That is not a defect in RMSE. RMSE is doing exactly what squared error says it should do.

The inventory problem is asking a different question.

## The optimal inventory decision is a quantile

For the cost function above, the expected cost is

[
C(q)
=
c_hmathbb E[(q-D)^+]
+
c_umathbb E[(D-q)^+].
]

Under standard regularity conditions, differentiating with respect to (q) gives

[
rac{dC(q)}{dq}
=
c_hF_D(q)
-
c_u[1-F_D(q)],
]

where (F_D) is the cumulative distribution function of demand.

Setting the derivative to zero,

[
c_hF_D(q)
=
c_u[1-F_D(q)].
]

Therefore,

[
F_D(q^ast)
=
rac{c_u}{c_u+c_h}.
]

The optimal stock level is

[
q^ast
=
F_D^{-1}
left(
rac{c_u}{c_u+c_h}
ight).
]

This is the classical newsvendor critical fractile.

The decision does not ask for the mean unless the cost structure and demand distribution make the mean the appropriate quantile.

It asks for a quantile determined by the relative cost of overage and underage.

This is the bridge between probabilistic forecasting and inventory control.

## A lower-RMSE forecast can produce more than twice the inventory cost

Now consider a concrete synthetic example.

Suppose demand follows

[
D
sim
mathcal N(100,15^2).
]

The mean demand is

[
mu=100
]

and the standard deviation is

[
sigma=15.
]

Assume one unit of excess inventory costs

[
c_h=1,
]

while one unit of shortage costs

[
c_u=9.
]

The critical fractile is therefore

[
alpha
=
rac{9}{9+1}
=
0.9.
]

The inventory-optimal quantity is the 90th percentile of demand:

[
q^ast
=
F_D^{-1}(0.9).
]

For a standard normal distribution,

[
Phi^{-1}(0.9)
approx
1.2816.
]

Therefore,

[
q^ast
=
100+1.2816(15)
approx
119.22.
]

Now compare two point forecasts.

**Forecast A** reports the conditional mean:

[
f_A=100.
]

**Forecast B** reports the inventory-optimal quantile:

[
f_B=119.22.
]

If forecasts are evaluated using RMSE, Forecast A is better.

For a constant forecast (f),

[
operatorname{MSE}(f)
=
operatorname{Var}(D)
+
[mathbb E(D)-f]^2.
]

For Forecast A,

[
operatorname{RMSE}_A
=
sqrt{15^2}
=
15.
]

For Forecast B,

[
operatorname{RMSE}_B
=
sqrt{
15^2+(119.22-100)^2
}.
]

Thus,

[
operatorname{RMSE}_B
approx
24.38.
]

According to RMSE,

[
A
]

is clearly superior.

Now evaluate the same two decisions using the inventory cost that motivated the forecast.

For normally distributed demand, define

[
z
=
rac{q-mu}{sigma}.
]

Expected excess inventory is

[
mathbb E[(q-D)^+]
=
(q-mu)Phi(z)
+
sigmaphi(z),
]

and expected shortage is

[
mathbb E[(D-q)^+]
=
(mu-q)[1-Phi(z)]
+
sigmaphi(z),
]

where (phi) and (Phi) are the standard normal density and distribution functions.

At

[
q=100,
]

we have

[
z=0
]

and

[
phi(0)
=
rac{1}{sqrt{2pi}}.
]

Expected excess inventory and expected shortage are both

[
15phi(0)
approx
5.984.
]

The expected operational cost is therefore

[
C_A
=
1(5.984)
+
9(5.984)
]

or

[
C_A
approx
59.84.
]

At

[
q=119.22,
]

the expected excess inventory is approximately

[
19.93,
]

while expected shortage falls to approximately

[
0.71.
]

The resulting expected cost is

[
C_B
=
1(19.93)
+
9(0.71)
]

or

[
C_B
approx
26.32.
]

The ranking reverses.

| Forecast | Point forecast | RMSE | Expected inventory cost |
| --- | ---: | ---: | ---: |
| A: conditional mean | 100.00 | 15.00 | 59.84 |
| B: 90th percentile | 119.22 | 24.38 | 26.32 |

Forecast A has substantially better RMSE.

Forecast B reduces the expected operational cost by more than half.

Nothing is inconsistent about this result.

The two metrics optimise different objectives.

## The forecast did not become worse when it moved away from the mean

It would be misleading to conclude that Forecast B is an inaccurate estimate of mean demand.

It is not trying to estimate mean demand.

The quantity

[
119.22
]

is an intentionally conservative demand quantile chosen because shortages are more expensive than excess stock.

If we ask for the expected demand, 119.22 is biased upward.

If we ask for the stock level that minimises the stated inventory cost, 100 is too low.

The statistical target is part of the forecasting specification.

This matters in machine learning because a model trained under squared-error loss is pushed toward a conditional mean. A model trained under quantile loss at level (alpha) is pushed toward the conditional (alpha)-quantile.

The training loss, forecast output, evaluation score, and downstream decision should therefore be coherent.

Otherwise, a forecasting pipeline can optimise one mathematical target and deploy the result into a system that needs another.

## Pinball loss is the inventory loss in another form

For quantile level

[
0<alpha<1,
]

the pinball loss can be written as

[
L_alpha(D,q)
=
egin{cases}
alpha(D-q), & Dge q,\
(1-alpha)(q-D), & D<q.
end{cases}
]

Ignoring a positive multiplicative constant, the inventory cost

[
c_h(q-D)^+
+
c_u(D-q)^+
]

is a pinball loss with

[
alpha
=
rac{c_u}{c_u+c_h}.
]

This equivalence is important.

It means that a demand quantile can be trained and evaluated using a statistically coherent scoring rule that is directly related to the operational asymmetry.

For

[
c_u=9
]

and

[
c_h=1,
]

the relevant quantile is

[
alpha=0.9.
]

A 0.9-quantile forecasting model should not be penalised for systematically lying above the conditional mean.

That upward shift is the decision.

## Operational value depends on the policy around the forecast

The newsvendor example deliberately removes many supply-chain details to isolate the loss-function mismatch.

Real replenishment systems contain additional state.

Inventory position can include on-hand stock, outstanding purchase orders, backorders, allocations, and sometimes reserved inventory.

A periodic-review order-up-to policy can be written schematically as

[
Q_t
=
max
left[
0,
S_t-IP_t
ight],
]

where (Q_t) is the new order, (S_t) is the order-up-to level, and (IP_t) is inventory position.

The target (S_t) depends on demand over the protection period.

If the review interval is (R) and lead time is (L), the relevant random quantity is not next-period demand.

It is cumulative demand over approximately

[
R+L.
]

Write this as

[
D^{(R+L)}
=
sum_{j=1}^{R+L}D_{t+j}.
]

A forecast can perform well one step ahead and still estimate the protection-period distribution poorly.

Inventory performance depends on the latter.

## Lead-time demand changes what forecast accuracy means

Suppose one-step demand errors are

[
e_{t+h}
=
D_{t+h}-hat D_{t+h}.
]

For lead time (L), the cumulative forecast error is

[
E_L
=
sum_{h=1}^{L}e_{t+h}.
]

Its variance is

[
operatorname{Var}(E_L)
=
sum_{h=1}^{L}
operatorname{Var}(e_{t+h})
+
2
sum_{h<k}
operatorname{Cov}(e_{t+h},e_{t+k}).
]

The covariance terms matter.

Multiplying one-step error variance by (L) implicitly assumes that forecast errors across horizons are independent and similarly distributed.

That assumption can be wrong even when the underlying demand process itself has little serial correlation.

Prak, Teunter, and Syntetos show that forecast-based safety-stock calculations can underestimate uncertainty when parameter estimation induces correlation among future forecast errors. In their analysed settings, conventional procedures can produce safety stocks that are materially too low and service levels below target.

The general lesson is broader than one formula.

Inventory decisions care about the distribution of cumulative demand during replenishment exposure.

A generic one-step RMSE does not fully describe that distribution.

## Bias can matter more than symmetric error magnitude

Consider two forecasting models with similar MAE.

Model A makes errors that are approximately symmetric around zero.

Model B produces persistent negative forecast errors:

[
mathbb E[D-hat D]>0.
]

Under a shortage-sensitive inventory policy, the models may perform very differently.

A small systematic underforecast can repeatedly lower order-up-to levels, leading to chronic stockouts.

A comparable overforecast may increase holding cost instead.

If shortage and holding costs are unequal, the sign of the forecast error has operational meaning.

Metrics that take an absolute value or square the error remove that sign.

That can be entirely appropriate for measuring prediction distance.

It can be inappropriate for measuring operational consequence.

This is why bias should be monitored separately even when a symmetric accuracy metric remains useful.

## Service level and expected cost are different objectives

Suppose management requires a cycle service level of

[
95%.
]

A policy may then choose a demand quantile near

[
0.95
]

for the relevant protection period, depending on the exact service definition and model assumptions.

That decision may not minimise economic cost.

If the implied shortage and holding costs correspond to a critical fractile of

[
0.80,
]

then choosing the 0.95 quantile intentionally carries more inventory than the newsvendor cost optimum.

This is not a mistake if the service requirement is genuine.

It shows that supply-chain objectives can be constrained.

A practical optimisation problem may look more like

[
min_{pi}
mathbb E[C(pi)]
]

subject to

[
operatorname{ServiceLevel}(pi)
ge
0.95.
]

Forecast evaluation should reflect this structure.

The model that achieves the lowest unconstrained expected cost can differ from the model or policy that best satisfies a service-level constraint.

## Cycle service level and fill rate should not be conflated

Even the phrase *service level* is ambiguous.

Cycle service level is commonly associated with the probability of completing a replenishment cycle without a stockout.

Fill rate concerns the fraction of demand units supplied immediately from stock.

These are not equivalent quantities.

A system can experience occasional large shortages and have a different fill rate from another system with frequent tiny shortages, even if both have similar stockout probabilities.

If the business objective is unit availability, evaluating a forecast-policy pair by stockout occurrence alone can be misleading.

If the objective is avoiding any stockout during a critical production run, fill rate may not capture the operational consequence sufficiently.

The forecasting objective should therefore inherit the service definition that matters to the supply chain.

## A better forecast can be neutralised by a poor policy

Suppose Model A provides a well-calibrated predictive distribution.

The replenishment system ignores the distribution and orders the point mean with no safety adjustment.

Model B provides a less accurate distribution, but the inventory policy uses its uncertainty correctly.

Model B can outperform operationally.

The result does not prove that B is the better forecasting model.

It proves that the combined system

[
	ext{forecast}
+
	ext{policy}
]

performed better.

This distinction is important when evaluating production systems.

Forecasting quality and policy quality are separate components.

A poor policy can hide the benefit of a better model.

A strong policy can compensate for some forecast weaknesses.

If the goal is to choose a forecasting model, the evaluation should control the replenishment policy.

If the goal is to choose a complete decision system, the combined policy should be compared.

Those are different experiments.

## Inventory backtesting should simulate decisions, not only errors

A supply-chain forecasting backtest should preserve the temporal order of decisions.

At time (t), the model should use only information available at (t).

It should generate the predictive quantities required by the replenishment policy.

The policy should determine the order.

The simulator should then advance time, receive previously placed orders according to lead time, observe demand, fulfil what it can, account for shortages, and update inventory state.

A simple cost function might be

[
C_t
=
c_hI_t^+
+
c_bB_t
+
c_omathbf 1(Q_t>0)
+
c_eE_t,
]

where

[
I_t^+
]

is positive ending inventory,

[
B_t
]

is backorder or lost-demand quantity,

[
c_o
]

is fixed ordering cost, and

[
E_t
]

represents emergency replenishment or another exception cost.

Total backtest cost is

[
C_{mathrm{total}}
=
sum_tC_t.
]

This simulation produces operational outputs that can be compared alongside forecast metrics.

Useful outputs may include:

- holding cost
- shortage cost
- lost sales
- backorder volume
- emergency shipments
- average inventory
- inventory turns
- fill rate
- cycle service level
- order frequency
- order variability

The point is not to replace forecasting metrics with one magical business metric.

It is to measure the consequences that the forecasting system was built to influence.

## The cost function should be specified before comparing models

If operational costs are chosen after the forecast results are visible, model evaluation becomes vulnerable to the same flexibility that affects statistical analysis elsewhere.

Suppose Model A performs better when shortage cost is

[
c_u=3,
]

while Model B performs better when

[
c_u=20.
]

Choosing 20 after seeing the result is not a neutral evaluation.

The cost assumptions should come from the business process where possible.

When costs are uncertain, sensitivity analysis is preferable to pretending that one precise number is known.

For example, evaluate the model ranking over

[
rac{c_u}{c_h}
in
{2,5,10,20}.
]

If Model B dominates across the range, the operational conclusion is robust.

If rankings reverse, the uncertainty in the cost ratio is decision-relevant information.

## Forecast accuracy is still useful

The argument should not be turned into the claim that RMSE, MAE, MASE, or related metrics are irrelevant to supply chains.

They are useful diagnostic tools.

Hyndman and Koehler show why some forecast accuracy measures behave poorly across series and motivate scaled measures such as MASE for comparative forecasting evaluation.

A forecasting model with very large errors is unlikely to support consistently good replenishment unless the policy is extraordinarily insensitive to those errors.

Accuracy metrics can also reveal whether a model improved after adding information such as promotions, weather, prices, calendar effects, or hierarchical structure.

The problem is the final inference:

[
	ext{lower forecast error}
Rightarrow
	ext{better supply-chain performance}.
]

That implication requires assumptions about the decision policy and loss function.

It is not a mathematical identity.

## Point forecasts hide the information inventory control actually needs

The newsvendor result depends on

[
F_D^{-1}(alpha).
]

That requires a predictive distribution or at least an estimate of the relevant quantile.

A mean forecast alone is insufficient.

Suppose two forecasting systems both predict mean demand of 100.

Model A believes

[
D
sim
mathcal N(100,5^2).
]

Model B believes

[
D
sim
mathcal N(100,25^2).
]

Their point forecasts are identical.

Their inventory implications are not.

For a 95th-percentile stock target,

[
q_A
approx
100+1.645(5)
=
108.23,
]

while

[
q_B
approx
100+1.645(25)
=
141.13.
]

The mean forecast contains no information about this difference in uncertainty.

A supply-chain forecasting system should therefore often produce more than one number.

Quantiles, predictive intervals, samples from the predictive distribution, or a full parametric distribution can provide the uncertainty information required by the policy.

## Probabilistic accuracy must also be decision-aware

Moving from point forecasts to probabilistic forecasts does not remove the evaluation problem.

A probabilistic forecast can be assessed with proper scoring rules such as the logarithmic score, CRPS, or quantile scores.

Those are statistically principled.

The downstream inventory decision can still weight parts of the distribution differently.

If extreme underforecasting is very expensive, calibration in the upper demand tail may matter more operationally than small improvements near the centre.

A model with slightly better overall CRPS can still estimate the 0.99 demand quantile poorly.

If the replenishment policy for a critical item depends on that quantile, the local failure matters.

Operational evaluation should therefore ask whether the distribution is accurate where the decision consumes it.

## Parameter uncertainty belongs inside the inventory problem

Suppose historical demand is assumed normal,

[
D_t
sim
mathcal N(mu,sigma^2).
]

We estimate

[
hatmu
]

and

[
hatsigma
]

from limited data.

A common shortcut inserts those estimates directly into the inventory formula as though they were known parameters.

This creates a predictive distribution that is too confident because it ignores estimation uncertainty.

Prak and Teunter analyse this interface explicitly. Their framework propagates uncertainty in estimated demand parameters into the predictive lead-time distribution before the inventory decision is made. They show that ignoring this uncertainty can generate insufficient safety stock, lower service, and higher cost.

This is a recurring data-science pattern.

Uncertain model parameters are converted into fixed inputs for an optimisation model.

The optimiser then behaves as though uncertainty has disappeared.

It has not.

The uncertainty has merely been omitted from the decision model.

## Forecast Value Added should be extended to decision value

Forecast Value Added asks whether a forecasting step improves accuracy relative to a baseline.

That is useful because sophisticated processes can make forecasts worse.

In an inventory setting, a second question should follow.

Did the improvement in the forecast change the replenishment decision enough to improve inventory performance?

Suppose a planner override reduces MAE from

[
12.0
]

to

[
11.7.
]

If both forecasts generate the same order quantity after lot-size rounding and safety-stock rules, the operational value of that improvement may be zero.

Conversely, a small change in a forecast near a reorder threshold can alter the decision substantially.

This suggests a decision-level analogue:

[
	ext{Decision Value Added}
=
C_{mathrm{baseline}}
-
C_{mathrm{new}}.
]

The name matters less than the principle.

Forecast improvements should eventually be evaluated through the decisions they affect.

## Aggregated accuracy can hide expensive failures

Suppose a retailer forecasts one thousand SKUs.

Nine hundred low-margin items are predicted extremely well.

One hundred critical high-margin items are predicted poorly.

An unweighted aggregate RMSE can still improve.

Operational performance can worsen.

The reason is weighting.

Forecast metrics usually weight observations according to their mathematical definition.

The business weights items through volume, margin, service criticality, substitution, shelf-life, production dependency, and shortage cost.

A supply-chain evaluation should make those weights explicit rather than assuming every forecast error has equal operational consequence.

This does not necessarily mean weighting forecast errors by revenue.

The correct weights depend on the decision.

A low-revenue component can have enormous shortage cost if it stops a production line.

A high-revenue item may have low shortage cost if substitution is easy.

Operational importance is a system property.

## Sales data can make the backtest optimistic

Inventory evaluation contains another trap.

Observed sales are not always equal to latent demand.

If a product stocks out, observed sales can be censored:

[
	ext{Sales}_t
=
min(D_t,I_t^{mathrm{available}}).
]

When inventory reaches zero, true demand can continue while recorded sales cannot.

A forecasting model trained on sales may therefore learn the consequences of historical inventory constraints.

An inventory simulation that then treats those sales as unconstrained demand can underestimate the cost of stockouts.

This issue deserves its own article, but it belongs in the evaluation framework because operational metrics are only meaningful if the demand process being simulated is credible.

Forecasting and inventory data cannot always be separated cleanly.

The inventory policy affects what demand becomes observable.

## Better evaluation uses two scorecards

For most supply-chain forecasting systems, I would keep two separate scorecards.

The first evaluates predictive quality.

It can include metrics such as:

[
	ext{MASE},
quad
	ext{RMSE},
quad
	ext{bias},
quad
	ext{quantile loss},
quad
	ext{coverage},
]

depending on the forecasting target.

The second evaluates operational performance under a fixed and documented replenishment policy.

It can include:

[
	ext{total cost},
quad
	ext{holding cost},
quad
	ext{shortage cost},
quad
	ext{fill rate},
quad
	ext{cycle service},
quad
	ext{average inventory}.
]

A model that improves both is easy to defend.

A model that improves forecast metrics while worsening operations needs investigation.

A model that worsens a generic accuracy metric while improving the policy objective may be entirely rational if the forecasting target changed to match the decision.

This separation also makes debugging easier.

It tells us whether failure originates in prediction or in the policy that transforms predictions into actions.

## Supply-chain data science should optimise the decision interface

The worked example gives the central result.

For

[
Dsimmathcal N(100,15^2),
]

Forecast A uses the mean

[
100
]

and obtains

[
operatorname{RMSE}=15.
]

Forecast B uses the 90th percentile

[
119.22
]

and obtains the worse RMSE

[
24.38.
]

Under shortage cost

[
c_u=9
]

and holding cost

[
c_h=1,
]

their expected inventory costs are approximately

[
59.84
]

and

[
26.32.
]

The lower-RMSE forecast loses.

The reason is not that predictive accuracy is unimportant.

The reason is that RMSE optimises a symmetric squared-error problem while the inventory system faces an asymmetric economic problem.

Once a forecast enters an operational process, model quality cannot be defined independently of the decision that consumes it.

The forecasting model, uncertainty representation, inventory policy, lead time, service objective, and cost structure form one decision system.

Supply-chain data science becomes useful when those pieces are evaluated together.

## References

Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746–762. https://doi.org/10.1198/jasa.2011.r10138

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001

Prak, D., & Teunter, R. (2019). A general method for addressing forecasting uncertainty in inventory models. *International Journal of Forecasting*, 35(1), 224–238. https://doi.org/10.1016/j.ijforecast.2017.11.004

Prak, D., Teunter, R., & Syntetos, A. (2017). On the calculation of safety stocks when demand is forecasted. *European Journal of Operational Research*, 256(2), 454–461. https://doi.org/10.1016/j.ejor.2016.06.035

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.

Zipkin, P. H. (2000). *Foundations of Inventory Management*. McGraw-Hill.
