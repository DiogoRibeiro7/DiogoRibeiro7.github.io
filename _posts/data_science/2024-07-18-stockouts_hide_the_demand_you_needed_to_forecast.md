---
permalink: '/data-science/stockouts_hide_the_demand_you_needed_to_forecast/'
title: 'Stockouts Hide the Demand You Needed to Forecast'
date: '2024-07-18'
categories:
- Data Science
tags:
- Supply Chain
- Demand Forecasting
- Inventory
- Censored Data
- Lost Sales
author_profile: false
classes: wide
seo_title: 'Stockouts Hide the Demand You Needed to Forecast'
seo_description: 'Observed sales are not always demand. When inventory caps what customers can buy, forecasting models learn censored sales unless the stockout process is modelled explicitly.'
seo_type: article
excerpt: >-
  Sales stop when inventory stops. Demand does not necessarily stop with them.
  A forecasting model trained on censored sales can therefore learn the inventory
  policy rather than the customer demand process.
summary: >-
  This article develops stockouts as a censoring problem. A Poisson example with
  true mean demand of eight units and inventory capped at seven units produces
  observed mean sales of only about 6.34 units, while nearly 69% of days end at
  the inventory ceiling. The article derives the correct censored likelihood,
  explains the feedback loop between low stock and low forecasts, and extends the
  problem to substitution, promotions, lost sales, exposure bias, and inventory-
  aware demand reconstruction.
keywords:
- censored demand
- lost sales
- stockout bias
- demand forecasting
- retail inventory
- substitution
why_this_exists: >-
  Forecasting pipelines frequently use historical sales as a direct proxy for
  demand. During stockouts this proxy is censored by the inventory system itself.
  This article makes that censoring explicit and shows why naive forecasting can
  underestimate demand precisely where inventory is already insufficient.
evidence: >-
  An original Poisson censoring example, the censored likelihood for stock-
  constrained sales, exact calculations of observed-sales bias and expected lost
  sales, and established operations-research work on demand estimation under
  unobserved lost sales and stockout-based substitution.
methodology: >-
  Define latent demand and observed sales through a minimum operator. Derive the
  expectation of observed sales, the probability of censoring, and the censored
  likelihood under Poisson demand. Compare naive estimation from observed sales
  with a likelihood that retains stockout information, then extend the framework
  to substitution and decision-feedback effects.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-data-science-dashboard.jpg
  og_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-dashboard.jpg
  twitter_image: /assets/images/headers/photo-data-science-dashboard.jpg
---

<!--
Development contract
Question: When can historical sales be treated as historical demand?
Claim: If available inventory constrains sales, the observed series is a censored version of latent demand. Treating it as fully observed demand systematically understates demand in stockout periods and can create a self-reinforcing forecasting and replenishment bias.
Counterclaim: Sales can be an adequate demand proxy when stock availability is effectively unconstrained, when stockouts are rare, or when customer demand is directly observed through orders, searches, requests, or backorders. The problem is not sales data itself but unmodelled censoring.
Evidence object: Poisson demand with mean eight, inventory cap seven, exact observed-sales expectation of about 6.34, censoring probability of about 68.7%, expected lost demand of about 1.66 units/day, and a censored Poisson likelihood.
Failure case: Assuming every zero or low sale is caused by stockout, imputing lost sales without modelling substitution or abandonment, or reconstructing latent demand with a model that simply reproduces its own assumptions.
Reader payoff: Recognise stockout censoring in historical data, distinguish demand from sales, write the correct likelihood contribution for censored observations, and design forecasting backtests that do not reward models for reproducing inventory constraints.
Exclusions: Estimating demand for a specific retailer or SKU without inventory-position data, prescribing a universal lost-sales imputation method, and claiming that all retail sales series require censoring correction.
-->

A forecasting system usually starts from a historical target.

In supply chains, that target is often sales.

The assumption is implicit:

$$
\text{sales}=\text{demand}.
$$

When inventory is available, the approximation may be good.

When inventory runs out, it is wrong by construction.

A customer can want a product that cannot be purchased because the shelf is empty, the warehouse is out of stock, the production component is unavailable, or the online inventory promise has already reached zero.

Observed sales then stop.

Latent demand may continue.

The forecasting problem therefore changes from ordinary prediction to censored-data inference.

This distinction matters because the inventory system that creates the censoring is usually the same system that later consumes the forecast. A weak replenishment policy can suppress observed sales, the forecasting model can interpret those suppressed sales as weak demand, and the next replenishment decision can become even smaller.

The forecast begins to learn the consequences of inventory scarcity as if they were customer preference.

## Sales are the minimum of demand and availability

Let latent demand during period $t$ be

$$
D_t.
$$

Let the maximum quantity available for sale be

$$
I_t.
$$

If unmet demand is lost and no backorder is recorded, observed sales are

$$
S_t
=
\min(D_t,I_t).
$$

This equation contains the entire censoring problem.

When

$$
D_t<I_t,
$$

we observe demand exactly:

$$
S_t=D_t.
$$

When

$$
D_t\ge I_t,
$$

we observe only

$$
S_t=I_t.
$$

The second case does not tell us the realised demand.

It tells us only that demand was at least as large as available inventory.

Formally,

$$
D_t\ge I_t.
$$

That is a censored observation.

Treating it as

$$
D_t=I_t
$$

throws away the inequality.

## A simple example shows how large the bias can be

Suppose true daily demand is Poisson:

$$
D
\sim
\operatorname{Poisson}(8).
$$

The true mean demand is therefore

$$
\mathbb E[D]=8.
$$

Now suppose inventory available each day is capped at

$$
I=7.
$$

Observed sales are

$$
S=\min(D,7).
$$

The expected observed sales are

$$
\mathbb E[S]
=
\sum_{d=0}^{6}
d,P(D=d)
+
7P(D\ge7).
$$

For

$$
D\sim\operatorname{Poisson}(8),
$$

this gives approximately

$$
\mathbb E[S]
\approx
6.336.
$$

The observed sales mean understates true mean demand by

$$
8-6.336
=
1.664
$$

units per day.

Expressed proportionally,

$$
\frac{8-6.336}{8}
\approx
20.8%.
$$

A forecasting model trained naively on sales is therefore being shown a process whose mean is about one fifth below the actual demand mean.

The forecasting algorithm did not create the bias.

The target variable did.

## Most observations can pile up at the inventory ceiling

The probability that observed sales equal the full available inventory is

$$
P(S=7)
=
P(D\ge7).
$$

For the Poisson example,

$$
P(D\ge7)
\approx
0.6866.
$$

So nearly

$$
68.7%
$$

of all days record sales of exactly seven units.

That repeated value can look like stable demand.

It is not.

It is the capacity of the inventory system to satisfy demand.

The sales histogram contains a point mass at the stock limit because many different latent demand values are mapped to the same observed sale:

$$
D=7,8,9,10,\ldots
$$

all become

$$
S=7.
$$

This is information loss.

A forecasting model cannot recover those values from sales alone unless additional assumptions or additional observations are introduced.

## Expected lost sales are the missing part of the mean

Because

$$
S=\min(D,I),
$$

we can write

$$
D
=
S+(D-I)^+,
$$

where

$$
x^+=\max(x,0)
$$

on periods where inventory constrains sales.

Taking expectations gives

$$
\mathbb E[D]
=
\mathbb E[S]
+
\mathbb E[(D-I)^+].
$$

Therefore,

$$
\mathbb E[(D-I)^+]
=
\mathbb E[D]-\mathbb E[S].
$$

In the example,

$$
\mathbb E[(D-7)^+]
\approx
1.664.
$$

That is the expected unobserved demand per day.

The quantity is not visible in the transaction table.

It is implied by the latent demand model.

This is why lost-sales estimation is an inference problem rather than a data-cleaning step.

## The correct likelihood keeps the censoring information

Suppose demand follows a Poisson model with unknown rate

$$
\lambda.
$$

For an uncensored day where

$$
S_t<I_t,
$$

we know

$$
D_t=S_t.
$$

The likelihood contribution is

$$
P(D_t=S_t\mid\lambda)
=
\frac{
e^{-\lambda}\lambda^{S_t}
}{
S_t!
}.
$$

For a censored day where

$$
S_t=I_t,
$$

we do not know the exact demand.

We know only

$$
D_t\ge I_t.
$$

The correct likelihood contribution is therefore

$$
P(D_t\ge I_t\mid\lambda)
=
1-
P(D_t<I_t\mid\lambda).
$$

For a Poisson model,

$$
P(D_t\ge I_t\mid\lambda)
=
1-
\sum_{d=0}^{I_t-1}
\frac{
e^{-\lambda}\lambda^d
}{
d!
}.
$$

The full likelihood is a product of exact-probability terms for uncensored observations and tail-probability terms for censored observations.

This is the information that a naive sales regression discards.

The censoring indicator is not a nuisance column.

It changes the likelihood.

## Replacing censored demand with the stock level biases the model downward

A naive estimator treats every observed sale as exact demand.

For the Poisson model, the sample mean of observed sales is then used as an estimator of the rate:

$$
\hat\lambda_{\mathrm{naive}}
=
\bar S.
$$

With enough data under the fixed inventory cap,

$$
\hat\lambda_{\mathrm{naive}}
\rightarrow
\mathbb E[\min(D,7)].
$$

In the example,

$$
\hat\lambda_{\mathrm{naive}}
\rightarrow
6.336
$$

even though the true value is

$$
\lambda=8.
$$

The estimator is not merely noisy.

It converges to the wrong quantity.

More data make the biased target more precisely estimated.

This is a supply-chain version of a general statistical warning:

A large dataset cannot repair a target variable that is systematically censored by the process being studied.

## Inventory can create a feedback loop in the training data

Suppose replenishment for the next period is partly based on the current demand forecast.

A simplified feedback loop is

$$
\text{low inventory}
\rightarrow
\text{more censoring}
\rightarrow
\text{lower observed sales}
\rightarrow
\text{lower forecast}
\rightarrow
\text{lower replenishment}.
$$

The data-generating process is therefore endogenous to the forecasting policy.

This differs from an ordinary time-series setting where the forecasting model passively observes the process.

Inventory decisions change which part of demand becomes observable.

The target distribution seen by the model can therefore depend on previous model outputs.

This creates a policy-data feedback problem.

## A stockout day is informative even when exact lost demand is unknown

Suppose the available quantity is

$$
I_t=12
$$

and observed sales equal

$$
S_t=12.
$$

The exact demand is unknown.

But the observation is not uninformative.

It implies

$$
D_t\ge12.
$$

A day with sales 12 and no stockout means something different.

If availability were

$$
I_t=30
$$

and sales were

$$
12,
$$

then demand is observed as 12.

The same sales quantity carries different statistical meaning depending on inventory availability.

This is why forecasting datasets need inventory-state information.

A table containing only date, SKU, and units sold cannot distinguish those cases.

## In-stock indicators are necessary but not always sufficient

Many retail datasets contain an in-stock flag.

That helps.

But a binary availability indicator may still be insufficient.

Suppose an item begins the day with eight units and sells out by noon.

The daily record may contain

$$
S_t=8
$$

and

$$
\text{stockout}=1.
$$

Demand during the first half of the day was at least eight.

Demand during the remaining hours is unobserved.

The amount of censoring depends on when the stockout occurred.

Sachs and Minner use the timing of sales events to infer unsatisfied demand patterns in a censored newsvendor setting. Their motivation is exactly this issue: a daily stockout indicator contains less information than the within-day sales process that led to the stockout.

Higher-frequency inventory and sales data can therefore improve demand reconstruction even when total daily lost sales remain unobserved.

## Backorders change the observation model

Not every stockout creates lost demand.

In some systems, unmet demand becomes a backorder.

Then the customer request may still be recorded.

If an order of ten units arrives while only six are available, the system may record

$$
D_t=10
$$

and create four backordered units.

In that case, demand is much closer to directly observed.

The inventory problem remains.

The censoring problem changes.

This is why the distinction between lost-sales and backorder systems is not only operational.

It is statistical.

A forecasting team should know what the transaction system records when supply is insufficient.

## Customer abandonment creates latent demand

In a lost-sales system, a customer can respond to stockout in several ways.

The customer may:

- abandon the purchase
- return later
- buy from another store
- order online
- choose another brand
- choose another size or variant
- buy a higher- or lower-priced substitute

Each response produces a different observed-data pattern.

The latent quantity of interest must therefore be defined carefully.

Is demand the preferred product demand before substitution?

Is it category demand?

Is it demand for any acceptable substitute?

Is it store-level demand or network-level demand?

There is no universal answer.

The forecast target should match the inventory decision.

## Substitution contaminates more than the stocked-out SKU

Consider two products, $A$ and $B$.

Let primary latent demands be

$$
D_A
$$

and

$$
D_B.
$$

Suppose product $A$ stocks out.

A fraction

$$
\rho
$$

of unmet demand for $A$ switches to $B$.

Then observed demand pressure on $B$ can be approximated by

$$
D_B^{\mathrm{obs}}
=
D_B
+
\rho(D_A-I_A)^+,
$$

subject to $B$'s own inventory constraint.

A naive forecasting pipeline now makes two errors.

It underestimates baseline demand for $A$.

It overestimates baseline demand for $B$.

When $A$ returns to stock, the apparent demand for $B$ may fall.

A model without availability features can interpret this as arbitrary time-series variation.

It is a cross-product inventory effect.

Transchel's work on stockout-based substitution formalises how customer switching changes optimal inventory decisions across an assortment. The important forecasting implication is that SKU demand series are not independent when customers substitute.

## Stockouts can distort promotion measurement

Suppose a promotion increases latent demand.

Inventory is insufficient.

Observed sales rise only until the product stocks out.

The apparent promotional lift is therefore capped by availability.

A naive estimator might compare sales before and during the promotion:

$$
\widehat{\text{lift}}
=
\bar S_{\mathrm{promo}}
-
\bar S_{\mathrm{baseline}}.
$$

If promotional periods stock out more often, the estimator is downward biased for demand lift.

This creates a causal-inference problem in addition to a forecasting problem.

The treatment changes demand.

The treatment also changes the probability that demand becomes censored.

Observed sales are therefore a post-treatment quantity filtered through inventory availability.

A forecast model trained on these observations can systematically underlearn promotion effects.

## Zero sales can mean very different things

A daily sale of zero may mean:

$$
D_t=0.
$$

It may also mean:

$$
I_t=0.
$$

Those are operationally opposite states.

The first suggests no customer demand.

The second can occur under high demand that exhausted inventory before the observation window began.

If the dataset does not distinguish them, a model can interpret stockout days as evidence of zero demand.

This is especially damaging for intermittent-demand forecasting.

Sparse sales are often treated as sparse demand.

Inventory censoring can create artificial intermittency.

## Historical availability is a confounder for descriptive demand analysis

Suppose Region A has lower sales than Region B.

A naive interpretation is

$$
\text{demand}_A
<
\text{demand}_B.
$$

But if Region A has lower availability, the observed difference can instead reflect inventory constraints.

Write

$$
S_r
=
\min(D_r,I_r).
$$

Then a regional sales comparison mixes demand and supply.

This matters in assortment planning, store clustering, market sizing, and capacity allocation.

A low-selling location may be low demand.

It may also be chronically understocked.

Sales alone cannot distinguish the two.

## Censoring is informative because inventory is not assigned randomly

In many textbook censoring models, the censoring mechanism may be assumed independent of the latent outcome after conditioning on covariates.

Inventory censoring often violates that simplification.

High-demand items receive more inventory.

Promotions trigger larger orders.

Managers intervene when they expect shortages.

Suppliers allocate scarce stock strategically.

Availability is therefore related to expected demand.

The censoring process is informative.

A realistic model may need

$$
P(D_t\mid X_t)
$$

and

$$
P(I_t\mid X_t,\text{policy history})
$$

or an explicit structural model of replenishment.

Simply deleting stockout observations can create another bias because stockouts tend to occur on systematically high-demand periods.

## Deleting stockout days is usually not neutral

A common practical workaround is:

> Remove all days where the SKU stocked out.

This avoids treating censored sales as exact demand.

It can still bias the dataset.

Stockout days are more likely when latent demand is high.

Conditioning on no stockout means selecting periods where demand was easier to satisfy.

The remaining sample can therefore have a lower demand distribution than the target population.

In the simple fixed-capacity setting,

$$
\text{keep day}
\iff
D<I.
$$

The retained sample follows a truncated distribution.

Its mean is below the unconditional mean.

Deleting the censored observations removes precisely the upper tail we are trying to understand.

## A censored likelihood is usually preferable to arbitrary imputation

Another workaround replaces a stockout day with an assumed demand value such as:

$$
S_t+20%.
$$

This creates a complete-looking dataset.

It also creates invented observations.

A model-based approach treats the lost amount as latent.

For parametric demand model (p$D_t\mid X_t,\theta$), a stockout contributes a survival probability:

$$
P(D_t\ge I_t\mid X_t,\theta).
$$

Uncensored days contribute ordinary density or probability terms.

The parameters are then estimated using both kinds of information.

The missing demand need not be imputed first.

If an imputed latent demand is useful operationally, it can be generated afterward from the conditional distribution

$$
p(
D_t
\mid
D_t\ge I_t,
X_t,
\hat\theta
).
$$

This preserves uncertainty rather than replacing every censored observation with one fabricated number.

## The posterior or conditional expectation is not the true lost sale

Suppose the model estimates

$$
\mathbb E[D_t\mid D_t\ge I_t,X_t]
=
10.4.
$$

That does not mean the true demand on that day was 10.4.

The latent demand may have been 8, 10, 15, or another value consistent with the model.

The expectation is a model-based summary.

This distinction matters when reconstructed demand is later used as if it had been directly observed.

A second forecasting model trained on point imputations can become overconfident because it ignores uncertainty in the reconstruction.

Multiple imputation, latent-variable estimation, Bayesian posterior prediction, or likelihood-based training can preserve more of that uncertainty.

## Policy changes can reveal previously hidden demand

Suppose inventory availability improves after a replenishment-policy change.

Sales increase.

A naive before-and-after analysis may conclude that customer demand increased.

But the latent demand distribution may be unchanged.

The system simply captured demand that had previously been censored.

This is another feedback between operations and measurement.

A forecasting model trained across the policy change can interpret higher post-change sales as a time trend.

An inventory-aware model can attribute part of the change to increased exposure.

This is conceptually similar to exposure problems in recommendation systems and advertising: outcomes are observed only when the system makes the opportunity available.

## Availability is an exposure variable

A customer cannot purchase inventory that is unavailable.

That makes product availability analogous to exposure.

Let

$$
A_t
$$

denote whether the item is available.

Observed purchase behaviour depends on both latent preference and exposure:

$$
P(\text{purchase})
=
P(
\text{purchase}
\mid
A_t=1
)
P(A_t=1).
$$

If

$$
A_t=0,
$$

a non-purchase contains little direct information about willingness to buy.

The same principle appears in ranking systems, digital advertising, and observational causal inference.

Absence of an observed action is not evidence of absence of latent demand when the opportunity for the action was removed.

## The forecasting target should be defined before reconstruction

There are at least three possible targets.

### Unconstrained primary demand

This is what customers would request for the SKU if it were always available.

It is useful for assortment and inventory planning.

### Realised demand after substitution

This includes customers who move between products when preferred items are unavailable.

It may be relevant for category-level inventory decisions.

### Network-level fulfilled demand

This records demand eventually served somewhere in the system after store switching, online fulfilment, or backordering.

It may be relevant for capacity planning.

These targets are different.

A censored-demand correction that estimates primary SKU demand may be inappropriate if the business decision concerns total category demand after substitution.

Demand reconstruction is not meaningful until the estimand is stated.

## Forecast accuracy should be evaluated against uncensored periods carefully

Suppose a model is evaluated only on periods where demand was fully observable.

This avoids comparing forecasts with censored targets.

It changes the evaluation population.

The test set now conditions on adequate inventory.

If stockouts occur preferentially during high-demand periods, the evaluation excludes the hardest and most operationally important cases.

A model can look accurate precisely because the backtest removes periods where demand exceeded expectations.

This is selection bias in model evaluation.

## Operational backtesting can evaluate latent-demand models indirectly

True lost demand is often unobserved, so direct forecast scoring is difficult.

Operational evaluation provides another route.

Suppose Model A and Model B imply different replenishment policies.

Replay both policies in a simulator or controlled experiment and compare:

- stockout frequency
- lost-sales estimates
- fulfilled demand
- revenue
- fill rate
- holding cost
- emergency replenishment
- substitution effects

If one model systematically improves decisions, that is evidence that its demand representation is more useful.

This is not proof that its reconstructed latent demand is numerically correct on every stockout day.

It evaluates the model for the operational task.

## Randomised inventory experiments can identify hidden demand more directly

When operationally feasible, deliberately varying inventory availability can provide information about censored demand.

Suppose similar stores or time periods receive different stock levels under random assignment.

Higher inventory exposes more of the upper demand tail.

Comparing realised sales across inventory conditions can help identify the relationship between availability and demand capture.

Such experiments have costs.

Overstocking can waste inventory.

Understocking can harm customers and revenue.

They are therefore not universally appropriate.

The conceptual point is that the censoring process can sometimes be manipulated to reveal information that passive historical data do not contain.

## Search, page views, and requests can provide auxiliary demand signals

Retail and digital systems may observe customer behaviour before purchase.

Examples include:

- product page views
- search queries
- add-to-cart events
- wait-list registrations
- attempted orders
- quote requests
- store enquiries
- substitute purchases

These signals do not equal demand automatically.

They can contain information about latent demand during stockout periods.

For example, if page views remain high after inventory reaches zero, the observed zero sales should not be interpreted as zero customer interest.

Auxiliary signals are particularly useful when combined with explicit availability data.

## Lost-sales models should be validated against periods with high availability

One practical validation strategy is to identify periods where inventory was sufficiently high that censoring was unlikely.

In those periods, observed sales approximate latent demand more closely.

A model trained with censoring correction can be checked against those less-constrained observations.

If the model systematically overpredicts even when availability is abundant, its lost-sales reconstruction may be too aggressive.

If it performs well in unconstrained periods and improves inventory performance in backtests, confidence increases.

No single check is decisive.

The goal is to create opportunities for the latent-demand model to fail.

## Stockout substitution creates a network demand problem

In multi-SKU and multi-location systems, censoring does not occur independently.

A stockout at one node can redirect demand to another.

Let product-location nodes form a network.

A stockout can move demand along edges representing:

- substitute SKU
- neighbouring store
- alternative fulfilment centre
- online channel
- delayed purchase

The observed sales vector is then partly generated by the inventory state of the whole network.

Forecasting each SKU-location series independently can mistake redirected demand for autonomous local demand.

This is one reason supply-chain demand modelling eventually becomes a network problem.

The inventory state changes the path through which demand becomes visible.

## Censored demand can make low-stock policies look self-consistent

Suppose a planner stocks seven units.

Observed average sales become

$$
6.34.
$$

The planner concludes that average demand is about six units and retains a low stock target.

The next data batch again contains sales near six.

The policy appears validated by its own censored observations.

This is a dangerous form of self-consistency.

The data are not independently confirming the policy.

The policy helped create the data.

Whenever a model output influences future observation opportunities, evaluation must account for the feedback loop.

## The Poisson example contains the entire warning

True demand was

$$
D
\sim
\operatorname{Poisson}(8).
$$

Available inventory was

$$
I=7.
$$

Observed sales were

$$
S=\min(D,7).
$$

The true mean was

$$
8.
$$

The observed mean was only

$$
6.336.
$$

Expected lost demand was

$$
1.664
$$

units per day.

And the inventory ceiling was hit on approximately

$$
68.7%
$$

of days.

A naive model trained on sales would therefore learn a systematically lower demand process.

The remedy is not simply to add a more sophisticated forecasting algorithm.

The target must be modelled correctly.

When inventory constrains sales, demand becomes a latent variable.

Stockouts hide the observations the forecasting system most needs to understand.

## References

Berk, E., Gürler, Ü., & Levine, R. A. (2007). Bayesian demand updating in the lost sales newsvendor problem: A two-moment approximation. *European Journal of Operational Research*, 182(1), 256–281. https://doi.org/10.1016/j.ejor.2006.08.035

Hill, R. M. (1992). Parameter estimation and performance measurement in lost sales inventory systems. *International Journal of Production Economics*, 28(2), 211–215. https://doi.org/10.1016/0925-5273(92)90033-4

Sachs, A.-L., & Minner, S. (2014). The data-driven newsvendor with censored demand observations. *International Journal of Production Economics*, 149, 28–36. https://doi.org/10.1016/j.ijpe.2013.04.039

Transchel, S. (2017). Inventory management under price-based and stockout-based substitution. *European Journal of Operational Research*, 262(3), 996–1008. https://doi.org/10.1016/j.ejor.2017.03.075

Zipkin, P. H. (2008). Old and new methods for lost-sales inventory systems. *Operations Research*, 56(5), 1256–1263.
