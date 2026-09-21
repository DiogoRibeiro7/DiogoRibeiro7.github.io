---
permalink: '/data-science/the_bullwhip_effect_as_variance_amplification/'
title: 'The Bullwhip Effect as Variance Amplification'
date: '2025-09-04'
categories:
- Data Science
tags:
- Supply Chain
- Logistics
- Bullwhip Effect
- Forecasting
- Inventory
- Time Series
author_profile: false
classes: wide
seo_title: 'The Bullwhip Effect as Variance Amplification'
seo_description: 'Why upstream orders can become much more volatile than customer demand. A worked order up to model shows how forecasting and lead time amplify variance even when demand is stationary.'
seo_type: article
excerpt: >-
  The bullwhip effect is not just a story about bad communication. Forecast
  updating, replenishment rules, lead time, batching, and local incentives can
  transform modest customer variation into much larger upstream order variation.
summary: >-
  This article derives the bullwhip effect as a variance amplification problem.
  Under independent demand, a moving average forecast and a simple order up to
  policy produce an analytical variance ratio of
  1 + 2L/n + 2(L/n)^2, where L is lead time and n is the forecast window.
  A worked example gives a bullwhip ratio of 2.5 even though customer demand is
  stationary. The article then develops the roles of lead time, forecast
  responsiveness, batching, price variation, shortage gaming, information
  sharing, capacity, and multi echelon feedback.
keywords:
- bullwhip effect
- variance amplification
- supply chain data science
- order variability
- demand forecasting
- order up to policy
why_this_exists: >-
  Bullwhip is often explained qualitatively as distorted information moving
  upstream. This article makes one source of that distortion explicit by
  deriving how a common forecasting and replenishment rule transforms demand
  variance into order variance.
evidence: >-
  An original analytical derivation for an order up to policy with moving
  average forecasting, exact variance calculations, a permanent demand shift
  example, and established work by Lee, Padmanabhan and Whang, Chen and
  colleagues, Sterman, and control theoretic analyses of replenishment rules.
methodology: >-
  Model customer demand as an independent stationary process. Forecast demand
  with an n period moving average and set the pipeline target proportional to
  lead time L. Derive the resulting order process, its mean, variance, and
  bullwhip ratio. Examine sensitivity to L and n, then extend the interpretation
  to batching, information sharing, capacity, and behavioural feedback.
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
Question: Why can upstream orders be substantially more variable than final customer demand?
Claim: Forecast updating and replenishment control can amplify demand variation even when customer demand itself is stationary. Lead time and forecast responsiveness determine the magnitude of this amplification under a simple order up to policy.
Counterclaim: Not every upstream variance increase is irrational or avoidable. Batch constraints, capacity, transport economics, promotions, deliberate smoothing, and service requirements can make different order profiles economically reasonable.
Evidence object: Independent demand model, n period moving average forecast, lead time L, exact order equation, analytical bullwhip ratio, sensitivity calculations, and a permanent demand step response.
Failure case: Treating all upstream variability as bullwhip, assuming that smoothing orders always improves the system, or interpreting centralized demand information as sufficient to eliminate amplification under every replenishment rule.
Reader payoff: Quantify one source of bullwhip, see how lead time and forecast windows enter the variance directly, and distinguish customer demand variability from variability created by the control policy itself.
Exclusions: Optimising a specific company's production schedule, recommending a universal replenishment rule, and claiming that the analytical model captures every source of supply chain amplification.
-->

Customer demand can be fairly stable while factory orders are not.

A retailer may sell roughly one hundred units per week, yet the distributor receives orders that move between eighty, one hundred, and one hundred and thirty. The manufacturer sees an even less stable order stream. Production then reacts to a pattern that looks much more volatile than the consumer market that ultimately created it.

This phenomenon is usually called the bullwhip effect.

The familiar description is that variability increases as one moves upstream through a supply chain. Lee, Padmanabhan, and Whang formalised the idea as information distortion in which orders become more variable than sales and identified several mechanisms that can create it, including demand signal processing, batching, price variation, and shortage gaming.

The phrase *information distortion* is useful, but it can sound more mysterious than the mathematics requires.

Even a transparent replenishment rule, applied rationally to stationary demand, can amplify variance.

The amplification appears because an order is not merely a copy of customer demand.

An order also corrects the desired inventory position and the desired pipeline.

Forecast revisions therefore enter the order stream.

Lead time determines how strongly those revisions matter.

## Orders contain demand replacement and forecast correction

Let customer demand in period $t$ be

$$
D_t.
$$

Assume for the moment that demand is independent over time with

$$
\mathbb E[D_t]=\mu
$$

and

$$
\operatorname{Var}(D_t)=\sigma^2.
$$

Suppose the retailer forecasts future demand using an $n$ period moving average,

$$
F_t
=
\frac{1}{n}
\sum_{j=0}^{n-1}
D_{t-j}.
$$

Let replenishment lead time be $L$ periods.

A simple order up to policy sets the desired pipeline approximately proportional to expected demand during lead time,

$$
S_t
=
LF_t.
$$

Ignore a constant safety stock for the moment. A constant buffer changes the level of the target but disappears when the target change is calculated.

The order in period $t$ must replace current demand and adjust the pipeline from the previous target to the new one:

$$
O_t
=
D_t
+
S_t-S_{t-1}.
$$

Substituting the target gives

$$
O_t
=
D_t
+
L(F_t-F_{t-1}).
$$

The moving average changes only because one new observation enters and one old observation leaves:

$$
F_t-F_{t-1}
=
\frac{D_t-D_{t-n}}{n}.
$$

Therefore,

$$
O_t
=
D_t
+
\frac{L}{n}
(D_t-D_{t-n}).
$$

Collecting terms,

$$
O_t
=
\left(
1+\frac{L}{n}
\right)D_t
-
\frac{L}{n}D_{t-n}.
$$

This equation contains the bullwhip mechanism.

The retailer does not simply pass current demand upstream.

It passes current demand plus a correction created by forecast updating.

## The mean order remains correct

The expected order is

$$
\mathbb E[O_t]
=
\left(
1+\frac{L}{n}
\right)\mu
-
\frac{L}{n}\mu.
$$

So

$$
\mathbb E[O_t]
=
\mu.
$$

On average, the retailer orders exactly the same quantity that customers demand.

There is no systematic overordering in the long run under this simple model.

Bullwhip is not a mean problem.

It is a variance problem.

## The variance is larger than customer demand variance

Because $D_t$ and $D_{t-n}$ are independent under the stated demand model,

$$
\operatorname{Cov}(D_t,D_{t-n})=0.
$$

Therefore,

$$
\operatorname{Var}(O_t)
=
\left(
1+\frac{L}{n}
\right)^2\sigma^2
+
\left(
\frac{L}{n}
\right)^2\sigma^2.
$$

Dividing by demand variance,

$$
B
=
\frac{
\operatorname{Var}(O_t)
}{
\operatorname{Var}(D_t)
}
$$

gives the bullwhip ratio

$$
B
=
\left(
1+\frac{L}{n}
\right)^2
+
\left(
\frac{L}{n}
\right)^2.
$$

Expanding,

$$
B
=
1
+
\frac{2L}{n}
+
2\left(
\frac{L}{n}
\right)^2.
$$

Whenever

$$
L>0,
$$

the ratio exceeds one.

The replenishment rule amplifies variance even though demand is independent, stationary, and unbiased.

No irrational manager is required.

No promotion is required.

No supplier shortage is required.

The control rule is enough.

## A simple numerical example produces 2.5 times the variance

Suppose customer demand has

$$
\mu=100
$$

and

$$
\sigma=10.
$$

Use an eight period moving average,

$$
n=8,
$$

with lead time

$$
L=4.
$$

The bullwhip ratio is

$$
B
=
1
+
\frac{2(4)}{8}
+
2\left(
\frac{4}{8}
\right)^2.
$$

Thus,

$$
B
=
1+1+0.5
=
2.5.
$$

Customer demand variance is

$$
100.
$$

Order variance is

$$
250.
$$

Customer demand standard deviation is

$$
10.
$$

Order standard deviation is

$$
\sqrt{250}
\approx
15.81.
$$

The average order remains

$$
100.
$$

The coefficient of variation therefore rises from

$$
\frac{10}{100}
=
0.10
$$

to approximately

$$
\frac{15.81}{100}
=
0.158.
$$

Nothing about the consumer market became more unstable.

The replenishment policy transformed the variability.

## Lead time increases amplification directly

Keep the moving average window fixed at

$$
n=8.
$$

If lead time is only one period,

$$
L=1,
$$

then

$$
B
=
1
+
\frac{2}{8}
+
2\left(
\frac{1}{8}
\right)^2
\approx
1.281.
$$

If

$$
L=4,
$$

then

$$
B=2.5.
$$

If

$$
L=8,
$$

then

$$
B
=
1+2+2
=
5.
$$

The relationship is nonlinear because the final term contains

$$
L^2.
$$

Long lead time does more than delay replenishment.

It magnifies the effect of every forecast revision on the desired pipeline.

This connects the bullwhip problem directly to the previous supply chain problem of stochastic lead time.

Longer and less predictable replenishment makes inventory harder to control.

Long lead time also makes forecast adjustments more consequential.

## Forecast smoothing reduces variance amplification

Now keep

$$
L=4
$$

but change the moving average window.

For

$$
n=4,
$$

the ratio is

$$
B
=
1+2+2
=
5.
$$

For

$$
n=8,
$$

we obtained

$$
B=2.5.
$$

For

$$
n=20,
$$

$$
B
=
1
+
\frac{8}{20}
+
2\left(
\frac{4}{20}
\right)^2.
$$

Therefore,

$$
B
=
1+0.4+0.08
=
1.48.
$$

A smoother forecast produces less order amplification.

That does not mean a longer averaging window is automatically better.

A smoother forecast responds more slowly when true demand changes.

The tradeoff is between responsiveness and variance amplification.

## A permanent demand increase produces temporary overordering

Suppose demand has been stable at

$$
100
$$

and then permanently increases to

$$
120.
$$

Use the previous parameters,

$$
L=4
$$

and

$$
n=8.
$$

Each new observation of 120 replaces one old observation of 100 in the moving average.

The forecast therefore increases by

$$
\frac{120-100}{8}
=
2.5
$$

units each period during the transition.

The desired pipeline changes by

$$
L(2.5)
=
10.
$$

Current demand is already

$$
120.
$$

The order becomes

$$
O_t
=
120+10
=
130
$$

during each period in which the moving average is still adjusting.

Once all eight historical observations have been replaced, the forecast reaches 120 and stops changing.

Orders then return to

$$
120.
$$

The retailer therefore sends an order stream of roughly 130 during the adjustment even though final customer demand has already stabilised at 120.

The extra ten units are not fake demand.

They are pipeline correction.

To an upstream stage that sees orders rather than consumer sales, they can look like continued demand growth.

## The upstream stage sees the control policy, not only the market

This is one of the most important distinctions in supply chain data.

An upstream supplier often observes

$$
O_t,
$$

not

$$
D_t.
$$

The supplier's observed demand therefore contains:

$$
\text{consumer demand}
+
\text{forecast revision}
+
\text{inventory correction}
+
\text{pipeline correction}
+
\text{batching}
+
\text{commercial behaviour}.
$$

A forecasting model trained on upstream orders is not modelling pure market demand.

It is modelling a market filtered through downstream policy.

That distinction becomes critical when a company tries to infer end customer behaviour from purchase orders.

## Bullwhip is measurable as a variance ratio

A simple empirical measure is

$$
B
=
\frac{
\operatorname{Var}(\text{orders})
}{
\operatorname{Var}(\text{downstream demand})
}.
$$

If

$$
B>1,
$$

order variance exceeds demand variance.

The ratio is useful but requires care.

Variance depends on time aggregation.

Weekly and daily measurements can produce different values.

Trends and seasonality can inflate variance if they are not treated consistently.

Demand and orders can have different means if inventories are being deliberately built or depleted.

Structural breaks can dominate the statistic.

A bullwhip estimate should therefore be interpreted together with the time scale and the operating regime.

## Variance amplification can exist with constant customer demand

Order batching provides an extreme illustration.

Suppose customer demand is perfectly constant:

$$
D_t=\mu.
$$

Customer demand variance is

$$
0.
$$

Now suppose a retailer orders only once every $k$ periods.

The order sequence is

$$
0,0,\ldots,0,k\mu
$$

over each $k$ period cycle.

The average order remains

$$
\mu.
$$

Its variance is positive.

In fact, over a complete batching cycle,

$$
\mathbb E[O^2]
=
\frac{1}{k}(k\mu)^2
=
k\mu^2.
$$

Therefore,

$$
\operatorname{Var}(O)
=
k\mu^2-\mu^2
=
(k-1)\mu^2.
$$

The classical variance ratio is not defined because demand variance is zero.

The operational lesson is still clear.

Order variability can be created entirely by ordering policy even when the underlying customer demand contains no variation.

## Batching can be economically rational

This example should not be interpreted as an argument that batching is irrational.

A company may face:

- fixed order costs
- full truck constraints
- pallet quantities
- minimum order quantities
- production setup costs
- container schedules
- administrative limits

Ordering every period can therefore be more expensive than consolidating demand.

The resulting upstream variance is partly the price of another operational objective.

Bullwhip analysis should therefore distinguish avoidable information distortion from deliberate economic batching.

The correct objective is not always to minimise order variance.

## Smoothing orders can increase inventory variance

An aggressive order up to policy corrects inventory position quickly.

That can create volatile orders.

A smoother replenishment rule can reduce order variance by correcting only part of the inventory gap each period.

Control theoretic analyses of supply chains make this tradeoff explicit.

Disney and Towill show that replenishment rules can be designed to reduce bullwhip by changing how rapidly inventory and pipeline discrepancies are corrected.

But smoothing has a cost.

If orders adjust slowly, inventory may deviate further from target after a demand change.

A supply chain therefore faces a control problem with competing objectives:

$$
\text{order smoothness}
$$

versus

$$
\text{inventory responsiveness}.
$$

Minimising bullwhip alone is not the complete operational objective.

## Price variation shifts demand through time

Suppose ordinary weekly customer consumption is stable.

A large temporary discount encourages buyers to purchase early.

Observed sales spike during the promotion and fall afterward.

If each supply chain stage forecasts independently from recent orders, the temporary shift can propagate upstream as a perceived demand regime change.

Lee, Padmanabhan, and Whang identified price variation as one of the classical sources of bullwhip for this reason.

The data science problem is that observed orders combine baseline consumption and intertemporal purchasing.

A model that does not represent promotion timing can turn a temporary transfer of demand across periods into an apparent change in demand level.

## Shortage gaming can inflate orders strategically

Suppose a supplier has limited capacity and allocates scarce inventory in proportion to customer orders.

A downstream buyer expecting rationing has an incentive to order more than it actually needs.

If the buyer needs 100 units but expects to receive only half of the requested quantity, it may order 200.

The supplier sees demand of 200.

When capacity later improves, the inflated orders disappear or are cancelled.

The supplier can misinterpret this reversal as a demand collapse.

This is not forecasting noise.

It is strategic behaviour generated by the allocation rule.

The ordering mechanism changed the data.

## Central demand information can reduce but not eliminate bullwhip

One obvious mitigation is to give upstream stages direct access to final customer demand.

Chen, Drezner, Ryan, and Simchi Levi analysed simple multi stage supply chains and showed that centralising demand information can reduce the bullwhip created by forecasting and lead times.

The qualification matters.

Information sharing does not automatically remove all amplification.

If each stage still uses a replenishment rule that reacts strongly to forecast changes or inventory gaps, policy induced variability can remain.

Information quality and control policy solve different parts of the problem.

## Point of sale data and order data answer different questions

Suppose a manufacturer receives both:

$$
\text{retailer sales}
$$

and

$$
\text{retailer orders}.
$$

The two series should not be treated as interchangeable features.

Retail sales are closer to final market demand.

Orders contain the retailer's response to that demand and to its own inventory state.

For short term production planning, orders may be operationally relevant because they determine what the manufacturer is expected to ship.

For market sensing, sales may be more informative.

A model should therefore state which process it is forecasting.

## Multi echelon systems create interacting feedback loops

Consider a chain with:

$$
\text{customer}
\rightarrow
\text{retailer}
\rightarrow
\text{distributor}
\rightarrow
\text{manufacturer}.
$$

The retailer observes customer demand and generates orders.

The distributor observes retailer orders and generates its own orders.

The manufacturer observes distributor orders.

Each stage therefore receives a signal that already contains downstream control behaviour.

It is tempting to multiply a one stage bullwhip ratio across stages.

That is generally too simple.

The first stage changes the autocorrelation structure of the order process.

The next stage therefore does not necessarily receive independent demand, which was an assumption in the simple derivation.

Exact multi echelon amplification depends on the full dynamic process.

This is why simulation, state space models, and control theoretic formulations are valuable for more realistic networks.

## Autocorrelation changes the variance formula

The simple derivation assumed

$$
\operatorname{Cov}(D_t,D_{t-n})=0.
$$

If demand is autocorrelated, then

$$
\operatorname{Var}(O_t)
=
\left(
1+\frac{L}{n}
\right)^2\sigma^2
+
\left(
\frac{L}{n}
\right)^2\sigma^2
-
2
\left(
1+\frac{L}{n}
\right)
\frac{L}{n}
\operatorname{Cov}(D_t,D_{t-n}).
$$

Positive autocorrelation at lag $n$ reduces this particular variance contribution.

Negative autocorrelation increases it.

The direction is not universal because the replenishment policy interacts with the demand dynamics.

A bullwhip formula should therefore never be applied without its assumptions.

## Forecast quality and bullwhip are related but different

A more accurate demand forecast can reduce unnecessary revisions.

That can reduce order variance.

But the relationship is not one to one.

A highly responsive forecast may track demand changes accurately while producing large period to period target revisions.

A smoother forecast may have larger short term prediction error while creating more stable orders.

The correct balance depends on:

- lead time
- service level
- capacity cost
- inventory cost
- production flexibility
- demand dynamics

This mirrors the earlier result that forecast accuracy and inventory performance are different objectives.

Bullwhip adds a third objective:

$$
\text{upstream variability}.
$$

## Capacity converts variance into cost

Suppose a plant has comfortable average capacity.

Mean demand is

$$
100.
$$

Plant capacity is

$$
125.
$$

If customer demand standard deviation is only 10, capacity exceedance is relatively rare.

If bullwhip raises order standard deviation to approximately

$$
15.8,
$$

the probability of an upstream order above 125 increases materially.

The mean demand did not change.

Capacity stress did.

Higher order variance can create:

- overtime
- subcontracting
- expedited transport
- unstable staffing
- queue growth
- missed production schedules
- larger safety stocks upstream

Variance therefore becomes an economic quantity.

## Capacity constraints can feed variance back downstream

Once upstream capacity is constrained, a second feedback loop appears.

High orders create backlog.

Backlog extends effective lead time.

Longer lead time encourages downstream stages to raise pipeline targets or safety stock.

Those larger targets can generate even larger orders.

The system can therefore create endogenous lead time:

$$
\text{large orders}
\rightarrow
\text{capacity congestion}
\rightarrow
\text{longer lead time}
\rightarrow
\text{larger pipeline target}
\rightarrow
\text{larger orders}.
$$

A model that assumes fixed lead time can miss this amplification entirely.

The bullwhip effect is therefore not only about information.

It can be a coupled demand, inventory, and capacity dynamic.

## Human adjustment can strengthen the feedback

Sterman's experimental work on dynamic decision making showed that people managing inventory systems can systematically misperceive feedback, particularly delays between actions and consequences.

This matters because supply chains contain exactly those delayed feedback structures.

A manager sees inventory fall.

The manager raises orders.

The replenishment is still in transit.

Inventory remains low.

The manager raises orders again.

When the delayed pipeline finally arrives, inventory overshoots.

Orders are then cut sharply.

The sequence can oscillate even when final demand is stable.

A forecasting system does not operate outside this behavioural context.

Manual overrides can become another feedback term.

## Forecast overrides should be evaluated for variance contribution

Suppose a statistical forecast is

$$
F_t.
$$

A planner applies an override

$$
A_t.
$$

The final forecast is

$$
F_t^{\mathrm{final}}
=
F_t+A_t.
$$

If replenishment responds to changes in the final forecast, volatile overrides can generate volatile orders even when they do not improve demand prediction.

Forecast Value Added should therefore be extended beyond accuracy.

For supply chain control, useful questions include:

$$
\operatorname{Var}(A_t),
$$

$$
\operatorname{Cov}(A_t,D_{t+1}-F_t),
$$

and the effect of overrides on

$$
\operatorname{Var}(O_t).
$$

An override that improves forecast error slightly but greatly increases upstream order variance may not improve the complete system.

## Bullwhip should be measured across several horizons

A single period variance ratio can miss important dynamics.

Suppose orders oscillate rapidly around demand.

Daily variance may be large.

Weekly aggregation can smooth the oscillation.

Another system may have slowly changing order waves that remain visible after aggregation.

Useful analysis can therefore examine:

$$
B_h
=
\frac{
\operatorname{Var}
\left(
\sum_{j=1}^{h}O_{t+j}
\right)
}{
\operatorname{Var}
\left(
\sum_{j=1}^{h}D_{t+j}
\right)
}
$$

for several horizons $h$.

This reveals whether amplification is primarily short term or persistent.

It also aligns the statistic with different capacity and planning horizons.

## Spectral analysis can reveal where amplification occurs

Variance collapses all frequencies into one number.

Supply chain oscillations often have structure across frequencies.

Let (f_D$\omega$) denote the spectral density of customer demand and (f_O$\omega$) the spectral density of orders.

A frequency specific amplification ratio is

$$
G(\omega)
=
\frac{
f_O(\omega)
}{
f_D(\omega)
}.
$$

If

$$
G(\omega)>1
$$

over a frequency band, the replenishment process amplifies variation at those frequencies.

This can distinguish high frequency order noise from slower inventory cycles.

Control theoretic approaches to bullwhip naturally use this perspective because a replenishment rule acts as a dynamic filter.

## Shared information should include inventory state, not only demand

Point of sale demand is valuable upstream.

It is not the whole state.

Suppose two retailers have identical customer demand.

Retailer A has full shelves and a healthy pipeline.

Retailer B has low on hand inventory and several delayed orders.

Their next purchase orders can differ substantially.

An upstream model that sees only consumer demand may therefore still fail to predict orders.

Useful shared state can include:

- customer demand
- inventory position
- outstanding orders
- expected receipts
- lead time
- promotions
- allocation rules
- planned policy changes

Information sharing reduces uncertainty when it reveals the variables that generate the order decision.

## Bullwhip can be reduced by changing the control law

Return to

$$
O_t
=
D_t
+
S_t-S_{t-1}.
$$

The full target correction

$$
S_t-S_{t-1}
$$

is not the only possible policy.

A damped rule can use only a fraction

$$
0<\gamma<1
$$

of the adjustment:

$$
O_t
=
D_t
+
\gamma(S_t-S_{t-1}).
$$

Smaller $\gamma$ reduces the immediate order reaction.

It also slows correction of inventory and pipeline errors.

Disney, Towill, and related control theoretic work analyse this type of tradeoff explicitly.

The right value is not determined by order variance alone.

It depends on the cost of production variability relative to the cost of inventory deviation and service failure.

## Bullwhip reduction is a multi objective problem

A supply chain can care about all of the following:

$$
\operatorname{Var}(O_t),
$$

$$
\operatorname{Var}(I_t),
$$

service level,

capacity utilisation,

backlog,

expediting cost,

holding cost,

and responsiveness to real demand shifts.

Reducing one can worsen another.

An optimisation problem can therefore take a form such as

$$
\min_\pi
\left[
\lambda_1
\operatorname{Var}(O_t)
+
\lambda_2
\operatorname{Var}(I_t)
+
\lambda_3
\mathbb E[\text{shortage}]
+
\lambda_4
\mathbb E[\text{capacity cost}]
\right].
$$

The replenishment policy

$$
\pi
$$

should be evaluated against the complete objective.

Bullwhip is one symptom of control performance.

It is not the only one.

## Data science should separate market variability from policy variability

Suppose upstream orders become more volatile.

Before concluding that the market has become unstable, decompose the order process.

Potential contributors include:

$$
\text{consumer demand variation}
$$

$$
+\text{forecast revision}
$$

$$
+\text{pipeline correction}
$$

$$
+\text{inventory correction}
$$

$$
+\text{batching}
$$

$$
+\text{promotion timing}
$$

$$
+\text{manual override}
$$

$$
+\text{allocation gaming}.
$$

Some components reflect external demand.

Others are generated internally.

This decomposition is important for root cause analysis.

A demand forecasting team should not be blamed for variability created by transport batching.

A production team should not interpret forecast correction as new consumer demand.

## The analytical example makes the mechanism visible

Under the simple model,

$$
F_t
=
\frac{1}{n}
\sum_{j=0}^{n-1}D_{t-j},
$$

and

$$
O_t
=
D_t+L(F_t-F_{t-1}).
$$

This becomes

$$
O_t
=
\left(
1+\frac{L}{n}
\right)D_t
-
\frac{L}{n}D_{t-n}.
$$

For independent demand,

$$
\frac{
\operatorname{Var}(O_t)
}{
\operatorname{Var}(D_t)
}
=
1
+
\frac{2L}{n}
+
2\left(
\frac{L}{n}
\right)^2.
$$

With

$$
L=4
$$

and

$$
n=8,
$$

the bullwhip ratio is

$$
2.5.
$$

The average order is still correct.

The variance is not inherited directly from the market.

It has been amplified by the replenishment rule.

That is the central data science lesson.

Supply chain observations are not passive measurements of demand.

They are often outputs of control systems.

To understand the data, we need to model the policy that generated them.

## References

Chen, F., Drezner, Z., Ryan, J. K., & Simchi-Levi, D. (2000). Quantifying the bullwhip effect in a simple supply chain: The impact of forecasting, lead times, and information. *Management Science*, 46(3), 436–443. https://doi.org/10.1287/mnsc.46.3.436.12069

Dejonckheere, J., Disney, S. M., Lambrecht, M. R., & Towill, D. R. (2003). Measuring and avoiding the bullwhip effect: A control theoretic approach. *European Journal of Operational Research*, 147(3), 567–590. https://doi.org/10.1016/S0377-2217(02)00369-7

Disney, S. M., & Towill, D. R. (2003). On the bullwhip and inventory variance produced by an ordering policy. *Omega*, 31(3), 157–167. https://doi.org/10.1016/S0305-0483(03)00028-8

Lee, H. L., Padmanabhan, V., & Whang, S. (1997). Information distortion in a supply chain: The bullwhip effect. *Management Science*, 43(4), 546–558. https://doi.org/10.1287/mnsc.43.4.546

Sterman, J. D. (1989). Modeling managerial behavior: Misperceptions of feedback in a dynamic decision making experiment. *Management Science*, 35(3), 321–339. https://doi.org/10.1287/mnsc.35.3.321
