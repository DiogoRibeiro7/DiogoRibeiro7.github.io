---
permalink: '/data-science/service_level_is_not_one_metric/'
title: 'Service Level Is Not One Metric'
date: '2025-01-16'
categories:
- Data Science
tags:
- Supply Chain
- Logistics
- Inventory
- Service Level
- OTIF
author_profile: false
classes: wide
seo_title: 'Service Level Is Not One Metric'
seo_description: 'Cycle service level, fill rate, ready rate and OTIF measure different failures. Two inventory systems can score identically on one service metric and very differently on another.'
seo_type: article
excerpt: >-
  "Service level" sounds like one number. In inventory and logistics it is a
  family of different probabilities and ratios, each weighting shortages,
  customers and time differently.
summary: >-
  This article separates cycle service level, unit fill rate, order fill rate,
  ready rate and OTIF. Two exact counterexamples show that systems can have the
  same fill rate and radically different stockout frequencies, or the same
  cycle service level and very different shortage severity. The discussion then
  connects service metrics to reorder points, order quantities, lost sales,
  promised delivery dates, multi echelon systems and decision oriented
  backtesting.
keywords:
- cycle service level
- fill rate
- OTIF
- inventory service
- stockout probability
- supply chain metrics
why_this_exists: >-
  Supply chain dashboards often report "service level" without defining the
  underlying event or denominator. That ambiguity can make inventory policies
  look equivalent when they expose customers and operations to very different
  shortage patterns.
evidence: >-
  Original exact counterexamples, analytical results for normal lead time demand,
  standard inventory control definitions and established operations management
  treatments of service measures.
methodology: >-
  Define service metrics from explicit random variables and denominators. Build
  pairs of shortage distributions that hold one metric constant while changing
  another. Extend the comparison to a continuous review inventory model and to
  order level delivery metrics.
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
Question: What does a reported supply chain "service level" actually measure?
Claim: Cycle service level, unit fill rate, order fill rate, ready rate and OTIF are different estimands. A policy can perform well on one and poorly on another because the metrics weight shortage frequency, shortage magnitude, customer orders and time differently.
Counterclaim: A single service metric can be sufficient when the operational failure mode is clear and the metric is explicitly defined. The problem is using the generic label "service level" as if the alternatives were interchangeable.
Evidence object: Two exact shortage distribution counterexamples, a normal lead time demand example, formal definitions of inventory and order level service measures and a multi echelon interpretation.
Failure case: Treating a higher value on one metric as universally better, comparing service percentages with different denominators, or reporting OTIF as though it were solely an inventory metric.
Reader payoff: Identify the event and denominator behind a service KPI, select a metric aligned with the business failure mode and design backtests that expose both shortage frequency and shortage severity.
Exclusions: Setting service targets for a real company without customer promises, shortage costs, demand data and replenishment constraints.
-->

"Service level" is one of the most common phrases in supply chain management and one of the least precise.

A planner may say that an item has 95% service. A warehouse dashboard may report 98%. A supplier scorecard may show 92%. A retailer may use the same phrase for shelf availability, while a logistics team means on time and in full delivery.

Those percentages are not comparable until the event and denominator are defined.

Inventory systems can count whether a stockout occurred, how many units were unavailable, how many customer orders were completely filled or how long inventory remained positive. Distribution systems can count whether orders arrived on time, in full, or both.

Each metric answers a legitimate question.

They are different questions.

This matters because a policy can look excellent under one service definition and poor under another. The difference is not cosmetic. It changes which replenishment policy is preferred.

## Cycle service level measures stockout frequency

Consider a replenishment cycle.

Let

$$
X
$$

denote shortage quantity during that cycle.

Cycle service level is

$$
CSL
=
P(X=0).
$$

Equivalently, if lead time demand is $D_L$ and the reorder point is $r$,

$$
CSL
=
P(D_L\le r).
$$

A cycle service level of 95% means that 95% of replenishment cycles are expected to complete without any shortage under the model.

It does not say how severe the remaining 5% of shortages are.

A shortage of one unit and a shortage of one thousand units both count as a failed cycle.

That is the defining property of the metric.

## Fill rate measures shortage magnitude

Unit fill rate asks a different question.

A common definition is the fraction of demanded units supplied immediately from stock.

For a continuous review ((Q,r)) system with expected shortage per replenishment cycle

$$
\mathbb E[X],
$$

a standard approximation is

$$
\beta
=
1-
\frac{
\mathbb E[X]
}{
Q
},
$$

where $Q$ is the replenishment quantity.

The numerator is expected shortage quantity.

Cycle service depends on

$$
P(X>0).
$$

Fill rate depends on

$$
\mathbb E[X].
$$

One counts how often shortage occurs.

The other weights shortage by magnitude.

Those quantities can move independently.

## The same fill rate can hide radically different stockout frequencies

Suppose replenishment quantity is

$$
Q=100.
$$

Consider Policy A.

Its shortage distribution is

$$
X_A
=
\begin{cases}
0, & p=0.95,\\
100, & p=0.05.
\end{cases}
$$

Expected shortage is

$$
\mathbb E[X_A]
=
0.95(0)+0.05(100)
=
5.
$$

Its unit fill rate is therefore

$$
\beta_A
=
1-\frac{5}{100}
=
0.95.
$$

Its cycle service level is

$$
CSL_A
=
P(X_A=0)
=
0.95.
$$

Now consider Policy B.

Its shortage distribution is

$$
X_B
=
\begin{cases}
0, & p=0.50,\\
10, & p=0.50.
\end{cases}
$$

Expected shortage is again

$$
\mathbb E[X_B]
=
0.50(0)+0.50(10)
=
5.
$$

So

$$
\beta_B
=
0.95.
$$

The fill rates are identical.

The cycle service levels are not.

For Policy B,

$$
CSL_B
=
0.50.
$$

One system stocks out in only 5% of cycles but occasionally fails badly.

The other stocks out in half of all cycles but each shortage is relatively small.

Both report a 95% unit fill rate.

A dashboard showing only fill rate cannot distinguish them.

## The same cycle service level can hide very different shortage severity

Now hold cycle service fixed.

Again let

$$
Q=100.
$$

Consider Policy C:

$$
X_C
=
\begin{cases}
0, & p=0.95,\\
1, & p=0.05.
\end{cases}
$$

Then

$$
CSL_C=0.95.
$$

Expected shortage is

$$
\mathbb E[X_C]
=
0.05.
$$

The fill rate is

$$
\beta_C
=
1-\frac{0.05}{100}
=
0.9995.
$$

Now consider Policy D:

$$
X_D
=
\begin{cases}
0, & p=0.95,\\
50, & p=0.05.
\end{cases}
$$

Its cycle service level is also

$$
0.95.
$$

Expected shortage is

$$
\mathbb E[X_D]
=
2.5.
$$

The fill rate is

$$
\beta_D
=
1-\frac{2.5}{100}
=
0.975.
$$

Both systems can advertise 95% cycle service.

One fills 99.95% of units.

The other fills 97.5%.

The difference exists because cycle service throws away shortage magnitude.

## Neither metric is wrong

The counterexamples do not show that cycle service is inferior to fill rate.

They show that the metrics define different failures.

Suppose a manufacturing line stops if one component is missing.

A one unit shortage can be nearly as disruptive as a fifty unit shortage.

Then the event

$$
X>0
$$

matters directly.

Cycle service is relevant.

Now suppose a retailer can tolerate occasional small backorders but cares strongly about the total volume of unfilled customer demand.

Then

$$
\mathbb E[X]
$$

can be more informative.

Fill rate becomes more aligned with the operational objective.

The correct service metric follows from the consequence of failure.

## A normal inventory model makes the distinction quantitative

Suppose lead time demand is approximately normal:

$$
D_L
\sim
\mathcal N(\mu_L,\sigma_L^2).
$$

Let

$$
\mu_L=500
$$

and

$$
\sigma_L=100.
$$

For a 95% cycle service target, choose

$$
r
=
\mu_L+z_{0.95}\sigma_L.
$$

Using

$$
z_{0.95}\approx1.645,
$$

the reorder point is

$$
r
=
500+1.645(100)
=
664.5.
$$

By construction,

$$
CSL
\approx
0.95.
$$

For normal lead time demand, expected shortage above $r$ is

$$
\mathbb E[(D_L-r)^+]
=
\sigma_L
\left[
\phi(z)
-
z(1-\Phi(z))
\right].
$$

At

$$
z=1.645,
$$

the standard normal loss function is approximately

$$
0.0209.
$$

Expected shortage is therefore approximately

$$
2.09
$$

units per replenishment cycle.

Now consider two order quantities.

If

$$
Q=20,
$$

then

$$
\beta
=
1-\frac{2.09}{20}
\approx
0.8955.
$$

If

$$
Q=200,
$$

then

$$
\beta
=
1-\frac{2.09}{200}
\approx
0.9896.
$$

The reorder point is identical.

The cycle service level is identical.

The fill rate differs because the same expected shortage is divided by a different replenishment quantity.

This is one reason service targets cannot be specified independently of inventory policy.

## Order fill rate introduces the customer order as the unit

Unit fill rate weights units.

Customer service can also be measured at the order level.

Let customer order $j$ contain requested quantity $Y_j$.

Define

$$
F_j
=
\begin{cases}
1, & \text{order }j\text{ is completely filled immediately},\\
0, & \text{otherwise}.
\end{cases}
$$

Order fill rate is

$$
OFR
=
\frac{
\sum_j F_j
}{
N
}.
$$

This treats every customer order equally.

A one unit order and a one thousand unit order each contribute one observation.

Unit fill rate does the opposite.

The larger order contributes much more weight because it contains more units.

The two metrics can therefore disagree even with the same underlying fulfilment data.

## A simple order example shows the denominator effect

Suppose there are ten customer orders.

Nine orders request one unit each.

One order requests ninety one units.

Total demand is

$$
100.
$$

Assume all nine small orders are filled completely.

The large order receives only 81 of its 91 units.

Units filled are

$$
9+81
=
90.
$$

Unit fill rate is

$$
90%.
$$

Only nine of ten orders are complete.

Order fill rate is also

$$
90%.
$$

Now change the pattern.

The large order is filled completely.

Only one of the nine small orders fails.

Suppose that failed order receives zero units.

Units filled are

$$
91+8
=
99.
$$

Unit fill rate is

$$
99%.
$$

Order fill rate remains

$$
90%.
$$

One metric says performance improved from 90% to 99%.

The other says nothing changed.

Both statements are correct because the denominators are different.

## Line fill rate creates another level of aggregation

A customer order may contain several product lines.

One order with ten lines can be partially fulfilled even when nine lines are available.

A line fill metric can define

$$
LFR
=
\frac{
\text{order lines completely filled}
}{
\text{total order lines}
}.
$$

This can differ from both unit fill rate and order fill rate.

A missing low volume line can fail the whole order while affecting very few units.

In B2B supply chains, that distinction can matter greatly because a customer may need the complete bill of materials rather than most individual units.

The operational failure is set completion, not unit count.

## Ready rate measures service across time

Another inventory service concept is the fraction of time the item is immediately available.

Let $I_t$ be on hand inventory.

A time based ready rate can be written as

$$
RR
=
P(I_t>0)
$$

or estimated empirically as

$$
\widehat{RR}
=
\frac{
\text{time with positive on hand inventory}
}{
\text{total observed time}
}.
$$

This metric weights time.

Cycle service weights replenishment cycles.

Fill rate weights demand units.

Order fill rate weights customer orders.

A system can look different depending on which dimension is used as the denominator.

## Ready rate can disagree with fill rate when demand is concentrated

Suppose an item is in stock 99% of the time.

That sounds excellent.

But suppose the 1% stockout period occurs during the year's largest promotion.

A large fraction of annual demand may be lost during that short interval.

Ready rate remains close to 99%.

Unit fill rate can be much lower.

Conversely, a low volume item can spend substantial time out of stock while affecting very few demand units.

Time availability and demand weighted availability are different quantities.

## OTIF is not an inventory metric

On time in full extends beyond inventory.

Define

$$
T_j=1
$$

if customer order $j$ arrives by the promised date.

Define

$$
F_j=1
$$

if it arrives in full.

Then

$$
OTIF
=
P(T_j=1,F_j=1).
$$

Empirically,

$$
\widehat{OTIF}
=
\frac{
\text{orders delivered on time and in full}
}{
\text{total eligible orders}
}.
$$

An order can be in full but late.

It can be on time but incomplete.

It can fail both dimensions.

Inventory availability influences the in full component.

Warehouse processing, transportation, carrier reliability, appointment scheduling and promise date logic influence the on time component.

Treating OTIF as a pure inventory KPI assigns failures to the wrong part of the system.

## On time and in full are not generally independent

It can be tempting to approximate

$$
P(T=1,F=1)
$$

as

$$
P(T=1)P(F=1).
$$

That requires independence.

In many logistics systems, timeliness and completeness are related.

A warehouse may hold an order until all lines become available.

That can improve in full performance while worsening timeliness.

Alternatively, a company may ship partial orders immediately.

That can improve on time performance while reducing in full performance.

The fulfilment policy creates dependence between the metrics.

The joint probability should therefore be measured directly when possible.

## Promise dates can manipulate OTIF without changing physical performance

Suppose average physical delivery time is four days.

A company promises delivery in five days.

Most orders are on time.

Now change the promise to seven days without changing operations.

OTIF can improve.

The logistics process did not become faster.

The service contract became less demanding.

This does not make OTIF invalid.

It means the promise definition is part of the metric.

Comparing OTIF across customers, regions or periods requires comparable promise logic.

A 98% OTIF against a ten day promise is not automatically superior to 95% against a three day promise.

## Backorder probability and backorder duration answer still different questions

Suppose stockouts create backorders.

One metric is

$$
P(B_t>0),
$$

the probability that backlog exists.

Another is expected backlog quantity,

$$
\mathbb E[B_t].
$$

A third is customer waiting time,

$$
\mathbb E[W].
$$

These quantities can move differently.

A system with frequent tiny backlogs may have high backorder probability and low expected waiting time.

A system with rare but severe supply interruptions may have low backorder frequency and very high waiting time when a failure occurs.

Again, the appropriate metric follows from the operational consequence.

## Lost sales make fill rate harder to estimate

If unmet demand is backordered, requested demand is often recorded.

If unmet demand is lost, the denominator can become partially unobserved.

Observed sales are

$$
S_t
=
\min(D_t,I_t).
$$

If true demand $D_t$ exceeds inventory, transaction data alone do not reveal the missing units.

A naive fill rate calculated as

$$
\frac{
\text{sales}
}{
\text{recorded demand}
}
$$

can therefore be biased upward when recorded demand excludes abandoned customers.

This connects service measurement directly to the censored demand problem.

A service KPI can look excellent because the failures removed themselves from the denominator.

## Customer substitution complicates service measurement further

Suppose product $A$ is unavailable.

A customer buys substitute $B$.

From a category perspective, demand may have been served.

From product $A$'s perspective, the requested SKU was not available.

From a revenue perspective, the outcome depends on the substitute price and margin.

From a customer preference perspective, the substitution may still be a service failure.

No single fill metric answers all four questions.

The metric must define what counts as fulfilled demand.

## Multi echelon systems separate local and customer service

Suppose a regional warehouse achieves high local fill rate to stores.

Stores still stock out because orders arrive too late or because the allocation policy sends inventory to the wrong locations.

The warehouse metric can be strong while final customer service is poor.

Conversely, stores can maintain high shelf availability by carrying large local buffers even when upstream service is unstable.

A multi echelon supply chain therefore has service metrics at several interfaces:

$$
\text{supplier}\rightarrow\text{DC},
$$

$$
\text{DC}\rightarrow\text{store},
$$

$$
\text{store}\rightarrow\text{customer}.
$$

High performance at one interface does not mathematically imply high end to end service.

Local KPIs can move inventory problems rather than solve them.

## Service metrics can create conflicting incentives

Suppose a warehouse is evaluated only on order fill rate.

Holding partially complete orders until missing items arrive can improve the chance that the order leaves in full.

It may worsen on time performance.

If the transportation team is measured only on on time delivery after dispatch, the waiting time inside the warehouse may disappear from its KPI.

Each local metric can be improved while customer lead time worsens.

This is a metric design problem rather than a forecasting problem.

The system optimises what each team is measured against.

## Inventory targets should state the service estimand explicitly

A statement such as

$$
\text{target service}=95%
$$

is incomplete.

A useful target looks more like:

$$
CSL=95%
$$

for a particular item class,

or

$$
\beta=98%
$$

for units demanded,

or

$$
OTIF=96%
$$

for customer orders under a specified promise definition.

The metric should also define:

- eligible demand
- treatment of cancellations
- treatment of substitutions
- partial shipments
- backorders
- measurement horizon
- customer promise rule

Without those definitions, two dashboards can report the same percentage and measure different operational events.

## Service should be evaluated jointly with inventory cost

A policy that raises safety stock generally raises service.

If only service is measured, the trivial solution is often to hold more inventory.

Inventory policy is a tradeoff.

A simple objective can be written as

$$
C(\pi)
=
c_h
\mathbb E[I]
+
c_b
\mathbb E[B]
+
c_o
\mathbb E[N_o]
+
c_e
\mathbb E[E],
$$

subject to a service constraint such as

$$
\beta(\pi)\ge0.98.
$$

Here $\pi$ represents the inventory policy.

The service metric appears as a constraint because it expresses the business requirement.

The cost function determines how the system meets it.

This is more informative than ranking policies by service alone.

## A policy can dominate on one service metric and lose on another

Suppose Policy P produces

$$
CSL=98%
$$

and

$$
\beta=96%.
$$

Policy Q produces

$$
CSL=90%
$$

and

$$
\beta=99.5%.
$$

Which is better?

The numbers alone cannot decide.

If the business needs almost every cycle to remain disruption free, P may align better.

If the business mainly cares about the proportion of units immediately supplied, Q may align better.

The question is not which service metric is mathematically superior.

The question is which operational failure is expensive.

## Service targets should be stress tested across the demand distribution

A nominal service level is model based.

Suppose a reorder point is designed for

$$
CSL=95%
$$

under a normal lead time demand model.

If actual demand has a heavier right tail, empirical service may be lower.

Backtesting should therefore estimate realised service:

$$
\widehat{CSL}
=
\frac{
\text{cycles without stockout}
}{
\text{completed cycles}
}
$$

and realised unit fill rate:

$$
\widehat{\beta}
=
\frac{
\text{units filled immediately}
}{
\text{units requested}
}.
$$

The model target and realised outcome should be shown together.

A 95% theoretical service target that consistently delivers 87% empirically is a calibration failure.

## Service needs confidence intervals too

A reported service percentage is estimated from finite data.

Suppose 95 of 100 cycles have no stockout.

The estimated cycle service level is

$$
0.95.
$$

Suppose instead 9,500 of 10,000 cycles succeed.

The estimate is also

$$
0.95.
$$

The second estimate is much more precise.

Operational dashboards often display both simply as 95%.

For a binomial approximation,

$$
SE(\hat p)
=
\sqrt{
\frac{
\hat p(1-\hat p)
}{
n
}
}.
$$

For (n=100),

$$
SE
\approx
0.0218.
$$

For (n=10000),

$$
SE
\approx
0.00218.
$$

The uncertainty differs by a factor of ten.

Service KPIs are statistical estimates and should be treated as such.

## Aggregated service can hide critical segments

Suppose a network achieves 98% unit fill rate overall.

High volume commodity items achieve 99.5%.

Critical spare parts achieve 82%.

The aggregate can remain high because large demand volumes dominate the denominator.

If the operational cost of a spare part shortage is high, the aggregate service metric hides the problem.

Segmented reporting can therefore be necessary by:

- SKU criticality
- customer class
- location
- supplier
- product family
- demand regime

The segmentation should follow operational consequences rather than arbitrary dashboard convenience.

## Averages across service metrics are usually meaningless

Suppose a dashboard reports:

$$
CSL=95%,
$$

$$
\beta=98%,
$$

and

$$
OTIF=92%.
$$

Taking their arithmetic mean,

$$
95%,
$$

creates a number with no clear probabilistic interpretation.

The metrics have different denominators and different events.

They are not repeated measurements of one latent "service" quantity unless a formal measurement model justifies such aggregation.

A composite score can be designed.

Its weights then become business assumptions.

Those assumptions should be explicit.

## The two counterexamples contain the central lesson

The first pair of policies had the same fill rate:

$$
\beta_A
=
\beta_B
=
95%.
$$

Their cycle service levels were

$$
95%
$$

and

$$
50%.
$$

The second pair had the same cycle service level:

$$
CSL_C
=
CSL_D
=
95%.
$$

Their fill rates were approximately

$$
99.95%
$$

and

$$
97.5%.
$$

The metrics disagree because they count different things.

Cycle service counts shortage events.

Fill rate counts shortage units.

Order fill rate counts complete customer orders.

Ready rate counts time.

OTIF counts orders that satisfy both completeness and timing.

None of those denominators is interchangeable.

"Service level" becomes scientifically and operationally useful only after the event and denominator are specified.

## References

Axsäter, S. (2015). *Inventory Control* (3rd ed.). Springer. https://doi.org/10.1007/978-3-319-15729-0

Hadley, G., & Whitin, T. M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.

Tempelmeier, H. (2011). Inventory management in supply networks: Problems, models, solutions. Books on Demand.

Zipkin, P. H. (2000). *Foundations of Inventory Management*. McGraw-Hill.
