---
author_profile: false
categories:
- Mathematics
classes: wide
title: 'Queueing Systems Fail Before Average Load Reaches Capacity'
excerpt: 'Utilization near one creates nonlinear growth in waiting time and queue length. Average capacity can therefore look adequate while service quality collapses.'
keywords:
- queueing theory
- Little's law
- M/M/1
- heavy traffic
- service systems
seo_title: 'Queueing Systems Fail Before Average Load Reaches Capacity'
seo_description: 'A mathematical draft on Little’s law, M/M/1 queues, finite capacity, variability, heavy traffic, and operational service levels.'
seo_type: article
summary: 'A planned article deriving basic queueing results and showing why utilization, variability and service-time distributions matter more than average throughput alone.'
tags:
- Queueing Theory
- Operations Research
- Stochastic Processes
- Service Systems
why_this_exists: 'Operational systems are often planned using average demand and average capacity, even though waiting time becomes highly nonlinear as utilization approaches one.'
evidence: 'Exact M/M/1 calculations, finite-capacity examples, simulation under non-exponential service times and heavy-traffic comparisons.'
methodology: 'Derive Little’s law and M/M/1 quantities, then hold mean arrival and service rates fixed while changing variability and capacity constraints.'
---

<!--
Development contract
Question: Why can a system with average service capacity above average demand still deliver terrible waiting times?
Claim: Queue performance depends on utilization and variability, and delay diverges nonlinearly as utilization approaches one even before average demand exceeds average capacity.
Counterclaim: Markovian queue formulas can be badly misleading when arrivals, service times, priorities or abandonment differ from their assumptions.
Evidence object: Exact M/M/1 calculations across utilization, M/G/1 comparison via Pollaczek-Khinchine, and a finite-capacity or abandonment example.
Failure case: Planning from utilization alone, using Little’s law as a causal formula, or assuming exponential service times without checking variability.
Reader payoff: Understand which queueing quantities govern waiting and when simulation or richer queue models are needed.
Exclusions: A full taxonomy of queueing networks.
-->

## Mathematical spine

For M/M/1 with arrival rate $\lambda$, service rate $\mu$, and

$$
\rho=\frac{\lambda}{\mu}<1,
$$

derive

$$
L=\frac{\rho}{1-\rho},
$$

$$
W=\frac{1}{\mu-\lambda},
$$

and

$$
L=\lambda W.
$$

Show numerically how moving from $\rho=0.8$ to $\rho=0.95$ changes waiting by far more than the 15 percentage-point utilization increase suggests.

Then introduce M/G/1 through Pollaczek-Khinchine,

$$
W_q
=
\frac{\lambda E[S^2]}
{2(1-\rho)},
$$

making service-time variability explicit.

## Worked examples

Compare two systems with identical mean service time but different variance. Add one finite-capacity example or Erlang-A abandonment example to show how observed throughput can hide blocked or abandoned demand.

## Reproducibility plan

Simulate M/M/1 and M/G/1 systems and compare empirical wait distributions with theoretical means, including heavy-tail service-time stress tests.

## Sources to develop

Kleinrock, L. (1975). *Queueing Systems, Volume 1*.

Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2008). *Fundamentals of Queueing Theory*.

Whitt, W. (2002). *Stochastic-Process Limits*.
