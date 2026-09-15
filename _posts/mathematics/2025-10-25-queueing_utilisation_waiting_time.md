---
permalink: '/mathematics/queueing_utilisation_waiting_time/'
title: 'Queueing: Why 90 Percent Utilisation Means Waiting'
categories:
- Mathematics
tags:
- Mathematical Modeling
- Stochastic Processes
- Optimization
- Monte Carlo
author_profile: false
seo_title: 'Queueing Theory: Utilisation and Waiting Time'
seo_description: 'Waiting time is not proportional to utilisation. It is flat until 70 percent and vertical after 90, doubles with service variability, drops five-fold with pooling, and lingers for hours after an overload. The formulas, checked by simulation.'
excerpt: >-
  A maintenance crew is busy 85 percent of the time and the planner wants
  95, because idle technicians are waste. The queue has other ideas. The
  last ten points of utilisation cost more waiting than the first eighty.
summary: >-
  Little's law, the utilisation curve of a single-server queue checked by
  simulation, why the curve is flat below 70 percent and vertical above 90,
  Kingman's formula for what service-time variability does to waiting, the
  five-fold gain from pooling four crews into one queue, how a single
  overloaded hour leaves a backlog that drains for four, why percentiles
  rather than means belong in a service target, and how to set a
  utilisation target from the wait the operation can afford.
keywords:
  - queueing theory
  - utilisation
  - waiting time
  - Little's law
  - Kingman's formula
  - pooling
  - capacity planning
classes: wide
date: '2025-10-25'
why_this_exists: >-
  Utilisation targets are set as if waiting were proportional to load, and
  the shape of the real relationship surprises every operation that runs
  into it. This post gives the formulas that describe it, checks each one by
  discrete-event simulation, and draws the capacity decisions that follow.
evidence: >-
  Simulated single-server and four-server queues with 200,000 jobs each,
  exponential arrivals, and deterministic, exponential and high-variability
  service times, plus a simulated day with one hour at 130 percent load.
methodology: >-
  Computes waiting times with the Lindley recursion and a multi-server
  first-come-first-served simulation, compares them with the M/M/1 formula,
  Kingman's approximation and the Erlang C formula, checks Little's law, and
  traces the backlog after a temporary overload hour by hour.
reviewed_at: '2026-09-10'
header:
  image: /assets/images/headers/photo-code.jpg
  og_image: /assets/images/headers/photo-code.jpg
  overlay_image: /assets/images/headers/photo-code.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-code.jpg
  twitter_image: /assets/images/headers/photo-code.jpg
---
A maintenance crew is busy 85 percent of the time, and the planner wants 95, because idle technicians are waste. The ticket queue has other ideas. Every operation that serves arriving work meets the same fact eventually: waiting time is not proportional to load. It is nearly flat until utilisation reaches about 70 percent, bends sharply through 80, and goes vertical past 90. The formulas that describe this are a century old, short, and checked here by simulation, and the capacity decisions they imply are the opposite of the planner's instinct.

## Little's Law

The one result that holds for any queue, whatever its arrival pattern or service rule, is Little's law: the average number of jobs in a system equals the arrival rate times the average time each spends there,

$$
L = \lambda W .
$$

It is a conservation statement, not a model, and it lets one quantity be inferred from the other two. A repair shop that receives 40 jobs a day and holds 120 in progress has a lead time of three days, whatever anyone's tracking system says. Applied to the waiting line alone, it turns a mean wait into a mean backlog and back.

## The Utilisation Curve

The simplest queue has one server, jobs arriving at random at rate $\lambda$, and service times that are random with mean $\tau$. Utilisation is $\rho = \lambda\tau$, the fraction of time the server is busy. When service times are exponentially distributed, the mean time a job waits before service starts is

$$
W_q = \frac{\rho}{1 - \rho}\,\tau .
$$

At 50 percent utilisation a job waits one service time on average. At 80 percent, four. At 90, nine. At 95, nineteen. The denominator is what does the damage: each step toward full utilisation removes a slice of the slack that absorbed randomness, and the queue grows without bound as $\rho$ approaches one.

The simulation uses the Lindley recursion, which computes each job's wait from the previous job's wait, service time and the gap between arrivals, and applies to any arrival and service distributions.

```python
import numpy as np
from math import factorial

def gg1_wait(arrivals, services):
    """Lindley recursion: waiting time in queue for a single FIFO server."""
    w = np.empty(len(services)); w[0] = 0.0
    for i in range(1, len(services)):
        w[i] = max(0.0, w[i-1] + services[i-1] - arrivals[i])
    return w

def ggc_wait(arrivals, services, c):
    """FIFO with c identical servers: each job takes the server that frees up first."""
    free = np.zeros(c); t = 0.0; w = np.empty(len(services))
    for i in range(len(services)):
        t += arrivals[i]
        k = np.argmin(free)
        start = max(t, free[k]); w[i] = start - t; free[k] = start + services[i]
    return w

rng = np.random.default_rng(0)
N = 200000
print("utilisation   simulated   formula rho/(1-rho)   95th percentile wait")
for rho in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    a = rng.exponential(1 / rho, N); s = rng.exponential(1.0, N)
    w = gg1_wait(a, s)[N // 10:]                   # drop the warm-up
    print(f"{rho:<13}{w.mean():>9.2f}{rho/(1-rho):>18.2f}{np.percentile(w, 95):>16.1f}")

rho = 0.8
a = rng.exponential(1 / rho, N); s = rng.exponential(1.0, N)
w = gg1_wait(a, s)
lam = 1 / a.mean()
print(f"Little's law at rho = 0.8: {lam:.3f} x {w.mean():.2f} = {lam * w.mean():.2f} jobs waiting on average")
```

| Utilisation | Mean wait, simulated | Mean wait, formula | 95th percentile wait |
| --- | --- | --- | --- |
| 0.50 | 1.02 | 1.00 | 4.7 |
| 0.70 | 2.30 | 2.33 | 8.6 |
| 0.80 | 4.09 | 4.00 | 14.3 |
| 0.90 | 9.15 | 9.00 | 30.0 |
| 0.95 | 19.96 | 19.00 | 63.1 |
| 0.99 | 66.85 | 99.00 | 208.5 |

All waits are in multiples of the mean service time. The simulation and the formula agree to within noise up to 95 percent. At 99 percent they do not, and the reason is itself a lesson: 200,000 jobs are not enough for a queue that close to saturation to settle into its long-run behaviour. A queue's memory is as long as its wait, and a real operation at 99 percent utilisation is never in steady state within a shift. It is always still recovering from the last surge.

Little's law checks out at 80 percent: an arrival rate of 0.802 times a mean wait of 3.98 gives 3.19 jobs waiting on average.

![Left: mean time waiting in queue, in units of the mean service time, against server utilisation for three levels of service-time variability, with simulated points for exponential service. The curves are flat below 70 percent and vertical above 90. Right: waiting times through a day in which one hour runs at 130 percent load and every other hour at 80; the backlog takes four hours to drain.](/assets/images/figures/queue_wait_vs_utilisation.png){: width="1664" height="640" loading="lazy"}

## Variability Is the Other Half

Utilisation is not the only input. Two queues at the same utilisation can have waits that differ by a factor of five, and the difference is variability. Kingman's approximation for a single server makes the dependence explicit:

$$
W_q \approx \frac{\rho}{1 - \rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \tau ,
$$

where $c_a$ and $c_s$ are the coefficients of variation of the times between arrivals and of the service times. The exponential case has $c_a^2 = c_s^2 = 1$ and the second factor is one, recovering the earlier formula.

```python
print("rho = 0.8: service-time variability")
print("service distribution    c_s^2   simulated wait   Kingman approximation")
for name, sgen, cs2 in (("deterministic", lambda n: np.ones(n), 0.0),
                        ("exponential", lambda n: rng.exponential(1.0, n), 1.0),
                        ("lognormal, cv 2", lambda n: rng.lognormal(-0.5 * np.log(5), np.sqrt(np.log(5)), n), 4.0)):
    a = rng.exponential(1 / rho, N); s = sgen(N)
    w = gg1_wait(a, s)[N // 10:]
    print(f"{name:<23}{cs2:>5.1f}{w.mean():>17.2f}{rho/(1-rho) * (1 + cs2) / 2:>18.2f}")
```

| Service times at 80% utilisation | $c_s^2$ | Simulated mean wait | Kingman |
| --- | --- | --- | --- |
| Deterministic | 0 | 2.03 | 2.00 |
| Exponential | 1 | 4.10 | 4.00 |
| Lognormal, coefficient of variation 2 | 4 | 9.95 | 10.00 |

Same arrival process, same utilisation, same server. Making the service time perfectly regular halves the wait; making it highly variable, which is what a mix of five-minute resets and three-hour rebuilds looks like, multiplies it by two and a half. The arrival side behaves the same way: scheduled arrivals have $c_a^2$ near zero and walk-ins have it near one, which is why appointment systems exist.

The operational reading is that variability reduction and capacity are substitutes. A crew that standardises its work so that jobs take more predictable times gets the same reduction in waiting as a crew that adds capacity to bring utilisation from 80 percent down to about 67, and it gets it without hiring.

## Pooling

Four crews, each with its own queue, each 80 percent busy. Or one queue feeding all four. Same people, same work, same utilisation.

```python
def erlang_c(c, a):
    num = a**c / factorial(c) * c / (c - a)
    den = sum(a**k / factorial(k) for k in range(c)) + num
    return num / den

a = rng.exponential(1 / 0.8, N); s = rng.exponential(1.0, N)
print(f"four separate queues at rho = 0.8: mean wait {gg1_wait(a, s)[N//10:].mean():.2f}")
a4 = rng.exponential(1 / 3.2, N); s4 = rng.exponential(1.0, N)
w4 = ggc_wait(a4, s4, 4)[N//10:]
pw = erlang_c(4, 3.2)
print(f"one pooled queue, four servers: mean wait {w4.mean():.2f}  "
      f"(Erlang C: P(wait) = {pw:.2f}, mean wait {pw / (4 - 3.2):.2f})")
```

Separate queues: a mean wait of 3.95 service times. One pooled queue with four servers: 0.75, in agreement with the Erlang C formula, which puts the probability of waiting at all at 60 percent. That is a five-fold reduction from a change in routing alone.

The mechanism is that separate queues let a job wait at one crew while another crew stands idle. Pooling makes every idle server available to every waiting job, and the benefit grows with the number of servers pooled. The costs are real too: pooled servers need to be interchangeable, which fights specialisation; a shared queue may add travel or handoffs; and a pooled system fails all at once rather than one crew at a time. Where those costs are small, pooling is the cheapest capacity there is.

## Recovering From an Overload

The utilisation curve describes steady state. Real load is not steady, and the queue integrates its history. A single hour at 130 percent load, in a day otherwise at 80 percent:

```python
rng = np.random.default_rng(3)
hours, per_hour = 24, 60
rates = np.where(np.arange(hours) == 8, 1.3, 0.8)           # hour 8 runs at 130% load
a, s = [], []
for h in range(hours):
    n = rng.poisson(rates[h] * per_hour)
    a.append(np.sort(rng.uniform(h * per_hour, (h + 1) * per_hour, n)))
    s.append(rng.exponential(1.0, n))
arr = np.concatenate(a); srv = np.concatenate(s)
inter = np.diff(np.concatenate([[0.0], arr]))
w = gg1_wait(inter, srv)
for h in (7, 8, 9, 10, 11, 12, 14):
    m = (arr >= h * per_hour) & (arr < (h + 1) * per_hour)
    print(f"hour {h:>2}: load {rates[h]:.1f}, mean wait of arrivals that hour {w[m].mean():.1f} minutes")
```

| Hour | Load | Mean wait of jobs arriving that hour |
| --- | --- | --- |
| 7 | 0.8 | 1.6 min |
| 8 | 1.3 | 17.7 min |
| 9 | 0.8 | 18.7 min |
| 10 | 0.8 | 14.4 min |
| 11 | 0.8 | 8.8 min |
| 12 | 0.8 | 4.1 min |
| 14 | 0.8 | 2.1 min |

Jobs arriving in the overloaded hour wait 18 minutes. So do jobs arriving in the hour after it, when the load is back to normal, and the backlog is still there at noon. The drain rate is the slack: at 80 percent utilisation the server clears the backlog at only 20 percent of its capacity, so the surplus of about 18 minutes' work that the hour leaves behind takes an hour and a half to remove in expectation, and with the randomness on top it is closer to four hours before waits return to normal. An average utilisation of 82 percent over the day describes none of this. The queue remembers peaks, and capacity has to be planned for them.

## What the Averages Hide

The 95th percentile column in the first table is three times the mean at every utilisation. Waiting times are skewed: most jobs wait little and a few wait a long time, and the few are the ones that generate complaints, missed deadlines and escalations. A service target expressed as a mean is easy to meet while a fifth of jobs wait an unacceptable time. At 90 percent utilisation the mean wait is nine service times and the 95th percentile is thirty. Targets, and reports against them, belong on percentiles.

## Where It Applies

The setting was a maintenance crew, and the same arithmetic governs any resource that serves arriving work: inference servers, where 90 percent GPU utilisation is a latency problem rather than an efficiency achievement; support desks; hospital beds and operating theatres; loading docks; build and test pipelines; call centres, which is where Erlang derived the pooled-queue formula in 1917. In each case there is a utilisation the operation can afford, given the wait it can tolerate and the variability it has, and it is rarely above 85 percent for a single server.

## What to Do

1. **Measure three things, not one**: utilisation, the variability of arrivals and the variability of service, since the wait depends on all of them.
2. **Set the utilisation target from the affordable wait**, reading it off the curve for your variability, rather than from a headcount or efficiency rule.
3. **Reduce variability before adding capacity.** Standardised work and scheduled arrivals move the curve down; they are cheaper than servers.
4. **Pool queues** wherever the servers can be made interchangeable, and expect gains of several-fold at the same headcount.
5. **Plan capacity for peaks**, because a queue integrates overloads and a busy hour costs an afternoon.
6. **Report and target percentiles**, since the mean wait is met while the tail fails.
7. **Simulate** the specific system, with its own arrival and service distributions, when a decision rides on the number. The recursion is six lines.

## References

- Little, J. D. C. (1961). A proof for the queuing formula L = λW. *Operations Research*, 9(3), 383-387.
- Kingman, J. F. C. (1961). The single server queue in heavy traffic. *Mathematical Proceedings of the Cambridge Philosophical Society*, 57(4), 902-904.
- Erlang, A. K. (1917). Solution of some problems in the theory of probabilities of significance in automatic telephone exchanges. *Elektroteknikeren*, 13, 5-13.
- Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. Wiley.
- Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2008). *Fundamentals of Queueing Theory* (4th ed.). Wiley.
- Hopp, W. J., & Spearman, M. L. (2011). *Factory Physics* (3rd ed.). Waveland Press.
