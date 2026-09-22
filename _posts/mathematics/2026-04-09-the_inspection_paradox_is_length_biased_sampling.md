---
permalink: '/mathematics/the_inspection_paradox_is_length_biased_sampling/'
title: 'The Inspection Paradox Is Length-Biased Sampling'
date: '2026-04-09'
categories:
- Mathematics
tags:
- Renewal Theory
- Inspection Paradox
- Stochastic Processes
- Reliability
- Queueing Theory
author_profile: false
classes: wide
seo_title: 'The Inspection Paradox Is Length-Biased Sampling'
seo_description: 'A random observation time over-samples long renewal intervals. Renewal theory explains the inspection paradox, residual life, equilibrium waiting times, and why average intervals can mislead.'
seo_type: article
excerpt: >-
  The interval seen at a random time is not distributed like an interval chosen
  at random from the event sequence. Long intervals occupy more time and are
  therefore more likely to be observed.
summary: >-
  This article develops the inspection paradox from renewal theory. If
  inter-arrival times have mean m, the interval containing a random observation
  time has size-biased distribution proportional to x dF(x) and mean E[X^2]/E[X].
  The equilibrium forward and backward recurrence times both have mean
  E[X^2]/(2E[X]). A two-point example with 90% one-hour intervals and 10%
  ten-hour intervals has ordinary mean 1.9 hours, yet a random observer lands in
  a ten-hour interval more than half the time and sees an average enclosing
  interval of 5.74 hours. The article then connects the result to the elementary
  renewal theorem, Poisson memorylessness, reliability, prevalence sampling and
  observational bias.
keywords:
- inspection paradox
- renewal process
- length biased sampling
- residual life
- forward recurrence time
- renewal theorem
why_this_exists: >-
  Means computed over events are often interpreted as though they describe what
  a randomly arriving observer will experience. Renewal theory shows that these
  are different sampling schemes. Sampling in event index gives the original
  interval distribution, while sampling in calendar time weights each interval
  in proportion to its duration.
evidence: >-
  Classical renewal theory, equilibrium age and residual-life distributions,
  length-biased sampling identities, exact two-point calculations and the
  elementary renewal theorem.
methodology: >-
  Define a renewal process through iid inter-arrival times, compare event-index
  and time-index sampling, derive the size-biased interval law and equilibrium
  residual-life distribution, then apply the identities to deterministic,
  exponential and high-variance interval models.
reviewed_at: '2026-09-22'
header:
  image: /assets/images/headers/photo-mathematics-probability.jpg
  og_image: /assets/images/headers/photo-mathematics-probability.jpg
  overlay_image: /assets/images/headers/photo-mathematics-probability.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-probability.jpg
  twitter_image: /assets/images/headers/photo-mathematics-probability.jpg
---

<!--
Development contract
Question: Why does a randomly arriving observer see longer intervals than the intervals obtained by sampling events directly?
Claim: Sampling a renewal process in calendar time induces length-biased sampling. An interval of length x occupies x times as much time as a unit interval and is therefore x times as likely to contain a random observation time.
Counterclaim: The inspection paradox does not imply that the forward waiting time always exceeds the ordinary mean interval. The expected residual life is m(1+CV^2)/2, so it exceeds m only when the squared coefficient of variation is greater than one.
Evidence object: Exact two-point interval example, size-biased interval distribution, equilibrium forward-recurrence formula and Poisson memoryless exception.
Failure case: Using event-average durations to describe cross-sectional observations, interpreting MTBF as the expected wait seen by a random observer, or invoking the bus paradox without checking the variability of headways.
Reader payoff: Distinguish event-index sampling from time-index sampling and know which mean is relevant for waiting, reliability, prevalence and duration-biased observation.
Exclusions: A full textbook treatment of renewal equations, a queueing-theory survey and a reliability-maintenance tutorial.
-->

Suppose a sequence of events is separated by independent intervals. A report lists the observed intervals and gives their arithmetic mean. It is tempting to interpret that mean as the duration a randomly arriving observer is likely to experience. Renewal theory shows that this interpretation is generally wrong, because choosing an interval uniformly from the event sequence and choosing a random time on the clock are different sampling experiments.

Long intervals occupy more calendar time than short intervals. A ten-hour gap creates ten times as many opportunities for a random observation time to land inside it as a one-hour gap. The interval containing a random time is therefore not distributed like an ordinary inter-arrival time. It is length biased.

This simple fact produces the inspection paradox, the waiting-time paradox and several closely related sampling effects in reliability, queueing, survival analysis and prevalence studies. The paradox is not a contradiction. It is a change of measure created by the observation scheme.

The cleanest way to see the mechanism is through a renewal process.

## A renewal process separates event time from observation time

Let

$$
X_1,X_2,\ldots
$$

be independent positive inter-arrival times with common distribution $F$ and finite mean

$$
m
=
E[X].
$$

Define renewal epochs

$$
S_n
=
X_1+\cdots+X_n,
$$

with

$$
S_0=0.
$$

The counting process

$$
N(t)
=
\max
\{
n:S_n\le t
\}
$$

records how many renewals have occurred by time $t$.

If an analyst chooses an interval by choosing an event index $i$ uniformly from a long list of renewals, the sampled duration behaves like

$$
X_i\sim F.
$$

Its mean is simply

$$
E[X]
=
m.
$$

Now consider a different experiment. Observe the system at a time selected uniformly from a very long calendar window, independently of the renewal process, and record the interval containing that observation time.

An interval of length $x$ contributes $x$ units of calendar time to the observation window. All else equal, it is therefore $x$ times as likely to be selected by time sampling as by event sampling.

The probability law changes.

This distinction is the entire inspection paradox.

## Random-time sampling produces a size-biased interval distribution

Let $L$ denote the length of the renewal interval containing a random stationary observation time.

For a continuous distribution with density $f$, the density of $L$ is

$$
f_L(x)
=
\frac{
x f(x)
}{
E[X]
}.
$$

More generally, without requiring a density, the size-biased law can be written as

$$
P(L\in dx)
=
\frac{
x
}{
E[X]
}
P(X\in dx).
$$

The normalizing constant is the ordinary mean interval because

$$
\int x\,P(X\in dx)
=
E[X].
$$

The expected observed interval length is therefore

$$
E[L]
=
\frac{
E[X^2]
}{
E[X]
}.
$$

Writing

$$
E[X^2]
=
\operatorname{Var}(X)
+
E[X]^2,
$$

we obtain

$$
E[L]
=
E[X]
+
\frac{
\operatorname{Var}(X)
}{
E[X]
}.
$$

Hence

$$
E[L]
\ge
E[X],
$$

with strict inequality whenever the interval distribution is non-degenerate.

The inflation is exactly

$$
\frac{
\operatorname{Var}(X)
}{
E[X]
}.
$$

The inspection paradox is therefore not caused by the mean being poorly estimated. Even if the event-level interval distribution is known perfectly, time sampling targets a different distribution.

Variance determines how different the two sampling schemes become.

## A process with mean interval 1.9 hours is usually observed inside a 10-hour gap

Consider an intentionally simple renewal process:

$$
P(X=1)
=
0.9,
$$

and

$$
P(X=10)
=
0.1.
$$

Ninety percent of intervals last one hour. Ten percent last ten hours.

The ordinary mean interval is

$$
E[X]
=
0.9(1)
+
0.1(10)
=
1.9
$$

hours.

If we choose one interval uniformly from the event sequence, the probability of selecting a ten-hour interval is only

$$
10\%.
$$

A random observation time sees something very different.

Under size-biased sampling,

$$
P(L=1)
=
\frac{
1(0.9)
}{
1.9
}
\approx
0.4737,
$$

while

$$
P(L=10)
=
\frac{
10(0.1)
}{
1.9
}
\approx
0.5263.
$$

Although only one interval in ten is long, a random observer is more likely than not to land inside a ten-hour interval.

The reason is mechanical. In every 19 hours of expected renewal time, the short intervals contribute about nine hours and the long intervals contribute about ten hours. Calendar time is divided nearly evenly between the two interval types even though event counts are not.

The second moment is

$$
E[X^2]
=
0.9(1^2)
+
0.1(10^2)
=
10.9.
$$

The mean length of the interval seen at a random time is therefore

$$
E[L]
=
\frac{
10.9
}{
1.9
}
\approx
5.74
$$

hours.

The ordinary event-average interval is 1.9 hours.

The randomly observed interval averages 5.74 hours.

Nothing about the process changed. Only the sampling scheme changed.

## Age and residual life split the observed interval

At observation time $t$, define the age of the current interval,

$$
A(t)
=
t-S_{N(t)},
$$

and the residual life,

$$
R(t)
=
S_{N(t)+1}-t.
$$

The enclosing interval length is

$$
L(t)
=
A(t)+R(t).
$$

In an equilibrium renewal process, the observation time is uniformly distributed within the selected interval conditional on its length.

Given

$$
L=x,
$$

the expected age and residual life are both

$$
\frac x2.
$$

Averaging over the size-biased interval distribution gives

$$
E[A]
=
E[R]
=
\frac12
E[L].
$$

Hence, provided the second moment is finite,

$$
E[A]
=
E[R]
=
\frac{
E[X^2]
}{
2E[X]
}.
$$

For the two-point example,

$$
E[A]
=
E[R]
=
\frac{
10.9
}{
2(1.9)
}
\approx
2.87
$$

hours.

A new interval drawn at a renewal epoch has expected total length 1.9 hours.

A random observer entering at an arbitrary time expects to wait 2.87 hours for the next renewal.

The forward wait is longer than the ordinary mean interval in this example because the interval distribution is highly variable.

This is the classical waiting-time paradox.

The same identity can be expressed through the coefficient of variation.

Let

$$
CV^2
=
\frac{
\operatorname{Var}(X)
}{
E[X]^2
}.
$$

Because

$$
E[X^2]
=
E[X]^2
\left(
1+CV^2
\right),
$$

the equilibrium residual mean is

$$
E[R]
=
\frac{
E[X]
}{
2
}
\left(
1+CV^2
\right).
$$

This formula adds an important qualification to the usual informal story.

The expected residual wait exceeds the ordinary mean interval only when

$$
CV^2>1.
$$

It equals the mean when

$$
CV^2=1,
$$

and it is smaller when

$$
CV^2<1.
$$

Length bias always makes the enclosing interval longer on average when there is any variability.

It does not always make the forward wait exceed the ordinary mean interval.

Those are different statements.

## The equilibrium residual-life distribution is an integrated tail

The residual-life distribution can be written without first conditioning on the enclosing interval.

Let

$$
\bar F(x)
=
P(X>x)
$$

be the survival function of the inter-arrival time.

In equilibrium,

$$
P(R>x)
=
\frac{
1
}{
E[X]
}
\int_x^\infty
\bar F(u)\,du.
$$

The residual-life distribution is therefore proportional to the integrated tail of the original interval distribution.

Differentiating when $F$ has a density gives the equilibrium residual density

$$
f_R(x)
=
\frac{
\bar F(x)
}{
E[X]
}.
$$

This formula explains why long tails are amplified.

If

$$
\bar F(x)
$$

decays slowly, substantial residual probability remains at large $x$ because the observer is preferentially located inside long intervals.

The mean follows by integrating the survival function:

$$
E[R]
=
\int_0^\infty
P(R>x)\,dx.
$$

Substituting the equilibrium tail gives

$$
E[R]
=
\frac{
1
}{
E[X]
}
\int_0^\infty
\int_x^\infty
\bar F(u)\,du\,dx.
$$

Interchanging the order of integration yields

$$
E[R]
=
\frac{
1
}{
E[X]
}
\int_0^\infty
u\bar F(u)\,du.
$$

Using the standard identity

$$
E[X^2]
=
2
\int_0^\infty
u\bar F(u)\,du,
$$

we recover

$$
E[R]
=
\frac{
E[X^2]
}{
2E[X]
}.
$$

The second moment appears because random-time observation weights duration by duration.

Long intervals influence the result twice: once because they are longer and therefore more likely to be sampled, and again because, once selected, they leave potentially long residual waits.

## Deterministic intervals, Poisson intervals and variable intervals behave differently

The coefficient-of-variation form makes three important cases transparent.

If every interval is exactly

$$
c,
$$

then

$$
CV=0.
$$

A random observer is uniformly located somewhere inside a deterministic interval, so

$$
E[R]
=
\frac c2.
$$

There is no length bias in the enclosing interval because every interval has the same duration.

The forward wait is half the interval on average.

Now suppose inter-arrival times are exponential with rate

$$
\lambda.
$$

Then

$$
E[X]
=
\frac1\lambda,
$$

and

$$
CV=1.
$$

The residual-life formula gives

$$
E[R]
=
\frac1\lambda
=
E[X].
$$

More strongly, the residual life itself is exponential with the same rate:

$$
P(R>x)
=
e^{-\lambda x}.
$$

This is the memoryless property.

Arriving at an arbitrary time is statistically equivalent, for the forward wait, to starting immediately after an event.

The enclosing interval is still length biased. Because

$$
E[X^2]
=
\frac{
2
}{
\lambda^2
},
$$

we have

$$
E[L]
=
\frac{
2/\lambda^2
}{
1/\lambda
}
=
\frac2\lambda.
$$

The average interval containing a random time is twice the ordinary mean interval.

The Poisson process removes the waiting-time paradox only in the forward direction.

It does not remove length bias in the total enclosing interval.

Finally, when

$$
CV>1,
$$

as in bursty or heavy-tailed inter-arrival processes, the expected residual wait exceeds the ordinary mean interval.

The more variable the intervals, the more misleading the event-level mean becomes for a random observer.

## The elementary renewal theorem governs the long-run event rate

The inspection paradox concerns what a random time sees.

Renewal theory also provides the long-run event frequency.

Under standard assumptions with

$$
E[X]<\infty,
$$

the elementary renewal theorem states that

$$
\frac{
E[N(t)]
}{
t
}
\to
\frac{
1
}{
E[X]
}
$$

as

$$
t\to\infty.
$$

A stronger almost-sure renewal law gives

$$
\frac{
N(t)
}{
t
}
\to
\frac{
1
}{
E[X]
}
$$

under familiar iid conditions.

The reciprocal mean interval is therefore the long-run event rate.

For the two-point example,

$$
\frac1{E[X]}
=
\frac1{1.9}
\approx
0.526
$$

renewals per hour.

This rate is perfectly consistent with the random observer seeing long intervals.

Event frequency and time occupancy answer different questions.

A process can generate many short intervals numerically while a large share of calendar time is spent inside a relatively small number of long intervals.

This distinction is central in reliability.

Suppose failures are followed by operational periods of highly variable length. The average interval between failure epochs describes event frequency.

A randomly chosen inspection time describes occupancy of operational intervals in proportion to how long they last.

Confusing the two leads directly to duration bias.

## Prevalence sampling is renewal length bias in another language

The same mathematics appears in survival studies.

Imagine episodes with durations

$$
X_1,X_2,\ldots.
$$

An incidence sample recruits episodes when they begin.

A prevalence sample observes the population at a random calendar time and recruits episodes currently in progress.

Long episodes are more likely to be active at the sampling time.

The observed duration distribution is therefore length biased.

This is why cross-sectional samples of ongoing disease, unemployment, hospitalization or device downtime can overrepresent long-duration cases even when episode onset rates are constant.

The mechanism is not confounding in the ordinary regression sense.

It is selection induced by exposure time.

A duration of ten days creates ten times as much opportunity for cross-sectional observation as a duration of one day.

The same issue appears in status dashboards.

If an operations team opens a dashboard at random times and records the outages currently active, the sample is weighted toward long outages.

The average outage duration among active incidents is not the average outage duration among all incidents.

Both are valid quantities.

They answer different questions.

## MTBF is not automatically the expected wait seen at a random time

Mean time between failures is often used as though it described how long an observer should expect to wait until the next failure.

For a renewal failure process with iid operating intervals,

$$
\mathrm{MTBF}
=
E[X].
$$

A random-time observer instead sees expected residual operating life

$$
E[R]
=
\frac{
E[X^2]
}{
2E[X]
}.
$$

These are equal only under special interval distributions, including the exponential case.

If operating times are nearly deterministic,

$$
CV<1,
$$

then

$$
E[R]
<
E[X].
$$

If operating times are highly variable,

$$
CV>1,
$$

then

$$
E[R]
>
E[X].
$$

The mean alone is therefore insufficient.

Two systems can share the same MTBF and have different random-time residual-life distributions because their interval variances differ.

For maintenance planning, this distinction matters whenever the observation time is not synchronized with a renewal epoch.

A planner commissioning a brand-new component begins at age zero.

A technician inspecting an installed component at an arbitrary calendar time sees a length-biased age distribution.

Those are different prediction problems.

## Alternating renewal processes turn the same idea into availability

Suppose a system alternates between an up period

$$
U
$$

and a down period

$$
D.
$$

The cycle length is

$$
C
=
U+D.
$$

If cycles are iid and have finite means, the renewal-reward theorem gives the long-run fraction of time the system is operational:

$$
A
=
\frac{
E[U]
}{
E[U]+E[D]
}.
$$

This is a time-weighted quantity.

It is not the fraction of episodes that are up, because every cycle contains exactly one up and one down episode.

If down durations vary substantially, a random inspection disproportionately encounters long outages.

The long-run unavailability is

$$
1-A
=
\frac{
E[D]
}{
E[U]+E[D]
},
$$

while the duration distribution of the outage seen conditional on observing the system down is the length-biased version of $D$.

Hence even after availability is known, the expected duration of the currently observed outage is

$$
\frac{
E[D^2]
}{
E[D]
},
$$

and the expected remaining downtime, under equilibrium sampling, is

$$
\frac{
E[D^2]
}{
2E[D]
}.
$$

A system can have acceptable long-run availability while producing very long residual waits during the relatively rare outages that dominate observed downtime.

This is another reason means alone are not enough for service-level planning.

## Event sampling and time sampling are different probability measures

The inspection paradox is often presented as a clever puzzle.

Its deeper mathematical lesson is about sampling measures.

Event-index sampling gives equal weight to each interval.

Time-index sampling gives weight proportional to interval duration.

If

$$
g(X)
$$

is any function of interval length, its event-average expectation is

$$
E[g(X)].
$$

The corresponding time-sampled expectation over the enclosing interval is

$$
E_{\mathrm{time}}[g(L)]
=
\frac{
E[Xg(X)]
}{
E[X]
}.
$$

The factor

$$
X
$$

is the change of measure.

Setting

$$
g(x)=x
$$

gives

$$
E[L]
=
\frac{
E[X^2]
}{
E[X]
}.
$$

Setting

$$
g(x)=\mathbf 1\{x>a\}
$$

gives

$$
P(L>a)
=
\frac{
E[
X\mathbf 1\{X>a\}
]
}{
E[X]
}.
$$

Large intervals are overrepresented in every time-sampled functional.

This perspective extends naturally to Palm probability, which formalizes the difference between observing a stochastic process at a typical time and observing it from a typical event.

A typical event sees one probability law.

A typical time sees another.

Many apparent paradoxes in point processes are really failures to specify which observer is being used.

## The bus paradox depends on headway variability, not on buses

Suppose buses have iid headways $X$ and a passenger arrives independently at a random time in equilibrium.

The expected passenger wait is

$$
E[R]
=
\frac{
E[X^2]
}{
2E[X]
}.
$$

Write this as

$$
E[R]
=
\frac{
E[X]
}{
2
}
\left(
1+CV^2
\right).
$$

If buses arrive exactly every ten minutes,

$$
CV=0,
$$

and the expected wait is five minutes.

If headways are exponential with mean ten minutes,

$$
CV=1,
$$

and the expected wait is ten minutes.

If bunching produces

$$
CV=1.5,
$$

then

$$
CV^2=2.25,
$$

and the expected wait becomes

$$
\frac{
10
}{
2
}
(1+2.25)
=
16.25
$$

minutes.

The paradox is therefore not that a passenger always waits longer than the published mean headway.

The paradox is that a random passenger does not sample headways uniformly.

Variability controls the difference.

The same formula applies to inspection intervals, packet gaps, machine failures, service episodes and many other renewal settings.

The bus is only the mnemonic.

## Random observation overweights persistence

The two-point example contains the whole argument.

Ninety percent of event intervals last one hour.

Only ten percent last ten hours.

The ordinary mean is 1.9 hours.

A random time nevertheless lands in a ten-hour interval with probability about 52.6%.

The enclosing interval averages 5.74 hours.

The expected remaining wait is 2.87 hours.

No estimator failed.

No sample was too small.

The observer changed.

Renewal theory makes this distinction precise. Sampling intervals by event index recovers the original inter-arrival distribution. Sampling the process at a random calendar time produces the size-biased law. The equilibrium age and residual life inherit the second moment, which is why variability matters so strongly.

This is the general lesson of the inspection paradox.

Long states are easier to observe because they persist.

Whenever observation probability grows with duration, the observed sample is not a neutral sample of episodes.

The sampling clock has become part of the probability model.

## References

Asmussen, S. (2003). *Applied Probability and Queues* (2nd ed.). Springer.

Cox, D. R. (1962). *Renewal Theory*. Methuen.

Feller, W. (1971). *An Introduction to Probability Theory and Its Applications, Volume II* (2nd ed.). Wiley.

Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press.

Smith, W. L. (1958). Renewal theory and its ramifications. *Journal of the Royal Statistical Society: Series B*, 20(2), 243–302.

Vardi, Y. (1982). Nonparametric estimation in the presence of length bias. *The Annals of Statistics*, 10(2), 616–620.
