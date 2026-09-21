---
permalink: '/statistics/competing_risks_recurrent_events_and_multistate_models_are_not_the_same_problem/'
title: 'Competing Risks, Recurrent Events, and Multi-State Models Are Not the Same Problem'
date: '2025-09-18'
categories:
- Statistics
tags:
- Survival Analysis
- Competing Risks
- Recurrent Events
- Multi-State Models
- Joint Models
author_profile: false
classes: wide
seo_title: 'Competing Risks, Recurrent Events, and Multi-State Models Are Different Problems'
seo_description: 'Competing risks, recurrent events, multi-state processes and joint longitudinal-survival models require different estimands, risk sets and stochastic structures. One time-to-first-event model cannot answer all of them.'
seo_type: article
excerpt: >-
  A single event time is only one possible event-history object. When causes
  compete, events recur, subjects move through intermediate states, or a
  longitudinal process evolves jointly with event risk, the estimand and risk
  set change with the scientific question.
summary: >-
  This article develops a unified event-history view of advanced survival
  analysis. Exact examples show why censoring competing events can overstate
  absolute incidence, why two processes can have identical time-to-first-event
  distributions while having very different recurrent-event burdens, and how a
  three-state illness-death model turns transition hazards into state occupancy
  probabilities. The discussion connects counting processes, cause-specific
  hazards, cumulative incidence, recurrent-event clocks, multi-state transition
  intensities and shared-parameter joint models.
keywords:
- competing risks
- recurrent events
- multi-state models
- joint longitudinal survival models
- event history
- counting processes
why_this_exists: >-
  Survival analysis is often introduced through one event time and one censoring
  indicator. That representation is insufficient when several event types
  compete, events can happen repeatedly, or intermediate states alter future
  risk. Compressing such data to one terminal event changes the scientific
  question rather than merely simplifying the analysis.
evidence: >-
  Counting-process survival theory, competing-risk cumulative-incidence
  identities, recurrent-event process calculations, Markov multi-state models
  and shared-parameter joint longitudinal-event formulations.
methodology: >-
  Start from the single-event counting-process representation and then alter one
  structural assumption at a time. Use exact constant-hazard examples to show
  how the estimand changes under competing risks, recurrent events and
  multi-state transitions before introducing joint longitudinal-survival models.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-statistics-survival.jpg
  og_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_image: /assets/images/headers/photo-statistics-survival.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-survival.jpg
  twitter_image: /assets/images/headers/photo-statistics-survival.jpg
---

<!--
Development contract
Question: How should survival analysis change when the event process is more complex than one terminal event?
Claim: Competing risks, recurrent events, multi-state processes and joint longitudinal-event models target different stochastic objects. Replacing them all with time to first event changes the estimand and can discard essential information.
Counterclaim: A simpler time-to-first-event analysis can be entirely appropriate when the scientific target is deliberately limited to the first event and later transitions are irrelevant to the decision.
Evidence object: Exact two-cause cumulative-incidence example, two recurrent-event processes with identical first-event distributions but different event burden, and a three-state illness-death model with closed-form occupancy probabilities.
Failure case: Treating competing events as independent censoring, collapsing recurrent histories into one event without stating the estimand change, interpreting hazards as probabilities, or using noisy longitudinal measurements as error-free time-varying covariates.
Reader payoff: Match the event-history model to the actual scientific process before choosing a Cox model, Fine-Gray model, recurrent-event model, multi-state model or joint model.
Exclusions: Repeating introductory Kaplan-Meier material, reproducing the existing dedicated competing-risk and recurrent-maintenance tutorials, or presenting one advanced survival model as universally preferable.
-->

A conventional survival dataset often contains one row per subject, one follow-up time and one indicator showing whether the event occurred. That representation is useful because many important questions genuinely concern one terminal transition. Time to death, time to first machine failure, time to customer churn and time to first hospital admission can all be scientifically meaningful outcomes. The difficulty begins when the underlying process contains more structure than the single-event representation preserves.

A patient can relapse and later die. A machine can fail, be repaired and fail again. A borrower can move from current to delinquent to default, or return from delinquency to current. A patient can be discharged before readmission becomes possible, die before relapse, or experience repeated admissions. A longitudinal biomarker can deteriorate gradually while event risk changes continuously with an unobserved latent trajectory. In each case, reducing the record to one event time is not merely an innocuous simplification. It defines a different stochastic object.

The correct model therefore begins with the event process rather than with a favourite survival function. Competing risks ask which event occurs first when several terminal causes are possible. Recurrent-event methods count events that can happen repeatedly. Multi-state models follow transitions among several states, some transient and some absorbing. Joint models connect a repeatedly measured longitudinal process with an event process when the two are statistically and scientifically dependent. These frameworks overlap, but they are not interchangeable because they answer different questions.

The common language that connects them is counting-process and transition-process notation. Once the event history is written explicitly, the differences among models become much clearer.

## The one-event survival model is a special case

For one terminal event, define the event time $T$ and the at-risk indicator

$$
Y(t)
=
\mathbf 1\{T\ge t\}.
$$

The counting process

$$
N(t)
=
\mathbf 1\{T\le t\}
$$

jumps from zero to one when the event occurs. The hazard function is

$$
\lambda(t)
=
\lim_{\Delta t\downarrow0}
\frac{
P(t\le T<t+\Delta t\mid T\ge t)
}{
\Delta t
}.
$$

The cumulative hazard is

$$
\Lambda(t)
=
\int_0^t
\lambda(u)\,du,
$$

and, under the usual continuous-time relationship,

$$
S(t)
=
P(T>t)
=
\exp[-\Lambda(t)].
$$

This is already a transition model, but one with only two states: event-free and absorbed. The subject begins in the event-free state, remains there until time $T$, and then makes one irreversible transition.

Many familiar survival methods exploit exactly this structure. Kaplan-Meier estimates the survival probability in the initial state. Cox regression models how covariates change the transition hazard out of that state. Nelson-Aalen estimates cumulative hazard. None of these ideas is conceptually limited to medicine. The same mathematics applies to machines, contracts, customers or any other units observed until one transition.

The limitation appears when $N(t)$ can jump more than once, when several different jumps are possible, or when the meaning of the risk set changes after an intermediate event. The problem is not that ordinary survival analysis becomes "less advanced". The random object itself has changed.

## Competing risks change probability because one cause removes the others

Suppose the first event can occur from one of $K$ causes. Let $T$ be the time to the first event and $J\in\{1,\ldots,K\}$ its cause. For cause $k$, the cause-specific hazard is

$$
\lambda_k(t)
=
\lim_{\Delta t\downarrow0}
\frac{
P(t\le T<t+\Delta t,\ J=k\mid T\ge t)
}{
\Delta t
}.
$$

The overall hazard is the sum

$$
\lambda(t)
=
\sum_{k=1}^K
\lambda_k(t),
$$

so the probability of remaining free of every event is

$$
S(t)
=
\exp
\left[
-\int_0^t
\sum_{k=1}^K
\lambda_k(u)\,du
\right].
$$

The absolute probability that cause $k$ occurs by time $t$ is the cumulative incidence function,

$$
F_k(t)
=
P(T\le t,\ J=k)
=
\int_0^t
S(u^-)\lambda_k(u)\,du.
$$

The survival term inside this integral is the crucial part. A subject can experience cause $k$ at time $u$ only if no competing event has already occurred. The real-world incidence of one cause therefore depends on the hazards of all competing causes.

An exact constant-hazard example shows why treating a competing event as ordinary censoring can change the answer. Suppose two first-event causes have hazards

$$
\lambda_1=0.08
$$

and

$$
\lambda_2=0.12
$$

per year. Overall event-free survival is

$$
S(t)
=
e^{-0.20t}.
$$

The cumulative incidence of cause 1 is

$$
F_1(t)
=
\int_0^t
e^{-0.20u}(0.08)\,du
=
\frac{0.08}{0.20}
\left(
1-e^{-0.20t}
\right).
$$

At five years,

$$
F_1(5)
=
0.4(1-e^{-1})
\approx
0.2528.
$$

Now imagine censoring cause 2 and estimating cause 1 as though the competing cause simply removed independent follow-up. The corresponding single-cause survival calculation produces

$$
1-e^{-0.08(5)}
\approx
0.3297.
$$

The hypothetical probability is about 33.0%, while the actual probability of experiencing cause 1 before cause 2 is about 25.3%. The single-cause calculation is roughly 30% larger.

Nothing is wrong with the cause-specific hazard $\lambda_1=0.08$. The problem comes from interpreting it as though cause 2 did not remove subjects from the event-free state. Cause-specific hazards are mechanism-oriented transition rates. Cumulative incidence is an absolute probability in the world where the competing processes actually operate.

This is why cause-specific Cox models and Fine-Gray models should not be treated as competing software choices for the same estimand. A cause-specific hazard model asks how covariates change the instantaneous cause-$k$ transition rate among subjects who are still free of every event. A Fine-Gray subdistribution model parameterises a different hazard connected to the cumulative incidence function. The coefficients need not have the same sign or interpretation when covariates also affect competing causes.

The blog already has a dedicated treatment of this distinction in [Competing Risks in Healthcare and Predictive Maintenance](/statistics/competing_risks_healthcare_predictive_maintenance/). The broader point here is structural: competing risks still allow only one first event, but the event type becomes part of the stochastic outcome. The data object is $(T,J)$ rather than $T$ alone.

## Recurrent events require a process rather than one event time

A recurrent-event history is different again because the same event type can happen repeatedly. Instead of a counting process that jumps at most once, define

$$
N_i(t)
=
\sum_{j\ge1}
\mathbf 1\{T_{ij}\le t\},
$$

where $T_{ij}$ is the time of subject $i$'s $j$th event. The quantity of interest may be an event rate, mean cumulative function, gap-time distribution, event-specific hazard or subject-level heterogeneity in event propensity.

Compressing this history to

$$
T_{i1},
$$

the time of the first event, can discard most of the process. More importantly, two recurrent-event mechanisms can have exactly the same first-event distribution and radically different long-run burden.

Consider two processes. In both, the first event has exponential rate

$$
\lambda_1=0.5
$$

per year, so

$$
P(T_1>t)
=
e^{-0.5t}.
$$

The probability of experiencing at least one event within three years is therefore identical in both processes,

$$
P(T_1\le3)
=
1-e^{-1.5}
\approx
0.7769.
$$

A time-to-first-event analysis cannot distinguish the processes.

In process A, events continue after the first according to a homogeneous Poisson process with rate $0.5$ per year. The expected number of events by three years is simply

$$
E[N_A(3)]
=
0.5(3)
=
1.5.
$$

In process B, the first event still occurs at rate $0.5$, but after the first event the recurrent event rate increases to

$$
\lambda_2=2
$$

per year. Conditional on first event time $T_1=s$, the expected number of additional events before time $t$ is

$$
\lambda_2(t-s).
$$

Therefore

$$
E[N_B(t)]
=
P(T_1\le t)
+
\lambda_2
E[(t-T_1)_+].
$$

For an exponential first-event time,

$$
E[(t-T_1)_+]
=
t-
\frac{
1-e^{-\lambda_1t}
}{
\lambda_1
}.
$$

At three years,

$$
E[(3-T_1)_+]
=
3-
\frac{
1-e^{-1.5}
}{
0.5
}
\approx
1.4463.
$$

Hence

$$
E[N_B(3)]
\approx
0.7769
+
2(1.4463)
=
3.6694.
$$

The two processes have the same probability of a first event by three years, 77.7%, but expected event counts of 1.5 and 3.67. Any model that keeps only the first event declares them identical on its chosen outcome even though their recurrent burden differs by more than a factor of two.

This is not an argument that time to first event is wrong. It is an argument that it estimates a narrower quantity. If the decision concerns whether any failure occurs before warranty expiry, first failure may be exactly the right endpoint. If the decision concerns maintenance workload, hospital admissions, seizure burden or repeated infections, discarding later events changes the estimand.

Recurrent-event models also differ in the clock they use. A total-time model measures every event on the original calendar or age scale. A gap-time model resets the clock after each event and models time since the previous event. The two clocks encode different scientific ideas. If risk depends on accumulated age, total time can be natural. If repair or treatment resets the system partially and recent history matters most, gap time may better represent the mechanism.

Risk-set construction changes as well. Andersen-Gill models use a counting-process formulation in which a subject can return to the risk set after an event. Prentice-Williams-Peterson formulations condition on event order, so the risk set for the third event contains only subjects who have already experienced two. Frailty models introduce latent subject heterogeneity, acknowledging that repeated events within one individual are not independent replicates.

The dedicated article [Recurrent Failures: Why Time to First Failure Throws Away Two Thirds of the Data](/statistics/recurrent_events_maintenance_mean_cumulative_function/) develops these ideas in a maintenance setting. The mathematical point here is that a recurrent history is a stochastic process $N(t)$, not a single event time.

## Multi-state models make the intermediate states explicit

Competing risks and recurrent events can both be represented within broader state-transition systems. A multi-state model defines a finite set of states and allows specified transitions among them. If $X(t)$ denotes the state occupied at time $t$, the fundamental quantities are transition probabilities

$$
P_{rs}(s,t)
=
P\{X(t)=s\mid X(s)=r\}
$$

and transition intensities

$$
\lambda_{rs}(t)
=
\lim_{\Delta t\downarrow0}
\frac{
P\{X(t+\Delta t)=s\mid X(t)=r\}
}{
\Delta t
},
$$

for allowed transitions $r\to s$.

Competing risks are a simple multi-state model with one initial state and several absorbing destination states. An illness-death model is richer because illness is an intermediate state. A subject can move from healthy to ill, healthy directly to death, or ill to death.

Consider a three-state model:

$$
0=\text{healthy},
\qquad
1=\text{ill},
\qquad
2=\text{dead}.
$$

Suppose the constant transition intensities are

$$
\lambda_{01}
=
\alpha
=
0.08,
$$

$$
\lambda_{02}
=
\beta
=
0.02,
$$

and

$$
\lambda_{12}
=
\gamma
=
0.05.
$$

Starting in state 0, the probability of remaining healthy through time $t$ is

$$
P_{00}(0,t)
=
e^{-(\alpha+\beta)t}.
$$

To be alive and ill at time $t$, the subject must enter illness at some time $u<t$ and then survive in the illness state until $t$. Hence

$$
P_{01}(0,t)
=
\int_0^t
e^{-(\alpha+\beta)u}
\alpha
e^{-\gamma(t-u)}
\,du.
$$

When

$$
\alpha+\beta\ne\gamma,
$$

this becomes

$$
P_{01}(0,t)
=
\alpha e^{-\gamma t}
\frac{
1-e^{-(\alpha+\beta-\gamma)t}
}{
\alpha+\beta-\gamma
}.
$$

At five years,

$$
P_{00}(0,5)
=
e^{-0.5}
\approx
0.6065,
$$

and

$$
P_{01}(0,5)
\approx
0.2756.
$$

The remaining probability is death,

$$
P_{02}(0,5)
=
1-P_{00}(0,5)-P_{01}(0,5)
\approx
0.1178.
$$

The five-year state distribution is therefore about 60.7% healthy, 27.6% alive after illness and 11.8% dead.

A single time-to-death model would retain only the final absorbed state and discard the intermediate illness history. A competing-risk model with "illness" and "death" as first events would distinguish which event occurs first but would stop following the post-illness mortality transition. The multi-state model preserves both transition sequence and state occupancy.

This matters because the scientific questions are often state-specific. How long is a patient expected to spend disease-free? What proportion of a fleet is degraded but still operational? What is the probability of returning from delinquency to current before default? How does an intervention affect the transition into illness versus the transition from illness to death? These are not reducible to one overall survival curve.

For nonparametric estimation, the Aalen-Johansen estimator generalises Kaplan-Meier to transition probabilities in multi-state and competing-risk settings. In Markov models, future transition behaviour depends on the current state rather than the full past history. Semi-Markov models relax this by allowing transition intensities to depend on duration in the current state or other aspects of history.

Choosing between Markov and semi-Markov formulations is therefore not merely computational. It encodes whether time since study origin, time since state entry or deeper event history carries predictive information about the next transition.

## The event clock and the risk set define the estimand

Advanced survival analysis can look like a collection of model names, but many differences reduce to two questions: which clock is running, and who belongs to the risk set at each instant?

In a first-event model, a subject leaves the risk set permanently after the event. In a competing-risk cause-specific model, any first event removes the subject from all cause-specific risk sets. In recurrent-event analysis, the subject can often return to risk after an event. In a multi-state model, the subject leaves one state-specific risk set and enters another.

The time scale can be age, calendar time, time since enrolment, time since treatment, time since previous event or time since entering the current state. The same raw event history can yield different hazards under different clocks because hazard is conditional on the chosen risk set and time scale.

This is why hazard ratios should not be treated as direct probability ratios. A hazard is a local transition rate among subjects currently eligible for that transition. Cumulative incidence and state occupancy probabilities integrate those rates while accounting for competing transitions and changing risk-set composition.

The distinction becomes especially important with treatment or policy variables that affect several transitions. A treatment can reduce the hazard of relapse while increasing another event that removes subjects from relapse risk. A maintenance policy can reduce catastrophic failure by increasing preventive replacement. A customer-retention intervention can reduce voluntary churn while increasing forced account closure through another rule. The absolute probability of one outcome depends on the entire transition system.

A compact mapping is useful:

| Scientific target | Natural event-history object |
| --- | --- |
| Time until any first event | Single-event survival |
| Probability of a particular first cause | Cumulative incidence under competing risks |
| Instantaneous mechanism-specific transition rate | Cause-specific hazard |
| Number or rate of repeated events | Recurrent counting process |
| Probability of occupying or reaching a state | Multi-state transition probabilities |
| Relationship between noisy longitudinal trajectory and event risk | Joint longitudinal-event model |

The table is not a hierarchy. It is a map from question to stochastic object.

## Joint models are needed when the longitudinal process is part of the event mechanism

A final complication appears when event risk depends on an evolving quantity that is itself measured with error. Suppose a patient has latent biomarker trajectory

$$
m_i(t),
$$

but observations are

$$
Y_{ij}
=
m_i(t_{ij})
+
\varepsilon_{ij}.
$$

A common longitudinal submodel is

$$
m_i(t)
=
x_i(t)^\top\beta
+
z_i(t)^\top b_i,
$$

where $b_i$ contains subject-specific random effects.

Suppose event hazard depends on the latent current biomarker value,

$$
\lambda_i(t)
=
\lambda_0(t)
\exp
\left[
\gamma^\top w_i
+
\alpha m_i(t)
\right].
$$

This is a shared-parameter joint model. The longitudinal measurements inform the latent process $m_i(t)$, and the event process depends on that latent trajectory rather than on noisy measurements treated as exact covariates.

A naive time-dependent Cox model might insert the most recent observed biomarker value directly,

$$
\lambda_i(t)
=
\lambda_0(t)
\exp[
\alpha Y_i(t)
].
$$

That can be problematic for two reasons. First, $Y_i(t)$ contains measurement error, producing an errors-in-variables problem. Second, the measurement schedule and event process can be informative. Subjects deteriorating rapidly may be measured more often, or death may terminate the longitudinal record. The observed biomarker process and event process are then statistically coupled.

Joint modelling does not solve every problem automatically. The association structure between $m_i(t)$ and hazard must be specified. Risk might depend on current value, slope, cumulative exposure or another functional. Random-effects distributions can be misspecified. Longitudinal measurement schedules may be informative beyond what the model captures.

The reason for joint modelling is conceptual rather than ceremonial. If the scientific claim concerns how an underlying longitudinal process relates to event risk, then the uncertainty in that process should propagate into survival inference.

The same idea extends beyond healthcare. Vibration or temperature trajectories can evolve jointly with machine failure. Account balances can evolve jointly with default. Usage trajectories can evolve jointly with churn. Once the longitudinal process is treated as latent rather than error-free, the event model and measurement model belong to one probabilistic system.

## Model choice should begin with the history that matters

The most common mistake in advanced survival analysis is not choosing the wrong regression family. It is defining the event history too narrowly before modelling begins.

If death prevents recurrence, death is part of the probability structure. If failures recur after repair, later failures are part of the outcome unless the scientific question is explicitly first failure. If subjects move through clinically or operationally meaningful intermediate states, state occupancy and transition order may matter more than a terminal event. If an evolving biomarker is noisy and tied to event risk, treating its observed values as fixed truth can distort both parts of the analysis.

Simpler models remain valuable. A single-event Cox model is often exactly right for a well-defined first-event estimand. Time to first event can be robust, interpretable and operationally relevant. Cause-specific hazards can be preferable to more complicated multi-state models when the scientific question concerns transition mechanisms. A multi-state model can become needlessly elaborate if intermediate states add no decision-relevant information.

Complexity should enter because the event process requires it, not because advanced methods are available.

The useful sequence is therefore to write down the possible histories before writing down the regression. What events can happen? Can they recur? Which events prevent others? Which transitions remain possible after each event? Does time reset after a transition? Are intermediate states scientifically meaningful? Is there a latent longitudinal process that changes risk? Which probability, rate, expected count or state occupancy is the actual target?

Once those questions are answered, the model class becomes much less mysterious.

Competing risks, recurrent events and multi-state models are not different levels of sophistication applied to one survival problem. They are models for different event histories. Joint longitudinal-survival models extend the same principle when the event process cannot be separated cleanly from a noisy evolving covariate.

The data structure should follow the history, and the estimand should follow the question. Only then should the regression model follow.

## References

Aalen, O. O., Borgan, Ø., & Gjessing, H. K. (2008). *Survival and Event History Analysis: A Process Point of View*. Springer.

Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). *Statistical Models Based on Counting Processes*. Springer.

Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis. *Statistical Methods in Medical Research*, 11(2), 91–115.

Cook, R. J., & Lawless, J. F. (2007). *The Statistical Analysis of Recurrent Events*. Springer.

Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496–509.

Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data* (2nd ed.). Wiley.

Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics: competing risks and multi-state models. *Statistics in Medicine*, 26(11), 2389–2430. https://doi.org/10.1002/sim.2712

Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event Data: With Applications in R*. Chapman & Hall/CRC.

Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
