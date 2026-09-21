---
permalink: '/mathematics/concentration_inequalities_quantify_how_random_sums_leave_their_typical_set/'
title: 'Concentration Inequalities Quantify How Random Sums Leave Their Typical Set'
date: '2026-02-26'
categories:
- Mathematics
tags:
- Probability
- Concentration Inequalities
- Hoeffding
- Bernstein
- Chernoff Bounds
author_profile: false
classes: wide
seo_title: 'Concentration Inequalities Quantify Random Deviations'
seo_description: 'Hoeffding, Bernstein and Chernoff bounds are finite-sample statements about rare deviations. Their sharpness depends on which information about the random variables is actually used.'
seo_type: article
excerpt: >-
  The law of large numbers says averages stabilize, and the central limit
  theorem describes their typical fluctuations. Concentration inequalities ask
  a different question: how unlikely is a finite-sample deviation of a given
  size, and what assumptions make that guarantee sharp?
summary: >-
  This article develops concentration inequalities from the exponential-moment
  method. Hoeffding's inequality is derived as a distribution-free bound for
  bounded variables, Bernstein's inequality is shown to use variance as well as
  range, and the binomial Chernoff bound is expressed through Bernoulli
  Kullback-Leibler divergence. In a rare-event example with n=1000 and p=0.01,
  the exact probability that the empirical rate reaches 0.03 is about 2.06e-7;
  Hoeffding gives 0.449, Bernstein 5.71e-6 and the KL-Chernoff bound 1.92e-6.
  The article then develops sub-Gaussian and sub-exponential variables,
  maxima via union bounds, bounded-difference inequalities, martingale
  extensions, and the limits of concentration under heavy tails and dependence.
keywords:
- concentration inequalities
- Hoeffding inequality
- Bernstein inequality
- Chernoff bound
- sub-Gaussian random variables
- large deviations
why_this_exists: >-
  Probability is often taught through expectations, variances and asymptotic
  approximations. Many scientific and algorithmic decisions instead require a
  finite-sample statement of the form: with probability at least 1-delta, how
  far can this random quantity move? Concentration inequalities provide that
  language, but different inequalities can differ by many orders of magnitude
  because they use different information about the data-generating process.
evidence: >-
  Classical exponential-moment bounds, Hoeffding's lemma, Bernstein and Chernoff
  inequalities, exact binomial tail probabilities, sub-Gaussian and
  sub-exponential Orlicz-type behaviour, bounded-difference inequalities and
  martingale concentration.
methodology: >-
  Derive concentration from Markov's inequality applied to exponential moments,
  specialize first to bounded variables through Hoeffding's lemma, then add
  variance information through Bernstein and exact Bernoulli structure through
  relative entropy. Use one rare-event Bernoulli example to compare the bounds
  numerically and then generalize the geometry to sub-Gaussian,
  sub-exponential and dependent settings.
reviewed_at: '2026-09-21'
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
Question: What can be said non-asymptotically about the probability that a random sum or estimator deviates substantially from its expectation?
Claim: Concentration inequalities convert structural assumptions such as boundedness, variance control or exponential moments into finite-sample exponential tail bounds. Their usefulness depends on matching the inequality to the information genuinely available.
Counterclaim: Concentration bounds are often conservative, sometimes dramatically so. Exact distributional calculations, saddlepoint methods, normal approximations or simulation can be much sharper when their assumptions are justified.
Evidence object: Exact Binomial(1000,0.01) tail probability for observing at least 30 successes, compared with Hoeffding, Bernstein and Bernoulli KL-Chernoff bounds.
Failure case: Applying bounded-variable inequalities to heavy-tailed data, treating a loose upper bound as an approximation to the actual probability, ignoring dependence, or choosing a bound after seeing which one produces the desired conclusion.
Reader payoff: Understand where exponential tail bounds come from, why Hoeffding can be too conservative, how variance-sensitive bounds improve it, and how concentration language extends to maxima, functions of many variables and sequential processes.
Exclusions: A catalogue of every concentration theorem, a machine-learning generalization-bound survey, and a measure-theoretic treatment of large-deviation principles.
-->

The law of large numbers tells us that an average of independent observations converges toward its expectation. The central limit theorem goes further and describes the scale and approximate shape of ordinary fluctuations around that expectation. Both results are foundational, but neither directly answers a common finite-sample question:

> Given the sample size I actually have, how unlikely is a deviation of this size?

That question appears everywhere. How unlikely is it that a defect rate estimated from one thousand components exceeds its nominal value by two percentage points? How far can an empirical mean move from its expectation with probability at most one in a million? How large can the maximum of ten thousand noisy coordinates become? How much can a function of many independent inputs change because the inputs fluctuate? How confident can we be that a stochastic algorithm has not deviated substantially from its average behaviour?

Concentration inequalities answer questions of this form by providing non-asymptotic tail bounds. Instead of waiting for

$$
n\to\infty,
$$

they give inequalities valid at a specified finite $n$.

The price is that a bound is usually not an exact probability. It is a guarantee derived from limited structural information. If all we know is that the variables lie in an interval, Hoeffding's inequality gives a distribution-free exponential tail. If we also know the variance is small, Bernstein-type inequalities can be much sharper. If we know the exact Bernoulli distribution, the Chernoff method can exploit its full moment-generating function and produce a relative-entropy bound that is sharper again.

These are not competing formulas for the same information. They answer the same tail question from increasingly rich assumptions.

The distinction becomes dramatic in rare-event problems.

## Exponential moments turn a tail probability into an optimization problem

Let

$$
S
$$

be a random variable whose upper tail we want to bound. For any

$$
\lambda>0,
$$

the event

$$
S\ge t
$$

is equivalent to

$$
e^{\lambda S}
\ge
e^{\lambda t}.
$$

Markov's inequality therefore gives

$$
P(S\ge t)
=
P
\left(
e^{\lambda S}
\ge
e^{\lambda t}
\right)
\le
e^{-\lambda t}
E[e^{\lambda S}].
$$

Because this is true for every positive $\lambda$,

$$
P(S\ge t)
\le
\inf_{\lambda>0}
\left\{
e^{-\lambda t}
E[e^{\lambda S}]
\right\}.
$$

If $S$ is centred,

$$
E[S]=0,
$$

it is convenient to write the cumulant-generating function

$$
\psi_S(\lambda)
=
\log E[e^{\lambda S}].
$$

Then

$$
P(S\ge t)
\le
\exp
\left[
-
\sup_{\lambda>0}
\{
\lambda t-\psi_S(\lambda)
\}
\right].
$$

The exponent is the convex conjugate of the log moment-generating function.

This is the core Chernoff method. Many concentration inequalities differ only in how the moment-generating function is bounded before the optimization over $\lambda$ is performed.

If the exact mgf is known, one can optimize it directly.

If only boundedness is known, Hoeffding's lemma supplies a universal quadratic mgf bound.

If variance and boundedness are known, Bernstein's method keeps more information and obtains a quadratic-to-linear transition in the exponent.

The geometry of the tail bound is therefore already contained in the log mgf.

## Hoeffding uses only the range

Suppose

$$
X_1,\ldots,X_n
$$

are independent and

$$
a_i
\le
X_i
\le
b_i
$$

almost surely.

Hoeffding's lemma states that for any centred bounded variable,

$$
E
\left[
e^{\lambda(X_i-E[X_i])}
\right]
\le
\exp
\left[
\frac{
\lambda^2(b_i-a_i)^2
}{
8
}
\right].
$$

Independence allows exponential moments to multiply. For

$$
S_n
=
\sum_{i=1}^n
\left(
X_i-E[X_i]
\right),
$$

we obtain

$$
E[e^{\lambda S_n}]
\le
\exp
\left[
\frac{
\lambda^2
}{
8
}
\sum_{i=1}^n
(b_i-a_i)^2
\right].
$$

Applying the Chernoff method and optimizing in $\lambda$ gives

$$
P(S_n\ge t)
\le
\exp
\left[
-
\frac{
2t^2
}{
\sum_{i=1}^n
(b_i-a_i)^2
}
\right].
$$

For independent variables in

$$
[0,1],
$$

the empirical mean

$$
\bar X_n
=
\frac1n
\sum_{i=1}^n X_i
$$

therefore satisfies

$$
P
\left(
\bar X_n-E[\bar X_n]
\ge
\varepsilon
\right)
\le
\exp
\left(
-2n\varepsilon^2
\right).
$$

The two-sided form is

$$
P
\left(
\left|
\bar X_n-E[\bar X_n]
\right|
\ge
\varepsilon
\right)
\le
2
\exp
\left(
-2n\varepsilon^2
\right).
$$

The inequality is powerful because it requires no knowledge of the distribution inside the interval. Bernoulli variables, continuous uniforms, bimodal bounded distributions and highly irregular bounded laws all receive the same guarantee.

That universality is also why the bound can be loose.

Hoeffding only knows that each observation lies somewhere in a range of width one. It does not know whether almost all mass is concentrated near zero, whether the variance is tiny, whether the distribution is symmetric or whether the exact mgf is available.

A distribution-free guarantee cannot exploit information it was never given.

## A rare Bernoulli event shows how much information Hoeffding discards

Let

$$
X_i
\sim
\operatorname{Bernoulli}(0.01)
$$

independently, with

$$
n=1000.
$$

The true mean is

$$
p=0.01.
$$

Suppose we observe an empirical rate of at least

$$
0.03.
$$

Equivalently,

$$
\sum_{i=1}^{1000}
X_i
\ge
30.
$$

The exact probability is the upper tail of a binomial distribution:

$$
P
\left[
\operatorname{Binomial}(1000,0.01)
\ge
30
\right]
\approx
2.06\times10^{-7}.
$$

This is a genuinely rare deviation.

Hoeffding sees only that each observation lies in

$$
[0,1]
$$

and that the empirical mean deviates upward by

$$
\varepsilon
=
0.03-0.01
=
0.02.
$$

Its one-sided bound is

$$
\exp
\left[
-2(1000)(0.02)^2
\right]
=
e^{-0.8}
\approx
0.449.
$$

The true probability is about

$$
2\times10^{-7}.
$$

Hoeffding's valid upper bound is approximately

$$
2.2\times10^6
$$

times larger than the actual probability.

Nothing is wrong with the theorem. It was asked to protect against every independent distribution on $[0,1]$ with the same mean, not to exploit the special low-variance structure of a Bernoulli variable with success probability 0.01.

This example is worth remembering because Hoeffding bounds are often reported numerically as though they were rough approximations to actual probabilities.

They are not.

A concentration inequality is a guarantee.

Its numerical tightness depends on how much information the theorem uses.

## Bernstein adds variance and becomes dramatically sharper

Suppose independent centred variables satisfy

$$
|X_i|
\le
M
$$

almost surely, and let

$$
v
=
\sum_{i=1}^n
\operatorname{Var}(X_i).
$$

A standard Bernstein inequality gives

$$
P
\left(
\sum_{i=1}^n
X_i
\ge
t
\right)
\le
\exp
\left[
-
\frac{
t^2
}{
2
\left(
v+\frac{Mt}{3}
\right)
}
\right].
$$

For the empirical mean of independent identically distributed variables with variance

$$
\sigma^2
$$

and centred deviations bounded by $M$,

$$
P
\left(
\bar X_n-\mu
\ge
\varepsilon
\right)
\le
\exp
\left[
-
\frac{
n\varepsilon^2
}{
2\sigma^2+\frac{2M\varepsilon}{3}
}
\right].
$$

The exponent has two regimes.

For small deviations,

$$
\varepsilon
\ll
\frac{
\sigma^2
}{
M
},
$$

the variance term dominates and the exponent behaves like

$$
-\frac{
n\varepsilon^2
}{
2\sigma^2
}.
$$

This is Gaussian-like concentration.

For large deviations, the linear term in $\varepsilon$ matters and the exponent behaves more like

$$
-\frac{
3n\varepsilon
}{
2M
}.
$$

The tail transitions from quadratic to approximately linear exponential decay.

Return to

$$
X_i
\sim
\operatorname{Bernoulli}(0.01).
$$

The variance is

$$
\sigma^2
=
p(1-p)
=
0.0099.
$$

For the centred Bernoulli deviation, we may use

$$
M\le1.
$$

At

$$
\varepsilon=0.02,
$$

Bernstein gives

$$
P
\left(
\bar X_n-p
\ge
0.02
\right)
\le
\exp
\left[
-
\frac{
1000(0.02)^2
}{
2(0.0099)
+
\frac{
2(0.02)
}{
3
}
}
\right].
$$

Numerically,

$$
P
\left(
\bar X_n\ge0.03
\right)
\le
5.71\times10^{-6}.
$$

Compare the three numbers:

| Quantity | Probability / upper bound |
| --- | ---: |
| Exact binomial probability | $2.06\times10^{-7}$ |
| Bernstein | $5.71\times10^{-6}$ |
| Hoeffding | $4.49\times10^{-1}$ |

Using one additional piece of information, the variance, improves the bound by almost five orders of magnitude.

The true probability is still smaller than Bernstein's guarantee by a factor of about 28.

There is more distributional information left to use.

## The Bernoulli Chernoff bound recovers the correct large-deviation geometry

For Bernoulli variables, the exact moment-generating function is known.

If

$$
X\sim\operatorname{Bernoulli}(p),
$$

then

$$
E[e^{\lambda X}]
=
1-p+pe^\lambda.
$$

For

$$
S_n
=
\sum_{i=1}^n X_i,
$$

independence gives

$$
E[e^{\lambda S_n}]
=
\left(
1-p+pe^\lambda
\right)^n.
$$

The Chernoff bound for

$$
S_n\ge nq
$$

with

$$
q>p
$$

is

$$
P(S_n\ge nq)
\le
\inf_{\lambda>0}
\exp
\left[
-n\lambda q
\right]
\left(
1-p+pe^\lambda
\right)^n.
$$

Optimizing yields

$$
P
\left(
\bar X_n\ge q
\right)
\le
\exp
\left[
-nD(q\|p)
\right],
$$

where

$$
D(q\|p)
=
q\log
\frac{
q
}{
p
}
+
(1-q)
\log
\frac{
1-q
}{
1-p
}
$$

is the Bernoulli Kullback-Leibler divergence.

For

$$
p=0.01,
\qquad
q=0.03,
$$

we have

$$
D(0.03\|0.01)
\approx
0.0131618.
$$

Therefore,

$$
P
\left(
\bar X_{1000}
\ge
0.03
\right)
\le
\exp
\left[
-1000(0.0131618)
\right]
\approx
1.92\times10^{-6}.
$$

The comparison is now:

| Quantity | Probability / upper bound |
| --- | ---: |
| Exact binomial probability | $2.06\times10^{-7}$ |
| KL-Chernoff | $1.92\times10^{-6}$ |
| Bernstein | $5.71\times10^{-6}$ |
| Hoeffding | $4.49\times10^{-1}$ |

The Chernoff exponent is still not exact at finite $n$, but it has the correct large-deviation rate.

For fixed

$$
q>p,
$$

the binomial tail behaves exponentially like

$$
P
\left(
\bar X_n\ge q
\right)
\approx
e^{-nD(q\|p)}
$$

up to subexponential factors under standard large-deviation asymptotics.

The relative entropy

$$
D(q\|p)
$$

is not an arbitrary algebraic artifact. It is the natural cost of forcing the empirical Bernoulli rate away from its true value.

This is the bridge between concentration inequalities and large-deviation theory.

## Hoeffding, Bernstein and Chernoff use different information

The rare-Bernoulli example can be summarized conceptually.

Hoeffding uses:

$$
0\le X_i\le1.
$$

Bernstein uses:

$$
0\le X_i\le1
$$

and

$$
\operatorname{Var}(X_i)=0.0099.
$$

The Bernoulli Chernoff bound uses the exact mgf,

$$
E[e^{\lambda X}]
=
1-p+pe^\lambda.
$$

The bounds improve because the assumptions become more informative.

This gives a general rule for concentration arguments:

> Use the weakest inequality that still exploits the strongest information you can justify.

A variance-sensitive bound is preferable to a range-only bound when the variance is known or estimable reliably.

An exact Chernoff calculation is preferable when the distributional form is credible.

A distribution-free bound remains valuable when those stronger assumptions would be questionable.

Sharpness is not free.

It comes from assumptions.

## Sub-Gaussian variables generalize Gaussian-type concentration

A centred random variable $X$ is called sub-Gaussian if there exists a scale parameter

$$
\sigma^2
$$

such that

$$
E[e^{\lambda X}]
\le
\exp
\left(
\frac{
\sigma^2\lambda^2
}{
2
}
\right)
$$

for every

$$
\lambda\in\mathbb R.
$$

Applying the Chernoff method gives

$$
P(X\ge t)
\le
\exp
\left(
-\frac{
t^2
}{
2\sigma^2
}
\right).
$$

A standard normal variable has this form exactly with

$$
\sigma^2=1.
$$

Bounded centred random variables are sub-Gaussian with an appropriate proxy variance because of Hoeffding's lemma.

If independent centred variables

$$
X_i
$$

are sub-Gaussian with parameters

$$
\sigma_i^2,
$$

then their sum is sub-Gaussian with parameter

$$
\sum_i
\sigma_i^2.
$$

Indeed,

$$
E
\left[
\exp
\left(
\lambda
\sum_i X_i
\right)
\right]
=
\prod_i
E[e^{\lambda X_i}]
$$

and therefore

$$
E
\left[
e^{\lambda\sum_i X_i}
\right]
\le
\exp
\left[
\frac{
\lambda^2
}{
2
}
\sum_i
\sigma_i^2
\right].
$$

This closure under sums explains why sub-Gaussian variables are so central in high-dimensional probability.

They behave like Gaussian variables at the level of tails even when their exact distributions are not Gaussian.

The terminology should still be used carefully. The parameter

$$
\sigma^2
$$

is often a variance proxy rather than the actual variance.

A random variable can have

$$
\operatorname{Var}(X)
<
\sigma^2
$$

while the mgf requires the larger scale for a valid global bound.

## Sub-exponential variables explain the Bernstein shape

Some variables have tails heavier than Gaussian but still possess exponential moments near zero.

A centred variable is commonly called sub-exponential if its mgf satisfies a bound of the form

$$
E[e^{\lambda X}]
\le
\exp
\left(
\frac{
\nu^2\lambda^2
}{
2
}
\right)
$$

for

$$
|\lambda|<\frac1b.
$$

The restriction on $\lambda$ changes the optimized tail.

One obtains a bound of the schematic form

$$
P(|X|\ge t)
\le
2
\exp
\left[
-c
\min
\left(
\frac{
t^2
}{
\nu^2
},
\frac{
t
}{
b
}
\right)
\right].
$$

For moderate deviations, the quadratic term dominates and the behaviour is Gaussian-like.

For large deviations, the linear term dominates.

This is the same two-regime geometry visible in Bernstein's inequality.

The terminology can be confusing because "sub-exponential" does not mean slower than every exponential tail. It refers to a class whose tails are controlled by exponential-type decay and whose sums satisfy Bernstein-style concentration.

Products of sub-Gaussian variables often become sub-exponential.

This matters in covariance estimation. Even when coordinates are sub-Gaussian, products such as

$$
X_iX_j
$$

have heavier tails, so concentration of sample covariances naturally uses sub-exponential tools rather than the same sub-Gaussian inequality applied blindly.

## Maxima introduce logarithms through the union bound

Suppose

$$
X_1,\ldots,X_m
$$

are centred sub-Gaussian variables with common parameter

$$
\sigma^2.
$$

For any

$$
t>0,
$$

the union bound gives

$$
P
\left(
\max_{1\le j\le m}
X_j
\ge t
\right)
\le
\sum_{j=1}^m
P(X_j\ge t).
$$

Therefore,

$$
P
\left(
\max_j X_j
\ge t
\right)
\le
m
\exp
\left(
-\frac{
t^2
}{
2\sigma^2
}
\right).
$$

To make this probability at most

$$
\delta,
$$

choose

$$
t
=
\sigma
\sqrt{
2\log
\frac{
m
}{
\delta
}
}.
$$

The maximum therefore grows on the scale

$$
\sqrt{\log m}
$$

rather than linearly with the number of coordinates.

This simple calculation appears throughout high-dimensional statistics.

If we inspect ten thousand noisy features, some large-looking feature is expected purely because there are many opportunities for fluctuation.

The logarithm enters because the tail probability for one coordinate decays exponentially while the number of opportunities grows multiplicatively.

The union bound can be loose when variables are strongly dependent.

Its value is that it requires no dependence assumptions at all.

More refined tools, including Gaussian comparison inequalities, chaining and entropy methods, exploit dependence and metric structure to obtain sharper bounds for suprema.

The elementary union-bound calculation already reveals the basic trade-off between tail decay and multiplicity.

## Bounded differences control functions, not only sums

Concentration is not limited to averages.

Let

$$
X_1,\ldots,X_n
$$

be independent, and consider a function

$$
f(X_1,\ldots,X_n).
$$

Suppose changing only coordinate $i$ can alter the function by at most

$$
c_i.
$$

Formally, for input vectors differing only in coordinate $i$,

$$
|f(x)-f(x')|
\le
c_i.
$$

McDiarmid's bounded-difference inequality gives

$$
P
\left(
f(X)
-
E[f(X)]
\ge
t
\right)
\le
\exp
\left[
-
\frac{
2t^2
}{
\sum_{i=1}^n
c_i^2
}
\right].
$$

This looks like Hoeffding because it is a functional extension of the same bounded-difference geometry.

For the sample mean of $[0,1]$ variables,

$$
f(X_1,\ldots,X_n)
=
\frac1n
\sum_i X_i,
$$

changing one observation can change the mean by at most

$$
c_i
=
\frac1n.
$$

Then

$$
\sum_i c_i^2
=
\frac1n,
$$

and McDiarmid recovers the Hoeffding-type bound

$$
P
\left(
\bar X-E[\bar X]\ge t
\right)
\le
e^{-2nt^2}.
$$

The functional form is much broader.

If one observation changes a statistic only slightly, the statistic can concentrate even when its exact distribution is difficult to compute.

This idea underlies algorithmic stability arguments, random graph functionals, bounded-loss empirical processes and many randomized combinatorial quantities.

Again, the sensitivity constants

$$
c_i
$$

are part of the model.

A statistic dominated by one observation has weak bounded-difference concentration.

A statistic averaging many small influences can concentrate strongly.

## Independence can be weakened through martingale concentration

Real data are often sequential or dependent.

The simplest concentration inequalities assume independence because exponential moments factor cleanly. Dependence does not make concentration impossible, but the structure has to be replaced by something else.

Let

$$
M_0,M_1,\ldots,M_n
$$

be a martingale with bounded increments,

$$
|M_k-M_{k-1}|
\le
c_k
$$

almost surely.

Azuma-Hoeffding gives

$$
P
\left(
M_n-M_0
\ge
t
\right)
\le
\exp
\left[
-
\frac{
t^2
}{
2
\sum_{k=1}^n
c_k^2
}
\right].
$$

The role formerly played by independent summands is now played by martingale differences.

This is useful when observations arrive adaptively but conditional expectations remain controlled.

Freedman's inequality adds a variance process and is the martingale analogue of Bernstein-style concentration. Roughly, if increments are bounded and predictable quadratic variation is small, deviations can be bounded more sharply than Azuma's range-only inequality.

The pattern is the same as before:

- bounded increments give a distribution-free guarantee;
- conditional variance gives a sharper guarantee;
- richer process information can sharpen it further.

Sequential dependence therefore changes the technical machinery without changing the underlying logic.

## Concentration and the central limit theorem answer different questions

The central limit theorem says that for iid variables with finite variance,

$$
\frac{
\sqrt n(\bar X_n-\mu)
}{
\sigma
}
\Rightarrow
N(0,1).
$$

This gives an asymptotic approximation to the distribution of fluctuations on the scale

$$
n^{-1/2}.
$$

Concentration inequalities provide finite-sample upper bounds, often valid uniformly over classes of distributions.

Neither dominates the other.

For moderate sample sizes and ordinary deviations, a normal approximation can be far sharper numerically than Hoeffding.

For a bounded variable with unknown distribution, Hoeffding gives a rigorous guarantee where a Gaussian approximation may not be justified.

For rare deviations that move farther from the mean as $n$ grows, large-deviation theory becomes more natural than the ordinary CLT.

The scales differ.

A fixed standardized deviation,

$$
\bar X_n-\mu
=
O(n^{-1/2}),
$$

belongs to the CLT regime.

A fixed absolute deviation,

$$
\bar X_n-\mu
\ge
\varepsilon
$$

for constant

$$
\varepsilon>0,
$$

becomes exponentially rare in $n$ and belongs naturally to concentration and large-deviation analysis.

This is why the binomial probability

$$
P(\bar X_n\ge0.03)
$$

with true

$$
p=0.01
$$

has an exponent involving

$$
nD(0.03\|0.01).
$$

The deviation is not shrinking with

$$
n.
$$

It is a large-deviation event.

Using a CLT formula outside its natural scale can be inaccurate even when $n$ is large.

## Sample complexity formulas are guarantees, not forecasts

Hoeffding's inequality can be inverted.

For independent

$$
X_i\in[0,1],
$$

we want

$$
P
\left(
|\bar X_n-\mu|
\ge
\varepsilon
\right)
\le
\delta.
$$

It is sufficient that

$$
2e^{-2n\varepsilon^2}
\le
\delta.
$$

Solving for

$$
n
$$

gives

$$
n
\ge
\frac{
\log(2/\delta)
}{
2\varepsilon^2
}.
$$

This looks like a sample-size formula.

It is a worst-case sufficient sample size over all distributions supported on

$$
[0,1].
$$

It should not be interpreted as the actual sample size required for a particular low-variance distribution.

Suppose

$$
\varepsilon=0.02
$$

and

$$
\delta=0.05.
$$

Hoeffding requires

$$
n
\ge
\frac{
\log 40
}{
2(0.02)^2
}
\approx
4612.
$$

If the underlying Bernoulli probability is only 0.01, variance-sensitive calculations can require substantially fewer observations for some one-sided questions.

The distinction is the same one that appeared in the rare-event example.

Distribution-free robustness costs sharpness.

When a sample-size argument is based on a concentration inequality, the assumptions and worst-case nature of the guarantee should be stated.

Otherwise a conservative theorem can be mistaken for a data-generating prediction.

## Heavy tails can break the usual concentration picture

Exponential concentration depends on exponential moments or comparable tail control.

If the data are heavy-tailed enough, the mgf may not exist for any positive

$$
\lambda.
$$

Then the Chernoff machinery cannot even begin.

A Pareto-distributed variable can have a finite mean and infinite variance.

For such variables, the ordinary sample mean can be dominated by rare extreme observations, and sub-Gaussian confidence widths are unjustified.

This does not mean finite-sample inference is impossible.

Robust estimators can restore concentration under weaker moment assumptions.

Median-of-means procedures split observations into groups, compute the mean within each group, and take the median of those means.

Under finite variance, such estimators can achieve sub-Gaussian-type deviation guarantees up to constants without requiring the original observations themselves to be sub-Gaussian.

Catoni-type estimators use robust influence functions to control the effect of extremes.

Truncation and winsorization can also produce concentration at the cost of bias.

The lesson is broader than the specific methods.

Concentration belongs to an estimator-plus-assumption pair.

If ordinary averages do not concentrate under the available tail assumptions, one can sometimes redesign the estimator rather than pretending the assumptions are stronger.

## A probability bound should be matched to the information actually available

The rare-Bernoulli example provides a compact hierarchy.

The exact probability was

$$
2.06\times10^{-7}.
$$

Hoeffding knew only the range and returned

$$
0.449.
$$

Bernstein also knew the variance and returned

$$
5.71\times10^{-6}.
$$

The Bernoulli Chernoff bound knew the exact mgf and returned

$$
1.92\times10^{-6}.
$$

Every bound was correct.

Their usefulness differed by more than five orders of magnitude.

This is the central practical lesson of concentration inequalities.

A theorem is not useful merely because it is valid.

It should exploit the information one can justify and no more.

If only boundedness is credible, Hoeffding is powerful because it makes almost no distributional assumptions.

If variance is reliably known or estimable, Bernstein can be far better.

If the exact model is justified, a distribution-specific Chernoff bound or exact calculation can be sharper again.

If tails are heavy, redesigning the estimator can be more honest than invoking an exponential bound that does not apply.

If data are dependent, martingale, mixing or process-specific concentration should replace iid arguments.

The same discipline applies when concentration is used inside more elaborate mathematics. Generalization bounds, random matrix theory, compressed sensing, online learning and high-dimensional statistics all rely on concentration. Their final guarantees are only as meaningful as the assumptions behind the tail bounds they inherit.

The law of large numbers tells us that randomness averages out.

Concentration inequalities quantify how quickly that happens and how unlikely it is to fail by a specified amount.

The answer depends on what we know about the randomness.

## References

Bernstein, S. N. (1924). On a modification of Chebyshev's inequality and of the error formula of Laplace. *Annals of the Scientific Institute Sav. Ukraine, Sect. Math.*, 1, 38–49.

Boucheron, S., Lugosi, G., & Massart, P. (2013). *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press.

Chernoff, H. (1952). A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. *The Annals of Mathematical Statistics*, 23(4), 493–507.

Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13–30.

McDiarmid, C. (1989). On the method of bounded differences. In *Surveys in Combinatorics*. Cambridge University Press.

Petrov, V. V. (1995). *Limit Theorems of Probability Theory*. Oxford University Press.

Vershynin, R. (2018). *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press.

Wainwright, M. J. (2019). *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press.
