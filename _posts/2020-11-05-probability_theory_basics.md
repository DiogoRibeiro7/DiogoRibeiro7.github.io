---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-11-05'
excerpt: An introduction to probability theory concepts every data scientist should
  know.
header:
  image: /assets/images/data_science_10.avif
  og_image: /assets/images/data_science_10.avif
  overlay_image: /assets/images/data_science_10.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_10.avif
  twitter_image: /assets/images/data_science_10.avif
keywords:
- Probability theory
- Random variables
- Distributions
- Data science
seo_description: Learn the core principles of probability theory, from random variables
  to common distributions, with practical examples for data science.
seo_title: Probability Theory Basics for Data Science
seo_type: article
summary: This post reviews essential probability concepts like random variables, expectation,
  and common distributions, illustrating how they underpin data science workflows.
tags:
- Probability
- Statistics
- Data Science
title: Probability Theory Basics for Data Science
---

Probability theory provides the mathematical foundation for modeling uncertainty. By understanding random variables and probability distributions, data scientists can quantify risks and make informed decisions.

## The Sample Space and Its Events

Every probabilistic argument starts with a sample space $\Omega$, the set of all outcomes an experiment can produce. A single roll of a die has $\Omega = \{1,2,3,4,5,6\}$; a request to a web service might have $\Omega = \{\text{success}, \text{timeout}, \text{error}\}$. An *event* is any subset of $\Omega$, and a probability measure $P$ assigns each event a number in $[0,1]$ subject to three constraints:

$$
P(\Omega) = 1, \qquad P(A) \ge 0, \qquad
P\left(\bigcup_{i} A_i\right) = \sum_i P(A_i) \ \text{for disjoint } A_i .
$$

These axioms look modest, but everything else follows from them: the complement rule $P(A^c) = 1 - P(A)$, the inclusion-exclusion formula $P(A \cup B) = P(A) + P(B) - P(A \cap B)$, and the whole apparatus of conditional probability.

The definition that does the most work in practice is conditioning. Given that $B$ has occurred, the probability of $A$ becomes

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0 .
$$

Two events are independent exactly when conditioning changes nothing, that is when $P(A \mid B) = P(A)$, equivalently $P(A \cap B) = P(A)P(B)$. Independence is an assumption you impose on a model, not a property you can read off a dataset, and assuming it where it does not hold is one of the most common sources of overconfident conclusions.

## Random Variables and Distributions

A random variable assigns numerical values to outcomes in a sample space. Formally it is a function $X: \Omega \to \mathbb{R}$, which lets us replace awkward talk about events with ordinary arithmetic on numbers.

Discrete random variables are described by a probability mass function $p(x) = P(X = x)$. Continuous ones are described by a density $f(x)$, where probability is recovered by integration:

$$
P(a \le X \le b) = \int_a^b f(x)\,dx .
$$

A density is not a probability. It can exceed 1, and $P(X = x) = 0$ for every single point in the continuous case. Only areas under the curve are meaningful.

Three distributions cover a surprising amount of applied work:

- **Binomial.** The number of successes in $n$ independent trials each with success probability $p$: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$. Conversion counts in an A/B test are binomial.
- **Poisson.** The number of events in a fixed interval when events occur independently at constant average rate $\lambda$: $P(X = k) = e^{-\lambda}\lambda^k / k!$. Arrivals at a queue, defects per batch, and alerts per hour are all naturally Poisson.
- **Normal.** The bell curve, with density $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$. It earns its ubiquity from the Central Limit Theorem rather than from any claim that real data is normal.

Knowing these distributions helps in selecting appropriate models and estimating parameters, but the choice should be driven by the mechanism generating the data. A Poisson model is appropriate because events are independent and rare, not because the histogram happens to look skewed.

## Expectation and Variance

Two fundamental measures of a random variable are its **expected value** and **variance**. The expected value represents the long-run average:

$$
E[X] = \sum_x x\,p(x) \quad \text{(discrete)}, \qquad
E[X] = \int_{-\infty}^{\infty} x f(x)\,dx \quad \text{(continuous)} .
$$

The variance measures how spread out the outcomes are, defined as the expected squared deviation from the mean:

$$
\operatorname{Var}(X) = E\big[(X - E[X])^2\big] = E[X^2] - (E[X])^2 .
$$

Two properties make these tractable. Expectation is linear without qualification, so $E[aX + bY] = aE[X] + bE[Y]$ whether or not $X$ and $Y$ are related. Variance is not linear: $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$, and

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y),
$$

so the familiar rule that variances add is a statement about *uncorrelated* variables, not a general fact. Forgetting the covariance term is why risk models built on correlated assets understate their exposure.

## Why the Normal Distribution Keeps Appearing

The Central Limit Theorem explains why so many aggregate quantities look normal even when the underlying data does not. If $X_1, \dots, X_n$ are independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$, then the standardised sample mean converges in distribution to a standard normal:

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) .
$$

Note what the theorem does and does not promise. It describes the behaviour of the *mean*, not of individual observations, and it requires finite variance. Heavy-tailed data such as income, city sizes, or financial returns may violate that condition badly enough that the approximation is useless at any realistic sample size.

The $\sqrt{n}$ in the denominator is the practical payoff: the standard error of a sample mean shrinks in proportion to the square root of the sample size. Quadrupling your data halves your uncertainty, which sets realistic expectations about what more data can buy you.

## A Short Simulation

Simulation is often the fastest way to check a probabilistic claim you are unsure about. The following demonstrates the Central Limit Theorem using an exponential distribution, which is strongly skewed:

```python
import numpy as np

rng = np.random.default_rng(42)

# Exponential population: heavily skewed, mean = 1, variance = 1
n, trials = 30, 20_000
samples = rng.exponential(scale=1.0, size=(trials, n))
means = samples.mean(axis=1)

print(f"population mean     : 1.000")
print(f"mean of sample means: {means.mean():.3f}")
print(f"predicted std error : {1.0 / np.sqrt(n):.3f}")
print(f"observed std error  : {means.std(ddof=1):.3f}")
```

The sample means cluster tightly around the population mean with a spread close to $\sigma/\sqrt{n}$, even though single draws from the population look nothing like a bell curve. Raising `n` tightens the agreement; lowering it to 2 or 3 shows the approximation breaking down.

## Where Intuition Fails

Probability is notorious for producing answers that feel wrong. Two failures are worth internalising because they recur constantly in applied work.

The first is ignoring base rates. A test that is 99% accurate for a condition affecting 1 in 10,000 people still produces mostly false positives, because the 1% error rate applies to a vastly larger healthy population than the 0.01% who are ill. Bayes' theorem makes this precise:

$$
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)} .
$$

The second is treating a conjunction as more likely than its parts. $P(A \cap B)$ can never exceed $\min(P(A), P(B))$, yet a detailed, plausible-sounding scenario routinely strikes people as more probable than the vaguer claim it entails.

Both errors share a root cause: reasoning from how representative a story feels rather than from how much probability mass it can actually claim.

## Bringing It Together

Mastering probability theory enables data scientists to better interpret model outputs and reason about uncertainty in real-world applications. A p-value, a confidence interval, and a posterior distribution are all statements about probability measures, and none of them can be read correctly without knowing what the underlying random variable is and what assumptions were imposed on it.

The practical discipline is to state the sample space, name the distribution and justify it mechanically, and check whether the independence and finite-variance assumptions your method depends on actually hold. Most statistical errors in production are not arithmetic mistakes; they are assumptions that were never examined.

## References

- Ross, S. (2019). *A First Course in Probability* (10th ed.). Pearson.
- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
- Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press.
- Kahneman, D., & Tversky, A. (1973). On the psychology of prediction. *Psychological Review*, 80(4), 237-251.
