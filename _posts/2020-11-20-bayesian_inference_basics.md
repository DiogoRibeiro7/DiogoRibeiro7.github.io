---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-11-20'
excerpt: Explore the fundamentals of Bayesian inference and how prior beliefs combine
  with data to form posterior conclusions.
header:
  image: /assets/images/data_science_12.jpg
  og_image: /assets/images/data_science_12.jpg
  overlay_image: /assets/images/data_science_12.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_12.jpg
  twitter_image: /assets/images/data_science_12.jpg
keywords:
- Bayesian statistics
- Priors
- Posterior distributions
- Data science
seo_description: An overview of Bayesian inference, demonstrating how to update prior
  beliefs with new evidence to make data-driven decisions.
seo_title: Bayesian Inference Explained
seo_type: article
summary: Learn how Bayesian inference updates prior beliefs into posterior distributions,
  providing a flexible framework for reasoning under uncertainty.
tags:
- Bayesian Statistics
- Statistical Modeling
- Statistics
title: Bayesian Inference Explained
---

Bayesian inference offers a powerful perspective on probability, treating unknown quantities as distributions that update when new evidence appears.

## Priors and Posteriors

The process begins with a **prior distribution** that captures our initial beliefs about a parameter. After observing data, we apply Bayes' theorem to obtain the **posterior distribution**, reflecting how our beliefs should change:

$$
p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)},
\qquad p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta .
$$

The three pieces have distinct jobs. The likelihood $p(y \mid \theta)$ says how plausible the observed data is under each candidate parameter value. The prior $p(\theta)$ encodes what was believed beforehand. The denominator $p(y)$, the marginal likelihood, is a normalising constant that makes the posterior integrate to one.

Because the denominator does not depend on $\theta$, the relationship is often written as a proportionality, which is all you need to identify the posterior's shape:

$$
p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta) .
$$

This is the deepest difference from frequentist practice. A frequentist treats $\theta$ as a fixed unknown constant and asks how the data would vary across hypothetical repetitions. A Bayesian treats $\theta$ as a random variable with a distribution and asks what the observed data implies about it. The distinction is not merely philosophical: it changes what the resulting intervals mean.

## A Worked Example

Suppose you are estimating a conversion rate $\theta$. The natural likelihood for $k$ successes in $n$ trials is binomial, and the conjugate prior is the Beta distribution:

$$
\theta \sim \mathrm{Beta}(\alpha, \beta), \qquad k \mid \theta \sim \mathrm{Binomial}(n, \theta) .
$$

Multiplying likelihood by prior and collecting terms in $\theta$ gives a posterior of the same family:

$$
\theta \mid k \sim \mathrm{Beta}(\alpha + k,\; \beta + n - k) .
$$

The update rule is simply to add successes to the first parameter and failures to the second. This makes the role of the prior concrete: $\mathrm{Beta}(1,1)$ is uniform and contributes nothing, while $\mathrm{Beta}(30,70)$ carries about as much weight as having already seen 100 observations at a 30% rate.

```python
import numpy as np
from scipy import stats

alpha_prior, beta_prior = 1, 1     # uniform prior
k, n = 12, 100                     # 12 conversions out of 100

post = stats.beta(alpha_prior + k, beta_prior + n - k)

print(f"posterior mean : {post.mean():.4f}")
print(f"95% credible   : {post.ppf(0.025):.4f} to {post.ppf(0.975):.4f}")
print(f"P(theta > 0.10): {1 - post.cdf(0.10):.3f}")
```

That last line is the practical appeal. "There is an 81% probability the rate exceeds 10%" is a direct statement about the parameter, and it is the sentence most people mistakenly believe a frequentist confidence interval provides.

## Credible Intervals Are Not Confidence Intervals

A 95% **credible interval** contains 95% of the posterior probability mass, so it genuinely means there is a 95% probability the parameter lies inside, given the model and prior.

A 95% **confidence interval** makes no such claim about the specific interval you computed. It says that the *procedure*, repeated over many hypothetical datasets, produces intervals containing the true value 95% of the time. For any particular interval, the parameter is either inside or not.

The two often land in similar places numerically, especially with weak priors and plenty of data, which conceals how different the underlying statements are.

## Choosing Priors

The prior is where most objections to Bayesian methods land, and where the most care is warranted.

A **weakly informative** prior rules out absurd values while letting the data dominate. For a log-odds parameter, a normal prior with standard deviation 2.5 permits any plausible effect size but excludes values corresponding to probabilities indistinguishable from 0 or 1.

A **flat** prior is not automatically neutral. Uniformity is not preserved under reparameterisation: a prior that is flat in $\theta$ is not flat in $\log \theta$ or in the odds. Claiming to have "used no prior" is usually a claim about one particular parameterisation.

An **informative** prior encodes genuine external knowledge, such as previous experiments or physical constraints. This is a feature when the knowledge is real and a liability when it is wishful.

The honest response to prior sensitivity is to test it. Refit under several defensible priors and report whether the conclusion moves. If it does, the data is not carrying the argument and you should say so.

## When Conjugacy Runs Out

Conjugate pairs like Beta-Binomial and Normal-Normal are convenient but rare. For realistic models the posterior has no closed form, and the integral in the denominator is intractable.

Markov Chain Monte Carlo solves this by constructing a Markov chain whose stationary distribution is the posterior, then drawing correlated samples from it. Modern samplers such as the No-U-Turn Sampler use gradient information to explore high-dimensional posteriors efficiently, and tools like PyMC, Stan, and NumPyro handle the machinery.

Sampling introduces its own diagnostics. The $\hat{R}$ statistic compares variance within and between chains and should sit very close to 1; effective sample size measures how much independent information the correlated draws carry; divergent transitions in Hamiltonian samplers signal geometry the sampler could not traverse and must not be ignored. A posterior summary from an unconverged chain is not an approximation of the right answer, it is an arbitrary one.

## Why Use Bayesian Methods?

Bayesian techniques are particularly useful when data is scarce or when incorporating domain knowledge is essential. They provide a coherent approach to uncertainty that can complement or outperform classical methods in many situations.

The strengths worth naming are these. Uncertainty propagates automatically: any function of the parameters gets a full distribution rather than a point estimate plus a delta-method approximation. Hierarchical models handle grouped data naturally, letting groups with little data borrow strength from the population. Sequential updating is built in, since today's posterior is tomorrow's prior. And there is no reliance on asymptotic approximations, which matters when samples are small.

The costs are equally real. Computation is orders of magnitude more expensive than fitting a closed-form estimator. Results depend on modelling choices that must be defended rather than assumed away. And the diagnostics require genuine attention, because a sampler can fail quietly.

Neither framework is universally correct. The useful question is which one answers the question you actually have, with assumptions you are willing to state out loud.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.
- Kruschke, J. K. (2014). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
