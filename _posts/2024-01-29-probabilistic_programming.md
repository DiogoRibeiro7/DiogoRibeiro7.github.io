---
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-01-29'
excerpt: Explore Markov Chain Monte Carlo (MCMC) methods, specifically the Metropolis algorithm, and learn how to perform Bayesian inference through Python code.
header:
  image: /assets/images/data_science_2.jpg
  og_image: /assets/images/data_science_9.jpg
  overlay_image: /assets/images/data_science_2.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.jpg
  twitter_image: /assets/images/data_science_9.jpg
keywords:
- MCMC
- Metropolis algorithm
- Bayesian inference
- Markov Chain Monte Carlo
- probabilistic programming
- Bayesian statistics
- statistical modeling
- Python code for MCMC
- data science
- machine learning
- python
seo_description: A practical explanation of MCMC and the Metropolis algorithm, focusing on Bayesian inference with Python code examples to make the concepts accessible.
seo_title: 'Demystifying MCMC: A Hands-On Guide to Bayesian Inference'
seo_type: article
subtitle: Understanding the Metropolis Algorithm Through Code
tags:
- Data Science
- Mathematical Modeling
- Statistical Methods
- Machine Learning
- Statistical Analysis
- Probability
- Probabilistic Programming
- Bayesian Statistics
- python
title: 'Demystifying MCMC: A Practical Guide to Bayesian Inference'
---

In my talks about probabilistic programming and Bayesian statistics, I often keep the explanation of inference high-level, treating it as a sort of "black box". Probabilistic programming's advantage is that it doesn't require deep knowledge of the inference mechanism to construct models, although understanding it is beneficial.

Once, while introducing a Bayesian model to my CEO, who is new to Bayesian statistics, he questioned the inference process, the part I usually simplify. He asked, "How does the inference work? How do we obtain samples from the posterior?"

I could have simply said, "MCMC generates samples from the posterior distribution by creating a reversible Markov-chain with the target posterior distribution as its equilibrium." But is such a technical explanation helpful? My criticism of math and stats education is its focus on complex math rather than the underlying intuition, which is often simpler. I had to spend hours deciphering these concepts myself.

This blog post aims to elucidate the intuition behind MCMC sampling, specifically the random-walk Metropolis algorithm, using code rather than formulas.

Let's start by examining Bayes' formula:

$$
P(\theta \mid X) = \frac{P(X \mid \theta) \cdot P(\theta)}{P(X)}
$$

Where:

- $$P(\theta \mid X)$$ is the posterior probability of the parameters ($$\theta$$) given the data ($$X$$).
- $$P(X \mid \theta)$$ is the likelihood of the data given the parameters.
- $$P(\theta)$$ is the prior probability of the parameters.
- $$P(X)$$ is the probability of the data (also known as the evidence).

The formula calculates the probability of our model parameters given our data. We multiply the prior (our initial belief before data) with the likelihood (our assumption about data distribution). The numerator is straightforward, but the denominator, the evidence, is challenging as it requires integrating over all possible parameter values.

Since direct computation is difficult, we consider approximations. One method is to sample from the posterior distribution using Monte Carlo methods, but this requires solving and inverting Bayes' formula, a complex task.

Alternatively, we might construct an ergodic, reversible Markov chain whose equilibrium distribution matches our posterior. This process is simplified using Markov chain Monte Carlo (MCMC) algorithms.

For our example, we'll use Python libraries like numpy and scipy. Our goal is to estimate the posterior of the mean (mu), assuming a known standard deviation, from data points drawn from a normal distribution centered at zero.

Here's our Python setup:

```python
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')
np.random.seed(123)

# Generating sample data
data = np.random.randn(20)
ax = plt.subplot()
sns.distplot(data, kde=False, ax=ax)
_ = ax.set(title='Histogram of observed data', xlabel='x', ylabel='# observations')
```

We define our model as a normal distribution. Conveniently, we can compute the posterior analytically in this case:

```python
def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

x = np.linspace(-1, 1, 500)
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.)
ax.plot(x, posterior_analytical)
ax.set(xlabel='mu', ylabel='belief', title='Analytical posterior')
sns.despine()
```

Next, we implement the MCMC sampler. We start with an initial mu value, propose a new mu, and decide whether to accept it based on the likelihood of the data given the proposed mu:

```python
def sampler(data, samples=4, mu_init=.5, proposal_width=.5, plot=False, mu_prior_mu=0, mu_prior_sd=1.):
    mu_current = mu_init
    posterior = [mu_current]
    for i in range(samples):
        mu_proposal = norm(mu_current, proposal_width).rvs()

        likelihood_current = norm(mu_current, 1).pdf(data).prod()
        likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()

        prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
        prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)

        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal

        p_accept = p_proposal / p_current

        accept = np.random.rand() < p_accept

        if accept:
            mu_current = mu_proposal
        posterior.append(mu_current)
        
    return np.array(posterior)
```

This algorithm navigates towards more probable values of mu but sometimes accepts less likely values, ensuring exploration of the parameter space. The sampler generates samples representing the posterior distribution of the model, confirmed by comparing the histogram of these samples to the analytically computed posterior.

In conclusion, while this post simplifies some aspects, it aims to clarify the concept of MCMC and the Metropolis sampler. This foundation should help in understanding more technical discussions of MCMC algorithms like Hamiltonian Monte Carlo, which function similarly but with more sophisticated proposal mechanisms.
