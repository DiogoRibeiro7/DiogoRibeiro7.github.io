---
permalink: '/statistics/cluster_randomised_experiments_stores_not_customers/'
title: 'Cluster-Randomised Experiments: When You Randomise Stores and Analyse Customers'
categories:
- Statistics
tags:
- Experimental Design
- A/B Testing
- Hypothesis Testing
- Statistics
author_profile: false
seo_title: 'Cluster-Randomised Experiments and the Intraclass Correlation'
seo_description: 'When treatment is assigned to stores, regions or teams and outcomes are measured on customers, a test that counts customers as independent declares differences that do not exist. A simulation shows false positive rates above 50 percent, the design effect that explains them, and why more clusters beat more customers.'
excerpt: >-
  Twenty stores are randomised, ten to each arm, and the 4,000 customers
  are compared with a t-test. With no true effect and only 5 percent of
  the outcome variance between stores, the test finds a significant
  difference in 52 percent of experiments.
summary: >-
  Why customers in the same store are not independent observations, the
  intraclass correlation and the design effect, a simulation of A/A
  experiments showing the customer-level test failing at any positive
  correlation and the store-level test holding its level, the standard
  error the customer-level test reports against the true spread of the
  estimate, why power depends on the number of clusters far more than on
  customers per cluster, the floor on precision that no amount of
  customers removes, and what a small number of clusters can and cannot
  establish.
keywords:
  - cluster randomisation
  - intraclass correlation
  - design effect
  - experimental design
  - A/B testing
  - mixed models
  - statistical power
classes: wide
date: '2026-04-02'
why_this_exists: >-
  Many treatments can only be assigned to whole units, a store, a sales
  team, a region, a school, and the temptation is to analyse the
  individuals inside them because there are so many. This post measures
  what that does to the false positive rate and to power, so that the
  design and the analysis match the unit that was randomised.
evidence: >-
  Simulated experiments with 8 to 200 stores randomised 50/50 and 40 to
  2,000 customers per store, an outcome with 10 units of standard
  deviation split between and within stores by an intraclass correlation
  from 0 to 0.10, with no true effect and with an effect of 2; 2,000
  replications per cell.
methodology: >-
  Compares the false positive rate of a customer-level two-sample test
  and a test on store means, the reported standard errors against the
  true spread of the estimate, the design effect 1 + (m - 1) rho against
  the observed inflation, and power as the number of stores and the
  number of customers per store vary.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-supercomputer.jpg
  og_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-supercomputer.jpg
  twitter_image: /assets/images/headers/photo-supercomputer.jpg
---
The new checkout flow cannot be switched on for some customers and off for others in the same store, so the experiment randomises stores: ten get the new flow, ten keep the old one. Over the month, 4,000 customers pass through, and the analysis compares the 2,000 in treated stores with the 2,000 in control stores using a two-sample t-test. The difference is significant at the 5 percent level, and the flow is rolled out.

In the simulation this post is built on, with no true effect at all, that analysis produces a significant result in 52 percent of experiments. The customers are real, but they are not 4,000 independent observations. Customers in the same store share the store: its location, its staff, its stock, its clientele. The experiment randomised twenty things, and a test that believes it randomised four thousand is wrong by roughly the ratio between them.

## The Intraclass Correlation

Outcomes of customers in the same store are correlated because part of each outcome belongs to the store. The intraclass correlation $\rho$ is the share of the outcome's variance that sits between stores rather than within them. It is usually small, a few percent, and it is usually ignored on that basis, which is the error.

The variance of a mean over $m$ customers from one store does not fall like $\sigma^2/m$. It falls like $\sigma^2[1 + (m-1)\rho]/m$, and the bracket is the design effect: the factor by which the sample is less informative than an independent sample of the same size. With 200 customers per store and $\rho = 0.05$, the design effect is 11. Four thousand customers carry the information of 365 independent ones, and a test that assumes 4,000 uses standard errors too small by a factor of $\sqrt{11}$.

## A Simulation

Stores are randomised 50/50. Each store has its own effect drawn from a distribution whose variance is the between-store share of a total variance of 100; each customer adds noise carrying the rest. Two tests are run on every experiment: a two-sample test on all customers, and a two-sample test on the store means with the number of stores minus two as the degrees of freedom.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def draw(n_clusters, per_cluster, icc, effect=0.0, r=rng, sd=10.0):
    """Stores randomised 50/50; customers nested in stores; icc = share of variance between stores."""
    sd_b = sd * np.sqrt(icc); sd_w = sd * np.sqrt(1 - icc)
    z = np.repeat([0, 1], n_clusters // 2); r.shuffle(z)
    store_eff = r.normal(0, sd_b, n_clusters)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    y = 50 + store_eff[cluster] + effect * z[cluster] + r.normal(0, sd_w, len(cluster))
    return z, cluster, y

def customer_level(z, cluster, y):
    """Customers treated as independent: a two-sample test on all of them."""
    t = z[cluster]
    a, b = y[t == 0], y[t == 1]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return b.mean() - a.mean(), se, 2 * stats.norm.sf(abs((b.mean() - a.mean()) / se))

def cluster_level(z, cluster, y):
    """Average each store first, then a t-test on store means with stores - 2 degrees of freedom."""
    means = np.array([y[cluster == c].mean() for c in range(z.size)])
    a, b = means[z == 0], means[z == 1]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = (b.mean() - a.mean()) / se
    return b.mean() - a.mean(), se, 2 * stats.t.sf(abs(t), z.size - 2)

reps = 2000
for n_c, per, icc in ((20, 200, 0.0), (20, 200, 0.01), (20, 200, 0.05), (20, 200, 0.10), (8, 500, 0.05), (100, 40, 0.05)):
    hits = np.zeros(2)
    for _ in range(reps):
        z, cl, y = draw(n_c, per, icc)
        hits += np.array([customer_level(z, cl, y)[2], cluster_level(z, cl, y)[2]]) < 0.05
    print(f"{n_c} stores x {per}, icc {icc:.2f}: customer-level {hits[0]/reps:.1%}, store-level {hits[1]/reps:.1%}, "
          f"design effect {1 + (per - 1) * icc:.1f}")
```

**False positive rates in A/A experiments**, where the right answer is 5 percent.

| Stores | Customers per store | Intraclass correlation | Customer-level test | Store-level test | Design effect |
| --- | --- | --- | --- | --- | --- |
| 20 | 200 | 0.00 | 4.9% | 4.2% | 1.0 |
| 20 | 200 | 0.01 | 26.2% | 5.9% | 3.0 |
| 20 | 200 | 0.05 | 52.0% | 4.5% | 11.0 |
| 20 | 200 | 0.10 | 67.4% | 5.9% | 20.9 |
| 8 | 500 | 0.05 | 70.2% | 5.5% | 26.0 |
| 100 | 40 | 0.05 | 25.6% | 4.4% | 3.0 |

With no between-store variance the two tests agree, and the customer-level test is fine. With one percent of the variance between stores, a correlation most analysts would call negligible, it rejects a true null one time in four. At five percent, one time in two; at ten, two times in three. The store-level test holds its level in every row, at the cost of a little conservatism when there are few stores. The two rows at the bottom make the design point: eight big stores are worse than twenty medium ones, and a hundred small ones bring the customer-level test's error down to 26 percent, which is still five times what was promised.

![False positive rate of A/A experiments with 20 stores of 200 customers against the intraclass correlation, for the customer-level test and the store-level test. The customer-level test passes 5 percent at any positive correlation; the store-level test stays at its level.](/assets/images/figures/cluster_randomisation_false_positives.png){: width="1152" height="672" loading="lazy"}

## What the Standard Error Should Have Been

The customer-level test is not wrong about the estimate, only about its uncertainty. The estimated difference is the same either way; the standard error is what differs.

```python
ests, se_cust, se_clus = [], [], []
for _ in range(reps):
    z, cl, y = draw(20, 200, 0.05)
    e, s1, _ = customer_level(z, cl, y); _, s2, _ = cluster_level(z, cl, y)
    ests.append(e); se_cust.append(s1); se_clus.append(s2)
print(f"true sd of the estimate {np.std(ests):.2f}; customer-level SE {np.mean(se_cust):.2f}; store-level SE {np.mean(se_clus):.2f}")
```

| 20 stores × 200 customers, intraclass correlation 0.05 | Value |
| --- | --- |
| True standard deviation of the estimated effect across experiments | 1.04 |
| Standard error reported by the customer-level test | 0.32 |
| Standard error reported by the store-level test | 1.03 |
| Effective sample size: 4,000 customers / design effect 11 | 365 |

The customer-level test reports a standard error a third of the truth, which is $1/\sqrt{11}$ to the second decimal. The store-level test reports the truth. The effective sample size, 365 customers' worth of information from 4,000 customers, is the honest description of what the experiment collected, and it is what a power calculation should have started from.

## Where Power Comes From

Once the unit of analysis is the store, the question of how to make the experiment more powerful has an answer that surprises people: not by observing more customers.

```python
for n_c, per in ((20, 200), (20, 2000), (50, 200), (100, 200), (200, 200)):
    hits = 0
    for _ in range(reps):
        z, cl, y = draw(n_c, per, 0.05, effect=2.0)
        hits += cluster_level(z, cl, y)[2] < 0.05
    print(f"{n_c} stores x {per} customers: power {hits/reps:.0%}")
```

**Power for a true effect of 2 units**, four percent of the mean, with an intraclass correlation of 0.05.

| Stores | Customers per store | Total customers | Power |
| --- | --- | --- | --- |
| 20 | 200 | 4,000 | 43% |
| 20 | 2,000 | 40,000 | 45% |
| 50 | 200 | 10,000 | 83% |
| 100 | 200 | 20,000 | 99% |
| 200 | 200 | 40,000 | 100% |

Ten times the customers in the same twenty stores raise power from 43 to 45 percent. Two and a half times the stores, with the same customers per store, raise it to 83 percent. Forty thousand customers in twenty stores are worth less than ten thousand in fifty. The reason is the variance of a store mean, $\sigma_b^2 + \sigma_w^2/m$: the within-store part shrinks with more customers, the between-store part does not, and once $m$ is a few hundred the between-store part is nearly all that is left.

```python
sd_b2, sd_w2 = 100 * 0.05, 100 * 0.95
for m in (50, 200, 2000, 100000):
    print(f"m = {m}: variance of a store mean {sd_b2 + sd_w2 / m:.2f} (floor {sd_b2:.2f})")
```

| Customers per store | Variance of a store mean | Floor set by between-store variance |
| --- | --- | --- |
| 50 | 6.90 | 5.00 |
| 200 | 5.47 | 5.00 |
| 2,000 | 5.05 | 5.00 |
| 100,000 | 5.00 | 5.00 |

Past a couple of hundred customers per store the mean is as precise as it will ever get, and the only way to a more precise experiment is more stores. That is the number to negotiate for when the experiment is designed, and the number the customer-level analysis hides by pretending the customers were the sample.

## When There Are Only a Few Clusters

Eight stores is not an unusual experiment; it may be all the stores a region has. The store-level test is still valid, but it is working with six degrees of freedom.

```python
hits = 0; ests = []
for _ in range(reps):
    z, cl, y = draw(8, 500, 0.05, effect=2.0)
    e, s, p = cluster_level(z, cl, y); hits += p < 0.05; ests.append(e)
print(f"8 stores x 500: power {hits/reps:.0%}, estimate {np.mean(ests):.2f} +/- {np.std(ests):.2f}")
```

With eight stores and a true effect of 2, the store-level test has 16 percent power and the estimate has a standard deviation of 1.6, most of a whole effect. Such an experiment is honest and nearly uninformative, which is better than the customer-level alternative, which is dishonest and confident: on the same eight stores it declared a nonexistent effect 70 percent of the time. When the number of clusters is fixed and small, the useful responses are to pair or stratify stores on their pre-period outcomes before randomising, to adjust for store-level covariates, or to switch to a design such as a stepped wedge or switchback that lets each store serve as its own control over time. None of these turns eight stores into eighty, but each recovers part of the between-store variance that a plain comparison leaves in the noise.

## The Analysis That Matches the Design

The store-means test used here is the simplest valid analysis, and it is a good one when clusters are of similar size. A mixed model with a random intercept per store is its natural generalisation: it weights stores by their information, adjusts for customer-level and store-level covariates, and estimates the intraclass correlation rather than assuming it. Cluster-robust standard errors on a customer-level regression are a third option, valid as the number of clusters grows and unreliable below thirty or so. What all three share, and what the naive analysis lacks, is a standard error built from the number of things that were randomised.

## What to Do

1. **Analyse at the unit that was randomised**, or with a method that accounts for it: store means, a mixed model with a store random effect, or cluster-robust standard errors with enough clusters.
2. **Estimate the intraclass correlation from pre-period data** before the experiment; even 0.01 is enough to break a customer-level test.
3. **Plan the sample in clusters, not customers.** The design effect $1 + (m-1)\rho$ converts customers into effective observations, and past a few hundred per cluster the conversion is nearly zero.
4. **Buy power with more clusters**, and stratify or pair them on pre-period outcomes when the count is small.
5. **Report the number of clusters** with every result, next to the number of customers; the first is the sample size that matters.
6. **Run the A/A check** on historical data by randomly labelling stores: a customer-level test that rejects far more than 5 percent of the time has told you the intraclass correlation is not zero.

## References

- Donner, A., & Klar, N. (2000). *Design and Analysis of Cluster Randomization Trials in Health Research*. Arnold.
- Hayes, R. J., & Moulton, L. H. (2017). *Cluster Randomised Trials* (2nd ed.). Chapman and Hall/CRC.
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Hussey, M. A., & Hughes, J. P. (2007). Design and analysis of stepped wedge cluster randomized trials. *Contemporary Clinical Trials*, 28(2), 182-191.
