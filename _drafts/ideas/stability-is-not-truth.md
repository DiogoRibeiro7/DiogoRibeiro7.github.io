---
author_profile: false
categories:
- Statistics
classes: wide
title: 'Stability Is Not Truth'
excerpt: A continuous Gaussian population can produce highly stable clusters. A negative control shows what bootstrap agreement establishes and what it leaves unanswered.
header:
  image: /assets/images/headers/constellation.jpg
  og_image: /assets/images/headers/constellation.jpg
  overlay_image: /assets/images/headers/constellation.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/constellation.jpg
  twitter_image: /assets/images/headers/constellation.jpg
keywords:
- clustering stability
- negative controls
- bootstrap
- latent classes
- representation sensitivity
- longitudinal clustering
seo_title: 'Stability Is Not Truth'
seo_description: 'Reproduce stable k-means partitions in a continuous population, compare a discrete-class control, and examine how representation changes the answer.'
seo_type: article
summary: 'An analytic Gaussian example and a controlled bootstrap experiment distinguish reproducible partitions from evidence for discrete latent groups.'
tags:
- Clustering
- Statistical Inference
- Simulation
- Reproducibility
why_this_exists: 'Stable clustering is often treated as evidence for natural classes. A reproducible negative control tests that inference directly.'
evidence: 'Original Gaussian quantization calculation and three 100-bootstrap experiments on 1,000 synthetic observations, including a separated mixture control.'
methodology: 'Fix k at two, compare bootstrap assignments with a reference fit, reweight the same continuous data, and distinguish global agreement from assignment uncertainty.'
reviewed_at: 2026-09-18
---

<!--
Development contract
Question: Can a highly reproducible partition arise without discrete latent classes?
Claim: Stability establishes repeatability under specified perturbations, not the existence of the proposed classes.
Counterclaim: Stable segmentation may still be useful, and stability can help distinguish candidates under an explicit cluster model.
Evidence object: Gaussian distortion calculation, continuous negative control, discrete mixture positive control, and representation perturbation.
Failure case: A single Gaussian control does not characterize all continuous populations or all clustering algorithms.
Reader payoff: Add controls and assignment diagnostics before interpreting clusters as substantive types.
Exclusions: Selecting the best clustering algorithm, proving universal consistency, and claiming that latent classes never exist.
-->

A clustering pipeline returns two groups. Refit it on bootstrap samples and most observations remain with the same neighbors. The adjusted Rand index is usually above 0.9.

That is evidence that the pipeline produces a repeatable partition under the chosen perturbation. It does not, by itself, show that the population contains two discrete types.

The distinction is easy to miss because stability sounds like confirmation from new evidence. But the repeated fits may all be reproducing the same geometric cut through a continuous population. We can demonstrate this with data whose generating process contains no class variable at all.

## A stable boundary can be an approximation to a continuum

Start with one continuous variable:

$$
X\sim N(0,1).
$$

Suppose we insist on describing it with two centroids. The population k-means objective is

$$
R(c_1,c_2)=E\left[\min\{(X-c_1)^2,(X-c_2)^2\}\right].
$$

Consider the symmetric split at zero. The optimal centroid within each half is its conditional mean:

$$
c_-=E[X\mid X<0]=-\sqrt{\frac2\pi},\qquad
c_+=E[X\mid X>0]=\sqrt{\frac2\pi}.
$$

Writing $a=\sqrt{2/\pi}$, the squared-error risk of this solution is

$$
E[(|X|-a)^2]
=E[X^2]-2aE|X|+a^2
=1-\frac2\pi
\approx0.3634.
$$

One centroid at zero has risk 1. Using two centroids therefore gives a substantial improvement in reconstruction, even though we generated the data from one smooth distribution.

The improvement is real. It is the benefit of representing a continuum with more representatives. It is not evidence that a hidden categorical variable generated the observations.

In more dimensions, unequal spread can favor a particular direction for the split. When resampling leaves that geometry nearly unchanged, the algorithm can repeatedly recover a similar boundary. Stability and the absence of a discrete class mechanism are perfectly compatible.

This limitation is part of the theoretical discussion of clustering stability: behavior depends on the objective, its solutions, and the perturbation scheme, rather than on an algorithm-independent definition of a true cluster. [Von Luxburg, *Clustering Stability: An Overview*](https://arxiv.org/abs/1007.1075).

## Build a negative control and a positive control

Our negative control contains 1,000 independent points with

$$
X_1\sim N(0,9),\qquad X_2\sim N(0,1),\qquad X_1\perp X_2.
$$

This is a single elongated Gaussian cloud. There is no sampled class label. We nevertheless fit k-means with $k=2$.

The positive control contains a genuine binary generating variable:

$$
G\sim\operatorname{Bernoulli}(1/2),
$$

$$
X_1=3(2G-1)+\epsilon_1,\qquad X_2=\epsilon_2,
$$

where $\epsilon_1$ and $\epsilon_2$ are independent $N(0,0.4^2)$ variables, independent of $G$. These two components are deliberately well separated. The labels are retained for evaluation and are not supplied to k-means.

The negative control asks whether high stability is possible without a discrete generating class. The positive control checks that the same procedure can recover a clear discrete signal. Neither control is intended as a realistic model of every application.

## Specify exactly what stability means

For each dataset, fit a reference k-means model. Then repeat the following 100 times:

1. Resample 1,000 rows with replacement.
2. Fit k-means with two clusters and 20 initializations.
3. Assign all 1,000 original points to their nearest fitted centroid.
4. Compare those assignments with the reference assignments using adjusted Rand index, or ARI.

ARI compares partitions through pairwise membership and is unchanged by swapping label names. A value of 1 means identical partitions; its chance adjustment permits values near zero or below zero for weak agreement. The precise definition and implementation are documented in [scikit-learn's ARI reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).

The reference fit and evaluation points come from the original dataset. This is a conditional bootstrap repeatability diagnostic, not held-out predictive accuracy or independent population replication. Twenty initializations reduce sensitivity to poor local solutions; they do not prove that every fit reaches a global optimum.

For a representation check, we also transform the negative control to

$$
\phi(X)=\left(\frac{X_1}{3},\,3X_2\right).
$$

The transformed coordinates have variances 1 and 9. The transformation is invertible and adds no information. It changes which differences dominate Euclidean distance.

## A reproducible experiment

The complete experiment below uses NumPy, scikit-learn, and threadpoolctl. The reported run used Python 3.13, NumPy 2.3.5, SciPy 1.15.3, and scikit-learn 1.6.1. Numerical libraries or later versions may change the last digits or a boundary assignment.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

def stability(x, seed=4321, bootstraps=100):
    rng = np.random.default_rng(seed)
    n = len(x)
    reference = KMeans(n_clusters=2, n_init=20, random_state=0).fit(x)
    labels = reference.labels_
    scores = []
    assignments = []

    for b in range(bootstraps):
        indices = rng.integers(0, n, n)
        model = KMeans(n_clusters=2, n_init=20,
                       random_state=b + 1).fit(x[indices])
        predicted = model.predict(x)
        scores.append(adjusted_rand_score(labels, predicted))

        # With two labels, choose the orientation agreeing most with reference.
        # ARI needs no alignment; the per-point frequencies below do.
        if np.mean(predicted == labels) < 0.5:
            predicted = 1 - predicted
        assignments.append(predicted)

    frequency = np.mean(assignments, axis=0)
    ambiguous = np.mean((frequency > 0.1) & (frequency < 0.9))
    return labels, np.quantile(scores, [0.1, 0.5, 0.9]), ambiguous

with threadpool_limits(limits=1):
    rng = np.random.default_rng(20260918)
    continuous = rng.normal(size=(1000, 2)) * [3, 1]
    group = rng.integers(0, 2, 1000)
    discrete = np.column_stack([
        3 * (2 * group - 1) + rng.normal(0, 0.4, 1000),
        rng.normal(0, 0.4, 1000),
    ])

    results = {}
    datasets = {
        "continuous": continuous,
        "discrete": discrete,
        "reweighted": continuous * [1 / 3, 3],
    }
    for name, x in datasets.items():
        labels, quantiles, ambiguous = stability(x)
        results[name] = labels
        print(name, "ARI quantiles", np.round(quantiles, 3),
              "ambiguous fraction", f"{ambiguous:.3f}")

    print("discrete recovery",
          adjusted_rand_score(group, results["discrete"]))
    print("between representations",
          adjusted_rand_score(results["continuous"], results["reweighted"]))
```

The results are:

| Data and representation | 10th percentile ARI | Median ARI | 90th percentile ARI | Ambiguous assignment fraction |
| --- | --- | --- | --- | --- |
| Continuous Gaussian, original coordinates | 0.879 | 0.937 | 0.980 | 5.4% |
| Separated two-component mixture | 1.000 | 1.000 | 1.000 | 0.0% |
| Same continuous Gaussian, reweighted coordinates | 0.824 | 0.906 | 0.996 | 8.0% |

The mixture reference partition also has ARI 1 against the known generating labels in this run. That is successful recovery in an intentionally easy positive control.

The continuous negative control produces a median bootstrap agreement of 0.937. Its high stability is not a false calculation. The unjustified step would be to interpret that agreement as evidence of two generating types.

The experiment fixes $k=2$; it does not estimate the number of groups. Maximizing stability without further constraints would also have to contend with the trivial one-cluster partition, which always assigns every observation together.

These percentiles describe the 100 perturbations of each particular dataset. They are not confidence limits for a population stability parameter. Establishing behavior across sample sizes, covariance structures, or overlap levels would require additional independently generated datasets and a broader experiment. One reproducible counterexample is enough for the narrower point: high stability does not logically imply discrete latent classes.

## Two stable answers can disagree with each other

The ARI between the reference partitions from the original and reweighted continuous data is approximately $-0.001$. Each representation is reasonably stable under its own bootstrap perturbations, yet their reference partitions have almost no adjusted agreement.

The reason is visible in the metric. Euclidean distance after transformation corresponds to

$$
d_\phi(x,x')^2
=\frac{(x_1-x'_1)^2}{9}+9(x_2-x'_2)^2.
$$

Relative to the original metric, the weight of the second coordinate compared with the first increases by a factor of 81. The algorithm is being asked to preserve different distinctions.

A reasonable objection is that this reweighting was chosen deliberately to change the answer. It was. The example does not show that every plausible preprocessing choice causes dramatic disagreement. It shows that stability within one representation cannot validate the representation itself.

In an application, weights should reflect measurement units, noise, and the question being asked. If the features were two unrelated measurements in different units, raw Euclidean distance would already contain an arbitrary weighting decision. If their scale had a justified physical meaning, rescaling might discard relevant structure. There is no universal instruction to standardize everything.

The necessary step is to defend the metric and examine credible alternatives. Reporting only the most stable version after trying many representations introduces an additional selection problem.

## Global agreement can hide the uncertain observations

The table's final column uses a simple diagnostic. After aligning each bootstrap's two label names with the reference, define

$$
q_i=\frac1B\sum_{b=1}^B I\{C_i^{(b)}=1\}.
$$

An observation is counted as ambiguous when $0.1<q_i<0.9$. The thresholds are descriptive choices. The label frequency is not a posterior probability that the observation belongs to a real class.

In the original continuous control, 54 of 1,000 points meet that criterion despite the high median ARI. Global agreement gives limited visibility into where the boundary moves. If an assigned group determines an action, those observations deserve individual attention.

A label-invariant alternative is the co-assignment matrix

$$
S_{ij}=\frac1B\sum_{b=1}^B I\{C_i^{(b)}=C_j^{(b)}\}.
$$

This records how often each pair is assigned together. Our protocol predicts labels for every original point in every fit, so every pair has $B$ evaluations. A protocol comparing only jointly sampled observations would need a denominator specific to each pair.

Co-assignment avoids forcing a single consensus partition to hide disagreements, but it still measures reproducibility of the selected procedure. It cannot turn bootstrap frequencies into probabilities that a latent ontology is correct.

## Why this matters for longitudinal clustering

Suppose each subject has a trajectory

$$
Y_i(t)=a_i+b_i t+\epsilon_i(t).
$$

If $(a_i,b_i)$ varies continuously across subjects, a clustering algorithm can still create repeatable “high versus low” or “increasing versus decreasing” groups. The labels discretize a continuum. Whether that discretization is useful is separate from whether the population contains distinct trajectory-generating classes.

Representations change the question. Raw trajectories emphasize level and amplitude. Centering each subject removes level. Normalizing amplitude can emphasize shape. Estimated slope features discard deviations from a straight line. Each choice preserves some distinctions and removes others.

The simulation above operates on known two-dimensional features. It is not a benchmark of noisy trajectory estimation. In a trajectory pipeline, refit data-dependent preprocessing within each resample if the intended uncertainty includes estimating that preprocessing. Resample subjects or the appropriate independent units; treating individual time points as independent subjects changes the experiment.

The site's [analysis of sampling and representation uncertainty](/statistics/sampling_uncertainty_can_dominate_representation_uncertainty/) separates those sources of disagreement. The additional question here is whether even their joint stability justifies a discrete-class interpretation. A continuous negative control shows why that conclusion needs further evidence.

## What stability is still good for

Stability can reveal an unreliable operational partition. It can identify boundary cases, expose dependence on initialization, and compare sensitivity under perturbations that reflect measurement or sampling uncertainty. Under an explicit model of cluster structure, it can also contribute to assessing candidate partitions.

A stable segmentation may be useful even if no natural classes exist. A system assigning workloads to two resource pools needs a defensible allocation rule; it does not necessarily need evidence for two kinds of workload. That rule should be evaluated by capacity, latency, and allocation cost as well as repeatability.

Conversely, real but overlapping classes can be difficult to recover, and small samples can produce unstable assignments. Low stability does not prove that the generating process is continuous. The positive control succeeds because separation is large; reducing it changes the recovery problem.

Before giving clusters substantive names, I would want to see the following:

1. A definition of the claim: operational segmentation, density structure, or latent generating classes.
2. A perturbation protocol that preserves the sampling units and includes the fitted parts of the pipeline.
3. Continuous controls with relevant covariance and noise, plus discrete controls spanning plausible separation and imbalance.
4. Results across defensible representations, alongside pairwise or individual assignment diagnostics.
5. External evidence or an explicit generative argument for any claim that the clusters correspond to distinct mechanisms.

Even external associations need interpretation: cutting a continuous severity variable into groups can produce different outcomes without revealing distinct subtypes. Compare that explanation with the proposed categorical one.

The supported statement from a stability analysis is specific: this algorithm, representation, number of clusters, and perturbation scheme repeatedly produced similar assignments. Turning that result into a claim about what kinds of entities exist in the population requires a separate argument.
