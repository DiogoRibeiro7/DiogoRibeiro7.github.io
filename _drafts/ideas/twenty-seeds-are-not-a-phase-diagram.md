---
author_profile: false
categories:
- Statistics
classes: wide
excerpt: A simulation can produce a stable-looking mean while the qualitative regime map built from it remains unstable. This article explains why Monte Carlo uncertainty must be judged relative to the scientific claim.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- Monte Carlo simulation
- Monte Carlo error
- simulation studies
- phase diagrams
- statistical validation
seo_description: Why small simulation budgets can create unstable phase diagrams and why Monte Carlo uncertainty should be judged relative to the scientific claim.
seo_title: Twenty Seeds Are Not a Phase Diagram
seo_type: article
summary: A technical note on Monte Carlo replication, threshold instability, and why qualitative conclusions can remain fragile after means appear stable.
tags:
- Statistics
- Simulation
- Reproducibility
title: 'Twenty Seeds Are Not a Phase Diagram'
---

A simulation study can be numerically correct and still tell the wrong scientific story.

The failure often appears when noisy Monte Carlo summaries are converted into categorical statements: stable versus unstable, positive versus negative, regime A versus regime B, or a phase transition at one location rather than another. Once a threshold is involved, the scientific object is no longer only a mean. It is also the classification induced by a noisy estimate.

The title is deliberately provocative. Twenty replications are not automatically wrong. If two regimes are very well separated, even a small simulation can be informative. The point is narrower:

> **A replication count is not evidence by itself. Its adequacy depends on the estimand, the Monte Carlo uncertainty, and the distance to the decision boundary used by the conclusion.**

## The hidden random variable in the conclusion

Suppose a simulation produces a binary event on each replication. Let

$$
X_i \in \{0,1\},
\qquad
P(X_i=1)=p.
$$

After $N$ independent replications, define

$$
\widehat p_N
=
\frac{1}{N}\sum_{i=1}^{N}X_i,
$$

and classify the cell according to

$$
\text{class}=\mathbf 1\{\widehat p_N>0.5\}.
$$

There are now two stochastic objects in the analysis: the estimator $\widehat p_N$ and the qualitative label obtained after thresholding it.

Those are not equally stable.

If the true probability is

$$
p=0.51,
$$

then the population quantity is technically on the positive side of the boundary. But with only 21 independent replications, the probability that the observed majority lands on the other side is

$$
P\!\left(\operatorname{Binomial}(21,0.51)\le 10\right)
\approx 0.463.
$$

So a genuine 51/49 split can look like the opposite regime in almost half of such experiments.

The problem is not the random-number generator. The problem is the strength of the conclusion being extracted from weak separation.

## Monte Carlo error should be visible

For this Bernoulli example,

$$
\operatorname{MCSE}(\widehat p_N)
=
\sqrt{\frac{p(1-p)}{N}}.
$$

Since

$$
p(1-p)\le \frac14,
$$

the worst-case Monte Carlo standard error satisfies

$$
\operatorname{MCSE}(\widehat p_N)
\le
\frac{1}{2\sqrt N}.
$$

This gives a useful scale:

| $N$ | Maximum MCSE | Approx. $1.96\times$ MCSE |
| ---: | ---: | ---: |
| 20 | 0.112 | 0.219 |
| 50 | 0.071 | 0.139 |
| 100 | 0.050 | 0.098 |
| 200 | 0.035 | 0.069 |
| 500 | 0.022 | 0.044 |
| 1,000 | 0.016 | 0.031 |

The last column is not an exact confidence interval. It is a scale calculation showing how much Monte Carlo variation may remain around a proportion near one half.

The relevant question is therefore not

$$
\text{“Is }N=100\text{ a respectable number?”}
$$

but

$$
\boxed{
\text{“Is the Monte Carlo uncertainty small relative to the claim I am making?”}
}
$$

## Means can look stable while regimes are unstable

Suppose an experiment compares two methods through a per-replication difference $\Delta_i$. The Monte Carlo mean

$$
\overline\Delta_N
=
\frac{1}{N}\sum_{i=1}^{N}\Delta_i
$$

may move very little when we increase the number of replications.

Now define

$$
q=P(\Delta_i<0)
$$

and use the rule $q>0.5$ to define a qualitative regime.

The mean and the majority probability answer different questions. A small positive $E[\Delta]$ can coexist with a value of $q$ close to one half, especially when the distribution is heterogeneous, skewed, or concentrated near zero.

So we can have

$$
\overline\Delta_N
\approx
\text{stable}
$$

while

$$
\mathbf 1\{\widehat q_N>0.5\}
\approx
\text{unstable}.
$$

This distinction matters whenever the scientific claim depends on a sign, majority, ranking, crossing, stability class, or phase boundary.

## A phase diagram is a collection of decisions

Suppose a study evaluates a grid of parameter settings and assigns each cell to a regime.

The resulting phase diagram can look deterministic on the page. Statistically, however, it is a collection of classification decisions made from finite Monte Carlo samples.

If neighbouring cells lie close to a decision boundary, the apparent topology can change under replication:

$$
A\rightarrow C
$$

in one run, but

$$
A\rightarrow B\rightarrow C
$$

in another.

The issue is not merely whether each cell has a sensible point estimate. It is whether the **adjacency, ordering, and transition structure** are stable enough to support the scientific story being told about the diagram.

## Replication should depend on distance to the decision boundary

There is no useful universal rule saying that 20, 50, 100, or 1,000 replications are enough.

The required replication depends on at least four things:

1. the performance measure being estimated;
2. its Monte Carlo variance;
3. the threshold or comparison used to construct the scientific conclusion;
4. the distance between the population quantity and that boundary.

A cell with $p\approx0.95$ is a very different inferential problem from one with $p\approx0.51$.

A practical design is therefore adaptive in precision rather than adaptive in storytelling.

### Stage 1: broad screening

Use a moderate number of replications to explore the parameter space and locate potentially interesting regions.

### Stage 2: identify dangerous cells

Flag cells close to a decision boundary, cells with large Monte Carlo error, and cells responsible for a qualitative topology change.

### Stage 3: escalate replication

Allocate additional independent replications until the uncertainty is small enough for the intended claim, or conclude that the boundary cannot be located precisely with the available computational budget.

### Stage 4: challenge the topology

Ask whether the first transition, ordering, or phase label survives the escalation.

### Stage 5: report the instability

If a candidate transition disappears, report that fact rather than rewriting the history of the experiment as though the final topology had always been obvious.

## Screening is not confirmation

Adaptive replication introduces another risk.

If a large parameter grid is searched using noisy estimates, the most interesting cells are selected, and then those same noisy estimates are treated as confirmation, selection itself can exaggerate the apparent signal.

A clean design separates the roles of the replications, for example

$$
\text{screening seeds}
\quad\perp\quad
\text{confirmation seeds}.
$$

The exact design can vary. The principle should not:

$$
\boxed{
\text{Do not let a noisy search result silently become its own confirmation.}
}
$$

## Failed transitions are useful results

Suppose a 20-replication screen suggests

$$
A\rightarrow C.
$$

At 100 replications the transition becomes

$$
A\rightarrow B,
$$

and at 200 replications it remains $A\rightarrow B$.

The disappearing $A\rightarrow C$ transition is not wasted computation. It reveals that the topology itself was sensitive to Monte Carlo error.

That tells us which regions need stronger evidence and which summaries are too close to a classification boundary to support categorical claims.

A falsified transition can improve a study more than another stable average.

## Report the hierarchy, not only the final number

For important simulation performance measures, I would report at least:

- the Monte Carlo estimate;
- a Monte Carlo standard error or another appropriate simulation-uncertainty measure;
- the number of successful replications contributing to the estimate;
- the rule used to transform the estimate into any qualitative label;
- any replication escalation used for boundary cases.

If a phase diagram is central to the conclusion, the underlying continuous statistics should also be retained. A colour map is easier to interpret when we can see how far each cell lies from the boundary that created the colour.

## Three different notions of convergence

For simulation work, it is useful to separate three questions:

$$
\boxed{
\text{Is the estimated performance measure numerically stable?}
}
$$

$$
\boxed{
\text{Is its Monte Carlo uncertainty small enough for the desired precision?}
}
$$

$$
\boxed{
\text{Is the qualitative scientific conclusion stable?}
}
$$

Those are not the same question.

The last one can be the hardest because thresholding destroys information. Two estimates that differ only slightly can receive different categorical labels, while two cells with very different distances from the boundary can be displayed with the same colour.

## The broader lesson

Monte Carlo replication is not a cosmetic parameter chosen once at the top of a notebook. It is part of the inferential design.

If the conclusion is continuous, study the uncertainty of the continuous estimand. If the conclusion is a threshold, ordering, or phase transition, study the uncertainty of that decision too.

The right replication count is not 20, 100, or 1,000 in the abstract. It is the amount of simulation needed to make the uncertainty small enough **relative to the scientific claim**.

And when the claim is about topology, that can require substantially more evidence than a stable-looking mean.

## References

- Koehler E, Brown E, Haneuse S. *On the Assessment of Monte Carlo Error in Simulation-Based Statistical Analyses*. The American Statistician. 2009;63(2):155–162.
- Morris TP, White IR, Crowther MJ. *Using simulation studies to evaluate statistical methods*. Statistics in Medicine. 2019;38(11):2074–2102.
