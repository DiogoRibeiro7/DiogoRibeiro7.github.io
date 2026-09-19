---
permalink: '/statistics/when_one_simulation_metric_tells_wrong_story/'
title: 'When One Simulation Metric Tells the Wrong Story'
categories:
- Statistics
- Data Science
tags:
- Monte Carlo Simulation
- Statistical Computing
- Parameter Estimation
- Numerical Optimisation
- Reproducibility
author_profile: false
seo_title: 'Monte Carlo Studies Need More Than One Recovery Metric'
seo_description: 'Median error, tolerance success rates, optimiser disagreement and boundary hits answer different questions. A simulation can rank the same parameter region differently depending on which diagnostic you report.'
excerpt: >-
  A parameter setting can have the smaller typical estimation error and
  still be much more sensitive to starting values. Another can look poor
  under a tolerance threshold while remaining numerically stable. The
  contradiction disappears once we stop asking one simulation metric to
  answer several different questions.
summary: >-
  Why median absolute log error, tolerance based recovery rates, optimiser
  disagreement and boundary hits should be treated as complementary Monte
  Carlo diagnostics rather than interchangeable measures of estimator
  quality, with a finite sample example in which the rankings genuinely
  disagree.
keywords:
  - Monte Carlo simulation
  - simulation study
  - parameter recovery
  - numerical optimisation
  - statistical diagnostics
  - estimator evaluation
classes: wide
date: '2026-09-01'
why_this_exists: >-
  Simulation studies often compress estimator behaviour into one number.
  That is convenient, but it can turn a multidimensional numerical problem
  into a misleading ranking. A recent parameter recovery experiment made
  the failure unusually visible because point error, tolerance success,
  start sensitivity and boundary behaviour did not always agree.
evidence: >-
  A fixed seed recovery grid with repeated random samples, two sample sizes,
  multiple parameter regimes and three separated optimisation starts per
  replication. The article reports selected diagnostics from that grid while
  leaving the underlying unpublished model out of the discussion.
methodology: >-
  Compares median absolute log error, the fraction of estimates within 20
  percent of the true value, disagreement across separated optimisation
  starts and the frequency with which the retained fit approaches a search
  boundary. The diagnostics are interpreted as distinct functionals rather
  than collapsed into a single score.
reviewed_at: '2026-09-14'
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
---

A Monte Carlo study often ends with a table and a winner.

Estimator A has the smaller RMSE. Estimator B has the smaller bias. One parameter region has better coverage. Another has a larger fraction of estimates within some tolerance. We pick the metric that seems most relevant, rank the methods, and move on.

That workflow is attractive because it produces a clean conclusion. It can also be wrong.

I was recently running a finite sample parameter recovery experiment in which the details of the underlying model were not the interesting part. The interesting part was that four perfectly reasonable diagnostics sometimes ranked the same parameter settings differently.

None of the diagnostics was broken. They were answering different questions.

The mistake would have been to force them into one story.

## One Parameter, Four Questions

Suppose a model contains a positive parameter $\theta$, and a simulation produces estimates

$$
\widehat\theta_1,\ldots,\widehat\theta_R.
$$

A natural error measure for a positive parameter is the absolute log error

$$
E_r
=
\left|\log\frac{\widehat\theta_r}{\theta}\right|.
$$

Its median,

$$
M
=
\operatorname{median}(E_1,\ldots,E_R),
$$

has a useful interpretation. It measures a typical multiplicative error and treats overestimation and underestimation symmetrically on the log scale.

But a practitioner may care about a different question. How often is the estimate close enough to the truth for the intended use?

For a 20 percent tolerance we can define

$$
P_{20}
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbf 1
\left(
0.8\theta
\le
\widehat\theta_r
\le
1.2\theta
\right).
$$

Now we already have two diagnostics, and they need not agree.

The median asks where the middle of the error distribution sits. The tolerance rate asks how much probability lies inside a fixed interval. Those are different functionals of the sampling distribution.

The numerical procedure introduces two more questions.

If the likelihood is difficult, I may fit every simulated sample from several separated starting values. If the resulting parameter vectors disagree materially, the sample has exposed sensitivity to the optimisation landscape. Define

$$
D
=
\frac{\text{replications with material start disagreement}}{R}.
$$

Finally, if optimisation is constrained to a broad numerical box, I also want to know how often the retained solution approaches that box. A boundary rate

$$
B
=
\frac{\text{replications near a search boundary}}{R}
$$

is not an estimation error measure at all. It is a warning that the numerical safeguard may be influencing the fit.

So a simulation that reports $M$, $P_{20}$, $D$, and $B$ is not reporting four versions of the same statistic. It is asking four different questions:

1. How large is the typical multiplicative error?
2. How often is the estimate practically close to the truth?
3. How sensitive is the fit to the optimiser's starting point?
4. How often does the numerical search press against its allowed region?

Trying to collapse those questions into one score throws information away.

## A Case Where the Rankings Disagree

The following values come from a fixed seed recovery grid in a model I am currently studying. I have deliberately removed the model specific details here because they are irrelevant to the methodological point.

Two parameter regions, A and B, are fitted to repeated random samples. Every replication is fitted from three separated starting points.

```python
import pandas as pd

results = pd.DataFrame(
    [
        {
            "regime": "low information",
            "n": 501,
            "region": "A",
            "median_log_error": 0.282,
            "within_20pct": 0.225,
            "start_disagreement": 0.200,
            "boundary_rate": 0.000,
        },
        {
            "regime": "low information",
            "n": 501,
            "region": "B",
            "median_log_error": 0.207,
            "within_20pct": 0.475,
            "start_disagreement": 0.475,
            "boundary_rate": 0.050,
        },
        {
            "regime": "low information",
            "n": 1001,
            "region": "A",
            "median_log_error": 0.220,
            "within_20pct": 0.450,
            "start_disagreement": 0.325,
            "boundary_rate": 0.000,
        },
        {
            "regime": "low information",
            "n": 1001,
            "region": "B",
            "median_log_error": 0.232,
            "within_20pct": 0.475,
            "start_disagreement": 0.475,
            "boundary_rate": 0.025,
        },
        {
            "regime": "higher information",
            "n": 1001,
            "region": "A",
            "median_log_error": 0.068,
            "within_20pct": 0.875,
            "start_disagreement": 0.000,
            "boundary_rate": 0.000,
        },
        {
            "regime": "higher information",
            "n": 1001,
            "region": "B",
            "median_log_error": 0.230,
            "within_20pct": 0.475,
            "start_disagreement": 0.050,
            "boundary_rate": 0.075,
        },
    ]
)

print(results.to_string(index=False))
```

| Regime | $n$ | Region | Median log error | Within 20% | Start disagreement | Boundary rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Low information | 501 | A | 0.282 | 22.5% | 20.0% | 0.0% |
| Low information | 501 | B | 0.207 | 47.5% | 47.5% | 5.0% |
| Low information | 1001 | A | 0.220 | 45.0% | 32.5% | 0.0% |
| Low information | 1001 | B | 0.232 | 47.5% | 47.5% | 2.5% |
| Higher information | 1001 | A | 0.068 | 87.5% | 0.0% | 0.0% |
| Higher information | 1001 | B | 0.230 | 47.5% | 5.0% | 7.5% |

The first pair is the interesting one.

At $n=501$, region B looks better if I report only parameter recovery. Its median log error is smaller, $0.207$ instead of $0.282$, and almost half of its estimates fall within 20 percent of the truth compared with only 22.5 percent for region A.

A paper with one recovery metric could easily stop there.

But B also disagrees across optimisation starts in 47.5 percent of the replications, compared with 20 percent for A, and 5 percent of its fits end close to a numerical boundary.

So which region is better?

That question is underspecified.

B gives better point recovery in this finite sample experiment. A gives more stable numerical optimisation. Those statements can both be true.

At $n=1001$, the ambiguity becomes even clearer. The median errors are almost tied, with A slightly smaller, while the 20 percent success rate still slightly favours B. Start disagreement remains much larger for B.

If I insist that every diagnostic must identify the same winner, I have to ignore part of the evidence.

## Why Median Error and Tolerance Rates Can Reverse

The first disagreement is statistical rather than numerical.

Let

$$
Z
=
\log\frac{\widehat\theta}{\theta}.
$$

The median absolute log error is

$$
\operatorname{median}(|Z|).
$$

The 20 percent success rate is

$$
P\left(\log 0.8 \le Z \le \log 1.2\right).
$$

There is no theorem saying these two quantities must rank two sampling distributions in the same order.

Imagine one estimator whose errors are tightly concentrated just outside the 20 percent tolerance and another whose distribution has a sharper central spike but longer tails. The first can have the smaller median absolute error while the second places more probability inside the chosen tolerance.

The threshold matters too. Changing 20 percent to 10 percent or 30 percent can change the ranking again.

That does not make tolerance metrics bad. It means they encode a decision.

A statement such as

$$
P\left(\left|\widehat\theta/\theta-1\right|\le0.2\right)=0.75
$$

is meaningful if 20 percent corresponds to a real scientific or operational tolerance. It is much less meaningful if 20 percent was chosen because it made a convenient table.

The median has the opposite strength and weakness. It is threshold free and robust, but it can hide the shape of the distribution around a practically important cutoff.

Report both when both questions matter.

## Optimiser Disagreement Is Not Estimation Error

The more subtle mistake is to interpret optimiser sensitivity as though it were another measure of statistical error.

It is not.

Suppose three optimisation runs begin at separated points

$$
\phi^{(1)}_0,\qquad
\phi^{(2)}_0,\qquad
\phi^{(3)}_0,
$$

and produce fitted vectors

$$
\widehat\phi^{(1)},\qquad
\widehat\phi^{(2)},\qquad
\widehat\phi^{(3)}.
$$

If those fits differ substantially, several things may be happening. The objective may be flat in some direction. There may be several local minima. Finite sample noise may have produced a ridge. Two parameters may be weakly separated. The optimisation method itself may be struggling.

None of those possibilities is equivalent to saying that the final retained estimate is far from the truth.

A sample can produce a good estimate after selecting the best of several starts while simultaneously revealing an unstable objective surface.

That is exactly why the low information example above is useful. Region B can have better point recovery and worse start sensitivity at the same time.

If the goal is a simulation paper, that distinction matters. If the goal is software that somebody else will use on one real dataset, it matters even more.

The user of the fitted model does not get to average over 40 simulated datasets. They get one dataset and one objective surface.

## Boundary Hits Tell Yet Another Story

Constrained optimisation is often unavoidable in difficult parameter recovery problems. A broad search box prevents pathological numerical excursions and keeps the simulation finite.

But a bound is supposed to be a safeguard, not an estimator.

If fitted values repeatedly land near it, the boundary has become part of the answer.

That is why I like to record a boundary indicator separately rather than silently accepting every converged optimisation run.

For a parameter vector $\phi$ with lower and upper bounds $L_j$ and $U_j$, one simple diagnostic is to flag a fit when a coordinate falls within a small fraction of its allowed interval:

$$
\min\left(
\frac{\widehat\phi_j-L_j}{U_j-L_j},
\frac{U_j-\widehat\phi_j}{U_j-L_j}
\right)
<\varepsilon.
$$

A nonzero boundary rate does not automatically invalidate the estimator. It tells us that the chosen box and the sampling behaviour are interacting and deserve inspection.

That information should not be mixed into an RMSE or a composite score. It is a diagnostic about the numerical experiment itself.

## When the Diagnostics Do Agree

Multiple metrics do not mean that every conclusion becomes ambiguous.

Look at the higher information regime in the table.

At $n=1001$, region A has median log error $0.068$ against $0.230$ for B. Its within 20 percent recovery rate is 87.5 percent against 47.5 percent. There is no start disagreement and no boundary hit, while B still shows both.

Here the evidence points in the same direction.

That is a much stronger conclusion than declaring A the winner because one metric happened to be smaller.

Agreement across diagnostics is informative precisely because disagreement was possible.

## Do Not Solve This With a Composite Score

A tempting response is to combine everything:

$$
S
=
w_1 M
-w_2 P_{20}
+w_3 D
+w_4 B.
$$

Now every simulation cell has one number again.

I would resist that unless the weights have a real decision theoretic meaning.

Otherwise the composite score simply hides the disagreement behind arbitrary constants. Is a ten percentage point increase in start disagreement worth a 0.05 reduction in median log error? Is one boundary hit equivalent to five failed tolerance recoveries?

There is no statistical answer to those questions without a utility function.

Keeping the diagnostics separate is not indecision. It is an honest representation of a problem with several dimensions.

## A Better Simulation Table

For difficult estimation problems, I now prefer a table that separates at least three layers.

The first layer describes statistical recovery:

* median or mean error on an appropriate scale
* a robust spread such as an interquartile range
* a scientifically meaningful tolerance probability

The second layer describes numerical behaviour:

* convergence rate
* disagreement across multiple starts
* boundary frequency
* objective differences between competing solutions

The third layer describes uncertainty in the Monte Carlo experiment itself:

* number of replications
* Monte Carlo standard errors for reported rates
* fixed or reproducibly generated seeds

A rate computed from 40 replications has standard error

$$
\sqrt{\frac{p(1-p)}{40}},
$$

which is at most about $0.079$. A difference of two or three percentage points should therefore not be narrated like a discovery. A difference of forty percentage points is another matter.

The simulation needs uncertainty quantification too.

## What I Would Report

If I had to reduce the whole argument to one reporting rule, it would be this:

> Report one metric for typical error, one metric tied to practical recovery, and at least one diagnostic for numerical stability.

For positive parameters, a reasonable starting set is

$$
\boxed{
\operatorname{median}\left|\log(\widehat\theta/\theta)\right|,
\quad
P\left(\left|\widehat\theta/\theta-1\right|\le\delta\right),
\quad
P(\text{start disagreement}),
\quad
P(\text{boundary})
}
$$

with $\delta$ chosen for a substantive reason.

None of these is universally best. That is the point.

A simulation study should not be designed to manufacture a ranking. It should reveal how an estimator behaves.

Sometimes the metrics agree and the conclusion becomes stronger. Sometimes they disagree and the disagreement is the result.

The wrong response to that disagreement is to search for the one statistic that restores a clean story.

The better response is to ask what each statistic was measuring in the first place.
