---
permalink: '/statistics/a_negative_monte_carlo_result_is_still_a_result/'
title: 'A Negative Monte Carlo Result Is Still a Result'
categories:
- Statistics
- Data Science
tags:
- Monte Carlo Simulation
- Statistical Computing
- Scientific Method
- Numerical Analysis
- Reproducibility
author_profile: false
seo_title: 'Why Negative Monte Carlo Results Matter'
seo_description: 'A simulation study is useful when it disproves the story you expected. Negative Monte Carlo results can expose weak theory, inadequate diagnostics, numerical error and overconfident claims.'
excerpt: >-
  A simulation that refuses to confirm the theory you hoped to see is not a failed
  simulation. It may be the most useful part of the project. The difficult part is
  resisting the temptation to tune the experiment until the preferred story comes
  back.
summary: >-
  A practical framework for treating negative Monte Carlo results as scientific
  evidence. The article separates sampling noise, numerical error, diagnostic
  mismatch and genuine theoretical failure, and shows how a simulation can narrow
  a claim without making the research weaker.
keywords:
  - Monte Carlo study
  - negative result
  - simulation study
  - numerical diagnostics
  - statistical research
  - reproducibility
classes: wide
date: '2026-09-14'
why_this_exists: >-
  Simulation studies are often written as if their purpose were to confirm a
  theoretical expectation. In real methodological work, a broader grid or a more
  accurate numerical calculation can overturn the expected pattern. That is not a
  nuisance to hide; it is evidence about the limits of the claim.
evidence: >-
  Motivated by a recent parameter-recovery study in which an expected weak region
  was not uniformly worse in finite samples, and a later higher-accuracy
  information calculation showed that the original local diagnostic had been
  interpreted too broadly.
methodology: >-
  Distinguishes four explanations for an unexpected simulation result: Monte Carlo
  noise, implementation error, diagnostic mismatch and a genuinely false or too
  broad claim. Uses small reproducible examples to show how replication and
  numerical refinement separate those cases.
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

A Monte Carlo study is not supposed to agree with you.

It is supposed to answer the experiment you actually ran.

That sounds obvious, but simulation studies are unusually easy to turn into confirmation machines. We choose the data-generating process, the parameter grid, the sample sizes, the estimators, the metrics and often the plots too. When a result looks inconvenient, there are many legitimate-looking ways to change the experiment until it looks more familiar.

Increase the sample size.

Use a different error metric.

Drop a difficult parameter region.

Change the optimiser.

Increase the number of starts.

Smooth the curve.

Average over another axis.

All of those changes can be scientifically defensible. They can also quietly convert

$$
\text{test the claim}
$$

into

$$
\text{find a design where the claim looks true}.
$$

A negative Monte Carlo result is therefore not a failed simulation.

Sometimes it is the point where the research finally becomes interesting.

## What I Mean by a Negative Result

I do not mean only a result with a negative numerical sign.

A negative Monte Carlo result is any simulation outcome that does not support the ordering, threshold, monotonicity or qualitative pattern that motivated the experiment.

Suppose theory or a local approximation suggests that parameter region A should be easier to estimate than region B.

A natural finite-sample hypothesis is

$$
E\{L(\widehat\theta_A,\theta_A)\}
<
E\{L(\widehat\theta_B,\theta_B)\},
$$

for some loss function \(L\).

Then the simulation may return

$$
\widehat L_A
\approx
\widehat L_B,
$$

or even

$$
\widehat L_A
>
\widehat L_B.
$$

That is negative evidence for the finite-sample claim.

It is not automatically evidence that the theory is wrong.

There are at least four possibilities:

1. the Monte Carlo experiment is too noisy;
2. the implementation or numerical method is wrong;
3. the theoretical diagnostic and the simulation metric answer different questions;
4. the original claim is genuinely false or too broad.

The research problem is to distinguish them.

## The First Temptation: Explain It Away

Suppose I expect method A to outperform method B, but after 20 simulation replications I obtain

$$
\overline L_A=0.24,
\qquad
\overline L_B=0.21.
$$

The easiest story is that 20 replications are not enough.

That may be true.

But "Monte Carlo noise" should itself be quantified rather than used as a rhetorical escape hatch.

If \(D_r=L_{A,r}-L_{B,r}\) is the paired loss difference in replication \(r\), then the Monte Carlo standard error of the mean difference is

$$
\operatorname{MCSE}(\overline D)
=
\frac{s_D}{\sqrt R}.
$$

The relevant question is not whether the observed ordering is surprising to us. It is whether the experiment is precise enough to distinguish the competing conclusions.

A useful reporting unit is therefore

$$
\overline D
\pm
2\operatorname{MCSE}(\overline D).
$$

If this interval is wide and crosses zero comfortably, the honest result is

> the current simulation cannot resolve the ordering.

That is already information.

It means the proposed difference is not large relative to finite-sample and Monte Carlo variability under the current design.

## A Tiny Example

Here is a deliberately simple demonstration.

Suppose estimator A has slightly smaller expected squared error than estimator B, but the variance across generated datasets is large.

```python
import numpy as np

rng = np.random.default_rng(14)

R = 20
loss_a = rng.normal(loc=0.20, scale=0.16, size=R)
loss_b = rng.normal(loc=0.23, scale=0.16, size=R)

d = loss_a - loss_b

mean_difference = d.mean()
mcse = d.std(ddof=1) / np.sqrt(R)

print(mean_difference, mcse)
```

With only 20 replications, it is perfectly possible to observe the wrong ordering.

The correct response is not to delete the cell.

It is to increase replication if the ordering matters scientifically.

```python
for R in (20, 100, 1_000, 10_000):
    rng = np.random.default_rng(14)
    loss_a = rng.normal(0.20, 0.16, R)
    loss_b = rng.normal(0.23, 0.16, R)
    d = loss_a - loss_b

    print(
        R,
        d.mean(),
        d.std(ddof=1) / np.sqrt(R),
    )
```

As \(R\) grows, Monte Carlo error shrinks like

$$
R^{-1/2}.
$$

If the unexpected pattern disappears under adequate replication, the negative result was about simulation precision.

If it remains, we have learned something else.

## Replication Does Not Fix Numerical Error

More Monte Carlo replications reduce simulation error.

They do not repair a biased numerical calculation.

This distinction matters because methodological research often nests one numerical approximation inside another:

$$
\text{simulation}
\rightarrow
\text{fit estimator}
\rightarrow
\text{optimisation}
\rightarrow
\text{quadrature or integration}
\rightarrow
\text{summary}.
$$

A systematic error at an inner layer survives averaging.

If each replication computes

$$
\widetilde T_r=T_r+b+\varepsilon_r,
$$

where \(b\) is numerical bias, then

$$
\overline{\widetilde T}
\xrightarrow{R\to\infty}
E[T]+b.
$$

Running one million replications only estimates the wrong target more precisely.

This is why an unexpected Monte Carlo result should trigger numerical diagnostics before it triggers interpretation.

## A Pattern I Have Learned to Distrust

One particularly dangerous pattern is this:

1. a theoretical argument predicts a special parameter region;
2. a coarse numerical calculation appears to show a feature there;
3. a simulation is designed around that feature;
4. the simulation behaves inconsistently;
5. the temptation is to blame the simulation.

The correct next step may instead be to improve step 2.

A numerical dip, peak or boundary can be an approximation artefact.

If the scientific interpretation depends on a local feature of a curve, I want to see convergence under refinement.

For a numerical approximation \(I_h(\theta)\) with resolution parameter \(h\), the basic question is

$$
I_h(\theta)
\longrightarrow
I(\theta)
$$

as \(h\to0\).

A plot at one resolution is not enough.

## Coarse Quadrature Can Invent Geometry

Consider an integral

$$
I(\theta)
=
\int f(z;\theta)\,dz.
$$

Suppose we approximate it using \(m\) midpoint evaluations:

$$
I_m(\theta)
=
\frac{1}{m}
\sum_{j=1}^{m}
f(z_j;\theta).
$$

If the integrand changes rapidly in some parameter region, a coarse grid can underestimate or overestimate the integral non-uniformly in \(\theta\).

That can manufacture apparent curvature:

$$
I_m(\theta_0)
<
I_m(\theta_0-\varepsilon),
\qquad
I_m(\theta_0)
<
I_m(\theta_0+\varepsilon),
$$

while the converged quantity has no local minimum at all.

The resulting plot can look exactly like theoretical confirmation.

Then a broader Monte Carlo experiment refuses to obey it.

That is not the simulation failing.

It may be the simulation telling us to revisit the numerical object we trusted first.

## Convergence Should Be an Experiment Too

Numerical convergence deserves the same discipline as Monte Carlo convergence.

For a sequence of grids

$$
m_1<m_2<m_3<\cdots,
$$

record

$$
I_{m_k}(\theta)
$$

at the scientifically important parameter values.

Then examine relative changes such as

$$
\frac{|I_{m_{k+1}}-I_{m_k}|}{|I_{m_{k+1}}|}.
$$

A numerical claim should stop moving before a scientific claim is built on top of it.

If the ordering changes when moving from \(m=100\) to \(m=1{,}000\), then the problem is not ready for Monte Carlo storytelling.

## Local Theory and Finite-Sample Recovery Are Not the Same Object

Even when the numerical calculation is correct, a theoretical diagnostic may not imply the finite-sample ranking we first attach to it.

Suppose a parameter \(\theta\) appears weakly in one analytical feature of a model because

$$
\frac{\partial g(\theta)}{\partial\theta}=0
$$

at some \(\theta_0\).

It is tempting to call \(\theta_0\) a weak-identification point.

But the estimator may use much more than the single feature \(g\).

The full Fisher information is

$$
I(\theta)
=
E\left[
\left(
\frac{\partial}{\partial\theta}
\log p(X;\theta)
\right)^2
\right].
$$

There is no general reason why

$$
g'(\theta_0)=0
$$

must imply

$$
I(\theta_0)
\text{ is locally minimal}.
$$

The local feature may stop carrying information while other parts of the likelihood still do.

A negative recovery simulation can expose exactly this overreach.

## The Most Valuable Negative Result Is Often a Narrower Claim

There is a tendency to view a narrower conclusion as a weaker paper.

I usually think the opposite.

Compare these statements:

> Region B is weakly identified and should generally be excluded.

and

> A particular local diagnostic becomes stationary in region B, but finite-sample recovery follows the full information structure and B is not uniformly inferior.

The second claim is narrower.

It is also much more informative.

It tells us which mathematical mechanism is real, which inference was too broad, and what diagnostic should actually guide practice.

The Monte Carlo study has not destroyed the theory.

It has separated a valid local fact from an invalid global interpretation.

That is scientific progress.

## A Negative Ranking Can Reveal Metric Mismatch

Unexpected results can also arise because two diagnostics measure different things.

Suppose estimator A has smaller median absolute error,

$$
\operatorname{median}|\widehat\theta_A-\theta|
<
\operatorname{median}|\widehat\theta_B-\theta|,
$$

but estimator B has a larger probability of falling within a tolerance:

$$
P\left(
\left|
\frac{\widehat\theta_B}{\theta}-1
\right|
\le0.2
\right)
>
P\left(
\left|
\frac{\widehat\theta_A}{\theta}-1
\right|
\le0.2
\right).
$$

There is no contradiction.

The first summary describes the middle of the error distribution.

The second describes mass inside a fixed interval.

If a theory predicts one but the simulation reports the other, apparent disagreement may be diagnostic mismatch rather than theoretical failure.

The right response is to state which quantity the theory actually predicts.

## Do Not Promote a Pilot Into a Theorem

Small simulation grids are useful for debugging designs and identifying promising regions.

They are dangerous when their patterns are promoted too quickly.

A pilot with

$$
R=5
$$

or

$$
R=20
$$

replications per cell is excellent for finding broken code, obvious failures and approximate effect sizes.

It is usually poor evidence for statements like

> method A is uniformly better across the parameter space.

Uniform statements are expensive.

If the grid has \(K\) cells and each cell has noisy estimates, then the chance that at least one cell produces an apparently contradictory ordering grows with \(K\).

A broad claim therefore needs enough replication to distinguish genuine heterogeneity from the expected scatter of many noisy comparisons.

## Pre-Specify the Question, Not the Answer

A useful discipline is to write down before running the large simulation:

- the estimand or performance quantity;
- the main contrast;
- the grid axes;
- the number of replications;
- the stopping rule, if adaptive;
- which diagnostics will be treated as primary;
- what would count as evidence against the motivating claim.

The last item is particularly important.

If there is no possible simulation outcome that would change the conclusion, then the simulation is decoration.

A falsifiable simulation question might be

$$
H:
\quad
D(\theta)=L_A(\theta)-L_B(\theta)<0
\quad
\text{for all }\theta\in\Theta_0.
$$

Then a well-replicated cell with

$$
D(\theta)>0
$$

is not an inconvenience.

It directly narrows \(H\).

## Freeze the Raw Evidence

One practical way to avoid rewriting the experiment after seeing the result is to keep raw simulation outputs as the evidence layer.

For each replication, store at least

- the seed;
- the exact design cell;
- the estimator settings;
- the raw performance metrics;
- convergence indicators;
- boundary hits or numerical warnings.

Then summaries are derived from those records.

This makes it harder to quietly change the analysis after seeing the first table.

It also makes alternative summaries possible without rerunning the experiment.

The distinction is

$$
\text{raw evidence}
\neq
\text{preferred summary}.
$$

That is especially useful when the preferred summary turns out not to be the right one.

## Fixed Seeds Are Not the Same as Cherry-Picked Seeds

Reproducible simulation often uses fixed seeds.

That is good practice when the seed set is chosen independently of the outcomes and retained as part of the experimental design.

It becomes a problem when seeds are rerun until the visual pattern looks cleaner.

A useful rule is to freeze the replication set before interpreting the results:

$$
\mathcal S
=
\{s_1,\ldots,s_R\}.
$$

If replication must be increased, extend it systematically:

$$
\mathcal S'
=
\mathcal S
\cup
\{s_{R+1},\ldots,s_{R'}\},
$$

rather than replacing inconvenient seeds.

Reproducibility does not rescue selection bias if the random seeds themselves were selected after inspection.

## Separate Debugging Runs From Evidence Runs

During development I expect to run many small, disposable simulations.

Those are engineering tests.

Their purpose is to answer questions such as:

- does the estimator recover an easy case?
- does the code reproduce under a fixed seed?
- do two implementations agree?
- do failure modes occur where expected?

A production Monte Carlo study has a different purpose.

It should have frozen controls, adequate replication, deterministic persistence and a clear claim boundary.

Mixing the two creates a subtle problem: exploratory results start appearing in prose before the experiment has enough precision to support them.

I prefer the separation

$$
\boxed{
\text{pilot}
\rightarrow
\text{freeze design}
\rightarrow
\text{production simulation}
\rightarrow
\text{claim}
}
$$

rather than treating every intermediate output as paper evidence.

## When an Unexpected Result Appears

My own sequence is now roughly this.

### Check the pairing

If two methods or parameter regions are supposed to be compared on the same simulated dataset, verify that they actually are.

Paired designs can reduce variance dramatically, but only if the pairing is real.

### Check the invariants

Things that mathematically cannot change under the intervention should not change in the simulation.

If they do, the experiment is confounded or the code is wrong.

### Quantify Monte Carlo uncertainty

Do not use "probably noise" as an explanation without an MCSE or repeated-seed distribution.

### Check numerical convergence

Refine quadrature, optimiser tolerances, starting values and search bounds where they could influence the conclusion.

### Compare multiple diagnostics

Error magnitude, tolerance rates, optimiser disagreement and boundary behaviour are different objects.

### Revisit the theoretical implication

Ask whether the theorem or local calculation really predicts the finite-sample quantity being simulated.

### Narrow the claim if necessary

This is not the last resort.

It is often the correct result.

## A Scientific Workflow Can Be Written as a Decision Tree

When a simulation contradicts expectation, I find the following conceptual tree useful:

$$
\text{unexpected result}
$$

$$
\downarrow
$$

$$
\begin{array}{c}
\text{Is Monte Carlo error large?}\\
\downarrow\\
\text{increase replication if needed}
\end{array}
$$

then

$$
\begin{array}{c}
\text{Is the numerical calculation converged?}\\
\downarrow\\
\text{refine implementation / quadrature / optimisation}
\end{array}
$$

then

$$
\begin{array}{c}
\text{Does theory predict this exact metric?}\\
\downarrow\\
\text{align diagnostic and estimand}
\end{array}
$$

and finally

$$
\begin{array}{c}
\text{Does the contradiction remain?}\\
\downarrow\\
\boxed{\text{revise the claim}}
\end{array}
$$

The final box is not failure.

It is the part of the workflow that protects the research from becoming self-confirming.

## Negative Results Can Improve the Model

A simulation that contradicts a proposed mechanism can tell us where to look next.

Suppose recovery is unexpectedly good in a supposedly weak region.

Possible explanations include:

- another part of the likelihood carries more information than expected;
- finite-sample bias offsets variance;
- constraints regularize an otherwise weak parameter;
- the parameterization changes effective geometry;
- the optimiser has different sensitivity across regions;
- the theoretical approximation is only local.

Each is a new research question.

The negative result converts a vague story into a sharper mechanism test.

## The Publication Bias Exists Inside Projects Too

We usually discuss publication bias at the journal level.

There is a smaller version inside individual research projects.

Researchers naturally remember the simulation that produced a clean figure and forget the five earlier versions that did not.

Code repositories make it possible to resist that tendency.

Keep the failed pilot.

Keep the old table if it motivated a correction.

Write in the commit or research notes why the interpretation changed.

That history is useful because it records the actual path from conjecture to evidence.

A polished final paper may present only the final argument, but the research process benefits from preserving the route that falsified the earlier one.

## Do Not Turn Every Null Result Into a Grand Conclusion

There is an equal and opposite mistake.

A simulation that fails to distinguish two methods does not prove they are equivalent.

If

$$
\overline D\approx0
$$

with large MCSE, the conclusion is low precision, not equality.

Likewise, one reversal in a noisy cell does not establish that the theoretical ordering is false everywhere.

Negative evidence has to respect its own uncertainty.

The responsible language is often

> we do not observe a stable ordering under this finite-sample design

rather than

> the methods are the same.

The point is not to celebrate negative results indiscriminately.

It is to treat them with the same evidential discipline as positive ones.

## The Best Simulation Studies Are Adversarial

A useful Monte Carlo study should try to break the method or claim being studied.

That means including regimes where assumptions are strained, signal is weak, nuisance parameters are difficult and numerical optimisation is uncomfortable.

An experiment designed only around easy conditions answers a much less interesting question.

If method A wins everywhere because every cell was chosen where A is expected to win, the simulation has little falsifying power.

I prefer to think of the grid as an adversarial design:

$$
\Theta
=
\Theta_{\text{easy}}
\cup
\Theta_{\text{ambiguous}}
\cup
\Theta_{\text{stress}}.
$$

The stress cells are not there to embarrass the method.

They define its boundary of reliability.

## A More Useful Definition of Success

The wrong success criterion for a Monte Carlo project is

$$
\text{simulation agrees with theory}.
$$

A better one is

$$
\boxed{
\text{simulation changes our confidence in a precise claim.}
}
$$

Sometimes confidence increases because the predicted pattern survives broad replication.

Sometimes confidence decreases because the ordering is weak or unstable.

Sometimes the claim changes because the numerical approximation was inadequate.

Sometimes a local theorem survives while a global interpretation does not.

All four are results.

## The Practical Rule

When a Monte Carlo experiment gives an inconvenient answer, do not immediately ask

> how do I make the result cleaner?

Ask

> what would have to be true for this result to be trustworthy?

Then test those conditions.

Check the seeds.

Check the pairing.

Check the MCSE.

Check the numerical convergence.

Check the theoretical implication.

If the result survives, keep it.

The simulation has done exactly what it was supposed to do.

It prevented a plausible mathematical story from becoming a stronger claim than the evidence supports.

That is not a failed Monte Carlo study.

That is a successful one.
