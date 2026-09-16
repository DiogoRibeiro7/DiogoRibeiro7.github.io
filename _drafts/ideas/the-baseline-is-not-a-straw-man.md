---
author_profile: false
categories:
- Data Science
classes: wide
excerpt: A good baseline is not a weak opponent designed to make a sophisticated model look impressive. It is the scientific control that tells us what complexity actually added.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- baselines
- benchmarks
- ablation studies
- model comparison
- machine learning evaluation
- scientific method
seo_description: Why strong baselines, matched tuning budgets, uncertainty estimates, and ablations are necessary to know whether model complexity really adds value.
seo_title: The Baseline Is Not a Straw Man
seo_type: article
summary: A baseline-first approach to modelling in which complexity must justify itself against credible references under fair evaluation.
tags:
- Data Science
- Modelling
- Machine Learning
title: 'The Baseline Is Not a Straw Man'
---

A baseline is often treated as the model we intend to beat.

That is too weak a role.

A good baseline is a **scientific control**. It tells us which part of the problem was already solvable before we added complexity.

If the baseline is deliberately poor, the comparison may make a sophisticated model look impressive without teaching us much.

The question is not merely

> Did the new model score higher?

It is

$$
\boxed{
\text{What did the extra complexity actually contribute?}
}
$$

That question is harder, because it requires a fair comparison, a credible reference, and some account of uncertainty.

## Baseline, benchmark, ablation, and control are not the same thing

These terms are often used loosely, but they play different roles.

### Naive reference

A naive reference answers:

> Is this problem easier than doing something trivial?

Examples include:

- predicting the historical mean;
- persistence forecasting;
- majority-class prediction;
- last-observation-carried-forward;
- random ranking.

A naive reference is useful, but beating it is rarely enough.

### Credible baseline

A credible baseline is the strongest simple method that a competent practitioner would reasonably try before introducing the new idea.

Examples might include:

- regularized linear or logistic regression;
- a seasonal forecasting model;
- gradient boosting on well-defined tabular features;
- a transparent state-space model;
- a hand-designed temporal representation paired with a conventional classifier.

This is the comparison that usually matters most.

### Benchmark

A benchmark is a standardized evaluation setting: dataset, split, metric, protocol, and often a collection of reference methods.

A benchmark can contain weak or strong baselines. It is the **evaluation arena**, not the method itself.

### Ablation or control

An ablation asks what happens when part of a model or pipeline is removed.

If a model contains components $A$, $B$, and $C$, then an ablation might compare

$$
A+B+C
$$

with

$$
A+B,
\qquad
A+C,
\qquad
B+C.
$$

That is different from comparing the full method against an external baseline. Ablations tell us which components matter **inside the proposed system**.

A strong evaluation often needs all four roles.

## Start with the estimand and decision problem

Before choosing an algorithm, define what is being estimated or predicted and how success will be judged.

A sensible sequence is

$$
\text{estimand or decision target}
\rightarrow
\text{credible baseline}
\rightarrow
\text{validation protocol}
\rightarrow
\text{uncertainty}
\rightarrow
\text{complexity}.
$$

Starting with the model reverses the logic. We end up asking where a technique can be applied rather than what the problem requires.

This matters because the “best” model depends on the objective.

A classifier optimized for AUROC may not be the best model when false negatives are very expensive. A forecasting model with lower average RMSE may still be inferior if it is badly calibrated in the region where decisions are made.

The baseline should therefore be chosen relative to the actual decision problem, not merely because it is familiar.

## A strong baseline should capture obvious structure

A credible baseline should encode the parts of the problem that are already easy to justify.

For time series, that may mean trend, seasonality, and persistence.

For tabular prediction, it may mean a regularized generalized linear model or a tuned tree ensemble.

For longitudinal data, it may mean scientifically meaningful summary features plus a transparent clustering or classification model.

The point is not that simple methods are always better.

The point is that we should not attribute gains to sophisticated machinery until the obvious structure has already been accounted for.

A weak baseline creates an easy opponent. A strong baseline creates an informative experiment.

## Fair comparison requires fair tuning

One of the easiest ways to create the illusion of progress is to give the new method a large optimization budget and compare it with an untuned baseline.

Suppose the proposed model receives:

- extensive hyperparameter search;
- several random seeds;
- architecture selection;
- early stopping;
- learning-rate schedules;
- augmentation tuning;
- careful feature preprocessing.

Meanwhile the baseline is run once with default settings.

The observed gap now combines at least two effects:

$$
\text{method effect}
+
\text{optimization-budget effect}.
$$

Those are not the same thing.

The recurrent-neural-network study by Melis, Dyer, and Blunsom is a useful example: after careful hyperparameter tuning, standard recurrent architectures were much more competitive than headline comparisons had suggested.

Lucic and colleagues found a similar problem in generative models: relative rankings changed substantially with hyperparameter optimization and random initialization.

The practical rule is simple:

$$
\boxed{
\text{Compare methods under comparable optimization effort.}
}
$$

That does not mean every method needs the exact same number of trials. Different model classes have different search spaces. It means the evaluation should not systematically handicap the baseline.

## Variance is part of the comparison

A single score is rarely enough for stochastic learning systems.

Suppose two methods produce

$$
0.842
\quad\text{and}\quad
0.848
$$

on one evaluation run.

That difference is meaningless without knowing how much variation comes from:

- random initialization;
- data splitting;
- minibatch order;
- stochastic augmentation;
- hyperparameter selection;
- finite test-sample uncertainty.

A fair comparison should therefore report a distribution or uncertainty measure rather than treating the largest observed number as the result.

Bouthillier and colleagues make this point directly in their work on variance in machine-learning benchmarks: benchmark scores contain multiple sources of randomness, and ignoring them can overstate differences between methods.

A useful decomposition is

$$
\widehat M
=
M
+
\varepsilon_{\text{data}}
+
\varepsilon_{\text{training}}
+
\varepsilon_{\text{tuning}}
+
\varepsilon_{\text{evaluation}}.
$$

Not every experiment can estimate every component separately, but pretending they do not exist is worse.

## The test set is not a tuning instrument

Repeatedly trying variants until one performs well on the test set turns the test set into part of the optimization loop.

This is easy to do informally:

1. train model;
2. inspect test result;
3. adjust preprocessing or architecture;
4. run again;
5. keep the best version.

At that point the final test score is optimistic.

The same problem can affect baselines. If the new method is selected from fifty experiments but the baseline is a single fixed run, the comparison is asymmetric in a different way.

A cleaner design separates:

$$
\text{development}
\quad\perp\quad
\text{final evaluation}.
$$

Nested validation, held-out evaluation, or a pre-specified protocol can all help, depending on the setting.

The exact mechanism matters less than the principle: **the final comparison should not be the same feedback signal used to design the competing models**.

## Complexity should be a hypothesis

Rather than assuming a more flexible model will help, formulate what it is expected to capture.

For example,

$$
H_1:
\text{nonlinear interactions contain predictive information absent from the additive baseline.}
$$

or

$$
H_1:
\text{long-range temporal dependence matters beyond the short-memory baseline.}
$$

or

$$
H_1:
\text{learned representations preserve discriminative structure lost by hand-designed summaries.}
$$

Now complexity has a purpose that can be falsified.

If the complex model wins, inspect whether it wins where the hypothesis predicted.

If a transformer improves only on examples where the simple model already fails for obvious data-quality reasons, that does not establish the proposed mechanism.

If a nonlinear model beats a linear baseline only after adding features unavailable to the baseline, the experiment has changed more than one thing at a time.

A strong comparison isolates the reason complexity is supposed to help.

## Ablations answer a different question

Suppose a new model beats a strong baseline.

We still do not know why.

If the proposed system contains:

- a new encoder;
- a special loss;
- an auxiliary objective;
- a larger parameter budget;
- additional preprocessing;
- more training data;

then the headline comparison changes several variables simultaneously.

Ablations help decompose the gain.

For instance:

| Variant | Encoder | New loss | Extra data | Score |
| --- | --- | --- | --- | ---: |
| Baseline | no | no | no | 0.81 |
| A | yes | no | no | 0.82 |
| B | yes | yes | no | 0.825 |
| Full | yes | yes | yes | 0.84 |

Now the experiment begins to explain **where the improvement came from**.

The baseline tells us whether the whole system is better. The ablation tells us which parts deserve credit.

## Complexity can improve the wrong metric

Suppose a forecasting model reduces RMSE by 2% but requires ten times the infrastructure, is poorly calibrated where decisions matter, and degrades sharply under distribution shift.

Was the model better?

That depends on the decision problem.

Model comparison may need to include:

- calibration;
- stability;
- latency;
- memory use;
- inference cost;
- training cost;
- interpretability constraints;
- failure modes;
- uncertainty quality;
- robustness under shift.

For production systems, one useful way to think about the comparison is

$$
\text{net value}
=
\text{predictive gain}
-
\text{operational cost}
-
\text{risk cost}.
$$

That expression is schematic rather than universal, but it captures an important point: a statistically significant predictive improvement need not be operationally important.

## Strong baselines expose leakage and weak validation

Simple models are diagnostic tools.

If a trivial baseline performs implausibly well, investigate:

- target leakage;
- temporal leakage;
- duplicate observations;
- train-test contamination;
- preprocessing fitted on the full dataset;
- target-derived features;
- entity overlap across splits.

Likewise, if every sophisticated model performs dramatically better than a sensible baseline, that gap deserves explanation.

Large gains are possible. They are also claims that should survive scrutiny.

A baseline is therefore useful even when it loses badly. It helps us understand whether the problem itself, the data pipeline, or the model is doing the work.

## A failed complex model is a result

Suppose a neural model and a simple statistical model perform essentially the same after careful validation and matched tuning effort.

That is not a disappointing experiment.

It may imply that:

- the simpler representation already captures most of the available signal;
- the sample is too small for the flexible model to estimate additional structure reliably;
- the proposed mechanism does not matter in this regime;
- the evaluation metric cannot detect the claimed advantage;
- the additional complexity mainly increases variance.

Each possibility is scientifically useful.

The wrong response is to keep tuning until a small test-set difference appears.

Hand’s “illusion of progress” argument is still relevant here: improvements in classifier technology can look more substantial than they are when evaluations are performed on convenient benchmarks or under assumptions that do not reflect the actual application distribution.

Progress should therefore be judged against the problem we care about, not only against the score table we inherited.

## Benchmark leadership is not the same as methodological superiority

Benchmarks are useful because they standardize comparison.

They are also incomplete abstractions.

A benchmark fixes choices about:

- data distribution;
- labels;
- metric;
- split;
- preprocessing;
- computational budget;
- acceptable external data;
- sometimes even reporting conventions.

A method can be excellent on that benchmark and still be a poor choice elsewhere.

Conversely, a method may look modest on a leaderboard while being preferable because it is more stable, cheaper, interpretable, calibrated, or robust.

So I would interpret benchmark results as conditional statements:

$$
\boxed{
\text{Under this dataset, metric, protocol, and budget, method A outperformed method B.}
}
$$

That is much stronger scientifically than saying simply “A is better.”

## A practical evaluation ladder

For a new modelling idea, I would usually want something like the following.

### 1. Naive reference

Can we beat a trivial strategy?

### 2. Strong transparent baseline

Can we beat a credible conventional method that captures obvious structure?

### 3. Fair tuning

Were both systems given serious, documented optimization effort?

### 4. Repeated evaluation

Does the result survive randomness in training, splitting, or sampling?

### 5. Ablation

Which components of the proposed method create the gain?

### 6. Mechanism check

Does the method improve where the motivating hypothesis predicted?

### 7. Robustness

Does the result survive reasonable changes in preprocessing, seeds, splits, metrics, and operating conditions?

### 8. Operational comparison

Is the gain worth the additional cost, latency, maintenance, and risk?

Only then does the headline number begin to tell a useful story.

## The practical rule

I would not ask

> What is the strongest model we can build?

I would ask

> What is the simplest credible model that captures the structure we can currently defend, and what evidence says we need more?

That changes the role of the baseline completely.

$$
\boxed{
\text{The baseline is not the opponent. It is part of the experimental design.}
}
$$

If the complex method wins, the baseline tells us the gain was not trivial.

If it ties, the baseline tells us complexity may not be necessary.

If it loses, the baseline tells us something even more valuable: our modelling hypothesis was wrong, at least under the conditions we tested.

That is what a useful control is supposed to do.

## References

- Hand DJ. *Classifier Technology and the Illusion of Progress*. Statistical Science. 2006;21(1):1–14. DOI: [10.1214/088342306000000060](https://doi.org/10.1214/088342306000000060).
- Melis G, Dyer C, Blunsom P. *On the State of the Art of Evaluation in Neural Language Models*. International Conference on Learning Representations, 2018.
- Lucic M, Kurach K, Michalski M, Gelly S, Bousquet O. *Are GANs Created Equal? A Large-Scale Study*. Advances in Neural Information Processing Systems, 2018.
- Bouthillier X, Delaunay P, Bronzi M, et al. *Accounting for Variance in Machine Learning Benchmarks*. Proceedings of Machine Learning and Systems, 2021.
- Sculley D, Holt G, Golovin D, et al. *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems, 2015.
