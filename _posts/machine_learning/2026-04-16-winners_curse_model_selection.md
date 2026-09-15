---
permalink: '/machine-learning/winners_curse_model_selection/'
title: "The Winner's Curse in Model Selection: Why the Best Validation Score Is Too Good"
categories:
- Machine Learning
tags:
- Model Evaluation
- Model Selection
- Hyperparameter Tuning
- Statistics
author_profile: false
seo_title: "The Winner's Curse in Model Selection"
seo_description: 'Choosing the best of many models on a validation set and reporting its score overstates its accuracy, by an amount that grows with the number of candidates and shrinks with the validation size. A simulation puts numbers on it and shows what a held-out test set fixes.'
excerpt: >-
  A hundred hyperparameter configurations are compared on a thousand
  validation cases. The winner scores 82.4 percent. Its true accuracy is
  80 percent, and there is a one-in-six chance it is the best configuration
  at all.
summary: >-
  Why the maximum of noisy estimates is biased upward, how large the bias is
  for model selection as a function of the number of candidates and the
  validation set size, how often the selected model is actually the best and
  how far it falls short, why a fresh test set removes the bias entirely, a
  closed-form check against the expected maximum of normal variables, and
  the reporting and design habits that keep tuning honest.
keywords:
  - winner's curse
  - model selection
  - validation set
  - optimism bias
  - hyperparameter search
  - nested cross-validation
  - selection bias
classes: wide
date: '2026-04-16'
why_this_exists: >-
  Tuning reports routinely quote the best validation score as the model's
  expected performance. This post measures how far that number sits above
  the truth in realistic settings, so that teams can size validation sets,
  budget searches and hold out a test set with a clear idea of what they
  are buying.
evidence: >-
  Simulated selection of the best of k candidate configurations with true
  accuracies near 80 percent, scored on validation sets of 200, 1,000 and
  5,000 items, for k from 1 to 500 and spreads of true accuracy from zero to
  five points, 4,000 replications per cell.
methodology: >-
  Records the gap between the winner's validation score and its true
  accuracy, the probability that the winner is the truly best candidate and
  its expected shortfall, the error of a fresh test set of the same size,
  and checks the optimism against the expected maximum of k standard normal
  variables scaled by the binomial standard error.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/constellation.jpg
  og_image: /assets/images/headers/constellation.jpg
  overlay_image: /assets/images/headers/constellation.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/constellation.jpg
  twitter_image: /assets/images/headers/constellation.jpg
---
The search ran overnight: a hundred combinations of learning rate, depth, regularisation and feature set, each trained and scored on the same thousand-case validation split. The best configuration reached 82.4 percent accuracy, two and a half points above the incumbent, and the number went into the slide. In production the model ran at 80 percent, which is what the incumbent did.

Nothing went wrong in production. The validation set did its job and the search did its job. What went wrong is the last step, reading the best score as the winner's accuracy. Any one configuration's validation score is an unbiased estimate of its true accuracy, but the largest of a hundred such estimates is not an unbiased estimate of anything. It is the largest, and it got to be the largest partly by being good and partly by being lucky on that particular thousand cases. The luck does not transfer.

## The Maximum of Noisy Estimates

Suppose $k$ candidates have true accuracies $p_1, \dots, p_k$ and the validation set gives estimates $\hat p_i = p_i + e_i$ with independent errors of standard deviation $\sigma_v = \sqrt{p(1-p)/n}$. Selecting the largest $\hat p_i$ favours candidates with large positive $e_i$. If the true accuracies were all equal, the winner would be the candidate with the largest error, and its validation score would exceed its true accuracy by $\sigma_v$ times the expected maximum of $k$ standard normal variables: about 1.16 for five candidates, 1.87 for twenty, 2.51 for a hundred.

With a thousand validation cases at 80 percent accuracy, $\sigma_v$ is 1.26 points. The expected optimism of the best of a hundred equal candidates is therefore about 3.2 points. Real candidates differ, which reduces the bias, because a truly better candidate wins on merit more often than on luck. The simulation measures both regimes.

## A Simulation

Each replication draws $k$ true accuracies from a normal distribution centred on 80 percent with a one-point spread, scores each on a validation set of $n$ items as a binomial count, and selects the largest score. Three quantities are recorded: the winner's validation score minus its true accuracy, whether the winner is in fact the best candidate, and how far its true accuracy falls short of the best.

```python
import numpy as np

rng = np.random.default_rng(0)
base_acc, spread = 0.80, 0.01       # true accuracies of candidate configurations

def select(k, n_val, reps=4000, spread=spread, r=rng, n_test=None):
    """Pick the best of k configurations on a validation set of n_val items."""
    opt = regret = is_best = test_gap = 0.0
    for _ in range(reps):
        true = np.clip(r.normal(base_acc, spread, k), 0.5, 0.99)
        val = r.binomial(n_val, true) / n_val
        w = val.argmax()
        opt += val[w] - true[w]            # validation score minus the winner's true accuracy
        regret += true.max() - true[w]     # distance from the truly best configuration
        is_best += w == true.argmax()
        if n_test:
            test_gap += r.binomial(n_test, true[w]) / n_test - true[w]
    return opt / reps, regret / reps, is_best / reps, test_gap / reps

print("optimism of the winner's validation score, accuracy points")
print(f"{'k':>5}{'n=200':>10}{'n=1000':>10}{'n=5000':>10}")
for k in (1, 2, 5, 20, 100, 500):
    print(f"{k:>5}" + "".join(f"{select(k, n)[0]*100:>10.2f}" for n in (200, 1000, 5000)))
```

**Optimism of the winning score**, in accuracy points, by number of candidates and validation size.

| Candidates | 200 cases | 1,000 cases | 5,000 cases |
| --- | --- | --- | --- |
| 1 | 0.1 | 0.0 | 0.0 |
| 2 | 1.4 | 0.6 | 0.2 |
| 5 | 3.0 | 1.1 | 0.3 |
| 20 | 4.8 | 1.8 | 0.5 |
| 100 | 6.3 | 2.4 | 0.7 |
| 500 | 7.6 | 2.9 | 0.8 |

A single candidate has no optimism: its score is an honest estimate. Comparing two already introduces half a point at a thousand cases. The hundred-configuration search on a thousand cases, the situation in the opening, overstates the winner by 2.4 points, which is the whole of the improvement the slide reported. On two hundred cases the same search overstates by more than six points, enough to make a random configuration look like a breakthrough.

The bias grows slowly in $k$, roughly with the expected maximum of $k$ normals, so a search five times larger adds about half a point, and shrinks with $1/\sqrt{n}$, so five times the validation data cuts it by more than half. The validation set is the lever.

![Average gap between the winning configuration's validation accuracy and its true accuracy, against the number of configurations compared, for validation sets of 200, 1,000 and 5,000 items. The gap grows with the number of candidates and shrinks with validation size; a fresh test set has no gap.](/assets/images/figures/winners_curse_optimism.png){: width="1152" height="672" loading="lazy"}

**Is the winner the best model?** Optimism is the bias in the reported number. The second cost is that the search does not necessarily return the best configuration, and a validation set too small to rank candidates gives back a merely adequate one.

```python
print("probability the winner is the truly best configuration, and its shortfall (points)")
for k in (2, 5, 20, 100):
    o1, r1, b1, _ = select(k, 1000)
    o5, r5, b5, _ = select(k, 5000)
    print(f"k={k:>4}: n=1000 P(best) {b1:.0%} shortfall {r1*100:.2f}   n=5000 P(best) {b5:.0%} shortfall {r5*100:.2f}")
```

| Candidates | Winner is the best, 1,000 cases | Shortfall (points) | Winner is the best, 5,000 cases | Shortfall (points) |
| --- | --- | --- | --- | --- |
| 2 | 71% | 0.2 | 84% | 0.1 |
| 5 | 47% | 0.5 | 68% | 0.2 |
| 20 | 28% | 0.7 | 54% | 0.2 |
| 100 | 16% | 1.0 | 42% | 0.3 |

With a hundred candidates and a thousand cases, the winner is the best configuration one time in six, and on average it is a full point worse than the best one available. That is a second, separate loss: the report overstates the winner by 2.4 points, and the winner itself is a point below what the search could have delivered with enough validation data to see the differences. Both losses fall when the validation set grows, and neither falls when the search grows.

**A fresh test set removes the bias.** The remedy is old: select on one sample, estimate on another. The test set was not used to choose, so the winner's score on it has no selection bias.

```python
print("held-out test set, k = 100")
for n in (200, 1000, 5000):
    o, _, _, tg = select(100, n, n_test=n)
    print(f"n_val = n_test = {n:>5}: validation optimism {o*100:+.2f}, test-set error {tg*100:+.2f}")
```

| Cases in each set | Validation optimism | Test-set error |
| --- | --- | --- |
| 200 | +6.3 | +0.0 |
| 1,000 | +2.4 | +0.0 |
| 5,000 | +0.7 | +0.0 |

The test set is unbiased at every size. It is still noisy, with a standard error of 1.3 points at a thousand cases, but the noise is symmetric and reported honestly by a confidence interval, whereas the validation optimism is a one-directional error that no interval on the validation score can reveal.

## When Candidates Really Differ

The one-point spread of true accuracies is typical of a hyperparameter grid, most of whose cells are variations on the same model. When the candidates differ more, the best one wins on merit and the curse weakens.

```python
print("spread of true accuracy, n = 1000")
for s in (0.0, 0.01, 0.03, 0.05):
    o20, _, b20, _ = select(20, 1000, spread=s)
    o100, _, b100, _ = select(100, 1000, spread=s)
    print(f"spread {s:.2f}: k=20 optimism {o20*100:.2f} P(best) {b20:.0%}   k=100 optimism {o100*100:.2f} P(best) {b100:.0%}")
```

| Spread of true accuracy | 20 candidates, optimism | Winner is the best | 100 candidates, optimism | Winner is the best |
| --- | --- | --- | --- | --- |
| 0 points | 2.3 | 6% | 3.1 | 1% |
| 1 point | 1.8 | 29% | 2.4 | 16% |
| 3 points | 0.8 | 65% | 0.9 | 58% |
| 5 points | 0.4 | 80% | 0.4 | 77% |

At zero spread, all candidates identical, the optimism is 3.1 points for a hundred candidates, against the 3.2 predicted by $\sigma_v$ times the expected maximum of a hundred standard normals, which is the check that the simulation and the closed form agree. As the spread grows the winner is more often the genuine best, and its score is less inflated, because a candidate three points ahead does not need luck to win. The practical reading is that the curse is worst exactly where searches are largest: fine grids over similar configurations, where the differences the search is trying to resolve are smaller than the noise of the validation set.

## Corrections Short of a Test Set

A held-out test set is the clean answer and the one to use when the data allow it. When they do not, three approaches reduce the bias without a fresh sample.

**Nested cross-validation** runs the whole selection procedure inside each outer fold and scores the selected model on the outer held-out fold. The outer estimate is unbiased for the performance of the procedure, "search over these candidates and pick the best", which is the thing that will be deployed. It costs $k$ times the outer folds in training runs.

**Shrinkage** treats the observed scores as noisy draws around the true accuracies and pulls the winner toward the mean of the candidates by a factor set by the ratio of validation noise to candidate spread. For the hundred-candidate, one-point-spread, thousand-case case, the true spread is 1 point and the noise is 1.26, so the winner's excess over the candidate mean should be discounted by roughly $1 - 1^2/(1^2 + 1.26^2) \approx 0.6$. This is Efron's empirical Bayes correction and it recovers most of the bias when the spread can be estimated, which requires more than a handful of candidates.

**Report the whole distribution.** A search that returns a hundred scores has measured the noise: the spread of the scores across candidates that should be nearly equivalent is the spread of validation error. Reporting the winner alongside the distribution of runner-up scores makes the reader ask whether the winner stands out of it, and a winner within one standard error of the tenth-best configuration has not been shown to be better than any of them.

## What to Do

1. **Never report a selected score as an estimate.** The best validation score is the maximum of the search; the model's expected accuracy needs a sample the search did not see.
2. **Hold out a test set before tuning starts**, size it for the precision the decision needs, and score the winner on it once.
3. **Size the validation set for the search, not the model.** With $k$ candidates the optimism is about $\sqrt{p(1-p)/n}$ times the expected maximum of $k$ normals; if that is larger than the smallest improvement worth having, the validation set cannot resolve the search.
4. **Prefer fewer, more different candidates** over fine grids. A hundred near-identical configurations bring a large curse and little to choose between.
5. **Use nested cross-validation** when data are too scarce for a separate test set, and report the outer-fold score as the estimate.
6. **Read the runner-up scores.** A winner that is not clearly separated from the field has not been selected on evidence, and any configuration in the field would have done as well.

## References

- Jensen, D. D., & Cohen, P. R. (2000). Multiple comparisons in induction algorithms. *Machine Learning*, 38, 309-338.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
- Efron, B. (2011). Tweedie's formula and selection bias. *Journal of the American Statistical Association*, 106(496), 1602-1614.
- Tibshirani, R. J., & Tibshirani, R. (2009). A bias correction for the minimum error rate in cross-validation. *The Annals of Applied Statistics*, 3(2), 822-829.
- Thaler, R. H. (1988). Anomalies: the winner's curse. *Journal of Economic Perspectives*, 2(1), 191-202.
