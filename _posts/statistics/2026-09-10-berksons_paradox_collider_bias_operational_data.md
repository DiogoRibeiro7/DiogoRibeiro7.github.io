---
permalink: '/statistics/berksons_paradox_collider_bias_operational_data/'
title: "Berkson's Paradox: How Selecting the Cases Worth Looking At Invents Correlations"
categories:
- Statistics
tags:
- Causal Inference
- Selection Bias
- Data Quality
- Statistics
author_profile: false
seo_title: "Berkson's Paradox and Collider Bias in Operational Data"
seo_description: 'When records enter a dataset because they cleared a bar that depends on two things, those two things become negatively correlated in the data even if they are independent in the world. A simulation of escalated support tickets shows the correlation appear, when regressions survive it, and when a model trained on the selected cases fails.'
excerpt: >-
  Among escalated tickets, severity and customer value are correlated at
  minus 0.55. Across all tickets they are independent. Nothing about the
  tickets changed; the escalation rule did the correlating.
summary: >-
  What a collider is and why conditioning on one creates association, a
  simulation of support tickets escalated on the sum of severity and
  customer value showing the induced negative correlation and how it
  strengthens with selectivity, which analyses on the selected records
  survive (a regression that includes every selection input) and which do
  not (marginal relationships, and any regression once the selection used
  something unrecorded), the cost to a prediction model trained on the
  selected cases, and how to recognise and defuse the selection in
  practice.
keywords:
  - Berkson's paradox
  - collider bias
  - selection bias
  - conditioning on a collider
  - sample selection
  - spurious correlation
  - training data selection
classes: wide
date: '2026-09-10'
why_this_exists: >-
  Operational datasets are built from the cases somebody chose to act on:
  escalated tickets, approved loans, admitted patients, funded projects,
  flagged transactions. Every one of those choices is a collider, and
  analysts routinely study the selected set as if it were a sample. This
  post shows exactly what that does, on a simulation small enough to
  reason about, and separates the analyses it ruins from the ones it
  spares.
evidence: >-
  A simulated population of 200,000 support tickets with independent
  severity and customer value, escalated when a score built from the two
  clears a threshold, at selection shares from 50 to 1 percent; a second
  version in which the escalation score also uses an unrecorded difficulty
  that drives resolution time.
methodology: >-
  Measures the correlation between the two attributes among escalated
  tickets against the share escalated, compares regressions of resolution
  time on the attributes in the population and among escalated tickets
  with and without each attribute and with the selection score as a
  control, and evaluates a model trained on escalated tickets against one
  trained on all tickets, overall and on the subgroups where they differ.
reviewed_at: '2026-09-11'
header:
  image: /assets/images/headers/photo-code.jpg
  og_image: /assets/images/headers/photo-code.jpg
  overlay_image: /assets/images/headers/photo-code.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-code.jpg
  twitter_image: /assets/images/headers/photo-code.jpg
---
The support analytics team looks at escalated tickets, because those are the ones that cost money. In that set, the severity of the problem and the value of the customer are strongly and negatively correlated: the high-value customers' escalations are mostly minor, and the severe escalations mostly come from small accounts. Someone proposes an explanation involving how account managers shield their large clients. Someone else builds a model on the escalated tickets and finds that customer value predicts faster resolution.

Across all tickets, severity and customer value are independent. The correlation of minus 0.55 in the escalated set was made by the escalation rule. A ticket gets escalated when severity plus customer value clears a bar, and a set defined that way contains severe tickets from small accounts and mild tickets from large accounts, but almost no mild tickets from small accounts, because those never clear the bar, and almost no severe tickets from large accounts, because there are few of them to begin with. Take a square of independent points, keep only the corner above a diagonal line, and the corner is a negatively sloped strip.

This is Berkson's paradox. Berkson described it in 1946 for hospital patients, where two independent diseases appear negatively associated among the admitted because either disease can get a patient admitted. The general form is now called collider bias: when a variable is caused by two others, conditioning on it, by selection, stratification or adjustment, creates an association between the two causes.

## A Simulation

Two hundred thousand tickets, each with a severity and a customer value drawn independently from standard normal distributions. Escalation depends on their sum plus some judgement noise, and the top 15 percent of scores are escalated. Resolution time depends on severity alone.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200000
severity = rng.normal(0, 1, n)
value = rng.normal(0, 1, n)
score = severity + value + rng.normal(0, 0.5, n)          # what the escalation decision is based on
escalated = score > np.quantile(score, 0.85)              # the top 15 percent get escalated

print(f"all tickets:       corr(severity, value) = {np.corrcoef(severity, value)[0, 1]:+.3f}")
print(f"escalated tickets: corr(severity, value) = {np.corrcoef(severity[escalated], value[escalated])[0, 1]:+.3f}")
for q in (0.5, 0.7, 0.85, 0.95, 0.99):
    sel = score > np.quantile(score, q)
    print(f"  escalate top {1-q:>4.0%}: {np.corrcoef(severity[sel], value[sel])[0, 1]:+.3f}")
```

| Tickets kept | Correlation of severity and customer value |
| --- | --- |
| All | -0.003 |
| Top 50% by score | -0.40 |
| Top 30% | -0.48 |
| Top 15% (the escalated set) | -0.55 |
| Top 5% | -0.61 |
| Top 1% | -0.65 |

The correlation is zero in the population and minus 0.55 among the escalated, and it gets stronger the more selective the rule is. There is no mechanism in the simulation linking the two attributes; the rule alone produces the pattern, and it would produce it for any two independent inputs to any selection that depends on their sum.

![Correlation between severity and customer value among escalated tickets against the share of tickets escalated, when escalation depends on the sum of the two. The population correlation is zero; the correlation in the selected set falls from minus 0.40 at 50 percent selected to minus 0.65 at 1 percent.](/assets/images/figures/berkson_selection_correlation.png){: width="1152" height="672" loading="lazy"}

The rule does not have to be a sum. Escalating a ticket if severity is high *or* customer value is high, the hospital version, keeps 29 percent of tickets and gives a correlation of minus 0.56 among them. Any rule under which one attribute can substitute for the other in getting a record selected has the same effect.

## Which Analyses Survive

Not every analysis on the escalated tickets is wrong, and it matters to know which. Resolution time in the simulation is four hours plus three hours per unit of severity, plus noise, and customer value has no effect on it.

```python
hours = 4 + 3 * severity + rng.normal(0, 2, n)

def ols(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]

both = np.column_stack([severity, value])
print("population, both:          ", ols(both, hours).round(2))
print("escalated, both:           ", ols(both[escalated], hours[escalated]).round(2))
print("escalated, severity alone: ", ols(severity[escalated, None], hours[escalated]).round(2))
print("escalated, value alone:    ", ols(value[escalated, None], hours[escalated]).round(2))
print("escalated, both + score:   ", ols(np.column_stack([both, score])[escalated], hours[escalated]).round(2))
```

| Regression of resolution hours | Severity coefficient | Customer value coefficient |
| --- | --- | --- |
| All tickets, both attributes | +3.00 | -0.01 |
| Escalated only, both attributes | +2.97 | -0.04 |
| Escalated only, severity alone | +2.99 | |
| Escalated only, customer value alone | | -1.66 |
| Escalated only, both attributes and the escalation score | +2.93 | -0.07 |

The regression with both attributes is fine on the escalated tickets: 2.97 and minus 0.04 against a truth of 3 and 0. Selection that depends only on variables in the model does not bias the conditional mean of the outcome given those variables, because within any combination of severity and value, which tickets got escalated is unrelated to the outcome. What the selection distorts is the joint distribution of the predictors, and that is where the damage shows: the regression on customer value alone gives minus 1.66 hours per unit, a large and entirely spurious effect. It arises because, among escalated tickets, high value means low severity, and low severity means fast resolution. The team member who found that "customer value predicts faster resolution" found the escalation rule.

Adding the escalation score as a control changes little here, because the model already contains the score's inputs. It becomes the useful move in the next case.

## When the Selection Used Something You Do Not Have

Escalation decisions are made by people, and people use information that never reaches the table. Suppose the agent's judgement of how hard a ticket will be to resolve enters the escalation score, and also, because it is a real property of the ticket, drives the resolution time.

```python
difficulty = rng.normal(0, 1, n)                                  # judged by the agent, never logged
score2 = severity + value + difficulty + rng.normal(0, 0.5, n)
esc2 = score2 > np.quantile(score2, 0.85)
hours2 = 4 + 3 * severity + 2 * difficulty + rng.normal(0, 2, n)

print("population, both:", ols(both, hours2).round(2))
print("escalated, both: ", ols(both[esc2], hours2[esc2]).round(2))
```

| Regression of resolution hours, both attributes | Severity | Customer value |
| --- | --- | --- |
| All tickets | +3.00 | +0.00 |
| Escalated only | +2.00 | -0.96 |

Now the regression with both attributes is biased too, and in both coefficients. Severity's effect is understated by a third and customer value acquires an effect of almost minus one hour per unit. The mechanism is the same collider, one level up: among escalated tickets, a high-value ticket got in with less difficulty than a low-value one, difficulty drives hours, and so value appears to reduce hours. Nothing in the recorded data can remove this, because the variable that would need to be controlled for was never recorded. The only fix is upstream: analyse all tickets, not the escalated ones, or record the judgement that went into the escalation.

## The Model That Was Trained on the Selected Cases

The practical version of the problem is a prediction model. Resolution-time models are trained on escalated tickets because those have the careful records, and then applied to every ticket.

```python
def fit_predict(mask, y):
    X = np.column_stack([np.ones(n), severity, value])
    b = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
    return X @ b

pred_sel = fit_predict(esc2, hours2)                   # trained on escalated tickets
pred_all = fit_predict(np.ones(n, bool), hours2)       # trained on all tickets
rmse = lambda p: np.sqrt(np.mean((hours2 - p) ** 2))
print(f"RMSE on all tickets: escalated-trained {rmse(pred_sel):.2f}, population-trained {rmse(pred_all):.2f}")
hi, lo = value > 1.5, severity < -1
print(f"high-value tickets: mean error {np.mean(pred_sel[hi] - hours2[hi]):+.2f} vs {np.mean(pred_all[hi] - hours2[hi]):+.2f}")
print(f"low-severity tickets: mean error {np.mean(pred_sel[lo] - hours2[lo]):+.2f} vs {np.mean(pred_all[lo] - hours2[lo]):+.2f}")
```

| Applied to all tickets | Trained on escalated tickets | Trained on all tickets |
| --- | --- | --- |
| Root mean squared error | 4.65 hours | 2.83 hours |
| Mean error on high-value tickets | +1.53 hours | -0.02 hours |
| Mean error on low-severity tickets | +4.96 hours | +0.02 hours |

The model from the escalated set is not slightly worse; its error is two thirds larger overall, and on the tickets least like the training set, the routine low-severity ones, it overpredicts resolution time by five hours. Those are the tickets whose absence from the training data defined the selection. In the first version of the simulation, where escalation depended only on recorded attributes, the same model was as good as one trained on everything, because the conditional mean it was learning was the same in the selected set. The difference between the two versions is invisible from inside the escalated data, which is the point.

## Recognising the Pattern

A collider is any variable that two others feed into, and a dataset is a collider whenever membership in it depends on more than one thing. The pattern is easy to recognise once the question is asked in the right form: *why is this record in front of me?*

**Approved applications.** Loans, insurance, admissions. An applicant with a weak score and strong income can be approved, and so can the reverse; among the approved, score and income are negatively related, and a default model trained on the approved book learns relationships that do not hold for applicants.

**Flagged or reviewed cases.** Fraud alerts, quality inspections, audits. The review queue is the set that cleared a threshold on several signals, and the signals are negatively associated within it.

**Published or funded work.** Studies enter the literature by being either novel or rigorous; among published studies the two trade off, whether or not they do among studies conducted.

**Hired candidates, surviving products, retained customers.** Anything that has passed a multi-criterion filter. The talent-and-polish version of the paradox, where the two are independent in the population and negatively correlated among people hired on their sum, is the same computation with a stricter bar.

```python
sel = (severity > 1) | (value > 1)                     # escalate on either criterion
print(f"either rule: {sel.mean():.0%} selected, corr {np.corrcoef(severity[sel], value[sel])[0, 1]:+.3f}")
talent, polish = rng.normal(0, 1, n), rng.normal(0, 1, n)
for q in (0.9, 0.99):
    hired = talent + polish > np.quantile(talent + polish, q)
    print(f"hire top {1-q:.0%}: corr among hired {np.corrcoef(talent[hired], polish[hired])[0, 1]:+.3f}")
```

Hiring a tenth of applicants on the sum of two independent qualities gives a correlation of minus 0.71 among the hired; hiring a hundredth gives minus 0.83. The more exclusive the filter, the more the survivors look as if the qualities trade off.

The defensive habits follow from the mechanism. Analyse the population the selection was applied to, when it exists. When only the selected set exists, treat every marginal relationship between selection inputs as suspect, keep the selection inputs in any model together, and record the inputs to the decision, especially the human ones, so that the selection can be conditioned on. And when a correlation in operational data seems to demand a story, first check whether the data were selected on both of the things being correlated.

## What to Do

1. **Ask how records entered the dataset**, and list the variables the selection depended on. If the outcome, or anything correlated with it, is on the list, no analysis of the selected set alone will recover the population relationship.
2. **Never read a marginal correlation between two selection inputs** in a selected dataset as a fact about the world.
3. **Keep all selection inputs in the model together.** A regression or prediction model that conditions on everything the selection used is unbiased for the conditional mean; one that drops any of them is not.
4. **Record the judgement that went into the decision.** If the escalation, approval or flag used a human assessment, log it; it is the variable that makes the selected data usable later.
5. **Train on the population the model will score**, or, when the labels exist only for the selected cases, say so and test the model on a sample of unselected cases before trusting it on them.
6. **Prefer selection-aware methods** when the selection cannot be undone: sample-selection models when the selection inputs are recorded, and inverse-probability weighting by the selection probability when it can be estimated.

## References

- Berkson, J. (1946). Limitations of the application of fourfold table analysis to hospital data. *Biometrics Bulletin*, 2(3), 47-53.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Elwert, F., & Winship, C. (2014). Endogenous selection bias: the problem of conditioning on a collider variable. *Annual Review of Sociology*, 40, 31-53.
- Hernán, M. A., Hernández-Díaz, S., & Robins, J. M. (2004). A structural approach to selection bias. *Epidemiology*, 15(5), 615-625.
- Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.
- Griffith, G. J., Morris, T. T., Tudball, M. J., et al. (2020). Collider bias undermines our understanding of COVID-19 disease risk and severity. *Nature Communications*, 11, 5749.
