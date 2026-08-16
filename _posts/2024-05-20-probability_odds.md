---
author_profile: false
categories:
- Statistics
classes: wide
date: '2024-05-20'
excerpt: Discover the difference between probability and odds in biostatistics, and
  how these concepts apply to data science and machine learning. A clear explanation
  of event occurrence and likelihood.
header:
  image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_3.jpg
  overlay_image: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_1.jpg
  twitter_image: /assets/images/data_science_3.jpg
keywords:
- Probability vs odds
- Biostatistics probability
- Understanding odds in statistics
- Event occurrence likelihood
- Statistical analysis in data science
seo_description: The key differences between probability and odds in biostatistics, with clear examples and applications in data science.
seo_title: Understanding Probability and Odds in Biostatistics
seo_type: article
subtitle: A Clear Explanation of Two Key Concepts in Biostatistics
summary: This article provides a detailed explanation of probability and odds, exploring
  their definitions, differences, and applications in biostatistics, data science,
  and machine learning.
tags:
- Probability
- Statistical Modeling
- Mathematical Modeling
- Statistics
- Data Science
- Machine Learning
title: Understanding Probability and Odds
---

## Introduction

In the field of biostatistics, understanding the likelihood of events is crucial. Two fundamental concepts used to describe this likelihood are probability and odds. While they are related, they provide different perspectives on the likelihood of an event occurring. This article will clarify the difference between these two concepts and illustrate how they can be used interchangeably.

## What is Probability?

### Definition

Probability is a measure of the likelihood that a particular event will occur. It is calculated as the ratio of the number of favorable outcomes to the total number of possible outcomes.

### Formula

$$\text{Probability (P)} = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

### Example

Using the example of rolling a die to get a number greater than 4:

- **Favorable outcomes**: 5 and 6 (2 outcomes)
- **Total outcomes**: 1, 2, 3, 4, 5, 6 (6 outcomes)
- **Probability**: $$\frac{2}{6} = \frac{1}{3}$$

Probability is bounded: it always lies between 0 and 1. That boundedness is convenient for interpretation but awkward for modelling, and it is the main reason odds exist as a separate quantity.

## What are Odds?

### Definition

Odds are a ratio comparing the probability that an event will occur to the probability that it will not occur.

### Formula

$$\text{Odds} = \frac{\text{Probability of the event occurring}}{\text{Probability of the event not occurring}}$$
$$\text{Odds} = \frac{P}{1 - P}$$

### Example

Using the same die example to get a number greater than 4:

- **Favorable outcomes**: 5 and 6 (2 outcomes)
- **Unfavorable outcomes**: 1, 2, 3, 4 (4 outcomes)
- **Probability of event occurring**: $$\frac{2}{6} = \frac{1}{3}$$
- **Probability of event not occurring**: $$1 - \frac{1}{3} = \frac{2}{3}$$
- **Odds**: $$\frac{\frac{1}{3}}{\frac{2}{3}} = \frac{1}{3} \times \frac{3}{2} = \frac{1}{2}$$ or 1:2

Unlike probability, odds are unbounded above. As $P$ approaches 1 the odds grow without limit, while as $P$ approaches 0 the odds approach 0. The range is therefore $[0, \infty)$ rather than $[0,1]$.

## Converting Between Probability and Odds

### From Probability to Odds

$$\text{Odds} = \frac{P}{1 - P}$$

### From Odds to Probability

$$P = \frac{\text{Odds}}{1 + \text{Odds}}$$

A few reference points are worth memorising, because they make published odds figures immediately readable:

| Probability | Odds | Stated as |
|---|---|---|
| 0.10 | 0.111 | 1:9 |
| 0.25 | 0.333 | 1:3 |
| 0.50 | 1.000 | 1:1 (even) |
| 0.75 | 3.000 | 3:1 |
| 0.90 | 9.000 | 9:1 |

Notice the asymmetry. Probabilities of 0.10 and 0.90 are equidistant from 0.5, but their odds, 0.111 and 9, are not equidistant from 1. Taking logarithms restores the symmetry, which is exactly why the log-odds scale is the one statistical models use.

## The Log-Odds Scale

The **logit** is the natural logarithm of the odds:

$$
\operatorname{logit}(P) = \log\left(\frac{P}{1-P}\right).
$$

This transformation maps the interval $(0,1)$ onto the entire real line. A probability of 0.5 becomes 0, probabilities above 0.5 become positive, and those below become negative, symmetrically. Its inverse is the logistic function:

$$
P = \frac{1}{1 + e^{-\operatorname{logit}(P)}} .
$$

The practical value is that an unbounded quantity can be modelled with a linear predictor without any risk of producing an impossible probability. A linear model fitted directly to $P$ can predict 1.4 or $-0.2$; a linear model fitted to the logit cannot.

## Why This Matters: The Odds Ratio

The odds ratio compares the odds of an outcome between two groups:

$$
\text{OR} = \frac{\text{odds in exposed group}}{\text{odds in unexposed group}}
= \frac{P_1 / (1 - P_1)}{P_0 / (1 - P_0)} .
$$

For a standard two-by-two table with counts $a, b, c, d$:

| | Outcome present | Outcome absent |
|---|---|---|
| **Exposed** | $a$ | $b$ |
| **Unexposed** | $c$ | $d$ |

the odds ratio reduces to the cross-product $\text{OR} = ad / bc$.

This is the quantity logistic regression estimates. A fitted coefficient $\beta$ is a difference in log-odds, so $e^{\beta}$ is an odds ratio: the multiplicative change in odds for a one-unit increase in the predictor. An odds ratio of 1 means no association, above 1 means increased odds, below 1 means decreased odds.

Odds ratios also have a property no other measure of association shares: they can be estimated validly from case-control studies, where participants are selected by outcome rather than by exposure. Because the cross-product is invariant to how the two outcome groups were sampled, the odds ratio survives a design that makes relative risk uncomputable. This is the reason odds dominate epidemiology.

## Odds Ratio Is Not Relative Risk

The most common error in reading medical literature is treating an odds ratio as though it were a risk ratio. Relative risk compares probabilities directly:

$$
\text{RR} = \frac{P_1}{P_0}, \qquad \text{OR} = \frac{P_1/(1-P_1)}{P_0/(1-P_0)} .
$$

When the outcome is rare, the denominators $1-P_1$ and $1-P_0$ are both close to 1 and the two measures nearly coincide. When the outcome is common they diverge sharply, and the odds ratio is always further from 1 than the risk ratio.

Consider a treatment that raises an outcome's probability from 0.40 to 0.60:

- Relative risk: $\frac{0.60}{0.40} = 1.5$, a 50% increase in risk.
- Odds ratio: $\frac{0.60/0.40}{0.40/0.60} = \frac{1.5}{0.667} = 2.25$.

Reporting "the odds more than doubled" is accurate; reporting "the risk more than doubled" is not, and the difference is large enough to change a clinical decision. The rare-disease approximation that justifies conflating them fails badly above roughly 10% baseline prevalence.

```python
def odds(p):
    return p / (1 - p)

def odds_ratio(p1, p0):
    return odds(p1) / odds(p0)

for p1, p0 in [(0.60, 0.40), (0.011, 0.010), (0.30, 0.20)]:
    rr = p1 / p0
    print(f"P1={p1:<6} P0={p0:<6} RR={rr:.3f}  OR={odds_ratio(p1, p0):.3f}")
```

Running this makes the pattern obvious: for the rare pair the two agree to three decimals, and for the common pairs they do not.

## Summary

- **Probability** tells us how likely an event is to happen out of the total number of possible outcomes, and is bounded between 0 and 1.
- **Odds** compare the likelihood of the event happening to the likelihood of it not happening, and range from 0 to infinity.
- **Log-odds** map probability onto the whole real line, which is what makes logistic regression possible.
- **Odds ratios** are the natural output of logistic regression and are valid in case-control designs, but they overstate relative risk whenever the outcome is common.

## Conclusion

Understanding the concepts of probability and odds is essential for accurately describing the likelihood of events in biostatistics. By mastering these concepts, you can better interpret and communicate statistical findings.

The practical discipline is to state which scale you are using and convert deliberately. When reporting to a clinical or general audience, converting an odds ratio back to absolute risks under a stated baseline prevalence is almost always clearer than quoting the ratio itself, because absolute risk is what actually informs a decision.

## References

- Agresti, A. (2018). *An Introduction to Categorical Data Analysis* (3rd ed.). Wiley.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Davies, H. T. O., Crombie, I. K., & Tavakoli, M. (1998). When can odds ratios mislead? *BMJ*, 316(7136), 989-991.
- Zhang, J., & Yu, K. F. (1998). What's the relative risk? A method of correcting the odds ratio in cohort studies of common outcomes. *JAMA*, 280(19), 1690-1691.
