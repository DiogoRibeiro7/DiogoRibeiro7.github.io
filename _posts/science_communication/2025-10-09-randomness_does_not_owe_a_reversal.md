---
permalink: '/science-communication/randomness_does_not_owe_a_reversal/'
title: 'Randomness Does Not Owe Us a Reversal'
date: '2025-10-09'
categories:
- Science Communication
tags:
- Probability
- Scientific Literacy
- Randomness
- Statistical Reasoning
author_profile: false
classes: wide
seo_title: 'Why a Random Streak Does Not Make the Opposite Outcome Due'
seo_description: 'Four heads in a row mean different things for a fair coin, a bag sampled without replacement, and a coin of unknown bias. Exact calculations explain why.'
seo_type: article
excerpt: >-
  Four heads in a row do not make tails due on an independent fair coin.
  But history can matter in other mechanisms. Three examples show exactly
  when the next probability changes and why.
summary: >-
  A fair coin, a shrinking bag, and a coin with unknown bias produce different
  predictions after the same streak. The comparison separates independence,
  changing composition, and learning about an uncertain mechanism.
keywords:
- gambler fallacy
- random streaks
- independent events
- conditional probability
why_this_exists: >-
  The phrase randomness has no memory is too broad to handle many real
  situations. This article compares three explicit mechanisms instead of
  asking readers to memorise a slogan about streaks.
evidence: >-
  Original exact conditional probabilities, enumeration checks, and a figure
  comparing a fair coin, sampling without replacement, and an unknown coin.
methodology: >-
  Condition each mechanism on the same initial sequence of heads, derive the
  next-event probabilities, and identify what information or physical state
  changes between observations.
reviewed_at: '2026-09-18'
header:
  image: /assets/images/headers/photo-dice.jpg
  og_image: /assets/images/headers/photo-dice.jpg
  overlay_image: /assets/images/headers/photo-dice.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-dice.jpg
  twitter_image: /assets/images/headers/photo-dice.jpg
---

<!--
Development contract
Question: Does a streak of one outcome change the probability of that outcome next?
Claim: The answer depends on the mechanism, not on a need for a short sequence to balance itself.
Counterclaim: Previous observations can change a physical system or inform an unknown probability.
Evidence object: Three exactly specified mechanisms, a conditional probability table, and an original figure.
Failure case: Claims about a fair independent coin do not transfer automatically to sport, weather, or sampling without replacement.
Reader payoff: Ask what changes in the mechanism or knowledge before predicting a reversal from a streak.
Exclusions: Betting advice, tests of physical coin fairness, and a review of sporting streak studies.
-->

Suppose a fair coin produces four heads in a row. If each toss is independent, the chance of heads next is still 50%. The previous results have not created a debt that the next toss must repay.

That is the mistake behind saying that tails is now “due.” A short sequence is being asked to enforce a balance that the probability model never promised.

But there is a second mistake to avoid: assuming that previous observations can never matter in a random process. Sometimes they change what remains in the system. Sometimes they change what we know about it.

We can separate those cases by giving the same four-head streak to three different mechanisms.

## First mechanism: a known fair, independent coin

In this model, every toss has probability one-half of heads, regardless of the previous outcomes. “Independent” is doing essential work here.

Before any tosses, the probability of four initial heads is

$$
\frac12\times\frac12\times\frac12\times\frac12
=\frac1{16}.
$$

That is 6.25%. Once those four heads have happened, however, we are asking a new question: what comes next?

Both five-heads and four-heads-then-tails sequences had probability $1/32$ in advance. Conditional on the first four results being heads, they remain equally likely alternatives for the fifth result.

The distinction is between the probability of the entire sequence before observing it and the probability of the remaining outcome after part of it is known.

For repeated independent trials with a fixed success probability, the binomial model describes the total number of successes. It does not require each small block of trials to contain the same number of each outcome. [NIST's binomial distribution reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm).

## Balance can improve without a compensating streak

Imagine that the next 100 tosses after those four heads contain 50 heads and 50 tails. Across all 104 tosses, there would then be 54 heads and 50 tails.

The four-head excess has not been erased. Yet the fraction of heads has moved from 100% to about 51.9%, simply because the original imbalance is now a smaller part of a longer record.

Exactly 50 heads in the next 100 tosses is an illustration, not a prediction. The expected number is 50, with variation around it.

This explains why a long-run tendency toward a proportion does not require short-run correction. A fixed early imbalance can become less important as the denominator grows. The coin does not need to increase the probability of tails to achieve that.

## Second mechanism: a bag whose contents change

Now replace the coin with ten tokens in a bag: five marked H and five marked T. Draw uniformly at random and do not put a token back.

Suppose the first four tokens are all H. The bag now contains one H and five T tokens. The next probability of H is therefore

$$
\frac1{6}\approx16.7\%.
$$

Here the previous observations matter because drawing tokens changes the available population. The mechanism contains a finite stock with a fixed composition at the start.

T is now more likely, but that conclusion comes from counting the remaining tokens. It does not follow from a general rule that random outcomes must reverse after a streak.

If every drawn token were replaced and the bag mixed again, the H probability would return to one-half. Changing that one procedure changes the relevant probability model.

## Third mechanism: a coin whose bias is unknown

Consider a different experiment. Someone randomly chooses one of two coins, with equal probability:

- Coin L produces heads with probability 25%.
- Coin H produces heads with probability 75%.

The selected coin is then tossed repeatedly. Tosses are independent given which coin was selected. We do not know its identity.

Before seeing any results, the probability of heads is 50%, because the two possible biases average to one-half.

After four initial heads, the evidence favours Coin H. That sequence is $0.75^4/0.25^4=81$ times as likely under Coin H as under Coin L. With equal starting probabilities, the updated probabilities for the coin's identity are $81/82$ and $1/82$.

The chance of heads next becomes

$$
\frac{81}{82}\times0.75+
\frac{1}{82}\times0.25
=\frac{61}{82}
\approx74.4\%.
$$

The physical coin has not changed. Our information about it has. The streak shifts the prediction toward more heads, rather than toward a compensating tail.

This is a deliberately specified uncertainty model. It does not mean that four heads from an ordinary coin establish a 74.4% probability for its next toss.

## Put the three answers side by side

| After four initial H outcomes | Probability of H next | What makes history relevant? |
| --- | ---: | --- |
| Known fair, independent coin | 50.0% | Nothing changes in the specified mechanism |
| Five-H, five-T bag without replacement | 16.7% | The remaining contents have changed |
| Unknown 25%- or 75%-heads coin | 74.4% | We have learned about the coin's identity |

![After an initial streak of heads, the next-head probability stays at one-half for a fair independent coin, decreases for a bag sampled without replacement, and increases for an unknown coin chosen from two specified biases.](/assets/images/figures/science_streaks_three_mechanisms.png){: width="1337" height="732" loading="lazy"}

*Original exact calculation. The bag starts with five H and five T tokens. The unknown coin is selected once, with equal prior probability of a 25% or 75% heads rate.*

The visible history is the same. The answers differ because the mechanisms differ.

This is why a streak in a sports team, a sequence of rainy days, or repeated manufacturing failures cannot be settled by announcing that a fair coin has no memory. Those situations require evidence about their own processes. An independent-coin argument does not establish independence in a different system.

## A better question than “is a reversal due?”

Ask what would make the next probability change. Has the pool of possible outcomes changed? Has the process itself changed? Have the observations taught us something about an unknown feature?

If the answer is none of those, and the trials really are independent with known probabilities, a recent imbalance does not create a correction mechanism.

If the answer is yes, use that mechanism to calculate or estimate the next probability. The bag and unknown-coin examples show why merely labelling a process random is insufficient.

## Reproduce the comparison

```python
for heads in range(5):
    fair = 0.5
    bag = (5 - heads) / (10 - heads)
    unknown = (0.25**(heads + 1) + 0.75**(heads + 1)) / (
        0.25**heads + 0.75**heads
    )
    print(f"{heads} initial heads: fair={fair:.1%}, "
          f"bag={bag:.1%}, unknown={unknown:.1%}")
```

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces the chart. Its unknown-coin probabilities are also checked by enumerating and weighting every five-toss sequence.

*Archive note: dated 9 October 2025 for this collection; written and source-checked on 18 September 2026.*
