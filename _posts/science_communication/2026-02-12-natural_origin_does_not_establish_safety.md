---
permalink: '/science-communication/natural_origin_does_not_establish_safety/'
title: 'Natural Origin Does Not Establish Safety'
date: '2026-02-12'
categories:
- Science Communication
tags:
- Chemistry
- Scientific Literacy
- Risk Communication
- Consumer Claims
author_profile: false
classes: wide
seo_title: 'How to Assess Claims That Natural Means Safe'
seo_description: 'A worked concentration-and-amount example shows why an origin label cannot establish safety, and which information is needed to assess a chemical exposure.'
seo_type: article
excerpt: >-
  Natural describes an origin. To assess safety, we also need to know the
  substance, the amount, the route, and the conditions of use. A simple
  calculation shows why concentration alone is not enough either.
summary: >-
  Three hypothetical samples separate concentration from total amount.
  Worked intake, uncertainty, and detection-limit examples extend that comparison
  into a practical way to examine natural, chemical-free, and low-concentration
  claims without assuming that synthetic means safe.
keywords:
- natural means safe myth
- chemical exposure
- concentration versus amount
- hazard and risk
why_this_exists: >-
  Debunking an origin label should help the reader assess the replacement
  claim. This article supplies a worked exposure calculation and a small
  set of specific questions that a meaningful safety comparison must answer.
evidence: >-
  Original calculations and a figure for three fictional samples, with
  primary explanations from NCCIH, EFSA, and EPA about chemicals, exposure,
  risk assessment, and the interpretation of measurements below detection limits.
methodology: >-
  Hold the identity of a hypothetical substance fixed, vary concentration
  and volume, and separate the resulting amount from any claim about
  absorption, toxicity, product quality, or clinical benefit. Carry units through
  an intake calculation and distinguish scenario bounds from statistical intervals.
reviewed_at: '2026-09-18'
header:
  image: /assets/images/headers/photo-microscope.jpg
  og_image: /assets/images/headers/photo-microscope.jpg
  overlay_image: /assets/images/headers/photo-microscope.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-microscope.jpg
  twitter_image: /assets/images/headers/photo-microscope.jpg
---

<!--
Development contract
Question: Can a natural-origin or low-concentration label establish that a product is safe?
Claim: Origin and concentration alone do not identify the substance, exposure, or resulting risk.
Counterclaim: Production methods and source materials can create relevant differences in composition and quality.
Evidence object: Concentration-times-volume and body-mass-normalised intake calculations, uncertainty bounds, a detection-limit example, and an original figure.
Failure case: Equal chemical amounts do not guarantee equal absorption or effects in different formulations and users.
Reader payoff: Ask for a named substance, units, use conditions, and relevant evidence instead of inferring safety from a label.
Exclusions: Product recommendations, safe-dose thresholds, and an assessment of any named supplement or pesticide.
-->

A natural origin does not, by itself, establish that a substance is safe. The National Center for Complementary and Integrative Health makes this point using examples of naturally occurring substances with very different effects. It also explains that natural products contain chemicals. [NCCIH explanation](https://www.nccih.nih.gov/health/know-science/natural-doesnt-mean-better).

That correction should not turn into a new shortcut in which synthetic automatically means safe. Neither origin label answers the complete question.

For a useful comparison, we need to know what the substance is, how much is present, how exposure occurs, and what evidence applies to those conditions. “Natural” might describe part of its history. It does not supply all of that missing information.

## Identify the claim hidden inside the label

Imagine two posts about the same product. One says that it comes from a plant. The other concludes that it cannot cause harm because it comes from a plant.

The second post adds a universal safety claim. Evidence for the source of an ingredient does not establish that broader conclusion.

The distinction works in both directions. Identifying a chemical hazard is not yet an estimate of the risk under a particular use. EFSA distinguishes the potential to cause harm from the likelihood of harm under specific exposure conditions. [EFSA's explanation of hazard and risk](https://www.efsa.europa.eu/en/glossary/hazard).

A useful response therefore asks the claim to become more specific. Which ingredient, which effect, which amount, and which conditions are being discussed?

## Concentration is only part of the arithmetic

Consider a fictional substance X in three hypothetical liquid samples. X has exactly the same chemical identity in all three. We are calculating the amount present in a chosen volume, before making any claim about biological absorption or effects.

| Sample | Concentration of X | Volume considered | Amount of X |
| --- | ---: | ---: | ---: |
| A | 10 mg per mL | 2 mL | 20 mg |
| B | 1 mg per mL | 30 mL | 30 mg |
| C | 1 mg per mL | 20 mL | 20 mg |

The calculation is

$$
\text{amount} = \text{concentration}\times\text{volume}.
$$

The units show why it works: milligrams per millilitre multiplied by millilitres leaves milligrams.

Sample B has a concentration ten times lower than A, but the selected volume is fifteen times larger. It therefore contains 50% more X in total. Sample C has the same low concentration as B, yet its smaller volume contains the same amount of X as A.

These are invented samples and volumes, not instructions for consuming a real product. No safe or harmful threshold for X has been defined.

![Three hypothetical samples are compared by concentration and total amount. A has the highest concentration, but B contains the greatest amount in its selected volume. A and C each contain 20 milligrams.](/assets/images/figures/science_concentration_and_amount.png){: width="1465" height="681" loading="lazy"}

*Original calculation for one fictional substance. Concentration alone does not rank the amounts in the chosen volumes. Neither panel establishes a health risk or a safety threshold.*

A headline saying “90% lower concentration” can therefore be numerically correct while leaving the practical exposure comparison unanswered. We need the corresponding volumes as well.

EFSA's exposure explanation uses the same general accounting idea for food: combine information about substance levels with information about how much food is consumed over a specified period. [EFSA on exposure](https://www.efsa.europa.eu/en/glossary/exposure).

## An amount still does not tell us the whole effect

The worked table answers one question: how much X is in each specified volume?

It does not tell us what proportion reaches a particular part of the body, how quickly that happens, or what effect follows. Those questions require additional evidence.

Even the comparison between A and C has a limited meaning. They contain equal amounts of X in the chosen volumes. That equality does not establish that two real products would have identical formulations, other ingredients, exposure routes, or users.

Suppose a report provides only “20 mg,” without naming the substance. That number cannot distinguish 20 mg of one compound from 20 mg of another. The unit measures an amount, not a universal biological effect.

Suppose it names the substance but leaves out the time window. A one-time amount and the same amount repeatedly encountered are different exposure descriptions. The missing detail should be requested rather than silently filled in by the reader.

The lesson is to make each comparison answer the question it actually supports. More complete arithmetic is useful, but arithmetic cannot substitute for evidence about the substance and the conditions.

## Follow the units from a laboratory result to an intake estimate

A concentration measurement and an exposure estimate can use similar-looking units while describing different quantities. Consider a second fictional example involving the same unspecified substance X in a food. This is an exercise in accounting; X still has no assigned toxicity or safe level.

Suppose the reported concentration is 5 mg of X per kg of food, and the hypothetical consumption is 0.30 kg of that food per day. Multiplying gives

$$
5\ \frac{\text{mg X}}{\text{kg food}}
\times0.30\ \frac{\text{kg food}}{\text{day}}
=1.5\ \frac{\text{mg X}}{\text{day}}.
$$

If the hypothetical consumer's body mass is 60 kg, the intake per unit body mass is

$$
\frac{1.5\ \text{mg/day}}{60\ \text{kg body mass}}
=0.025\ \frac{\text{mg}}{\text{kg body mass}\cdot\text{day}}.
$$

The first “kg” referred to the food. The final “kg” refers to the consumer. Dropping those labels can make unlike quantities appear comparable. Likewise, converting milligrams into micrograms changes the numerical value by a factor of 1,000 without changing the physical amount.

This is an estimated external intake through one specified route. It does not establish the amount absorbed, the concentration in a target tissue, or the biological response. It also assumes the reported concentration represents the food being consumed and that the consumption amount is appropriate for the stated period.

A useful exposure assessment makes those assumptions visible. EPA's description of human health risk assessment separates hazard identification, dose-response assessment, exposure assessment, and risk characterisation. Our arithmetic addresses only part of the exposure component. [EPA risk-assessment framework](https://www.epa.gov/risk/conducting-human-health-risk-assessment).

## Uncertain inputs produce uncertain amounts

The neat value 0.025 can conceal substantial uncertainty. Suppose we only know that the concentration is between 3 and 7 mg/kg, consumption between 0.10 and 0.40 kg/day, and body mass between 50 and 90 kg in the scenarios of interest.

The lowest intake per unit body mass among those combinations is

$$
\frac{3\times0.10}{90}\approx0.0033\ \frac{\text{mg}}{\text{kg}\cdot\text{day}},
$$

and the highest is

$$
\frac{7\times0.40}{50}=0.056\ \frac{\text{mg}}{\text{kg}\cdot\text{day}}.
$$

That range is a **scenario bound**, not a 95% confidence interval. We have supplied input ranges but no probability distributions, sampling design, or dependence structure. There is no basis for assigning a probability to different positions inside the range.

Combining all upper or lower extremes may also produce a scenario that rarely occurs in practice. People with different body masses may consume different quantities, and concentrations may vary across products or occasions. Dependence between inputs matters if the aim is a population distribution rather than an envelope of allowed combinations.

The exercise still identifies which measurements would improve the estimate. The intake changes proportionally with concentration and consumption, and inversely with body mass. If consumption is the least well characterised input, adding more decimal places to a laboratory concentration will not resolve that uncertainty.

For a headline, the practical lesson is to distinguish a measured value from the assumptions used to translate it into exposure. A claim may cite a precise concentration while relying on an unstated or unrealistic consumption pattern. The precision of one input does not make the whole calculation precise.

## “Not detected” and “absent” answer different questions

A laboratory report may say that X was not detected above a reporting threshold. It is tempting to rewrite that as “there is no X,” just as it is tempting to treat any detected amount as automatically harmful. Both shortcuts omit information.

For a simple illustration, assume an idealised reporting rule that labels every concentration below 0.10 mg/kg as not detected and every concentration above it as detected. A concentration of 0.02 mg/kg would receive the first label even though it is not zero. A more sensitive test with a threshold of 0.01 mg/kg would detect the same sample without any change in its composition.

Real analytical methods do not have perfectly sharp error-free boundaries. Their detection and quantification limits are operational properties of a method and its measurement conditions. The ideal threshold simply makes the logical distinction visible. EPA's data-quality guidance discusses nondetects as censored measurements rather than substituting “zero” or “not present.” [EPA guidance, Section 4.7](https://www.epa.gov/sites/default/files/2015-06/documents/g9-final.pdf).

Under our ideal threshold assumption, a result below 0.10 mg/kg combined with consumption of 0.30 kg/day would imply an amount below 0.03 mg/day from that source. That upper bound does not establish safety: the identity of X and relevant effect evidence are still missing. It does show why an analytical reporting threshold and a health threshold are different concepts.

Comparisons over time need the methods too. A report that detections have increased could reflect a real concentration change, a lower detection threshold, a changed sample population, or a combination. The appropriate response is to examine comparable measurements, not to assume either contamination or reassurance from the detection count alone.

## Match the evidence to the exact claim

Different forms of evidence answer different parts of a product claim. A compositional analysis can establish that a named compound is present in the tested sample. It does not, by itself, establish clinical benefit, absence of harm under every use, or the composition of every future batch.

A biological experiment can test a defined effect under defined conditions. Applying its result to another route, concentration, duration, or population requires an argument about why that comparison is appropriate. The required bridge cannot be replaced by the word “natural,” but neither is it supplied by the word “laboratory.”

Evidence about benefit and evidence about harm also need to be kept separate. A claimed benefit does not make every exposure acceptable, and a hazard finding does not by itself quantify risk at an ordinary exposure. A useful assessment states the outcome being evaluated and combines the relevant exposure and response information.

This changes how we read a reassuring testimonial. “I used it and felt fine” describes one person's experience over some period. It leaves the substance amount, the monitored outcomes, delayed effects, and the experience of other users unspecified. The experience can be sincere without supporting the universal claim that no one can be harmed.

The same discipline applies to a frightening anecdote. Temporal order alone does not isolate a cause, and a reported symptom does not establish that an ingredient produced it. The reader should ask what comparison and measurements would distinguish the proposed explanation from plausible alternatives.

## Can natural and synthetic products differ?

Yes. Rejecting an origin-based guarantee does not establish that every pair of products is equivalent.

A manufacturing or extraction process can leave a product with a particular composition. A broad label may refer to a mixture rather than one identified compound. NCCIH notes that herbal products can contain many chemical constituents, including constituents that are not fully identified. [NCCIH explanation](https://www.nccih.nih.gov/health/know-science/natural-doesnt-mean-better).

Those are reasons to examine what is actually in the product. They are not reasons to rank every natural mixture above every manufactured substance, or the reverse.

An informative comparison would specify the ingredient identities and relevant quality measurements. If a claim concerns impurities, it should present evidence about impurities. If it concerns effectiveness, it should present evidence about the claimed benefit. The origin label cannot do all these jobs at once.

This also explains why “chemical-free” is an unhelpful description of a material product. Water and the components of food are chemicals too. EFSA's overview treats chemicals as ordinary constituents of food, while distinguishing the assessments needed for substances that may raise safety concerns. [EFSA on chemicals in food](https://www.efsa.europa.eu/en/topics/topic/chemicals-food).

## Four questions that improve the comparison

For an online safety claim, try translating the promotional language into a small evidence record:

| Question | Useful information |
| --- | --- |
| What is present? | Identified substance or defined mixture |
| How much is encountered? | Concentration, amount used, frequency, and units |
| Under which conditions? | Route, formulation, population, and duration |
| What supports the conclusion? | Evidence about the specified effect under relevant conditions |

The table is a way to identify missing information. It is not a method for establishing a safe dose at home.

When a claim provides only an origin label, the appropriate conclusion is that its evidence is incomplete. When a claim provides a hazard name but no exposure information, a different piece is missing. Neither omission is repaired by a frightening or reassuring adjective.

## A correction worth keeping short

> Natural describes where something comes from. Safety depends on what it is and the conditions of exposure. A concentration or an ingredient name alone does not tell us the complete risk.

This correction leaves room for meaningful differences between products. It asks those differences to be measured and explained, rather than guessed from the label.

## Reproduce the amount calculation

```python
samples = [("A", 10, 2), ("B", 1, 30), ("C", 1, 20)]

for name, milligrams_per_ml, volume_ml in samples:
    amount_mg = milligrams_per_ml * volume_ml
    print(f"{name}: {milligrams_per_ml} mg/mL x {volume_ml} mL = {amount_mg} mg")
```

The intake scenarios preserve their units through the calculation:

```python
scenarios = [
    ("illustrative", 5, 0.30, 60),
    ("lower bound", 3, 0.10, 90),
    ("upper bound", 7, 0.40, 50),
]
for label, mg_per_kg_food, kg_food_per_day, kg_body in scenarios:
    intake = mg_per_kg_food * kg_food_per_day / kg_body
    print(label, f"{intake:.4f} mg/(kg body mass * day)")
```

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces the original concentration and amount panels. For another example of a correct percentage leaving out essential context, see [Read the Starting Risk Before the Percentage](/science-communication/read_the_starting_risk_before_the_percentage/).

*Archive note: dated 12 February 2026 for this collection; written and source-checked on 18 September 2026.*
