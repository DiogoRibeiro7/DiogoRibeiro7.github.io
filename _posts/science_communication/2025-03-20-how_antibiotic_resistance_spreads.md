---
permalink: '/science-communication/how_antibiotic_resistance_spreads/'
title: 'How Antibiotic Resistance Spreads Through Bacteria'
date: '2025-03-20'
categories:
- Science Communication
tags:
- Microbiology
- Antibiotic Resistance
- Evolution
- Public Health
author_profile: false
classes: wide
seo_title: 'Antibiotic Resistance Changes Bacteria, Not Your Body’s Immunity'
seo_description: 'A simple population calculation explains how resistant bacteria can become more common even while their numbers fall, and why resistance can spread between people.'
seo_type: article
excerpt: >-
  Antibiotic resistance concerns microbes and medicines. A worked example
  shows how a rare resistant group can become a large share of the survivors
  without the bacteria learning or the human body becoming immune to a drug.
summary: >-
  Three hypothetical selective bottlenecks separate bacterial numbers from
  bacterial proportions. The article connects this model to selection and
  transmission while making clear why it cannot determine treatment decisions.
keywords:
- antibiotic resistance explained
- bacterial selection
- antimicrobial resistance
- resistance myths
why_this_exists: >-
  Describing resistance as a person becoming used to antibiotics obscures how
  it develops and spreads. The original calculation makes the population
  mechanism visible and distinguishes a changing share from a growing count.
evidence: >-
  Original calculations and a two-panel figure for hypothetical sensitive and
  resistant populations, with CDC explanations of resistance and transmission.
methodology: >-
  Apply explicitly invented survival fractions to two bacterial groups, track
  counts and composition separately, and identify the omitted biological
  processes that prevent this illustration from being a treatment model.
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
Question: How can resistant bacteria become a larger share of a population without a person becoming immune to an antibiotic?
Claim: Differential survival changes population composition, while transmission can move resistant organisms or genes between hosts.
Counterclaim: Resistance evolution also involves mutation, gene transfer, growth, and a changing environment.
Evidence object: A two-group selection calculation, separate count and proportion plots, and primary public-health explanations.
Failure case: Fixed survival fractions and no regrowth do not represent a clinical treatment course.
Reader payoff: Locate resistance in the microbes and distinguish changes in frequency from changes in absolute number.
Exclusions: Antibiotic selection, dosing, treatment duration, and diagnosis of a failed treatment.
-->

Antibiotic resistance means that bacteria can withstand an antibiotic that would otherwise act against them. It does not mean that a person's body has become immune to the medicine. The CDC makes this distinction explicitly in its explanation of antimicrobial resistance. [CDC overview](https://www.cdc.gov/antimicrobial-resistance/about/index.html).

The difference matters. If resistance were simply a personal habit formed by taking medicine, someone who had never used an antibiotic might seem protected from it. But resistant bacteria can spread between people, and bacteria can exchange resistance genes. A person's own prescription history is not the whole history of the microbes they encounter. [CDC on how resistance spreads](https://www.cdc.gov/antimicrobial-resistance/causes/index.html).

To understand one part of the process, we can begin with a population in which resistant bacteria are already present in small numbers.

## Start with a rare minority

Imagine 100,000 bacteria of the same species in a hypothetical laboratory system. Of these, 99,900 belong to a group sensitive to a particular antibiotic, while 100 belong to a resistant group.

The resistant group is initially only 0.1% of the population.

Now impose a selective bottleneck: under the chosen conditions, suppose 1% of the sensitive group survives and 80% of the resistant group survives. These fractions are invented to make the arithmetic visible. They are not measurements for a named species, drug, concentration, or patient.

After this bottleneck, the model contains:

- 999 sensitive bacteria, down from 99,900;
- 80 resistant bacteria, down from 100.

Both groups have become smaller. Yet the resistant share has increased from 0.1% to about 7.4%, because the sensitive group fell much more sharply.

That is selection in the calculation. There is no need to assume that every surviving bacterium changed its identity or learned what the drug was trying to do.

## Count bacteria and calculate their share separately

Repeat the same hypothetical bottleneck twice more, without adding growth or new bacteria between steps:

| Bottlenecks completed | Sensitive model count | Resistant model count | Resistant share of model counts |
| --- | ---: | ---: | ---: |
| 0 | 99,900 | 100 | 0.1% |
| 1 | 999 | 80 | 7.4% |
| 2 | 9.99 | 64 | 86.5% |
| 3 | 0.0999 | 51.2 | 99.8% |

Fractional entries are expected counts in this mathematical illustration, not claims that a fraction of an individual bacterium exists. The displayed shares are ratios of those model counts. A real finite population would have integer counts and random variation, including the possibility that a small group disappears.

![The first panel shows both sensitive and resistant expected counts falling through three hypothetical bottlenecks. The second shows the resistant share rising from 0.1 percent to almost 100 percent.](/assets/images/figures/science_antibiotic_selection.png){: width="1465" height="697" loading="lazy"}

*Original selection model. Sensitive and resistant survival fractions are fixed at 1% and 80%. There is no growth, mutation, transmission, or immune response. The horizontal axis does not represent prescribed doses.*

The plots answer different questions. The first asks how many bacteria remain. The second asks which group makes up those survivors.

After the third step, resistance is almost universal in the model's remaining population, even though the resistant count itself has nearly halved. Saying only “resistance increased” would leave this distinction hidden.

## Why the proportions change so quickly

The resistant group survives each bottleneck at 80 times the fraction of the sensitive group: $0.80/0.01=80$.

Before the first bottleneck, the ratio of resistant to sensitive counts is $100/99{,}900$. Afterwards it is $80/999$. The latter ratio is 80 times larger.

Each repeated bottleneck multiplies the resistant-to-sensitive ratio by the same factor. Starting from a very small fraction does not prevent a group from becoming dominant if the conditions repeatedly favour it strongly enough.

This is a statement about relative survival under the assumed conditions. It does not say that resistant bacteria grow faster in every environment, that all antibiotics select in the same way, or that every exposure produces these changes.

The model identifies a mechanism that can alter a population's composition. Establishing its strength in a real system requires measurements.

## Real bacteria add more routes to the story

Selection acts on differences. Our example supplies those differences at the start and then holds them fixed.

Real bacterial populations can also acquire new genetic changes and share genetic material. The CDC describes mobile genetic elements as one way resistance can move between germs. Resistant organisms can then spread through routes involving people, animals, food, and the environment. [CDC transmission explanation](https://www.cdc.gov/antimicrobial-resistance/causes/index.html).

These processes answer different questions. Selection helps explain why a variant becomes more common under particular conditions. Genetic change and gene transfer help explain where a capability comes from. Transmission helps explain how organisms or their resistance mechanisms reach new settings.

Keeping those questions separate avoids two opposite errors: suggesting that antibiotics make every bacterium deliberately adapt, or suggesting that resistant bacteria must always have been present in exactly their current form.

## Resistance is specific enough to require evidence

Calling bacteria “resistant” needs a reference: resistant to which antimicrobial, assessed under which conditions? A bacterium does not have to resist every medicine to create a serious treatment problem. [CDC overview](https://www.cdc.gov/antimicrobial-resistance/about/index.html).

Likewise, an unsuccessful treatment is an observation requiring assessment, not a diagnosis of resistance on its own. The two-population table contains no symptoms, laboratory results, drug concentrations in tissue, or competing explanations for illness.

Its numbers therefore cannot tell someone which medicine to take or how long to take it. They also cannot tell a clinician whether a particular patient's infection has cleared. Treating each row as another dose would silently turn a teaching example into a clinical model that it was never designed to be.

## What the distinction changes in a conversation

If someone says, “My body has become resistant to antibiotics,” a useful correction is:

> Antibiotic resistance concerns bacteria and the medicines acting on them. Some bacteria may survive better than others, and resistant bacteria or resistance genes can spread. The explanation is about the microbes, not the body becoming accustomed to the drug.

This wording also avoids assigning the entire problem to one person's behaviour. A resistant infection can reflect transmission and a much wider history of selection. Understanding the mechanism makes infection prevention and careful antibiotic use parts of the same public-health problem.

The calculation offers another useful question for a graph or headline: did the number of resistant bacteria rise, did their proportion rise, or did both rise? Those descriptions can imply very different population changes.

## Reproduce the model

```python
sensitive, resistant = 99_900.0, 100.0

for bottleneck in range(4):
    share = resistant / (sensitive + resistant)
    print(f"{bottleneck}: sensitive={sensitive:g}, "
          f"resistant={resistant:g}, resistant share={share:.1%}")
    sensitive *= 0.01
    resistant *= 0.80
```

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces the two panels. The accompanying checks verify that the resistant share increases while both model counts decrease.

*Archive note: dated 20 March 2025 for this collection; written and source-checked on 18 September 2026.*
