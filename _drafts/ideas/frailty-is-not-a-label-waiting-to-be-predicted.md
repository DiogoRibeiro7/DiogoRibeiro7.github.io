---
author_profile: false
categories:
- Healthcare
classes: wide
title: 'Frailty Is Not a Label Waiting to Be Predicted'
excerpt: Frailty can mean a phenotype, an accumulation of deficits, a clinical judgement, or a multidimensional vulnerability construct. A model trained on one definition does not automatically predict the others.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- frailty
- aging
- clinical prediction
- phenotype
- frailty index
- digital health
seo_title: 'Frailty Is Not a Label Waiting to Be Predicted'
seo_description: 'Why frailty prediction depends on the operational definition of frailty, and why phenotype, deficit accumulation, and clinical vulnerability should not be treated as interchangeable labels.'
seo_type: article
summary: 'Frailty is a clinical construct with competing operational definitions. Predictive models inherit those definitions and should be evaluated against the decisions they are intended to support.'
tags:
- Frailty
- Clinical Prediction
- Aging
- Measurement
why_this_exists: 'ML papers often treat frailty as if it were a naturally observed binary label. In reality the label is constructed from a clinical definition.'
evidence: 'Fried frailty phenotype, Rockwood deficit-accumulation index, and clinical frailty literature.'
methodology: 'Compare operational definitions, derive how label construction changes prevalence and error, and separate construct prediction from outcome prediction.'
reviewed_at: 2026-09-19
---

<!--
Development contract
Question: What exactly does a model predict when the target is “frailty”?
Claim: Frailty is an operationalised construct, so model performance depends on which definition created the label.
Counterclaim: Different definitions may still capture a shared vulnerability dimension and can be pragmatically useful.
Evidence object: Fried phenotype, deficit-accumulation index, and threshold counterexample.
Failure case: The article does not argue that frailty is unreal; it argues that measurement choices matter.
Reader payoff: Define the clinical construct before optimising the classifier.
Exclusions: A review of all geriatric risk scores.
-->

A machine-learning paper predicts frailty with 88% accuracy.

The first question should not be which algorithm achieved the score.

It should be:

> What was called frailty?

That question is not semantic.

Frailty is a clinically important construct, but it has multiple operational definitions.

A model trained on one definition learns that definition.

## The Fried phenotype defines one construct

The Fried frailty phenotype uses criteria including weight loss, exhaustion, weakness, slow walking speed, and low physical activity.

People meeting enough criteria are classified as frail.

This definition has clear clinical intuition.

It also embeds specific measurement choices and thresholds.

A model predicting this label is partly learning those choices.

## The frailty index defines another

The Rockwood deficit-accumulation approach counts a broader set of health deficits.

A frailty index can be written schematically as

$$
FI = rac{	ext{number of deficits present}}{	ext{number of deficits assessed}}.
$$

This produces a continuous measure.

It represents frailty differently from the phenotype model.

Two people can therefore be classified differently under the two frameworks without either system being internally inconsistent.

## Turning a continuous construct into a binary label loses information

Suppose a frailty index ranges from 0 to 1.

A threshold converts it into

$$
Y = I(FI > c).
$$

Now a person at 0.24 and a person at 0.01 receive the same label if (c=0.25).

A person at 0.26 and another at 0.60 also receive the same label.

The classifier is then asked to reproduce a thresholded construct.

This can make model performance look cleaner than the clinical reality.

## Label disagreement creates an upper bound problem

If two accepted frailty definitions disagree materially, then “ground truth” depends on the operationalisation.

A model can disagree with one label and agree with another.

That is not necessarily model error.

It may be construct disagreement.

This should affect how accuracy is interpreted.

## Predicting frailty is not the same as predicting outcomes

Frailty is often used because it predicts outcomes such as falls, hospitalisation, disability, institutionalisation, and mortality.

But if the real decision concerns those outcomes, it may be more direct to model them.

A frailty classifier is useful when the construct itself has clinical meaning.

It is less useful when frailty is merely an intermediate label inserted because it is familiar.

The estimand should match the decision.

## Passive sensing complicates the construct further

A wearable or smart-home system may infer gait speed, activity, sleep, or mobility patterns.

These are related to frailty.

They are not frailty itself.

A digital frailty model therefore combines two layers of operationalisation:

sensor-derived proxies;
clinical frailty definition.

Measurement error exists in both.

## Prevalence depends on the definition

A change in criteria changes the number of people labelled frail.

That affects:

- class balance;
- positive predictive value;
- apparent calibration;
- fairness across subgroups.

Comparing two frailty classifiers without checking label definition can therefore be meaningless.

They may not be solving the same task.

## Clinical utility should anchor the model

A useful frailty model should answer a decision question.

Does it help identify people who need comprehensive geriatric assessment?

Does it improve fall prevention?

Does it change discharge planning?

Does it identify reversible vulnerability?

Those questions are stronger than “does the model reproduce a binary label?”

## Frailty is multidimensional

Physical performance is important.

So are cognition, nutrition, comorbidity, social vulnerability, and functional reserve.

Different models emphasise different dimensions.

The existence of multiple definitions reflects this multidimensionality.

That is not a reason to abandon the construct.

It is a reason to make the construct explicit.

## Conclusion

Frailty is not a natural binary variable waiting in the dataset.

It is a clinical construct made operational through definitions.

The Fried phenotype, deficit-accumulation index, and clinical judgement capture overlapping but non-identical aspects of vulnerability.

A predictive model inherits whichever definition created its target.

Before optimising the classifier, define what the label means and why predicting it changes care.

## References

- Fried LP, Tangen CM, Walston J, et al. Frailty in older adults: evidence for a phenotype. *J Gerontol A*. 2001.
- Rockwood K, Mitnitski A. Frailty in relation to the accumulation of deficits. *J Gerontol A*. 2007.
- Clegg A, Young J, Iliffe S, Rikkert MO, Rockwood K. Frailty in elderly people. *Lancet*. 2013.
- Rockwood K, Song X, MacKnight C, et al. A global clinical measure of fitness and frailty in elderly people. *CMAJ*. 2005.
