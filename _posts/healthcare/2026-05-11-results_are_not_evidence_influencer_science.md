---
permalink: '/healthcare/results_are_not_evidence_influencer_science/'
title: 'Results Are Not Evidence: When Popularity, Testimonials and Sales Pretend to Be Science'
date: '2026-05-11'
categories:
- Healthcare
tags:
- Science Communication
- Influencers
- Evidence Based Practice
- Causal Inference
- Testimonials
- Conflicts of Interest
author_profile: false
classes: wide
seo_title: 'Results Are Not Evidence: Influencers, Testimonials and Scientific Authority'
seo_description: 'A critical look at how health and fitness influencers turn testimonials, client results, popularity and commercial success into substitutes for causal evidence.'
seo_type: article
excerpt: >-
  A client can lose weight. A programme can sell. An influencer can become famous.
  All three observations can be true without proving the scientific explanation
  used to market the method. Results are observations. Evidence requires a
  denominator, a counterfactual and a way to separate signal from selection.
summary: >-
  This article examines how health and fitness influencing can confuse testimonials
  with causal evidence, popularity with expertise, and commercial success with
  scientific validation. It explains selection bias, survivorship bias, regression
  to the mean, multi-component interventions, conflicts of interest and parasocial
  trust, then proposes a practical standard for genuinely evidence-based communication.
keywords:
- fitness influencers
- health influencers
- testimonials
- anecdotal evidence
- causal inference
- conflicts of interest
- science based
- social media health misinformation
why_this_exists: >-
  Social media rewards confidence, visual transformation and commercial proof.
  Science asks different questions: compared with what, in whom, with what
  denominator, under what uncertainty, and after accounting for alternative
  explanations? The gap between those two systems creates a predictable kind of
  pseudo-evidence.
evidence: >-
  Systematic reviews of influencer effects on health, conflict-of-interest
  reporting in social-media health communication, parasocial relationships and
  health persuasion, together with regulatory guidance on health testimonials
  and endorsements.
methodology: >-
  Treat public popularity and commercial success as social variables rather than
  epistemic ones. Analyse testimonial claims using causal inference concepts:
  counterfactuals, selection mechanisms, missing denominators, regression to the
  mean, co-interventions and outcome reporting.
reviewed_at: '2026-09-19'
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
Question: Do client transformations, popularity and commercial success validate the scientific claims used by health and fitness influencers?
Claim: No. They can demonstrate reach, satisfaction or observed outcomes, but causal scientific claims require information that testimonials and popularity do not provide.
Counterclaim: Real-world results are not worthless; they can generate hypotheses, reveal feasibility and provide useful pragmatic information when denominators and methods are transparent.
Evidence object: Systematic reviews of influencer health effects, conflict-of-interest disclosure studies, parasocial-health meta-analysis and advertising standards for testimonials.
Failure case: Treating all commercial activity as corrupt, all influencers as incompetent, or assuming academic credentials automatically guarantee good science.
Reader payoff: A practical framework for separating genuine results from causal evidence and for recognising when “science-based” has become a branding device.
Exclusions: Named individuals, personality judgements, private motives and claims about any specific influencer's intentions.
-->

There is a sentence that appears constantly in health and fitness arguments:

> **But I get results.**

Sometimes the evidence is a photograph.

Sometimes it is a client who lost 20 kilograms.

Sometimes it is a list of people who improved their blood tests.

Sometimes it is a waiting list for coaching.

Sometimes it is a large following.

Sometimes it is the financial success of the business itself.

The underlying argument is usually:

$$
\text{visible success}
\Rightarrow
\text{the method is scientifically correct}.
$$

That implication is false.

A result can be genuine.

A transformation can be impressive.

A client can sincerely believe that a programme changed their life.

A business can become extremely successful.

None of those observations, by themselves, identifies **why** the result occurred.

That distinction is the difference between marketing evidence and scientific evidence.

## A result is an observation, not a causal explanation

Suppose a person joins a coaching programme and loses 12 kg.

We observe

$$
Y_{\text{after}} - Y_{\text{before}} = -12\text{ kg}.
$$

That is a real observation.

But the causal question is different:

$$
Y(1)-Y(0),
$$

where

- $Y(1)$ is the outcome under the intervention;
- $Y(0)$ is the outcome that would have occurred for the same person, over the same period, without that intervention.

We never observe both for the same person at the same time.

That missing quantity is the **counterfactual**.

Scientific study design exists largely because the counterfactual is unobservable.

A before-and-after photograph tells us $Y(1)$.

It does not tell us $Y(0)$.

Without a credible approximation of $Y(0)$, the causal effect is unknown.

## “It worked” and “I know why it worked” are different claims

This is one of the most important distinctions in health communication.

Suppose a person starts a programme that includes:

- calorie tracking;
- more protein;
- fewer ultra-processed foods;
- resistance training;
- more walking;
- improved sleep;
- alcohol reduction;
- weekly accountability;
- a supplement;
- a rule banning seed oils.

Six months later, the person has lost weight and feels better.

The programme worked in the ordinary sense that a positive outcome occurred while the person followed it.

But what caused the improvement?

The observed effect can be written schematically as

$$
\Delta Y
=
\beta_1 X_1
+
\beta_2 X_2
+
\cdots
+
\beta_p X_p
+
\varepsilon.
$$

If ten variables changed simultaneously, success does not identify which $\beta_j$ mattered.

The programme may work while the explanation attached to it is wrong.

For example:

> “My clients stopped eating seed oils and lost weight, therefore seed oils were making them fat.”

The actual mechanism might instead be:

$$
\text{less takeaway food}
+
\text{lower energy intake}
+
\text{higher protein}
+
\text{more structured meals}
\rightarrow
\text{weight loss}.
$$

A genuine outcome does not validate every story told about the outcome.

## Testimonials hide the denominator

A testimonial answers:

> Can I find someone for whom this appeared to work?

Science asks:

> How often did it work among everyone exposed to the intervention?

Those are different questions.

Imagine a programme enrols 1,000 people.

After one year:

- 80 achieve a spectacular result;
- 220 achieve a modest result;
- 300 show little change;
- 250 stop participating;
- 150 cannot be contacted.

A marketing page can show the best 20 transformations and be factually accurate.

Every photograph can be real.

Every quotation can be genuine.

But the visible sample is

$$
S=20,
$$

while the relevant denominator is

$$
N=1000.
$$

The testimonial page estimates something like

$$
P(\text{success}\mid\text{selected for promotion}),
$$

not

$$
P(\text{success}\mid\text{joined programme}).
$$

Those probabilities can be radically different.

This is why the U.S. Federal Trade Commission's guidance on endorsements explicitly distinguishes exceptional testimonials from results consumers can generally expect. The broader principle is not specifically American: **selected success stories do not establish typical effectiveness**.

Source:

- [Federal Trade Commission: Endorsements and Testimonials](https://www.ftc.gov/news-events/topics/truth-advertising/advertisement-endorsements)

## The people who disappear matter

Successful clients are visible.

Unsuccessful clients often disappear.

This creates survivorship bias.

If we study only people who completed a programme, we condition on a variable that may itself depend on motivation, early success, money, time, injury, illness, expectations and satisfaction.

Let $C=1$ indicate programme completion.

Then the quantity

$$
E[Y\mid C=1]
$$

does not necessarily estimate

$$
E[Y]
$$

for everyone who started.

People who remain in a demanding programme may be systematically different from those who leave.

They may have:

- more time;
- fewer injuries;
- higher income;
- better adherence;
- stronger initial motivation;
- greater early response.

A testimonial archive is therefore often a dataset generated by a strong selection mechanism.

The correct question is not merely:

> How good are the visible results?

It is:

> Who became visible?

## Regression to the mean can look like treatment success

People often seek help when something is unusually bad.

Pain is unusually severe.

Weight reaches a personal maximum.

Sleep is particularly poor.

A biomarker produces a worrying result.

Symptoms are at their worst.

Extreme measurements tend, partly because of random variation, to be followed by less extreme measurements.

This is **regression to the mean**.

Let

$$
Y_t = \mu + \varepsilon_t.
$$

If treatment begins when $\varepsilon_t$ is unusually positive or negative, a later value can move toward $\mu$ even when the intervention has no effect.

This does not mean the improvement is imaginary.

The improvement happened.

The problem is causal attribution.

Without an appropriate comparison, natural variation can be mistaken for efficacy.

## Natural history is another invisible comparator

Many conditions fluctuate.

Back pain changes.

Skin conditions flare and improve.

Gastrointestinal symptoms vary.

Minor injuries heal.

Fatigue changes with sleep and stress.

If someone begins a supplement at the worst point of a fluctuating condition and improves over the following weeks, the sequence

$$
\text{take product}
\rightarrow
\text{feel better}
$$

is psychologically compelling.

But temporal order alone does not establish causality.

Formally,

$$
X_t < Y_{t+1}
$$

does not prove

$$
X_t \rightarrow Y_{t+1}.
$$

The untreated natural history remains a competing explanation.

## “I have hundreds of clients” does not solve the problem

Increasing the number of anecdotes does not automatically turn them into a controlled study.

One hundred selected testimonials are still selected.

Ten thousand followers reporting success in comments are still self-selected reporters.

A large uncontrolled dataset can estimate some quantities very precisely while remaining badly biased.

This is a basic statistical fact:

$$
\text{variance} \downarrow
\not\Rightarrow
\text{bias} \downarrow.
$$

With enough biased observations, we can become extremely confident about the wrong quantity.

That is why sample size does not rescue bad sampling.

## Popularity is not peer review

Another common move is:

> If I were wrong, I would not have this audience.

That is not a scientific argument.

Follower count measures some mixture of:

- reach;
- entertainment value;
- communication skill;
- platform timing;
- branding;
- consistency;
- controversy;
- emotional resonance;
- network effects;
- advertising;
- usefulness;
- trust.

Accuracy may contribute.

It is not the only variable.

Let popularity be $P$ and evidential validity be $E$.

There is no general law

$$
P \uparrow \Rightarrow E \uparrow.
$$

A highly accurate scientist may communicate badly.

A charismatic communicator may simplify aggressively.

A false claim may be more shareable than a qualified one.

A systematic review published in *Social Science & Medicine* found that influencers can measurably affect health behaviours and outcomes, with both beneficial and harmful effects depending on context. That capacity to influence is evidence of persuasive power, not of universal epistemic reliability.

Source:

- [Powell & Pring, 2024, *The impact of social media influencers on health outcomes*](https://pubmed.ncbi.nlm.nih.gov/38070305/)

## Commercial success validates a market, not a mechanism

A profitable programme demonstrates that people are willing to buy it.

That can be useful information.

It may indicate that the service solves a practical problem, provides accountability, creates community or communicates effectively.

But revenue is not a biological endpoint.

If

$$
R = \text{revenue}
$$

and

$$
T = \text{truth of scientific claim},
$$

then there is no necessary implication

$$
R \uparrow \Rightarrow T=1.
$$

History contains profitable products that worked, profitable products that did nothing, and profitable products that caused harm.

Markets optimise for willingness to pay.

Scientific methods optimise, imperfectly, for reducing uncertainty about claims.

Those objectives sometimes align.

They are not identical.

## The dangerous step is using success as immunity from criticism

There is a subtle rhetorical pattern that appears frequently online.

A scientific claim is challenged.

Instead of answering with evidence, the response becomes:

- look at my clients;
- look at my following;
- look at my business;
- look at my physique;
- look at my years of experience;
- look at how many people trust me.

All of those facts may be true.

None addresses the proposition under dispute.

This is an argument from status.

The structure is

$$
\text{I am successful}
\therefore
\text{my claim is correct}.
$$

Science tries to remove precisely this dependency.

A result should be able to survive when the speaker's name is deleted.

## Expertise matters, but authority has a domain

The opposite extreme is also wrong.

Expertise matters.

Experience matters.

A coach who has worked with thousands of people may recognise practical patterns that a researcher has never encountered.

But expertise is domain-specific.

Being excellent at programme adherence does not automatically confer expertise in endocrinology.

Being a successful bodybuilder does not establish expertise in toxicology.

Running a profitable nutrition business does not establish expertise in causal inference.

We should think of expertise as a vector:

$$
\mathbf{E}
=
(E_1,E_2,\ldots,E_p),
$$

not a scalar

$$
E=\text{expert}.
$$

High competence in one coordinate does not imply high competence in all coordinates.

The internet often turns local expertise into global authority.

## The halo of “science-based”

The phrase **science-based** has become a brand category.

Sometimes it means something valuable:

- citing primary research;
- distinguishing effect size from significance;
- discussing uncertainty;
- updating beliefs when evidence changes;
- separating mechanism from outcomes;
- acknowledging limitations.

Sometimes it means something else:

- papers appear on screen;
- technical words are used;
- a mechanistic diagram is shown;
- references are listed;
- the conclusion was decided before the literature search began.

The relevant distinction is not whether citations exist.

It is whether the reasoning is auditable.

A scientific claim should permit the audience to ask:

> What evidence would make you change your mind?

If the answer is

> nothing, because my clients get results,

then the process is no longer scientific.

## A citation does not automatically make a claim evidence-based

Citation dumping is another common defence.

Ten papers are placed below a post.

The post is now described as science-based.

But a reference can fail to support a claim in many ways.

The cited paper may be:

- in mice;
- in cells;
- observational;
- underpowered;
- studying a different dose;
- studying a different population;
- measuring a biomarker rather than the claimed clinical outcome;
- testing a different intervention;
- contradicted by stronger evidence.

The correct relationship is not

$$
\text{citation present}
\Rightarrow
\text{claim established}.
$$

It is

$$
\text{claim strength}
\leq
\text{strength of relevant evidence}.
$$

That inequality is violated constantly.

## Mechanistic sophistication can hide causal weakness

Technical explanations are persuasive.

Insulin.

Cortisol.

Inflammation.

Mitochondria.

Autophagy.

Dopamine.

Gut permeability.

Neuroplasticity.

Each term refers to real biology.

But a mechanism can be real without being the dominant explanation for an outcome.

Suppose

$$
X \rightarrow M
$$

and

$$
M \rightarrow Y
$$

are biologically plausible.

That still does not establish that changing $X$ in humans meaningfully changes $Y$.

The size of the effect matters.

Competing pathways matter.

Compensation matters.

Dose matters.

Time matters.

The body is not a diagram with one arrow.

## Confidence is not calibration

Influencer media rewards certainty.

Scientific communication often produces sentences such as:

> The evidence suggests a small benefit in this population, although heterogeneity is substantial and long-term outcomes remain uncertain.

That is accurate.

It is terrible short-form content.

The algorithmically improved version becomes:

> This fixes inflammation.

Confidence increases shareability.

It can also destroy calibration.

A well-calibrated communicator should be 60% confident about some claims, 90% about others and willing to say “we do not know” when appropriate.

If every statement is delivered with maximum certainty, vocal confidence carries no information about evidential confidence.

In statistical terms, the probability forecast is miscalibrated.

## Parasocial trust changes the evidence environment

Followers do not experience an influencer as an anonymous paper.

They see the same face repeatedly.

They hear personal stories.

They observe daily routines.

They watch family life, training, meals and failures.

Over time, this can produce a parasocial relationship: a one-sided sense of familiarity and trust.

A 2026 meta-analysis of 58 studies found that parasocial relationships in health contexts were associated with perceived credibility, health-information seeking and sharing, while also being associated with lower resistance to persuasion.

That is not inherently bad.

Trust can improve public-health communication.

But it changes how claims are processed.

The audience may increasingly evaluate

$$
P(\text{claim true}\mid\text{I trust this person})
$$

rather than

$$
P(\text{claim true}\mid\text{quality of evidence}).
$$

Those are not the same posterior.

Source:

- [Meta-analysis of parasocial relationships and health, 2026](https://pubmed.ncbi.nlm.nih.gov/41830519/)

## Money does not make a claim false

Commercial incentives deserve a more careful argument than:

> they sell something, therefore they are lying.

That inference is invalid.

Researchers receive salaries.

Doctors are paid.

Universities compete for grants.

Publishers make money.

Pharmaceutical companies can produce excellent trials.

A coach can recommend a product they sell and still be correct.

The relevant issue is **conflict of interest**, not automatic corruption.

A conflict of interest exists when a secondary interest can create incentives that compete with the primary interest.

It changes the risk of bias.

It does not determine the truth value of every statement.

Formally,

$$
P(T\mid COI)
\neq 0
$$

and

$$
P(\neg T\mid COI)
\neq 1.
$$

The appropriate response is disclosure, scrutiny and stronger evidential discipline.

## But commercial incentives change what deserves scrutiny

Health communication becomes ethically different when a scientific claim is attached to:

- an affiliate code;
- a supplement;
- a paid programme;
- a laboratory test;
- a subscription;
- a product line;
- a consultation funnel.

Now one message performs two functions simultaneously:

$$
\text{information}
+
\text{sales}.
$$

That does not make the information false.

It does make transparency essential.

A systematic review in *BMJ Open* found that conflict-of-interest disclosure in social-media health communication was frequently poor. Across included studies, reporting of conflicts was generally low, and financial relationships with industry were common in several studied professional groups.

The authors also identified evidence suggesting that conflicts may be associated with the content of posts, although the evidence base was limited.

Source:

- [Helou et al., 2023, *BMJ Open*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10432670/)

## Ethics begins where the audience cannot see the missing data

A seller knows things the audience may not know.

How many people joined?

How many left?

How many asked for refunds?

How many had no meaningful change?

How many experienced adverse effects?

How were the impressive examples selected?

Were photographs taken under comparable conditions?

Did successful clients also receive other interventions?

Were the expected results calculated before selecting testimonials?

This creates information asymmetry.

Let the provider know dataset

$$
D_{\text{all}}
$$

while the audience sees

$$
D_{\text{selected}}.
$$

If

$$
D_{\text{selected}}
\subsetneq
D_{\text{all}},
$$

the ethical question is not merely whether anything shown is false.

It is whether selection creates a misleading impression of the whole distribution.

That is a much higher standard than technical truthfulness.

## Before-and-after photographs are particularly weak scientific objects

A before-and-after image is powerful because human vision is persuasive.

But scientifically it usually lacks:

- standardised lighting;
- standardised posture;
- identical hydration;
- identical glycogen state;
- identical clothing;
- identical camera distance;
- blinded assessment;
- body-composition measurement;
- a control;
- a denominator.

Even if perfectly honest, it establishes little beyond visible change in one selected person.

A photograph can be useful documentation.

It is not a randomized trial compressed into JPEG format.

## The strongest evidence would often be boring

Imagine two coaches.

Coach A shows 40 extraordinary transformations.

Coach B publishes:

- number enrolled;
- baseline characteristics;
- dropout rate;
- median outcome;
- interquartile range;
- adverse events;
- proportion meeting predefined targets;
- follow-up duration;
- protocol changes;
- missing-data handling.

Coach A will probably produce better marketing.

Coach B provides better evidence.

This tension is structural.

Social media selects for exceptional examples.

Science needs distributions.

## Results should come with uncertainty

Suppose average weight loss is

$$
\bar X = 7.2\text{ kg}.
$$

That number means little without dispersion.

If

$$
SD = 1.0\text{ kg},
$$

results are concentrated.

If

$$
SD = 10.0\text{ kg},
$$

experience varies enormously.

The same mean can hide very different populations.

Reporting only the best result is worse.

It discards the distribution entirely.

An ethical results culture should prefer:

$$
\text{distribution}
>
\text{maximum}.
$$

Not because maxima are fake.

Because maxima are rarely representative.

## The “my clients are proof” argument is unfalsifiable when failures disappear

Suppose a client succeeds.

The method worked.

Suppose another client fails.

The explanation becomes:

- poor adherence;
- insufficient commitment;
- wrong mindset;
- did not follow the protocol;
- underlying inflammation;
- more time needed.

Some of those explanations may genuinely be correct.

But if every failure is assigned to the client while every success is assigned to the method, the causal model is asymmetric.

We have:

$$
\text{success} \Rightarrow \text{method}
$$

and

$$
\text{failure} \Rightarrow \text{client}.
$$

A theory constructed this way cannot lose.

That is not a strength.

It is a failure of falsifiability.

## Science requires permission to be wrong

One of the deepest differences between scientific culture and authority culture is what happens after error.

Science has mechanisms, imperfect but explicit, for correction:

- replication;
- peer criticism;
- reanalysis;
- corrections;
- retractions;
- updated meta-analysis;
- new trials;
- revised guidelines.

Influencer authority can create the opposite incentive.

A confident claim becomes part of the brand.

Thousands of people repeat it.

Products are built around it.

Changing position then carries reputational and commercial cost.

This creates **belief lock-in**.

The stronger the brand is tied to being right, the harder it becomes to say:

> I overstated that.

Yet that sentence is one of the clearest signs of scientific integrity.

## Selling and science are not incompatible

There is nothing inherently unethical about making money from expertise.

Researchers consult.

Clinicians practise privately.

Coaches charge for their time.

Companies sell useful products.

The problem begins when commercial optimisation changes the rules for truth.

A defensible model is

$$
\text{commercial activity}
+
\text{transparent incentives}
+
\text{proportionate claims}
+
\text{auditable evidence}.
$$

The dangerous model is

$$
\text{commercial activity}
+
\text{authority}
+
\text{selected success}
+
\text{unfalsifiable explanation}.
$$

The distinction is not capitalism versus science.

It is evidence versus persuasion.

## What genuine “science-based” communication should look like

A communicator does not need to be an academic.

They do not need to publish papers.

They do not need a PhD.

But if science is invoked as authority, there should be observable standards.

A useful checklist is:

| Question | Strong scientific behaviour |
| --- | --- |
| What is the denominator? | Show everyone relevant, not only successes |
| What is the comparator? | Explain what would likely happen otherwise |
| What else changed? | Identify co-interventions |
| How uncertain is the result? | Show variation, not only averages or maxima |
| What evidence supports the mechanism? | Distinguish mechanism from clinical outcome |
| What evidence contradicts the claim? | Discuss it rather than hiding it |
| Is something being sold? | Make the relationship obvious |
| What would change your mind? | Give a falsifiable answer |
| Did your view change? | Correct the record publicly |
| Is this outside your expertise? | Reduce certainty rather than expanding authority |

None requires institutional status.

All require epistemic discipline.

## A useful hierarchy

There is also a practical hierarchy for evaluating claims.

### Level 1: Testimonial

> This happened to me.

Useful for hypothesis generation.

Very weak for general causal inference.

### Level 2: Case series

> This happened to several selected people.

More information.

Still vulnerable to selection and lack of comparator.

### Level 3: Complete programme outcomes

> Here is what happened to everyone who enrolled, including dropouts.

Much more informative about real-world performance.

Still observational.

### Level 4: Comparative observational evidence

> Comparable people using different approaches had different outcomes.

Potentially useful, but confounding remains.

### Level 5: Randomized comparative evidence

> Assignment attempts to balance known and unknown confounders.

Stronger causal inference when conducted well.

### Level 6: Replicated synthesis

> Multiple studies, populations and investigators converge.

Usually much harder for one charismatic story to overturn.

The mistake is presenting Level 1 with the confidence of Level 6.

## “But science can be wrong”

Of course.

Science is often wrong.

That is not an argument for replacing it with testimonials.

It is an argument for better science.

The relevant comparison is not

$$
\text{science}
\quad\text{versus}\quad
\text{certainty}.
$$

It is

$$
\text{methods designed to detect error}
\quad\text{versus}\quad
\text{methods designed to persuade}.
$$

A randomized trial can be badly designed.

A meta-analysis can be biased.

A guideline can become outdated.

But those failures can be investigated because the methods are visible.

A testimonial has almost no error-detection machinery.

## Popularity should increase responsibility, not certainty

Influence changes the ethical stakes.

A person speaking to 200 followers and a person speaking to 2 million may make the same factual error.

The second error can propagate much further.

A larger audience therefore should not justify:

> I must know what I am doing because people follow me.

A more defensible conclusion is:

> More people may act on what I say, so I should be more careful about what I claim.

Formally,

$$
\text{responsibility}
\propto
\text{reach}
\times
\text{potential harm}.
$$

Not every bad nutrition claim causes serious harm.

But when communication expands into medication, cancer, hormones, mental health, cardiovascular disease or supplement safety, the cost of overconfidence rises quickly.

## Influencers can also be excellent science communicators

This article is not an argument against influencers.

Some of the best science communication occurs outside universities.

Influencers can:

- translate technical evidence;
- reach audiences researchers cannot;
- model healthy behaviour;
- make useful information memorable;
- correct misinformation rapidly;
- build communities around beneficial change.

A systematic review of influencer interventions found both positive and negative health effects depending on context.

The platform is not the problem.

The evidential standard is.

Source:

- [Powell & Pring, 2024](https://pubmed.ncbi.nlm.nih.gov/38070305/)

## The real divide is not academic versus influencer

That divide is mostly noise.

The more useful distinction is:

$$
\text{auditable}
\quad\text{versus}\quad
\text{authority-dependent}.
$$

An auditable communicator shows enough of the reasoning that someone else can challenge it.

An authority-dependent communicator ultimately asks:

> Do you trust me?

Science should survive the answer:

> No. Show me.

## Conclusion

Results matter.

Experience matters.

Client outcomes matter.

Commercial success can matter.

Popularity matters if the question is communication.

None of those variables is worthless.

They simply answer different questions.

A transformation can show that change is possible.

It cannot identify the causal mechanism by itself.

A hundred testimonials can show that satisfied clients exist.

They cannot estimate effectiveness without the denominator.

A large audience can demonstrate influence.

It cannot establish biological truth.

A profitable business can demonstrate demand.

It cannot validate a scientific claim.

And when scientific authority is used to sell something, the appropriate standard should become **higher**, not lower.

The essential distinction is:

$$
\boxed{
\text{results are data points; evidence is a method for interpreting them}
}
$$

The moment popularity, testimonials or revenue are used as protection against criticism, science has stopped being the standard.

Authority has replaced it.

---

## References

1. Powell J, Pring T. **The impact of social media influencers on health outcomes: Systematic review.** *Social Science & Medicine*. 2024;340:116472. https://pubmed.ncbi.nlm.nih.gov/38070305/

2. Helou V, Mouzahem F, Makarem A, et al. **Conflict of interest and funding in health communication on social media: a systematic review.** *BMJ Open*. 2023;13:e072258. https://pmc.ncbi.nlm.nih.gov/articles/PMC10432670/

3. **A Meta-Analysis of Parasocial Relationships and Health: Examining Theoretical Mechanisms, Health Outcomes, and Moderating Factors.** *Health Communication*. 2026. https://pubmed.ncbi.nlm.nih.gov/41830519/

4. Li Y, Liu Z, Liu F. **Parasocial Engagement With Social Media Influencers and Mental Health Outcomes: Systematic Review and Meta-Analysis.** *JMIR Mental Health*. 2026;13:e96331. https://pubmed.ncbi.nlm.nih.gov/42623515/

5. Federal Trade Commission. **Advertisement Endorsements: The FTC's Endorsement Guides.** https://www.ftc.gov/news-events/topics/truth-advertising/advertisement-endorsements

6. Federal Trade Commission. **Health Products Compliance Guidance.** https://www.ftc.gov/business-guidance/resources/health-products-compliance-guidance

7. Federal Trade Commission. **Advertising FAQs: A Guide for Small Business — Endorsements and Testimonials.** https://www.ftc.gov/business-guidance/resources/advertising-faqs-guide-small-business

8. World Health Organization. **Infodemic.** https://www.who.int/health-topics/infodemic

---

*This article discusses general standards of evidence and health communication. It does not refer to, assess or diagnose any particular influencer or individual. Commercial interests are relevant to transparency and potential bias; they are not evidence that a person is dishonest or that a claim is automatically false.*
