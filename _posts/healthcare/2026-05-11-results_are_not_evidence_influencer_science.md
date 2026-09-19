---
permalink: '/healthcare/results_are_not_evidence_influencer_science/'
title: 'When Results Become Rhetoric: Evidence, Authority and Commercial Incentives in Online Health Communication'
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
seo_title: 'Results, Testimonials and Scientific Authority in Online Health'
seo_description: 'In a commercial programme with 60,164 clients, 6.6% were still attending after a year. The best twenty results of a programme that does nothing look like the best twenty of one that works, once it has enough clients. What client results, audience size and revenue can and cannot establish.'
seo_type: article
excerpt: >-
  Client results can be genuine without supporting the explanation used to sell
  a method. A wall of the twenty best transformations says more about how many
  clients a business has had than about what its programme does, and the
  published numbers on retention, spontaneous improvement and disclosure show
  how large the gap is.
summary: >-
  Three claims travel together in online health communication: that a client
  improved, that the programme caused it, and that the seller's explanation is
  right. This essay separates them using published data on retention in a
  commercial programme, an exact calculation of what selected testimonials show,
  trials with untreated arms, studies of accuracy and disclosure on social media,
  and the n-of-1 trial as the honest version of personal evidence.
keywords:
- health influencers
- fitness influencers
- causal inference
- testimonials
- science communication
- scientific authority
- conflicts of interest
- evidence based practice
why_this_exists: >-
  Online health communication often blurs three different questions: whether a
  person improved, whether an intervention caused that improvement, and whether
  the explanation offered for the improvement is scientifically correct. Those
  questions require different forms of evidence, and the size of the gap between
  them can be put in numbers.
evidence: >-
  Retention and weight loss among 60,164 clients of a commercial programme
  (Finley et al., 2007); a meta-analysis of 37 trials with untreated, placebo and
  active arms (Krogsbøll et al., 2009); the range of response to one training
  programme in 585 people (Hubal et al., 2005); accuracy of 510 posts from
  Instagram accounts with at least 100,000 followers (Denniss et al., 2024);
  reviews and measurements of disclosure (Helou et al., 2023; Mathur et al.,
  2018); and an exact calculation of the expected mean of the best twenty results
  among n clients.
methodology: >-
  Figures from published studies are quoted from their abstracts and used only
  for arithmetic that a reader can repeat. The testimonial-wall calculation
  assumes normally distributed individual results and is exact under that
  assumption; it is checked against simulation in the test suite. Commercial
  relationships are treated as a source of bias to be disclosed, not as proof of
  dishonesty. No individual or business is assessed.
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
Question: What can client results, testimonials, popularity and commercial success legitimately establish in health communication?
Claim: They can document experience, feasibility, reach and demand, but they do not establish causal efficacy or validate the scientific mechanism used to explain an outcome.
Counterclaim: Real-world outcomes are not worthless. Complete and transparently reported programme data can provide useful observational evidence and generate hypotheses.
Evidence object: Published retention data from a commercial programme, an exact order-statistics calculation of what a wall of selected results shows, three-armed trials with untreated arms, measured accuracy and disclosure on social media, and n-of-1 trials.
Failure case: Treating every commercial relationship as corrupt, every testimonial as useless, or academic credentials as sufficient proof of correctness.
Reader payoff: A rigorous way to distinguish observed success from causal evidence and persuasive authority from scientific authority, with the size of each gap in numbers.
Exclusions: Named individuals, personality judgements, speculation about motives and private conduct.
-->

The successful health influencer rarely presents as an advertiser. The model that works is epistemic: the same person sells a programme, a supplement or a coaching service and supplies the theory of why it works, so that the sales proposition and the scientific proposition arrive together. When the theory is challenged, one answer comes back more often than any other, which is that the method produces results. There are transformations, satisfied clients, better laboratory values, a large audience and a profitable business.

All of that may be true, and none of it answers the question that was asked. An observed outcome is not a causal effect, and a causal effect is not a correct explanation of its mechanism. A client can improve, the programme can have contributed, and the account of why it worked can still be wrong. This essay takes those three claims apart and, wherever published data allow, puts a number on the distance between them.

## Three Claims That Travel Together

Suppose someone joins a weight-management programme and loses 12 kg in six months. The difference between before and after is an observation. The causal claim is about something else: in the potential-outcomes notation, the effect of the programme on person $i$ is $Y_i(1) - Y_i(0)$, the outcome with the programme minus the outcome the same person would have had over the same six months without it. Only one of the two is ever observed. Randomisation recovers the missing one on average across a group, and careful observational designs try to approximate it (Hernán & Robins, 2020). A testimonial does neither, because it reports $Y_i(1)$ and leaves the reader to assume that $Y_i(0)$ would have been no change at all.

The third claim is further away still. A typical online programme changes many things at once: resistance training, daily walking, protein, alcohol, home cooking, sleep, self-weighing, weekly accountability, and a supplement. If health improves, the bundle deserves credit as a way of organising behaviour. It does not follow that each component mattered, or that the biological story told about one of them is right, because that component was never varied while the others were held fixed. A client can say truthfully that the programme worked for them while the seller's conclusion, that this proves a theory about insulin or cortisol or inflammation, has no support from the same facts.

Mechanistic vocabulary makes the third claim feel stronger than it is. Insulin, cortisol, mitochondrial function, autophagy and gut permeability are real phenomena, and naming one does not show that it is the pathway through which a particular intervention acts. A pathway can exist and contribute almost nothing to the outcome of interest, compensating systems can cancel it, the dose that matters may never be reached in people, and a biomarker can move without any change in the risk of disease. Plausibility of mechanism is a reason to run a study. It is not the result of one.

## The Denominator, Measured Once

The problem with testimonials is not that they are false. It is that the reader is shown $P(\text{success} \mid \text{featured})$, which is close to one by construction, and wants $P(\text{success} \mid \text{enrolled})$, which needs the people who were not featured. Commercial programmes almost never publish that denominator. One did. Finley and colleagues (2007) analysed the records of all 60,164 adults who enrolled in a large commercial weight-loss programme over one year, and followed each of them for 52 weeks.

![Bar chart of retention among 60,164 clients of a commercial weight-loss programme: 73% still attending at week 4, 42% at week 13, 22% at week 26 and 6.6% at week 52. Those still attending had lost 8.3%, 12.6% and 15.6% of their body weight at weeks 13, 26 and 52. Data from Finley and colleagues, 2007.](/assets/images/figures/results_rhetoric_retention.png){: width="1152" height="704" loading="lazy"}

The clients who stayed a year lost 15.6% of their body weight on average, which is a large and clinically meaningful amount, and they were 6.6% of those who enrolled: about 3,970 people out of 60,164, or one in fifteen. More than a quarter, some 16,200 people, had left within four weeks, having lost 1.1%. Every transformation photograph that such a business could publish would be genuine, and a reader who took those photographs as typical would be wrong about 93.4% of the clients.

Two further points follow from the same table. The first is that staying is not independent of succeeding: people who lose weight early have a reason to continue and people who do not have a reason to stop, so the completers' average would be high even if the programme added nothing to what they would have achieved anyway. The second is that the weight of those who left is known only at their last visit. The study cannot say what the programme caused, because it has no comparison group, and its authors conclude only that the programme can be effective for those who remain in it. What it shows is how different the visible results and the enrolled population can be when both are measured honestly.

This asymmetry also protects a method from ever being wrong. When a client succeeds, the programme is credited. When a client fails, the explanation moves to adherence, motivation or individual biology. Those explanations are sometimes correct, but a system that assigns successes to the method and failures to the participant cannot be contradicted by any outcome, and a claim that no outcome could contradict is not supported by the outcomes that happen to agree with it.

## What the Best Twenty Show

A frequent reply is that the evidence is not one anecdote but hundreds. Numbers reduce random error and do nothing about selection, and with testimonials they make matters worse in a way that can be computed. Suppose individual results vary around a mean effect $\delta$ with standard deviation $\sigma$, and a business with $n$ clients displays its best $k$. The expected mean on display is

$$
\mathbb{E}\left[\bar{X}_{\text{wall}}\right] = \delta + \sigma\, m(n, k),
$$

where $m(n, k)$ is the expected mean of the $k$ largest of $n$ standard normal values. The effect of the programme enters once, as $\delta$. Everything else is selection, and it grows with $n$ without limit. When $k$ is small beside $n$, write $p = k/n$ for the share of clients on display and $z_p = \Phi^{-1}(1 - p)$ for the point that cuts off that share of a normal distribution. A good approximation is then

$$
m(n, k) \approx \frac{\varphi(z_p)}{p},
$$

the mean of a normal distribution above its upper $p$ quantile. The figure uses the exact value, which for a wall of twenty is 1.74 with 200 clients, 2.41 with 1,000, 3.16 with 10,000 and 3.78 with 100,000.

![Line chart of the expected mean result among the best twenty clients, in standard deviations of individual results, against the number of clients on a logarithmic axis from one hundred to one million. Three parallel rising lines: a programme with no effect goes from 1.4 at 100 clients to 2.4 at 1,000 and 4.3 at a million; effects of half and of one standard deviation lie that much higher. A dashed line marks that a large effect with 1,000 clients and no effect with 24,000 clients produce the same wall, 3.4.](/assets/images/figures/results_rhetoric_testimonial_wall.png){: width="1152" height="736" loading="lazy"}

Consider a programme with a genuinely large effect, one standard deviation, and 1,000 clients. Its wall shows 3.41 standard deviations, of which 71% is selection and 29% is the programme. A programme with no effect at all shows the same wall once it has had about 24,000 clients, and a more impressive one after that. To give the unit a size, take the spread that Finley's 26-week cohort showed, 5.1% of body weight: the best twenty of a programme that does nothing then average a loss of 12.3% of body weight with 1,000 clients and 19.3% with 100,000. The wall is a measurement of how many people have passed through the business. That is why "hundreds of clients" is not an answer to the charge of selection, and why the largest operators will always have the most striking results whatever they sell.

The normal distribution is an assumption, but the wide spread of individual results is not. Hubal and colleagues (2005) put 585 adults through an identical twelve-week arm-training programme and measured the biceps by magnetic resonance imaging. The change in muscle size ranged from a loss of 2% to a gain of 59%, and the change in one-repetition strength from nothing to 250%. One programme, delivered the same way to everyone, produced both the person with nothing to show and the person whose photograph would sell it.

## Improvement That Needs No Treatment

People seek help at unusual moments: pain at its worst, weight at a personal maximum, an alarming blood test. Extreme measurements tend to be followed by less extreme ones whether or not anything is done, which is regression to the mean (Barnett et al., 2005), and many conditions fluctuate or resolve on their own. A [companion article]({{ '/science-communication/what_before_and_after_testimonials_can_establish/' | relative_url }}) derives the size of that effect for a recruited cohort. Here the question is how large the untreated improvement is in practice, and there are trials built to measure it.

Krogsbøll and colleagues (2009) collected 37 randomised trials, with 2,900 patients across eight conditions, that had three arms: no treatment, placebo and active treatment. Measured as change from baseline in standard deviations, patients given nothing improved by 0.24, those given placebo by 0.44 and those given the active treatment by 1.01. A treated patient's before-and-after story therefore contains 0.24 of improvement that needed no treatment and a further 0.20 that needed only the ritual of one. The treatment itself accounts for 56% of the change, so the before-and-after difference overstates what the treatment did by a factor of 1.8, and that is for treatments that work. For depression the untreated share of the improvement was 35%, and for acute pain 25%.

A visible physique is subject to the same reasoning with more causes in play. Response to training varies enormously between people, as the arm-training study shows, and training is not the only thing that builds muscle. In a randomised trial, men given testosterone for ten weeks who did no exercise at all added 9 kg to their bench press, while men given placebo and no exercise lost 1 kg (Bhasin et al., 1996). This says nothing about any individual. It says that a body is an outcome with several causes, among them inheritance, years of training, pharmacology and photography, and that displaying the outcome does not identify which of them produced it, still less that it was the method on sale.

## What an Audience Measures

The second substitution is of audience size for correctness. Popularity has many causes, among them skill at communication, consistency, visual presentation, controversy, timing and advertising, and accuracy is at most one of them. Whether it is one of them can be tested. Denniss and colleagues (2024) assessed nutrition posts from 47 Australian Instagram accounts with at least 100,000 followers each. Of 510 posts checked against dietary guidelines and evidence databases, 44.7% contained inaccuracies. Of 676 rated for quality, 34.8% were poor, 59.2% mediocre, 6.1% good and none excellent.

The associations matter more than the averages. Posts written by dietitians or nutritionists were far more likely to be accurate, with an odds ratio of 4.69, and posts about supplements far less, with an odds ratio of 0.23. Neither the number of followers nor a verified account had any relationship with quality or accuracy, and engagement tended to be higher for posts of lower quality. A large audience is evidence of influence, and influence is real: a systematic review found effects of influencers on health behaviour in both directions, from twelve studies that the reviewers judged to be mostly of poor methodological quality (Powell & Pring, 2024). Nothing in that loop of reach, social proof and further reach tests the claim being repeated.

Expertise is also specific to a domain. A successful strength coach may know a great deal about adherence and programme design, and that knowledge does not extend by itself to oncology, toxicology, endocrinology or causal inference. The same holds inside universities, where a professorship in one field confers nothing in another. The useful question about any speaker is what, exactly, they are expert in, and whether the claim in front of the reader falls inside it.

## Money, Disclosure and What Is Withheld

Revenue is sometimes offered as a harder kind of validation, on the argument that a method which did not work would fail in the market. Markets measure demand, and demand responds to branding, identity, convenience, community, price, novelty and persuasion as well as to efficacy. Useful products are profitable, useless ones can be, and harmful ones have stayed profitable for a long time. When a claim about physiology is answered with growth in sales, a commercial quantity has been substituted for the biological one that was in dispute.

The opposite simplification is just as poor. Clinicians are paid, researchers compete for grants, universities sell education and drug companies fund trials, and none of that settles whether what they say is true. A financial interest is a source of bias, which is why it has to be visible, and the measurements say that it usually is not. A systematic review of 17 studies found that the share of health posts reporting a conflict of interest ranged from 0% to 60% and was mostly low, and that where physicians' payments from industry could be checked against a public register, between 98.7% and 100% of the relationships went unmentioned (Helou et al., 2023). Across more than 500,000 YouTube videos and 2.1 million Pinterest pins, about 10% of the content carrying affiliate links disclosed them at all (Mathur et al., 2018).

Disclosure of payment is the smaller half of the matter. A provider knows, or could know, how many people enrolled, how many left, how many did not improve, how many complained and how the featured cases were chosen, and the audience sees the featured cases. A communication can be truthful in every sentence and misleading as a whole. Advertising regulators have reached the same conclusion from another direction: the Endorsement Guides of the United States Federal Trade Commission treat an exceptional testimonial as potentially deceptive unless it comes with what consumers can generally expect, and say that a line such as "results not typical" does not repair it.

## Personal Evidence, Done Properly

None of this means that an individual's experience is worthless. It means that experience has to be collected in a way that can distinguish the treatment from everything else that changes, and for one person there is a design that does so. In an n-of-1 trial a single patient receives the treatment and a placebo in a randomised sequence of periods, blind, with the outcome recorded throughout. Guyatt and colleagues (1990) ran such trials as a service for doctors who were unsure whether a drug was helping a particular patient: of 70 begun, 57 were completed and 50 gave a definite answer.

The result worth remembering is what the answers did. In 15 trials, 39% of those where the doctor's prior plan was recorded, the result changed the plan, and in 11 of those the doctor stopped a drug that had been meant to continue indefinitely. These were drugs that the doctor had planned to continue, on the strength of how the patient seemed to be doing. When the experience was arranged so that it could have come out otherwise, in more than a third of cases it did. "It works for me" is a hypothesis, and a testable one.

## What a Results Claim Should Contain

A business that wants its results to count as evidence has a better object to offer than its best transformation, which is its cohort. The standard is not exotic. It is what Finley and colleagues published for a commercial programme, and each item answers a specific question that a reader is otherwise forced to guess at.

| What is reported | What it lets a reader work out |
| :--- | :--- |
| Everyone who enrolled, and over what period | The denominator that every other figure is a share of |
| How many were still there at each follow-up | Whether the results describe clients or survivors |
| The outcome and its timing, fixed in advance | Whether the measure was chosen because it looked good |
| Mean, spread and the share who got worse | The typical result and the range, not the best case |
| What is known about those who left | How far the completers' figures can be trusted |
| A comparison group, or a statement that there is none | How much of the change needs the programme to explain it |
| Adverse events and complaints | The cost side, which testimonials never show |
| Payments, affiliate links and products sold | What the speaker gains if the claim is believed |
{: .table-prose}

Such a report would still have limits. Clients select themselves, confounding remains, measurement is imperfect and the programme changes over time. The difference is that the limits are stated where a reader can weigh them. The label "science-based" should be read the same way, as a claim about method and not a badge. Citations do not establish it, since twenty references can be to animals, to mechanisms, to other doses and other populations. What establishes it is whether a reader can reconstruct the argument: the claim, the population, the outcome, the studies that bear on it directly, the size and uncertainty of the effect, the evidence against, and the observation that would change the speaker's mind.

That last item is where commerce and science pull hardest against each other. Once a claim is attached to a public identity or a product line, revising it is expensive, and the scientific habit of updating conflicts with the commercial value of consistency. The tension cannot be removed, only managed in the open. Reach raises the stakes, because an inaccurate claim made to hundreds of thousands of people alters diets, supplement use and decisions about medication, and the care owed to a claim should rise with its consequences as it moves from exercise technique towards disease, hormones and drugs.

Real-world results are not trivial. They show that change is possible, that a programme is feasible, that people value a service and that something deserves a proper study. They do not show what caused the change, and the numbers above say how wide that gap is: one client in fifteen still present at a year, a wall of results that is 71% selection even when the programme works well, a before-and-after difference 1.8 times the effect of a treatment that is effective, accuracy unrelated to audience size, and one affiliate link in ten disclosed. The divide that matters is not between academics and influencers. It is between a case that rests on who is speaking and what they can display, and one that gives the claim, the denominator, the comparison, the limits and the conditions under which it would be withdrawn.

## References

- Barnett, A. G., van der Pols, J. C., & Dobson, A. J. (2005). Regression to the mean: what it is and how to deal with it. *International Journal of Epidemiology*, 34(1), 215-220. <https://doi.org/10.1093/ije/dyh299>
- Bhasin, S., Storer, T. W., Berman, N., Callegari, C., Clevenger, B., Phillips, J., Bunnell, T. J., Tricker, R., Shirazi, A., & Casaburi, R. (1996). The effects of supraphysiologic doses of testosterone on muscle size and strength in normal men. *New England Journal of Medicine*, 335(1), 1-7. <https://doi.org/10.1056/NEJM199607043350101>
- Denniss, E., Lindberg, R., Marchese, L. E., & McNaughton, S. A. (2024). #Fail: the quality and accuracy of nutrition-related information by influential Australian Instagram accounts. *International Journal of Behavioral Nutrition and Physical Activity*, 21(1), 16. <https://doi.org/10.1186/s12966-024-01565-y>
- Federal Trade Commission. Advertisement endorsements: the Endorsement Guides and related guidance. <https://www.ftc.gov/news-events/topics/truth-advertising/advertisement-endorsements>
- Finley, C. E., Barlow, C. E., Greenway, F. L., Rock, C. L., Rolls, B. J., & Blair, S. N. (2007). Retention rates and weight loss in a commercial weight loss program. *International Journal of Obesity*, 31(2), 292-298. <https://doi.org/10.1038/sj.ijo.0803395>
- Guyatt, G. H., Keller, J. L., Jaeschke, R., Rosenbloom, D., Adachi, J. D., & Newhouse, M. T. (1990). The n-of-1 randomized controlled trial: clinical usefulness. Our three-year experience. *Annals of Internal Medicine*, 112(4), 293-299. <https://doi.org/10.7326/0003-4819-112-4-293>
- Helou, V., Mouzahem, F., Makarem, A., Noureldine, H. A., El-Khoury, R., Al Oweini, D., Halak, R., Hneiny, L., Khabsa, J., & Akl, E. A. (2023). Conflict of interest and funding in health communication on social media: a systematic review. *BMJ Open*, 13(8), e072258. <https://doi.org/10.1136/bmjopen-2023-072258>
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. <https://miguelhernan.org/whatifbook>
- Hubal, M. J., Gordish-Dressman, H., Thompson, P. D., Price, T. B., Hoffman, E. P., Angelopoulos, T. J., Gordon, P. M., Moyna, N. M., Pescatello, L. S., Visich, P. S., Zoeller, R. F., Seip, R. L., & Clarkson, P. M. (2005). Variability in muscle size and strength gain after unilateral resistance training. *Medicine & Science in Sports & Exercise*, 37(6), 964-972.
- Krogsbøll, L. T., Hróbjartsson, A., & Gøtzsche, P. C. (2009). Spontaneous improvement in randomised clinical trials: meta-analysis of three-armed trials comparing no treatment, placebo and active intervention. *BMC Medical Research Methodology*, 9, 1. <https://doi.org/10.1186/1471-2288-9-1>
- Mathur, A., Narayanan, A., & Chetty, M. (2018). Endorsements on social media: an empirical study of affiliate marketing disclosures on YouTube and Pinterest. *Proceedings of the ACM on Human-Computer Interaction*, 2(CSCW), 119. <https://doi.org/10.1145/3274388>
- Powell, J., & Pring, T. (2024). The impact of social media influencers on health outcomes: systematic review. *Social Science & Medicine*, 340, 116472. <https://doi.org/10.1016/j.socscimed.2023.116472>

*This essay concerns standards of evidence in public health communication. It does not assess any named individual, infer motives, or treat commercial activity as evidence of dishonesty.*
