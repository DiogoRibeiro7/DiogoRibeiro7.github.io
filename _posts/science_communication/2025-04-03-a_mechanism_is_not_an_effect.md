---
permalink: '/science-communication/a_mechanism_is_not_an_effect/'
title: 'A Mechanism Is Not an Effect'
date: '2025-04-03'
categories:
- Science Communication
tags:
- Mechanistic Evidence
- Causal Inference
- Surrogate Endpoints
- Scientific Reasoning
- Evidence
author_profile: false
classes: wide
seo_title: 'Mechanistic Evidence Does Not by Itself Establish an Effect'
seo_description: 'A causal mechanism can be real while the net effect is small, absent, or harmful. A quantitative model and a historical clinical trial show why.'
seo_type: article
excerpt: >-
  Establishing one causal pathway does not establish the total effect of an
  intervention. Competing pathways, dose, scale, and surrogate outcomes can
  change both the magnitude and the direction of the result.
summary: >-
  This article examines what mechanistic evidence can and cannot establish.
  A simple causal model separates pathway specific effects from the total
  effect of an intervention, followed by a historical example from the
  Cardiac Arrhythmia Suppression Trial in which a treatment successfully
  changed its intended intermediate outcome while increasing mortality.
  The discussion then considers surrogate endpoints, dose response,
  competing pathways, and the legitimate role of mechanistic evidence.
keywords:
- mechanistic evidence
- causal mechanism
- causal inference
- surrogate endpoint
- scientific reasoning
- total causal effect
why_this_exists: >-
  Scientific arguments often move from a plausible mechanism to a claim about
  a real world outcome without establishing the size of the pathway, the
  relevance of the exposure, or the contribution of competing pathways. This
  article makes that inferential gap explicit.
evidence: >-
  A mathematical decomposition of pathway effects, the Cardiac Arrhythmia
  Suppression Trial, methodological work on surrogate endpoints and
  mechanistic reasoning, and the IARC framework for integrating mechanistic
  evidence with other evidence streams.
methodology: >-
  Express the total causal effect as the sum of contributions through multiple
  pathways. Construct numerical examples in which a genuine beneficial
  mechanism coexists with a null or harmful total effect. Compare this with
  empirical evidence from a randomised trial in which suppression of an
  intermediate cardiac outcome failed to predict mortality.
reviewed_at: '2025-04-03'
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
Question: What can be inferred from evidence that an intervention acts through a scientifically plausible mechanism?
Claim: Evidence for one causal pathway is evidence about that pathway, not automatically about the sign or magnitude of the total effect on the outcome of interest.
Counterclaim: Mechanistic evidence is scientifically valuable and can sometimes contribute strongly to causal inference, particularly when the relevant chain is well characterised and competing pathways are constrained.
Evidence object: Linear causal decomposition, numerical counterexamples, the Cardiac Arrhythmia Suppression Trial, dose response reasoning, and methodological literature on surrogate endpoints and mechanistic evidence.
Failure case: Treating all mechanistic evidence as weak, assuming randomised trials reveal mechanisms automatically, or concluding that an intervention is ineffective merely because its proposed mechanism is incomplete.
Reader payoff: Distinguish evidence that a pathway exists from evidence about the total effect, identify the missing quantities in a mechanism based argument, and recognise when intermediate outcomes cannot substitute for the outcome that actually matters.
Exclusions: Evaluating a named contemporary product, providing medical advice, claiming that mechanisms never establish causation, and treating one historical clinical trial as representative of all surrogate endpoint failures.
-->

A common scientific argument has a simple structure. An intervention changes some biological, chemical, or physical quantity. That quantity is related to an outcome. Therefore, the intervention is expected to change the outcome.

Nothing in this chain is necessarily false. The intervention may genuinely act on the proposed mechanism. The intermediate quantity may genuinely participate in the process that produces the outcome. The direction of that particular pathway may even be understood correctly. The conclusion can nevertheless fail because an outcome is rarely determined by one pathway in isolation.

The distinction is between demonstrating **a mechanism** and estimating **the total effect produced by the system in which that mechanism operates**.

This is not an argument against mechanistic evidence. Mechanisms are often essential to scientific explanation. They help distinguish plausible causal structures, guide experiments, explain heterogeneity, identify adverse effects, and determine whether results obtained in one setting can reasonably generalise to another. The problem begins when evidence about one part of a causal system is treated as if it had already determined the behaviour of the whole system.

## A pathway and a total effect are different quantities

Let an intervention be represented by $A$, where $A=1$ denotes exposure to the intervention and $A=0$ denotes its absence. Suppose the intervention changes an intermediate quantity $M$, which then contributes to an outcome $Y$.

A simple linear representation is

$$
M = \mu_M + \alpha A + \varepsilon_M
$$

and

$$
Y = \mu_Y + \beta M + \gamma A + \varepsilon_Y.
$$

The parameter $\alpha$ represents the effect of the intervention on the proposed mechanism. The parameter $\beta$ describes the contribution of the intermediate quantity to the outcome. The parameter $\gamma$ represents every contribution of the intervention to the outcome that is not transmitted through $M$.

Under this simplified model, the expected total effect of the intervention is

$$
\Delta Y = \alpha\beta + \gamma.
$$

Evidence that the mechanism exists may establish that $\alpha\neq 0$. If the relation between $M$ and $Y$ is itself causal, there may also be evidence about $\beta$. Neither result determines the total effect unless something is known about $\gamma$.

Consider an intervention that reduces an undesirable intermediate quantity by 10 units, so that

$$
\alpha=-10.
$$

Suppose each unit reduction in that quantity improves the final outcome by $0.2$ units. The contribution through the proposed mechanism is then

$$
\alpha\beta=(-10)(0.2)=-2.
$$

If lower values of $Y$ are preferable, the mechanism contributes a two unit benefit.

Now suppose the intervention also produces another physiological change whose net contribution to the same outcome is

$$
\gamma=3.
$$

The total effect is

$$
\Delta Y=-2+3=1.
$$

The proposed beneficial mechanism is real. Its direction is correct. Its effect is large enough to measure. The intervention nevertheless worsens the final outcome by one unit because another pathway is larger.

No contradiction has occurred. The error would have been to identify $\alpha\beta$ with $\Delta Y$.

Real systems usually contain considerably more than one omitted pathway. A slightly more realistic representation is

$$
\Delta Y
=
\alpha_1\beta_1
+
\alpha_2\beta_2
+
\cdots
+
\alpha_k\beta_k,
$$

where the intervention changes several intermediate processes and each contributes to the final outcome. Interactions between pathways can make even this additive form inadequate, but the simpler expression is already enough to expose the central problem. Knowing one term in the sum does not identify the sum.

## A mechanism can be correct and still answer the wrong question

Mechanistic arguments are particularly persuasive when the intermediate quantity is strongly associated with the outcome. If people with higher values of $M$ tend to experience worse outcomes, reducing $M$ appears to be an obvious objective.

The reasoning is incomplete because three different statements have been combined.

The first is that $M$ predicts $Y$. The second is that changing $M$ changes $Y$. The third is that changing $M$ through a particular intervention changes $Y$ by the amount expected from the first two relationships.

These are not equivalent statements.

A variable can predict an outcome without being a useful intervention target. It may be a marker of another process rather than an important cause of the outcome. Even when it is causal, two interventions producing the same change in the variable need not have the same effect on the final outcome because they may differ in everything else they change.

The distinction becomes clearer in potential outcome notation. Let $S(a)$ denote the value of an intermediate or surrogate outcome under intervention state $a$, and let $Y(a)$ denote the corresponding final outcome. The average intervention effects are

$$
\tau_S
=
\mathbb E[S(1)-S(0)]
$$

and

$$
\tau_Y
=
\mathbb E[Y(1)-Y(0)].
$$

A positive or negative value of $\tau_S$ does not determine the sign of $\tau_Y$. Even a strong observational relationship between $S$ and $Y$ does not establish that interventions moving $S$ in a favourable direction will necessarily move $Y$ in the same direction.

This possibility is sometimes called the **surrogate paradox**. An intervention can improve a surrogate outcome, the surrogate can be associated with the clinical outcome in the expected direction, and the intervention can still worsen the clinical outcome.

This is not merely a theoretical construction.

## Suppressing an arrhythmia without improving survival

After myocardial infarction, ventricular premature depolarisations were known to be associated with subsequent sudden cardiac death. The clinical reasoning was understandable. If ventricular arrhythmias identify patients at increased risk, and a drug can suppress those arrhythmias, suppression might reduce the probability of arrhythmic death.

The Cardiac Arrhythmia Suppression Trial tested that hypothesis.

During the initial phase of the study, patients were treated with antiarrhythmic drugs and the ability of the drugs to suppress ventricular arrhythmias was measured. Patients whose arrhythmias were successfully suppressed could then enter the randomised comparison between continued active treatment and placebo.

The proposed intermediate effect was therefore not imaginary. The drugs could suppress the ventricular ectopic activity they were intended to suppress.

The clinical outcome did not follow.

In the preliminary 1989 report, the part of the trial involving encainide and flecainide was stopped because mortality was higher among patients receiving active treatment. Arrhythmic death or nonfatal cardiac arrest occurred in 33 of 730 patients receiving encainide or flecainide, compared with 9 of 725 receiving placebo. The reported relative risk was 3.6. Total mortality was also higher, with 56 deaths among 730 patients receiving active treatment and 22 among 725 receiving placebo, corresponding to a reported relative risk of 2.5.

The intended intermediate outcome moved in the desired direction while the outcome of greater clinical importance moved in the opposite direction.

The trial did not show that ventricular arrhythmias were irrelevant. Nor did it show that suppressing an arrhythmia can never be beneficial. It showed something more specific and more useful: successful suppression of that intermediate outcome by those drugs in that population did not establish a survival benefit.

The causal structure contained information that the intermediate measurement could not capture.

In terms of the earlier model, observing a favourable $\alpha\beta$ did not reveal the remaining pathways contributing to $\Delta Y$. The later analyses of the trial documented excess arrhythmic and cardiac mortality among patients assigned to active treatment. The precise biological explanation for the excess mortality was itself not completely established by the trial. Randomisation identified the harmful treatment effect more securely than it identified every mechanism responsible for that harm.

This distinction is worth preserving. Experiments can provide strong evidence about whether an intervention changes an outcome without providing a complete explanation of why that change occurs.

## The surrogate problem is a causal problem

A surrogate endpoint is useful because the outcome we care about may take years to observe, require a very large sample, or be difficult to measure directly. A biomarker, laboratory value, imaging result, or physiological measurement can often be obtained sooner.

The practical attraction is obvious. The inferential problem is harder.

For a surrogate $S$ to replace an outcome $Y$, it is not enough for $S$ to predict $Y$. It is not enough for treatment to change $S$. It is not even enough for both statements to hold simultaneously.

What is required is information about how intervention effects on $S$ relate to intervention effects on $Y$.

One particularly restrictive causal structure would be

$$
A \longrightarrow S \longrightarrow Y,
$$

with no other pathway from $A$ to $Y$. Under that structure, once the causal effect of $S$ on $Y$ is known, changes in the surrogate can potentially carry much more information about the final outcome.

But many real systems instead resemble

$$
A \longrightarrow S \longrightarrow Y
$$

while the intervention also affects other processes that eventually reach $Y$.

Those additional paths are exactly what the term $\gamma$ represented in the earlier equation.

This is why validation of surrogate endpoints is a substantive causal problem rather than a matter of calculating a sufficiently impressive correlation coefficient. Fleming and DeMets made this point in their 1996 discussion of surrogate endpoints, emphasising that interventions can affect clinical outcomes through mechanisms not adequately captured by the proposed surrogate. Subsequent statistical work has formalised conditions under which apparent improvement in a surrogate can coexist with harm in the final outcome.

The same reasoning applies outside medicine. A model may optimise an intermediate engineering metric while worsening reliability. An environmental intervention may reduce one measured pollutant while increasing another. An educational programme may raise performance on an assessment without improving the broader capability that the assessment was intended to represent.

Whenever the measured quantity lies somewhere between the intervention and the objective, the possibility of competing pathways needs to be considered.

## Direction is not magnitude

There is another inferential step hidden inside many mechanistic arguments. Showing that a process can occur does not establish that it occurs at a magnitude sufficient to produce the claimed real world effect.

Dose and exposure are part of the mechanism.

Suppose the response of a biological system to concentration $c$ follows a Hill type relationship,

$$
R(c)
=
R_{\max}
\frac{c^n}
{EC_{50}^{\,n}+c^n}.
$$

For the simple case $n=1$,

$$
R(c)
=
R_{\max}
\frac{c}
{EC_{50}+c}.
$$

At a concentration equal to one tenth of $EC_{50}$,

$$
R(0.1EC_{50})
\approx
0.091R_{\max}.
$$

At ten times $EC_{50}$,

$$
R(10EC_{50})
\approx
0.909R_{\max}.
$$

The same mechanism exists at both concentrations. The magnitude of the response differs by a factor of ten.

This matters whenever evidence is transferred between experimental conditions. A molecular interaction observed in vitro may be genuine, while the concentration required to produce a substantial response may not be reached in the relevant tissue under realistic exposure. Conversely, a modest effect at low exposure does not imply that larger exposure is harmless. The point is not that laboratory evidence is intrinsically weak. It is that a mechanism includes its operating conditions.

Temperature, concentration, duration, timing, route of exposure, receptor availability, feedback, and competing reactions can all determine whether a mechanism that is physically or biologically possible becomes quantitatively important.

A statement of the form “substance $X$ activates pathway $M$” therefore contains less information than it first appears to contain. To predict an outcome, we also need to know how strongly $X$ changes $M$ under the relevant conditions, how strongly $M$ contributes to the outcome, and what else $X$ changes at the same time.

## Biological plausibility is evidence, not a numerical effect estimate

The opposite error is to respond to failures of mechanistic reasoning by dismissing mechanisms altogether.

That position is no better.

Mechanistic evidence can strengthen causal inference. It can show that an observed statistical association has a physically or biologically credible route. It can identify conditions under which a causal effect should disappear or become stronger. It can reveal why an average treatment effect differs across populations. It can also provide evidence about outcomes that have not yet been observed directly.

Howick and colleagues argued that the weakness of mechanistic reasoning does not arise simply from the fact that it is mechanistic. Reliability depends on the quality and completeness of the inferential chain and on whether relevant complexity has been considered.

Formal evidence evaluation also uses mechanistic information. The International Agency for Research on Cancer evaluates mechanistic evidence as one stream of evidence in carcinogenic hazard identification, alongside evidence concerning cancer in humans and experimental animals. Its methodology explicitly considers the relevance of experimental systems to humans, the consistency of findings, gaps in the evidence, and the possibility that different mechanisms operate in different settings.

This is considerably more demanding than finding a molecular pathway that points in the desired direction.

The appropriate conclusion is therefore not that mechanisms are subordinate decorations attached to statistical evidence. Nor is it that a plausible pathway establishes the outcome before the outcome has been measured. Mechanistic and outcome evidence answer overlapping but different questions, and their evidential value depends on how those questions fit together.

## A complete mechanism claim requires several quantities

When a claim rests heavily on mechanism, five questions are particularly useful.

1. **Does the intervention change the proposed intermediate process under the conditions that actually matter?** Evidence obtained at a different dose, duration, tissue, organism, or experimental environment may establish possibility without establishing relevance.

2. **Is the intermediate process causally related to the final outcome?** Prediction is not sufficient. A biomarker can identify high risk without being a useful target for intervention.

3. **How large is the pathway specific effect?** Direction alone does not determine practical importance.

4. **What other pathways does the intervention alter?** A favourable contribution through one mechanism can be cancelled or reversed elsewhere in the system.

5. **Has the outcome of interest been measured directly when direct measurement is feasible?** When outcome data disagree with a mechanistic prediction, the disagreement requires explanation rather than automatic dismissal of the outcome.

These questions convert a vague appeal to plausibility into a set of empirical claims. Some can be answered experimentally. Others require observational evidence, pharmacokinetic or physical modelling, mediation analysis, or further mechanistic work.

The important change is that the mechanism is no longer being asked to provide information that has not actually been measured.

## Mechanistic evidence becomes stronger when it makes risky predictions

A mechanism gains evidential value when it predicts observations that were not used merely to construct the explanation.

Suppose a proposed mechanism implies that an effect should increase with exposure until saturation, disappear when a specific pathway is blocked, differ between populations according to a measurable biological characteristic, and emerge within a particular time interval. If those predictions are subsequently observed, the mechanism has survived opportunities to fail.

A verbal explanation that can accommodate every possible outcome provides much less information.

This is one reason quantitative mechanisms are valuable. They force assumptions into the open. A claim that an intervention “reduces inflammation” can remain vague enough to survive almost any observation. A model specifying which mediator changes, by how much, at what dose, over what period, and with what expected consequence for an outcome has substantially more ways to be wrong.

Those possible failures are scientifically useful.

A mechanism should therefore do more than make an observed result sound plausible after the fact. It should restrict the set of results that ought to be observed if the mechanism is important.

## From pathway evidence to causal evidence

The statement that an intervention acts on a mechanism can be a meaningful scientific result. It may establish one link in a causal chain and rule out competing explanations that would not produce the same observation.

What it does not automatically establish is the effect on an outcome several steps downstream.

The simplest mathematical reason is already contained in

$$
\Delta Y=\alpha\beta+\gamma.
$$

Mechanistic evidence may estimate $\alpha$, provide information about $\beta$, or sometimes constrain part of $\gamma$. The total causal effect depends on all of them.

The Cardiac Arrhythmia Suppression Trial provides an unusually clear empirical reminder of this distinction. The intermediate physiological target could be changed successfully, yet the clinically important outcome did not improve and instead became worse under the treatments studied. The mechanism was not useless information. It was incomplete information.

Scientific explanation requires mechanisms because effects without explanation leave important questions unanswered. Scientific evaluation requires outcome evidence because a mechanism considered in isolation may leave equally important pathways unmeasured.

The strongest inference usually comes when the two agree for reasons that were specified before the result was known.

## References

Baker, S. G. (2018). Five criteria for using a surrogate endpoint to predict treatment effect based on data from multiple previous trials. *Statistics in Medicine*, 37(4), 507–518. doi:10.1002/sim.7561.

Cardiac Arrhythmia Suppression Trial Investigators. (1989). Preliminary report: effect of encainide and flecainide on mortality in a randomized trial of arrhythmia suppression after myocardial infarction. *New England Journal of Medicine*, 321(6), 406–412. doi:10.1056/NEJM198908103210629.

Echt, D. S., Liebson, P. R., Mitchell, L. B., et al. (1991). Mortality and morbidity in patients receiving encainide, flecainide, or placebo. The Cardiac Arrhythmia Suppression Trial. *New England Journal of Medicine*, 324(12), 781–788. doi:10.1056/NEJM199103213241201.

Fleming, T. R., & DeMets, D. L. (1996). Surrogate end points in clinical trials: Are we being misled? *Annals of Internal Medicine*, 125(7), 605–613. doi:10.7326/0003-4819-125-7-199610010-00011.

Howick, J., Glasziou, P., & Aronson, J. K. (2010). Evidence based mechanistic reasoning. *Journal of the Royal Society of Medicine*, 103(11), 433–441.

International Agency for Research on Cancer. (2019). *Preamble to the IARC Monographs on the Identification of Carcinogenic Hazards to Humans*.

International Agency for Research on Cancer. (2025). *Key Characteristics associated End points for Evaluating Mechanistic Evidence of Carcinogenic Hazards*. IARC Monographs Technical Report.

Prentice, R. L. (1989). Surrogate endpoints in clinical trials: Definition and operational criteria. *Statistics in Medicine*, 8(4), 431–440. doi:10.1002/sim.4780080407.

VanderWeele, T. J. (2013). Surrogate measures and consistent surrogates. *Biometrics*, 69(3), 561–569.
