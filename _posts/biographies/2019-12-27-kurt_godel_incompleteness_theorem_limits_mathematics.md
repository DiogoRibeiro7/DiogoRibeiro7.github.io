---
author_profile: false
categories:
- Biographies
classes: wide
date: '2019-12-27'
excerpt: Kurt Gödel transformed mathematical logic through the completeness theorem, incompleteness theorems, consistency results for set theory, and a surprising solution of Einstein's field equations.
header:
  image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  og_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  overlay_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-heesch-solid.jpg
  twitter_image: /assets/images/headers/photo-mathematics-heesch-solid.jpg
keywords:
- Kurt Gödel
- incompleteness theorems
- mathematical logic
- completeness theorem
- constructible universe
seo_description: Kurt Gödel's work in logic, including the completeness and incompleteness theorems, the constructible universe, and his rotating-universe solution in general relativity.
seo_title: 'Kurt Gödel: Completeness, Incompleteness, and Formal Systems'
seo_type: article
summary: A mathematical biography of Kurt Gödel that states the incompleteness theorems with their actual hypotheses and places them alongside his work in set theory and relativity.
tags:
- Mathematical Logic
- Foundations
- Biographies
title: 'Kurt Gödel: Completeness, Incompleteness, and Formal Systems'
---

Kurt Gödel (1906–1978) changed mathematical logic twice in opposite-looking directions.

First, he proved a **completeness theorem** for first-order logic.

Then he proved **incompleteness theorems** for sufficiently strong formal theories containing arithmetic.

There is no contradiction between the two results.

They concern different notions of completeness.

Understanding that distinction is the best way to understand Gödel's work.

## From Brno to Vienna

Gödel was born in 1906 in Brünn, then part of Austria-Hungary and now Brno in the Czech Republic.

He studied at the University of Vienna, initially with broad interests in mathematics and physics.

He became involved with the intellectual world around the Vienna Circle but did not share its philosophical commitments in any simple way.

His doctoral work was in mathematical logic.

## The completeness theorem

Gödel's 1929 doctoral dissertation established the completeness of first-order predicate logic.

In modern notation, semantic consequence is written

$$
\Gamma\models\varphi,
$$

meaning every model satisfying the premises $\Gamma$ also satisfies $\varphi$.

Syntactic provability is written

$$
\Gamma\vdash\varphi.
$$

Gödel's completeness theorem says, roughly,

$$
\Gamma\models\varphi
\quad\Longrightarrow\quad
\Gamma\vdash\varphi.
$$

Combined with soundness,

$$
\Gamma\vdash\varphi
\quad\Longrightarrow\quad
\Gamma\models\varphi,
$$

we obtain

$$
\Gamma\models\varphi
\iff
\Gamma\vdash\varphi.
$$

First-order logic is therefore complete in the sense that every logically valid consequence has a formal proof.

## Why incompleteness does not contradict completeness

Gödel's later incompleteness theorems concern **theories expressed in first-order logic**, such as formal arithmetic.

A deductive logic can be complete while a particular effectively axiomatized theory formulated in that logic is incomplete.

The distinction is:

$$
\text{completeness of the logical proof system}
$$

versus

$$
\text{completeness of a theory's axioms}.
$$

This is one of the most frequently blurred distinctions in popular accounts of Gödel.

## Arithmetization of syntax

Gödel's technical breakthrough was to encode formulas, proofs, and metamathematical statements as integers.

A symbolic expression can be assigned a Gödel number.

Sequences of symbols and proof steps can then be represented arithmetically.

This lets arithmetic express statements about formal proofs inside arithmetic itself.

That self-reference is not an informal paradox.

It is built through precise coding and a diagonal construction.

## The first incompleteness theorem

A modern version can be stated as follows.

Let $T$ be a consistent, effectively axiomatized formal theory strong enough to represent a sufficient amount of elementary arithmetic.

Then $T$ is incomplete: there exists a sentence $G$ such that neither

$$
T\vdash G
$$

nor

$$
T\vdash\neg G.
$$

The exact hypotheses vary among formulations, and Gödel's original 1931 proof used a stronger consistency condition for part of the result.

Later refinements, including Rosser's theorem, weakened the consistency requirements.

The safe conclusion is not that **every** formal system is incomplete.

Weak systems can be complete.

The theorem applies to effectively axiomatized systems with enough arithmetic expressive power.

## “True but unprovable” needs qualification

Popular accounts often say:

> Gödel proved that every sufficiently powerful formal system contains true statements that cannot be proved.

That can be a useful intuition, but the word **true** requires a semantic interpretation.

For a sufficiently sound arithmetic theory, the Gödel sentence constructed for the theory is true in the intended standard model of the natural numbers while unprovable in the theory.

Mere consistency alone is not identical to semantic soundness.

This distinction matters because incompleteness is fundamentally a theorem about formal provability under specified assumptions.

## The second incompleteness theorem

Gödel's second theorem concerns consistency statements.

Let

$$
\operatorname{Con}(T)
$$

be an arithmetic sentence formalizing the claim that theory $T$ has no proof of contradiction.

Under standard derivability conditions, if $T$ is consistent and sufficiently strong, then

$$
T\nvdash\operatorname{Con}(T).
$$

This does not mean no consistency proof is possible in any stronger framework.

A stronger theory may prove the consistency of a weaker one.

The theorem blocks a sufficiently strong consistent theory from providing the relevant internal proof of its own consistency.

That was a direct obstacle to the strongest form of Hilbert's finitistic consistency program.

## Set theory and the constructible universe

Gödel's work did not end with incompleteness.

In the late 1930s he developed the constructible universe,

$$
L,
$$

an inner model of set theory built in a cumulative definability hierarchy.

Using this construction, Gödel proved relative consistency results for the Axiom of Choice and the Generalized Continuum Hypothesis.

Informally:

> If the standard axioms of set theory are consistent, then adding Choice and the Generalized Continuum Hypothesis does not create inconsistency.

Later, Paul Cohen proved complementary independence results using forcing.

Together, the work of Gödel and Cohen established that the Continuum Hypothesis cannot be settled from the usual Zermelo-Fraenkel axioms with Choice, assuming consistency.

## Gödel and computability

Gödel's incompleteness work and computability theory developed in close historical proximity, but the incompleteness theorem is not simply the Halting Problem in another form.

Turing's Halting Problem asks whether there is an algorithm deciding, for every program and input, whether execution eventually stops.

The answer is no.

Both results reveal limitations of effective formal procedures, and the methods are deeply related through coding and self-reference.

But they are distinct theorems with distinct hypotheses and conclusions.

## The Institute for Advanced Study

Gödel visited the Institute for Advanced Study in Princeton beginning in the 1930s and became a permanent member after leaving Europe.

He later joined the faculty.

His friendship with Albert Einstein became one of the better documented intellectual relationships at the Institute.

They often walked together and discussed mathematics, physics, and philosophy.

The friendship is historically interesting, but it should not overshadow Gödel's own work in physics.

## A rotating universe

In 1949, Gödel published a solution to Einstein's field equations describing a rotating cosmological model.

The spacetime contains closed timelike curves.

In such a geometry, there exist future-directed timelike paths that return to their starting spacetime event.

The result did not show that our universe permits practical time travel.

It showed that general relativity's field equations admit solutions with surprising global causal structure.

This raised deep questions about time in relativity.

## Philosophical views

Gödel held strong philosophical views, including a form of mathematical Platonism.

He believed mathematical objects and truths were not merely formal symbol manipulations.

His incompleteness theorems are sometimes used as if they mechanically proved Platonism.

They do not.

The mathematical theorems constrain formal systems.

Their philosophical interpretation remains an additional argument.

Gödel himself drew philosophical conclusions from his work, but theorem and interpretation should be separated.

## Later life

Gödel experienced serious health and psychiatric difficulties during his life.

Historical accounts document recurrent fears concerning poisoning and food, particularly late in life.

After his wife Adele was hospitalized and unable to prepare food for him, Gödel's intake declined severely.

He died in Princeton in January 1978 from malnutrition and inanition recorded in connection with a personality disturbance.

These biographical facts should be reported carefully rather than used as a dramatic explanation of his mathematics.

## Legacy

Gödel's work established several different kinds of limit and possibility:

$$
\text{first-order logic}
\rightarrow
\text{semantic completeness},
$$

$$
\text{formal arithmetic}
\rightarrow
\text{syntactic incompleteness},
$$

$$
\text{set theory}
\rightarrow
\text{relative consistency},
$$

$$
\text{general relativity}
\rightarrow
\text{unexpected causal geometry}.
$$

The lasting lesson is not that mathematics “failed.”

It is that formal systems themselves became mathematical objects whose powers and limitations could be proved.

## References

- Gödel, K. (1930). The completeness of the axioms of the functional calculus of logic.
- Gödel, K. (1931). On formally undecidable propositions of *Principia Mathematica* and related systems I.
- Gödel, K. (1940). *The Consistency of the Axiom of Choice and of the Generalized Continuum-Hypothesis with the Axioms of Set Theory*. Princeton University Press.
- Gödel, K. (1949). An example of a new type of cosmological solutions of Einstein's field equations of gravitation. *Reviews of Modern Physics*, 21(3), 447–450.
- Institute for Advanced Study. *Kurt Gödel: Life, Work, and Legacy*.
- Stanford Encyclopedia of Philosophy. *Gödel's Incompleteness Theorems*.
- Dawson, J. W. (1997). *Logical Dilemmas: The Life and Work of Kurt Gödel*. A K Peters.
