---
permalink: '/biographies/paulerdos/'
author_profile: false
categories:
- Biographies
classes: wide
date: '2023-08-22'
excerpt: Paul Erdős transformed twentieth-century combinatorics, number theory, graph theory, and probabilistic mathematics through an unusually collaborative style of research.
header:
  image: /assets/images/headers/photo-data-science-dashboard.jpg
  og_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_image: /assets/images/headers/photo-data-science-dashboard.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-dashboard.jpg
  twitter_image: /assets/images/headers/photo-data-science-dashboard.jpg
keywords:
- Paul Erdős
- Erdős biography
- Probabilistic method
- Extremal graph theory
- Number theory
- Combinatorics
- Erdős-Kac theorem
- Erdős-Rényi random graphs
- Erdős number
seo_description: A mathematical biography of Paul Erdős, focusing on his work in combinatorics, number theory, graph theory, probability, and his distinctive collaborative style.
seo_title: 'Paul Erdős: Problems, Proofs, and Collaboration'
seo_type: article
subtitle: Problems, Proofs, and a Collaborative Mathematics
tags:
- Number Theory
- Combinatorics
- Graph Theory
title: The Life and Mathematics of Paul Erdős
---

![Erdos Paul - The Life and Mathematics of Paul Erdős](/assets/images/Erdos_Paul.jpg){: width="1120" height="814" loading="lazy"}
<p align="center"><i>Paul Erdős</i></p>

Paul Erdős was one of the most prolific mathematicians of the twentieth century, but raw publication count is not the main reason his influence remains so visible. He helped shape modern combinatorics and probabilistic mathematics, posed an enormous number of problems that redirected research programmes, and developed a style of collaboration that connected otherwise separate mathematical communities. His work ranged across number theory, extremal combinatorics, graph theory, probability, set theory, geometry, and approximation theory.

The familiar anecdotes about Erdős living from a suitcase, moving between collaborators, and referring to particularly elegant arguments as proofs from "The Book" are part of the historical record. They are interesting because they reflect something real about how he worked, but they should not obscure the mathematics. Erdős mattered because he repeatedly found elementary-looking questions whose solutions required new ideas, and because he treated problem selection itself as a serious mathematical skill.

## Early life and mathematical formation

Erdős was born in Budapest in 1913 to two mathematics teachers. He showed unusual numerical ability as a child and entered Péter Pázmány University in Budapest, where he completed his doctorate in 1934. His early research was already in number theory, including work related to primes and arithmetic functions. As antisemitism intensified in Hungary and across Europe, he spent increasing amounts of time abroad, including periods in Britain and the United States.

His career did not follow the standard path of a permanent university appointment followed by students and a local research group. Instead, Erdős became a travelling collaborator. He would visit mathematicians, work intensely on open problems, move on, and often return later with new questions. This made him unusually effective at transmitting problems and methods between fields.

## Number theory: elementary methods with deep consequences

One of Erdős's characteristic strengths was the use of elementary arguments to obtain nontrivial asymptotic results. A famous example is his independent proof, with Atle Selberg, of the prime number theorem by elementary means. The theorem states that

$$
\pi(x) \sim \frac{x}{\log x},
$$

where $\pi(x)$ is the number of primes not exceeding $x$. Earlier proofs relied on complex analysis through the Riemann zeta function. The Selberg-Erdős argument showed that the theorem could instead be reached using real-variable and combinatorial ideas.

Another important result is the Erdős-Kac theorem. Let $\omega(n)$ denote the number of distinct prime factors of $n$. For a uniformly chosen integer $n\leq x$, the normalized quantity

$$
\frac{\omega(n)-\log\log n}{\sqrt{\log\log n}}
$$

converges in distribution to a standard normal random variable as $x\to\infty$. The result is remarkable because it links the deterministic arithmetic structure of integers with a probabilistic limit law. It became one of the foundational examples of probabilistic number theory.

Erdős also worked extensively on additive number theory, Diophantine approximation, distribution of prime factors, and extremal problems involving sets of integers. His style was often to isolate a concrete quantitative question and then attack it with a mixture of counting, averaging, probabilistic intuition, and clever inequalities.

## The probabilistic method

Perhaps Erdős's most influential methodological contribution was the probabilistic method. The central idea is simple: to prove that a combinatorial object with some desired property exists, place a probability distribution over a class of candidate objects and show that the probability of the desired property is positive. No explicit construction is required.

Suppose a random object $X$ is drawn from a finite class. If one can show

$$
\Pr(X\text{ has property }P) > 0,
$$

then at least one deterministic object with property $P$ must exist.

This principle sounds almost tautological, but Erdős demonstrated that it could solve difficult extremal problems. Random colourings, random graphs, and random subsets often provided existence proofs where direct constructions were elusive. The method later developed into a major branch of combinatorics, with tools such as the Lovász local lemma, concentration inequalities, randomised rounding, and entropy methods extending the original idea.

## Extremal graph theory and Ramsey theory

Erdős played a central role in the development of extremal graph theory, where one asks how large or dense a graph can be while avoiding a forbidden configuration. The Erdős-Stone theorem is one of the field's cornerstone results. It identifies the asymptotic extremal density of graphs that avoid a fixed subgraph $H$ in terms of the chromatic number $\chi(H)$.

If $\operatorname{ex}(n,H)$ denotes the maximum number of edges in an $n$-vertex graph containing no copy of $H$, then for $\chi(H)\ge 2$,

$$
\operatorname{ex}(n,H)
=
\left(1-\frac{1}{\chi(H)-1}+o(1)\right)\frac{n^2}{2}.
$$

The theorem shows that, asymptotically, the chromatic structure of the forbidden graph controls the extremal density.

Erdős also transformed Ramsey theory. Instead of constructing large graphs with no large clique and no large independent set, he showed probabilistically that such graphs exist. This gave lower bounds on Ramsey numbers and demonstrated the power of random structures in a subject that had traditionally been approached deterministically.

## Random graphs

The name Erdős is also inseparable from random graph theory. With Alfréd Rényi, he studied what is now called the Erdős-Rényi random graph model. In one formulation, $G(n,p)$, each of the $\binom{n}{2}$ possible edges is included independently with probability $p$.

The importance of the model is not that real networks are literally generated this way. Its importance is conceptual. It allows precise study of threshold phenomena: properties such as connectivity, isolated vertices, and giant components can appear abruptly as $p$ changes with $n$. This helped establish random graphs as mathematical objects in their own right and created a framework later used throughout combinatorics, probability, network science, and theoretical computer science.

## Collaboration as a mathematical method

Erdős published with hundreds of coauthors. The well-known "Erdős number" assigns Erdős number 1 to his direct collaborators, number 2 to their collaborators, and so on. It began as a mathematical joke, but it also reflects the unusually broad collaboration network he created.

His collaborative style was based on problems. Erdős often arrived with a notebook of conjectures and open questions, some accompanied by cash prizes. The prize amounts were not a ranking of mathematical importance in any formal sense; they were usually an informal signal of how difficult or interesting he believed a problem to be.

This way of working had consequences beyond his own papers. Problems moved rapidly across institutions and subfields. Younger mathematicians encountered important open questions directly. Ideas that might otherwise have remained local were circulated internationally. In that sense, Erdős functioned as a research network long before digital collaboration made such networks routine.

## "The Book" and proof aesthetics

Erdős frequently spoke of an imaginary "Book" in which God had written the most elegant proof of every theorem. The phrase was not theology in any systematic sense; it was shorthand for a mathematical aesthetic. A proof from "The Book" was short, surprising, inevitable in retrospect, and free of unnecessary machinery.

Martin Aigner and Günter Ziegler later used that idea as the title of *Proofs from THE BOOK*, a collection of especially elegant arguments. The phrase remains associated with Erdős because it captures his preference for elementary structure and conceptual economy.

## A legacy of problems as well as theorems

Erdős's influence cannot be summarized by a short list of named results. His contribution was distributed across thousands of problems, techniques, conjectures, collaborations, and partial results. Some of his questions have since been solved; others remain open. The mathematical areas he helped build, particularly extremal combinatorics, probabilistic combinatorics, and random graph theory, are now central fields.

His life also illustrates an important point about mathematical research. Major influence does not require a single grand theory. It can come from identifying the right questions, developing reusable proof methods, and creating intellectual connections across communities.

Erdős died in Warsaw in 1996 while attending a mathematical meeting. That setting was fitting. For most of his adult life, mathematics was not something he did in one institution at fixed hours. It was a continuous collaborative activity organised around problems.

## References

- Aigner, M., & Ziegler, G. M. (2018). *Proofs from THE BOOK* (6th ed.). Springer.
- Erdős, P., & Kac, M. (1940). The Gaussian law of errors in the theory of additive number theoretic functions. *American Journal of Mathematics*, 62(1), 738-742.
- Erdős, P., & Rényi, A. (1959). On random graphs I. *Publicationes Mathematicae*, 6, 290-297.
- Erdős, P., & Stone, A. H. (1946). On the structure of linear graphs. *Bulletin of the American Mathematical Society*, 52, 1087-1091.
- Hoffman, P. (1998). *The Man Who Loved Only Numbers*. Hyperion.
- Schechter, B. (1998). *My Brain Is Open: The Mathematical Journeys of Paul Erdős*. Simon & Schuster.

![N is number - The Life and Mathematics of Paul Erdős](/assets/images/n_is_number.jpeg){: width="174" height="289" loading="lazy"}
<p align="left"><i>N is a Number</i></p>

![Proofs from the book - The Life and Mathematics of Paul Erdős](/assets/images/proofs_from_the_book.png){: width="743" height="862" loading="lazy"}
<p align="center"><i>Proofs from THE BOOK</i></p>

![The man - The Life and Mathematics of Paul Erdős](/assets/images/the_man.jpg){: width="614" height="1000" loading="lazy"}
<p align="center"><i>The Man Who Loved Only Numbers</i></p>
