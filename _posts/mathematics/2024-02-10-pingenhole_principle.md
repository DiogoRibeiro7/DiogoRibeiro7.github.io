---
permalink: '/mathematics/pingenhole_principle/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-02-10'
excerpt: "The pigeonhole principle turns finite counting constraints into existence proofs. Its power lies in choosing the right objects and boxes."
header:
  image: /assets/images/headers/photo-mathematics-cryptography-blackboard.jpg
  og_image: /assets/images/headers/photo-mathematics-cryptography-blackboard.jpg
  overlay_image: /assets/images/headers/photo-mathematics-cryptography-blackboard.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-cryptography-blackboard.jpg
  twitter_image: /assets/images/headers/photo-mathematics-cryptography-blackboard.jpg
keywords:
- Pigeonhole principle
- Combinatorics
- Dirichlet principle
- Existence proofs
- Counting
seo_description: "A rigorous introduction to the pigeonhole principle, generalized bounds, number-theory examples, handshake counts, and lossless compression."
seo_title: "The Pigeonhole Principle: Counting Forces Existence"
seo_type: article
tags:
- Combinatorics
- Number Theory
title: "The Pigeonhole Principle: Counting Forces Existence"
toc: false
---

The pigeonhole principle says that finite capacity constraints force collisions.

If $N$ objects are placed into $m$ boxes, then at least one box contains at least

$$
\left\lceil\frac{N}{m}\right\rceil
$$

objects.

The statement is elementary. The difficult part in applications is choosing the right objects and boxes.

## Equal remainders

Take $n+1$ integers and reduce them modulo $n$. There are only $n$ possible remainders, so two integers have the same remainder and their difference is divisible by $n$.

## Repeating decimals

In long division by positive integer $q$, each step has a remainder in $\{0,1,\ldots,q-1\}$. If zero occurs, the decimal terminates. Otherwise, a nonzero remainder must eventually repeat, so the decimal expansion becomes periodic.

## Handshake counts

At a party of $n$ people, each person can have shaken between 0 and $n-1$ hands. Counts 0 and $n-1$ cannot both occur, so at most $n-1$ distinct handshake counts are possible. By pigeonhole, at least two people have the same count.

## Birthday collisions

With 366 possible birthdays, any group of 367 people must contain a shared birthday. This certainty statement is different from the birthday paradox, which asks for a collision probability in smaller random groups.

## Lossless compression

There are $2^n$ binary strings of length $n$, but only

$$
1+2+\cdots+2^{n-1}=2^n-1
$$

binary strings of length strictly less than $n$.

Therefore no injective lossless compressor can shorten every $n$-bit input. Some inputs must remain the same size or expand.

## Generalized form

If every box contained at most $r$ objects, then $m$ boxes could hold at most $mr$ objects. Hence if

$$
N>mr,
$$

some box contains at least $r+1$ objects.

## Do not confuse pigeonhole with every counting invariant

The mutilated-chessboard domino problem is usually proved by a coloring invariant, not by pigeonhole alone. Calling every elementary counting contradiction a pigeonhole argument obscures the actual mechanism.

## Conclusion

The practical pattern is

$$
\text{objects}>\text{available categories}
\Longrightarrow
\text{collision}.
$$

The theorem is simple because the creativity lies in constructing the right categories.

## References

- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics*.
- van Lint, J. H., & Wilson, R. M. (2001). *A Course in Combinatorics*.
