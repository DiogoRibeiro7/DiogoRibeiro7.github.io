---
permalink: '/mathematics/combinatorics_python/'
author_profile: false
categories:
- Mathematics
classes: wide
date: '2024-02-11'
excerpt: "Python can enumerate combinatorial objects, but combinatorics is primarily about counting without enumeration and understanding computational explosion."
header:
  image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  og_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-mathematics-fractal-grid.jpg
  twitter_image: /assets/images/headers/photo-mathematics-fractal-grid.jpg
keywords:
- Combinatorics Python
- itertools
- combinations
- permutations
- binomial coefficients
- dynamic programming
seo_description: "A practical guide to combinatorics in Python using exact counting, lazy iteration, combinations, permutations, products, and complexity-aware enumeration."
seo_title: "Combinatorics with Python: Count Before You Enumerate"
seo_type: article
tags:
- Python
- Combinatorics
- Probability
title: "Combinatorics with Python: Count Before You Enumerate"
toc: false
---

Python's itertools module makes it easy to generate permutations and combinations, but the most important computational lesson in combinatorics is often the opposite: do not enumerate if you only need the count.

## Permutations

The number of permutations of $n$ distinct objects is $n!$.

~~~python
from itertools import permutations

for ordering in permutations(("a", "b", "c")):
    print(ordering)
~~~

Do not convert a large permutation iterator to a list unless every object is genuinely needed. For $n=12$,

$$
12!=479001600.
$$

## Combinations

The number of unordered selections of size $k$ from $n$ objects is

$$
\binom{n}{k}=\frac{n!}{k!(n-k)!}.
$$

For exact counts, Python's standard library provides `math.comb`.

~~~python
from math import comb

count = comb(49, 6)
print(count)
~~~

Use `itertools.combinations` only when the actual subsets are needed.

## Lazy iteration does not remove combinatorial explosion

An iterator reduces memory consumption, but the number of generated objects is unchanged. If $\binom{n}{k}$ is enormous, iterating lazily can still take impractical time.

## Cartesian products

If option sets have sizes $m_1,\ldots,m_d$, the Cartesian product contains

$$
\prod_{j=1}^d m_j
$$

configurations. This is why naive hyperparameter grids can explode exponentially.

## Lottery probability

If six distinct numbers are drawn uniformly from 49 without order, a fixed ticket wins with probability

$$
P(\text{jackpot})=\frac{1}{\binom{49}{6}}.
$$

No simulation is needed because the sample space is finite and symmetric.

## Dynamic programming and recurrences

Many counting problems are better solved through recurrences than enumeration. Dynamic programming stores overlapping subproblems and is often the computational counterpart of a combinatorial recurrence.

## Inclusion-exclusion

For sets $A_1,\ldots,A_m$,

$$
\left|\bigcup_i A_i\right|
=
\sum_i|A_i|
-
\sum_{i<j}|A_i\cap A_j|
+\cdots.
$$

This counts configurations satisfying at least one condition without listing them all.

## Backtracking and pruning

When enumeration is necessary, constraints can prune the search tree. A partial assignment that already violates a condition need not be extended.

## Type-safe helper

~~~python
from __future__ import annotations

from math import comb


def lottery_probability(total_numbers: int, picks: int) -> float:
    """Return jackpot probability for an unordered draw without replacement."""
    if total_numbers <= 0:
        raise ValueError("total_numbers must be positive")
    if not 0 <= picks <= total_numbers:
        raise ValueError("picks must be between 0 and total_numbers")

    return 1.0 / comb(total_numbers, picks)
~~~

## Conclusion

Before enumerating a combinatorial space, ask how many objects it contains. That count often determines whether the problem should be enumerated, sampled, optimized, or solved symbolically.

## References

- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics*.
- Knuth, D. E. *The Art of Computer Programming, Volume 4A: Combinatorial Algorithms*.
