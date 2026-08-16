---
author_profile: false
categories:
- Economics
classes: wide
date: '2022-01-01'
excerpt: Explore exchange rate models like Purchasing Power Parity (PPP) and Uncovered Interest Parity (UIP), key frameworks in global economics.
header:
  image: /assets/images/data_science_2.avif
  og_image: /assets/images/data_science_2.avif
  overlay_image: /assets/images/data_science_2.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.avif
  twitter_image: /assets/images/data_science_2.avif
keywords:
- Exchange rate models
- Purchasing power parity
- Uncovered interest parity
- Currency valuation
redirect_from:
- '/mathematical economics/exchange_rate_models_understanding_ppp_uip/'
seo_description: Learn about exchange rate models such as Purchasing Power Parity (PPP) and Uncovered Interest Parity (UIP) and their role in international finance.
seo_title: 'Exchange Rate Models: PPP and UIP Explained'
seo_type: article
summary: An overview of exchange rate models with a focus on Purchasing Power Parity (PPP) and Uncovered Interest Parity (UIP), including their principles, applications, and limitations.
tags:
- Economics
title: 'Exchange Rate Models: Understanding PPP and UIP'
---

Exchange rates are among the hardest prices in economics to explain. Two parity conditions — Purchasing Power Parity and Uncovered Interest Parity — provide the standard theoretical anchors. Both are elegant, both are intuitive, and both are rejected by the data over short horizons. Understanding *why* they fail is more instructive than either condition alone.

## Purchasing Power Parity

PPP starts from the law of one price: an identical good should cost the same everywhere once prices are converted to a common currency. If it does not, arbitrage should close the gap.

**Absolute PPP** applies this to the whole price level:

$$
S = \frac{P}{P^*},
$$

where $S$ is the spot exchange rate in domestic currency per unit of foreign currency, $P$ the domestic price level and $P^*$ the foreign one. A basket costing £100 in Britain and <span class="tex2jax_ignore">$130</span> in the United States implies $S = 1.30$ dollars per pound.

**Relative PPP** is the weaker and more useful form. It claims not that levels equalise but that *changes* track inflation differentials:

$$
\frac{\Delta S}{S} \approx \pi - \pi^*,
$$

so a country with inflation 3 percentage points above its trading partner should see its currency depreciate by roughly 3% a year. Relative PPP can hold even when absolute PPP fails, which matters because absolute PPP fails badly.

### Why PPP Fails in the Short Run

The empirical record is unambiguous: deviations from PPP are large and persistent, with a half-life of roughly three to five years. As a short-run predictor it is close to useless.

Several mechanisms explain this. Many goods are simply not tradable — housing, haircuts, medical care — and no arbitrage force acts on them. Transport costs, tariffs and distribution margins create bands within which price differences persist without any profitable trade. Price stickiness means nominal prices adjust slowly while exchange rates move continuously, so the ratio is knocked away from parity far faster than goods markets can restore it. And national price indices weight different baskets, so $P$ and $P^*$ are not measuring the same thing.

The **Balassa-Samuelson effect** explains a systematic component of the deviation rather than treating it as noise. Rich countries have higher productivity in tradable sectors, which raises wages economy-wide, which raises the price of non-tradables, which makes their overall price level higher. Richer countries therefore have systematically overvalued currencies relative to absolute PPP — not an anomaly but a prediction.

This is why PPP-adjusted GDP comparisons are standard: market exchange rates systematically understate real incomes in poorer countries.

## Uncovered Interest Parity

UIP concerns financial rather than goods arbitrage. If domestic assets pay more than foreign ones, capital should flow in until expected returns equalise:

$$
i - i^* = \mathbb{E}\!\left[\frac{\Delta S}{S}\right],
$$

the interest differential should equal the expected rate of depreciation. A currency offering 5% while its partner offers 2% must be expected to depreciate by about 3%, otherwise the higher rate is free money.

The word **uncovered** is doing real work. Covered Interest Parity uses a forward contract to lock in the future rate, eliminating exchange risk, and CIP holds tightly in practice — it is close to an arbitrage identity, enforced by trade, and deviations were negligible until the 2008 crisis introduced funding frictions and balance-sheet costs that opened persistent small gaps.

UIP instead leaves the position exposed and relies on *expectations*. That difference is why one holds and the other does not.

### The Forward Premium Puzzle

UIP is not merely rejected by the data; it is rejected with the wrong sign. Regressing realised depreciation on the interest differential,

$$
\frac{\Delta S_{t+1}}{S_t} = \alpha + \beta (i_t - i_t^*) + \varepsilon_{t+1},
$$

UIP predicts $\beta = 1$. Estimates across currencies and periods typically come out near zero or negative, often around $-0.8$.

A negative coefficient means high-interest currencies tend to *appreciate* rather than depreciate — the opposite of the prediction. This is the **forward premium puzzle**, and it is the empirical basis of the carry trade: borrow in a low-interest currency, lend in a high-interest one, and historically earn both the interest differential and a currency gain.

Three explanations compete. A time-varying **risk premium** would mean investors demand compensation for holding high-interest currencies, so the excess return is payment for risk rather than a free lunch — consistent with the carry trade's tendency to produce steady small gains punctuated by severe crashes. **Peso problems** attribute the pattern to small probabilities of large devaluations that happen not to occur in sample, so measured returns overstate expected returns. And **expectational errors** hold that market forecasts are simply biased, which survey data on exchange rate expectations partly supports.

None of these is fully settled, which is unusual for a regularity documented this consistently for this long.

## Comparing the Two

| | PPP | UIP |
|---|---|---|
| Arbitrage in | Goods markets | Capital markets |
| Speed of adjustment | Very slow (half-life 3-5 years) | Should be immediate |
| Short-run empirical support | Poor | Poor, and wrong-signed |
| Long-run empirical support | Reasonable, especially relative PPP | Weak |
| Main practical use | Long-run valuation, income comparison | Benchmark for excess returns |

The complementarity is real: PPP anchors the real exchange rate over long horizons while UIP concerns nominal returns over short ones. Their combination is the basis of the monetary model, in which the exchange rate is driven by relative money supplies and output.

Meese and Rogoff's finding remains the disciplining fact: at horizons under a year, no structural exchange rate model — monetary, portfolio-balance, or otherwise — reliably beats a random walk out of sample. Decades of subsequent work has qualified this at longer horizons but not overturned it.

## What to Take From This

Neither condition should be used as a short-run forecast. Both are useful as equilibrium benchmarks: a currency far from its PPP value is not guaranteed to revert soon, but the deviation is informative, and half-lives of several years mean the pull is real even when it is slow.

The general lesson is worth more than either condition. Parity relationships describe forces that operate when nothing impedes them. Real markets are full of impediments — non-traded goods, sticky prices, risk premia, transaction costs, and expectations that are not rational in the model's sense. The gap between the parity condition and the data is not a failure of the theory so much as a measurement of those frictions, and that gap is usually the interesting quantity.

## References

- Rogoff, K. (1996). The purchasing power parity puzzle. *Journal of Economic Literature*, 34(2), 647-668.
- Fama, E. F. (1984). Forward and spot exchange rates. *Journal of Monetary Economics*, 14(3), 319-338.
- Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: do they fit out of sample? *Journal of International Economics*, 14(1-2), 3-24.
- Balassa, B. (1964). The purchasing-power parity doctrine: a reappraisal. *Journal of Political Economy*, 72(6), 584-596.
- Engel, C. (2014). Exchange rates and interest parity. In *Handbook of International Economics* (Vol. 4). Elsevier.
