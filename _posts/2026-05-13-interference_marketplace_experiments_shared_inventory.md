---
permalink: '/statistics/interference_marketplace_experiments_shared_inventory/'
title: 'Interference in Experiments: When Treated Users Take What Control Users Would Have Bought'
categories:
- Statistics
tags:
- A/B Testing
- Experimental Design
- Causal Inference
- Marketplaces
author_profile: false
seo_title: 'Interference and Spillovers in Marketplace Experiments'
seo_description: 'A user-level A/B test assumes one user’s treatment does not change another user’s outcome. In a marketplace with shared inventory it does, and the test reports a 20 percent lift where the true rollout lift is 3. A simulation shows the bias, why it happens, and the designs that recover the truth.'
excerpt: >-
  The ranking change raises purchase intent from 10 to 12 percent, and
  the A/B test reports a 20 percent lift in sales. Rolled out to everyone,
  it delivers 3 percent, because the warehouse holds 200 units a day and
  treated buyers were buying the units control buyers would have bought.
summary: >-
  The assumption every user-level test makes and where marketplaces
  break it, a simulated market with shared daily inventory in which the
  buyer-level split reports the same lift regardless of stock while the
  true rollout lift shrinks to nothing, the arithmetic of why the ratio
  survives a stock-out, two designs that recover the truth, randomising
  separate markets and switching the whole market between arms over
  time, what each costs in precision, the other direction in which
  spillovers bias a test, and how to recognise interference before the
  experiment runs.
keywords:
  - interference
  - SUTVA
  - marketplace experiments
  - spillover
  - cannibalisation
  - switchback design
  - cluster randomisation
classes: wide
date: '2026-05-13'
why_this_exists: >-
  Two-sided platforms, marketplaces, ad auctions and anything with a
  shared constraint are where the most confident A/B results are the
  most wrong, and the failure is invisible from inside the test. This post
  builds the smallest market in which it happens, measures it, and shows
  the designs that would have told the truth.
evidence: >-
  A simulated market with 2,000 buyers a day and a daily inventory from
  300 to 200 units, a treatment that raises purchase intent from 10 to
  12 percent, twenty-day experiments under a buyer-level split, a
  randomisation of twenty separate cities, and a day-by-day switchback;
  200 replications per design and inventory, with the true rollout lift
  computed from the same demand draws.
methodology: >-
  Compares each design's estimated lift with the lift from treating
  everyone versus no one, decomposes the buyer-level estimate for one
  day without noise to show why the stock-out preserves the ratio of
  intents, and reports the spread of each estimator across replications.
reviewed_at: '2026-09-12'
header:
  image: /assets/images/headers/photo-motherboard.jpg
  og_image: /assets/images/headers/photo-motherboard.jpg
  overlay_image: /assets/images/headers/photo-motherboard.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-motherboard.jpg
  twitter_image: /assets/images/headers/photo-motherboard.jpg
---
The new ranking puts the right products in front of buyers, and the A/B test proves it: buyers in the treatment arm purchase 20 percent more than buyers in the control arm, with a tight confidence interval and a p-value that ends the discussion. The ranking is rolled out to everyone. The following month, sales are up 3 percent.

Nothing was wrong with the test's statistics. What was wrong was an assumption the test never stated: that the treatment given to one buyer does not change the outcome of another. In this market the warehouse holds 200 units a day, and a buyer who was persuaded to purchase took a unit that a control buyer, arriving later, would otherwise have bought. The treatment arm's gain was partly the control arm's loss, and the difference between them measured the transfer, not the creation, of sales.

## The Assumption With the Long Name

The stable unit treatment value assumption, SUTVA, says each unit's outcome depends only on its own treatment. It is what makes a comparison of two randomised groups an estimate of what would happen if everyone got the treatment. It fails whenever units share something that the treatment changes: inventory, drivers, ad impressions, a matching pool, a social feed, a support queue, a budget. In a marketplace it fails by construction, because the market exists to make units compete for the same things.

When it fails, the user-level experiment still measures something, the difference between treated and untreated users inside a market where both exist, but that quantity is not the effect of rolling out the treatment, which is what the decision needs. The gap between them can be any size, and in the simulation below it is a factor of seven.

## A Simulated Market

Two thousand buyers arrive each day in random order, and each buys with probability 10 percent under the old ranking and 12 percent under the new one, if a unit is still in stock. Inventory is fixed per day. The true lift is computed by running two parallel worlds with the same demand draws, one with everyone treated and one with no one, and comparing total sales. Three experimental designs are then run on the same market.

```python
import numpy as np

rng = np.random.default_rng(0)
buyers_per_day = 2000
p_control, p_treat = 0.10, 0.12          # purchase probability given the item is available

def day_sales(intents, inventory, r=rng):
    """Buyers arrive in random order; the first `inventory` intents are served."""
    order = r.permutation(len(intents))
    served = np.cumsum(intents[order]) <= inventory
    sold = np.zeros(len(intents), bool)
    sold[order] = intents[order] & served
    return sold

def simulate(inventory, design, days=20, r=rng, n_markets=20):
    """Returns (estimated lift, true lift) for one experiment of `days` days."""
    truth_c = truth_t = 0.0
    est_t = est_c = 0.0
    for d in range(days):
        ic = r.random(buyers_per_day) < p_control          # intents if everyone were in control
        it = r.random(buyers_per_day) < p_treat            # intents if everyone were treated
        truth_c += min(ic.sum(), inventory); truth_t += min(it.sum(), inventory)
        if design == "buyer":                               # buyers split 50/50 inside one shared market
            z = r.integers(0, 2, buyers_per_day)
            sold = day_sales(np.where(z == 1, it, ic), inventory, r)
            est_t += sold[z == 1].sum() / (z == 1).mean(); est_c += sold[z == 0].sum() / (z == 0).mean()
        elif design == "market":                            # separate cities of the same size, each with its own stock
            zm = np.repeat([0, 1], n_markets // 2); r.shuffle(zm)
            st = sc = 0.0
            for m in range(n_markets):
                intents = r.random(buyers_per_day) < (p_treat if zm[m] else p_control)
                s = day_sales(intents, inventory, r).sum()
                if zm[m]: st += s
                else: sc += s
            est_t += st / zm.sum(); est_c += sc / (n_markets - zm.sum())
        elif design == "switchback":                        # the whole market treated on alternate days
            s = day_sales(it if d % 2 == 0 else ic, inventory, r).sum()
            if d % 2 == 0: est_t += 2 * s
            else: est_c += 2 * s
    return est_t / est_c - 1, truth_t / truth_c - 1

reps = 200
for inventory in (300, 230, 210, 200):
    print(f"inventory {inventory}")
    for design in ("buyer", "market", "switchback"):
        res = np.array([simulate(inventory, design) for _ in range(reps)])
        print(f"  {design:11} estimated {res[:, 0].mean():+.1%} (sd {res[:, 0].std():.1%})   true {res[:, 1].mean():+.1%}")
```

**Estimated and true lift by daily inventory.** Control demand is about 200 units a day and treated demand about 240.

| Daily inventory | True rollout lift | Buyer-level split | Separate cities randomised | Switchback by day |
| --- | --- | --- | --- | --- |
| 300 (ample) | +20% | +20.4% ± 3.5% | +20.1% ± 0.7% | +19.9% ± 3.0% |
| 230 | +13.9% | +19.8% ± 3.2% | +14.0% ± 0.5% | +14.0% ± 2.7% |
| 210 | +6.0% | +20.2% ± 3.5% | +6.0% ± 0.4% | +5.9% ± 1.8% |
| 200 (control demand) | +2.8% | +20.5% ± 3.8% | +2.8% ± 0.3% | +2.8% ± 1.3% |

With 300 units a day nothing binds, every design agrees, and the ranking's 20 percent is real. With 230 units the treated demand of 240 hits the ceiling, and the true lift from rolling out is 14 percent; the buyer-level test still says 20. At 210 the truth is 6 percent and the test says 20. At 200, where control demand already sells out the inventory, the ranking cannot raise sales at all beyond the 3 percent that comes from filling the days on which control demand fell short, and the buyer-level test says 20 percent with the same confidence as before. Its estimate does not depend on the inventory, which is the signature of the problem: it is measuring a quantity that the constraint does not enter.

![Estimated lift against daily inventory for the buyer-level split and for randomised cities, with the true rollout lift. Ample inventory gives 20 percent for all; as stock binds the truth falls toward zero while the buyer-level estimate stays at 20 percent.](/assets/images/figures/marketplace_interference_lift.png){: width="1152" height="672" loading="lazy"}

## Why the Ratio Survives a Stock-Out

The arithmetic for a single day with 210 units and no noise in the intents makes the mechanism plain. A thousand treated buyers have 120 intents and a thousand control buyers have 100, for 220 intents against 210 units. Buyers arrive in random order, so the stock-out removes the same fraction from each group: treated sales of about 114.5 and control sales of about 95.5. The ratio is still 1.2, and the test reports a 20 percent lift. Rolling out to everyone produces 240 intents capped at 210, against 200 uncapped under control, and the lift is 5 percent.

The treated buyers' extra intents did not create sales at the margin; they displaced control buyers' sales, one for one, once the stock was gone. The test is comparing the two groups' shares of a fixed pie and reading the shift in shares as growth in the pie. Every shared constraint produces the same structure: a fixed number of drivers, a daily ad budget, a matching pool of candidates, a support team's capacity. Whatever the treatment competes for, the control arm is on the other side of the competition.

## Designs That Tell the Truth

Both fixes work by making the unit of randomisation the unit inside which the constraint operates.

**Randomise markets.** Twenty cities, each with its own buyers and its own inventory, ten treated and ten control. Nothing crosses between them, so each city's sales are what a rollout would produce there, and the comparison is unbiased at every inventory level. The estimate is precise here because each city is a full-sized market observed for twenty days; with fewer or smaller markets it would be a cluster-randomised experiment with all of that design's costs, few clusters and a between-market variance that customers do not reduce.

**Switch the whole market over time.** On even days everyone is treated, on odd days no one is, and the comparison is between days. There is no cross-arm competition because there is never more than one arm at a time. The estimate is unbiased as long as the market resets between periods; here inventory is daily, so it does. In a market with carry-over, inventory or drivers or buyers that persist across the switch, the periods have to be long enough for the carry-over to fade, or the analysis has to model it. The spread is larger than the city design's because a twenty-day experiment gives ten days per arm and day-to-day demand noise is the only source of variance.

A third family, two-sided or interleaved designs that randomise both buyers and sellers and analyse the corners, estimates the interference directly rather than avoiding it, at the price of a more complex analysis. The choice among them is about what can be randomised: if the market is one city and cannot be split, the switchback is the design; if there are many comparable markets, randomise them.

## The Other Direction

Interference does not always inflate. When the treatment creates something others can use, a shared recommendation model that improves with more treated users, a social feature whose value grows with adoption, a referral that reaches control users, the control arm benefits from the treatment and the user-level test understates the effect, sometimes to zero. The diagnosis is the same in both directions: ask whether one user's treatment changes the world another user acts in, and if the answer is yes, the user-level difference is not the rollout effect, and its sign relative to the truth depends on whether the treatment consumes a shared resource or produces one.

## Recognising It Before Running the Test

The interference in the simulation is visible before any data are collected. The treatment raises demand; the supply is fixed; therefore treated demand can only be met from supply that control demand would have used. That sentence, or its equivalent for drivers, budget, impressions or capacity, is the test to apply at design time. A second check is available after a buyer-level test has run: if the treatment arm's gain is matched by a loss in the control arm relative to the pre-period, the sum is telling the truth that the difference hides.

## What to Do

1. **Name the shared resource** before the experiment. If treated and control units compete for it, a unit-level test measures the competition, not the effect.
2. **Randomise the unit that contains the constraint**: separate markets, regions or time periods, and analyse at that level.
3. **Use switchbacks when the market cannot be split**, with periods long enough for carry-over to clear, and compare periods rather than users.
4. **Check the total, not just the difference.** In a constrained market a real lift shows up in the sum of both arms; a pure transfer does not.
5. **Expect the bias in both directions**: a treatment that consumes a shared resource inflates the user-level estimate, one that produces a shared benefit deflates it.
6. **Report the design's assumption** with the result: "no interference between users" is a claim, and in a marketplace it is usually false.

## References

- Rubin, D. B. (1980). Randomization analysis of experimental data: the Fisher randomization test, comment. *Journal of the American Statistical Association*, 75(371), 591-593.
- Blake, T., & Coey, D. (2014). Why marketplace experimentation is harder than it seems: the role of test-control interference. *Proceedings of the Fifteenth ACM Conference on Economics and Computation*, 567-582.
- Bojinov, I., Simchi-Levi, D., & Zhao, J. (2023). Design and analysis of switchback experiments. *Management Science*, 69(7), 3759-3777.
- Johari, R., Li, H., Liskovich, I., & Weintraub, G. Y. (2022). Experimental design in two-sided platforms: an analysis of bias. *Management Science*, 68(10), 7069-7089.
- Holtz, D., Lobel, R., Liskovich, I., & Aral, S. (2020). Reducing interference bias in online marketplace pricing experiments. *arXiv preprint*, arXiv:2004.12489.
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with interference. *Journal of the American Statistical Association*, 103(482), 832-842.
