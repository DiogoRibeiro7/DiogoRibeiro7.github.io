"""Numbers and figures for "When Results Become Rhetoric".

    python generate_results_rhetoric_figures.py            # both figures and the numbers
    python generate_results_rhetoric_figures.py --dry-run  # the numbers only, no files written

Two things are computed. The first is arithmetic on the retention and weight-loss
figures that Finley et al. (2007) published for 60,164 clients of a commercial
programme. The second is exact: the expected mean of the best k results among n
clients whose individual results are normal, which is what a wall of testimonials
shows. The model needs only the standard library; matplotlib is imported to draw.
"""

import argparse
import json
from math import exp, lgamma, log, log1p
from random import Random
from statistics import NormalDist

# Finley CE, Barlow CE, Greenway FL, Rock CL, Rolls BJ, Blair SN. Retention rates and weight loss in a
# commercial weight loss program. Int J Obes 2007;31(2):292-298. Figures as given in the abstract.
FINLEY = {
    "enrolled": 60_164,
    "retained, % of enrolled": {0: 100.0, 4: 73.0, 13: 42.0, 26: 22.0, 52: 6.6},
    # mean and SD of weight lost, % of initial body weight, among clients still attending at that week
    "cohort loss, %": {13: (8.3, 3.3), 26: (12.6, 5.1), 52: (15.6, 7.5)},
    "loss of those who left in weeks 1-4, %": (1.1, 1.6),
}
# Krogsboll LT, Hrobjartsson A, Gotzsche PC. Spontaneous improvement in randomised clinical trials.
# BMC Med Res Methodol 2009;9:1. Change from baseline in standard deviations, 37 three-armed trials.
KROGSBOLL = {"no treatment": 0.24, "placebo": 0.44, "active treatment": 1.01}
STANDARD = NormalDist()
WALL = 20
SCALE = FINLEY["cohort loss, %"][26][1]      # an SD of 5.1% of body weight, to give the unit a size


def finley_summary():
    retained = FINLEY["retained, % of enrolled"]
    enrolled = FINLEY["enrolled"]
    return {
        "left within four weeks, %": round(100 - retained[4], 1),
        "left before a year, %": round(100 - retained[52], 1),
        "clients still attending at a year": round(enrolled * retained[52] / 100),
        "clients who left within four weeks": round(enrolled * (100 - retained[4]) / 100),
        "enrolled per client still attending at a year": round(100 / retained[52], 1),
    }


def log_binomial_pmf(j, n, p):
    return lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1) + j * log(p) + (n - j) * log1p(-p)


def wall_mean(clients, wall=WALL, effect=0.0, step=0.002):
    """Expected mean of the best `wall` results among `clients`, in standard deviations.

    Individual results are Normal(effect, 1), larger being better. A client with
    result x is on the wall when at most wall - 1 of the other clients did better,
    so the expected total on the wall is
    clients * integral of x * pdf(x) * P(Binomial(clients - 1, P(X > x)) <= wall - 1) dx,
    which is evaluated on a grid. No approximation of the order statistics is used.
    """
    if wall < 1 or clients < wall:
        raise ValueError("the wall cannot hold more results than there are clients")
    if clients == wall:
        return effect
    total = 0.0
    x = -9.0
    while x < 9.0:
        tail = 1 - STANDARD.cdf(x) if x < 0 else STANDARD.cdf(-x)
        expected_better = (clients - 1) * tail
        if expected_better < wall + 40 * (wall ** 0.5) + 40:        # beyond this the weight is below 1e-30
            if tail <= 0.0:
                on_wall = 1.0
            elif tail >= 1.0:
                on_wall = 0.0          # everybody else did better, and there are more of them than places
            else:
                on_wall = sum(exp(log_binomial_pmf(j, clients - 1, tail)) for j in range(min(wall, clients)))
            total += x * STANDARD.pdf(x) * min(on_wall, 1.0) * step
        x += step
    return effect + clients * total / wall


def wall_mean_by_simulation(clients, wall=WALL, effect=0.0, repeats=400, seed=20260919):
    rng = Random(seed)
    means = []
    for _ in range(repeats):
        results = sorted(rng.gauss(effect, 1.0) for _ in range(clients))
        means.append(sum(results[-wall:]) / wall)
    return sum(means) / repeats


def clients_needed(target, effect=0.0, wall=WALL):
    """Smallest number of clients, to two significant figures, whose wall reaches `target` standard deviations."""
    low, high = wall, 10 ** 8                     # wall_mean rises with the number of clients, so bisect
    while high - low > max(1, low // 1000):
        middle = (low + high) // 2
        low, high = (low, middle) if wall_mean(middle, wall, effect) >= target else (middle, high)
    digits = len(str(high)) - 2
    return round(high, -digits) if digits > 0 else high


def summary():
    sizes = (200, 1_000, 10_000, 100_000, 1_000_000)
    walls = {n: round(wall_mean(n), 2) for n in sizes}
    effective_small = round(wall_mean(1_000, effect=1.0), 2)
    return {
        "finley": finley_summary(),
        "wall of 20, no effect, by number of clients": walls,
        "wall of 20, effect of 1 SD, 1,000 clients": effective_small,
        "wall of 20, effect of 0.5 SD, 1,000 clients": round(wall_mean(1_000, effect=0.5), 2),
        "clients at which a programme with no effect matches it": clients_needed(effective_small),
        "share of the wall that is selection, effect of 1 SD, 1,000 clients": round(walls[1_000] / effective_small, 2),
        "wall of 20, no effect, % of body weight at the illustrative SD": {n: round(walls[n] * SCALE, 1) for n in (1_000, 100_000)},
        "share of the change in treated patients that the treatment explains": round(
            (KROGSBOLL["active treatment"] - KROGSBOLL["placebo"]) / KROGSBOLL["active treatment"], 2),
        "before-and-after change over the treatment's own effect": round(
            KROGSBOLL["active treatment"] / (KROGSBOLL["active treatment"] - KROGSBOLL["placebo"]), 1),
    }


def retention_figure(plt, hs):
    retained = FINLEY["retained, % of enrolled"]
    losses = FINLEY["cohort loss, %"]
    weeks = list(retained)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(range(len(weeks)), [retained[w] for w in weeks], color=hs.PALETTE[0], width=0.62)
    for position, week in enumerate(weeks):
        label = f"{retained[week]:g}%"
        if week in losses:
            label += f"\nlost {losses[week][0]:g}%"
        ax.text(position, retained[week] + 2, label, ha="center", va="bottom", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(["enrolled"] + [f"week {w}" for w in weeks[1:]])
    ax.set_ylim(0, 116)
    ax.set_xlabel("weight lost: mean among clients still attending, % of initial body weight")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("clients still attending, % of those enrolled")
    ax.set_title("The better the result, the fewer clients it describes")
    ax.grid(axis="x", visible=False)
    alt = (f"Bar chart of retention among {FINLEY['enrolled']:,} clients of a commercial weight-loss programme: "
           f"{retained[4]:g}% still attending at week 4, {retained[13]:g}% at week 13, {retained[26]:g}% at week 26 and "
           f"{retained[52]:g}% at week 52. Those still attending had lost {losses[13][0]}%, {losses[26][0]}% and "
           f"{losses[52][0]}% of their body weight at weeks 13, 26 and 52. Data from Finley and colleagues, 2007.")
    return fig, "results_rhetoric_retention", alt


def wall_figure(plt, hs):
    sizes = [round(100 * 10 ** (i / 6)) for i in range(25)]          # 100 to 1,000,000, six points a decade
    base = [wall_mean(n) for n in sizes]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = ((0.0, "a programme with no effect", hs.PALETTE[0]), (0.5, "a moderate effect, 0.5 SD", hs.PALETTE[1]),
              (1.0, "a large effect, 1 SD", hs.PALETTE[2]))
    for effect, label, colour in series:
        ax.plot(sizes, [effect + b for b in base], color=colour, label=label)
    small = wall_mean(1_000, effect=1.0)
    match = clients_needed(round(small, 2))
    ax.plot([1_000, match], [small, small], color=hs.INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.plot([1_000], [small], "o", color=hs.PALETTE[2], markersize=7, markeredgecolor=hs.SURFACE, markeredgewidth=2)
    ax.plot([match], [small], "o", color=hs.PALETTE[0], markersize=7, markeredgecolor=hs.SURFACE, markeredgewidth=2)
    ax.annotate(f"same wall: a large effect with 1,000 clients,\nno effect with {match:,}", xy=(match, small),
                xytext=(0, -78), textcoords="offset points", ha="center", fontsize=9, color=hs.INK_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": hs.INK_MUTED, "linewidth": 0.8, "shrinkB": 6})
    ax.set_xscale("log")
    ax.set_xlabel("number of clients the best twenty are chosen from")
    ax.set_ylabel("mean result on the wall, standard deviations")
    ax.set_ylim(0, 5.6)
    ax.set_title("A wall of the best twenty measures the size of the business")
    ax.legend(loc="upper left")
    alt = ("Line chart of the expected mean result among the best twenty clients, in standard deviations of individual "
           "results, against the number of clients on a logarithmic axis from one hundred to one million. Three parallel "
           f"rising lines: a programme with no effect goes from {base[0]:.1f} at 100 clients to {base[6]:.1f} at 1,000 and "
           f"{base[-1]:.1f} at a million; effects of half and of one standard deviation lie that much higher. A dashed line "
           f"marks that a large effect with 1,000 clients and no effect with {match:,} clients produce the same wall, {small:.1f}.")
    return fig, "results_rhetoric_testimonial_wall", alt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the numbers and write nothing")
    args = parser.parse_args()
    print(json.dumps(summary(), indent=1))
    if args.dry_run:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import housestyle as hs
    hs.use()
    for build in (retention_figure, wall_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:36} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
