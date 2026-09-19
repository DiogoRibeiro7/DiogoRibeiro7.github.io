"""Numbers and figures for "A Stool Sample Is Not a Diagnosis".

    python generate_microbiome_testing_figures.py            # both figures and the numbers
    python generate_microbiome_testing_figures.py --dry-run  # the numbers only, no files written

Three calculations, all exact. Closure: when one taxon's absolute count changes,
every other taxon's share changes although nothing happened to it. Flags: how many
of the taxa on a report fall outside a "healthy range" in a healthy person.
Repeatability: given the intraclass correlation of a taxon's abundance within one
person, how often a value flagged in one sample is flagged again in the next.
The model needs only the standard library; matplotlib is imported to draw.
"""

import argparse
import json
from math import sqrt
from statistics import NormalDist

STANDARD = NormalDist()
# Vandeputte D, et al. Nat Commun 2021;12:6740. 20 women sampled daily for six weeks. Share of genera whose
# abundance varied more within a person than between people (intraclass correlation below 0.5).
VANDEPUTTE = {"absolute abundance, %": 78, "relative abundance, %": 36}
# Servetas SL, et al. Commun Biol 2026;9:269. One homogenised stool sample, three kits to each of seven companies.
SERVETAS = {"companies": 7, "unique taxa": 1208, "genera found by all": 3, "fewest genera": 34, "most genera": 906}
# Berry SE, et al. Nat Med 2020;26:964-973. Variance in the glucose response after identical meals, % explained.
BERRY = {"meal macronutrients": 15.4, "gut microbiome": 6.0}


def apparent_change(share, fold):
    """Relative change in the share of every other taxon when one taxon with `share` multiplies by `fold`."""
    return 1 / (1 + share * (fold - 1)) - 1


def any_flag(taxa, coverage=0.95):
    return 1 - coverage ** taxa


def flagged_again(reliability, top=0.05, step=0.001):
    """P(second sample in its top share | first sample in its top share), bivariate normal, given the ICC."""
    if reliability >= 1:
        return 1.0
    cut = STANDARD.inv_cdf(1 - top)
    spread = sqrt(1 - reliability ** 2)
    total, x = 0.0, cut
    while x < 8.0:
        middle = x + step / 2
        total += STANDARD.pdf(middle) * (1 - STANDARD.cdf((cut - reliability * middle) / spread)) * step
        x += step
    return total / top


def summary():
    return {
        "other shares when a 30% taxon doubles, %": round(100 * apparent_change(0.30, 2)),
        "other shares when a 30% taxon rises tenfold, %": round(100 * apparent_change(0.30, 10)),
        "other shares when a 5% taxon rises tenfold, %": round(100 * apparent_change(0.05, 10)),
        "expected flags among 100 taxa with 95% ranges": round(0.05 * 100),
        "chance of at least one flag among 100 taxa, %": round(100 * any_flag(100), 1),
        "expected flags among 100 taxa with interquartile ranges": round(0.5 * 100),
        "flagged again, %": {str(icc): round(100 * flagged_again(icc), 1) for icc in (0.3, 0.5, 0.7, 0.9)},
        "microbiome over macronutrients, glucose response": round(BERRY["gut microbiome"] / BERRY["meal macronutrients"], 2),
    }


def closure_figure(plt, hs):
    folds = [10 ** (i / 50) for i in range(51)]           # one to tenfold
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for share, colour in zip((0.05, 0.15, 0.30), (hs.SEQUENTIAL[2], hs.SEQUENTIAL[4], hs.SEQUENTIAL[7])):
        ax.plot(folds, [100 * apparent_change(share, k) for k in folds], color=colour,
                label=f"the taxon that grew held {100 * share:.0f}% of the community")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["no change", "doubles", "fivefold", "tenfold"])
    ax.set_ylim(-80, 5)
    ax.set_xlabel("what happened to one taxon's absolute count")
    ax.set_ylabel("apparent change in every other taxon's share, %")
    ax.set_title("A percentage falls when its neighbour grows")
    ax.legend(loc="lower left")
    alt = ("Line chart of the apparent change in the relative abundance of every other taxon when a single taxon's absolute count "
           "rises between one and tenfold and nothing else changes. If the taxon that grew held 30% of the community, the other "
           f"shares fall {-100 * apparent_change(0.30, 2):.0f}% when it doubles and {-100 * apparent_change(0.30, 10):.0f}% when it "
           f"rises tenfold; if it held 5%, they fall {-100 * apparent_change(0.05, 10):.0f}% for a tenfold rise.")
    return fig, "microbiome_compositional_closure", alt


def repeat_figure(plt, hs):
    grid = [i / 100 for i in range(0, 100)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(5, color=hs.INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)), label="no relation between the two samples")
    ax.plot(grid, [100 * flagged_again(g) for g in grid], color=hs.PALETTE[0], label="a taxon with this repeatability")
    half = 100 * flagged_again(0.5)
    ax.plot([0.5], [half], "o", color=hs.PALETTE[1], markersize=8, markeredgecolor=hs.SURFACE, markeredgewidth=2)
    ax.annotate(f"intraclass correlation 0.5: {half:.0f}%", xy=(0.5, half), xytext=(-12, 26), textcoords="offset points",
                ha="right", fontsize=9, color=hs.INK_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": hs.INK_MUTED, "linewidth": 0.8})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("repeatability of the taxon within one person (intraclass correlation)")
    ax.set_ylabel("flagged results that are flagged again, %")
    ax.set_title("A flag that would not survive a second sample")
    ax.legend(loc="upper left")
    alt = ("Line chart of how often a taxon flagged as high, meaning in the top 5% of a healthy range, is flagged again in a second "
           "sample from the same person, against the taxon's intraclass correlation. It is 5% with no repeatability, "
           f"{100 * flagged_again(0.3):.0f}% at 0.3, {half:.0f}% at 0.5, {100 * flagged_again(0.7):.0f}% at 0.7 and "
           f"{100 * flagged_again(0.9):.0f}% at 0.9.")
    return fig, "microbiome_flag_repeatability", alt


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
    for build in (closure_figure, repeat_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:34} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
