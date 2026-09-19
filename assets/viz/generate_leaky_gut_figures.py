"""Numbers and figures for "Leaky Gut: Real Physiology, Weak Diagnosis".

    python generate_leaky_gut_figures.py            # both figures and the numbers
    python generate_leaky_gut_figures.py --dry-run  # the numbers only, no files written

Two calculations. The first asks what a positive result on a surrogate test says
about the quantity it stands for, given the correlation between them. The second
turns a published hazard ratio into absolute risks. The model needs only the
standard library; matplotlib is imported to draw.
"""

import argparse
import json
from math import atanh, sqrt, tanh
from statistics import NormalDist

STANDARD = NormalDist()
# Power N, et al. Front Physiol 2021;12:645303. Serum marker from a commercial kit against the
# lactulose-mannitol ratio in 39 healthy first-degree relatives of patients with Crohn's disease.
POWER = {"r squared": 0.004, "participants": 39}
# Turpin W, et al. Gastroenterology 2020;159:2092-2100. First-degree relatives followed for a median of 7.8 years.
TURPIN = {"relatives": 1420, "developed Crohn's disease": 50, "hazard ratio": 3.03, "interval": (1.64, 5.63)}
# Zhou Q, et al. Gut 2019;68:996-1002. Responders among those who completed eight weeks.
ZHOU = {"glutamine": (43, 54), "placebo": (3, 52)}
TOP = 0.20        # call the highest fifth of results "positive" and the highest fifth of true values "leaky"


def share_truly_high(correlation, top=TOP, step=0.001):
    """P(true value in its top share | test result in its top share) for a bivariate normal pair."""
    if correlation >= 1:
        return 1.0
    cut = STANDARD.inv_cdf(1 - top)
    spread = sqrt(1 - correlation ** 2)
    total, x = 0.0, cut
    while x < 8.0:
        middle = x + step / 2
        total += STANDARD.pdf(middle) * (1 - STANDARD.cdf((cut - correlation * middle) / spread)) * step
        x += step
    return total / top


def correlation_interval(r_squared, participants, z=1.959964):
    """Correlation and its 95% interval by Fisher's transformation; the sign is taken as positive."""
    r = sqrt(r_squared)
    centre, half = atanh(r), z / sqrt(participants - 3)
    return r, tanh(centre - half), tanh(centre + half)


def risks_by_test(share_abnormal, overall, ratio):
    """Risk with a normal and with an abnormal test, given the overall risk and the ratio between them."""
    normal = overall / (1 + (ratio - 1) * share_abnormal)
    return normal, ratio * normal


def summary():
    r, low, high = correlation_interval(POWER["r squared"], POWER["participants"])
    overall = TURPIN["developed Crohn's disease"] / TURPIN["relatives"]
    (g_yes, g_all), (p_yes, p_all) = ZHOU["glutamine"], ZHOU["placebo"]
    return {
        "correlation and 95% interval": [round(r, 3), round(low, 2), round(high, 2)],
        "variance explained, %": round(100 * POWER["r squared"], 1),
        "truly high among test positives, %": {
            "at the observed correlation": round(100 * share_truly_high(r), 1),
            "at the top of its interval": round(100 * share_truly_high(high), 1),
            "at 0.5": round(100 * share_truly_high(0.5), 1),
            "at 0.9": round(100 * share_truly_high(0.9), 1),
            "by chance": round(100 * TOP, 1),
        },
        "relatives who developed Crohn's disease, %": round(100 * overall, 1),
        "risk with a normal and an abnormal test, %": {
            str(q): [round(100 * v, 1) for v in risks_by_test(q, overall, TURPIN["hazard ratio"])] for q in (0.1, 0.2, 0.3)},
        "glutamine trial responders, %": [round(100 * g_yes / g_all, 1), round(100 * p_yes / p_all, 1)],
    }


def surrogate_figure(plt, hs):
    r, low, high = correlation_interval(POWER["r squared"], POWER["participants"])
    grid = [i / 100 for i in range(0, 100)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(0, high, color=hs.SEQUENTIAL[0], alpha=0.55, linewidth=0, label="95% interval for the commercial kit (39 people)")
    ax.axhline(100 * TOP, color=hs.INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)), label="picking people at random")
    ax.plot(grid, [100 * share_truly_high(g) for g in grid], color=hs.PALETTE[0], label="a test with this correlation")
    here = 100 * share_truly_high(r)
    ax.plot([r], [here], "o", color=hs.PALETTE[1], markersize=8, markeredgecolor=hs.SURFACE, markeredgewidth=2)
    ax.annotate(f"observed correlation {r:.2f}: {here:.0f}%", xy=(r, here), xytext=(34, -46), textcoords="offset points",
                fontsize=9, color=hs.INK_SECONDARY, arrowprops={"arrowstyle": "-", "color": hs.INK_MUTED, "linewidth": 0.8})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("correlation between the test and measured permeability")
    ax.set_ylabel("test positives whose permeability is truly high, %")
    ax.set_title("What a positive result is worth depends on the correlation")
    ax.legend(loc="upper left", fontsize=8.5)
    alt = (f"Line chart of the share of people with a positive test, defined as the highest fifth of results, whose true "
           f"permeability is also in the highest fifth, against the correlation between test and permeability. The share is "
           f"20% at zero correlation, {100 * share_truly_high(0.5):.0f}% at 0.5 and {100 * share_truly_high(0.9):.0f}% at 0.9. "
           f"The commercial kit's observed correlation of {r:.2f} gives {here:.0f}%, and the top of its 95% interval, "
           f"{high:.2f}, gives {100 * share_truly_high(high):.0f}%.")
    return fig, "leaky_gut_surrogate_test", alt


def relatives_figure(plt, hs):
    overall = TURPIN["developed Crohn's disease"] / TURPIN["relatives"]
    shares = (0.1, 0.2, 0.3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.36
    for k, (label, colour) in enumerate((("normal permeability test", hs.PALETTE[0]), ("abnormal permeability test", hs.PALETTE[1]))):
        values = [100 * risks_by_test(q, overall, TURPIN["hazard ratio"])[k] for q in shares]
        xs = [i + (k - 0.5) * width for i in range(len(shares))]
        ax.bar(xs, values, width=width * 0.94, color=colour, label=label)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.25, f"{v:.1f}%", ha="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xticks(range(len(shares)))
    ax.set_xticklabels([f"if {100 * q:.0f}% of relatives test abnormal" for q in shares])
    ax.set_ylim(0, 12)
    ax.set_ylabel("developed Crohn's disease in about eight years, %")
    ax.set_title("A threefold risk is still a small risk: most abnormal tests lead to nothing")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    example = risks_by_test(0.2, overall, TURPIN["hazard ratio"])
    alt = (f"Bar chart of the risk of developing Crohn's disease over about eight years among first-degree relatives of patients, "
           f"split by a normal or abnormal permeability test, for three assumptions about how many relatives test abnormal. The "
           f"overall risk is {100 * overall:.1f}% and the hazard ratio 3.03. If a fifth test abnormal, the risk is "
           f"{100 * example[0]:.1f}% after a normal test and {100 * example[1]:.1f}% after an abnormal one.")
    return fig, "leaky_gut_relatives_risk", alt


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
    for build in (surrogate_figure, relatives_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:28} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
