"""Numbers and figures for "Inflammation Is Not a Diagnosis".

    python generate_inflammation_marker_figures.py            # both figures and the numbers
    python generate_inflammation_marker_figures.py --dry-run  # the numbers only, no files written

The calculations concern C-reactive protein as a measurement: how far two results
from one person must differ before they mean a change, and how wide the range of
single results is around a person's usual level. A bivariate log-normal model of
repeat testing was tried and dropped: it predicted 51% reclassification where a
national survey observed 32%, and 41% where a cohort observed 69%, so the article
quotes the observations. The second figure collects published hazard ratios.
The model needs only the standard library; matplotlib is imported to draw.
"""

import argparse
import json
from math import exp, log, sqrt
from statistics import NormalDist

STANDARD = NormalDist()
Z = STANDARD.inv_cdf(0.975)
# EFLM Biological Variation Database (https://biologicalvariation.eu), C-reactive protein, six studies:
# within-subject CV, %. Macy EM, et al. Clin Chem 1997;43:52-58: analytical and within-subject CV and the
# critical difference the authors report, %.
EFLM_WITHIN = 34.7
MACY = {"analytical": 5.2, "within subject": 42.2, "published critical difference": 118}
# Bower JK, et al. Arch Intern Med 2012;172:1519-1521. NHANES, 541 people tested twice 18.9 days apart.
# Visser M, et al. JAMA 1999;282:2131-2135: 6.7% of US adults had CRP above 1 mg/dL.
BOWER = {"intraclass correlation": 0.77, "above 1 mg/dL who were below it on repeat, %": 32, "share above 1 mg/dL": 0.067}
DEGOMA_ICC = 0.62          # DeGoma EM, et al. Atherosclerosis 2012;224:274-279 (MESA, serial values in 255 people)
# Hazard or risk ratios with 95% intervals. The first row is a Mendelian randomisation estimate per 1 SD of
# genetically raised ln CRP; the others are randomised trials against placebo.
ESTIMATES = (
    ("CRP raised by genes (Mendelian randomisation)", 1.00, 0.90, 1.13, "no drug: tests whether CRP is a cause"),
    ("Methotrexate, CIRT, n = 4,786", 0.96, 0.79, 1.16, "did not lower CRP, IL-6 or IL-1β"),
    ("Canakinumab 150 mg, CANTOS, n = 10,061", 0.85, 0.74, 0.98, "lowered CRP 37 points more than placebo"),
    ("Colchicine, COLCOT, n = 4,745", 0.77, 0.61, 0.96, "after a recent myocardial infarction"),
    ("Colchicine, LoDoCo2, n = 5,522", 0.69, 0.57, 0.83, "chronic coronary disease"),
)
CANTOS = {"placebo": 4.50, "150 mg": 3.86, "fatal infection, canakinumab": 0.31, "fatal infection, placebo": 0.18}


def symmetric_change_value(within, analytical=0.0):
    return Z * sqrt(2) * sqrt(within ** 2 + analytical ** 2)


def lognormal_change_value(within, analytical=0.0):
    """Rise and fall, in %, that two results must exceed, for results that are log-normal within a person."""
    sigma = sqrt(log(1 + (within / 100) ** 2) + log(1 + (analytical / 100) ** 2))
    factor = exp(Z * sqrt(2) * sigma)
    return 100 * (factor - 1), 100 * (1 / factor - 1)


def single_result_interval(usual, within):
    """95% range of one result from a person whose usual (median) level is `usual`, log-normal within person."""
    sigma = sqrt(log(1 + (within / 100) ** 2))
    return usual * exp(-Z * sigma), usual * exp(Z * sigma)


def summary():
    rise, fall = lognormal_change_value(EFLM_WITHIN)
    prevented = CANTOS["placebo"] - CANTOS["150 mg"]
    infections = CANTOS["fatal infection, canakinumab"] - CANTOS["fatal infection, placebo"]
    return {
        "critical difference from Macy's components, %": round(symmetric_change_value(MACY["within subject"], MACY["analytical"])),
        "change needed, EFLM variation, log-normal, %": [round(rise), round(fall)],
        "single results from a usual level of 3 mg/L": [round(v, 1) for v in single_result_interval(3.0, EFLM_WITHIN)],
        "single results from a usual level of 1 mg/L": [round(v, 2) for v in single_result_interval(1.0, EFLM_WITHIN)],
        "CANTOS events prevented per 1,000 person-years": round(10 * prevented, 1),
        "CANTOS extra fatal infections per 1,000 person-years": round(10 * infections, 1),
        "CANTOS events prevented per extra fatal infection": round(prevented / infections, 1),
    }


def scale_figure(plt, hs):
    low, high = single_result_interval(3.0, EFLM_WITHIN)
    rows = (
        ("acute infection or injury", [(500, "above 500")], hs.PALETTE[7]),
        ("current smokers and never-smokers", [(1.35, "1.35 never"), (2.53, "2.53 smokers")], hs.PALETTE[1]),
        ("healthy blood donors", [(0.8, "median 0.8"), (3.0, "90th centile 3.0"), (10, "99th centile 10")], hs.PALETTE[0]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(low, high, color=hs.SEQUENTIAL[0], alpha=0.6, linewidth=0)
    ax.text(sqrt(low * high), 3.62, f"single results of one person whose\nusual level is 3 mg/L: {low:.1f} to {high:.1f}",
            ha="center", va="top", fontsize=8.5, color=hs.INK_SECONDARY)
    ax.vlines([3, 10], -0.7, 2.62, color=hs.INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))   # stop below the band's label
    for y, (label, points, colour) in enumerate(rows):
        ax.plot([v for v, _ in points], [y] * len(points), "o", color=colour, markersize=9, markeredgecolor=hs.SURFACE, markeredgewidth=2)
        for k, (value, text) in enumerate(points):
            ax.annotate(text, xy=(value, y), xytext=(0, 11 if k % 2 == 0 else -17), textcoords="offset points", ha="center",
                        fontsize=8.5, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 1500)
    ax.set_ylim(-0.7, 3.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xticks([0.1, 1, 3, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "3", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("C-reactive protein, mg/L (logarithmic scale)")
    ax.set_title("One marker, four orders of magnitude, and a wide band of noise")
    ax.grid(axis="y", visible=False)
    alt = ("Dot chart of C-reactive protein on a logarithmic axis from 0.05 to 1,500 mg/L. Healthy blood donors have a median of "
           "0.8, a 90th centile of 3.0 and a 99th centile of 10 mg/L; never-smokers average 1.35 and current smokers 2.53; acute "
           "infection or injury takes it above 500. A shaded band shows that single results from one person whose usual level is "
           f"3 mg/L range from {low:.1f} to {high:.1f} mg/L, across the cut-off of 3 used for cardiovascular risk.")
    return fig, "inflammation_crp_scale", alt


def trials_figure(plt, hs):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rows = list(reversed(ESTIMATES))
    for y, (label, ratio, low, high, note) in enumerate(rows):
        colour = hs.INK_MUTED if "Mendelian" in label else hs.PALETTE[0]
        ax.plot([low, high], [y, y], color=colour, linewidth=2)
        ax.plot([ratio], [y], "o", color=colour, markersize=8, markeredgecolor=hs.SURFACE, markeredgewidth=2)
        ax.text(1.27, y, f"{ratio:.2f} ({low:.2f} to {high:.2f})", va="center", fontsize=9, color=hs.INK_SECONDARY)
        ax.text(0.52, y - 0.34, note, va="center", fontsize=8, color=hs.INK_MUTED)
    ax.axvline(1.0, color=hs.BASELINE, linewidth=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1.75)
    ax.set_ylim(-0.75, len(rows) - 0.4)
    ax.set_xticks([0.5, 0.7, 1.0, 1.4])
    ax.set_xticklabels(["0.5", "0.7", "1.0", "1.4"])
    ax.minorticks_off()
    ax.set_xlabel("ratio of cardiovascular events (below 1 favours the intervention)")
    ax.set_title("Which anti-inflammatory strategies prevented heart attacks")
    ax.grid(axis="y", visible=False)
    alt = ("Forest plot of five estimates of the effect on cardiovascular events, with 95% intervals. "
           + " ".join(f"{label}: {ratio:.2f}, {low:.2f} to {high:.2f}." for label, ratio, low, high, _ in ESTIMATES))
    return fig, "inflammation_trials_forest", alt


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
    for build in (scale_figure, trials_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:30} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
