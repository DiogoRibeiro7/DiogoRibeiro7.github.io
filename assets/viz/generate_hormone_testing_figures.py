"""Numbers and figures for "Hormones Are Not a Single Balance".

    python generate_hormone_testing_figures.py            # both figures and the numbers
    python generate_hormone_testing_figures.py --dry-run  # the numbers only, no files written

Two calculations. The first is the chance that a healthy person has at least one
result outside a 95% reference interval when several analytes are measured, with
and without correlation between them. The second is the reference change value:
how far two results from one person must differ before biological variation alone
does not explain it. The model needs only the standard library; matplotlib is
imported to draw.
"""

import argparse
import json
from math import sqrt
from random import Random
from statistics import NormalDist

# Within-subject biological variation, CV in %, median of the studies in the EFLM Biological Variation
# Database (https://biologicalvariation.eu, read 2026-09-19); serum or plasma, healthy adults.
WITHIN_SUBJECT_CV = {
    "free T4": 4.8, "SHBG": 8.3, "FSH": 9.5, "testosterone": 14.3, "estradiol": 15.0, "cortisol": 16.2,
    "TSH": 17.8, "progesterone": 18.5, "DHEAS": 20.0, "LH": 25.2, "insulin": 25.4, "prolactin": 45.0,
    "free testosterone": 55.0,
}
# Casals G, et al. Clin Biochem 2011;44:665-668. Late-night salivary cortisol: analytical CV, within-subject
# CV and the reference change value the authors report, all in %.
CASALS = {"analytical": 15.4, "within subject": 34.1, "published RCV": 104}
# Danese E, et al. Clin Chem Lab Med 2024;62:2287-2293. Salivary cortisol, reference change value by time of day, %.
DANESE_RCV = (96, 245)
# Naugler C, Ma I. Can Fam Physician 2018;64:202-203. Mean abnormal result rate of 1,340 family physicians, %.
NAUGLER = {"observed abnormal": 8.6, "expected in the healthy": 5.0}
# Stanczyk FZ, et al. Menopause 2019;26:966-971. Measured content against a label of 0.5 mg and 100 mg.
STANCZYK = {"estradiol capsules, mg": (0.365, 0.551, 0.5), "progesterone capsules, mg": (90.8, 135.0, 100.0)}
# Anckaert E, et al. Pract Lab Med 2021;25:e00211. Median serum progesterone, nmol/L.
ANCKAERT_PROGESTERONE = {"follicular": 0.212, "luteal": 28.8}
Z = NormalDist().inv_cdf(0.975)


def any_flag(analytes, coverage=0.95):
    """Chance that at least one of `analytes` independent results falls outside its reference interval."""
    return 1 - coverage ** analytes


def any_flag_correlated(analytes, correlation, repeats=40_000, seed=20260919):
    """The same chance when every pair of analytes has the given correlation, by simulation."""
    rng = Random(seed)
    shared, own = sqrt(correlation), sqrt(1 - correlation)
    hits = 0
    for _ in range(repeats):
        common = rng.gauss(0, 1)
        if any(abs(shared * common + own * rng.gauss(0, 1)) > Z for _ in range(analytes)):
            hits += 1
    return hits / repeats


def reference_change_value(within_subject_cv, analytical_cv=0.0):
    """Smallest difference between two results, in %, that exceeds what variation alone gives 95% of the time."""
    return Z * sqrt(2) * sqrt(within_subject_cv ** 2 + analytical_cv ** 2)


def summary():
    observed, expected = NAUGLER["observed abnormal"], NAUGLER["expected in the healthy"]
    low, high, label = STANCZYK["estradiol capsules, mg"]
    p_low, p_high, p_label = STANCZYK["progesterone capsules, mg"]
    return {
        "chance of at least one flag, %": {n: round(100 * any_flag(n), 1) for n in (1, 5, 12, 20, 40)},
        "expected flags in a panel of 12": round(0.05 * 12, 1),
        "chance of at least one flag, 12 analytes, correlation 0.5, %": round(100 * any_flag_correlated(12, 0.5), 1),
        "share of abnormal results that are false positives, %": round(100 * expected / observed),
        "minimum reference change value, %": {k: round(reference_change_value(v)) for k, v in WITHIN_SUBJECT_CV.items()},
        "salivary cortisol reference change value from Casals's components, %": round(
            reference_change_value(CASALS["within subject"], CASALS["analytical"]), 1),
        "estradiol capsule content against label, %": [round(100 * (low / label - 1)), round(100 * (high / label - 1))],
        "progesterone capsule content against label, %": [round(100 * (p_low / p_label - 1)), round(100 * (p_high / p_label - 1))],
        "progesterone, luteal over follicular": round(ANCKAERT_PROGESTERONE["luteal"] / ANCKAERT_PROGESTERONE["follicular"]),
    }


def panel_figure(plt, hs):
    sizes = list(range(1, 41))
    correlated_sizes = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(sizes, [100 * any_flag(n) for n in sizes], color=hs.PALETTE[0], label="independent analytes")
    ax.plot(correlated_sizes, [100 * any_flag_correlated(n, 0.5, repeats=20_000) for n in correlated_sizes],
            color=hs.PALETTE[1], label="every pair correlated at 0.5")
    twelve = 100 * any_flag(12)
    ax.plot([12], [twelve], "o", color=hs.PALETTE[0], markersize=8, markeredgecolor=hs.SURFACE, markeredgewidth=2)
    ax.annotate(f"a dozen hormones: {twelve:.0f}%", xy=(12, twelve), xytext=(-10, 10), textcoords="offset points",
                ha="right", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 100)
    ax.set_xlabel("analytes measured in one healthy person")
    ax.set_ylabel("chance of at least one result outside its range, %")
    ax.set_title("The larger the panel, the more certain it is to find something")
    ax.legend(loc="lower right")
    alt = (f"Line chart of the chance that a healthy person has at least one result outside a 95% reference interval, "
           f"against the number of analytes measured. With independent analytes it rises from 5% for one test to "
           f"{100 * any_flag(12):.0f}% for twelve, {100 * any_flag(20):.0f}% for twenty and {100 * any_flag(40):.0f}% for forty. "
           f"With every pair of analytes correlated at 0.5 the curve is lower but still reaches about "
           f"{100 * any_flag_correlated(12, 0.5):.0f}% at twelve.")
    return fig, "hormone_panel_false_flags", alt


def change_figure(plt, hs):
    values = {k: reference_change_value(v) for k, v in WITHIN_SUBJECT_CV.items()}
    names = sorted(values, key=values.get)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    positions = range(len(names))
    ax.barh(positions, [values[n] for n in names], color=hs.PALETTE[0], height=0.62, label="serum, biological variation alone")
    for y, name in zip(positions, names):
        ax.text(values[name] + 3, y, f"{values[name]:.0f}%", va="center", fontsize=9, color=hs.INK_SECONDARY)
    top = len(names)
    ax.barh([top], [DANESE_RCV[1] - DANESE_RCV[0]], left=[DANESE_RCV[0]], color=hs.PALETTE[1], height=0.62,
            label="salivary cortisol, measured, by time of day")
    ax.text(DANESE_RCV[1] + 3, top, f"{DANESE_RCV[0]}% to {DANESE_RCV[1]}%", va="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_yticks(list(positions) + [top])
    ax.set_yticklabels(names + ["salivary cortisol"])
    ax.set_xlim(0, 300)
    ax.set_xlabel("difference between two results needed before it means a change, %")
    ax.set_title("Two hormone results have to differ by a lot to differ at all")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    alt = ("Horizontal bar chart of the reference change value, the smallest difference between two results from one person "
           "that biological variation alone would not produce 95% of the time. It is "
           + ", ".join(f"{values[n]:.0f}% for {n}" for n in names)
           + f". Measured values for salivary cortisol, which include assay variation, run from {DANESE_RCV[0]}% to {DANESE_RCV[1]}% "
           "depending on the time of day.")
    return fig, "hormone_reference_change_values", alt


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
    for build in (panel_figure, change_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:34} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
