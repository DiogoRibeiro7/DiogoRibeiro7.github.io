"""Numbers and figures for "Parasites Are Diagnosed by Species, Not by Symptom Lists".

    python generate_parasite_testing_figures.py            # both figures and the numbers
    python generate_parasite_testing_figures.py --dry-run  # the numbers only, no files written

Everything here is arithmetic on published test accuracies: how much a second and
a third sample add, what probability of infection is left after negative results,
and how far a symptom checklist can move a prior. The model needs only the
standard library; matplotlib is imported to draw.
"""

import argparse
import json

# Cartwright CP. J Clin Microbiol 1999;37(8):2408-2411. Patients with three stool specimens examined;
# a case is a parasite found in any of the three, so these are upper bounds on true sensitivity.
CARTWRIGHT = {"cases": 373, "found by the first specimen": 283, "found by the first two": 343}
# Branda JA, et al. Clin Infect Dis 2006;42(7):972-978. Sensitivity of the first of at least three specimens,
# and the negative predictive values the authors give at four prevalences among those tested.
BRANDA = {"sensitivity": 0.72, "published NPV, %": {0.05: 98, 0.10: 97, 0.15: 95, 0.20: 93}}
# Wendt S, et al. Dtsch Arztebl Int 2019;116(13):213-219. Adhesive tape test for pinworm.
TAPE = {"one morning": 0.50, "three mornings": 0.90}
# Garcia LS, Shimizu RY. J Clin Microbiol 1997;35(6):1526-1529. Giardia antigen immunoassays;
# specificity was 100% on 50 negative specimens.
GARCIA = {"lowest sensitivity": 0.94, "negative specimens": 50, "false positives": 0}
PRIORS = (0.20, 0.05, 0.01, 0.001)


def detected_by(samples, sensitivity):
    """Share of infections found by at least one of `samples` independent samples."""
    return 1 - (1 - sensitivity) ** samples


def left_after_negatives(prior, negatives, sensitivity, specificity=1.0):
    """Probability of infection after `negatives` independent negative results, by Bayes' rule."""
    infected = prior * (1 - sensitivity) ** negatives
    return infected / (infected + (1 - prior) * specificity ** negatives)


def likelihood_ratio(sensitivity, false_positive_share):
    return sensitivity / false_positive_share


def after_positive(prior, ratio):
    odds = prior / (1 - prior) * ratio
    return odds / (1 + odds)


def specificity_lower_bound(negatives, confidence=0.95):
    """One-sided lower limit for a specificity observed as negatives out of negatives correct."""
    return (1 - confidence) ** (1 / negatives)


def summary():
    first = CARTWRIGHT["found by the first specimen"] / CARTWRIGHT["cases"]
    two = CARTWRIGHT["found by the first two"] / CARTWRIGHT["cases"]
    s = BRANDA["sensitivity"]
    bound = specificity_lower_bound(GARCIA["negative specimens"])
    antigen = likelihood_ratio(GARCIA["lowest sensitivity"], 1 - bound)
    return {
        "stool examination, first specimen, %": round(100 * first, 1),
        "stool examination, two specimens, observed %": round(100 * two, 1),
        "stool examination, two specimens, if independent %": round(100 * detected_by(2, first), 1),
        "tape test, three mornings, if independent %": round(100 * detected_by(3, TAPE["one morning"]), 1),
        "negative predictive value at the published prevalences, %": {
            str(p): round(100 * (1 - left_after_negatives(p, 1, s)), 1) for p in BRANDA["published NPV, %"]},
        "left after negatives, %": {
            str(p): [round(100 * left_after_negatives(p, n, s), 3) for n in (1, 2, 3)] for p in PRIORS},
        "left after three negatives at half the sensitivity, %": {
            str(p): round(100 * left_after_negatives(p, 3, 0.5), 2) for p in PRIORS},
        "checklist likelihood ratio": {str(q): round(likelihood_ratio(0.95, q), 2) for q in (0.8, 0.5, 0.2)},
        "after a positive checklist, prior 1%, %": {
            str(q): round(100 * after_positive(0.01, likelihood_ratio(0.95, q)), 1) for q in (0.8, 0.5, 0.2)},
        "antigen test specificity, lower 95% limit, %": round(100 * bound, 1),
        "antigen test likelihood ratio at that limit": round(antigen, 1),
        "after a positive antigen test, prior 1%, %": round(100 * after_positive(0.01, antigen), 1),
    }


def sampling_figure(plt, hs):
    first = CARTWRIGHT["found by the first specimen"] / CARTWRIGHT["cases"]
    samples = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = (("stool examination for ova and parasites", first, hs.PALETTE[0],
               {1: first, 2: CARTWRIGHT["found by the first two"] / CARTWRIGHT["cases"]}),
              ("adhesive tape test for pinworm", TAPE["one morning"], hs.PALETTE[1],
               {1: TAPE["one morning"], 3: TAPE["three mornings"]}))
    for label, sensitivity, colour, observed in series:
        ax.plot(samples, [100 * detected_by(n, sensitivity) for n in samples], color=colour, label=f"{label}: if samples were independent")
        ax.plot(list(observed), [100 * v for v in observed.values()], "o", color=colour, markersize=8,
                markeredgecolor=hs.SURFACE, markeredgewidth=2, label=f"{label}: reported")
        for n, value in observed.items():
            above = value > detected_by(n, sensitivity) + 1e-9           # keep the label off the line
            ax.annotate(f"{100 * value:.0f}%", xy=(n, 100 * value), xytext=(0, 9 if above else -17),
                        textcoords="offset points", ha="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xlim(0.8, 3.25)
    ax.set_xticks(samples)
    ax.set_xticklabels(["one sample", "two, on separate days", "three, on separate days"])
    ax.set_ylim(40, 104)
    ax.set_ylabel("infections detected, %")
    ax.set_title("A missed infection is intermittent shedding, and sampling again finds it")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    alt = (f"Line chart of the share of infections detected by one, two and three samples taken on separate days. For stool "
           f"examination one specimen found {100 * first:.0f}% and two found 92%, close to the {100 * detected_by(2, first):.0f}% "
           f"expected if samples were independent. For the pinworm tape test one morning finds about 50% and three mornings about "
           f"90%, close to the {100 * detected_by(3, 0.5):.0f}% expected under independence.")
    return fig, "parasite_repeat_sampling", alt


def negatives_figure(plt, hs):
    s = BRANDA["sensitivity"]
    priors = [10 ** (-3 + 2.5 * i / 60) for i in range(61)]          # 0.1% to about 32%
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([100 * p for p in priors], [100 * p for p in priors], color=hs.INK_MUTED, linewidth=1.2,
            linestyle=(0, (3, 3)), label="before any test")
    shades = (hs.SEQUENTIAL[2], hs.SEQUENTIAL[4], hs.SEQUENTIAL[7])
    for negatives, colour in zip((1, 2, 3), shades):
        ax.plot([100 * p for p in priors], [100 * left_after_negatives(p, negatives, s) for p in priors], color=colour,
                label=f"after {negatives} negative stool examination" + ("" if negatives == 1 else "s"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.1, 1, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%"])
    ax.set_xlabel("probability of infection before testing")
    ax.set_ylabel("probability of infection that remains")
    ax.set_title("Each negative result divides what is left by about 3.6")
    ax.legend(loc="upper left")
    left = {p: [100 * left_after_negatives(p, n, s) for n in (1, 3)] for p in (0.05, 0.01)}
    alt = (f"Line chart on logarithmic axes of the probability of infection remaining after one, two and three negative stool "
           f"examinations, against the probability before testing, for a single-specimen sensitivity of {100 * s:.0f}%. The "
           f"lines are parallel and each lies below the last. A 5% prior falls to {left[0.05][0]:.1f}% after one negative result "
           f"and {left[0.05][1]:.2f}% after three; a 1% prior falls to {left[0.01][0]:.2f}% and {left[0.01][1]:.3f}%.")
    return fig, "parasite_negative_results", alt


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
    for build in (sampling_figure, negatives_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:30} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()
