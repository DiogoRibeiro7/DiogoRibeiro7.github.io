"""Reproduce synthetic polling examples; --dry-run prints without writing figures."""

import argparse
import json
from math import isfinite, sqrt


def finite_summary(yes_total, no_total, yes_seen, no_seen):
    """Exact finite-population bookkeeping, with population moments divided by N."""
    values = (yes_total, no_total, yes_seen, no_seen)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("Counts must be nonnegative integers")
    if (yes_total == 0 or no_total == 0 or yes_seen > yes_total
            or no_seen > no_total or yes_seen + no_seen == 0):
        raise ValueError("Need both population outcomes and a nonempty feasible sample")
    population, sample = yes_total + no_total, yes_seen + no_seen
    p, q, f = yes_total/population, yes_seen/sample, sample/population
    covariance = yes_seen/population - p*f
    rho = covariance/sqrt(p*(1-p)*f*(1-f)) if f < 1 else None
    return {"population": population, "sample": sample, "population_share": p,
            "respondent_share": q, "recorded_fraction": f,
            "yes_recording_rate": yes_seen/yes_total, "no_recording_rate": no_seen/no_total,
            "covariance": covariance, "data_defect_correlation": rho,
            "error": q-p, "population_sd": sqrt(p*(1-p)),
            "no_assumption_bounds": [f*q, f*q+1-f]}


def naive_interval(q, n):
    """An intentionally naive binomial normal interval, not a population guarantee."""
    if not isfinite(q) or not 0 <= q <= 1 or isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Use a proportion and a positive integer sample size")
    half_width = 1.96 * sqrt(q*(1-q)/n)
    return {"half_width": half_width, "lower": q-half_width, "upper": q+half_width}


def srs_standard_error(p, population, sample):
    """Design SE of a binary mean under simple random sampling without replacement."""
    if not 0 < p < 1 or not 1 <= sample <= population or population <= 1:
        raise ValueError("Use a nondegenerate proportion and valid population/sample sizes")
    return sqrt(p*(1-p)*(population-sample)/(sample*(population-1)))


def population_from_ratio(q, ratio):
    """Invert outcome-dependent recording with ratio = r_yes/r_no."""
    if not isfinite(q) or not 0 < q < 1 or not isfinite(ratio) or ratio <= 0:
        raise ValueError("Use an interior respondent share and a positive finite ratio")
    return q/(ratio*(1-q)+q)


def weighting_example(outcome_dependent=False):
    """Two population groups; recorded counts are stipulated, not random draws."""
    groups = [
        {"group": "A", "total": 40000, "yes": 36000, "seen_yes": 3600,
         "seen_no": 100 if outcome_dependent else 400},
        {"group": "B", "total": 60000, "yes": 24000, "seen_yes": 480,
         "seen_no": 180 if outcome_dependent else 720},
    ]
    population = sum(g["total"] for g in groups)
    sample = sum(g["seen_yes"] + g["seen_no"] for g in groups)
    for g in groups:
        g["recorded"] = g["seen_yes"] + g["seen_no"]
        g["weight"] = g["total"]/g["recorded"]
        g["population_share"] = g["yes"]/g["total"]
        g["respondent_share"] = g["seen_yes"]/g["recorded"]
    return {"groups": groups, "population": population, "sample": sample,
            "population_share": sum(g["yes"] for g in groups)/population,
            "unweighted_share": sum(g["seen_yes"] for g in groups)/sample,
            "weighted_share": sum(g["total"]*g["respondent_share"] for g in groups)/population}


def examples():
    main = finite_summary(12000000, 8000000, 960000, 160000)
    sizes = (70, 700, 7000, 70000, 700000, 1120000)
    size_examples = []
    for n in sizes:
        row = finite_summary(12000000, 8000000, 6*n//7, n//7)
        row["naive_interval"] = naive_interval(row["respondent_share"], n)
        size_examples.append(row)
    return {"main": main, "main_naive_interval": naive_interval(6/7, 1120000),
            "sample_size_examples": size_examples,
            "srs_1000_standard_error": srs_standard_error(.6, 20000000, 1000),
            "weighting_by_group_only": weighting_example(),
            "weighting_within_group_selection": weighting_example(True),
            "compatible_worlds": [finite_summary(yes, 20000000-yes, 960000, 160000)
                                  for yes in (10000000, 12000000, 15000000)],
            "sensitivity": {str(r): population_from_ratio(6/7, r) for r in (1, 2, 3, 4, 6, 8)}}


def draw_figures():
    import matplotlib.pyplot as plt
    from housestyle import INK_SECONDARY, PALETTE, save, use

    use()
    rows = examples()["sample_size_examples"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.get_layout_engine().set(rect=(0, .11, 1, .84))
    ax.errorbar([r["sample"] for r in rows], [100*r["respondent_share"] for r in rows],
                yerr=[100*r["naive_interval"]["half_width"] for r in rows], fmt="o-",
                color=PALETTE[0], capsize=5, label="Respondent share with naive binomial intervals")
    ax.axhline(60, color=PALETTE[1], ls="--", label="Population share: 60%")
    ax.annotate("85.714% among respondents", xy=(7000, 85.714), xytext=(7000, 91),
                ha="center", fontsize=10)
    ax.set(xscale="log", ylim=(53, 99), xlabel="Number of recorded responses (log scale)",
           ylabel="Support (%)", title="More responses narrow an interval around a selected population")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(.02, .008, "Synthetic population: 20 million people. Supporters are recorded at four times the rate of other people.\n"
             "Bars show q ± 1.96 sqrt[q(1−q)/n]; these are not valid uncertainty bounds for population support.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_poll_selection_precision")))

    rows = [weighting_example(), weighting_example(True)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.get_layout_engine().set(rect=(0, .12, 1, .83))
    for offset, key, color, label in ((-.19, "unweighted_share", PALETTE[0], "Unweighted"),
                                    (.19, "weighted_share", PALETTE[1], "Weighted to group totals")):
        xs, ys = [i+offset for i in range(2)], [100*r[key] for r in rows]
        ax.bar(xs, ys, width=.34, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y+1.1, f"{y:.2f}%", ha="center", fontsize=10)
    ax.axhline(60, color=INK_SECONDARY, ls="--", label="Population share: 60%", lw=1.3)
    ax.set(ylim=(0, 116), xticks=[0, 1], yticks=[0, 20, 40, 60, 80, 100],
           xticklabels=["Selection differs by group only", "Selection also differs by answer\nwithin each group"],
           ylabel="Estimated support (%)", title="Matching group totals does not guarantee matching opinions")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    fig.text(.02, .008, "Synthetic population: group A is 40% of people with 90% support; group B is 60% with 40% support.\n"
             "Both weighted samples reproduce the population's group proportions exactly.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_poll_selection_weighting")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(examples(), indent=2))
    if not args.dry_run:
        draw_figures()


if __name__ == "__main__":
    main()
