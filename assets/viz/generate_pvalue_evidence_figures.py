"""Reproduce the synthetic p-value examples; --dry-run writes no files."""

import argparse
import json
from math import erfc, exp, hypot, isfinite, log, sqrt
from statistics import NormalDist

NORMAL = NormalDist()


def two_sided_p(z):
    """Two-sided standard-normal tail probability, with a prespecified test."""
    if not isfinite(z):
        raise ValueError("z must be finite")
    return erfc(abs(z) / sqrt(2))


def rejection_probability(effect=2.0, alpha=0.05):
    """P(|Z| >= c) for Z ~ N(effect, 1); effect is measured in SE units."""
    if not isfinite(effect) or not 0 < alpha < 1:
        raise ValueError("Use a finite effect and 0 < alpha < 1")
    cutoff = NORMAL.inv_cdf(1 - alpha / 2)
    return (erfc((cutoff - effect) / sqrt(2))
            + erfc((cutoff + effect) / sqrt(2))) / 2


def selected_studies(prior_signal=0.1, alpha=0.05, effect=2.0, studies=10000):
    """Expected counts and composition when only rejected tests are retained."""
    if not 0 <= prior_signal <= 1 or not isfinite(studies) or studies <= 0:
        raise ValueError("Use a probability and a positive finite study count")
    power = rejection_probability(effect, alpha)
    signal_rejections = studies * prior_signal * power
    null_rejections = studies * (1 - prior_signal) * alpha
    selected_signal_probability = signal_rejections / (signal_rejections + null_rejections)
    return {"alpha": alpha, "power": power,
            "signal_rejections": signal_rejections, "null_rejections": null_rejections,
            "signal_nonrejections": studies * prior_signal * (1 - power),
            "null_nonrejections": studies * (1 - prior_signal) * (1 - alpha),
            "signal_given_rejection": selected_signal_probability,
            "repeat_rejection_given_first_rejection":
                selected_signal_probability * power + (1 - selected_signal_probability) * alpha}


def posterior_signal(z, prior_signal=0.1, effect=2.0):
    """Posterior for N(effect,1) versus N(0,1), given the signed observation."""
    if not all(isfinite(v) for v in (z, prior_signal, effect)) or not 0 <= prior_signal <= 1:
        raise ValueError("Use finite parameters and a probability")
    if prior_signal in (0, 1):
        return prior_signal
    log_odds = log(prior_signal) - log(1 - prior_signal) + effect * z - effect ** 2 / 2
    if log_odds >= 0:
        return 1 / (1 + exp(-log_odds))
    odds = exp(log_odds)
    return odds / (1 + odds)


def study_summary(estimate, se):
    """Normal-model p-value and confidence interval with known standard error."""
    if not isfinite(estimate) or not isfinite(se) or se <= 0:
        raise ValueError("Use a finite estimate and positive finite standard error")
    half_width = NORMAL.inv_cdf(.975) * se
    return {"estimate": estimate, "se": se, "p": two_sided_p(estimate / se),
            "ci95": [estimate - half_width, estimate + half_width]}


def examples():
    return {"z_2_p": two_sided_p(2), "z_2_likelihood_ratio": exp(2),
            "negative_z_2_likelihood_ratio": exp(-6),
            "thresholds": [selected_studies(alpha=a) for a in (.05, .01, .001)],
            "prior_sensitivity": [{"prior": p, "given_z_2": posterior_signal(2, p),
                                   "given_rejection": selected_studies(p)["signal_given_rejection"]}
                                  for p in (.01, .1, .5)],
            "study_a": study_summary(.20, .10), "study_b": study_summary(.19, .11),
            "study_difference": study_summary(.20 - .19, hypot(.10, .11)),
            "same_p_small_effect": study_summary(.002, .001),
            "same_p_large_effect": study_summary(2, 1),
            "any_rejection_20_independent_null_tests": 1 - .95 ** 20,
            "bonferroni_20_independent_null_tests": 1 - (1 - .05 / 20) ** 20}


def draw_figures():
    import matplotlib.pyplot as plt
    from housestyle import INK_MUTED, PALETTE, save, use

    use()
    rows = [selected_studies(alpha=a) for a in (.05, .01, .001)]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.get_layout_engine().set(rect=(0, .075, 1, .925))
    false = [r["null_rejections"] for r in rows]
    true = [r["signal_rejections"] for r in rows]
    ax.bar(range(3), false, width=.55, color=PALETTE[1], label="Null-generated rejections")
    ax.bar(range(3), true, width=.55, bottom=false, color=PALETTE[0], label="Signal-generated rejections")
    for x, row in enumerate(rows):
        total = row["signal_rejections"] + row["null_rejections"]
        ax.text(x, total + 18, f'{total:.1f} retained\n{100*row["signal_given_rejection"]:.1f}% signal',
                ha="center", va="bottom", fontsize=9)
    ax.set(xticks=range(3), xticklabels=["0.05", "0.01", "0.001"], ylim=(0, 1190),
           xlabel="Prespecified significance threshold", ylabel="Expected rejections per 10,000 studies",
           title="The retained studies depend on both error rates and the starting mixture")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.text(.02, .005, "Synthetic model: 90% null, 10% signal; signal Z ~ N(2, 1). Counts are expectations.",
             color=INK_MUTED, fontsize=8)
    print(json.dumps(save(fig, "science_pvalue_selected_studies")))

    fig, ax = plt.subplots(figsize=(9, 4.7))
    fig.get_layout_engine().set(rect=(0, .085, 1, .915))
    for y, (name, estimate, se) in enumerate((("Study B", .19, .11), ("Study A", .20, .10))):
        row = study_summary(estimate, se)
        ax.errorbar(estimate, y, xerr=NORMAL.inv_cdf(.975)*se, fmt="o", capsize=5,
                    color=PALETTE[0], markersize=7)
        ax.annotate(f'p = {row["p"]:.4f}', (.42, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10)
    ax.axvline(0, color=INK_MUTED, ls="--", lw=1)
    ax.set(xlim=(-.08, .60), ylim=(-.7, 1.7), yticks=[0, 1], yticklabels=["Study B", "Study A"],
           xlabel="Estimated effect and 95% normal-model confidence interval",
           title="Nearly identical estimates can receive opposite significance labels")
    fig.text(.02, .005, "Invented independent studies of the same estimand; the difference is 0.01 (SE 0.149).",
             color=INK_MUTED, fontsize=8)
    print(json.dumps(save(fig, "science_pvalue_study_comparison")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(examples(), indent=2))
    if not args.dry_run:
        draw_figures()


if __name__ == "__main__":
    main()
