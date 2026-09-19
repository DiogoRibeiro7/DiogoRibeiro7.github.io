"""Reproduce the synthetic before/after examples; --dry-run writes no files."""

import argparse
import json
from math import erfc, exp, isfinite, pi, sqrt
from random import Random
from statistics import NormalDist


def selected_moments(cutoff=65.0, mu=50.0, between_sd=8.0,
                     within_sd=6.0, baseline_count=1, effect=0.0):
    """Exact moments after selection on an average of independent baseline errors.

    Follow-up is one new observation. effect is added to that observation;
    negative values are beneficial when a lower score is preferable.
    """
    if not all(isfinite(x) for x in (cutoff, mu, between_sd, within_sd, effect)):
        raise ValueError("Model parameters must be finite")
    if between_sd < 0 or within_sd < 0 or between_sd + within_sd == 0:
        raise ValueError("Use nonnegative standard deviations with positive total variance")
    if isinstance(baseline_count, bool) or not isinstance(baseline_count, int) or baseline_count < 1:
        raise ValueError("baseline_count must be a positive integer")
    between_var, within_var = between_sd ** 2, within_sd ** 2
    baseline_var = between_var + within_var / baseline_count
    baseline_sd = sqrt(baseline_var)
    slope = between_var / baseline_var
    a = (cutoff - mu) / baseline_sd
    selected_fraction = 0.5 * erfc(a / sqrt(2))
    if selected_fraction == 0:
        raise ValueError("Selection probability underflows at this cutoff")
    tail_ratio = exp(-a * a / 2) / sqrt(2 * pi) / selected_fraction
    baseline_mean = mu + baseline_sd * tail_ratio
    followup_mean = mu + slope * (baseline_mean - mu) + effect
    selected_baseline_var = max(0.0, baseline_var * (1 + a * tail_ratio - tail_ratio ** 2))
    residual_var = max(0.0, between_var + within_var - between_var ** 2 / baseline_var)
    change_var = (1 - slope) ** 2 * selected_baseline_var + residual_var
    return {"selected_fraction": selected_fraction, "slope": slope,
            "baseline_mean": baseline_mean, "followup_mean": followup_mean,
            "mean_change": followup_mean - baseline_mean,
            "selected_baseline_variance": selected_baseline_var,
            "change_variance": change_var}


def simulate_pairs(n=2000, seed=20260919, effect=0.0):
    """Draw independent people with a shared stable component at both visits."""
    rng = Random(seed)
    pairs = []
    for _ in range(n):
        stable = rng.gauss(50, 8)
        pairs.append((stable + rng.gauss(0, 6), stable + rng.gauss(0, 6) + effect))
    return pairs


def examples():
    rho, sigma = 0.64, 10.0
    conditional = NormalDist(50 + rho * 20, sigma * sqrt(1 - rho * rho))
    large_drop = NormalDist(0, sqrt(2 * sigma * sigma * (1 - rho))).cdf(-10)
    return {
        "selected_cohort": selected_moments(),
        "interventions": {str(effect): selected_moments(effect=effect)
                          for effect in (-3.0, 0.0, 3.0)},
        "baseline_averaging": {str(k): selected_moments(baseline_count=k)
                               for k in (1, 4, 16)},
        "conditional_at_70": {"mean": conditional.mean,
                              "prediction_95": [conditional.inv_cdf(0.025), conditional.inv_cdf(0.975)]},
        "unselected_drop_at_least_10": large_drop,
        "at_least_one_large_drop_among_20_independent_people": 1 - (1 - large_drop) ** 20,
    }


def draw_figures():
    import matplotlib.pyplot as plt
    from housestyle import INK_MUTED, PALETTE, save, use

    use()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    pairs = simulate_pairs()
    for selected, color, label in ((False, INK_MUTED, "Not selected"),
                                   (True, PALETTE[1], "Selected: baseline at least 65")):
        points = [(x, y) for x, y in pairs if (x >= 65) == selected]
        ax.scatter(*zip(*points), s=13, color=color, alpha=0.55, edgecolors="none", label=label)
    ax.plot([10, 90], [10, 90], color=INK_MUTED, ls=":", label="No change in observed score")
    ax.plot([10, 90], [50 + .64 * (x - 50) for x in (10, 90)], color=PALETTE[0],
            label="Expected follow-up at each baseline")
    ax.axvline(65, color=PALETTE[1], ls="--", lw=1)
    ax.set(xlim=(10, 90), ylim=(10, 90), xlabel="Baseline score", ylabel="Follow-up score",
           title="Selection creates apparent improvement without an intervention")
    ax.legend(loc="upper left", fontsize=8.5)
    print(json.dumps(save(fig, "science_testimonial_selection")))

    fig, ax = plt.subplots(figsize=(9, 5.3))
    for effect, color, label in ((3, PALETTE[1], "Harmful: +3 relative to no intervention"),
                                 (0, PALETTE[0], "No intervention"),
                                 (-3, PALETTE[2], "Beneficial: -3 relative to no intervention")):
        row = selected_moments(effect=effect)
        ax.plot([0, 1], [row["baseline_mean"], row["followup_mean"]], marker="o", color=color)
        ax.annotate(f'{row["followup_mean"]:.2f}  {label}', (1, row["followup_mean"]),
                    xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.annotate("69.39", (0, selected_moments()["baseline_mean"]), xytext=(0, 9),
                textcoords="offset points", ha="center")
    ax.set(xlim=(-.1, 2.5), ylim=(57, 72), xticks=[0, 1],
           xticklabels=["Selected baseline", "Follow-up"], ylabel="Expected group mean score",
           title="All three trajectories improve from the selected baseline")
    ax.text(.01, .02, "Exact model expectations; lower scores are preferable. These are not observed trial results.",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED)
    print(json.dumps(save(fig, "science_testimonial_counterfactual")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(examples(), indent=2))
    if not args.dry_run:
        draw_figures()


if __name__ == "__main__":
    main()
