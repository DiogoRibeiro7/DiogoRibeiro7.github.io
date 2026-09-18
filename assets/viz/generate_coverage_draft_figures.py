"""Reproduce three synthetic examples for the Research/Environment/Healthcare drafts.

Run with --dry-run to print the calculations without writing figures. Plotting
dependencies are loaded only when exporting the PNGs; the models use stdlib.
"""

import argparse
import json
from math import comb, isfinite, sqrt
from statistics import variance


def balance_probabilities(size=20, high=10, treated=10):
    """Exact binary-covariate imbalance under complete randomisation."""
    if not (0 <= high <= size and 0 < treated < size):
        raise ValueError("Require 0 <= high <= size and 0 < treated < size")
    denominator = comb(size, treated)
    return [
        {"high_treated": k,
         "difference": k / treated - (high - k) / (size - treated),
         "probability": comb(high, k) * comb(size - high, treated - k) / denominator}
        for k in range(max(0, treated - (size - high)), min(high, treated) + 1)
    ]


def adjusted_sd(outcomes, covariates, treated, coefficient):
    """Randomisation SD with a fixed, externally chosen adjustment coefficient.

    Outcomes are the fixed untreated potential outcomes. A constant additive
    treatment effect does not change this variance. The coefficient is not fit.
    """
    if len(outcomes) != len(covariates) or not 0 < treated < len(outcomes):
        raise ValueError("Equal population lengths and two nonempty arms required")
    residuals = [y - coefficient * x for y, x in zip(outcomes, covariates)]
    return sqrt((1 / treated + 1 / (len(outcomes) - treated)) * variance(residuals))


def storage_ledger(generation, demand, capacity, power, initial=0.0,
                   charge_efficiency=0.9, discharge_efficiency=0.9, hours=1.0):
    """Greedy, no-export storage dispatch; powers are AC kW, state is stored kWh.

    Imports fill each shortfall. No grid charging, standby loss, reserve,
    degradation, or simultaneous charging/discharging is included.
    """
    values = [*generation, *demand, capacity, power, initial,
              charge_efficiency, discharge_efficiency, hours]
    if not all(isfinite(v) for v in values):
        raise ValueError("All inputs must be finite")
    if (len(generation) != len(demand) or capacity < 0 or power < 0
            or not 0 <= initial <= capacity or hours <= 0
            or not 0 < charge_efficiency <= 1 or not 0 < discharge_efficiency <= 1
            or any(v < 0 for v in [*generation, *demand])):
        raise ValueError("Invalid storage parameters or power series")
    state = initial
    rows = []
    for solar, load in zip(generation, demand):
        before = state
        surplus, deficit = max(0.0, solar - load), max(0.0, load - solar)
        charge = min(surplus, power, (capacity - state) / (charge_efficiency * hours))
        discharge = min(deficit, power, state * discharge_efficiency / hours)
        state = before + hours * (charge_efficiency * charge - discharge / discharge_efficiency)
        # Remove rounding at an active capacity bound, not physical overshoots.
        state = min(capacity, max(0.0, state))
        rows.append({"generation": solar, "demand": load, "before": before,
                     "charge": charge, "discharge": discharge, "after": state,
                     "imports": deficit - discharge, "curtailment": surplus - charge,
                     "loss": hours * ((1 - charge_efficiency) * charge
                                      + (1 / discharge_efficiency - 1) * discharge)})
    return rows


def adaptive_baseline(observations, alpha, initial=0.0):
    """Score before updating; None means no score and no baseline update."""
    if not 0 <= alpha <= 1 or not isfinite(initial):
        raise ValueError("Require finite initial state and alpha in [0, 1]")
    baseline = initial
    rows = []
    for observation in observations:
        before = baseline
        if observation is None:
            score = None
        else:
            if not isfinite(observation):
                raise ValueError("Use None for a missing observation")
            score = observation - before
            baseline += alpha * score
        rows.append({"observation": observation, "before": before,
                     "score": score, "after": baseline})
    return rows


def example_results():
    balance = balance_probabilities()
    x = [0] * 10 + [1] * 10
    noise = [-2, -1, 0, 1, 2] * 4
    y = [10 + 4 * xi + ei for xi, ei in zip(x, noise)]
    storage = []
    for capacity, power in ((6, 2), (6, 4), (10, 4)):
        rows = storage_ledger([0, 6, 6, 0, 0, 0], [2] * 6, capacity, power)
        storage.append({"capacity": capacity, "power": power, "rows": rows,
                        "imports_kwh": sum(r["imports"] for r in rows),
                        "curtailment_kwh": sum(r["curtailment"] for r in rows),
                        "loss_kwh": sum(r["loss"] for r in rows)})
    return {
        "balance": balance,
        "perfect_balance_probability": balance[5]["probability"],
        "imbalance_at_least_40_points": sum(r["probability"] for r in balance
                                            if abs(r["high_treated"] - 5) >= 2),
        "adjustment_sd": {str(b): adjusted_sd(y, x, 10, b) for b in (0, 2, 4, 8)},
        "storage": storage,
        "baseline": {str(a): adaptive_baseline([4.0] * 30, a) for a in (0, 0.02, 0.1, 0.25)},
    }


def export_figures(results):
    import matplotlib.pyplot as plt
    from housestyle import PALETTE, save, use

    use()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    balance = results["balance"]
    axes[0].bar([r["high_treated"] for r in balance],
                [100 * r["probability"] for r in balance], color=PALETTE[0])
    axes[0].set(xlabel="High-covariate units in treatment (of 10)",
                ylabel="Probability (%)", title="Perfect balance occurs in 34.4% of allocations",
                xticks=range(0, 11, 2))
    x = [0] * 10 + [1] * 10
    y = [10 + 4 * xi + ei for xi, ei in zip(x, [-2, -1, 0, 1, 2] * 4)]
    coefficients = [i / 10 for i in range(81)]
    axes[1].plot(coefficients, [adjusted_sd(y, x, 10, b) for b in coefficients])
    axes[1].axhline(adjusted_sd(y, x, 10, 0), ls="--", color=PALETTE[1], label="Unadjusted")
    axes[1].set(xlabel="Fixed adjustment coefficient", ylabel="SD of estimated effect",
                title="Adjustment helps when the coefficient is useful")
    axes[1].legend()
    outputs = [save(fig, "research_randomisation_balance")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(range(1, 7), [0, 6, 6, 0, 0, 0], label="Solar")
    axes[0].axhline(2, color=PALETTE[1], ls="--", label="Demand")
    axes[0].set(title="Equal total energy: 12 kWh", xlabel="Hour", ylabel="Power (kW)", xticks=range(1, 7))
    axes[0].legend()
    for index, case in enumerate(results["storage"]):
        label = f'{case["capacity"]} kWh / {case["power"]} kW'
        axes[1].plot(range(7), [0] + [r["after"] for r in case["rows"]],
                     marker="o", label=label)
        axes[2].bar(index, case["imports_kwh"], color=PALETTE[index])
        axes[2].text(index, case["imports_kwh"] + 0.07, f'{case["imports_kwh"]:.2f}', ha="center")
    axes[1].set(title="State depends on the whole path", xlabel="End of hour", ylabel="Stored energy (kWh)")
    axes[1].legend(fontsize=8)
    axes[2].set(title="Imports still needed", ylabel="Imported energy (kWh)", ylim=(0, 5.5),
                xticks=range(3), xticklabels=["6 kWh\n2 kW", "6 kWh\n4 kW", "10 kWh\n4 kW"])
    outputs.append(save(fig, "environment_storage_constraints"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for alpha, rows in results["baseline"].items():
        label = "Frozen" if alpha == "0" else f"alpha = {alpha}"
        axes[0].plot(range(1, 31), [r["before"] for r in rows], label=label)
        axes[1].plot(range(1, 31), [r["score"] for r in rows], label=label)
    axes[0].axhline(4, color="0.4", ls=":", label="Persistent observed level")
    axes[0].set(title="The reference follows the change", xlabel="Observed day after step", ylabel="Baseline before scoring")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    axes[1].axhline(2.5, color="0.4", ls=":", label="Illustrative flag threshold")
    axes[1].set(title="Deviation fades without recovery", xlabel="Observed day after step", ylabel="Observation minus baseline")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    outputs.append(save(fig, "healthcare_adaptive_baseline"))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print model results without writing PNGs")
    args = parser.parse_args()
    results = example_results()
    print(json.dumps(results if args.dry_run else export_figures(results), indent=2))


if __name__ == "__main__":
    main()
