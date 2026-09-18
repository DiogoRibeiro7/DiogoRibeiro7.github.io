"""Reproduce the figures and numerical tables for the autumn 2026 article batch.

Run from the repository root:
    python assets/viz/generate_2026_evidence_articles.py

Dependencies: NumPy and Matplotlib, plus the adjacent house-style files.
Examples are synthetic except for three archived BEA GDP releases, transcribed
with source URLs in assets/data/gdp_q1_2025_release_vintages.csv. No network
access is required. Physical constants use their exact SI definitions.
"""

import csv
from datetime import datetime
from math import exp, log2, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from housestyle import PALETTE, save, use


def longitudinal_budget():
    """Compare population and individual precision at a fixed observation count."""
    rng = np.random.default_rng(20260921)
    replications = 10_000
    designs = [(20, 50), (50, 20), (100, 10), (200, 5)]
    print("n, T, analytic population SE, simulated SE, analytic individual RMSE, simulated RMSE, SD(tau2)")
    for n, t in designs:
        noise = rng.normal(0, 3 / sqrt(t), size=(replications, n))
        theta = rng.normal(0, 1, size=(replications, n))
        means = theta + noise
        empirical_se = means.mean(axis=1).std(ddof=1)
        print(n, t, f"{sqrt((1 + 9 / t) / n):.6f}",
              f"{empirical_se:.6f}", f"{3 / sqrt(t):.6f}",
              f"{sqrt(np.mean(noise**2)):.6f}",
              f"{sqrt(2 / (n - 1)) * (1 + 9 / t):.6f}")

    t = np.array([2, 5, 10, 20, 25, 50, 100, 200])
    n = 1000 / t
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(t, np.sqrt((1 + 9 / t) / n), marker="o", color=PALETTE[0])
    axes[1].plot(t, 3 / np.sqrt(t), marker="o", color=PALETTE[1])
    axes[0].set(title="Population mean", ylabel="Standard error")
    axes[1].set(title="Individual level, without pooling", ylabel="RMSE")
    for ax in axes:
        ax.set(xscale="log", xlabel="Measurements per subject (T)")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200],
                      labels=["2", "5", "10", "20", "50", "100", "200"])
    fig.suptitle("1,000 observations: allocating more to each person has two effects")
    print(save(fig, "subjects_vs_measurements_2026")["path"])


def monitoring_worlds():
    """Enumerate exact joint distributions; no sampling approximation is used."""
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    score = np.where(x == 1, 0.9, 0.1)
    prediction = (score >= 0.5).astype(int)
    distributions = {
        "Reference": np.array([0.45, 0.05, 0.05, 0.45]),
        "Hidden reversal": np.array([0.05, 0.45, 0.45, 0.05]),
        "Input shift only": np.array([0.09, 0.01, 0.09, 0.81]),
    }
    values = []
    print("world, P(X=1), mean confidence, accuracy, Brier score")
    for name, probability in distributions.items():
        row = [probability @ x,
               probability @ np.maximum(score, 1 - score),
               probability @ (prediction == y),
               probability @ (score - y)**2]
        values.append(row)
        print(name, *(f"{v:.3f}" for v in row))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    locations = np.arange(3)
    values = np.asarray(values)
    axes[0].bar(locations - 0.17, values[:, 0], width=0.34,
                color=PALETTE[0], label="Positive prediction share")
    axes[0].bar(locations + 0.17, values[:, 1], width=0.34,
                color=PALETTE[1], label="Mean confidence")
    axes[1].bar(locations, values[:, 2], color=PALETTE[0], width=0.5)
    axes[0].set_title("Observable before labels arrive")
    axes[1].set_title("Accuracy requires outcomes")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.19), fontsize=8)
    labels = ["Reference", "Hidden\nreversal", "Input shift\nonly"]
    for ax in axes:
        ax.set_xticks(locations, labels)
        ax.set(ylim=(0, 1.05), ylabel="Proportion")
    for loc, accuracy in zip(locations, values[:, 2]):
        axes[1].text(loc, accuracy + 0.025, f"{accuracy:.0%}", ha="center")
    fig.suptitle("Identical monitoring signals can conceal a different accuracy")
    print(save(fig, "unlabelled_monitoring_worlds_2026")["path"])


def solve_conversion(steps, method="heun", rate=2.0):
    """Integrate a two-state conversion to t=1 and return endpoint and mass error."""
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be euler or heun")
    h = 1 / steps
    z = np.array([1.0, 0.0])
    mass_error = 0.0

    def rhs(state):
        return rate * state[0] * np.array([-1.0, 1.0])

    for _ in range(steps):
        k1 = rhs(z)
        if method == "euler":
            z = z + h * k1
        else:
            z = z + h * (k1 + rhs(z + h * k1)) / 2
        mass_error = max(mass_error, abs(z.sum() - 1))
    return z, mass_error


def numerical_verification():
    exact = np.array([exp(-2), 1 - exp(-2)])
    steps = np.array([5, 10, 20, 40, 80])
    errors = {}
    print("method, steps, endpoint error, maximum mass error, observed order")
    cases = [("Euler", "euler", 2), ("Heun", "heun", 2),
             ("Heun, wrong rate", "heun", 1.8)]
    for name, method, rate in cases:
        previous = None
        errors[name] = []
        for n in steps:
            endpoint, mass_error = solve_conversion(int(n), method, rate)
            error = float(np.max(np.abs(endpoint - exact)))
            errors[name].append(error)
            order = "--" if previous is None else f"{log2(previous / error):.3f}"
            print(name, n, f"{error:.9f}", f"{mass_error:.3e}", order)
            previous = error

    wrong_values = [solve_conversion(n, "heun", 1.8)[0][0] for n in [20, 40, 80]]
    self_order = log2(abs(wrong_values[0] - wrong_values[1])
                      / abs(wrong_values[1] - wrong_values[2]))
    print("Wrong-rate self-convergence order:", f"{self_order:.3f}")
    print("Wrong-rate limiting error:", f"{exp(-1.8) - exp(-2):.9f}")
    for method in ("euler", "heun"):
        n = 1
        while np.max(np.abs(solve_conversion(n, method)[0] - exact)) > 1e-4:
            n *= 2
        calls = n if method == "euler" else 2 * n
        print("First power-of-two design below 1e-4:", method, n,
              "steps,", calls, "RHS calls")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for index, (name, _, _) in enumerate(cases):
        ax.loglog(1 / steps, errors[name], marker="o", color=PALETTE[index], label=name)
    ax.set(xlabel="Step size h (smaller to the right)",
           ylabel="Maximum component error at t = 1",
           title="Refinement exposes an implementation converging to the wrong answer")
    ax.invert_xaxis()
    ax.set_xticks(1 / steps, labels=["0.2", "0.1", "0.05", "0.025", "0.0125"])
    ax.legend()
    print(save(fig, "numerical_verification_2026")["path"])


def wearable_alerts():
    """Expected counts for one evaluable opportunity per hypothetical person."""
    prevalence = np.array([0.01, 0.05, 0.20])
    true_positive = 10_000 * prevalence * 0.90
    false_negative = 10_000 * prevalence * 0.10
    false_positive = 10_000 * (1 - prevalence) * 0.05
    true_negative = 10_000 * (1 - prevalence) * 0.95
    ppv = true_positive / (true_positive + false_positive)
    print("prevalence, TP, FP, FN, TN, PPV")
    for row in zip(prevalence, true_positive, false_positive,
                   false_negative, true_negative, ppv):
        print(*(f"{value:.6g}" for value in row))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    positions = np.arange(3)
    ax.barh(positions, true_positive, color=PALETTE[0], label="True alerts")
    ax.barh(positions, false_positive, left=true_positive,
            color=PALETTE[1], label="False alerts")
    for pos, tp, fp, predictive_value in zip(positions, true_positive, false_positive, ppv):
        ax.text(tp + fp + 35, pos, f"{predictive_value:.1%} true", va="center")
    ax.set_yticks(positions, labels=["1%", "5%", "20%"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 2800), xlabel="Alerts per 10,000 people",
           ylabel="Condition prevalence",
           title="Same hypothetical test; different meaning of a positive alert")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    print(save(fig, "wearable_alert_denominators_2026")["path"])


def microwave_energy():
    """Separate photon energy from aggregate absorbed energy; no exposure model."""
    planck = 6.62607015e-34
    elementary_charge = 1.602176634e-19
    speed_of_light = 299792458
    frequency = 2.45e9
    microwave_ev = planck * frequency / elementary_charge
    green_ev = planck * speed_of_light / (550e-9 * elementary_charge)
    print("Microwave photon energy (eV):", microwave_ev)
    print("Green-light photon energy (eV):", green_ev)
    print("Green/microwave photon energy ratio:", green_ev / microwave_ev)
    print("Photons per second at 500 W absorbed:", 500 / (planck * frequency))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), layout="constrained")
    energies = [microwave_ev, green_ev]
    axes[0].scatter(energies, [0, 1], color=PALETTE[:2], s=75)
    axes[0].set(xscale="log", xlim=(1e-6, 1e2), ylim=(-0.6, 1.6),
                xlabel="Energy per photon (eV, logarithmic scale)",
                title="Energy of one photon")
    axes[0].set_yticks([0, 1], ["Microwave\n2.45 GHz", "Green light\n550 nm"])
    for value, pos, label in zip(energies, [0, 1], ["0.0000101 eV", "2.25 eV"]):
        axes[0].annotate(label, (value, pos), xytext=(0, 14),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].bar([0, 1], [30, 60], width=0.55, color=PALETTE[0])
    axes[1].set_xticks([0, 1], ["500 W", "1,000 W"])
    axes[1].set(xlabel="Assumed absorbed microwave power",
                ylabel="Total absorbed energy (kilojoules)", ylim=(0, 70),
                title="Energy absorbed in 60 seconds")
    for pos, value in enumerate([30, 60]):
        axes[1].text(pos, value + 1.5, f"{value} kJ", ha="center")
    print(save(fig, "microwave_photons_and_power_2026")["path"])


def load_gdp_vintages():
    """Read the three sourced releases; later revisions are outside this example."""
    path = Path(__file__).resolve().parents[1] / "data" / "gdp_q1_2025_release_vintages.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["released_at"] = datetime.fromisoformat(row["released_at"])
        row["value"] = float(row["value"])
    return sorted(rows, key=lambda row: row["released_at"])


def gdp_as_of(rows, cutoff):
    """Select within the CSV's single series/period; assume immediate availability."""
    eligible = [row for row in rows if row["released_at"] <= cutoff]
    return max(eligible, key=lambda row: row["released_at"])["value"] if eligible else None


def economic_vintages():
    rows = load_gdp_vintages()
    print("cutoff, contemporaneous Q1 GDP growth, below -0.4%")
    for date in ["2025-04-29", "2025-05-01", "2025-06-01", "2025-07-01"]:
        value = gdp_as_of(rows, datetime.fromisoformat(date + "T12:00:00+00:00"))
        decision = None if value is None else value < -0.4
        print(date, value, decision)
    print("Quarterly rate corresponding to -0.3% annualized:",
          100 * ((1 - 0.3 / 100)**0.25 - 1))

    dates = [row["released_at"] for row in rows]
    end = datetime.fromisoformat("2025-07-01T12:00:00+00:00")
    values = [row["value"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.step(dates + [end], values + [values[-1]], where="post",
            color=PALETTE[0], label="Estimate available at the time")
    ax.scatter(dates, values, color=PALETTE[0], zorder=3)
    ax.hlines(values[-1], dates[0], end, color=PALETTE[1], linestyle="--",
              label="Third estimate copied backwards (hindsight)")
    for date, value, label in zip(dates, values, ["Advance: -0.3%", "Second: -0.2%", "Third: -0.5%"]):
        ax.annotate(label, (date, value), xytext=(5, 12),
                    textcoords="offset points", fontsize=9)
    ax.set(ylim=(-0.6, -0.1), xlim=(dates[0], end), xlabel="Date information becomes available (2025, UTC)",
           ylabel="Real GDP growth (%, annualized)",
           title="One quarter of economic activity, three dated estimates")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=9)
    print(save(fig, "gdp_release_vintages_2026")["path"])


if __name__ == "__main__":
    use()
    longitudinal_budget()
    monitoring_worlds()
    numerical_verification()
    wearable_alerts()
    microwave_energy()
    economic_vintages()
