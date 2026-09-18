"""Reproduce six science-communication examples dated 2024 through 2026.

From the repository root:
    python assets/viz/generate_science_communication_figures.py --dry-run
    python assets/viz/generate_science_communication_figures.py

Dry-run prints the calculations without writing figures. Calculations use only
the standard library; figure generation also requires Matplotlib. All inputs
are illustrative assumptions, not measurements or fitted biological parameters.
"""

import argparse
from math import acos, atan, cos, log, pi, radians, sin, sqrt, tan
from statistics import NormalDist


def climate_tails(mean, sd=5):
    distribution = NormalDist(mean, sd)
    return distribution.cdf(0), 1 - distribution.cdf(15)


def normal_kl(mean_p, sd_p, mean_q, sd_q):
    """D(P || Q), in nats, for two univariate normal distributions."""
    if sd_p <= 0 or sd_q <= 0:
        raise ValueError("standard deviations must be positive")
    return log(sd_q / sd_p) + (sd_p**2 + (mean_p - mean_q)**2) / (2 * sd_q**2) - 0.5


def binary_kl(p, q):
    """D(Bernoulli(p) || Bernoulli(q)); this example uses interior probabilities."""
    if not 0 < p < 1 or not 0 < q < 1:
        raise ValueError("probabilities must be strictly between zero and one")
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


def climate_log_ratio(temperature):
    """Log density ratio: N(7, 25) versus N(5, 25)."""
    return ((temperature - 5)**2 - (temperature - 7)**2) / 50


def run_probability(tosses, run_length, heads_probability=0.5):
    """Probability of at least one all-heads run, allowing overlapping windows."""
    if tosses < 0 or run_length < 1 or not 0 <= heads_probability <= 1:
        raise ValueError("invalid toss count, run length, or probability")
    # States retain the trailing head count only while the target run is absent.
    states = [1.0] + [0.0] * (run_length - 1)
    for _ in range(tosses):
        states = [sum(states) * (1 - heads_probability)] + [
            mass * heads_probability for mass in states[:-1]
        ]
    return max(0.0, min(1.0, 1 - sum(states)))


def thermal_response(time_constant_days, period_days=365):
    """Amplitude fraction and peak lag for a linear periodically forced reservoir."""
    omega = 2 * pi / period_days
    return (1 / sqrt(1 + (omega * time_constant_days)**2),
            atan(omega * time_constant_days) / omega)


def solar_geometry(latitude, declination):
    """Ideal sphere and point Sun; return noon altitude, daylight, daily mean/S0."""
    phi, delta = radians(latitude), radians(declination)
    sunset = acos(max(-1, min(1, -tan(phi) * tan(delta))))
    noon_altitude = 90 - abs(latitude - declination)
    daylight = 24 * sunset / pi
    daily_mean = (sunset * sin(phi) * sin(delta)
                  + cos(phi) * cos(delta) * sin(sunset)) / pi
    return noon_altitude, daylight, max(0, daily_mean)


def selection_rows():
    """Expected counts through three hypothetical selective bottlenecks."""
    sensitive, resistant = 99_900.0, 100.0
    rows = []
    for step in range(4):
        rows.append((step, sensitive, resistant, resistant / (sensitive + resistant)))
        sensitive *= 0.01
        resistant *= 0.80
    return rows


def next_head_probabilities(heads):
    """Three distinct mechanisms, conditioned on an initial all-heads sequence."""
    if not isinstance(heads, int) or not 0 <= heads <= 5:
        raise ValueError("heads must be an integer from 0 to 5")
    fair = 0.5
    bag = (5 - heads) / (10 - heads)
    # Equal prior probability of a 25%-heads coin or a 75%-heads coin.
    mixture = (0.25**(heads + 1) + 0.75**(heads + 1)) / (0.25**heads + 0.75**heads)
    return fair, bag, mixture


def exposure_rows():
    # Name, concentration (mg/mL), volume (mL), amount (mg).
    scenarios = [("A", 10, 2), ("B", 1, 30), ("C", 1, 20)]
    return [(name, concentration, volume, concentration * volume)
            for name, concentration, volume in scenarios]


def risk_rows():
    # Hypothetical five-year risks in two populations of 1,000 people per group.
    scenarios = [("Population A", 20, 10), ("Population B", 2, 1)]
    return [(name, before, after, before - after, 1 - after / before)
            for name, before, after in scenarios]


def print_calculations():
    print("Climate: mean, P(below 0 C), P(above 15 C)")
    for mean in (5, 7):
        print(mean, *(f"{value:.6f}" for value in climate_tails(mean)))
    print("Climate KL, warmer || cooler: full reading, freezing indicator (nats)")
    print(normal_kl(7, 5, 5, 5), binary_kl(climate_tails(7)[0], climate_tails(5)[0]))
    print("Climate evidence under warmer model: n, mean, sd, P(log ratio < 0)")
    for n in (1, 10, 25, 100):
        evidence = NormalDist(0.08 * n, 0.4 * sqrt(n))
        print(n, evidence.mean, evidence.stdev, evidence.cdf(0))
    print("Seasons at 45 N: declination, noon altitude, daylight hours, daily mean/S0")
    for declination in (-23.44, 0, 23.44):
        print(declination, *(f"{value:.6f}" for value in solar_geometry(45, declination)))
    print("Seasons: latitude, December daylight/overhead hours, June daylight/overhead hours")
    for latitude in (0, 45, 70):
        winter, summer = solar_geometry(latitude, -23.44), solar_geometry(latitude, 23.44)
        print(latitude, winter[1], 24 * winter[2], summer[1], 24 * summer[2])
    print("Thermal reservoir: time constant (days), amplitude fraction, lag (days)")
    for tau in (10, 30, 90):
        print(tau, *thermal_response(tau))
    print("Resistance: bottleneck, sensitive, resistant, resistant fraction")
    for row in selection_rows():
        print(*(f"{value:.6g}" for value in row))
    print("Streak: initial heads, fair coin, bag, unknown coin")
    for heads in range(5):
        print(heads, *(f"{value:.6f}" for value in next_head_probabilities(heads)))
    print("Run search: tosses, probability of at least one four-head run")
    for tosses in (4, 20, 50, 100):
        print(tosses, run_probability(tosses, 4))
    print("Exposure: scenario, mg/mL, mL, mg")
    for row in exposure_rows():
        print(*row)
    print("Risk: population, before/1000, after/1000, difference/1000, relative reduction")
    for row in risk_rows():
        print(*row)


def plot_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from housestyle import PALETTE, save, use

    use()

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    temperatures = [i / 10 for i in range(-150, 251)]
    for index, mean in enumerate((5, 7)):
        distribution = NormalDist(mean, 5)
        density = [distribution.pdf(value) for value in temperatures]
        ax.plot(temperatures, density, color=PALETTE[index], label=f"Mean {mean} C")
        ax.fill_between(temperatures, density,
                        where=[value <= 0 for value in temperatures],
                        color=PALETTE[index], alpha=0.18)
    ax.axvline(0, color="#52514e", linestyle="--", linewidth=1)
    ax.set(xlabel="Temperature (C)", ylabel="Probability density",
           title="A warmer distribution still includes freezing days")
    ax.legend()
    print(save(fig, "science_weather_climate_shift")["path"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(temperatures, [climate_log_ratio(x) for x in temperatures], color=PALETTE[1])
    axes[0].axhline(0, color="#52514e", linewidth=1)
    axes[0].axvline(6, color="#52514e", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Observed temperature (C)", ylabel="Log likelihood ratio (nats)",
                title="One reading can favour either model", xlim=(-10, 20))
    sample_sizes = list(range(1, 101))
    z = NormalDist().inv_cdf(0.95)
    for sign, name, color in ((1, "Warmer generates data", PALETTE[1]),
                              (-1, "Cooler generates data", PALETTE[0])):
        means = [sign * 0.08 * n for n in sample_sizes]
        widths = [z * 0.4 * sqrt(n) for n in sample_sizes]
        axes[1].plot(sample_sizes, means, color=color, label=name)
        axes[1].fill_between(sample_sizes, [m - w for m, w in zip(means, widths)],
                             [m + w for m, w in zip(means, widths)], color=color, alpha=0.15)
    axes[1].axhline(0, color="#52514e", linewidth=1)
    axes[1].set(xlabel="Independent observations", ylabel="Total log likelihood ratio (nats)",
                title="Expected evidence and central 90% bands")
    axes[1].legend(fontsize=8, loc="upper left")
    print(save(fig, "science_climate_kl_evidence")["path"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["December\nsolstice", "Equinox", "June\nsolstice"]
    north = [solar_geometry(45, d) for d in (-23.44, 0, 23.44)]
    south = [solar_geometry(-45, d) for d in (-23.44, 0, 23.44)]
    for index, (name, rows) in enumerate((("45 N", north), ("45 S", south))):
        axes[0].plot(range(3), [r[1] for r in rows], marker="o", color=PALETTE[index], label=name)
        axes[1].plot(range(3), [24 * r[2] for r in rows], marker="o", color=PALETTE[index], label=name)
    for ax in axes:
        ax.set_xticks(range(3), labels)
        ax.legend()
    axes[0].set(title="Length of daylight", ylabel="Hours", ylim=(0, 24))
    axes[1].set(title="Daily incoming solar energy", ylabel="Equivalent hours of overhead sunlight", ylim=(0, 10))
    fig.suptitle("Opposite seasons at a fixed Earth-Sun distance")
    print(save(fig, "science_seasons_tilt_geometry")["path"])

    rows = selection_rows()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for column, name, color in ((1, "Sensitive", PALETTE[0]), (2, "Resistant", PALETTE[1])):
        axes[0].semilogy([r[0] for r in rows], [r[column] for r in rows], marker="o", label=name, color=color)
    axes[0].set(ylabel="Expected number (logarithmic scale)", title="Both numbers decrease")
    axes[0].legend()
    axes[1].plot([r[0] for r in rows], [100 * r[3] for r in rows], marker="o", color=PALETTE[1])
    axes[1].set(ylabel="Resistant share (%)", ylim=(0, 105), title="The resistant share increases")
    for ax in axes:
        ax.set(xlabel="Hypothetical selective bottlenecks")
        ax.set_xticks(range(4))
    print(save(fig, "science_antibiotic_selection")["path"])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for column, name in enumerate(("Known fair coin", "Bag without replacement", "Unknown 25% or 75% coin")):
        ax.plot(range(5), [next_head_probabilities(k)[column] for k in range(5)],
                marker="o", color=PALETTE[column], label=name)
    ax.set(xlabel="Initial consecutive heads observed", ylabel="Probability of heads next",
           ylim=(0, 1), title="The same streak means different things in different mechanisms")
    ax.set_xticks(range(5))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.19), fontsize=9)
    print(save(fig, "science_streaks_three_mechanisms")["path"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1))
    rows = exposure_rows()
    axes[0].bar([r[0] for r in rows], [r[1] for r in rows], color=PALETTE[0], width=0.6)
    axes[1].bar([r[0] for r in rows], [r[3] for r in rows], color=PALETTE[1], width=0.6)
    axes[0].set(title="Concentration", ylabel="mg per mL", ylim=(0, 12))
    axes[1].set(title="Amount in the chosen volume", ylabel="mg", ylim=(0, 36))
    for ax in axes:
        ax.set(xlabel="Hypothetical sample")
    for pos, row in enumerate(rows):
        axes[1].text(pos, row[3] + 1, f"{row[2]} mL: {row[3]} mg", ha="center", fontsize=9)
    fig.suptitle("A lower concentration can deliver a larger amount")
    print(save(fig, "science_concentration_and_amount")["path"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, row in zip(axes, risk_rows()):
        name, before, after, difference, relative = row
        ax.bar([0, 1], [before, after], color=PALETTE[:2], width=0.55)
        ax.set_xticks([0, 1], ["Comparison", "Intervention"])
        ax.set(title=f"{name}: {relative:.0%} lower risk", ylim=(0, 24),
               xlabel=f"{difference} fewer events per 1,000 people")
        for pos, value in enumerate((before, after)):
            ax.text(pos, value + 0.5, str(value), ha="center")
    axes[0].set_ylabel("Events per 1,000 people over five years")
    print(save(fig, "science_relative_absolute_risk")["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print calculations without writing figures")
    args = parser.parse_args()
    print_calculations()
    if args.dry_run:
        print("No figures written (--dry-run).")
    else:
        plot_figures()


if __name__ == "__main__":
    main()
