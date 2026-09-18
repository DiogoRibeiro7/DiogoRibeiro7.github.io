"""Reproduce the Gaussian mean-shift CUSUM example; --dry-run writes no files."""

import argparse
import json
from math import isfinite
from random import Random


def cusum(data, mu_0, delta, h):
    """Return every upper-CUSUM score and the first alarm's zero-based index.

    delta is a positive target mean shift, in observation units. The reference
    allowance is delta / 2. h uses the same units as the accumulated score.
    Scores continue after the first alarm for illustration; they do not reset.
    """
    if not all(isfinite(v) for v in (mu_0, delta, h)) or delta <= 0 or h <= 0:
        raise ValueError("Use a finite baseline and positive finite delta and h")
    score, scores, alarm_index = 0.0, [], None
    for index, value in enumerate(data):
        if not isfinite(value):
            raise ValueError("CUSUM requires finite observations")
        score = max(0.0, score + value - mu_0 - delta / 2)
        scores.append(score)
        if alarm_index is None and score > h:
            alarm_index = index
    return scores, alarm_index


def example():
    rng = Random(42)
    mu_0, mu_1, sigma = 0.0, 2.0, 1.0
    first_changed_index = 60
    data = ([rng.gauss(mu_0, sigma) for _ in range(first_changed_index)]
            + [rng.gauss(mu_1, sigma) for _ in range(40)])
    delta, h = mu_1 - mu_0, 5.0
    scores, alarm_index = cusum(data, mu_0, delta, h)
    return {"data": data, "scores": scores, "alarm_index": alarm_index,
            "first_changed_index": first_changed_index, "delta": delta, "h": h}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = example()
    print(json.dumps(result, indent=2))
    if args.dry_run:
        return

    import matplotlib.pyplot as plt
    from housestyle import PALETTE, save, use

    use()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    indices = range(len(result["data"]))
    axes[0].plot(indices, result["data"], linewidth=1.5, label="Observed value")
    axes[0].set(ylabel="Observation", title="A mean shift and a later alarm are different events")
    axes[1].plot(indices, result["scores"], label="Upper CUSUM")
    axes[1].axhline(result["h"], ls=":", color=PALETTE[1], label="Threshold h = 5")
    axes[1].set(xlabel="Observation index (zero-based)", ylabel="CUSUM score")
    for axis in axes:
        axis.axvline(result["first_changed_index"], ls="--", color=PALETTE[1], label="First changed index: 60")
        if result["alarm_index"] is not None:
            axis.axvline(result["alarm_index"], ls="-.", color=PALETTE[2],
                         label=f'First alarm index: {result["alarm_index"]}')
        axis.legend(loc="upper left", fontsize=8)
    print(json.dumps(save(fig, "sequential_cusum_worked_example"), indent=2))


if __name__ == "__main__":
    main()
