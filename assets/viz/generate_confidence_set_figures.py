"""Figures for "Confidence Sets Are Not Just Intervals".

    python generate_confidence_set_figures.py            # both figures
    python generate_confidence_set_figures.py --dry-run  # compute, write nothing

The model is Y = (theta^2 - 1)^2 + noise. Its inverted 95% set has a closed form,
which the first figure draws next to the p-value curve; the second shows a
profile over a nuisance parameter whose two basins a local optimiser confuses.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import housestyle as hs
from housestyle import PALETTE as P

ALPHA = 0.05
SIGMA = 0.08
Z = norm.isf(ALPHA / 2)


def accepted_set(y, sigma=SIGMA, z=Z):
    """Exact {theta : |y - (theta^2 - 1)^2| < z sigma} as sorted open intervals."""
    low, high = max(y - z * sigma, 0.0), y + z * sigma
    if high <= 0:
        return []
    a, b = np.sqrt(low), np.sqrt(high)               # accepted |theta^2 - 1| lies in (a, b)
    squares = [(1 + a, 1 + b)]                        # intervals for theta^2
    if max(1 - b, 0.0) < 1 - a:
        squares.append((max(1 - b, 0.0), 1 - a))
    pieces = []
    for u_low, u_high in squares:
        if u_low == 0.0:
            pieces.append((-np.sqrt(u_high), np.sqrt(u_high)))
        else:
            pieces += [(-np.sqrt(u_high), -np.sqrt(u_low)), (np.sqrt(u_low), np.sqrt(u_high))]
    merged = []
    for low, high in sorted(pieces):                  # the two branches touch when a = 0
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def p_value(theta, y, sigma=SIGMA):
    return 2 * norm.sf(np.abs(y - (theta**2 - 1) ** 2) / sigma)


def components_figure():
    theta = np.linspace(-1.8, 1.8, 3601)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, y in zip(axes, (0.04, 0.50)):
        pieces = accepted_set(y)
        ax.plot(theta, p_value(theta, y), color=P[0], lw=2.2, label="p-value of each candidate")
        ax.axhline(ALPHA, color=hs.INK_MUTED, lw=1.4, ls=":", label="level, 0.05")
        for k, (low, high) in enumerate(pieces):
            ax.axvspan(low, high, color=P[0], alpha=0.14, lw=0,
                       label="accepted: the confidence set" if k == 0 else None)
        ax.plot([pieces[0][0], pieces[-1][1]], [-0.07, -0.07], color=P[1], lw=3,
                solid_capstyle="butt", label="convex hull of the set", clip_on=False)
        ax.set_ylim(-0.12, 1.05)
        ax.set_xlabel("candidate value of the parameter")
        ax.set_title(f"observed {y:.2f}: {len(pieces)} components")
    axes[0].set_ylabel("p-value")
    axes[0].legend(loc="center", fontsize=8.5)          # the rejected region around zero is empty
    fig.tight_layout()
    alt = ("Two panels plotting the p-value of each candidate parameter value against the candidate, "
           "with the 0.05 level as a dotted line and the accepted regions shaded. With an observation "
           "of 0.04 the accepted set is two separate intervals around minus one and plus one; with an "
           "observation of 0.50 it is four. A bar under each panel marks the convex hull, which runs "
           "across the wide rejected region around zero.")
    return fig, "confidence_set_components", alt


def profile_figure():
    sigma, psi, y = 0.06, 0.40, 0.0
    g = lambda lam: (lam**2 - 1) ** 2 - 0.3 * lam
    stat = lambda lam: ((y - psi - g(lam)) / sigma) ** 2
    lam = np.linspace(-1.7, 1.7, 2001)

    fig, ax = plt.subplots()
    ax.plot(lam, stat(lam), color=P[0], lw=2.4, label="test statistic along the nuisance parameter")
    ax.axhline(Z**2, color=hs.INK_MUTED, lw=1.4, ls=":", label="critical value, 3.84")
    for start, colour, text, offset in ((-1.2, P[1], "local optimiser started at -1.2\nstops here: reject", (10, 12)),
                                        (1.2, P[2], "started at 1.2: the true minimum, 2.48\naccept (p = 0.115)", (-150, 28))):
        found = float(minimize(lambda v: stat(v[0]), x0=[start], method="BFGS").x[0])
        ax.plot([found], [stat(found)], marker="o", ms=9, color=colour,
                markeredgecolor=hs.SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(text, (found, stat(found)), xytext=offset, textcoords="offset points",
                    fontsize=9, color=hs.INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set_xlabel("nuisance parameter")
    ax.set_ylabel("test statistic for the target value 0.40")
    ax.set_title("Two basins: which one the optimiser finds decides the inference")
    ax.legend(loc="upper center")
    alt = ("The test statistic for one target value plotted against a nuisance parameter on a logarithmic "
           "axis. The curve has two basins. The left one bottoms out at about 134, far above the critical "
           "value of 3.84, and a local optimiser started there stops and rejects the target. The right "
           "basin reaches 2.48, below the critical value, so the target value is in fact accepted.")
    return fig, "confidence_set_profile_basins", alt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="compute the figures and write nothing")
    args = parser.parse_args()
    hs.use()
    for build in (components_figure, profile_figure):
        fig, slug, alt = build()
        if args.dry_run:
            plt.close(fig)
            print(f"  would write {slug}")
            continue
        result = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:32} {result['width']}x{result['height']}")
        print("  " + result["markdown"][:80] + "...")


if __name__ == "__main__":
    main()
