"""Ideal two-path/marker calculations; --dry-run prints without writing figures.

Core calculations use only the standard library. Matplotlib is needed solely
to regenerate the two figures. These are model predictions, not measured data.
"""

import argparse
import cmath
import json
from math import cos, isfinite, pi, prod, sin, sqrt


def check_overlap(gamma):
    if not isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Use a real marker overlap between zero and one")


def state(phi, gamma):
    """Joint amplitudes C[path][marker] before the final path recombination.

    Marker states are d0=(1,0) and d1=(gamma,sqrt(1-gamma**2)).
    Path alternatives have equal amplitudes and relative phase phi.
    """
    check_overlap(gamma)
    if not isfinite(phi):
        raise ValueError("The phase must be finite")
    phase = cmath.exp(1j*phi)
    return ((1/sqrt(2), 0j),
            (phase*gamma/sqrt(2), phase*sqrt(1-gamma*gamma)/sqrt(2)))


def reduced_path_state(phi, gamma):
    """Trace over the two orthogonal marker basis states."""
    amplitudes = state(phi, gamma)
    return [[sum(amplitudes[a][k]*complex(amplitudes[b][k]).conjugate() for k in range(2))
             for b in range(2)] for a in range(2)]


def marker_basis(beta=0, eta=0):
    """An arbitrary orthonormal marker basis, including a complex relative phase."""
    if not isfinite(beta) or not isfinite(eta):
        raise ValueError("Basis angles must be finite")
    return ((cos(beta), cmath.exp(1j*eta)*sin(beta)),
            (-cmath.exp(-1j*eta)*sin(beta), cos(beta)))


def joint_probabilities(phi, gamma, beta=0, eta=0):
    """Born probabilities P[path port +/-, marker result 0/1]."""
    amplitudes = state(phi, gamma)
    paths = ((1/sqrt(2), 1/sqrt(2)), (1/sqrt(2), -1/sqrt(2)))
    markers = marker_basis(beta, eta)
    return [[abs(sum(complex(paths[a][p]).conjugate()*complex(markers[j][k]).conjugate()
                     * amplitudes[p][k] for p in range(2) for k in range(2)))**2
             for j in range(2)] for a in range(2)]


def metrics(gamma):
    check_overlap(gamma)
    eigenvalues = [(1+gamma)/2, (1-gamma)/2]
    density = reduced_path_state(.73, gamma)
    purity = sum(density[i][j]*density[j][i] for i in range(2) for j in range(2)).real
    distinguishability = sqrt(1-gamma*gamma)
    return {"overlap": gamma, "visibility": gamma,
            "distinguishability": distinguishability,
            "optimal_path_guess_probability": (1+distinguishability)/2,
            "path_purity": purity, "path_eigenvalues": eigenvalues,
            "port_plus_at_phase_zero": sum(joint_probabilities(0, gamma)[0]),
            "port_plus_at_phase_pi": sum(joint_probabilities(pi, gamma)[0])}


def environment_overlap(overlaps):
    """Product overlap for conditionally factorised pure environment records."""
    overlaps = tuple(overlaps)
    for gamma in overlaps:
        check_overlap(gamma)
    return prod(overlaps)


def examples():
    rows = []
    for fraction in (0, .5, 1):
        joint = joint_probabilities(fraction*pi, 0, beta=pi/4)
        rows.append({"phase_over_pi": fraction, "joint_probabilities": joint,
                     "expected_counts_10000": [[round(10000*p) for p in row] for row in joint],
                     "port_plus_marginal": sum(joint[0]),
                     "port_plus_given_marker": [joint[0][j]/sum(joint[a][j] for a in range(2))
                                                for j in range(2)]})
    p_plus = sum(joint_probabilities(0, .6)[0])
    return {"marker_cases": [metrics(gamma) for gamma in (1, .6, 0)],
            "independent_trials_at_overlap_point6_phase_zero": {
                "trials": 10000, "expected_plus_count": 10000*p_plus,
                "plus_count_standard_deviation": sqrt(10000*p_plus*(1-p_plus))},
            "eraser": rows,
            "environment_visibility": {str(m): environment_overlap([.95]*m) for m in (0, 1, 20, 100)}}


def draw_figures():
    import matplotlib.pyplot as plt
    from housestyle import INK_SECONDARY, PALETTE, save, use

    use()
    phases = [2*pi*i/400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.get_layout_engine().set(rect=(0, .12, 1, .83))
    for gamma, color, style in ((1, PALETTE[0], "-"), (.6, PALETTE[1], "--"), (0, PALETTE[2], ":")):
        values = [sum(joint_probabilities(phi, gamma)[0]) for phi in phases]
        ax.plot([p/pi for p in phases], values, color=color, ls=style,
                label=f"Marker overlap {gamma:g}; visibility {gamma:g}")
    ax.set(xlim=(0, 2), ylim=(-.02, 1.2), yticks=[0, .25, .5, .75, 1],
           xticks=[0, .5, 1, 1.5, 2], xlabel="Controlled phase / π",
           ylabel="Probability of the + output", title="Interference follows the overlap of physical marker states")
    ax.legend(loc="upper center", ncol=1, fontsize=9)
    fig.text(.02, .008, "Ideal equal-amplitude two-path model; all marker outcomes are included.\n"
             "The calculation changes physical correlations. It contains no parameter for awareness or intention.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_quantum_marker_visibility")))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.get_layout_engine().set(rect=(0, .12, 1, .83))
    joints = [joint_probabilities(phi, 0, beta=pi/4) for phi in phases]
    for j, color, style, label in ((0, PALETTE[0], "-", "Given marker +"),
                                   (1, PALETTE[1], "--", "Given marker −")):
        values = [table[0][j]/sum(table[a][j] for a in range(2)) for table in joints]
        ax.plot([p/pi for p in phases], values, color=color, ls=style, label=label)
    ax.plot([p/pi for p in phases], [sum(table[0]) for table in joints], color=PALETTE[2],
            ls=":", lw=2.5, label="All marker outcomes combined")
    ax.set(xlim=(0, 2), ylim=(-.02, 1.2), yticks=[0, .25, .5, .75, 1],
           xticks=[0, .5, 1, 1.5, 2], xlabel="Controlled phase / π",
           ylabel="Probability of the + path output", title="Quantum erasure reveals complementary conditional fringes")
    ax.legend(loc="upper center", ncol=1, fontsize=9)
    fig.text(.02, .008, "Orthogonal path markers measured in a complementary basis; each marker result has probability 1/2.\n"
             "Sorting by the marker record changes the subset. The unsorted path probability stays at 1/2.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_quantum_eraser_conditioning")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(examples(), indent=2))
    if not args.dry_run:
        draw_figures()


if __name__ == "__main__":
    main()
