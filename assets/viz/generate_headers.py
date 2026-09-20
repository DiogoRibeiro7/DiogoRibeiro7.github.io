"""Generate original header images for posts.

Every image in assets/images/headers is produced here, so the pool can be
regenerated or extended without licensing questions. Run from this directory:

    python generate_headers.py             # all headers
    python generate_headers.py walks grid  # named headers only

Each generator draws one 16:9 composition on a dark ground in the house
palette, so a white title reads over it in the post hero. Files are written
as JPEG at 1600 x 900.
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from housestyle import PALETTE as P, SEQUENTIAL

HEADERS = {}
OUTDIR = Path(__file__).resolve().parent.parent / "images" / "headers"
W, H, DPI = 16, 9, 100          # 1600 x 900 pixels
GROUND = "#101a2e"              # dark navy ground shared by every header
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#16223b"] + SEQUENTIAL[1:] + ["#f4f8ff"])
DEEP_CMAP = LinearSegmentedColormap.from_list("deep", ["#101a2e", "#15305c", "#1f4f94", "#2a78d6", "#5a98e2"])


def header(slug):
    def deco(fn):
        HEADERS[slug] = fn
        return fn
    return deco


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=GROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(GROUND)
    ax.set_axis_off()
    return fig, ax


# --------------------------------------------------------------------------
@header("walks")
def walks():
    r = np.random.default_rng(1)
    fig, ax = canvas()
    n, steps = 48, 400
    x = np.linspace(0, W, steps)
    for i in range(n):
        y = np.cumsum(r.normal(0, 0.12, steps)) + H / 2 + r.normal(0, 1.2)
        ax.plot(x, y, color=P[i % 4], lw=1.4, alpha=0.55)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("field")
def field():
    r = np.random.default_rng(2)
    fig, ax = canvas()
    xs, ys = np.meshgrid(np.linspace(0, W, 400), np.linspace(0, H, 225))
    z = np.zeros_like(xs)
    for _ in range(14):
        cx, cy, s, a = r.uniform(0, W), r.uniform(0, H), r.uniform(1.5, 4), r.uniform(-1, 1)
        z += a * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s ** 2))
    ax.contourf(xs, ys, z, levels=18, cmap=DEEP_CMAP, alpha=0.95)
    ax.contour(xs, ys, z, levels=18, colors="#0b1426", linewidths=0.6, alpha=0.6)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("constellation")
def constellation():
    r = np.random.default_rng(3)
    fig, ax = canvas()
    centres = r.uniform([1, 1], [W - 1, H - 1], (7, 2))
    pts, cols = [], []
    for k, c in enumerate(centres):
        m = r.integers(40, 90)
        pts.append(r.normal(c, [1.4, 0.9], (m, 2))); cols += [P[k % len(P)]] * m
    pts = np.vstack(pts)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for i, j in tree.query_pairs(0.9):
        ax.plot(pts[[i, j], 0], pts[[i, j], 1], color="#6f8bb8", lw=0.5, alpha=0.35)
    ax.scatter(pts[:, 0], pts[:, 1], s=26, c=cols, alpha=0.85, edgecolors="none")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("waves")
def waves():
    r = np.random.default_rng(4)
    fig, ax = canvas()
    x = np.linspace(0, W, 1200)
    for i in range(36):
        f, ph, amp, off = r.uniform(0.3, 1.4), r.uniform(0, 6.3), r.uniform(0.4, 1.6), r.uniform(0.5, H - 0.5)
        y = off + amp * np.sin(f * x + ph) * np.sin(0.25 * x + ph / 2)
        ax.plot(x, y, color=P[i % 3], lw=1.1, alpha=0.5)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("network")
def network():
    r = np.random.default_rng(5)
    fig, ax = canvas()
    pts = r.uniform([0.5, 0.5], [W - 0.5, H - 0.5], (140, 2))
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    deg = np.zeros(len(pts))
    for i, j in tree.query_pairs(1.7):
        ax.plot(pts[[i, j], 0], pts[[i, j], 1], color="#4f6b9e", lw=0.7, alpha=0.5)
        deg[i] += 1; deg[j] += 1
    ax.scatter(pts[:, 0], pts[:, 1], s=18 + 9 * deg, c=[P[int(d) % 4] for d in deg], alpha=0.9, edgecolors=GROUND, linewidths=0.8)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("blocks")
def blocks():
    r = np.random.default_rng(6)
    fig, ax = canvas()
    nx, ny = 32, 18
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    z = np.sin(xs / 4.0) * np.cos(ys / 3.0) + 0.6 * r.normal(0, 1, (ny, nx))
    ax.imshow(z, cmap=DEEP_CMAP, extent=[0, W, 0, H], interpolation="nearest", alpha=0.95)
    for i in range(nx + 1):
        ax.axvline(i * W / nx, color=GROUND, lw=1.2)
    for j in range(ny + 1):
        ax.axhline(j * H / ny, color=GROUND, lw=1.2)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("spiral")
def spiral():
    fig, ax = canvas()
    n = 1400
    i = np.arange(n)
    theta = i * np.pi * (3 - np.sqrt(5))
    rad = 0.19 * np.sqrt(i)
    x, y = W / 2 + rad * np.cos(theta), H / 2 + rad * np.sin(theta)
    ax.scatter(x, y, s=8 + 0.03 * i, c=i, cmap=SEQ_CMAP, alpha=0.9, edgecolors="none")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("lissajous")
def lissajous():
    r = np.random.default_rng(8)
    fig, ax = canvas()
    t = np.linspace(0, 2 * np.pi, 4000)
    for k in range(9):
        a, b, d = r.integers(2, 7), r.integers(2, 7), r.uniform(0, np.pi)
        ax.plot(W / 2 + 6.5 * np.sin(a * t + d), H / 2 + 3.6 * np.sin(b * t), color=P[k % len(P)], lw=0.9, alpha=0.45)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("skyline")
def skyline():
    r = np.random.default_rng(9)
    fig, ax = canvas()
    x = np.linspace(0, W, 90)
    for layer, (mu, sd, col) in enumerate([(5, 3.2, P[0]), (9, 2.4, P[1]), (12, 1.6, P[2])]):
        y = 0.6 * H * np.exp(-((x - mu) ** 2) / (2 * sd ** 2)) * (1 + 0.25 * r.normal(0, 1, len(x)))
        ax.bar(x, np.clip(y, 0, None), width=W / 90 * 0.8, bottom=0, color=col, alpha=0.55 - 0.1 * layer)
    xx = np.linspace(0, W, 600)
    ax.plot(xx, 0.62 * H * np.exp(-((xx - 8) ** 2) / (2 * 3.5 ** 2)), color="#f4f8ff", lw=2, alpha=0.7)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("sparklines")
def sparklines():
    r = np.random.default_rng(10)
    fig, ax = canvas()
    cols, rows = 8, 5
    for i in range(cols):
        for j in range(rows):
            x0, y0 = i * W / cols + 0.25, j * H / rows + 0.3
            x = np.linspace(x0, x0 + W / cols - 0.5, 60)
            y = y0 + 0.45 * H / rows * (0.5 + 0.5 * np.cumsum(r.normal(0, 0.18, 60)).clip(-1, 1))
            ax.plot(x, y, color=P[(i + j) % 4], lw=1.5, alpha=0.8)
            ax.plot([x0, x0 + W / cols - 0.5], [y0, y0], color="#33456a", lw=0.8)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("cells")
def cells():
    r = np.random.default_rng(11)
    fig, ax = canvas()
    from scipy.spatial import Voronoi
    pts = r.uniform([-2, -2], [W + 2, H + 2], (90, 2))
    vor = Voronoi(pts)
    for k, region in enumerate(vor.regions):
        if not region or -1 in region:
            continue
        poly = vor.vertices[region]
        ax.fill(poly[:, 0], poly[:, 1], color=P[k % len(P)], alpha=0.22 + 0.05 * (k % 3), edgecolor=GROUND, linewidth=2.2)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("noise")
def noise():
    r = np.random.default_rng(12)
    fig, ax = canvas()
    xs, ys = np.meshgrid(np.linspace(0, 1, 640), np.linspace(0, 1, 360))
    z = np.zeros_like(xs)
    for k in range(1, 7):
        fx, fy, ph = r.uniform(1, 4) * k, r.uniform(1, 4) * k, r.uniform(0, 6.3)
        z += np.sin(2 * np.pi * (fx * xs + fy * ys) + ph) / k
    ax.imshow(z, cmap=DEEP_CMAP, extent=[0, W, 0, H], interpolation="bilinear")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


# --------------------------------------------------------------------------
# Essay headers. The nine below are not abstract textures: each one draws the
# argument of a single article, using that article's own published figures, so
# the piece gets a hero nobody else on the site shares. No text is drawn, since
# the theme lays the title over the image.


def logx(value, lo, hi):
    """Map a positive quantity onto the 0..W canvas by its logarithm."""
    return W * (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))


@header("science-based-brand")
def science_based_brand():
    """What a body of evidence carries, and how far the claim built on it goes.

    Each row is one claim. The blue length is the part the evidence supports;
    the orange overhang is the part supplied by the language around it. The
    essay is about that overhang rather than about who is speaking.
    """
    r = np.random.default_rng(21)
    fig, ax = canvas()
    rows, x0 = 13, 1.15
    for k in range(rows):
        y = 0.7 + k * (H - 1.4) / (rows - 1)
        evidence = r.uniform(1.0, 4.4)
        claim = min(evidence + abs(r.normal(0, 3.6)) ** 1.1 + 0.5, W - x0 - 0.7)
        ax.plot([x0, x0 + claim], [y, y], color=P[1], lw=20, alpha=0.88,
                solid_capstyle="butt", zorder=2)
        ax.plot([x0, x0 + evidence], [y, y], color=P[0], lw=20, alpha=1.0,
                solid_capstyle="butt", zorder=3)
    ax.axvline(x0, color="#8fa6cd", lw=1.8, alpha=0.85, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("inflammation-marker")
def inflammation_marker():
    """C-reactive protein over four orders of magnitude, with its own noise.

    Each row is one person measured six times. The shaded band is the range a
    single result can take when the usual level is 3 mg/L, and it straddles
    the cut-off drawn through it; the red points are the same people during an
    acute episode, which is where the marker really does carry information.
    """
    r = np.random.default_rng(22)
    fig, ax = canvas()
    lo, hi = 0.05, 1500.0

    for decade in [0.1, 1, 10, 100, 1000]:          # the four orders of magnitude
        ax.axvline(logx(decade, lo, hi), color="#2b3c5f", lw=1.0, alpha=0.9, zorder=1)

    band = (0.9, 11.0)                              # a single result from a usual level of 3 mg/L
    ax.axvspan(logx(band[0], lo, hi), logx(band[1], lo, hi),
               color=P[0], alpha=0.11, zorder=1)
    for edge in band:
        ax.axvline(logx(edge, lo, hi), color=P[0], lw=1.2, alpha=0.4, zorder=2)
    ax.axvline(logx(3.0, lo, hi), color="#f4f8ff", lw=1.5, alpha=0.7,
               ls=(0, (5, 4)), zorder=5)            # the cut-off used for risk

    for k in range(11):                             # eleven people, six results each
        y = 0.75 + k * (H - 1.5) / 10
        usual = 10 ** r.uniform(np.log10(0.4), np.log10(6.0))
        reps = list(usual * np.exp(r.normal(0, 0.62, 6)))
        acute = k in (1, 4, 8)                      # three of them are ill on one day
        if acute:
            reps[-1] = 10 ** r.uniform(np.log10(150), np.log10(900))
        xs = logx(np.clip(reps, lo, hi), lo, hi)
        order = np.argsort(xs)
        ax.plot(xs[order], [y] * 6, color="#5f7bab", lw=1.1, alpha=0.6, zorder=2)
        ax.scatter(xs[:-1], [y] * 5, s=66, color=P[0], alpha=0.9,
                   edgecolors=GROUND, linewidths=1.0, zorder=3)
        ax.scatter([xs[-1]], [y], s=140 if acute else 66,
                   color=P[7] if acute else P[0], alpha=0.95,
                   edgecolors=GROUND, linewidths=1.1, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("aspartame-fruit-dose")
def aspartame_fruit_dose():
    """Methanol by source on one logarithmic axis, from a can to a lethal dose.

    The essay's argument is that the comparison with fruit only becomes a
    safety argument once the doses are written down, so the header is the dose
    axis: milligrams from each source, published ranges, same scale.
    """
    r = np.random.default_rng(23)
    fig, ax = canvas()
    lo, hi = 1.0, 1e5
    sources = [                     # milligrams of methanol, published ranges
        (3, 20, P[0]),              # one can of diet drink
        (3, 160, P[0]),             # a 250 mL glass of fruit juice
        (280, 280, P[3]),           # the acceptable daily intake at 70 kg
        (300, 600, P[2]),           # made by the body in a day
        (400, 1400, P[2]),          # the pectin in 1 kg of apples
        (21000, 70000, P[7]),       # minimum lethal dose at 70 kg
    ]
    for decade in [10, 100, 1000, 10000]:
        ax.axvline(logx(decade, lo, hi), color="#2b3c5f", lw=1.1, alpha=0.9, zorder=1)

    for k, (low, high, colour) in enumerate(sources):
        y = H - 0.95 - k * (H - 1.9) / (len(sources) - 1)
        ax.plot([0.35, W - 0.35], [y, y], color="#24375c", lw=1.4, alpha=0.95, zorder=1)
        for decade in [10, 100, 1000, 10000]:       # a tick on every row's own scale
            ax.plot([logx(decade, lo, hi)] * 2, [y - 0.16, y + 0.16],
                    color="#33486f", lw=1.4, alpha=0.95, zorder=1)
        x0, x1 = logx(low, lo, hi), logx(high, lo, hi)
        if x1 - x0 < 0.25:                          # a point value still needs a mark
            x0, x1 = x0 - 0.12, x1 + 0.12
        ax.plot([x0, x1], [y, y], color=colour, lw=30, alpha=0.2,
                solid_capstyle="round", zorder=2)
        doses = r.uniform(x0, x1, 70)               # the range as the doses inside it
        ax.scatter(doses, y + r.normal(0, 0.13, 70), s=r.uniform(14, 58, 70),
                   color=colour, alpha=0.62, edgecolors="none", zorder=3)
        ax.plot([x0, x1], [y, y], color=colour, lw=5.0, alpha=0.95,
                solid_capstyle="round", zorder=4)
        ax.scatter([x0, x1], [y, y], s=150, color=colour, alpha=1.0,
                   edgecolors=GROUND, linewidths=1.6, zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("results-rhetoric")
def results_rhetoric():
    """Every client as a dot, and the best twenty lifted out onto a wall.

    Selection is the whole essay: the wall on the right is drawn from the
    extreme tail of the distribution on the left, and grows more impressive
    the more clients the business has had.
    """
    r = np.random.default_rng(24)
    fig, ax = canvas()
    n, bins = 640, 46
    values = r.normal(0, 1, n)
    edges = np.linspace(-3.4, 3.4, bins + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    cut = np.sort(values)[-20]
    spacing = (H - 1.4) / max(np.bincount(idx, minlength=bins).max(), 1)

    seen = np.zeros(bins, dtype=int)
    chosen = []
    for value, b in zip(values, idx):
        x = 0.6 + (b + 0.5) * (9.0 - 0.6) / bins
        y = 0.7 + seen[b] * spacing
        seen[b] += 1
        if value >= cut:
            ax.scatter([x], [y], s=52, color=P[1], alpha=1.0,
                       edgecolors=GROUND, linewidths=0.8, zorder=4)
            chosen.append((x, y))
        else:
            ax.scatter([x], [y], s=28, color="#4f6b9e", alpha=0.7,
                       edgecolors="none", zorder=2)

    tiles = [(10.8 + c * 1.05, 6.9 - row * 1.45) for row in range(4) for c in range(5)]
    for (x, y), (tx, ty) in zip(chosen, tiles):
        ax.plot([x, tx - 0.36], [y, ty], color=P[1], lw=0.8, alpha=0.32, zorder=1)
    for tx, ty in tiles:
        ax.add_patch(plt.Rectangle((tx - 0.36, ty - 0.5), 0.72, 1.0,
                                   color=P[1], alpha=0.88, ec=GROUND, lw=1.4, zorder=3))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("scientific-provenance")
def scientific_provenance():
    """A claim's chain back to its evidence, thinning at every step.

    Observations feed analyses, analyses feed syntheses, syntheses feed the
    public claim on the right. The edges fade as the chain travels, which is
    what the essay means by provenance disappearing while authority remains.
    """
    r = np.random.default_rng(25)
    fig, ax = canvas()
    layers = [22, 11, 5, 1]
    cols = [1.0, 5.3, 9.6, 13.6]
    pos = []
    for count, cx in zip(layers, cols):
        ys = np.linspace(0.6, H - 0.6, count) if count > 1 else np.array([H / 2])
        pos.append([(cx + r.normal(0, 0.16), y) for y in ys])

    for depth, (left, right) in enumerate(zip(pos, pos[1:])):
        alpha = 0.5 - 0.145 * depth              # the trail dims as it travels
        for j, (rx, ry) in enumerate(right):
            parents = [left[i] for i in range(len(left))
                       if abs(i * len(right) / len(left) - j) < 1.25]
            for lx, ly in parents or [left[min(j, len(left) - 1)]]:
                mid = (lx + rx) / 2
                t = np.linspace(0, 1, 40)
                bx = (1 - t) ** 2 * lx + 2 * (1 - t) * t * mid + t ** 2 * rx
                by = (1 - t) ** 2 * ly + 2 * (1 - t) * t * ((ly + ry) / 2) + t ** 2 * ry
                ax.plot(bx, by, color="#6f8bb8", lw=1.0, alpha=alpha, zorder=1)

    for depth, nodes in enumerate(pos):
        xs = [p[0] for p in nodes]
        ys = [p[1] for p in nodes]
        ax.scatter(xs, ys, s=[80, 170, 320, 820][depth], color=P[[0, 0, 2, 1][depth]],
                   alpha=0.95, edgecolors=GROUND, linewidths=1.2, zorder=3)
    ax.scatter([pos[-1][0][0]], [pos[-1][0][1]], s=5200, color=P[1],
               alpha=0.15, edgecolors="none", zorder=2)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("one-study-not-enough")
def one_study_not_enough():
    """Sixty runs of the essay's own low-power experiment.

    Twenty per arm and a true effect of 0.3 standard deviations: most intervals
    cover nothing, and the few that reach significance sit well to the right of
    the truth. That gap is the effect inflation the article computes.
    """
    r = np.random.default_rng(26)
    fig, ax = canvas()
    n_studies, true_effect, se = 60, 0.3, np.sqrt(2 / 20)
    estimates = r.normal(true_effect, se, n_studies)
    lo, hi = -1.5, 2.1

    def mx(v):
        return 0.7 + (np.clip(v, lo, hi) - lo) * (W - 1.4) / (hi - lo)

    ax.axvline(mx(0.0), color="#f4f8ff", lw=1.5, alpha=0.55, zorder=4)
    ax.axvline(mx(true_effect), color=P[2], lw=1.5, alpha=0.8,
               ls=(0, (5, 4)), zorder=4)
    for k, est in enumerate(estimates):
        y = 0.45 + k * (H - 0.9) / (n_studies - 1)
        low, high = est - 1.96 * se, est + 1.96 * se
        significant = low > 0
        colour = P[1] if significant else "#546d9f"
        ax.plot([mx(low), mx(high)], [y, y], color=colour,
                lw=2.8 if significant else 1.8,
                alpha=0.95 if significant else 0.68, solid_capstyle="round", zorder=2)
        ax.scatter([mx(est)], [y], s=56 if significant else 30, color=colour,
                   alpha=0.95 if significant else 0.75,
                   edgecolors=GROUND, linewidths=0.8, zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("study-design-question")
def study_design_question():
    """Five designs side by side rather than stacked into a ladder.

    Randomised allocation, a cohort followed forward, a case-control study
    looking back, a dose-response experiment and a pooled synthesis: each
    panel answers a different question, which is the essay's whole claim.
    """
    r = np.random.default_rng(27)
    fig, ax = canvas()
    panels = 5
    width = W / panels
    for k in range(1, panels):
        ax.axvline(k * width, color="#2b3c5f", lw=1.2, alpha=0.9, zorder=1)

    def centre(k):
        return k * width + width / 2

    # 1. randomised allocation: one node splitting into two arms
    c = centre(0)
    ax.scatter([c - 1.0], [H / 2], s=380, color=P[3], alpha=0.95,
               edgecolors=GROUND, linewidths=1.4, zorder=4)
    for sign, colour in ((1, P[0]), (-1, P[1])):
        ax.plot([c - 1.0, c + 0.15], [H / 2, H / 2 + sign * 2.0],
                color=colour, lw=2.4, alpha=0.85, zorder=2)
        ys = H / 2 + sign * 2.0 + np.tile([-0.55, 0.0, 0.55], 3)
        xs = c + 0.15 + np.repeat([0.0, 0.45, 0.9], 3)
        ax.scatter(xs, ys, s=64, color=colour, alpha=0.9,
                   edgecolors=GROUND, linewidths=0.8, zorder=3)

    # 2. a cohort followed forward in time
    c = centre(1)
    for j in range(7):
        y = 1.3 + j * (H - 2.6) / 6
        ax.annotate("", xy=(c + 1.1, y), xytext=(c - 1.1, y),
                    arrowprops=dict(arrowstyle="-|>", color=P[0], lw=1.6, alpha=0.8))
        ax.scatter([c - 1.1], [y], s=62, color=P[0], alpha=0.9,
                   edgecolors=GROUND, linewidths=0.8, zorder=3)

    # 3. case-control: start from the outcome and look back
    c = centre(2)
    for j in range(7):
        y = 1.3 + j * (H - 2.6) / 6
        colour = P[7] if j % 2 else P[6]
        ax.annotate("", xy=(c - 1.1, y), xytext=(c + 1.1, y),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6, alpha=0.8))
        ax.scatter([c + 1.1], [y], s=62, color=colour, alpha=0.9,
                   edgecolors=GROUND, linewidths=0.8, zorder=3)

    # 4. a mechanistic dose-response curve
    c = centre(3)
    x = np.linspace(c - 1.15, c + 1.15, 200)
    y = H / 2 + 2.4 / (1 + np.exp(-3.2 * (x - c))) - 1.2
    ax.plot(x, y, color=P[2], lw=2.8, alpha=0.95, zorder=3)
    dx = np.linspace(c - 1.0, c + 1.0, 7)
    dy = H / 2 + 2.4 / (1 + np.exp(-3.2 * (dx - c))) - 1.2 + r.normal(0, 0.17, 7)
    ax.scatter(dx, dy, s=76, color=P[2], alpha=0.9, edgecolors=GROUND, linewidths=0.9, zorder=4)

    # 5. a pooled synthesis: intervals and a diamond
    c = centre(4)
    for j, (est, half) in enumerate([(-0.45, 0.62), (0.2, 0.5), (0.5, 0.7), (-0.1, 0.42)]):
        y = H - 1.5 - j * 1.35
        ax.plot([c + est - half, c + est + half], [y, y], color=P[0], lw=3.0,
                alpha=0.9, solid_capstyle="round", zorder=2)
        ax.scatter([c + est], [y], s=90, marker="s", color=P[0], alpha=0.95,
                   edgecolors=GROUND, linewidths=0.8, zorder=3)
    ax.add_patch(plt.Polygon([(c - 0.7, 1.95), (c + 0.05, 2.35),
                              (c + 0.8, 1.95), (c + 0.05, 1.55)],
                             color=P[1], alpha=0.95, ec=GROUND, lw=1.2, zorder=3))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("meta-analysis-pooling")
def meta_analysis_pooling():
    """The essay's five-study example drawn as the forest plot it describes.

    Effects of 0.10 to 0.90 at a common standard error of 0.10 pool to 0.43.
    The narrow diamond is the common-effect interval, the bar below it the
    random-effects interval, and the widest bar the prediction interval, which
    crosses zero.
    """
    fig, ax = canvas()
    effects = [0.10, 0.15, 0.20, 0.80, 0.90]
    se = 0.10
    lo, hi = -1.05, 1.9

    def mx(v):
        return 0.8 + (v - lo) * (W - 1.6) / (hi - lo)

    ax.axvline(mx(0.0), color="#f4f8ff", lw=1.6, alpha=0.6, zorder=4)
    for k, est in enumerate(effects):
        y = H - 0.9 - k * 1.05
        ax.plot([mx(est - 1.96 * se), mx(est + 1.96 * se)], [y, y],
                color=P[0], lw=3.2, alpha=0.9, solid_capstyle="round", zorder=2)
        ax.scatter([mx(est)], [y], s=200, marker="s", color=P[0], alpha=0.95,
                   edgecolors=GROUND, linewidths=1.0, zorder=3)

    pooled, pooled_se = 0.43, se / np.sqrt(len(effects))
    ax.axvline(mx(pooled), color=P[1], lw=1.2, alpha=0.35, ls=(0, (4, 4)), zorder=1)
    y = 3.05
    ax.add_patch(plt.Polygon([(mx(pooled - 1.96 * pooled_se), y), (mx(pooled), y + 0.5),
                              (mx(pooled + 1.96 * pooled_se), y), (mx(pooled), y - 0.5)],
                             color=P[1], alpha=0.95, ec=GROUND, lw=1.2, zorder=4))
    for y, (low, high), colour in [(1.95, (0.09, 0.77), P[3]),
                                   (0.95, (-0.88, 1.74), P[7])]:
        ax.plot([mx(low), mx(high)], [y, y], color=colour, lw=6.0, alpha=0.9,
                solid_capstyle="butt", zorder=3)
        for edge in (low, high):                 # end caps on the wider intervals
            ax.plot([mx(edge), mx(edge)], [y - 0.38, y + 0.38], color=colour,
                    lw=3.0, alpha=0.95, zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


@header("preregistration-protocol")
def preregistration_protocol():
    """The essay's analysis space: four by three by two by two, forty-eight ends.

    One outcome, one covariate set, one transformation and one exclusion rule
    are lit up as the path written down in advance; the other forty-seven stay
    dim, available and unrecorded.
    """
    fig, ax = canvas()
    branching = [4, 3, 2, 2]
    cols = np.linspace(0.9, W - 1.4, len(branching) + 1)
    levels = [[H / 2]]
    for count in branching:
        prev = levels[-1]
        span = (H - 1.0) / len(prev)          # the room this parent's children share
        nxt = []
        for parent in prev:
            offs = (np.arange(count) - (count - 1) / 2) * span / count
            nxt += list(parent + offs * 0.92)
        levels.append(nxt)

    chosen = {0: 0}                       # the path a protocol pins down in advance
    for depth, count in enumerate(branching):
        chosen[depth + 1] = chosen[depth] * count + (1 if count > 1 else 0)

    for depth, count in enumerate(branching):
        left, right = levels[depth], levels[depth + 1]
        for j, ry in enumerate(right):
            i = j // count
            lit = (i == chosen[depth] and j == chosen[depth + 1])
            t = np.linspace(0, 1, 30)
            mid = (cols[depth] + cols[depth + 1]) / 2
            bx = (1 - t) ** 2 * cols[depth] + 2 * (1 - t) * t * mid + t ** 2 * cols[depth + 1]
            by = (1 - t) ** 2 * left[i] + 2 * (1 - t) * t * ((left[i] + ry) / 2) + t ** 2 * ry
            ax.plot(bx, by, color=P[1] if lit else "#4a6096",
                    lw=3.0 if lit else 1.0, alpha=0.95 if lit else 0.6,
                    zorder=3 if lit else 1)

    for depth, ys in enumerate(levels):
        size = [280, 180, 110, 64, 40][depth]
        for j, y in enumerate(ys):
            lit = j == chosen[depth]
            ax.scatter([cols[depth]], [y], s=size * (1.7 if lit else 1.0),
                       color=P[1] if lit else P[0], alpha=0.95 if lit else 0.75,
                       edgecolors=GROUND, linewidths=1.0, zorder=4 if lit else 2)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig


def save(fig, slug):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=GROUND)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    out = OUTDIR / f"{slug}.jpg"
    img.save(out, "JPEG", quality=82, optimize=True, progressive=True)
    return out, img.size, out.stat().st_size


def main(names):
    chosen = names or sorted(HEADERS)
    for slug in chosen:
        if slug not in HEADERS:
            print(f"  ?? unknown header: {slug}")
            continue
        out, size, nbytes = save(HEADERS[slug](), slug)
        print(f"  wrote {out.name:18} {size[0]}x{size[1]}  {nbytes // 1024} KB")
    print(f"\n{len(chosen)} headers -> {OUTDIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
