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
