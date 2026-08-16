"""House chart style helpers for figures embedded in blog posts.

Import this before plotting so every figure on the site shares one palette,
one set of mark specs, and one export path.

    from housestyle import use, save, PALETTE, SEQUENTIAL, DIVERGING

    fig, ax = plt.subplots()
    ax.plot(x, y, color=PALETTE[0], label="Observed")
    ax.legend()
    save(fig, "arima_forecast")

Palette provenance: validated with the dataviz skill's validator against the
site's real surface (#f7f9fa). Do not hand-edit the hexes; if the palette must
change, re-run the validator and update house.mplstyle in step.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE / "house.mplstyle"

# Where generated figures land, and how posts reference them.
FIGDIR = HERE.parent / "images" / "figures"
URL_PREFIX = "/assets/images/figures"

SURFACE = "#f7f9fa"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed slot order. Never cycle past slot 8: fold into "Other" or facet.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Scatter/bubble put every pair on screen at once; only these three validate
# under --pairs all. Cap those forms at three series.
PALETTE_ALLPAIRS = PALETTE[:3]

# Single hue, light -> dark, for continuous magnitude.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#256abf", "#184f95", "#0d366b"]

# Two poles with a neutral gray midpoint; never a hue at the middle.
DIVERGING = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
             "#f0a3a3", "#e34948", "#a82725", "#6b1614"]

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


def use():
    """Activate the house style. Call once before creating figures."""
    plt.style.use(str(STYLE))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


def sequential_cmap(name="house_seq"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def diverging_cmap(name="house_div"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIVERGING)


def label_end(ax, x, y, text, color, dx=6, dy=0):
    """Direct-label a series at its final point.

    Label selectively - the endpoint, the extreme, or the one series the story
    is about. Never a value on every point.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", va="center",
                color=INK_SECONDARY, fontsize=9.5)
    ax.plot([x], [y], marker="o", markersize=5, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)


def save(fig, slug, alt=None):
    """Write the figure as PNG and return the markdown to paste into a post."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{slug}.png"
    fig.savefig(out)
    plt.close(fig)
    w, h = fig.get_size_inches() * fig.dpi
    md = (f'![{alt or slug}]({URL_PREFIX}/{slug}.png)'
          f'{{: width="{int(w)}" height="{int(h)}" loading="lazy"}}')
    return {"path": str(out), "markdown": md, "width": int(w), "height": int(h)}
