"""Figures for "A Data Lake Is a Directory With Rules", drawn from data_lake_benchmarks.json.

    python generate_data_lake_figures.py            # both figures
    python generate_data_lake_figures.py --dry-run  # read the results, write nothing

The benchmarks write several gigabytes of temporary files and take minutes, so they
run separately (data_lake_benchmarks.py) and their results are kept beside this file.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import housestyle as hs
from housestyle import PALETTE as P

RESULTS = json.loads(Path(__file__).with_name("data_lake_benchmarks.json").read_text(encoding="utf-8"))


def label_seconds(value):
    if value >= 1:
        return f"{value:.1f} s"
    return f"{value * 1000:.0f} ms" if value >= 0.01 else f"{value * 1000:.1f} ms"


def formats_figure():
    sizes = RESULTS["size in MB"]
    names = list(sizes)
    values = [sizes[n] for n in names]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [5, 6]})

    # one quantity, one hue: the same table under each encoding
    ax.barh(names[::-1], values[::-1], color=P[0], height=0.62)
    for y, v in enumerate(values[::-1]):
        ax.text(v + max(values) * 0.015, y, f"{v:.0f} MB", va="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("size on disk, MB")
    ax.set_title(f"{RESULTS['rows'] / 1e6:.0f} million events, eight columns")
    ax.grid(axis="y", visible=False)

    columns = RESULTS["column sizes in MB"]
    order = sorted(columns["arrival order"], key=columns["arrival order"].get, reverse=True)
    base = np.arange(len(order))[::-1]
    height = 0.38
    for k, (key, label, colour) in enumerate((("arrival order", "rows in arrival order", P[0]),
                                              ("sorted by day", "rows sorted by day, then country", P[1]))):
        vals = [columns[key][c] for c in order]
        ys = base + (0.5 - k) * height
        bx.barh(ys, vals, height=height * 0.92, color=colour, label=label)
        for y, v in zip(ys, vals):
            bx.text(v + 0.35, y, f"{v:.1f}" if v >= 0.1 else f"{v:.2f}", va="center", fontsize=8.5, color=hs.INK_SECONDARY)
    bx.set_yticks(base)
    bx.set_yticklabels(order)
    bx.set_xlim(0, max(max(columns[k].values()) for k in columns) * 1.18)
    bx.set_xlabel("MB of each column inside the zstd file")
    bx.set_title("Sorting moves bytes between columns")
    bx.legend(loc="lower right")
    bx.grid(axis="y", visible=False)
    fig.tight_layout()
    c = columns
    alt = (f"Two bar charts. Left: six million events take {sizes['CSV']:.0f} MB as CSV and {sizes['Parquet, uncompressed']:.0f}, "
           f"{sizes['Parquet, snappy']:.0f} and {sizes['Parquet, zstd']:.0f} MB as Parquet with no compression, snappy and zstd; "
           f"the zstd file sorted by day is slightly larger at {sizes['Parquet, zstd, sorted by day']:.0f} MB. Right: the size of each "
           f"column inside the zstd file in arrival order and after sorting by day. Sorting shrinks the day column from "
           f"{c['arrival order']['event_day']:.1f} MB to almost nothing and the country column likewise, and swells the event id "
           f"column from {c['arrival order']['event_id']:.1f} to {c['sorted by day']['event_id']:.1f} MB; the other columns do not change.")
    return fig, "data_lake_formats_and_columns", alt


def layouts_figure():
    layouts = RESULTS["layouts"]
    seconds = RESULTS["seconds"]["layouts"]
    names = list(layouts)
    series = [("whole table", "sum over the whole table", P[0]), ("one day", "one day of 365", P[1]),
              ("one country", "one country of ten", P[2]), ("listing the files", "listing the files, reading nothing", hs.INK_MUTED)]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    width = 0.2
    base = np.arange(len(names))
    for k, (key, label, colour) in enumerate(series):
        values = [seconds[n][key] for n in names]
        xs = base + (k - 1.5) * width
        ax.bar(xs, values, width=width * 0.9, color=colour, label=label)
        for x, v in zip(xs, values):
            ax.text(x, v * 1.18, label_seconds(v), ha="center", fontsize=7.5, color=hs.INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylim(top=max(max(v.values()) for v in seconds.values()) * 4)
    ax.set_xticks(base)
    count = lambda number, noun: f"{number:,} {noun}" + ("" if number == 1 else "s")
    ax.set_xticklabels([f"{n}\n{count(layouts[n]['files'], 'file')} in {count(layouts[n]['folders'], 'folder')}\n{layouts[n]['MB']:.0f} MB"
                        for n in names], fontsize=9)
    ax.set_ylabel("seconds, logarithmic scale")
    ax.set_title("The same six million rows in four layouts: the files cost more than they save")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    described = " ".join(
        f"{n.capitalize()}, {count(layouts[n]['files'], 'file')}: {label_seconds(seconds[n]['whole table'])}, "
        f"{label_seconds(seconds[n]['one day'])} and {label_seconds(seconds[n]['one country'])}, "
        f"with {label_seconds(seconds[n]['listing the files'])} to list the files." for n in names)
    alt = ("Grouped bars on a logarithmic time axis: the time to sum the whole table, to query one day and to query one "
           f"country, and the time to list the files alone, for the same six million events in four layouts. {described}")
    return fig, "data_lake_partition_layouts", alt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="read the results and write nothing")
    args = parser.parse_args()
    hs.use()
    for build in (formats_figure, layouts_figure):
        fig, slug, alt = build()
        if args.dry_run:
            plt.close(fig)
            print(f"  would write {slug}")
            continue
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:34} {out['width']}x{out['height']}")


if __name__ == "__main__":
    main()
