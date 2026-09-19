"""Figures for "A Database for Analysis", drawn from database_benchmarks.json.

    python generate_database_figures.py            # both figures
    python generate_database_figures.py --dry-run  # read the results, write nothing

The benchmarks take minutes and depend on the machine, so they are run separately
(database_benchmarks.py) and their results are kept beside this file.
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

RESULTS = json.loads(Path(__file__).with_name("database_benchmarks.json").read_text(encoding="utf-8"))


def label_seconds(value):
    if value >= 1:
        return f"{value:.1f} s"
    if value >= 0.001:
        return f"{value * 1000:.0f} ms" if value >= 0.01 else f"{value * 1000:.1f} ms"
    return f"{value * 1e6:.0f} µs"


def engines_figure():
    seconds = RESULTS["seconds"]
    cpus = RESULTS["machine"]["cpus"]
    queries = list(seconds)
    series = [("sqlite", "SQLite: rows, one thread", P[0]),
              ("duckdb, 1 thread", "DuckDB: columns, one thread", P[1]),
              (f"duckdb, {cpus} threads", f"DuckDB: columns, {cpus} threads", P[2])]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    height = 0.26
    base = np.arange(len(queries))[::-1]
    for k, (key, label, colour) in enumerate(series):
        values = [seconds[q][key] for q in queries]
        ys = base + (1 - k) * height
        ax.barh(ys, values, height=height * 0.9, color=colour, label=label)
        for y, v in zip(ys, values):
            ax.text(v * 1.12, y, label_seconds(v), va="center", fontsize=8.5, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 60)
    ax.set_yticks(base)
    ax.set_yticklabels(queries)
    ax.set_xlabel("seconds, logarithmic scale")
    ax.set_title("The same five questions asked of a row store and a column store")
    ax.legend(loc="upper right")               # the top rows end well short of one second
    ax.grid(axis="y", visible=False)
    alt = ("Horizontal bars on a logarithmic time axis for five queries on five million orders. For a sum of one "
           "column, a one-day filter, a group-by and a three-table join, SQLite takes between 0.3 and 11 seconds "
           "and DuckDB between 2 and 270 milliseconds. For two thousand lookups by key the order reverses: "
           "SQLite takes 34 milliseconds and DuckDB more than a second.")
    return fig, "database_rows_versus_columns", alt


def index_figure():
    index = RESULTS["index"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [3, 2]})

    steps = index["one day"]
    names = ["no index:\nscan", "index on\nthe day", "covering\nindex"]
    values = [s["seconds"] for s in steps]
    # colour follows the access path in both panels: a scan, or a walk through an index
    ax.bar(names, values, color=[P[0], P[1], P[1]], width=0.6)
    for x, v in enumerate(values):
        ax.text(x, v * 1.25, label_seconds(v), ha="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 3)
    ax.set_ylabel("seconds, logarithmic scale")
    ax.set_title(f"One day of 730: {index['rows matched by one day']:,} rows")
    ax.grid(axis="x", visible=False)

    broad = index["a third of the table"]
    names = ["index, as the\nplanner chose", "scan,\nforced"]
    values = [broad["seconds with the chosen plan"], broad["seconds with a forced scan"]]
    bx.bar(names, values, color=[P[1], P[0]], width=0.6)
    for x, v in enumerate(values):
        bx.text(x, v * 1.25, label_seconds(v), ha="center", fontsize=9, color=hs.INK_SECONDARY)
    bx.set_yscale("log")
    bx.set_ylim(0.1, 120)
    bx.set_title("A third of the table")
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(color=P[0], label="full scan"), Patch(color=P[1], label="through an index")], loc="upper right")
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    alt = ("Two bar charts on logarithmic time axes. Left: a filter matching one day of 730 takes 335 "
           "milliseconds as a full scan, 118 with an index on the day and 0.6 with an index that also holds "
           "the summed column. Right: a filter matching a third of the table takes 27.5 seconds through the "
           "index the planner chose and half a second as a forced scan.")
    return fig, "database_index_helps_and_hurts", alt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="read the results and write nothing")
    args = parser.parse_args()
    hs.use()
    for build in (engines_figure, index_figure):
        fig, slug, alt = build()
        if args.dry_run:
            plt.close(fig)
            print(f"  would write {slug}")
            continue
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:34} {out['width']}x{out['height']}")


if __name__ == "__main__":
    main()
