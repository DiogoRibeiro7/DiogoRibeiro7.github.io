"""Benchmarks behind "A Data Lake Is a Directory With Rules".

    python data_lake_benchmarks.py --phase facts      # write the files; sizes, statistics, counts
    python data_lake_benchmarks.py --phase timings    # time the queries over those files
    python data_lake_benchmarks.py --clean            # delete the working directory

Six million synthetic events are written as CSV and as Parquet in several layouts,
all with DuckDB so that one writer produces every file, and the same questions are
asked of each. The facts do not depend on how busy the machine is; the timings do,
so they are a separate phase, refused while the machine is busy; every valid run is
kept and the least disturbed one is reported. Results accumulate in
data_lake_benchmarks.json. Peak memory is about 2 GB and the files take about 1 GB.
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

N = 6_000_000
DAYS = 365
SEED = 20260919
ROW_GROUP = 122_880
THE_DAY = 200
OUT = Path(__file__).with_name("data_lake_benchmarks.json")
WORK = Path(tempfile.gettempdir()) / "datalake_bench"

FORMATS = {
    "CSV": ("events.csv", "FORMAT csv, HEADER"),
    "Parquet, uncompressed": ("events_uncompressed.parquet", f"FORMAT parquet, COMPRESSION uncompressed, ROW_GROUP_SIZE {ROW_GROUP}"),
    "Parquet, snappy": ("events_snappy.parquet", f"FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {ROW_GROUP}"),
    "Parquet, zstd": ("events_zstd.parquet", f"FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP}"),
}
SORTED = "events_sorted_by_day.parquet"
LAYOUTS = {
    "one file": None,
    "by month": "month",
    "by day": "event_day",
    "by day and country": "event_day, country",
}
QUERIES = {
    "whole table": "SELECT sum(amount) FROM {src}",
    "one day": f"SELECT count(*), sum(amount) FROM {{src}} WHERE event_day = {THE_DAY}",
    "one country": "SELECT count(*), sum(amount) FROM {src} WHERE country = 'PT'",
}


def make_frame():
    rng = np.random.default_rng(SEED)
    countries = np.array(["PT", "ES", "FR", "DE", "UK", "IT", "NL", "BE", "SE", "PL"])
    devices = np.array(["web", "ios", "android", "store"])
    return pd.DataFrame({
        "event_id": np.arange(N, dtype=np.int64),
        "event_day": rng.integers(0, DAYS, N).astype(np.int32),      # rows arrive in no particular order
        "user_id": np.minimum((rng.pareto(1.2, N) * 5000).astype(np.int64), 499_999),
        "country": countries[rng.choice(10, N, p=[.22, .18, .15, .13, .1, .08, .05, .04, .03, .02])],
        "device": devices[rng.integers(0, 4, N)],
        "product_id": np.minimum((rng.pareto(1.1, N) * 60).astype(np.int64), 4_999),
        "quantity": rng.integers(1, 6, N).astype(np.int32),
        "amount": np.round(rng.lognormal(3.2, 0.9, N), 2),
    })


def posix(path):
    return str(path).replace("\\", "/")


def tree(path):
    files = [p for p in Path(path).rglob("*") if p.is_file()]
    return len(files), len({p.parent for p in files}), round(sum(p.stat().st_size for p in files) / 1e6, 1)


def column_sizes(path):
    meta = pq.ParquetFile(path).metadata
    sizes = {}
    for group in range(meta.num_row_groups):
        for index in range(meta.num_columns):
            column = meta.row_group(group).column(index)
            sizes[column.path_in_schema] = sizes.get(column.path_in_schema, 0) + column.total_compressed_size
    return {name: round(size / 1e6, 2) for name, size in sizes.items()}


def groups_that_may_hold(path, day):
    meta = pq.ParquetFile(path).metadata
    index = meta.schema.names.index("event_day")
    stats = [meta.row_group(g).column(index).statistics for g in range(meta.num_row_groups)]
    return sum(1 for s in stats if s.min <= day <= s.max), meta.num_row_groups


def layout_source(name):
    root = WORK / ("layout_" + name.replace(" ", "_"))
    return root, f"read_parquet('{posix(root)}/**/*.parquet', hive_partitioning = true)"


def facts(con, result):
    WORK.mkdir(parents=True, exist_ok=True)
    frame = make_frame()  # noqa: F841  (DuckDB reads the local variable)
    con.execute("CREATE OR REPLACE TABLE events AS SELECT * FROM frame")
    del frame
    sizes = {}
    for label, (name, options) in FORMATS.items():
        con.execute(f"COPY events TO '{posix(WORK / name)}' ({options})")
        sizes[label] = round((WORK / name).stat().st_size / 1e6, 1)
    con.execute(f"COPY (SELECT * FROM events ORDER BY event_day, country) TO '{posix(WORK / SORTED)}' "
                f"({FORMATS['Parquet, zstd'][1]})")
    sizes["Parquet, zstd, sorted by day"] = round((WORK / SORTED).stat().st_size / 1e6, 1)
    result["size in MB"] = sizes

    plain = WORK / FORMATS["Parquet, zstd"][0]
    result["column sizes in MB"] = {"arrival order": column_sizes(plain), "sorted by day": column_sizes(WORK / SORTED)}
    total = sum(result["column sizes in MB"]["arrival order"].values())
    result["share of the file in amount"] = round(result["column sizes in MB"]["arrival order"]["amount"] / total, 3)

    result["row groups that may hold one day"] = {}
    for label, path in (("arrival order", plain), ("sorted by day", WORK / SORTED)):
        hits, groups = groups_that_may_hold(path, THE_DAY)
        result["row groups that may hold one day"][label] = {"may hold it": hits, "of": groups}

    layouts = {}
    for name, keys in LAYOUTS.items():
        root, _ = layout_source(name)
        shutil.rmtree(root, ignore_errors=True)
        start = time.perf_counter()
        if keys is None:
            root.mkdir(parents=True)
            con.execute(f"COPY events TO '{posix(root / 'part-0.parquet')}' ({FORMATS['Parquet, zstd'][1]})")
        else:
            select = "SELECT *, event_day // 31 AS month FROM events" if keys == "month" else "SELECT * FROM events"
            con.execute(f"COPY ({select}) TO '{posix(root)}' (FORMAT parquet, COMPRESSION zstd, PARTITION_BY ({keys}))")
        files, folders, mb = tree(root)
        layouts[name] = {"files": files, "folders": folders, "MB": mb, "rows per file": round(N / files),
                         "seconds to write": round(time.perf_counter() - start, 1)}
    result["layouts"] = layouts
    result["writer threads"] = con.execute("SELECT current_setting('threads')").fetchone()[0]


def other_load(seconds=4.0):
    """CPU cores' worth of work done by other processes over a short window: all of them, and the busiest one."""
    me = os.getpid()
    before = {}
    for proc in psutil.process_iter():
        try:
            before[proc.pid] = sum(proc.cpu_times()[:2])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(seconds)
    used = [0.0]
    for proc in psutil.process_iter():
        try:
            # pid 0 is the System Idle Process on Windows: its "CPU time" is the time nobody used
            if proc.pid not in (0, me) and proc.pid in before:
                used.append((sum(proc.cpu_times()[:2]) - before[proc.pid]) / seconds)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"all other processes": round(sum(used), 2), "the busiest of them": round(max(used), 2)}


def too_busy(load, threads, batch_job):
    """Why timings taken under this load would be wrong, or None.

    A desktop in use is never idle, so the test is not for silence. It is that the
    cores this run needs are free, and that nothing which looks like a batch job is
    running: that would make the timings noisy, and the run would slow it down.
    """
    free = os.cpu_count() - load["all other processes"]
    if free < threads + 4:
        return f"other processes are using {load['all other processes']} of {os.cpu_count()} cores, which leaves too few for {threads} threads"
    if load["the busiest of them"] > batch_job:
        return f"another process is using {load['the busiest of them']} cores by itself, which looks like a batch job"
    return None


def timed_together(queries, rounds=3):
    """Median seconds of each query, keyed like the queries.

    The rounds are interleaved: a few seconds of somebody else's work then lands on
    a few runs of every query, where the median drops it, and not on every run of one.
    """
    runs = {key: [] for key in queries}
    for _ in range(rounds):
        for key, (fn, repeat) in queries.items():
            fn()  # warm-up, not counted
            for _ in range(repeat):
                start = time.perf_counter()
                fn()
                runs[key].append(time.perf_counter() - start)
    return {key: round(statistics.median(values), 4) for key, values in runs.items()}


def timings(con, result):
    run = lambda sql: (lambda: con.execute(sql).fetchall())
    csv = f"read_csv('{posix(WORK / FORMATS['CSV'][0])}')"
    plain = f"read_parquet('{posix(WORK / FORMATS['Parquet, zstd'][0])}')"
    by_day = f"read_parquet('{posix(WORK / SORTED)}')"
    every_column = ("SELECT count(*), sum(amount), avg(quantity), count(DISTINCT user_id), max(product_id), min(event_day), "
                    "max(event_id) FROM {src} WHERE device = 'web' AND country <> 'XX'")
    # the one-day question handed the day's folder, instead of being left to find it
    folder = layout_source("by day")[0] / f"event_day={THE_DAY}"
    queries = {
        ("sum of one column", "CSV"): (run(QUERIES["whole table"].format(src=csv)), 3),
        ("sum of one column", "Parquet"): (run(QUERIES["whole table"].format(src=plain)), 5),
        ("a query that needs all eight columns", "CSV"): (run(every_column.format(src=csv)), 3),
        ("a query that needs all eight columns", "Parquet"): (run(every_column.format(src=plain)), 5),
        ("one day of 365", "arrival order"): (run(QUERIES["one day"].format(src=plain)), 5),
        ("one day of 365", "sorted by day"): (run(QUERIES["one day"].format(src=by_day)), 5),
        ("one day, given its folder",): (run(f"SELECT count(*), sum(amount) FROM read_parquet('{posix(folder)}/*.parquet')"), 5),
    }
    for name in LAYOUTS:
        root, src = layout_source(name)
        repeat = 2 if name == "by day and country" else 3
        for label, sql in QUERIES.items():
            queries[("layouts", name, label)] = (run(sql.format(src=src)), repeat)
        # how much of that is finding the files: list them and do nothing else
        queries[("layouts", name, "listing the files")] = (run(f"SELECT count(*) FROM glob('{posix(root)}/**/*.parquet')"), repeat)
    seconds = {}
    for key, value in timed_together(queries).items():
        level = seconds
        for part in key[:-1]:
            level = level.setdefault(part, {})
        level[key[-1]] = value
    result["seconds"] = seconds
    result["timings machine"] = {"threads": con.execute("SELECT current_setting('threads')").fetchone()[0],
                                 "cores used by other processes, before": result.pop("_load_before"),
                                 "cores used by other processes, after": other_load()}


def flat(seconds, prefix=()):
    for key, value in seconds.items():
        if isinstance(value, dict):
            yield from flat(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def keep_least_disturbed(result):
    """Keep every valid run, and report the one with the smallest geometric mean.

    A laptop changes its clock speed with its temperature, so whole runs differ by a
    factor of two or three while the ratios inside a run barely move. Interference
    only ever adds time, so the fastest run is the closest to the machine itself,
    and the spread of each ratio across the runs says how far to trust it.
    """
    runs = result.setdefault("timing runs", [])
    runs.append({"seconds": result["seconds"], "machine": result["timings machine"]})
    best = min(runs, key=lambda r: statistics.geometric_mean([v for _, v in flat(r["seconds"])]))
    result["seconds"], result["timings machine"] = best["seconds"], best["machine"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", choices=("facts", "timings", "all"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--clean", action="store_true", help="delete the working directory and exit")
    parser.add_argument("--batch-job", type=float, default=1.5,
                        help="refuse to take timings while any other process uses more CPU cores than this")
    args = parser.parse_args()
    if args.clean:
        shutil.rmtree(WORK, ignore_errors=True)
        print(f"removed {WORK}")
        return
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    result["machine"] = {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "platform": platform.platform(), "cpus": os.cpu_count()}
    result.update({"rows": N, "days": DAYS, "row group": ROW_GROUP})
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET memory_limit = '2GB'")
    con.execute(f"SET temp_directory = '{posix(WORK / 'spill')}'")
    if args.phase in ("facts", "all"):
        facts(con, result)
    if args.phase in ("timings", "all"):
        load = other_load()
        reason = too_busy(load, args.threads, args.batch_job)
        if reason:
            sys.exit(f"no timings taken: {reason}. Try again when the machine is quieter.")
        result["_load_before"] = load
        timings(con, result)
        reason = too_busy(result["timings machine"]["cores used by other processes, after"], args.threads, args.batch_job)
        if reason:
            sys.exit(f"timings discarded, the machine became busy during the run: {reason}.")
        keep_least_disturbed(result)
    con.close()
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
