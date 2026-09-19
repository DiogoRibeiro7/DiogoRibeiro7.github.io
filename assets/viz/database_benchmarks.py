"""Benchmarks behind "A Database for Analysis": one data set, a row store and a column store.

    python database_benchmarks.py        # about five minutes; writes database_benchmarks.json

Five million synthetic orders with customer and product tables are loaded into SQLite
(rows, B-trees, one thread) and DuckDB (columns, vectorised, one thread and all of
them). Timings are medians of repeated runs on one machine. Read them for their
ratios, which are stable, not for their absolute values, which are not.
"""

import json
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

N_ORDERS = 5_000_000
N_CUSTOMERS = 200_000
N_PRODUCTS = 5_000
SEED = 20260919
OUT = Path(__file__).with_name("database_benchmarks.json")

QUERIES = {
    "sum of one column": "SELECT sum(amount) FROM orders",
    "one day of 730": "SELECT count(*), sum(amount) FROM orders WHERE order_day = 365",
    "group by channel": "SELECT channel, count(*), avg(amount) FROM orders GROUP BY channel",
    "star join": ("SELECT c.country, p.category, sum(o.amount) FROM orders o "
                  "JOIN customers c ON c.customer_id = o.customer_id "
                  "JOIN products p ON p.product_id = o.product_id "
                  "WHERE o.status = 'paid' GROUP BY c.country, p.category"),
}
ONE_DAY = QUERIES["one day of 730"]
BROAD = "SELECT count(*), sum(quantity) FROM orders WHERE order_day < 243"   # a third of the table


def make_data():
    rng = np.random.default_rng(SEED)
    countries = np.array(["PT", "ES", "FR", "DE", "UK", "IT", "NL", "BE", "SE", "PL"])
    segments = np.array(["consumer", "business", "public"])
    categories = np.array([f"cat_{k:02d}" for k in range(40)])
    customers = {
        "customer_id": np.arange(N_CUSTOMERS, dtype=np.int64),
        "country": countries[rng.integers(0, len(countries), N_CUSTOMERS)],
        "segment": segments[rng.choice(3, N_CUSTOMERS, p=[0.7, 0.25, 0.05])],
        "signup_day": rng.integers(0, 1500, N_CUSTOMERS).astype(np.int64),
    }
    products = {
        "product_id": np.arange(N_PRODUCTS, dtype=np.int64),
        "category": categories[rng.integers(0, len(categories), N_PRODUCTS)],
        "list_price": np.round(rng.lognormal(3.0, 0.8, N_PRODUCTS), 2),
    }
    # a few customers and products account for most orders, as they do in practice
    cust = np.minimum((rng.pareto(1.2, N_ORDERS) * 4000).astype(np.int64), N_CUSTOMERS - 1)
    prod = np.minimum((rng.pareto(1.1, N_ORDERS) * 60).astype(np.int64), N_PRODUCTS - 1)
    qty = rng.integers(1, 6, N_ORDERS).astype(np.int64)
    orders = {
        "order_id": np.arange(N_ORDERS, dtype=np.int64),
        "customer_id": cust,
        "product_id": prod,
        "order_day": rng.integers(0, 730, N_ORDERS).astype(np.int64),
        "quantity": qty,
        "amount": np.round(products["list_price"][prod] * qty * rng.uniform(0.8, 1.0, N_ORDERS), 2),
        "status": np.array(["paid", "refunded", "cancelled"])[rng.choice(3, N_ORDERS, p=[0.93, 0.04, 0.03])],
        "channel": np.array(["web", "app", "store", "phone"])[rng.integers(0, 4, N_ORDERS)],
    }
    return customers, products, orders


def timed(fn, repeat=5, warmup=1):
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return round(statistics.median(runs), 4)


def as_rows(table):
    return list(zip(*(table[c].tolist() for c in table)))


def main():
    work = Path(tempfile.mkdtemp(prefix="dbbench_"))
    customers, products, orders = make_data()
    result = {
        "machine": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                    "duckdb": duckdb.__version__, "platform": platform.platform(), "cpus": os.cpu_count()},
        "rows": {"orders": N_ORDERS, "customers": N_CUSTOMERS, "products": N_PRODUCTS},
    }

    # ------------------------------------------------------------ SQLite: rows and B-trees
    lite_path = work / "shop.sqlite"
    lite = sqlite3.connect(lite_path)
    lite.executescript("""
        PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, country TEXT, segment TEXT, signup_day INTEGER);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, category TEXT, list_price REAL);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                             order_day INTEGER, quantity INTEGER, amount REAL, status TEXT, channel TEXT);
    """)
    with lite:
        lite.executemany("INSERT INTO customers VALUES (?,?,?,?)", as_rows(customers))
        lite.executemany("INSERT INTO products VALUES (?,?,?)", as_rows(products))
        lite.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", as_rows(orders))
    lite.execute("ANALYZE")
    size = {"sqlite": round(lite_path.stat().st_size / 1e6, 1)}

    # ------------------------------------------------------------ DuckDB: columns
    duck_path = work / "shop.duckdb"
    duck = duckdb.connect(str(duck_path))
    duck.execute("CREATE TABLE orders (order_id BIGINT PRIMARY KEY, customer_id BIGINT, product_id BIGINT, "
                 "order_day BIGINT, quantity BIGINT, amount DOUBLE, status VARCHAR, channel VARCHAR)")
    for name, table in (("customers", customers), ("products", products), ("orders", orders)):
        frame = pd.DataFrame(table)  # noqa: F841  (DuckDB reads the local variable)
        duck.execute(f"INSERT INTO {name} SELECT * FROM frame" if name == "orders" else f"CREATE TABLE {name} AS SELECT * FROM frame")
    duck.execute("CHECKPOINT")
    size["duckdb"] = round(duck_path.stat().st_size / 1e6, 1)

    seconds = {}
    for name, sql in QUERIES.items():
        seconds[name] = {"sqlite": timed(lambda: lite.execute(sql).fetchall(), repeat=3)}
        for threads in (1, os.cpu_count()):
            duck.execute(f"SET threads = {threads}")
            seconds[name][f"duckdb, {threads} thread" + ("s" if threads > 1 else "")] = timed(lambda: duck.execute(sql).fetchall())
    keys = np.random.default_rng(1).integers(0, N_ORDERS, 2000).tolist()
    seconds["2,000 lookups by key"] = {
        "sqlite": timed(lambda: [lite.execute("SELECT * FROM orders WHERE order_id = ?", (k,)).fetchone() for k in keys], repeat=3),
        f"duckdb, {os.cpu_count()} threads": timed(lambda: [duck.execute("SELECT * FROM orders WHERE order_id = ?", (k,)).fetchone() for k in keys], repeat=3),
    }
    duck.execute("SET threads = 1")
    seconds["2,000 lookups by key"]["duckdb, 1 thread"] = timed(
        lambda: [duck.execute("SELECT * FROM orders WHERE order_id = ?", (k,)).fetchone() for k in keys], repeat=3)
    result["seconds"] = seconds

    # ------------------------------------------------------------ what an index changes
    plan = lambda sql: [row[3] for row in lite.execute("EXPLAIN QUERY PLAN " + sql).fetchall()]
    index = {"rows matched by one day": lite.execute("SELECT count(*) FROM orders WHERE order_day = 365").fetchone()[0],
             "one day": [{"index": "none", "plan": plan(ONE_DAY), "seconds": seconds["one day of 730"]["sqlite"]}]}
    start = time.perf_counter()
    lite.execute("CREATE INDEX idx_orders_day ON orders(order_day)")
    index["seconds to build one index"] = round(time.perf_counter() - start, 2)
    index["one day"].append({"index": "(order_day)", "plan": plan(ONE_DAY), "seconds": timed(lambda: lite.execute(ONE_DAY).fetchall())})
    lite.execute("CREATE INDEX idx_orders_day_amount ON orders(order_day, amount)")
    lite.execute("ANALYZE")
    lite.commit()
    index["one day"].append({"index": "(order_day, amount), covering", "plan": plan(ONE_DAY),
                             "seconds": timed(lambda: lite.execute(ONE_DAY).fetchall())})
    size["sqlite with two indexes"] = round(lite_path.stat().st_size / 1e6, 1)
    index["a third of the table"] = {
        "share of rows": round(lite.execute("SELECT avg(order_day < 243) FROM orders").fetchone()[0], 3),
        "plan chosen": plan(BROAD),
        "seconds with the chosen plan": timed(lambda: lite.execute(BROAD).fetchall(), repeat=2, warmup=0),
        "seconds with a forced scan": timed(lambda: lite.execute(BROAD.replace("FROM orders", "FROM orders NOT INDEXED")).fetchall(), repeat=3),
    }
    result["index"] = index

    # ------------------------------------------------------------ what a commit costs
    ins = sqlite3.connect(work / "inserts.sqlite")
    ins.executescript("PRAGMA journal_mode = DELETE; PRAGMA synchronous = FULL; CREATE TABLE t (a INTEGER, b REAL, c TEXT);")
    sample = [(i, i * 0.5, "x") for i in range(2000)]
    start = time.perf_counter()
    for row in sample:
        ins.execute("INSERT INTO t VALUES (?,?,?)", row)
        ins.commit()
    each = time.perf_counter() - start
    start = time.perf_counter()
    with ins:
        ins.executemany("INSERT INTO t VALUES (?,?,?)", sample)
    once = time.perf_counter() - start
    result["insert 2,000 rows"] = {"commit after every row": round(each, 2), "one transaction": round(once, 4), "ratio": round(each / once)}

    extra = [(N_ORDERS + i,) + row[1:] for i, row in enumerate(as_rows({k: v[:200_000] for k, v in orders.items()}))]
    start = time.perf_counter()
    with lite:
        lite.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", extra)
    with_indexes = time.perf_counter() - start
    lite.executescript(f"DELETE FROM orders WHERE order_id >= {N_ORDERS}; DROP INDEX idx_orders_day; DROP INDEX idx_orders_day_amount;")
    start = time.perf_counter()
    with lite:
        lite.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", extra)
    result["insert 200,000 rows"] = {"with two secondary indexes": round(with_indexes, 2), "with none": round(time.perf_counter() - start, 2)}

    # ------------------------------------------------------------ the same table as files
    duck.execute(f"SET threads = {os.cpu_count()}")
    for label, options in (("parquet, zstd", "FORMAT parquet, COMPRESSION zstd"), ("csv", "FORMAT csv, HEADER")):
        target = work / ("orders." + label.split(",")[0])
        duck.execute(f"COPY (SELECT * FROM orders WHERE order_id < {N_ORDERS}) TO '{target.as_posix()}' ({options})")
        size[label] = round(target.stat().st_size / 1e6, 1)
    result["size in MB"] = size

    lite.close(); duck.close(); ins.close()
    shutil.rmtree(work, ignore_errors=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
