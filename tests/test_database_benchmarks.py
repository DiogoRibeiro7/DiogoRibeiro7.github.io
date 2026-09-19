"""The claims of "A Database for Analysis" have to hold in the results file beside the benchmark.

The benchmark is too slow for the test suite, so its results are kept in
assets/viz/database_benchmarks.json. Rerunning it on another machine changes the
timings; these checks say which statements in the article depend on them.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "assets" / "viz" / "database_benchmarks.json").read_text(encoding="utf-8"))
POST = (ROOT / "_posts" / "data_science" / "2026-09-19-a_database_for_analysis_rows_columns_indexes.md").read_text(encoding="utf-8")
ANALYTICAL = ("sum of one column", "one day of 730", "group by channel", "star join")


def test_column_store_wins_analytical_queries_on_one_thread():
    for query in ANALYTICAL:
        timing = RESULTS["seconds"][query]
        assert timing["sqlite"] / timing["duckdb, 1 thread"] > 5, query


def test_row_store_wins_point_lookups():
    timing = RESULTS["seconds"]["2,000 lookups by key"]
    assert timing["duckdb, 1 thread"] / timing["sqlite"] > 5


def test_covering_index_beats_plain_index_beats_scan():
    scan, plain, covering = (step["seconds"] for step in RESULTS["index"]["one day"])
    assert scan > plain > covering
    assert "COVERING INDEX" in RESULTS["index"]["one day"][2]["plan"][0]


def test_broad_filter_is_slower_through_the_index_than_as_a_scan():
    broad = RESULTS["index"]["a third of the table"]
    assert "USING INDEX" in broad["plan chosen"][0]
    assert broad["seconds with the chosen plan"] > 5 * broad["seconds with a forced scan"]


def test_lookup_cost_predicts_the_broad_index_scan():
    lookup = RESULTS["seconds"]["2,000 lookups by key"]["sqlite"] / 2000
    broad = RESULTS["index"]["a third of the table"]
    predicted = lookup * RESULTS["rows"]["orders"] * broad["share of rows"]
    assert 0.5 < predicted / broad["seconds with the chosen plan"] < 2


def test_commit_per_row_and_indexes_slow_writes():
    assert RESULTS["insert 2,000 rows"]["ratio"] > 50
    inserts = RESULTS["insert 200,000 rows"]
    assert inserts["with two secondary indexes"] > 5 * inserts["with none"]


def test_headline_numbers_in_the_post_match_the_results():
    seconds = RESULTS["seconds"]
    quoted = {
        "32×": seconds["sum of one column"]["sqlite"] / seconds["sum of one column"]["duckdb, 1 thread"],
        "97×": seconds["group by channel"]["sqlite"] / seconds["group by channel"]["duckdb, 1 thread"],
        "42×": seconds["star join"]["sqlite"] / seconds["star join"]["duckdb, 1 thread"],
    }
    for text, value in quoted.items():
        assert text in POST, text
        assert round(value) == int(re.sub(r"\D", "", text)), (text, value)
    assert f"{RESULTS['insert 2,000 rows']['ratio']:,}" in POST
