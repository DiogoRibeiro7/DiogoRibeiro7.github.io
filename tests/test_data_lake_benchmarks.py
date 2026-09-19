"""The claims of "A Data Lake Is a Directory With Rules" have to hold in the results file beside the benchmark.

The benchmark writes a gigabyte of files and is too slow for the test suite, so its
results are kept in assets/viz/data_lake_benchmarks.json. The sizes and counts are
the same on any machine; the timings are not, and these checks say which statements
in the article depend on them.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "assets" / "viz" / "data_lake_benchmarks.json").read_text(encoding="utf-8"))
POST = (ROOT / "_posts" / "data_science" / "2026-09-19-a_data_lake_is_a_directory_with_rules.md").read_text(encoding="utf-8")
SIZES = RESULTS["size in MB"]
COLUMNS = RESULTS["column sizes in MB"]
LAYOUTS = RESULTS["layouts"]
SECONDS = RESULTS["seconds"]
RUNS = RESULTS["timing runs"]


def ratios(of):
    """One ratio of two timings, in every run that was kept."""
    return [of(run["seconds"]) for run in RUNS]


def test_each_step_of_the_format_shrinks_the_file():
    order = ("CSV", "Parquet, uncompressed", "Parquet, snappy", "Parquet, zstd")
    assert all(SIZES[a] > SIZES[b] for a, b in zip(order, order[1:]))
    # half of the saving comes before any compression
    before_compression = (SIZES["CSV"] - SIZES["Parquet, uncompressed"]) / (SIZES["CSV"] - SIZES["Parquet, zstd"])
    assert 0.45 < before_compression < 0.55


def test_a_narrow_query_touches_a_quarter_of_the_file():
    assert RESULTS["share of the file in amount"] < 0.25


def test_sorting_makes_the_statistics_useful():
    groups = RESULTS["row groups that may hold one day"]
    assert groups["arrival order"]["may hold it"] == groups["arrival order"]["of"]
    assert groups["sorted by day"]["may hold it"] == 1


def test_sorting_moves_bytes_between_columns_and_the_file_grows():
    arrival, by_day = COLUMNS["arrival order"], COLUMNS["sorted by day"]
    assert by_day["event_day"] < 0.01 * arrival["event_day"]
    assert by_day["country"] < 0.05 * arrival["country"]
    assert by_day["event_id"] > 3 * arrival["event_id"]
    for untouched in ("user_id", "device", "product_id", "quantity", "amount"):
        assert abs(by_day[untouched] / arrival[untouched] - 1) < 0.01, untouched
    assert SIZES["Parquet, zstd, sorted by day"] > SIZES["Parquet, zstd"]


def test_files_multiply_and_small_files_cost_bytes():
    assert [LAYOUTS[n]["folders"] for n in LAYOUTS] == [1, 12, 365, 3650]
    assert LAYOUTS["by day"]["files"] > 3 * LAYOUTS["by day"]["folders"]
    assert LAYOUTS["by day and country"]["files"] > 3 * LAYOUTS["by day and country"]["folders"]
    assert LAYOUTS["by day and country"]["MB"] > 1.5 * LAYOUTS["one file"]["MB"]


def test_no_run_was_kept_while_a_batch_job_was_running():
    assert len(RUNS) >= 3
    for run in RUNS:
        for moment in ("before", "after"):
            load = run["machine"][f"cores used by other processes, {moment}"]
            assert load["the busiest of them"] <= 1.5
            assert load["all other processes"] + run["machine"]["threads"] + 4 <= RESULTS["machine"]["cpus"]


def test_the_reported_run_is_one_of_the_runs():
    assert SECONDS in [run["seconds"] for run in RUNS]


def test_parquet_beats_csv_most_on_a_narrow_query():
    narrow = ratios(lambda s: s["sum of one column"]["CSV"] / s["sum of one column"]["Parquet"])
    every = "a query that needs all eight columns"
    wide = ratios(lambda s: s[every]["CSV"] / s[every]["Parquet"])
    assert 25 < min(narrow) and max(narrow) < 45      # "between 29 and 41"
    assert 3.5 < min(wide) and max(wide) < 6.5        # "four to six times"


def test_sorted_file_answers_the_one_day_question_faster():
    gain = ratios(lambda s: s["one day of 365"]["arrival order"] / s["one day of 365"]["sorted by day"])
    assert 5 <= min(gain) and max(gain) <= 7          # "between five and seven times"


def test_a_coarse_partition_helps_the_query_that_uses_it():
    gain = ratios(lambda s: s["layouts"]["one file"]["one day"] / s["layouts"]["by month"]["one day"])
    assert 1.8 < min(gain) and max(gain) < 3          # "two to three times faster"
    for query in ("whole table", "one country"):
        cost = ratios(lambda s: s["layouts"]["by month"][query] / s["layouts"]["one file"][query])
        assert 1.15 < min(cost) and max(cost) < 1.75, query   # "a fifth to three quarters more"


def test_fine_partitions_are_slower_for_every_question():
    against_one_file = lambda layout, query: ratios(lambda s: s["layouts"][layout][query] / s["layouts"]["one file"][query])
    one_day = against_one_file("by day", "one day")
    assert 4.5 < min(one_day) and max(one_day) < 7.6                  # "five to seven times slower"
    whole = against_one_file("by day", "whole table")
    assert 15 < min(whole) and max(whole) < 27                        # "about twenty times"
    whole = against_one_file("by day and country", "whole table")
    assert 145 < min(whole) and max(whole) < 345                      # "between 150 and 340"
    country = against_one_file("by day and country", "one country")
    assert 45 < min(country) and max(country) < 105                   # "45 to 100 times"


def test_finding_the_files_is_most_of_the_cost():
    for layout in ("by day", "by day and country"):
        share = ratios(lambda s: s["layouts"][layout]["listing the files"] / s["layouts"][layout]["one day"])
        assert min(share) > 0.7, layout
    direct = ratios(lambda s: s["layouts"]["by day"]["one day"] / s["one day, given its folder"])
    assert 50 < min(direct) and max(direct) < 90                      # "between 50 and 90 times less"


def test_numbers_quoted_in_the_post_match_the_results():
    finest = LAYOUTS["by day and country"]
    quoted = [
        f"{SIZES['CSV']} MB", f"{SIZES['Parquet, uncompressed']} MB", f"{SIZES['Parquet, snappy']} MB", f"{SIZES['Parquet, zstd']} MB",
        f"{SIZES['Parquet, zstd, sorted by day']} MB",
        f"{LAYOUTS['by day']['files']:,} files", f"{finest['files']:,} files", f"{finest['rows per file']} rows", f"{finest['MB']} MB",
        f"{round(100 * (finest['MB'] / LAYOUTS['one file']['MB'] - 1))}% more",
        f"{round(100 * RESULTS['share of the file in amount'])}%",
    ]
    for text in quoted:
        assert text in POST, text
