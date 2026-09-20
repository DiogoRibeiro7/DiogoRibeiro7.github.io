---
permalink: '/data-science/a_data_lake_is_a_directory_with_rules/'
title: 'A Data Lake Is a Directory With Rules'
categories:
- Data Science
tags:
- Data Engineering
- Databases
- SQL
- Python
author_profile: false
seo_title: 'A Data Lake Is a Directory With Rules: Parquet, Sorting, Partitions and Small Files, Measured'
seo_description: 'Six million events written as CSV and as Parquet in four layouts. Parquet with zstd is 27% of the CSV and answers a narrow query 30 to 40 times faster, sorting lets a filter skip 48 of 49 row groups, and partitioning by day makes every query slower because 2,190 small files cost more to find than to read.'
excerpt: >-
  A data lake is files in folders plus the conventions that make them usable.
  The same six million rows answer a question in 10 milliseconds or in 2.7
  seconds depending on three of those conventions: the file format, the order
  of the rows, and how many files the data are cut into.
summary: >-
  One table written as CSV and as Parquet, sorted and unsorted, and in four
  partition layouts: what the format saves and why, how sorting turns file
  statistics into an index and what it costs, why fine partitions make every
  query slower, where small files come from, and what a directory cannot do
  without a table format.
keywords:
  - data lake
  - Parquet
  - partitioning
  - small files problem
  - zone maps
  - DuckDB
  - table formats
classes: wide
date: '2026-09-19'
why_this_exists: >-
  The previous article measured what the layout of bytes inside a database does
  to a query. Most analytical data never enters a database: it sits in files, and
  the advice about those files is usually a list of products. The rules that
  matter are few, and their effects are large enough to measure on a laptop.
evidence: >-
  Six million synthetic events with eight columns, written by DuckDB 1.5 as CSV,
  as Parquet with three compression settings, as Parquet sorted by day, and in
  four partition layouts from one file to 25,557. File and column sizes and
  row-group statistics are read from the files. Twenty-three statements are timed
  on four threads, in four separate runs of the benchmark.
methodology: >-
  Sizes, counts and statistics are properties of the files and do not depend on
  the machine. Timings are medians of six to fifteen runs in three interleaved
  rounds, with the files in the operating system's cache, and the benchmark
  refuses to take them while another process looks like a batch job. Whole runs
  differed by up to a factor of three on a laptop that changes its clock speed,
  so the text quotes the least disturbed run and gives each ratio with the range
  it took over the four runs. The script and its results file are in the
  repository, and a test checks the article's claims against that file. Table
  formats are described and not measured.
reviewed_at: '2026-09-19'
header:
  image: /assets/images/headers/photo-network-cables.jpg
  og_image: /assets/images/headers/photo-network-cables.jpg
  overlay_image: /assets/images/headers/photo-network-cables.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-network-cables.jpg
  twitter_image: /assets/images/headers/photo-network-cables.jpg
---

A data lake sounds like a product, and it is a directory. More exactly it is a set of files in folders, usually in object storage, together with conventions about their format, their layout and the way changes to them are published. Everything that makes a lake fast, cheap or trustworthy comes from those conventions, and everything that makes one slow, expensive or wrong comes from their absence. The second kind has a name of its own: a data swamp.

The [previous article]({{ '/data-science/a_database_for_analysis_rows_columns_indexes/' | relative_url }}) measured what the layout of bytes inside a database does to a query. This one does the same for files. It uses six million synthetic events with eight columns, written in several ways by one writer, DuckDB, and read back by the same engine on four threads, so that the only thing that changes between two measurements is the rule under test. The sizes and counts below are properties of the files and would be the same on any machine. The timings are not, so they are given as ratios, each with the range it took over four runs of the benchmark.

## The File Format Does Most of the Work

The first rule is the format, and it has the largest effect for the least effort. A CSV file is a row store with no types and no index: every query parses every character of every row. Parquet is a column store in a file. Rows are cut into *row groups*, 122,880 rows each here, and within a row group each column is stored by itself, encoded according to its type and then compressed. A footer records where every column of every row group starts, and the smallest and largest value it holds. The design descends from Google's Dremel (Melnik et al., 2010), and Zeng and colleagues (2023) measure its trade-offs in detail.

| Format | Size | Against CSV |
| :--- | ---: | ---: |
| CSV | 216.7 MB | 1.00 |
| Parquet, uncompressed | 138.9 MB | 0.64 |
| Parquet, snappy | 87.8 MB | 0.41 |
| Parquet, zstd | 57.7 MB | 0.27 |

Half of the saving comes before any compression, from storing numbers as numbers and repeated strings as small integers that point into a dictionary. The other half comes from compressing each column by itself, which works far better than compressing rows, because neighbouring values of one column resemble each other and neighbouring fields of one row do not. The choice between snappy and zstd is a choice between a little less processor time and a third less storage, and for data that are written once and read many times zstd is the better default.

![Two bar charts. Left: six million events take 217 MB as CSV and 139, 88 and 58 MB as Parquet with no compression, snappy and zstd; the zstd file sorted by day is slightly larger at 62 MB. Right: the size of each column inside the zstd file in arrival order and after sorting by day. Sorting shrinks the day column from 6.8 MB to almost nothing and the country column likewise, and swells the event id column from 6.1 to 19.5 MB; the other columns do not change.](/assets/images/figures/data_lake_formats_and_columns.png){: width="1536" height="704" loading="lazy"}

The size is the smaller benefit. The larger one is that a query reads only the columns it names. The `amount` column is 12.9 MB of the 57.7 MB file, 23% of it, so `sum(amount)` touches less than a quarter of the bytes and none of the text. That sum takes 0.42 s from the CSV file and 14 ms from the Parquet file, a ratio that stayed between 29 and 41 over the four runs. The advantage narrows when a query needs everything: a filter and six aggregates that between them use all eight columns take 0.59 s against 0.12 s, four to six times faster, because the whole file now has to be decompressed and only the parsing of text is saved. A columnar format makes narrow questions cheap, and it earns its place because most analytical questions are narrow.

## Sorting Is the Index

A Parquet file has no B-tree. What it has are the minimum and maximum of every column in every row group, kept in the footer, and an engine that reads the footer first can skip any row group whose range excludes the value it is looking for. The idea is Moerkotte's small materialised aggregates (1998), usually called a zone map. It costs almost nothing to store, and whether it is of any use depends entirely on the order of the rows.

The events arrive in no particular order of day, so every one of the 49 row groups holds days from 0 to 364 and a filter on one day can skip none of them. Sort the rows by day before writing and each row group covers about seven and a half consecutive days, so the same filter can skip 48 of the 49. The file's own metadata shows the difference, and it is worth looking, because nothing else will say whether the sort was done.

```python
import pyarrow.parquet as pq

for name in ("events_zstd.parquet", "events_sorted_by_day.parquet"):
    meta = pq.ParquetFile(name).metadata
    day = meta.schema.names.index("event_day")
    ranges = [meta.row_group(g).column(day).statistics for g in range(meta.num_row_groups)]
    hits = sum(1 for s in ranges if s.min <= 200 <= s.max)
    print(f"{name:30} {hits:2} of {meta.num_row_groups} row groups may hold day 200;"
          f" the first covers days {ranges[0].min} to {ranges[0].max}")
```

```text
events_zstd.parquet            49 of 49 row groups may hold day 200; the first covers days 0 to 364
events_sorted_by_day.parquet    1 of 49 row groups may hold day 200; the first covers days 0 to 7
```

The query for one day takes 13 ms on the file in arrival order and 1.9 ms on the sorted one, between five and seven times faster in every run, from the same format and with no index to build or maintain. It is not 49 times faster because at two milliseconds the fixed costs dominate: opening the file, reading the footer and planning the query take longer than reading one row group.

Sorting is not free, and the right panel of the figure shows where the cost lands. The day column shrinks from 6.8 MB to 0.01 MB and the country column from 2.3 MB to 0.02 MB, because a sorted column is a few long runs of one value. But `event_id` was a sequence in arrival order, which compresses to 6.1 MB, and the sort scrambles it: it grows to 19.5 MB. The sorted file is *larger*, 61.9 MB against 57.7 MB. Sorting moves disorder from the columns that were sorted into the columns that were correlated with the old order. It is still the right choice when queries filter by day, but it is a trade, and the column sizes in the metadata are where to check what was paid.

## Folders as a Coarse Index

The second way to skip data is to put them in different places. Under the convention called Hive partitioning, a folder named `event_day=200` states a fact about every row beneath it, so an engine can decide from the path alone whether to open a file. It is an index made of directory names, and it is the most over-used feature of data lakes. The same table was written in four layouts, each by one `COPY ... PARTITION_BY` statement.

| Layout | Folders | Files | Size | Rows per file |
| :--- | ---: | ---: | ---: | ---: |
| One file | 1 | 1 | 57.7 MB | 6,000,000 |
| By month | 12 | 12 | 56.0 MB | 500,000 |
| By day | 365 | 2,190 | 65.9 MB | 2,740 |
| By day and country | 3,650 | 25,557 | 97.0 MB | 235 |

![Grouped bars on a logarithmic time axis: the time to sum the whole table, to query one day and to query one country, and the time to list the files alone, for the same six million events in four layouts. One file, 1 file: 9.5 ms, 16 ms and 18 ms, with 0.5 ms to list the files. By month, 12 files: 16 ms, 5.7 ms and 23 ms, with 1.7 ms to list the files. By day, 2,190 files: 241 ms, 99 ms and 348 ms, with 73 ms to list the files. By day and country, 25,557 files: 2.7 s, 1.1 s and 1.3 s, with 939 ms to list the files.](/assets/images/figures/data_lake_partition_layouts.png){: width="1536" height="736" loading="lazy"}

The monthly layout behaves as the theory says. The query for one day reads one folder of twelve and runs two to three times faster than on the single file, 5.7 ms against 16 ms. The questions that cannot use the partition pay for it, but not much: between a fifth and three quarters more time, a few milliseconds. That is what a partition is for: a filter that nearly every query carries, on a column with few values, leaving files that are still large.

The daily layout ought to serve the one-day question better still, since it reads 1/365 of the data and not 1/12. It is five to seven times *slower* than the single file, 99 ms against 16 ms, and the grey bars show why. Before the engine can discard 364 folders it has to learn that they exist, and listing 2,190 files takes 73 ms, which is most of the query. Handed the path of the folder, the same question takes 1.3 ms, between 50 and 90 times less. The rows for one day were never the cost, and finding them was. The questions that cannot use the partition do worse, since they open every file: the sum over the whole table takes about twenty times as long as it does on one file.

Partitioning by day and country multiplies the files by nearly twelve, and everything follows from that. Listing them takes about a second. The sum over the whole table takes 2.7 s against 9.5 ms, between 150 and 340 times slower depending on the run. Even the query for one country, which the layout was presumably built to serve, takes 45 to 100 times longer than scanning the single file and filtering it. All of this is on a local disk, where finding a file costs a few tens of microseconds. In object storage a listing is a sequence of network requests, which on Amazon S3 return at most a thousand names each, and every file opened is at least one more request, so the same layouts are slower in absolute terms and their ranking does not change.

## Where Small Files Come From

Nobody decides to have 25,000 files. They arrive as a product: the number of partitions, multiplied by the number of pieces the writer cuts each partition into, multiplied by the number of times the job runs. DuckDB's partitioned writer keeps a bounded number of files open and flushes its buffers at intervals; the settings are `partitioned_write_max_open_files`, 100 by default, and `partitioned_write_flush_threshold`, 524,288 rows. With twelve partitions neither limit matters and each month is one file. With 365 the writer has to close files and start new ones, and every day ends up in six pieces of about 2,700 rows. With 3,650 partitions there are seven pieces each, 25,557 files that average 235 rows. A distributed job with two hundred tasks does the same thing at a larger scale, every night.

Each of those files carries its own footer, schema and dictionaries, and a column of 235 values barely compresses. The same six million rows take 57.7 MB in one file, 65.9 MB in the daily layout and 97.0 MB in the finest one, 68% more bytes for identical data. The time is worse than the space, as the previous section showed, because every file has to be found and opened and its footer read before a single value is used. In object storage each of those steps also has a price, since requests are billed.

The remedy is a rule and a job. The rule is to partition by a column with few values that nearly every query filters on, and to stop while the files are still large. The Parquet documentation suggests row groups of hundreds of megabytes, and a file should hold at least one: for this table that means month and not day, and at its real size it means no partitions at all. The job is *compaction*, which rewrites the many small files of a partition as a few large sorted ones, and which runs after the writes and before the reads.

## What a Directory Cannot Do

Formats, sorting and partitioning make a directory fast. They do not make it a table, and the difference appears as soon as two processes touch it. A reader that lists the directory while a writer is halfway through a job sees some of the new files and not others, and computes a total that was never true. A job that fails after writing half its files leaves them there for the next reader. Nothing stops a writer from adding a file whose `amount` column is text. And there is no way to change or delete one row, only to rewrite the file that holds it.

Table formats, of which Apache Iceberg and Delta Lake are the most widely used, fix this without moving the data (Armbrust et al., 2020). They add a small log beside the files that says exactly which files make up each version of the table. A write becomes visible by adding one entry to that log, atomically, so a reader sees all of a change or none of it. The schema lives in the log and is enforced. Old versions stay readable, which makes a query reproducible as of a date. And the log carries the statistics of every file, so an engine plans from the log and never lists the directory, which removes the cost measured two sections ago.

With those pieces a lake offers most of what a warehouse does, which is the argument of the lakehouse paper (Armbrust et al., 2021). None of it is measured here. Doing so needs a catalogue and more than one process, the interesting failures are concurrent ones, and that deserves an article of its own. What the measurements above do show is why the log has to exist: a directory listing is the slowest and least reliable part of reading a lake, and a table format is, among other things, a way of never doing one.

## The Rules

The title promised rules, and the measurements reduce to six. Store analytical data as Parquet with zstd, never as CSV. Sort the rows within each file by the column that most queries filter on, and read the row-group statistics to confirm that it worked. Partition only by a column with few values that almost every query names, and stop while the files are still hundreds of megabytes. Count the files a job writes, since the number is partitions times pieces times runs, and compact them. Publish a batch by writing it somewhere else and renaming the folder, or use a table format, so that no reader ever sees half of it. And keep the expected schema in code that the write path checks, because a directory accepts anything.

A laptop and DuckDB are enough to apply all of this to tens of gigabytes, and the same rules hold at petabytes, where the engine has a different name and the bill for ignoring them has more digits. The benchmark is [`data_lake_benchmarks.py`](https://github.com/DiogoRibeiro7/blog-reproducibility/blob/main/scripts/benchmarks/data_lake_benchmarks.py) in the [blog-reproducibility repository](https://github.com/DiogoRibeiro7/blog-reproducibility), with its recorded results beside it under `data/engineering/`. It writes about a gigabyte of temporary files, takes a few minutes, and declines to time anything while the machine is busy.

## References

- Armbrust, M., Das, T., Sun, L., Yavuz, B., Zhu, S., Murthy, M., Torres, J., van Hovell, H., Ionescu, A., Łuszczak, A., Świtakowski, M., Szafrański, M., Li, X., Ueshin, T., Mokhtar, M., Boncz, P., Ghodsi, A., Paranjpye, S., Senster, P., Xin, R., & Zaharia, M. (2020). Delta Lake: high-performance ACID table storage over cloud object stores. *Proceedings of the VLDB Endowment*, 13(12), 3411-3424.
- Armbrust, M., Ghodsi, A., Xin, R., & Zaharia, M. (2021). Lakehouse: a new generation of open platforms that unify data warehousing and advanced analytics. *Proceedings of the 11th Conference on Innovative Data Systems Research (CIDR)*.
- Melnik, S., Gubarev, A., Long, J. J., Romer, G., Shivakumar, S., Tolton, M., & Vassilakis, T. (2010). Dremel: interactive analysis of web-scale datasets. *Proceedings of the VLDB Endowment*, 3(1), 330-339.
- Moerkotte, G. (1998). Small materialized aggregates: a light weight index structure for data warehousing. *Proceedings of the 24th International Conference on Very Large Data Bases*, 476-487.
- Zeng, X., Hui, Y., Shen, J., Pavlo, A., McKinney, W., & Zhang, H. (2023). An empirical evaluation of columnar storage formats. *Proceedings of the VLDB Endowment*, 17(2), 148-161.
