---
author_profile: false
categories:
- Data Science
classes: wide
date: '2025-06-06'
excerpt: Discover the essential steps of Exploratory Data Analysis (EDA) and how to
  gain insights from your data before building models.
header:
  image: /assets/images/data_science_5.jpg
  og_image: /assets/images/data_science_5.jpg
  overlay_image: /assets/images/data_science_5.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_5.jpg
  twitter_image: /assets/images/data_science_5.jpg
keywords:
- Exploratory data analysis
- Data visualization
- Python
- Pandas
- Data cleaning
permalink: '/data-science/exploratory_data_analysis_intro/'
redirect_from:
- '/data science/exploratory_data_analysis_intro/'
seo_description: Learn the fundamentals of Exploratory Data Analysis using Python,
  including data cleaning, visualization, and summary statistics.
seo_title: Beginner's Guide to Exploratory Data Analysis (EDA)
seo_type: article
summary: This guide covers the core principles of Exploratory Data Analysis, demonstrating
  how to inspect, clean, and visualize datasets to uncover patterns and inform subsequent
  modeling steps.
tags:
- Exploratory Data Analysis
- Data Science
- Python
- Data Visualization
title: 'Exploratory Data Analysis: A Beginner''s Guide'
---

Exploratory Data Analysis (EDA) is the process of examining a dataset to understand its main characteristics before applying more formal statistical modeling or machine learning. By exploring your data upfront, you can identify patterns, spot anomalies, and test assumptions that might otherwise go unnoticed.

The point is not to produce charts. It is to find out what the data can and cannot support before you commit to a model, because most modelling failures trace back to something that was visible in the raw data and never looked at.

## Inspecting the Data

The first step is getting to know the dataset. Load it into a DataFrame, then examine the column names, data types, and a few example rows to confirm everything parsed correctly.

```python
import pandas as pd

df = pd.read_csv("readings.csv")

print(df.shape)                       # rows, columns
print(df.dtypes)                      # silent parsing failures show up here
print(df.head())
print(df.describe(include="all").T)   # numeric and categorical together
print(df.isna().mean().sort_values(ascending=False).head(10))
```

Two of these deserve attention. `dtypes` catches the most common silent failure: a numeric column read as `object` because a few rows contain `"N/A"`, a stray comma, or a footnote marker. Every later calculation on that column will either fail or quietly do something else.

The missing-value proportions matter more than the counts. A column that is 3% missing is a nuisance; one that is 60% missing is a decision about whether the column exists at all.

## Cleaning and Preparing

Real-world datasets contain missing values, duplicate rows, and inconsistent formats. Cleaning involves handling these — removing or imputing missing values, correcting data types, standardising text fields.

Before imputing anything, ask *why* a value is absent. If it is missing because a sensor fails when readings run high, imputing the mean systematically erases the extreme values you most needed. That is a modelling decision disguised as a cleaning step, and it should be made deliberately.

Duplicates deserve the same scepticism. `df.duplicated().sum()` counts exact repeats, but the more damaging kind is a near-duplicate — the same entity recorded twice with a different ID or a whitespace difference. Check for duplication on the columns that should uniquely identify a record, not on the whole row.

Record every transformation. A cleaning step that lives only in a notebook cell you later edited is not reproducible, and the number you report will not be recoverable six months on.

## Summary Statistics and What They Hide

Descriptive statistics give a quick read on central tendency and spread, but each one conceals a specific failure mode.

The mean is pulled by outliers; comparing it against the median tells you about skew immediately. A mean well above the median means a long right tail, which is the normal shape for income, duration, and count data. Standard deviation assumes a roughly symmetric spread; on skewed data the interquartile range is more honest.

Correlation is the most over-read of all. Pearson's $r$ measures *linear* association only, so a perfect parabola scores near zero. It is also acutely sensitive to outliers: a single extreme point can manufacture a strong correlation between unrelated variables or mask a real one.

Anscombe's quartet is the standard demonstration — four datasets with identical means, variances, correlations and regression lines that look nothing alike when plotted. It is the argument for not stopping at the summary table.

```python
num = df.select_dtypes("number")
print((num.mean() - num.median()).sort_values())   # skew indicator
print(num.corr(numeric_only=True).round(2))
print(num.corr(method="spearman").round(2))        # rank-based, outlier-resistant
```

Comparing Pearson against Spearman is a cheap diagnostic. Where they disagree sharply, the relationship is either non-linear or driven by a handful of points.

## Visualising Distributions and Relationships

Visualisation is central to EDA because shape is what summaries throw away. Histograms and box plots reveal the distribution of numerical variables, bar charts summarise categorical counts, and scatter plots expose relationships between features.

Match the plot to the question. A histogram shows the shape of one variable and is sensitive to bin width, so try more than one. A box plot compares many groups compactly but hides multimodality entirely — a box plot of a two-humped distribution looks identical to a single-humped one with the same quartiles. When the sample is small enough, plot every point.

Watch for the specific patterns that change what you do next: bimodality, which usually means two populations mixed together; a spike at zero, which often means "not measured" encoded as a number; values piled at a boundary, which suggests censoring or a clipped instrument; and repeated identical values, which can mean a stuck sensor or a default being recorded.

## Common Pitfalls

The biggest risk in EDA is that looking hard at data will always turn something up. If you test enough hypotheses generated by the same data you explored, the interesting ones will include a fair number of coincidences. Treat anything found during exploration as a hypothesis to be checked on data you have not looked at, not as a result.

Related traps recur often enough to name. Drawing conclusions from small subgroups produces the most extreme apparent effects, because small samples vary most. Deleting outliers because they are inconvenient rather than because they are wrong removes exactly the observations that carry information about failure modes. And exploring the test set at all leaks information into every decision you subsequently make.

Simpson's paradox deserves particular attention: a relationship visible in aggregate can reverse within every subgroup. Before trusting an aggregate trend, check whether it survives when you condition on the obvious grouping variable.

## Interactive Tools

Jupyter notebooks let you mix code and commentary so findings are documented as you go. Plotly and Altair add interactivity to charts, and dashboards in Streamlit let stakeholders explore the data themselves.

Profiling libraries such as `ydata-profiling` generate a full report — distributions, correlations, missing-value patterns, and warnings — in a single call. These are excellent for the first pass and a poor substitute for the second, because they cannot tell you which of the anomalies they surface actually matter for your question.

## Where EDA Leads

EDA serves as the foundation for everything downstream. What you learn here decides which transformations are needed, which features are usable, which model families are plausible, and which questions the data simply cannot answer.

A disciplined pass answers a short list before any modelling starts: what does one row represent, how much data is missing and why, what shape does the target variable have, which features are degenerate or duplicated, and what would have to be true for this dataset to be misleading. Getting those answers early is far cheaper than discovering them from a model that has already been trained, tuned, and reported.

## References

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician*, 27(1), 17-21.
- Wickham, H., & Grolemund, G. (2017). *R for Data Science*. O'Reilly Media.
- Matejka, J., & Fitzmaurice, G. (2017). Same stats, different graphs: generating datasets with varied appearance and identical statistics. *Proceedings of CHI*, 1290-1294.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.
