---
author_profile: false
categories:
- Data Science
- Machine Learning
- Python
classes: wide
date: '2024-07-17'
header:
  image: /assets/images/data_science_2.jpg
  overlay_image: /assets/images/data_science_2.jpg
  teaser: /assets/images/data_science_2.jpg
tags:
- Outlier Detection
- Machine Learning Algorithms
- Data Analysis
title: Interpretable Outlier Detection with Counts Outlier Detector (COD)
---

## Overview of the Counts Outliers Detector (COD)

COD extends the idea of histogram-based outlier detection and supports multi-dimensional histograms. This allows COD to identify outliers that are rare combinations of 2, 3, or more values, as well as the rare single values that can be detected by standard (1d) histogram-based methods such as HBOS. It can catch unusual single values such as heights of 7'2", and can also catch where a person has an age of 2 and a height of 5'10".

We look at 2d histograms first, but COD can support histograms up to 6 dimensions (we describe below why it does not go beyond this, and in fact, using only 2 or 3 or 4 dimensions will often work best).

A 2d histogram can be viewed similarly as a heatmap. In the image below we see a histogram in 2d space where the data in each dimension is divided into 13 bins, creating 169 (13 x 13) bins in the 2d space. We can also see one point (circled) that is an outlier in the 2d space. This point is in a bin with very few items (in this case, only one item) and so can be identified as an outlier when examining this 2d space.

This point is not an outlier in either 1d space; it is not unusual in the x dimension or the y dimension, so would be missed by HBOS and other tools that examine only single dimensions at a time.

As with HBOS, COD creates a 1d histogram for each single feature. But then, COD also creates a 2d histogram like this for each pair of features, so is able to detect any unusual pairs of values. The same idea can then be applied to any number of dimensions. It is more difficult to draw, but COD creates 3d histograms for each triple of features (each bin is a cube), and so on. Again, it calculates the counts (using the training data) in each bin and is able to identify outliers: values (or combinations of values) that appear in bins with unusually low counts.

## The Curse of Dimensionality

Although it’s effective to create histograms based on each set of 2, 3, and often more features, it is usually infeasible to create a histogram using all features, at least if there are more than about 6 or 7 features in the data. Due to what’s called the curse of dimensionality, we may have far more bins than data records.

For example, if there are 50 features, even using only 2 bins per feature, we would have 2 to the power of 50 bins in a 50d histogram, which is certainly many orders of magnitude greater than the number data records. Even with only 20 features (and using 2 bins per feature), we would have 2 to the power of 20, over one million, bins. Consequently, we can end up with most bins having no records, and those bins that do have any, containing only one or two items.

Most data is relatively skewed and there are usually associations between the features, so the affect won’t be as strong as if the data were spread uniformly through the space, but there will still likely be far too many features to consider at once using a histogram-based method for outlier detection.

Fortunately though, this is actually not a problem. It’s not necessary to create high-dimensional histograms; low-dimensional histograms are quite sufficient to detect the most relevant (and most interpretable) outliers. Examining each 1d, 2d and 3d space, for example, is sufficient to identify each unusual single value, pair of values, and triple of values. These are the most comprehensible outliers and, arguably, the most relevant (or at least typically among the most relevant). Where desired (and where there is sufficient data), examining 4d, 5d or 6d spaces is also possible with COD.

## The COD Algorithm

The approach taken by COD is to first examine the 1d spaces, then the 2d spaces, then 3d, and so on, up to at most 6d. If a table has 50 features, this will examine (50 choose 1) 1d spaces (finding the unusual single values), then (50 choose 2) 2d spaces (finding the unusual pairs of values), then (50 choose 3) 3d spaces (finding the unusual triples of values), and so on. This covers a large number of spaces, but it means each record is inspected thoroughly and that anomalies (at least in lower dimensions) are not missed.

Using histograms also allows for relatively fast calculations, so this is generally quite tractable. It can break down with very large numbers of features, but in this situation virtually all outlier detectors will eventually break down. Where a table has many features (for example, in the dozens or hundreds), it may be necessary to limit COD to 1d spaces, finding only unusual single values — which may be sufficient in any case for this situation. But for most tables, COD is able to examine even up to 4 or 5 or 6d spaces quite well.

Using histograms also eliminates the distance metrics used by many outlier detector methods, including some of the most well-used. While very effective in lower dimensions methods, such as LOF, kNN, and several others use all features at once and can be highly susceptible to the curse of dimensionality in higher dimensions. For example, kNN identifies outliers as points that are relatively far from their k nearest neighbors. This is a sensible and generally affective approach, but with very high dimensionality, the distance calculations between points can become highly unreliable, making it impossible to identify outliers using kNN or similar algorithms.

By examining only small dimensionalities at a time, COD is able to handle far more features than many other outlier detection methods.

## Limiting Evaluation to Small Dimensionalities

To see why it’s sufficient to examine only up to about 3 to 6 dimensions, we look at the example of 4d outliers. By 4d outliers, I’m referring to outliers that are rare combinations of some four features, but are not rare combinations of any 1, 2, or 3 features. That is, each single feature, each pair of features, and each triple of features is fairly common, but the combination of all four features is rare.

This is possible, and does occur, but is actually fairly uncommon. For most records that have a rare combination of 4 features, at least some subset of two or three of those features will usually also be rare.

One of the interesting things I discovered while working on this and other tools is that most outliers can be described based on a relatively small set of features. For example, consider a table (with four features) representing house prices, we may have features for: square feet, number of rooms, number of floors, and price. Any single unusual value would likely be interesting. Similarly for any pair of features (e.g. low square footage with a large number of floors; or low square feet with high price), and likely any triple of features. But there’s a limit to how unusual a combination of all four features can be without there being any unusual single value, unusual pair, or unusual triple of features.

By checking only lower dimensions we cover most of the outliers. The more dimensions covered, the more outliers we find, but there are diminishing returns, both in the numbers of outliers, and in their relevance.

Even where some legitimate outliers may exist that can only be described using, say, six or seven features, they are most likely difficult to interpret, and likely of lower importance than outliers that have a single rare value, or single pair, or triple of rare values. They also become difficult to quantify statistically, given the numbers of combinations of values can be extremely large when working with beyond a small number of features.

By working with small numbers of features, COD provides a nice middle ground between detectors that consider each feature independently (such as HBOS, z-score, inter-quartile range, entropy-based tests and so on) and outlier detectors that consider all features at once (such as Local Outlier Factor and KNN).

## How COD Removes Redundancy in Explanations

Counts Outlier Detector works by first examining each column individually and identifying all values that are unusual with respect to their columns (the 1d outliers).

It then examines each pair of columns, identifying the rows with pairs of unusual values within each pair of columns (the 2d outliers). The detector then considers sets of 3 columns (identifying 3d outliers), sets of 4 columns (identifying 4d outliers), and so on.

At each stage, the algorithm looks for instances that are unusual, excluding values or combinations already flagged in lower-dimensional spaces. For example, in the table of people described above, a height of 7'2" would be rare. Given that, any combination of age and height (or height and anything else), where the height is 7'2", will be rare, simply because 7'2" is rare. As such, there is no need to identify, for example, a height of 7'2" and age of 25 as a rare combination; it is rare only because 7'2" is rare and reporting this as a 2d outlier would be redundant. Reporting it strictly as a 1d outlier (based only on the height) provides the clearest, simplest explanation for any rows containing this height.

So, once we identify 7'2" as a 1d outlier, we do not include this value in checks for 2d outliers, 3d outliers, and so on. The majority of values (the more typical heights relative to the current dataset) are, however, kept, which allows us to further examine the data and identify unusual combinations.

Similarly, any rare pairs of values in 2d spaces are excluded from consideration in 3d and higher-dimensional spaces; any rare triples of values in 3d space will be excluded from 4d and higher-dimensional spaces; and so on.

So, each anomaly is reported using as few features as possible, which keeps the explanations of each anomaly as simple as possible.

Any row, though, may be flagged numerous times. For example, a row may have an unusual value in Column F; an unusual pair of values in columns A and E; another unusual pair of values in D and F; as well as an unusual triple of values in columns B, C, D. The row’s total outlier score would be the sum of the scores derived from these.

## Interpretability

We can identify, for each outlier in the dataset, the specific set of features where it is anomalous. This, then, allows for quite clear explanations. And, given that a high fraction of outliers are outliers in 1d or 2d spaces, most explanations can be presented visually (examples are shown below).

## Scoring

Counts Outlier Detector takes its name from the fact it examines the exact count of each bin. In each space, the bins with unusually low counts (if any) are identified, and any records with values in these bins are identified as having an anomaly in this sense.

The scoring system then used is quite simple, which further supports interpretability. Each rare value or combination is scored equivalently, regardless of the dimensionality or the counts within the bins. Each row is simply scored based on the number of anomalies found.

This can loose some fidelity (rare combinations are scored the same as very rare combinations), but allows for significantly faster execution times and more interpretable results. This also avoids any complication, and any arbitrariness, weighting outliers in different spaces. For example, it may not be clear how to compare outliers in a 4d space vs in a 2d space. COD eliminates this, treating each equally. So, this does trade-off some detail in the scores for interpretability, but the emphasis of the tool is interpretability, and the effect on accuracy is small (as well as being positive as often as negative — treating anomalies equivalently provides a regularizing effect).

By default, only values or combinations that are strongly anomalous will be flagged. This process can be tuned by setting a threshold parameter.