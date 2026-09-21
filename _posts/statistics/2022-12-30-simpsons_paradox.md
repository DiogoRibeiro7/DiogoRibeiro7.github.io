---
permalink: '/statistics/simpsons_paradox/'
author_profile: false
categories:
- Statistics
classes: wide
date: '2022-12-30'
excerpt: Simpson's Paradox shows how aggregated data can lead to misleading trends. Learn the theory behind this paradox, its practical implications, and how to analyze data rigorously.
header:
  image: /assets/images/headers/photo-statistics-dice-coins.jpg
  og_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_image: /assets/images/headers/photo-statistics-dice-coins.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-dice-coins.jpg
  twitter_image: /assets/images/headers/photo-statistics-dice-coins.jpg
seo_description: 'The theoretical foundations of Simpson''s Paradox, and how lurking variables and data aggregation produce contradictory statistical conclusions.'
seo_title: 'Simpson''s Paradox and Lurking Variables'
seo_type: article
tags:
- Correlation
- Data Quality
- Data Visualization
title: 'Simpson’s Paradox: Theoretical Foundations and Implications in Data Analysis'
---

Simpson's paradox is a reversal or substantial change between a marginal association and conditional associations after stratifying on a third variable. This paradox is widely misunderstood and can lead to erroneous conclusions if data is not analyzed carefully. It reveals the complexities of data aggregation and emphasizes the necessity of considering lurking variables to avoid false interpretations.

This article will explore the theoretical foundations of Simpson's Paradox, discuss real-world examples, and highlight why a rigorous approach to data analysis is essential in preventing misinterpretation.

## 1. The Origins and Theoretical Foundations of Simpson’s Paradox

Simpson’s Paradox was first described by Edward H. Simpson in 1951, although Karl Pearson and Udny Yule had identified similar phenomena as early as the late 19th century. The paradox is particularly intriguing because it defies our intuitive understanding of statistics and data relationships. It is essentially a problem of **data aggregation**, where the combined data tells a different story than when it is divided into meaningful subgroups.

### Understanding the Statistical Paradox

At the heart of Simpson’s Paradox is the concept of **marginal** versus **conditional** relationships in data:

- **Marginal relationships** are observed in the aggregate, ignoring any third variables.
- **Conditional relationships** are observed after accounting for the third variable (or stratifying the data by that variable).

The paradox arises when the marginal relationship (based on aggregated data) between two variables reverses or is significantly different from the conditional relationships within each subgroup.

Consider a simple example where two variables, $$ X $$ and $$ Y $$, are positively correlated in the aggregate data. However, when the data is broken down by a third variable, $$ Z $$, the correlation between $$ X $$ and $$ Y $$ becomes negative or disappears entirely in each subgroup of $$ Z $$. The marginal and conditional associations are both mathematically correct summaries of different distributions. Which one is scientifically relevant depends on the sampling and causal structure.

### Formal Mathematical Representation

Simpson’s Paradox can be formalized using **conditional probabilities** and the law of total probability. Suppose we have two binary variables, $$ A $$ and $$ B $$, and a third variable $$ C $$, which is categorical. The paradox can be described mathematically as follows:

Let:

$$
P(A | B) = \frac{a_1}{a_1 + b_1}, \quad P(A | \neg B) = \frac{a_2}{a_2 + b_2}
$$

Here, $$ a_1 $$ and $$ b_1 $$ represent the number of positive and negative outcomes for $$ A $$ when $$ B $$ is true, and $$ a_2 $$ and $$ b_2 $$ represent the outcomes when $$ B $$ is false.

However, if you split the data by the third variable $$ C $$ (such as subgroups $$ C_1 $$ and $$ C_2 $$), you might find:

$$
P(A | B, C_1) \neq P(A | B, C_2)
$$

The aggregated probabilities $$ P(A|B) $$ and $$ P(A|\neg B) $$ suggest one relationship, but the subgroup probabilities may tell a completely different story, leading to contradictory conclusions.

This mathematical framework highlights the crucial role of weighting and distribution in Simpson’s Paradox. Aggregating the data without considering these subtleties can lead to misleading interpretations.

### Causal Inference and Simpson’s Paradox

In the context of **causal inference**, Simpson’s Paradox often points to a **lurking variable** (the third variable) that affects both the outcome and the predictor. The third variable may be a confounder, a mediator, a collider, or simply a stratification variable. Conditioning is not automatically the correct move.

A classic example of this occurs in medicine, where an overall treatment might seem less effective due to an uneven distribution of patients across various risk categories. The treatment's efficacy, when viewed within individual risk categories, might actually be higher, but this is obscured when the data is aggregated.

## 2. Real-World Examples of Simpson’s Paradox

Simpson’s Paradox appears in various domains such as public health, economics, and social sciences. These examples demonstrate how intuitive conclusions drawn from data can be wrong when not carefully analyzed.

### Example 1: University Admission and Gender Bias

One of the most well-known examples of Simpson’s Paradox involves gender bias in university admissions. Suppose an analysis of overall admission data shows that men are accepted at a higher rate than women. At first glance, this seems to suggest bias against women. However, when the data is stratified by department, a different pattern emerges.

In the Berkeley admissions example, application patterns across departments with different admission rates generated a striking marginal association. The example is useful precisely because adjustment changes the interpretation; it should not be reduced to a generic rule that stratification always removes bias.

This example illustrates the dangers of relying on aggregated data, especially when subgroup characteristics (such as department competitiveness) differ dramatically.

### Example 2: Smoking and Health Outcomes

Consider a study examining the relationship between smoking and recovery rates from a particular illness. At first, the aggregated data might show that smokers have a higher recovery rate compared to non-smokers. However, upon stratifying the data by age, the paradox emerges: younger patients (who are more likely to recover regardless of smoking status) might disproportionately be smokers, while older patients, who are less likely to recover, tend to be non-smokers. Within each age group, smoking might actually correlate with worse recovery outcomes, but this is hidden in the aggregated data.

### Example 3: Treatment Efficacy in Medical Trials

In clinical trials comparing two treatments, aggregated data might suggest that Treatment A is more effective than Treatment B. However, when stratifying patients by age or health condition, it could become clear that Treatment B is more effective for specific subgroups, such as younger or healthier patients. This discrepancy can result from unequal distributions of patients across subgroups, leading to a paradoxical interpretation of the data.

## 3. Lurking Variables and the Role of Aggregated Data

The presence of **lurking variables** plays a critical role in Simpson’s Paradox. A lurking variable is a hidden factor that influences both the dependent and independent variables, thereby creating misleading correlations or masking the true relationships.

### Why Aggregation Leads to Misinterpretation

Aggregated data ignores subgroup heterogeneity, which can lead to skewed results. When data is aggregated across different groups, the relative sizes of these groups can dramatically alter the overall trends. This is especially true when the groups differ in characteristics such as size, response rate, or baseline risk.

In the context of Simpson’s Paradox, when subgroups have dramatically different weights (or sizes), the aggregated results may not accurately represent the true relationship within any single group. This emphasizes the importance of stratified or subgroup analysis, where the data is analyzed separately within meaningful divisions.

### Unequal Weighting and Its Effects

When combining data across subgroups, the paradox emerges from unequal weighting of group-specific probabilities. For instance, if one subgroup has far more data points than another, the aggregated result may reflect the trends of the larger subgroup, even if the smaller subgroup has a stronger internal correlation.

Simpson’s Paradox can also occur when the underlying distributions of the lurking variables differ between subgroups, which causes an unintended reversal of trends in the aggregate data.

## 4. Visualizing Simpson’s Paradox

Visualization can be a powerful tool for understanding Simpson’s Paradox. The paradox is often best illustrated through graphs that depict the relationships in both aggregated and stratified data.

### Scatter Plots and Grouping by Lurking Variables

A common way to visualize Simpson’s Paradox is through a scatter plot where the relationship between two variables appears to show a specific trend, such as a positive correlation. By color-coding or otherwise marking subgroups based on a third variable (the lurking variable), it becomes clear that the relationship between the two variables may differ dramatically within each subgroup.

For example, in an aggregated scatter plot, the relationship between variables $$ X $$ and $$ Y $$ might appear positive. However, when the data is stratified by the lurking variable $$ Z $$, the individual subgroup plots may show negative correlations, revealing the paradox.

### Bar Charts: Aggregated vs. Stratified Data

Bar charts can also be used to visualize Simpson’s Paradox by comparing the aggregated data to subgroup data. For instance, comparing overall success rates for two treatments in aggregate and then breaking it down by age group or health condition can reveal the paradox and demonstrate how misleading the aggregated data can be.

## 5. Implications for Data Analysts and Statisticians

Simpson’s Paradox serves as a crucial reminder for anyone working with data: aggregated data can be misleading, and deeper analysis is often required to uncover the true relationships between variables. Data analysts and statisticians must be cautious when interpreting trends and correlations, particularly in complex datasets where multiple variables are at play.

### Best Practices to Avoid Misinterpretation

1. **Do not stratify mechanically:** decide whether the third variable should be adjusted for from the study design and causal question. Conditioning on a collider can create bias rather than remove it.
2. **Use Visualizations:** Graphical representations of both aggregated and subgrouped data can help in identifying cases of Simpson’s Paradox.
3. **Understand the Data’s Context:** Simpson’s Paradox highlights the importance of understanding the context and underlying factors influencing your data. Never assume that the aggregated trends tell the whole story.
4. **Modeling Appropriately:** Use regression models or causal inference techniques to account for potential lurking variables and correctly model the relationships between variables.

## Conclusion

Simpson’s Paradox reveals the hidden complexities of data interpretation and statistical analysis. It challenges our assumptions about aggregated data and underscores the importance of considering lurking variables. By understanding the theoretical foundations of the paradox, recognizing its real-world implications, and adopting best practices in data analysis, we can avoid drawing misleading conclusions from our data.


## A weighted-average derivation

Suppose $Z$ indexes strata. Then

$$
P(Y=1mid X=x)
=
sum_z
P(Y=1mid X=x,Z=z)
P(Z=zmid X=x).
$$

Even if

$$
P(Y=1mid X=1,Z=z)
>
P(Y=1mid X=0,Z=z)
$$

for every stratum, the marginal comparison can reverse because the weights

$$
P(Z=zmid X=1)
$$

and

$$
P(Z=zmid X=0)
$$

differ.

That is the arithmetic engine of Simpson reversal.

## Causal graphs decide whether adjustment helps

Consider three possibilities.

### Confounder

$$
Zightarrow X,
qquad
Zightarrow Y.
$$

Conditioning on $Z$ can help recover a causal effect under appropriate assumptions.

### Mediator

$$
Xightarrow Zightarrow Y.
$$

Conditioning on $Z$ changes the estimand from a total effect toward a direct-effect question.

### Collider

$$
Xightarrow Zleftarrow Y.
$$

Conditioning on $Z$ can create a spurious association.

So the instruction “look for lurking variables and control for them” is incomplete.

## Standardization

When comparing groups with different stratum composition, one can standardize to a common distribution:

$$
P^ast(Y=1mid X=x)
=
sum_z
P(Y=1mid X=x,Z=z)
w_z,
$$

where the same weights $w_z$ are used for both exposure groups.

This makes the comparison explicit rather than leaving the marginal weights to differ automatically.

## References

- Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *Journal of the Royal Statistical Society: Series B*, 13(2), 238–241.
- Blyth, C. R. (1972). On Simpson's paradox and the sure-thing principle. *Journal of the American Statistical Association*, 67(338), 364–366.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
