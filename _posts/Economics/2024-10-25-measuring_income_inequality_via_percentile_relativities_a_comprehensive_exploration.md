---
author_profile: false
categories:
- Economics
classes: wide
date: '2024-10-25'
excerpt: This article delves deeply into percentile relativity indices, a novel approach to measuring income inequality, offering fresh insights into income distribution and its societal implications.
header:
  image: /assets/images/data_science_16.jpg
  og_image: /assets/images/data_science_16.jpg
  overlay_image: /assets/images/data_science_16.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_16.jpg
  twitter_image: /assets/images/data_science_16.jpg
keywords:
- Percentile relativities
- Income inequality
- Gini coefficient
- Inequality measurement
- Statistical indices
- Statistics
- Social sciences
- Python
- python
seo_description: An in-depth analysis of percentile-based measures of income inequality, comparing traditional metrics like the Gini Index with novel approaches developed by Brazauskas, Greselin, and Zitikis.
seo_title: Measuring Income Inequality via Percentile Relativities
seo_type: article
summary: This article explores the measurement of income inequality through percentile relativities, comparing it with traditional metrics like the Gini Index. It discusses new inequality indices, their application in real-world data, and their policy implications.
tags:
- Income inequality
- Percentile relativities
- Gini index
- Statistical measures
- Inequality indices
- Statistics
- Social sciences
- Python
- python
title: 'Measuring Income Inequality via Percentile Relativities: A Comprehensive Exploration'
---

Income inequality has long been a topic of interest for economists, policymakers, and statisticians. As societies continue to evolve and grow, understanding how wealth and income are distributed among their populations becomes increasingly crucial for maintaining social equity and fairness. One of the prominent ways of quantifying inequality is through statistical measures, which have been refined over the years to capture both simple and complex aspects of distributional imbalances.

This article delves deeply into a particular approach to measuring income inequality—*percentile relativities*. This method, recently discussed by Brazauskas, Greselin, and Zitikis (2024), provides a new perspective on quantifying inequality by focusing on income comparisons between different percentiles of the population. By using percentile-based indices, we can offer fresh insights into how income distribution behaves across varying segments of society.

## Historical Context: Traditional Measures of Inequality

### Gini Index

The **Gini index** (or Gini coefficient), introduced by Corrado Gini in 1914, is perhaps the most well-known measure of income inequality. It is a single summary statistic that provides a snapshot of income distribution within a population. The Gini index ranges from 0 to 1, where 0 represents perfect equality (everyone has the same income) and 1 represents perfect inequality (all income is concentrated in a single individual).

Mathematically, the Gini index can be expressed as:

$$
G = 1 - \int_0^1 \left( \frac{\text{mean of those below Q(p)}}{\text{mean of all}} \right) 2p \, dp
$$

While the Gini index has been extensively used, it has certain limitations, particularly when attempting to capture more localized aspects of inequality (e.g., income distribution among specific groups, or the difference between the very rich and the very poor).

### Lorenz Curve

Another common approach is the **Lorenz curve**, which plots cumulative income or wealth against the cumulative population. The farther the curve is from the line of perfect equality, the greater the level of inequality. However, the Lorenz curve, like the Gini index, offers a broad-stroke view of inequality and may miss specific nuances of the distribution.

### Quantile-based Approaches

Quantile-based methods have gained attention in recent years, as they allow for a more granular comparison of income distributions across different percentiles of the population. Instead of summarizing the entire distribution in a single index, these methods focus on relative differences between groups at different points in the distribution, such as the comparison between the 10th and 90th percentiles, or the bottom 20% with the top 20%.

## Measuring Income Inequality via Percentile Relativities

The approach introduced by Brazauskas, Greselin, and Zitikis in their 2024 study offers a novel way to measure inequality through **percentile relativities**. The core idea of percentile relativities is to compare the income levels at different percentiles of the population in a systematic way, using a set of indices that provide insights into inequality across the income spectrum.

### The Three Percentile Relativity Indices

Brazauskas, Greselin, and Zitikis propose three primary strategies for comparing incomes across percentiles:

1. **Strategy 1:** Compare the median income of the poorest $$ p \times 100\% $$ of the population with the median income of the entire population. This leads to an index that reflects how the lower-income population compares with the overall population.

   The equality curve is defined as:

   $$
   \psi_1(p) = \frac{Q(p/2)}{Q(1/2)}
   $$

   Averaging over all values of $$ p $$ gives the inequality index:

   $$
   \Psi_1 = 1 - \int_0^1 \frac{Q(p/2)}{Q(1/2)} \, dp
   $$

2. **Strategy 2:** Compare the median income of the poorest $$ p \times 100\% $$ with the median income of the remaining $$ (1-p) \times 100\% $$ of the population. This focuses on how the income of the poorest compares to the non-poor population.

   The equality curve here is:

   $$
   \psi_2(p) = \frac{Q(p/2)}{Q(1/2 + p/2)}
   $$

   The inequality index derived from this is:

   $$
   \Psi_2 = 1 - \int_0^1 \frac{Q(p/2)}{Q(1/2 + p/2)} \, dp
   $$

3. **Strategy 3:** Compare the median income of the poorest $$ p \times 100\% $$ with the median income of the richest $$ p \times 100\% $$. This index measures the disparity between the poorest and the richest segments of the population.

   The equality curve is defined as:

   $$
   \psi_3(p) = \frac{Q(p/2)}{Q(1 - p/2)}
   $$

   The corresponding inequality index is:

   $$
   \Psi_3 = 1 - \int_0^1 \frac{Q(p/2)}{Q(1 - p/2)} \, dp
   $$

### Interpreting the Indices

Each of the three indices provides a unique perspective on income inequality. **$$\Psi_1$$** focuses on how the poorest compare to the entire population, making it particularly useful in assessing overall poverty. **$$\Psi_2$$** looks at the gap between the poorest and the non-poor, offering insights into the middle and lower income brackets. Finally, **$$\Psi_3$$** highlights the disparity between the poorest and the richest, making it useful for examining extremes in income distribution.

These indices are particularly valuable because they allow us to analyze income inequality at different points in the distribution, rather than providing a single summary statistic that might obscure important details.

## Income Transfers and the Impact on Inequality

Another key aspect discussed by Brazauskas, Greselin, and Zitikis is the concept of **income transfers** and their effect on inequality. Income transfers occur when wealth is redistributed from one segment of the population to another, typically through mechanisms such as taxation, welfare programs, or direct financial assistance.

For example, suppose an individual from the well-off segment of the population (H) transfers a certain amount of money to someone from the struggling segment (L). Mathematically, we can represent this as:

$$
L \overset{c}{\longleftarrow} H
$$

where $$ c $$ is the amount of money transferred from H to L. The authors explore how such transfers affect the different percentile relativity indices.

### Effects on $$\Psi_1$$

When a transfer occurs between a well-off individual and a struggling one, the index $$\Psi_1$$ generally decreases. This means that the income distribution becomes more equal, as the income of the poorest is brought closer to that of the median. However, if both individuals involved in the transfer are well-off or both are struggling, the index $$\Psi_1$$ remains unchanged.

### Effects on $$\Psi_2$$

The index $$\Psi_2$$ behaves similarly to $$\Psi_1$$, but it is sensitive to both the size and direction of the transfer. If a large enough amount is transferred between two well-off individuals, $$\Psi_2$$ may increase, indicating greater inequality among the non-poor. Conversely, a transfer from a well-off individual to a struggling one will decrease $$\Psi_2$$, as the gap between the poor and the non-poor narrows.

### Effects on $$\Psi_3$$

The index $$\Psi_3$$ responds to transfers between the poorest and the richest segments of the population. When wealth is transferred from the richest to the poorest, $$\Psi_3$$ decreases, indicating a reduction in the disparity between the two groups. On the other hand, transfers within the same group (either among the poor or the rich) do not affect $$\Psi_3$$.

## Empirical Application of the Percentile Relativity Indices

To illustrate the practical application of the percentile relativity indices, Brazauskas, Greselin, and Zitikis present a case study involving income data from the European Community Household Panel (ECHP) and the EU Statistics on Income and Living Conditions (EU-SILC). By applying the indices to income data from different countries, the authors are able to assess the levels of inequality across Europe and track changes over time.

### Case Study: European Capital Incomes in 2001 and 2018

The study examines income inequality in Europe using data from 2001 and 2018. The indices $$\Psi_1$$, $$\Psi_2$$, and $$\Psi_3$$ are calculated for each country, and the results reveal notable differences in income inequality across European nations. Some countries, such as Sweden and Denmark, exhibit relatively low levels of inequality, while others, like Italy and Greece, show higher levels of inequality.

Moreover, the study highlights how income inequality has evolved over time. In many European countries, income inequality increased between 2001 and 2018, particularly in Southern Europe. This trend is captured by the rising values of the percentile relativity indices, particularly $$\Psi_3$$, which measures the gap between the poorest and the richest.

### Policy Implications

The findings of the case study have important policy implications. Countries with high levels of inequality may need to implement stronger redistributive policies to reduce the gap between the rich and the poor. The percentile relativity indices provide a useful tool for policymakers to assess the effectiveness of these policies and identify areas where further intervention may be needed.

## Mathematical Properties and Extensions of the Percentile Relativity Indices

In addition to their practical applications, the percentile relativity indices have several important mathematical properties that make them a robust tool for measuring income inequality. These properties include:

- **Monotonicity:** The indices are monotonic with respect to income transfers, meaning that transfers from the rich to the poor will always reduce inequality, while transfers in the opposite direction will increase it.
- **Scale Invariance:** The indices are scale-invariant, meaning that they are unaffected by proportional changes in income (e.g., if all incomes double, the indices remain unchanged).
- **Symmetry:** The indices are symmetric with respect to the income distribution, meaning that they treat the rich and the poor in a balanced way, without favoring one group over the other.

The authors also discuss possible extensions of the indices to account for other factors that may influence income distribution, such as changes in the overall economic environment or shifts in demographic patterns.

## Conclusion: The Future of Inequality Measurement

The work of Brazauskas, Greselin, and Zitikis represents a significant contribution to the field of income inequality measurement. By introducing the concept of percentile relativities, they provide a new and flexible tool for analyzing income distribution that overcomes many of the limitations of traditional measures like the Gini index and Lorenz curve.

As income inequality continues to be a pressing issue in both developed and developing countries, the percentile relativity indices offer valuable insights that can help policymakers design more effective interventions to reduce inequality. Whether through redistributive taxation, social welfare programs, or other mechanisms, addressing income inequality will remain a key challenge for governments worldwide.

In future research, the percentile relativity approach could be expanded to explore other dimensions of inequality, such as wealth inequality, access to education, and healthcare disparities. By applying these tools to a broader range of socioeconomic factors, we can gain a deeper understanding of the root causes of inequality and work towards creating more equitable societies.

Ultimately, the percentile relativity indices represent a powerful and versatile framework for studying income inequality, offering both theoretical rigor and practical applicability. As such, they are likely to play an increasingly important role in the ongoing efforts to understand and address the complex issue of inequality in the 21st century.

## Appendix: Python Code for Percentile Relativity Indices

```python
import numpy as np

# Function to calculate quantile
def quantile(data, p):
    return np.percentile(data, p * 100)

# Function for Strategy 1
def psi_1(data, p):
    return quantile(data, p / 2) / quantile(data, 0.5)

def psi_1_index(data):
    p_values = np.linspace(0, 1, 100)
    psi_values = [psi_1(data, p) for p in p_values]
    return 1 - np.mean(psi_values)

# Function for Strategy 2
def psi_2(data, p):
    return quantile(data, p / 2) / quantile(data, 0.5 + p / 2)

def psi_2_index(data):
    p_values = np.linspace(0, 1, 100)
    psi_values = [psi_2(data, p) for p in p_values]
    return 1 - np.mean(psi_values)

# Function for Strategy 3
def psi_3(data, p):
    return quantile(data, p / 2) / quantile(data, 1 - p / 2)

def psi_3_index(data):
    p_values = np.linspace(0, 1, 100)
    psi_values = [psi_3(data, p) for p in p_values]
    return 1 - np.mean(psi_values)

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    data = np.random.lognormal(mean=0, sigma=1, size=1000)

    # Calculate the indices
    psi_1_result = psi_1_index(data)
    psi_2_result = psi_2_index(data)
    psi_3_result = psi_3_index(data)

    # Print results
    print(f"Psi 1 Index: {psi_1_result:.4f}")
    print(f"Psi 2 Index: {psi_2_result:.4f}")
    print(f"Psi 3 Index: {psi_3_result:.4f}")
```
