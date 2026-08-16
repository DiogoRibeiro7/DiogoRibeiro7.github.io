---
author_profile: false
categories:
- Research
classes: wide
date: '2024-06-02'
header:
  image: /assets/images/data_science_8.avif
  og_image: /assets/images/data_science_8.avif
  overlay_image: /assets/images/data_science_8.avif
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.avif
  twitter_image: /assets/images/data_science_8.avif
redirect_from:
- '/healthcare education/statistical methods/data interpretation/nursing practice/professional development/explain_nurse/'
seo_description: How weighted moving averages and standard deviation work in health care, explained for nurses and clinical decision-making.
seo_title: Weighted Moving Average in Health Care
seo_type: article
tags:
- Healthcare
- Statistical Modeling
- Time Series
- Descriptive Statistics
title: Explaining Weighted Moving Average and Standard Deviation in Health Care
---

In nursing, understanding basic statistical concepts can enhance decision-making and patient care. Two important statistical measures are the weighted moving average and standard deviation. These tools help in analyzing trends and variability in patient data, making it easier to identify significant changes and patterns.

Both answer questions that come up on every shift. Is this patient's glucose actually trending upward, or is today's reading just noise? Is this blood pressure unusual for *this* patient, or only unusual compared with the ward average? Neither question can be answered by looking at a single number.

## What a Weighted Moving Average Does

The weighted moving average (WMA) smooths a series by averaging recent values, but unlike a plain average it does not treat every reading as equally informative. More recent observations receive larger weights, so the smoothed value tracks the patient's current state rather than being dragged backwards by readings from several days ago.

For a window of $n$ readings $x_1, \dots, x_n$ with weights $w_1, \dots, w_n$:

$$
\text{WMA} = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}.
$$

Dividing by the sum of the weights is what keeps the result in the same units as the original measurement. A common choice is linear weights, where the oldest reading in the window gets weight 1, the next gets 2, and so on up to $n$ for the newest.

The contrast with a simple moving average is the whole point. A simple average over five days responds to a real change only after that change has worked its way through most of the window. A weighted average starts moving immediately, because the newest reading carries the most influence.

## A Worked Example

Take ten days of fasting blood glucose from one patient, in mg/dL:

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Glucose | 100 | 105 | 98 | 110 | 102 | 107 | 95 | 115 | 108 | 101 |

Using a five-day window with linear weights 1, 2, 3, 4, 5 on days 6 through 10:

$$
\text{WMA} = \frac{1(107) + 2(95) + 3(115) + 4(108) + 5(101)}{1 + 2 + 3 + 4 + 5}
= \frac{107 + 190 + 345 + 432 + 505}{15} = \frac{1579}{15} \approx 105.3 .
$$

The simple average of those same five days is 105.2, which looks almost identical here — and that similarity is itself informative. When a series has no trend, weighting changes very little. The two diverge when something is actually moving, which is exactly when you want to notice.

Now the standard deviation across all ten days. The mean is

$$
\bar{x} = \frac{100 + 105 + 98 + 110 + 102 + 107 + 95 + 115 + 108 + 101}{10} = \frac{1041}{10} = 104.1 .
$$

The sample standard deviation uses $n-1$ in the denominator:

$$
s = \sqrt{\frac{\sum_i (x_i - \bar{x})^2}{n - 1}} = \sqrt{\frac{328.9}{9}} \approx 6.05 \ \text{mg/dL}.
$$

So this patient sits around 104 mg/dL, and a typical day lands within about 6 mg/dL of that. A reading of 115 is roughly 1.8 standard deviations above the mean — elevated, but not remarkable for this person. A reading of 140 would be nearly six standard deviations out and would deserve attention regardless of what the guideline threshold says.

```python
import numpy as np

glucose = np.array([100, 105, 98, 110, 102, 107, 95, 115, 108, 101])

weights = np.arange(1, 6)                      # oldest -> newest
window = glucose[-5:]
wma = np.sum(weights * window) / weights.sum()

mean = glucose.mean()
sd = glucose.std(ddof=1)                       # ddof=1 -> sample SD

print(f"weighted moving average : {wma:.1f} mg/dL")
print(f"simple average (5 days) : {window.mean():.1f} mg/dL")
print(f"overall mean            : {mean:.1f} mg/dL")
print(f"standard deviation      : {sd:.2f} mg/dL")
print(f"z-score of the 115 read : {(115 - mean) / sd:.2f}")
```

Note `ddof=1`. NumPy defaults to the population standard deviation, which divides by $n$ and slightly understates variability when you are working from a sample of readings rather than the complete record.

## Reading the Standard Deviation Clinically

Standard deviation measures how far a typical observation falls from the mean, in the original units. Low variability means a patient's measurements cluster tightly; high variability means they scatter.

The clinically useful move is to compare a patient against their own history rather than a population range. A resting heart rate of 92 is unremarkable in a population but may be a genuine signal in someone whose readings have sat at 68 ± 4 for a month. Population reference ranges answer "is this value normal for people"; a personal standard deviation answers "is this value normal for this person", and deterioration usually shows up in the second question first.

Two cautions matter at the bedside. Standard deviation assumes the readings are broadly symmetric around the mean, and it is sensitive to outliers, since deviations are squared before averaging. One transcription error of 1000 mg/dL will inflate the standard deviation enough to hide everything else. And a small number of readings gives an unreliable estimate: with five observations the standard deviation is itself very noisy, so treat early estimates as provisional.

## Choosing a Window and Weights

The window length is a trade-off, not a setting to be looked up. A short window reacts quickly to genuine change but also to noise; a long window is stable but slow. Match it to how fast the thing you are monitoring can actually move — post-operative observations change over hours, HbA1c over months.

An exponentially weighted moving average is often more convenient than linear weights, because it needs no fixed window at all:

$$
\text{EWMA}_t = \alpha x_t + (1 - \alpha)\,\text{EWMA}_{t-1}, \qquad 0 < \alpha \le 1 .
$$

Each new value is blended with the previous smoothed value, so the whole history contributes with geometrically decaying influence. A larger $\alpha$ tracks change faster; a smaller one smooths harder. This is the same machinery behind statistical process control charts used in clinical quality improvement, where an EWMA chart signals when a process has shifted away from its established baseline.

## Where These Measures Mislead

Smoothing is a form of information loss, and that is its purpose — but it also means a smoothed series will always lag a genuine turning point. If a patient deteriorates suddenly, the moving average will understate the change for exactly as long as the window is wide. Never let a smoothed value override a raw observation that looks alarming.

Averages also conceal direction. A patient oscillating between 70 and 140 mg/dL and a patient holding steady at 105 can produce the same mean, and even similar standard deviations if the oscillation is regular. Plot the raw series alongside the smoothed one; the shape carries information the summary cannot.

Finally, none of this establishes causation. A trend in the numbers tells you something changed, not what changed it, and the clinical explanation still has to come from assessment.

## Putting It Into Practice

Understanding and applying statistical measures like the weighted moving average and standard deviation can greatly enhance nursing practice. By using these tools, nurses can better interpret patient data, identify significant trends, and make informed decisions to improve patient outcomes.

The practical habit is modest: record enough readings to establish a personal baseline, look at the trend rather than the latest value alone, express deviations in that patient's own standard deviations, and keep the raw series visible next to whatever you have smoothed.

## References

- Montgomery, D. C. (2019). *Introduction to Statistical Quality Control* (8th ed.). Wiley.
- Benneyan, J. C., Lloyd, R. C., & Plsek, P. E. (2003). Statistical process control as a tool for research and healthcare improvement. *Quality and Safety in Health Care*, 12(6), 458-464.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Altman, D. G., & Bland, J. M. (2005). Standard deviations and standard errors. *BMJ*, 331(7521), 903.
