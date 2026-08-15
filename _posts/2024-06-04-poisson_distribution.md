---
author_profile: false
categories:
- Data Science
- Statistics
- R Programming
- Probability and Statistics
- Data Analysis
classes: wide
date: '2024-06-04'
header:
  image: /assets/images/data_science_1.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_1.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_1.jpg
  twitter_image: /assets/images/data_science_7.jpg
seo_description: How to model count events with the Poisson distribution in R, from data preparation to fitting and assessing the model.
seo_title: Modeling Count Events with Poisson Distribution in R
seo_type: article
tags:
- Poisson distribution
- Count data
- Statistical modeling
- Time series analysis
- Event data
- Data preparation
- R code
- Probability
- P-value analysis
- Statistical testing
- R
title: Modeling Count Events with Poisson Distribution in R
---

In this article, we will explore how to model count events, such as activations of certain types of events, using the Poisson distribution in R. We will also discuss how to determine if an observed count belongs to the Poisson distribution.

## Introduction to Poisson Distribution

The Poisson distribution is often used to model the number of events occurring within a fixed interval of time or space when these events occur with a known constant mean rate and independently of the time since the last event. It is defined by a single parameter, λ (lambda), which represents the average rate of occurrence.

Formally, the probability of observing exactly $k$ events is

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \dots
$$

The distribution has a property that turns out to be its most useful diagnostic: its mean and variance are both equal to $\lambda$.

$$
E[X] = \operatorname{Var}(X) = \lambda .
$$

Three assumptions underpin the model, and each can fail in practice:

- **Constant rate.** Events occur at the same average rate throughout the interval. Hourly web traffic violates this badly, since the rate depends on time of day.
- **Independence.** One event does not change the probability of another. Clustered events, such as retries after a failure, violate this.
- **No simultaneity.** Two events cannot occur at exactly the same instant.

When the rate genuinely varies over time in a known way, an inhomogeneous Poisson process with rate $\lambda(t)$ is the appropriate generalisation.

## Step-by-Step Implementation

### Step 1: Data Collection and Preparation

First, we need to gather and prepare our event data. For this example, let's assume we have timestamps of events.

```r
# Load necessary libraries
library(dplyr)
library(tidyr)      # complete() lives here, not in dplyr
library(lubridate)

# Example data: event timestamps
data <- data.frame(
  timestamp = c('2024-06-01 08:00:00', '2024-06-01 08:15:00', '2024-06-01 09:30:00', 
                '2024-06-01 11:00:00', '2024-06-01 11:30:00', '2024-06-02 08:30:00', 
                '2024-06-02 10:00:00', '2024-06-02 10:15:00', '2024-06-02 10:45:00')
)

# Convert timestamps to datetime format and extract hour
data <- data %>%
  mutate(timestamp = ymd_hms(timestamp),
         hour = hour(timestamp))

# Count events per hour
event_counts <- data %>%
  count(hour) %>%
  complete(hour = 0:23, fill = list(n = 0))

print(event_counts)
```

Note the `complete()` step. Without it, hours with no events are simply absent from the counted data, and every subsequent statistic would be computed only over hours where something happened. That would bias λ upward substantially, because the zeros carry real information about the rate.

### Step 2: Fitting the Poisson Model

Next, we will calculate the rate parameter (λ) for the Poisson distribution based on our data.

```r
# Calculate the mean rate (lambda) for the Poisson distribution
lambda_estimate <- mean(event_counts$n)
cat("Estimated rate (lambda):", lambda_estimate, "events per hour\n")
```

The sample mean is not an arbitrary choice here: it is the maximum likelihood estimator for λ. Differentiating the Poisson log-likelihood and solving gives $\hat{\lambda} = \bar{x}$ exactly. Its standard error is $\sqrt{\hat{\lambda}/n}$, which is worth reporting alongside the estimate.

### Step 3: Using the Model for Predictions

We can use the Poisson distribution to predict the number of events in future intervals.

```r
# Predict the probability of observing 0 to 9 events in an hour
predicted_probs <- dpois(0:9, lambda_estimate)
names(predicted_probs) <- 0:9

cat("Predicted probabilities for 0 to 9 events occurring in an hour:\n")
print(predicted_probs)
```

### Step 4: Measuring if a Certain Count Belongs to the Distribution

To determine if an observed count belongs to the Poisson distribution, we calculate the p-value.

```r
# Example observed count
observed_count <- 3

# Calculate the probability of observing the given count
probability <- dpois(observed_count, lambda_estimate)
cat("Probability of observing exactly", observed_count, "events:", probability, "\n")

# Calculate the cumulative probability for observed counts less than the given count
cumulative_prob_lower <- ppois(observed_count - 1, lambda_estimate)

# Calculate the cumulative probability for observed counts greater than or equal to the given count
cumulative_prob_upper <- 1 - ppois(observed_count - 1, lambda_estimate)

# Two-sided p-value: probability of observing a count as extreme as or more extreme than the observed count
p_value <- 2 * min(cumulative_prob_lower, cumulative_prob_upper)
cat("Two-sided p-value for observing", observed_count, "events:", p_value, "\n")
```

This answers a narrow question: is *this single count* surprising, assuming the Poisson model is correct? It does not test the model itself, which is a different and usually more important question.

## Testing Whether the Data Is Poisson At All

Two checks address the model rather than an individual observation.

The **dispersion test** exploits the mean-equals-variance property directly. Define the dispersion index

$$
D = \frac{s^2}{\bar{x}} .
$$

Under a true Poisson process $D \approx 1$. Values well above 1 indicate overdispersion, and values below 1 indicate underdispersion, which arises when events are more regularly spaced than randomness would produce.

```r
mu  <- mean(event_counts$n)
s2  <- var(event_counts$n)
n   <- length(event_counts$n)

dispersion <- s2 / mu
cat("Dispersion index:", round(dispersion, 3), "\n")

# Formal test: (n-1) * D is approximately chi-square with n-1 df
stat <- (n - 1) * dispersion
p_disp <- 2 * min(pchisq(stat, n - 1), 1 - pchisq(stat, n - 1))
cat("Dispersion test p-value:", round(p_disp, 4), "\n")
```

The **chi-square goodness-of-fit test** compares the full observed frequency distribution against the expected Poisson frequencies:

```r
obs_table <- table(factor(event_counts$n, levels = 0:max(event_counts$n)))
k_vals    <- as.integer(names(obs_table))

expected <- dpois(k_vals, lambda_estimate) * n
expected[length(expected)] <- expected[length(expected)] +
  n * (1 - ppois(max(k_vals), lambda_estimate))   # absorb the upper tail

# Pool categories so every expected count is at least 5
keep <- expected >= 5
chi_stat <- sum((as.numeric(obs_table)[keep] - expected[keep])^2 / expected[keep])
df <- sum(keep) - 1 - 1        # minus one for the estimated lambda
cat("Chi-square:", round(chi_stat, 3), "on", df, "df,",
    "p =", round(1 - pchisq(chi_stat, df), 4), "\n")
```

Two details matter. Degrees of freedom lose an extra one because λ was estimated from the same data. And the test is unreliable when expected counts fall below about 5, hence the pooling. With only nine events, as in this toy dataset, neither test has meaningful power; these procedures need a few hundred intervals to be informative.

## When Poisson Fails: Overdispersion

In real count data, overdispersion is the rule rather than the exception. Website visits, insurance claims, and machine faults almost always show variance exceeding the mean, usually because the rate itself varies between periods rather than staying constant.

The standard remedy is the **negative binomial** distribution, which adds a dispersion parameter $\theta$ and allows

$$
\operatorname{Var}(X) = \mu + \frac{\mu^2}{\theta} .
$$

As $\theta \to \infty$ this collapses back to the Poisson. Fitting is a one-line change:

```r
library(MASS)
fit_pois <- glm(n ~ 1, family = poisson, data = event_counts)
fit_nb   <- glm.nb(n ~ 1, data = event_counts)

AIC(fit_pois, fit_nb)      # lower AIC indicates the better fit
```

An excess of zeros beyond even what a negative binomial predicts points to a zero-inflated or hurdle model, where a separate process decides whether any event can occur at all.

Ignoring overdispersion does not usually bias the estimate of the mean rate, but it makes standard errors far too small, producing confidence intervals that are much narrower than the data supports and significance where none exists.

## Conclusion

In this article, we demonstrated how to use the Poisson distribution to model count events in R. We covered the steps of data preparation, fitting the Poisson model, using the model for predictions, and measuring if an observed count belongs to the distribution. The Poisson distribution provides a useful framework for analyzing count data and making probabilistic predictions about future event occurrences.

By using the p-value, we can assess whether an observed count is consistent with the Poisson distribution. A low p-value suggests that the observed count is unlikely under the assumed Poisson model, indicating it may be an outlier or that the model's assumptions need to be reconsidered.

The habit worth building is to check dispersion before trusting any Poisson result. It is a single line of code, it catches the most common way this model fails, and it points directly at the alternative that fixes it.

## References

- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.
- Agresti, A. (2018). *An Introduction to Categorical Data Analysis* (3rd ed.). Wiley.
- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.). Springer.
