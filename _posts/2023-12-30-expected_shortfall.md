---
title: "Comparing Value at Risk (VaR) and Expected Shortfall (ES): A Data-Driven Analysis"
categories:
- Data Science
- Financial Risk Management
tags:
- Value at Risk
- Expected Shortfall
- Financial Crisis
- Risk Models
author_profile: false
seo_title: "VaR vs Expected Shortfall: A Data-Driven Analysis"
seo_description: "An in-depth analysis of Value at Risk (VaR) and Expected Shortfall (ES) as risk assessment models, comparing their performance during different market conditions."
excerpt: "A comprehensive comparison of Value at Risk (VaR) and Expected Shortfall (ES) in financial risk management, with a focus on their performance during volatile and stable market conditions."
classes: wide
---

## Risk Calculation Models: Value at Risk (VaR) vs. Expected Shortfall (ES)

Before the financial crisis of 2008, financial institutions predominantly used Value at Risk (VaR) to calculate minimum capital requirements for market risk. However, the crisis revealed significant deficiencies in this approach. Many institutions lacked the capital buffers necessary to withstand severe market shocks, prompting regulatory bodies to reassess risk management frameworks. In response, the Basel Committee on Banking Supervision introduced the Fundamental Review of the Trading Book (FRTB), advocating a shift from VaR to Expected Shortfall (ES) to provide a more robust risk measurement.

ES measures average losses exceeding the VaR level, offering a more comprehensive view of potential losses under adverse market conditions. This article compares VaR and ES, highlighting their strengths and weaknesses through the lens of the 2008 financial crisis and the more stable market environment of 2017.

## Defining Value at Risk (VaR) and Expected Shortfall (ES)

### Value at Risk (VaR)

VaR quantifies the maximum potential loss over a specified time period at a given confidence level. For example, a 10-day VaR at the 95% confidence level answers the question: "What is the maximum loss amount that the portfolio will not surpass over a 10-day period with a probability of 95%?"

Mathematically, VaR is the $$1 - \alpha$$ percentile of the loss distribution:

$$
\text{VaR}_{\alpha} = F_X^{-1}(1 - \alpha)
$$

where $$X$$ is a random variable representing losses, $$F_X$$ is the cumulative distribution function of losses over the given period, and $$\alpha$$ is the confidence level. In the distribution, the right tail represents the maximum potential loss, while the left tail indicates negative returns (earnings).

### Expected Shortfall (ES)

Expected Shortfall (ES) measures the expected loss when losses exceed the VaR threshold. In other words, it estimates the average loss beyond the VaR percentile, providing insight into the severity of losses in the tail of the distribution. ES is mathematically defined as:
$$
\text{ES}_{\alpha} = \mathbb{E}[X | X > \text{VaR}_{\alpha}]
$$
where $$\mathbb{E}$$ denotes the expected value. ES thus accounts for the risk of extreme losses, offering a more comprehensive risk assessment than VaR.

## Limitations and Risks of VaR and ES

Both VaR and ES have limitations, primarily due to their reliance on the underlying loss distribution model, which can be based on historical data or statistical distributions (e.g., normal distribution). Some key considerations include:

- **Tail Risk Estimation:** Using a normal distribution may underestimate tail risks, while historical data might not adequately represent future market conditions, particularly extreme events.
- **Model Assumptions:** Accurate risk measurement depends on the validity of model assumptions. Misestimating the distribution of returns can lead to incorrect risk assessments.
- **Scope of Risk:** VaR and ES quantify market risk but do not account for other risks such as liquidity, operational, and model risk. A comprehensive risk management approach should incorporate multiple risk measures and frameworks.

## Case Study: VaR and ES During Different Market Conditions

To illustrate the differences between VaR and ES, we compare their performance during two distinct market periods:

1. **Financial Crisis (April 2008 - April 2009):** A period marked by high volatility and severe market stress.
2. **Stable Market (January 2017 - December 2017):** A period characterized by low volatility and steady market growth due to quantitative easing.

### Visual Analysis

- **2008 Financial Crisis:** The DAX 30 index experienced rapid declines and significant daily swings, with daily returns occasionally dropping as much as -7%.
- **2017 Stable Market:** The index showed steady growth with relatively low volatility, and daily returns fluctuated within a narrow range around zero.

### Computational Analysis

We will calculate the VaR and ES for a portfolio with an initial value of $10,000 over a 10-day holding period at a 95% confidence level. The steps involve:

1. Calculating the daily return and daily loss.
2. Estimating the 95th percentile of the loss distribution for VaR.
3. Calculating ES by averaging the losses exceeding the VaR threshold.
4. Scaling the one-day risk measures to a 10-day period assuming a normal distribution of returns.

### Results and Insights

- **2008 Financial Crisis:** 
  - **Expected Return:** -$196.59
  - **VaR (95%):** $1,366.41
  - **ES:** $1,841.86
  - **Interpretation:** High volatility led to substantial potential losses. ES provided a more comprehensive estimate, indicating the average loss beyond the VaR threshold.

- **2017 Stable Market:**
  - **Expected Return:** $43.01
  - **VaR (95%):** $354.74
  - **ES:** $423.22
  - **Interpretation:** Lower volatility resulted in smaller potential losses. Both VaR and ES values were closer, reflecting a more stable environment.

### Conclusion

The comparison demonstrates that while both VaR and ES are valuable tools for risk assessment, ES provides a more detailed view of tail risks by considering the severity of losses beyond the VaR threshold. During volatile periods like the 2008 financial crisis, ES offers greater insights into potential losses, while VaR provides a baseline measure. In stable markets, both measures converge, indicating lower overall risk.

## Python Code: Calculating VaR and ES

Below is a Python code snippet that calculates the 10-day VaR and ES at the 95% confidence level for a simulated portfolio using historical returns.

```python
import numpy as np
import pandas as pd

# Simulated daily returns (e.g., historical data)
np.random.seed(42)
daily_returns = np.random.normal(0, 0.02, 252)  # 252 trading days in a year

# Portfolio initial value
initial_value = 10000

# Calculate daily portfolio values
daily_portfolio_values = initial_value * (1 + daily_returns).cumprod()

# Calculate daily losses
daily_losses = initial_value - daily_portfolio_values

# 1-day VaR at 95% confidence level
confidence_level = 0.95
var_95 = np.percentile(daily_losses, 100 * (1 - confidence_level))

# Expected Shortfall (ES) - average loss beyond VaR
es_95 = daily_losses[daily_losses > var_95].mean()

# Scaling to 10-day VaR and ES assuming normal distribution
holding_period = 10
var_95_10day = var_95 * np.sqrt(holding_period)
es_95_10day = es_95 * np.sqrt(holding_period)

print(f"1-Day VaR (95%): ${var_95:.2f}")
print(f"1-Day ES (95%): ${es_95:.2f}")
print(f"10-Day VaR (95%): ${var_95_10day:.2f}")
print(f"10-Day ES (95%): ${es_95_10day:.2f}")
```

### Explanation of the Code

The code simulates daily returns using a normal distribution. It calculates the portfolio's daily losses based on these returns. The 1-day VaR and ES at a 95% confidence level are calculated using the loss distribution. To obtain 10-day VaR and ES, the daily measures are scaled using the square root of the holding period, assuming a normal distribution.

### Conclusion

By providing insights into both maximum potential losses (VaR) and the severity of tail risks (ES), this analysis underscores the importance of incorporating ES into risk management strategies, particularly in volatile market conditions. However, careful consideration of model assumptions and input data is crucial for accurate risk measurement.
