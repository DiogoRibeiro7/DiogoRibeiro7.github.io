# Logrank Test in Survival Analysis

## Basics of the Logrank Test

### Purpose

The Logrank test is used to test the null hypothesis that there is no difference between the survival probabilities of two groups at any time point.

### Key Concepts

1. **Observed Events (O)**: The actual number of events (e.g., deaths) that occur in each group at a specific time point.
2. **Expected Events (E)**: The number of events that would be expected in each group if there were no difference between the groups.
3. **Chi-Square Analysis**: The test uses a 2x2 contingency table to perform a Chi-Square test at each time point where an event occurs.

## How the Logrank Test Works

1. **Event Times**: For each time point where an event occurs, the Logrank test compares the observed and expected events in both groups.
   
2. **Calculation of Expected Events**:
   \[
   E_i = \frac{R_i \cdot (O_1 + O_2)}{N_1 + N_2}
   \]
   - \( E_i \): Expected events in group i.
   - \( R_i \): Risk set (number of subjects at risk) in group i at the event time.
   - \( O_1 \): Observed events in group 1.
   - \( O_2 \): Observed events in group 2.
   - \( N_1 \): Number of subjects at risk in group 1.
   - \( N_2 \): Number of subjects at risk in group 2.

3. **Chi-Square Statistic**:
   \[
   \chi^2 = \sum \frac{(O - E)^2}{E}
   \]
   - This is calculated for each event time and summed to get the overall Chi-Square statistic.

4. **P-Value**: The overall Chi-Square statistic is used to determine the p-value, which indicates whether there is a significant difference between the survival distributions of the two groups.

## Handling Censoring

### Censoring

Censoring occurs when the event of interest (e.g., death) has not occurred for some subjects during the study period, either because they are lost to follow-up or the study ends before they experience the event.

### Impact on Logrank Test

The Logrank test accommodates censoring by adjusting the risk set (R) at each event time. Only those subjects who are still under observation and at risk of experiencing the event are included in the calculations for expected events.

## Significance vs. Effect Size

The Logrank test only assesses whether there is a significant difference between groups, not the size of the difference. To estimate the size of the difference, the Cox Proportional Hazards Model is often used.

## Summary

The Logrank test is essential for comparing survival distributions, especially in the presence of censoring. It builds on fundamental statistical concepts like Chi-Square tests, making a solid understanding of basic statistics crucial. Although it tells you if there is a difference between groups, it does not provide the magnitude of the difference, which requires further analysis using models like the Cox Proportional Hazards Model.

Feel free to ask further questions or explore more resources on this topic. Understanding these foundational principles will strengthen your grasp of more advanced survival analysis techniques.
