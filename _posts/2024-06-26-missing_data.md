---
author_profile: false
categories:
- Epidemiology
- Data Science
- Medical Research
- Statistics
- Clinical Studies
classes: wide
date: '2024-06-26'
header:
  image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
subtitle: Strategies and Guidelines for Ensuring Valid Results
tags:
- Missing Data
- Multiple Imputation
- Complete Case Analysis
- Missing Data Mechanisms
- MCAR
- MAR
- MNAR
- Data Imputation
- Research Methodology
- Statistical Analysis
title: Handling Missing Data in Clinical Research
---

# Abstract

Missing data are a common issue in clinical research, potentially leading to biased results and reduced statistical power if not properly addressed. Effective handling of missing data is critical for the validity and reliability of research findings. This article provides a comprehensive review of the mechanisms of missing data, including Missing Completely at Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). It explores statistical methods for identifying the nature of missing data and discusses various approaches to handle missing data, with a particular emphasis on Multiple Imputation (MI) as the preferred method for MAR data. The article also addresses the limitations of traditional methods such as Complete Case Analysis (CCA) and Single Imputation, highlighting their potential to introduce bias. Through detailed guidelines and practical applications, this article aims to equip researchers with the necessary tools to manage missing data effectively, thereby enhancing the robustness and credibility of their clinical research outcomes.

## Introduction

In clinical research, the presence of missing data is almost inevitable. Whether due to participant drop-out, incomplete survey responses, or errors in data collection, missing data can significantly impact the validity and reliability of study findings. Properly handling missing data is crucial because ignoring it can lead to biased results, reduced statistical power, and potentially misleading conclusions.

### Importance of Handling Missing Data in Clinical Research

The integrity of clinical research depends on accurate and complete data. Missing data can distort the analysis, leading to erroneous estimates and weakened evidence. For instance, if data are missing non-randomly, the analysis might overlook important trends or relationships, compromising the study's conclusions. Therefore, researchers must implement robust strategies to address missing data to preserve the validity of their findings.

### Overview of Potential Biases Introduced by Ignoring Missing Data

Ignoring missing data can introduce several types of biases:

- **Selection Bias**: When the missing data are related to specific subgroups within the study population, leading to unrepresentative samples.
- **Attrition Bias**: Particularly relevant in longitudinal studies, where loss of participants over time can skew results.
- **Information Bias**: Occurs when the absence of data affects the measurement of key variables, leading to inaccurate estimates.

Each of these biases can significantly alter the study outcomes, highlighting the necessity for meticulous handling of missing data.

### Aim of the Article

This article aims to provide a comprehensive guide on identifying and managing missing data in clinical research. By exploring various missing data mechanisms, such as Missing Completely at Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR), the article will outline statistical methods for diagnosing the nature of missing data. Furthermore, it will discuss and compare different techniques for handling missing data, emphasizing the advantages of Multiple Imputation (MI) over traditional methods like Complete Case Analysis (CCA) and Single Imputation. Through detailed guidelines and practical examples, this article seeks to equip researchers with the necessary tools to enhance the accuracy and credibility of their research findings, even in the presence of missing data.

## Missing Data Mechanisms

Understanding the mechanisms behind missing data is crucial for determining the appropriate method for handling it. Rubin (1976) identified three primary mechanisms of missing data: Missing Completely at Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). Each mechanism has distinct characteristics and implications for data analysis.

### Missing Completely at Random (MCAR)

#### Definition and Examples

Missing Completely at Random (MCAR) occurs when the probability of data being missing is entirely unrelated to any observed or unobserved data. In other words, the missingness is purely random and does not depend on any specific value of the data. For instance, if a dataset includes blood pressure measurements and some data points are missing because participants accidentally skipped the measurement, this missingness would be considered MCAR.

Another example of MCAR might be a situation where survey responses are missing due to random technical issues, such as internet connectivity problems, that affect participants equally regardless of their characteristics or responses.

#### Implications for Data Analysis

When data are MCAR, the missingness is non-informative, meaning that the analysis conducted on the observed data is unbiased and can be considered representative of the entire dataset. Under the MCAR assumption, several methods can be used to handle missing data:

- **Complete Case Analysis (CCA)**: In this method, only cases with complete data are analyzed. Since the missing data are random, excluding them does not introduce bias, although it may reduce statistical power due to the smaller sample size.
- **Multiple Imputation (MI)**: While not strictly necessary under MCAR, MI can still be used to increase the efficiency of the analysis by retaining more data points and thus enhancing the statistical power.

It is important to test whether the MCAR assumption holds before deciding on the appropriate handling method. Statistical tests, such as Little’s MCAR test, can be employed to assess if the data are indeed MCAR. If the MCAR assumption is valid, simpler methods like CCA can be applied without introducing bias. However, if the data do not meet the MCAR criteria, more sophisticated techniques like MI are required to account for the potential biases associated with missing data.

Understanding and correctly identifying MCAR is a foundational step in the process of dealing with missing data in clinical research. It ensures that researchers apply the most appropriate and effective strategies to maintain the validity and reliability of their study outcomes.

### Missing at Random (MAR)

#### Definition and Examples

Missing at Random (MAR) occurs when the probability of data being missing is related to other observed data but not to the values of the missing data itself. This means that the missingness can be explained by other variables within the dataset. For example, in a clinical study, if the probability of missing blood pressure readings is higher for older participants, but not directly related to the blood pressure values themselves, the missingness would be considered MAR.

Another example might be a study on the effects of a new medication where participants with higher baseline anxiety levels are more likely to drop out, resulting in missing follow-up data on anxiety scores. The missingness is related to the observed baseline anxiety levels but not the follow-up anxiety scores themselves.

#### Strategies for Handling MAR Data

Handling MAR data requires more sophisticated techniques than those used for MCAR because the missingness is systematically related to other observed variables. The following strategies are commonly employed to handle MAR data effectively:

- **Multiple Imputation (MI)**: This method involves creating multiple sets of imputations for the missing values, analyzing each set separately, and then pooling the results to produce final estimates. MI accounts for the uncertainty of the imputed values by introducing random variation in the imputation process. The steps involved in MI are:
  - **Imputation Phase**: Missing values are replaced with multiple sets of plausible values generated from an imputation model that includes variables related to the missing data.
  - **Analysis Phase**: Each imputed dataset is analyzed using standard statistical methods.
  - **Pooling Phase**: The results from each analysis are combined to produce a single set of estimates and standard errors, reflecting the uncertainty due to missing data.

- **Maximum Likelihood (ML) Methods**: These methods estimate the parameters of interest by maximizing the likelihood function based on the observed data. ML methods make use of all available data and are particularly useful in the context of structural equation modeling and mixed-effects models.

- **Inverse Probability Weighting (IPW)**: This method involves weighting the observed cases by the inverse of their probability of being missing. It aims to create a pseudo-population in which the missing data mechanism is ignorable, allowing for unbiased estimation of parameters.

- **Full Information Maximum Likelihood (FIML)**: Similar to ML, FIML uses all available data to estimate model parameters directly, without imputing the missing values. It is particularly effective in the context of regression models and longitudinal data analysis.

Each of these methods has its advantages and limitations, and the choice of method depends on the specific context and data structure of the study. It is essential to include all relevant covariates in the imputation model to accurately reflect the relationships between variables and reduce bias.

By carefully applying these strategies, researchers can mitigate the impact of missing data and ensure the robustness and validity of their findings in studies where data are MAR.

### Missing Not at Random (MNAR)

#### Definition and Examples

Missing Not at Random (MNAR) occurs when the probability of data being missing is related to the unobserved data itself. In other words, the missingness depends on the value of the variable that is missing. This situation is particularly problematic because the mechanism behind the missing data cannot be explained by other observed variables in the dataset.

For example, consider a study measuring blood pressure where individuals with extremely high blood pressure readings are less likely to return for follow-up visits due to health concerns. Here, the missing blood pressure readings are directly related to the actual (unobserved) blood pressure values, making the data MNAR.

Another example might be a survey on income levels where individuals with very high or very low incomes are less likely to disclose their income, resulting in missing data that are dependent on the income values themselves.

#### Challenges in Addressing MNAR Data

Addressing MNAR data is significantly more challenging than dealing with MCAR or MAR data due to the inherent dependence of the missingness on the unobserved data. Some of the primary challenges include:

- **Unobservable Mechanism**: Since the missingness is related to the unobserved data, it is impossible to test directly whether the data are MNAR. This makes it difficult to determine the extent of bias introduced by the missing data.
- **Lack of Validity in Standard Methods**: Standard imputation methods, such as Multiple Imputation (MI) or Maximum Likelihood (ML), assume that data are at least MAR. These methods are not valid under MNAR because they do not account for the dependence of missingness on the unobserved values.
- **Bias in Estimates**: Analyses performed under the assumption of MCAR or MAR may yield biased results if the data are actually MNAR, leading to incorrect conclusions and potentially misleading evidence.

Despite these challenges, there are several strategies researchers can use to mitigate the impact of MNAR data:

- **Sensitivity Analysis**: Conducting sensitivity analyses to assess how robust the study findings are to different assumptions about the missing data mechanism. By comparing results under various scenarios, researchers can gauge the potential impact of MNAR on their conclusions.
- **Pattern-Mixture Models**: These models divide the data into different patterns based on the observed data and model the missingness within each pattern. While complex, they provide a way to account for MNAR mechanisms by explicitly modeling the relationship between the missing data and the observed data.
- **Selection Models**: These models explicitly model the process that leads to missing data. By incorporating assumptions about the missing data mechanism into the analysis, selection models can help address MNAR, although the results depend heavily on the validity of these assumptions.
- **Use of External Data**: Incorporating external data sources or prior knowledge can sometimes help address MNAR issues. For instance, using data from similar studies or known distributions to inform the imputation process can provide additional context for the missingness.

While no single method can fully overcome the challenges posed by MNAR data, combining these strategies and conducting thorough sensitivity analyses can help researchers better understand and address the biases introduced by MNAR. By acknowledging the limitations and carefully interpreting the results, researchers can enhance the robustness and credibility of their findings even in the presence of MNAR data.

## Exploring Missing Data

Once the presence of missing data is identified, it is essential to explore the nature of the missingness to determine the appropriate handling methods. One critical aspect of this exploration is to test whether the missing data are Missing Completely at Random (MCAR).

### Statistical Tests for MCAR

Several statistical tests can help determine if the missing data mechanism is MCAR. These tests assess whether the missingness is independent of both observed and unobserved data. Two commonly used tests for this purpose are t-tests and logistic regression analyses.

#### T-tests

T-tests can be used to compare the means of observed and missing data groups. By creating an indicator variable that flags whether a data point is missing, researchers can compare the means of this variable against other observed variables.

- **Procedure**:
  - Create a binary indicator variable (e.g., 0 for observed data, 1 for missing data).
  - Conduct independent samples t-tests to compare the means of the observed and missing groups for each variable.
  
- **Interpretation**:
  - If there is no significant difference between the means of the observed and missing groups for all variables, the data can be considered MCAR.
  - Significant differences suggest that the data may not be MCAR, indicating that the missingness is related to the observed data, thus potentially MAR or MNAR.

- **Example**: 
  - In a clinical study, a t-test might compare the mean age of participants with missing blood pressure data to those with observed blood pressure data. If the means are significantly different, the missingness is likely not MCAR.

#### Logistic Regression Analyses

Logistic regression analyses can be used to model the probability of missingness based on other observed variables. This method provides a more comprehensive assessment by considering multiple predictors simultaneously.

- **Procedure**:
  - Create a binary dependent variable indicating missingness (e.g., 0 for observed, 1 for missing).
  - Use logistic regression to predict the probability of missingness based on other observed variables.
  
- **Interpretation**:
  - If none of the predictor variables are significantly associated with the missingness indicator, the data may be considered MCAR.
  - Significant associations suggest that the missingness is related to the observed data, indicating MAR or MNAR.

- **Example**:
  - In a study, logistic regression might assess whether demographic variables (age, gender, income) predict the likelihood of missing follow-up data. Significant predictors would indicate the data are not MCAR.

By using these statistical tests, researchers can gain insights into the missing data mechanism and make informed decisions about the appropriate handling methods. Proper identification of MCAR is crucial as it allows for simpler and less biased analysis methods, such as Complete Case Analysis (CCA), whereas MAR or MNAR requires more sophisticated techniques like Multiple Imputation (MI) or sensitivity analyses.

### Little’s MCAR Test

#### Explanation and Application

Little’s MCAR test is a statistical test specifically designed to determine whether the missing data in a dataset are Missing Completely at Random (MCAR). This test provides a more rigorous and formal assessment compared to t-tests and logistic regression analyses.

#### Explanation

Little's MCAR test, also known as Little's Test for Missing Completely at Random, evaluates the null hypothesis that the data are MCAR. The test is based on comparing the means and covariances of the observed data and checking for significant differences.

- **Null Hypothesis (H0)**: The missing data are MCAR.
- **Alternative Hypothesis (H1)**: The missing data are not MCAR.

The test generates a chi-square statistic by comparing the expected and observed distributions of missing data patterns. If the p-value associated with the chi-square statistic is high (typically above 0.05), the null hypothesis cannot be rejected, indicating that the data are likely MCAR. Conversely, a low p-value suggests that the data are not MCAR.

#### Application

To apply Little’s MCAR test, researchers typically use statistical software such as SPSS, SAS, or R, which have built-in functions to perform the test. Below are the steps for applying Little’s MCAR test:

1. **Preparation**: Ensure that the dataset is appropriately formatted, with missing data indicated by a consistent symbol or code (e.g., NA).

2. **Performing the Test**:
   - **SPSS**:
     - Go to `Analyze` > `Missing Value Analysis`.
     - Select the variables and specify the options for the MCAR test.
     - SPSS will produce output including Little’s MCAR test statistic and p-value.
   - **R**:
     - Use the `BaylorEdPsych` package, which includes the `LittleMCAR` function.
     ```R
     install.packages("BaylorEdPsych")
     library(BaylorEdPsych)
     result <- LittleMCAR(data)
     print(result)
     ```
   - **SAS**:
     - Use the `PROC MI` procedure, which can include Little’s MCAR test as part of its output.
     ```sas
     proc mi data=your_data nimpute=0;
        var your_variables;
     run;
     ```

3. **Interpreting Results**:
   - Review the chi-square statistic and associated p-value from the output.
   - If the p-value > 0.05, the data are likely MCAR.
   - If the p-value ≤ 0.05, the data are not MCAR, indicating that another missing data mechanism (MAR or MNAR) might be present.

#### Example

Consider a clinical trial dataset with variables such as age, gender, treatment group, and follow-up measurements. Researchers want to test whether the missing follow-up measurements are MCAR.

- They input the dataset into SPSS and run Little’s MCAR test via the Missing Value Analysis menu.
- The output shows a chi-square value of 12.34 with a p-value of 0.08.
- Since the p-value is greater than 0.05, the researchers fail to reject the null hypothesis, suggesting that the missing follow-up measurements are likely MCAR.

Little’s MCAR test provides a robust and comprehensive method for evaluating the randomness of missing data. When the test indicates that data are MCAR, researchers can proceed with simpler analysis methods, confident that the missingness is non-informative. However, if the test suggests that data are not MCAR, researchers must consider more advanced techniques, such as Multiple Imputation (MI), to address the missingness appropriately.

### Distinguishing MAR from MNAR

#### Inherent Challenges and Limitations

One of the most significant challenges in handling missing data is distinguishing between Missing At Random (MAR) and Missing Not At Random (MNAR). The inherent difficulty lies in the fact that MNAR involves missingness that is directly related to the unobserved data, making it impossible to fully diagnose using the observed data alone.

- **Unobservable Mechanism**: Since MNAR depends on the values of the missing data themselves, there is no direct way to observe or measure the missingness mechanism. This makes it inherently challenging to determine whether the missing data mechanism is MAR or MNAR.
- **Assumption Dependence**: Analyses typically assume MAR because it is a more manageable and testable assumption compared to MNAR. However, this reliance on assumptions can lead to biased results if the true mechanism is MNAR.
- **Limited Diagnostic Tools**: While statistical tests can help identify MCAR, there are no definitive tests to distinguish MAR from MNAR. Researchers often rely on theoretical justifications and the nature of the data to hypothesize the missing data mechanism.

Due to these challenges, researchers must be cautious and adopt comprehensive strategies to account for the possibility of MNAR.

#### Importance of Sensitivity Analyses
Given the difficulty in distinguishing MAR from MNAR, sensitivity analyses play a crucial role in understanding the potential impact of different missing data mechanisms on study results. Sensitivity analyses involve varying the assumptions about the missing data mechanism and observing how these changes affect the study outcomes.

- **Purpose**: Sensitivity analyses assess the robustness of the study's conclusions under different missing data assumptions. They help determine the extent to which the results depend on the chosen mechanism for handling missing data.
- **Methods**:
  - **Pattern-Mixture Models**: These models divide the data into different patterns based on the observed data and analyze each pattern separately, allowing researchers to explore how different missingness patterns affect the results.
  - **Selection Models**: These models explicitly model the process that leads to missing data, incorporating assumptions about the missing data mechanism. By adjusting the parameters of the selection model, researchers can examine the sensitivity of their findings to different MNAR scenarios.
  - **Worst-Case and Best-Case Scenarios**: Researchers can analyze the data under extreme assumptions about the missingness. For example, in a worst-case scenario, missing values might be assumed to be the least favorable to the hypothesis, while in a best-case scenario, they might be assumed to be the most favorable.
  - **Delta Adjustment**: This method involves adjusting the imputed values by a certain amount (delta) to reflect different assumptions about the missingness. Researchers can then observe how the analysis results change with different delta values.

- **Example**: In a clinical trial studying the effectiveness of a new drug, researchers may suspect that patients with the most severe side effects are less likely to report their outcomes (MNAR). Sensitivity analyses can be conducted by varying the assumptions about the missing side effects data (e.g., assuming the worst-case scenario where all missing data represent severe side effects). By comparing the results under different scenarios, researchers can gauge the robustness of their conclusions regarding the drug's effectiveness.

- **Interpretation**: Sensitivity analyses provide a range of possible outcomes, helping researchers understand the potential biases introduced by different missing data mechanisms. If the study conclusions remain consistent across various scenarios, researchers can be more confident in their findings. Conversely, if the results vary widely, it indicates that the study is sensitive to the missing data mechanism, warranting cautious interpretation.

Distinguishing MAR from MNAR is a complex but essential task in clinical research. While it is often impossible to definitively classify the missing data mechanism, sensitivity analyses offer a valuable tool for exploring the impact of different assumptions and ensuring the robustness of study conclusions. By incorporating sensitivity analyses into their research, scientists can better understand the potential biases introduced by missing data and make more informed decisions about their findings.

## Methods to Deal with Missing Data

### Complete Case Analysis (CCA)

#### Description and Limitations

Complete Case Analysis (CCA), also known as listwise deletion, is a method for handling missing data where only the cases with complete data across all variables are included in the analysis. This approach is straightforward and easy to implement, as it involves simply excluding any records with missing values from the dataset.

- **Description**: In CCA, all observations (or cases) that have missing values for any of the variables included in the analysis are removed. The analysis is then conducted using only the subset of complete cases. This method assumes that the remaining data are representative of the entire dataset.
  
- **Limitations**:
  - **Reduction in Sample Size**: One of the primary drawbacks of CCA is the potential loss of a significant portion of the dataset, which can lead to a substantial reduction in sample size. This reduction decreases the statistical power of the study, making it harder to detect true effects.
  - **Bias Introduction**: If the data are not MCAR, CCA can introduce bias into the analysis. Specifically, if the missingness is related to the outcome or predictor variables, the complete cases may not be representative of the original population, leading to biased estimates.
  - **Inefficiency**: Even if the data are MCAR, CCA is generally less efficient than other methods, such as Multiple Imputation (MI), because it does not utilize all available information. This inefficiency can result in wider confidence intervals and less precise estimates.
  - **Not Feasible for High Missing Data**: In datasets with a high proportion of missing data, CCA may not be feasible as it could result in an unacceptably small sample size, undermining the reliability of the analysis.

#### Appropriate Situations for CCA

Despite its limitations, there are specific scenarios where CCA can be a valid and appropriate method for handling missing data:

- **MCAR Data**: CCA is most appropriate when the missing data mechanism is Missing Completely at Random (MCAR). In this scenario, the missingness is unrelated to any observed or unobserved variables, meaning that the complete cases are a random subset of the original dataset. Under MCAR, CCA can produce unbiased estimates.
  
- **Small Proportion of Missing Data**: When the proportion of missing data is very small (e.g., less than 5%), the impact of excluding incomplete cases may be minimal. In such cases, the loss of data and potential bias introduced by CCA are limited, making it a viable option.
  
- **Sensitivity Analyses**: CCA can be used as part of a sensitivity analysis to compare results with other missing data methods. By analyzing both complete cases and imputed datasets, researchers can assess the robustness of their findings across different handling methods.
  
- **Secondary Analyses**: In some situations, CCA might be used for secondary or exploratory analyses where the primary focus is not on the missing data but rather on generating preliminary insights. However, any conclusions drawn from such analyses should be interpreted with caution.

#### Example

Consider a clinical study examining the effects of a new medication on blood pressure. The dataset includes variables such as age, gender, baseline blood pressure, and follow-up blood pressure measurements. Suppose that 3% of the follow-up blood pressure measurements are missing at random due to technical errors in recording.

- **Application of CCA**: Researchers decide to use CCA, excluding the 3% of cases with missing follow-up measurements. Given the small proportion of missing data and the assumption that the missingness is MCAR, CCA is applied without significant concerns about bias.
- **Results Interpretation**: The analysis of the complete cases yields results that are considered unbiased and reliable. However, researchers also perform sensitivity analyses using Multiple Imputation (MI) to ensure that their findings are robust to different missing data assumptions.

While Complete Case Analysis (CCA) has notable limitations, it can be an appropriate method for handling missing data under specific conditions, particularly when the missing data mechanism is MCAR or when the proportion of missing data is very small. Researchers must carefully consider the context and potential biases before deciding to use CCA, and it is often advisable to complement CCA with more sophisticated methods like MI to ensure robust and reliable results.

### Single Imputation Methods

Single imputation methods are techniques where each missing value is replaced with a single estimated value, allowing the dataset to be analyzed as if it were complete. Common single imputation methods include mean imputation, regression imputation, and last observation carried forward (LOCF). However, these methods have significant drawbacks and are generally not recommended due to their potential to introduce bias and reduce the accuracy of statistical inferences.

#### Mean Imputation

- **Description**: Mean imputation involves replacing missing values with the mean of the observed values for that variable. For example, if a dataset has missing values for age, each missing age value would be replaced with the average age of all participants who provided their age.
- **Drawbacks**:
  - **Bias Introduction**: Mean imputation does not account for the natural variability in the data, leading to biased estimates of statistical parameters. It artificially reduces the variance of the imputed variable, making the data appear less variable than it actually is.
  - **Underestimation of Standard Errors**: By reducing variability, mean imputation leads to underestimation of standard errors, which can result in overly narrow confidence intervals and increased Type I error rates.
  - **Distortion of Relationships**: Mean imputation can distort relationships between variables. For instance, correlations between the imputed variable and other variables may be attenuated, leading to incorrect conclusions about the strength and direction of associations.

#### Regression Imputation

- **Description**: Regression imputation involves predicting the missing values based on a regression model that uses other observed variables. For example, if income data are missing, a regression model using variables such as age, education, and occupation might be used to predict and replace the missing income values.
- **Drawbacks**:
  - **Model Dependence**: The accuracy of regression imputation depends on the correctness of the model used for prediction. If the model is misspecified or if important predictors are omitted, the imputed values will be biased.
  - **Underestimation of Variability**: Similar to mean imputation, regression imputation does not fully capture the variability of the missing data, leading to underestimated standard errors and inflated Type I error rates.
  - **Distortion of Multivariate Relationships**: Regression imputation can preserve the relationship between the imputed variable and predictors in the model, but it can distort relationships with other variables not included in the model. This can result in biased estimates in multivariate analyses.

#### Last Observation Carried Forward (LOCF)

- **Description**: LOCF is a method often used in longitudinal studies where the last observed value of a variable is used to replace all subsequent missing values. For example, if a participant's blood pressure is measured at multiple time points and later measurements are missing, the last available blood pressure reading is used to fill in the missing values.
- **Drawbacks**:
  - **Assumption of No Change**: LOCF assumes that the value of the variable remains constant over time, which is often unrealistic in longitudinal studies. This assumption can lead to biased estimates, particularly if the variable is expected to change.
  - **Artificial Stability**: By carrying forward the last observation, LOCF artificially stabilizes the data, reducing variability and leading to underestimated standard errors and inflated Type I error rates.
  - **Potential for Bias**: LOCF can introduce systematic bias, especially if the missingness is related to changes in the variable over time. For example, if participants with worsening symptoms are more likely to drop out, LOCF will underestimate the true deterioration.

#### Why These Methods Are Not Recommended

Despite their simplicity, single imputation methods are generally not recommended due to their significant drawbacks:

- **Loss of Variability**: All single imputation methods fail to capture the true variability in the data, leading to biased parameter estimates and underestimated standard errors.
- **Introduction of Bias**: These methods often introduce systematic bias, particularly when the missing data mechanism is not MCAR.
- **Distortion of Relationships**: Single imputation methods can distort the relationships between variables, leading to incorrect conclusions about associations and effects.
- **False Precision**: By failing to account for the uncertainty associated with missing data, single imputation methods provide a false sense of precision, increasing the risk of Type I errors.

In contrast, more advanced methods like Multiple Imputation (MI) and Maximum Likelihood (ML) address these issues by incorporating the uncertainty associated with missing data and better preserving the relationships among variables. These methods are recommended for handling missing data in clinical research to ensure robust and unbiased results.

### Multiple Imputation (MI)

#### Overview and Benefits of MI

Multiple Imputation (MI) is a sophisticated method for handling missing data that addresses many of the limitations of single imputation methods. MI involves creating multiple versions of the dataset, each with different imputed values, to reflect the uncertainty about the missing data. These datasets are then analyzed separately, and the results are combined to produce final estimates.

- **Overview**: MI generates multiple imputed datasets by replacing missing values with a set of plausible values that are drawn from a distribution that reflects the uncertainty about the missing data. Each imputed dataset is then analyzed using standard statistical methods, and the results are combined (pooled) to produce estimates that account for the variability introduced by the imputation process.

- **Benefits**:
  - **Preserves Variability**: Unlike single imputation methods, MI preserves the natural variability in the data, leading to more accurate estimates of standard errors and confidence intervals.
  - **Reduces Bias**: MI reduces bias by incorporating the relationships between variables into the imputation model, making it more robust under MAR and even some MNAR scenarios.
  - **Valid Inference**: MI allows for valid statistical inference by appropriately accounting for the uncertainty associated with missing data.
  - **Flexibility**: MI can be applied to a wide range of data types and models, including continuous, categorical, and longitudinal data.

#### Phases of MI: Imputation, Analysis, and Pooling

The MI process consists of three main phases: imputation, analysis, and pooling.

1. **Imputation Phase**:
   - **Generating Imputed Datasets**: In this phase, each missing value is replaced with multiple plausible values to create several imputed datasets. The number of imputed datasets (m) typically ranges from 5 to 20.
   - **Imputation Model**: An imputation model is used to generate the plausible values based on the observed data. This model includes all variables that are related to the missing data and those that are predictive of the missing values.
   - **Adding Noise**: Random noise is added to the imputed values to reflect the uncertainty about the missing data, ensuring that each imputed dataset is slightly different.

2. **Analysis Phase**:
   - **Analyzing Each Imputed Dataset**: Each of the m imputed datasets is analyzed separately using the same statistical methods that would have been used if the data were complete.
   - **Standard Analysis**: The analyses performed can include any statistical procedures appropriate for the research question, such as regression analysis, ANOVA, or logistic regression.

3. **Pooling Phase**:
   - **Combining Results**: The results from the analyses of the m imputed datasets are combined to produce a single set of estimates. This involves pooling the estimates of parameters (e.g., means, regression coefficients) and their standard errors.
   - **Rubin’s Rules**: The pooled estimates are calculated using Rubin’s Rules, which account for both within-imputation variability (variability within each imputed dataset) and between-imputation variability (variability between the imputed datasets).

#### Recommended Imputation Techniques (e.g., Predictive Mean Matching)

Several techniques can be used for the imputation phase, depending on the nature of the data and the assumptions about the missing data mechanism. One of the most recommended techniques is Predictive Mean Matching (PMM).

- **Predictive Mean Matching (PMM)**:
  - **Description**: PMM is a semi-parametric method that imputes missing values by matching each missing value with the observed value that has the closest predicted value (from a regression model). The observed value is then used as the imputed value.
  - **Advantages**:
    - **Prevents Unrealistic Imputations**: By using observed values for imputation, PMM avoids imputing unrealistic values that are outside the range of observed data.
    - **Flexibility**: PMM can be used with continuous and categorical variables and is robust to departures from normality.
  - **Procedure**: 
    - Fit a regression model to predict the missing values based on observed data.
    - For each missing value, identify a set of observed values with predicted values close to the missing value’s predicted value.
    - Randomly select one of these observed values and use it as the imputed value.

- **Other Techniques**:
  - **Fully Conditional Specification (FCS)**: Also known as Multivariate Imputation by Chained Equations (MICE), FCS iteratively imputes each variable with missing data using a series of conditional models, one for each variable.
  - **Bayesian Methods**: These methods use Bayesian models to generate imputed values, incorporating prior distributions and updating them with the observed data.
  - **Multivariate Normal Imputation**: Assumes that the data follow a multivariate normal distribution and uses this assumption to generate imputed values. This method is appropriate for continuous data that approximately follow a normal distribution.

#### Example

Consider a clinical study examining the effects of a new drug on cholesterol levels, with some missing values in the follow-up measurements.

1. **Imputation Phase**:
   - Using PMM, the missing cholesterol values are imputed by matching each missing value with observed values that have similar predicted values based on a regression model using age, gender, baseline cholesterol, and treatment group.

2. **Analysis Phase**:
   - Each of the 10 imputed datasets is analyzed using linear regression to assess the effect of the new drug on cholesterol levels.

3. **Pooling Phase**:
   - The results from the 10 analyses are pooled using Rubin’s Rules to produce final estimates of the drug’s effect, along with appropriate standard errors and confidence intervals.

In conclusion, Multiple Imputation (MI) is a powerful and flexible method for handling missing data, providing more accurate and valid inferences compared to single imputation methods. By following the imputation, analysis, and pooling phases, and using recommended techniques like Predictive Mean Matching (PMM), researchers can effectively manage missing data and enhance the robustness of their study findings.

## Guidelines for Imputation

Effective imputation requires careful planning and consideration of various factors to ensure that the imputed data are as accurate and unbiased as possible. The following guidelines outline key principles for constructing and implementing a robust imputation strategy.

### Importance of a Well-Specified Imputation Model

#### Ensuring Accuracy and Validity

A well-specified imputation model is crucial for generating plausible and unbiased imputed values. The model should accurately reflect the relationships between variables and incorporate all relevant information.

- **Comprehensive Variable Selection**: Include all variables that are correlated with the missing data or that can predict the missing values. This helps to ensure that the imputed values are informed by the underlying data structure.
- **Avoiding Overfitting**: While it is important to include relevant variables, avoid overfitting the imputation model by including too many variables, especially those that do not contribute meaningful information.
- **Checking Model Assumptions**: Ensure that the assumptions underlying the imputation model are met. For example, if using a linear regression model for imputation, check for linearity, normality of residuals, and homoscedasticity.

### Inclusion of Outcome Variables in the Imputation Model

#### Mitigating Bias

Including the outcome variable in the imputation model is essential for reducing bias and improving the validity of the results. This practice ensures that the relationships between predictors and outcomes are maintained, even with imputed data.

- **Consistency Across Imputations**: By including the outcome variable, the imputation process accounts for the relationships between the outcome and the predictors, leading to more consistent and reliable estimates.
- **Bias Reduction**: Omitting the outcome variable from the imputation model can lead to biased estimates of the effects of predictors on the outcome, particularly if the missing data mechanism is related to the outcome.
- **Practical Application**: In a study analyzing the effect of a treatment on blood pressure, include the outcome (blood pressure) in the imputation model along with other predictors such as age, weight, and baseline blood pressure.

### Handling Situations with High Percentages of Missing Data

#### Strategies for Effective Imputation

When dealing with datasets that have a high percentage of missing data, special strategies are needed to ensure the reliability of the imputation process and the subsequent analysis.

- **Assess the Missing Data Mechanism**: Before proceeding with imputation, assess whether the missing data mechanism is MCAR, MAR, or MNAR. This assessment informs the choice of imputation method and helps identify potential biases.
- **Multiple Imputation (MI)**: Use Multiple Imputation, which creates several imputed datasets to reflect the uncertainty about the missing values. This method is particularly effective for handling high percentages of missing data as it provides a range of plausible values.
- **Model Complexity**: For datasets with high missingness, consider using more complex imputation models, such as Fully Conditional Specification (FCS) or Bayesian methods, which can better capture the relationships between variables.
- **Sensitivity Analysis**: Perform sensitivity analyses to evaluate the robustness of the results under different assumptions about the missing data mechanism. This helps to understand the impact of high missingness on the study conclusions.
- **Reducing Dimensionality**: In cases where a large number of variables have missing data, consider dimension reduction techniques such as principal component analysis (PCA) before imputation to simplify the data structure.

### Example

Consider a longitudinal study with 60% missing data for follow-up cholesterol measurements. The missingness is suspected to be related to baseline cholesterol levels (MAR).

1. **Well-Specified Imputation Model**:
   - Include variables such as age, gender, baseline cholesterol, diet, and exercise habits in the imputation model.
   - Ensure the imputation model assumptions are met, checking for linear relationships and normal distribution of residuals.

2. **Inclusion of Outcome Variables**:
   - Include follow-up cholesterol measurements (the outcome variable) in the imputation model to maintain the relationship with predictors like diet and exercise.

3. **Handling High Missingness**:
   - Use Multiple Imputation to create 20 imputed datasets, reflecting the uncertainty about the missing values.
   - Perform sensitivity analyses to assess how different assumptions about the missing data mechanism affect the study conclusions.
   - Consider using PCA to reduce the dimensionality if there are many predictors with missing data.

Researchers can improve the accuracy and reliability of their imputation process, ensuring that their analyses remain valid and unbiased, even in the presence of significant missing data.

## Case Studies and Practical Applications

### Examples of Successful MI Application in Clinical Research

#### Case Study 1: Cardiovascular Risk Assessment

In a large-scale study assessing cardiovascular risk factors, researchers faced missing data for several key variables, including cholesterol levels, blood pressure, and smoking status. The missing data were MAR, as they were related to demographic variables such as age and socioeconomic status.

- **Application of MI**: Researchers employed Multiple Imputation (MI) to handle the missing data. They included relevant covariates such as age, gender, diet, physical activity, and existing health conditions in the imputation model.
- **Outcomes**: The use of MI allowed for the retention of a larger sample size, enhancing the study’s statistical power. The final analysis produced more accurate and generalizable estimates of the associations between cardiovascular risk factors and outcomes, compared to complete case analysis.

#### Case Study 2: Diabetes Management Study

A longitudinal study on diabetes management included periodic measurements of HbA1c levels, body mass index (BMI), and adherence to medication. Missing data were present due to dropouts and incomplete follow-up visits, with an overall missing rate of 15%.

- **Application of MI**: Researchers used MI with Predictive Mean Matching (PMM) to impute missing values. The imputation model incorporated baseline HbA1c, BMI, age, gender, and medication adherence.
- **Outcomes**: MI provided valid estimates and maintained the integrity of the longitudinal data. Sensitivity analyses confirmed the robustness of the study’s conclusions about the effectiveness of the diabetes management interventions.

#### Case Study 3: Mental Health and Substance Use Survey

A cross-sectional survey on mental health and substance use faced significant missing data for sensitive questions about substance use, with a missing rate of 25%. The data were suspected to be MNAR due to the stigma associated with substance use.

- **Application of MI and Sensitivity Analysis**: Researchers applied MI and conducted sensitivity analyses using pattern-mixture models to assess the impact of different missing data assumptions. They included auxiliary variables like mental health status, demographic factors, and non-sensitive survey responses in the imputation model.
- **Outcomes**: While MI could not fully address the potential MNAR mechanism, sensitivity analyses provided a range of plausible outcomes, allowing researchers to better understand the potential biases and make informed conclusions.

### Common Pitfalls and How to Avoid Them

#### Pitfall 1: Ignoring the Missing Data Mechanism

- **Issue**: Failing to properly assess and account for the missing data mechanism can lead to biased results.
- **Solution**: Always perform preliminary analyses to understand the nature of the missing data (MCAR, MAR, MNAR). Use diagnostic tools like Little’s MCAR test and explore patterns of missingness with logistic regression.

#### Pitfall 2: Overfitting the Imputation Model

- **Issue**: Including too many variables or irrelevant variables in the imputation model can lead to overfitting, reducing the quality of imputations.
- **Solution**: Carefully select variables for the imputation model based on their relevance and relationship with the missing data. Avoid unnecessary complexity by including only those predictors that improve the imputation quality.

#### Pitfall 3: Ignoring the Uncertainty of Imputations

- **Issue**: Treating imputed values as if they were observed data without accounting for the uncertainty can result in underestimated standard errors and overly confident conclusions.
- **Solution**: Use Multiple Imputation (MI) instead of single imputation methods. MI accounts for the uncertainty by creating multiple imputed datasets and combining the results using Rubin’s Rules.

#### Pitfall 4: Inconsistent Imputation and Analysis Models

- **Issue**: Using different models for imputation and analysis can lead to inconsistencies and biased estimates.
- **Solution**: Ensure that the imputation model includes all variables used in the analysis model. Consistency between the imputation and analysis phases is crucial for valid results.

#### Pitfall 5: Neglecting to Perform Sensitivity Analyses

- **Issue**: Relying solely on one method of handling missing data without checking the robustness of the results can be problematic, especially if the missing data mechanism is uncertain.
- **Solution**: Conduct sensitivity analyses to explore how different assumptions about the missing data mechanism affect the results. This provides insights into the potential impact of missing data on study conclusions.

### Example of Avoiding Pitfalls

In a clinical trial investigating a new cancer treatment, researchers faced 20% missing data in follow-up tumor size measurements. They initially considered using CCA but recognized the potential for bias due to the high missing rate.

- **Strategy**: Researchers used MI with a well-specified model that included baseline tumor size, age, treatment group, and other relevant covariates. They performed sensitivity analyses to assess the impact of different missing data mechanisms, including MAR and MNAR scenarios.
- **Outcome**: The combined use of MI and sensitivity analyses provided robust and reliable estimates, avoiding the common pitfalls associated with handling missing data. The study’s conclusions about the treatment’s efficacy were validated across different assumptions.

By following these guidelines and being aware of common pitfalls, researchers can effectively manage missing data, ensuring the validity and reliability of their clinical research findings.

## Future Directions and Ongoing Research

### Current Trends in MI Research

Multiple Imputation (MI) continues to evolve as a vital technique for handling missing data in various fields, particularly in clinical research. Several current trends in MI research are aimed at improving its applicability, efficiency, and robustness.

#### 1. Advanced Computational Methods

- **Machine Learning Integration**: Incorporating machine learning algorithms into the imputation process to handle complex and high-dimensional data more effectively. Techniques such as random forests and neural networks are being explored for their potential to improve imputation accuracy.
- **Parallel Processing**: Utilizing parallel computing to speed up the MI process, making it feasible to handle large datasets and multiple imputed datasets more efficiently.

#### 2. Improved Imputation Models

- **Bayesian Imputation Methods**: Developing more sophisticated Bayesian imputation models that provide a flexible framework for incorporating prior information and handling different types of missing data mechanisms.
- **Multilevel and Longitudinal Data**: Enhancing imputation methods for multilevel (hierarchical) and longitudinal data to better capture the complexities of these data structures. Research is focusing on models that account for within-subject correlation over time and between-group variations.

#### 3. Software Development

- **User-Friendly Tools**: Creating more user-friendly software packages and interfaces to make MI accessible to a broader range of researchers. Tools like R’s `mice` and `Amelia` packages are continuously being updated with new features and improvements.
- **Integration with Mainstream Software**: Integrating MI capabilities into mainstream statistical software like SPSS, SAS, and Stata to streamline the workflow for researchers.

### Potential Areas for Further Development

While MI has made significant strides, several areas still require further development to enhance its application and reliability.

#### 1. Handling MNAR Data

- **Robust Imputation Models**: Developing imputation models that can better handle Missing Not At Random (MNAR) data. Current research is exploring methods to incorporate external information and sensitivity analysis techniques to address MNAR issues more effectively.
- **Sensitivity Analysis Methods**: Advancing sensitivity analysis methods to provide more robust tools for assessing the impact of MNAR data on study conclusions. This includes developing standardized protocols and software tools for conducting sensitivity analyses.

#### 2. Imputation of Complex Data Types

- **Mixed Data Types**: Improving imputation methods for datasets that include a mix of continuous, categorical, and ordinal data. Techniques that can seamlessly handle these mixed data types are crucial for many applied research settings.
- **High-Dimensional Data**: Enhancing methods to impute high-dimensional data, such as genetic data and large-scale surveys, where traditional imputation methods may struggle with the sheer volume and complexity of the data.

#### 3. Real-Time Data Imputation

- **Adaptive Imputation**: Developing real-time or adaptive imputation methods that can be applied as data are collected, particularly in clinical trials and longitudinal studies. This can help address missing data issues promptly and reduce the impact on subsequent analyses.
- **Automated Systems**: Creating automated systems for real-time imputation in electronic health records (EHRs) and other dynamic data environments, ensuring that analyses remain up-to-date and accurate as new data become available.

#### 4. Enhancing Interpretability and Transparency

- **Imputation Diagnostics**: Developing better diagnostic tools to assess the quality of imputed values and the adequacy of imputation models. These tools can help researchers understand the implications of their imputation choices and improve transparency.
- **Reporting Standards**: Establishing and promoting standardized reporting guidelines for MI, ensuring that studies transparently report their imputation methods, assumptions, and the impact of missing data on their findings.

### Example

Consider a large-scale epidemiological study investigating the genetic and environmental factors contributing to heart disease. This study involves high-dimensional genetic data, longitudinal health records, and survey responses with varying degrees of missing data.

1. **Advanced Computational Methods**:
   - Researchers employ machine learning algorithms to impute missing genetic data, leveraging the patterns identified in the high-dimensional data.
2. **Improved Imputation Models**:
   - Bayesian models are used to incorporate prior knowledge from previous studies, enhancing the imputation of environmental and behavioral data.
3. **Handling MNAR Data**:
   - Sensitivity analyses are conducted to evaluate the robustness of the findings under different MNAR assumptions, using newly developed software tools.
4. **Real-Time Data Imputation**:
   - Adaptive imputation methods are implemented to handle the continuously updated EHR data, ensuring the analyses reflect the most current information.
5. **Enhancing Interpretability and Transparency**:
   - Researchers use diagnostic tools to assess the quality of the imputed values and report their imputation methods and findings according to standardized guidelines.

By focusing on these future directions and ongoing research areas, the field of Multiple Imputation can continue to advance, providing researchers with powerful tools to handle missing data and improve the reliability of their findings.

## Conclusion

### Summary of Key Points

Handling missing data is a critical aspect of clinical research that, if not properly addressed, can lead to biased results, reduced statistical power, and invalid conclusions. This article has provided a comprehensive overview of the mechanisms of missing data, methods to explore and handle missing data, and guidelines for effective imputation.

- **Mechanisms of Missing Data**: We explored the three primary mechanisms of missing data—Missing Completely at Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). Understanding these mechanisms is essential for choosing the appropriate method to handle missing data.
- **Exploring Missing Data**: Statistical tests, such as t-tests, logistic regression analyses, and Little’s MCAR test, can help determine the nature of missingness. These tools are crucial for diagnosing whether data are MCAR, MAR, or MNAR.
- **Methods to Deal with Missing Data**:
  - **Complete Case Analysis (CCA)**: While easy to implement, CCA has significant limitations, particularly when data are not MCAR. It can lead to biased estimates and reduced sample size.
  - **Single Imputation Methods**: Methods like mean imputation, regression imputation, and last observation carried forward (LOCF) are generally not recommended due to their potential to introduce bias and underestimate variability.
  - **Multiple Imputation (MI)**: MI is a robust method that addresses the limitations of single imputation by generating multiple imputed datasets, analyzing each separately, and pooling the results. It preserves variability, reduces bias, and allows for valid statistical inferences.
- **Guidelines for Imputation**: The importance of a well-specified imputation model, inclusion of outcome variables in the imputation model, and strategies for handling high percentages of missing data were highlighted. These guidelines help ensure the accuracy and reliability of the imputation process.
- **Case Studies and Practical Applications**: Examples of successful MI applications in clinical research illustrated the effectiveness of MI in various contexts. Common pitfalls, such as ignoring the missing data mechanism and overfitting the imputation model, were discussed along with strategies to avoid them.
- **Future Directions and Ongoing Research**: Current trends in MI research include advanced computational methods, improved imputation models, software development, handling MNAR data, and enhancing interpretability and transparency. These advancements aim to make MI more accessible, efficient, and reliable.

### Final Recommendations for Researchers on Handling Missing Data

To ensure the validity and reliability of their findings, researchers should adopt comprehensive and systematic approaches to handle missing data. The following recommendations summarize best practices:

1. **Assess the Missing Data Mechanism**: Use diagnostic tools to determine whether the data are MCAR, MAR, or MNAR. This assessment is crucial for selecting the appropriate imputation method.
2. **Use Multiple Imputation (MI)**: Whenever possible, employ MI to handle missing data. MI provides robust and unbiased estimates by accounting for the uncertainty associated with missing values.
3. **Specify the Imputation Model Carefully**: Include all relevant variables in the imputation model, ensuring it reflects the relationships between variables. Avoid overfitting by selecting variables that contribute meaningful information.
4. **Include Outcome Variables in the Imputation Model**: Incorporating outcome variables in the imputation model helps maintain the integrity of the relationships between predictors and outcomes, reducing bias.
5. **Perform Sensitivity Analyses**: Conduct sensitivity analyses to explore how different assumptions about the missing data mechanism impact the results. This practice enhances the robustness and credibility of the study findings.
6. **Report Imputation Methods Transparently**: Follow standardized reporting guidelines to clearly describe the imputation methods, assumptions, and the impact of missing data on the results. Transparency in reporting enhances the reproducibility and reliability of research.

By following these recommendations, researchers can effectively manage missing data, thereby improving the accuracy and credibility of their clinical research outcomes. Addressing missing data thoughtfully and systematically ensures that the results are robust and reflective of the true underlying relationships in the data.

## References

1. Schafer JL, Graham JW. Missing data: our view of the state of the art. Psychol Methods. 2002;7(2):147-177. doi:10.1037/1082-989X.7.2.147

2. Rubin DB. Inference and missing data. Biometrika. 1976;63(3):581-592. doi:10.1093/biomet/63.3.581

3. Héraud-Bousquet V, Larsen C, Carpenter J, Desenclos JC, Le Strat Y. Practical considerations for sensitivity analysis after multiple imputation applied to epidemiological studies with incomplete data. BMC Med Res Methodol. 2012;12:73. doi:10.1186/1471-2288-12-73

4. Hsu CH, He Y, Hu C, Zhou W. A multiple imputation-based sensitivity analysis approach for data subject to missing not at random. Stat Med. 2020;39:3756-3771. doi:10.1002/sim.8693

5. Enders CK. Applied Missing Data Analysis. New York, NY: The Guilford Press; 2010.

6. Heymans MW, Eekhout I. Applied Missing Data Analysis with SPSS and RStudio. 2019. Available at https://bookdown.org/mwheymans/bookmi/. Accessed May 1, 2019.

7. Eekhout I, de Boer RM, Twisk JW, de Vet HC, Heymans MW. Missing data: a systematic review of how they are reported and handled. Epidemiology. 2012;23:729-732. doi:10.1097/EDE.0b013e3182576cdb

8. Austin PC, White IR, Lee DS, van Buuren S. Missing data in clinical research: a tutorial on multiple imputation. Can J Cardiol. 2021;37(9):1322-1331. doi:10.1016/j.cjca.2021.05.013

9. Groenwold RH, Donders AR, Roes KC, Harrell FE Jr, Moons KG. Dealing with missing outcome data in randomized trials and observational studies. Am J Epidemiol. 2012;175:210-217. doi:10.1093/aje/kwr302

10. Twisk JW, de Boer MR, de Vente W, Heymans MW. Multiple imputation of missing values was not necessary before performing a longitudinal mixed-model analysis. J Clin Epidemiol. 2013;66:1022-1028. doi:10.1016/j.jclinepi.2013.03.017

11. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York: John Wiley & Sons; 1987.

12. Buuren S. Flexible Imputation of Missing Data. 2nd ed. London, UK: Chapman and Hall/CRC; 2018.

13. Collins LM, Schafer JL, Kam CM. A comparison of inclusive and restrictive strategies in modern missing data procedures. Psychol Methods. 2001;6(4):330-351. doi:10.1037/1082-989X.6.4.330

14. Moons KG, Donders RA, Stijnen T, Harrell FE Jr. Using the outcome for imputation of missing predictor values was preferred. J Clin Epidemiol. 2006;59:1092-1101. doi:10.1016/j.jclinepi.2006.01.009

15. Lee KJ, Carlin JB. Multiple imputation for missing data: fully conditional specification versus multivariate normal imputation. Am J Epidemiol. 2010;171:624-632. doi:10.1093/aje/kwp425

16. White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. Stat Med. 2011;30:377-399. doi:10.1002/sim.4067

17. Marshall A, Altman DG, Holder RL, Royston P. Combining estimates of interest in prognostic modelling studies after multiple imputation: current practice and guidelines. BMC Med Res Methodol. 2009;9:57. doi:10.1186/1471-2288-9-57

18. van Buuren S, Groothuis-Oudshoorn K. MICE: Multivariate Imputation by Chained Equations in R. J Stat Softw. 2011;45:1-67. doi:10.18637/jss.v045.i03

19. Heymans MW. Miceafter: Data Analysis and Pooling After Multiple Imputation. R package version 0.1.0. 2021. Available at https://mwheymans.github.io/miceafter/. Accessed April 10, 2021.

20. Robitzsch A, Grund S. miceadds: Some additional multiple imputation functions, especially for ‘mice’. R package version 3.11-6. 2021. Available at https://CRAN.R-project.org/package=miceadds. Accessed October 18, 2021.

21. Heymans MW. Psfmi: Prediction Model Pooling, Selection and Performance Evaluation Across Multiply Imputed Datasets. R package version 1.0.0. 2021. Available at https://mwheymans.github.io/psfmi/. Accessed May 15, 2021.

22. Jolani S, Debray TP, Koffijberg H, van Buuren S, Moons KG. Imputation of systematically missing predictors in an individual participant data meta-analysis: a generalized approach using MICE. Stat Med. 2015;34:1841-1863. doi:10.1002/sim.6451

23. Resche-Rigon M, White IR. Multiple imputation by chained equations for systematically and sporadically missing multilevel data. Stat Methods Med Res. 2018;27(6):1634-1649. doi:10.1177/0962280216666564

24. Eekhout I, de Vet HC, Twisk JW, Brand JP, de Boer MR, Heymans MW. Missing data in a multi-item instrument were best handled by multiple imputation at the item score level. J Clin Epidemiol. 2014;67:335-342. doi:10.1016/j.jclinepi.2013.09.009

25. Eekhout I, de Vet HC, de Boer MR, Twisk JW, Heymans MW. Passive imputation and parcel summaries are both valid to handle missing items in studies with many multi-item scales. Stat Methods Med Res. 2018;27(4):1128-1140. doi:10.1177/0962280216656565

26. Brand J, van Buuren S, le Cessie S, van den Hout W. Combining multiple imputation and bootstrap in the analysis of cost-effectiveness trial data. Stat Med. 2019;38:210-220. doi:10.1002/sim.7991

27. Austin PC, Lee DS, Ko DT, White IR. Effect of variable selection strategy on the performance of prognostic models when using multiple imputation. Circ Cardiovasc Qual Outcomes. 2019;12:e005927. doi:10.1161/CIRCOUTCOMES.119.005927

28. Wahl S, Boulesteix AL, Zierer A, Thorand B, van de Wiel MA. Assessment of predictive performance in incomplete data by combining internal validation and multiple imputation. BMC Med Res Methodol. 2016;16:144. doi:10.1186/s12874-016-0244-6

29. Lee KJ, Tilling KM, Cornish RP, Little RJA, Bell ML, Goetghebeur E, et al. STRATOS Initiative. Framework for the treatment and reporting of missing data in observational studies: the treatment and reporting of missing data in observational studies framework. J Clin Epidemiol. 2021;134:79-88. doi:10.1016/j.jclinepi.2021.01.008

30. Little RJ, D'Agostino R, Cohen ML, Dickersin K, Emerson SS, Farrar JT, et al. The prevention and treatment of missing data in clinical trials. N Engl J Med. 2012;367:1355-1360. doi:10.1056/NEJMsr1203730

31. Sterne JA, White IR, Carlin JB, Spratt M, Royston P, Kenward MG, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. BMJ. 2009;338:b2393. doi:10.1136/bmj.b2393