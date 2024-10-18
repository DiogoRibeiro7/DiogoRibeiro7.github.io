---
author_profile: false
categories:
- Statistics
classes: wide
date: '2019-12-31'
excerpt: Let's examine why multiple imputation, despite being popular, may not be
  as robust or interpretable as it's often considered. Is there a better approach?
header:
  image: /assets/images/data_science_20.jpg
  og_image: /assets/images/data_science_20.jpg
  overlay_image: /assets/images/data_science_20.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_20.jpg
  twitter_image: /assets/images/data_science_20.jpg
keywords:
- Multiple imputation
- Missing data
- Single stochastic imputation
- Deterministic sensitivity analysis
seo_description: Exploring the issues with multiple imputation and why single stochastic
  imputation with deterministic sensitivity analysis is a superior alternative.
seo_title: 'The Case Against Multiple Imputation: An In-depth Look'
seo_type: article
summary: Multiple imputation is widely regarded as the gold standard for handling
  missing data, but it carries significant conceptual and interpretative challenges.
  We will explore its weaknesses and propose an alternative using single stochastic
  imputation and deterministic sensitivity analysis.
tags:
- Multiple imputation
- Missing data
- Data imputation
title: A Deep Dive into Why Multiple Imputation is Indefensible
---

# Why Multiple Imputation is Indefensible: A Deep Dive

In the realm of statistical analysis, missing data is an issue that nearly every data analyst encounters at some point. The need to deal with incomplete data sets has spurred the development of various methods to impute or estimate the missing values. Among the most popular of these methods is **multiple imputation**. This technique, endorsed by many as the "gold standard" for handling missing data, is widely applied across various fields of research, including medical studies, social sciences, and economics. 

On the surface, multiple imputation appears to be a robust, theoretically sound approach that takes into account the uncertainty associated with missing data by generating multiple plausible versions of the data set and averaging the results across these imputed versions. Yet, beneath this seemingly reasonable approach lies a series of troubling theoretical issues, particularly in the interpretation of the results. The purpose of this article is to critically examine the fundamental problems with multiple imputation, particularly its lack of correspondence to empirical reality, and to propose an alternative approach that better preserves the falsifiability and interpretability of statistical inference.

## The Basics of Multiple Imputation

To understand the shortcomings of multiple imputation, it’s crucial to first comprehend how the method works. Multiple imputation can be summarized as follows:

1. **Creating Multiple Copies of the Data Set**: In the first step, multiple copies (often referred to as "imputed data sets") are created from the original observed data. Each of these copies contains the observed values from the original data set, but the missing values are imputed (or "filled in") using a statistical model that is based on the observed data.
   
2. **Imputing Missing Values Stochastically**: For each imputed data set, a slightly different model for the missing data is specified. These models incorporate randomness, meaning the imputed values for the missing data points differ across the copies. The randomness stems from a sampling process that reflects the uncertainty about the true values of the missing data.

3. **Applying the Analysis to Each Data Set**: The statistical analysis of interest (e.g., estimating a population mean, regression coefficients, etc.) is applied separately to each of the imputed data sets. 

4. **Combining the Results**: The results from the analyses of the imputed data sets are then pooled or combined to form a single result. This pooling typically involves taking the average of the estimates and accounting for the variability between them to reflect the uncertainty due to missing data.

On its surface, this approach appears sound and intuitive. By generating multiple plausible data sets and combining the results, the method ostensibly accounts for the uncertainty surrounding the missing values and provides a more robust estimate than single imputation methods. However, as we’ll explore next, this reasoning is flawed.

## The Fundamental Problems with Multiple Imputation

While the mechanics of multiple imputation seem to suggest a solid approach to addressing missing data, there are significant theoretical problems that make it indefensible as a method for statistical inference. These problems lie primarily in the interpretation of both the imputation process and the final results.

### 1. The Super-Population Fallacy

One of the core problems with multiple imputation is the conceptual framework it assumes for the imputation process. The method relies on the idea that we can treat the unknown parameters of the missing data model as though they were drawn from some sort of "super-population." This means that each imputed data set corresponds to a different draw of the missing data parameters, as if these parameters were random variables from a distribution.

However, this notion of a super-population of model parameters is purely hypothetical and does not correspond to any observable reality. In practice, there is no super-population from which the true parameters are sampled; the parameters are fixed but unknown. The process of stochastically sampling parameters for the missing data model does not reflect any real-world process but rather a theoretical construct that lacks empirical grounding. This disconnect from reality is one of the key reasons multiple imputation is indefensible.

In essence, multiple imputation treats the unknown, fixed true parameters of the missing data model as though they were subject to random variation, when in fact they are fixed and deterministic. This leads to a situation where the process of imputing missing values based on these randomly sampled parameters has no basis in objective reality.

### 2. Ambiguity in the Final Results: The Mean of What?

The second major problem with multiple imputation is the ambiguity surrounding the final pooled result. After performing the analysis on each imputed data set, the results are averaged to produce a single estimate. But what, exactly, does this pooled estimate represent?

In most cases, the final result is an average of the estimates obtained from each imputed data set. While averaging seems reasonable, it leads to an interpretational conundrum: what exactly is this average a mean of? Is it the mean of multiple hypothetical realities, none of which correspond to the actual data-generating process? Since the multiple imputation process is based on the generation of multiple, slightly different data sets—each based on a different missing data model—the final result represents an amalgamation of inferences drawn from several hypothetical models, none of which can be empirically verified or falsified.

This raises serious questions about the interpretability of the final result. In essence, multiple imputation asks us to accept the mean of estimates derived from different, unobservable, and unverifiable models as though it represents the true estimate we seek. But without a clear mapping to reality, this pooled result lacks a coherent interpretation. What we are left with is a mean of estimates that have been generated from hypothetical models, and it is not clear what this mean actually tells us about the data or the underlying population.

### 3. Unfalsifiability and Lack of Empirical Investigation

Science relies on the principle of falsifiability—the idea that a theory or hypothesis should be testable and capable of being proven wrong. However, multiple imputation introduces an element of unfalsifiability into the analysis. Since the multiple imputation process involves generating multiple hypothetical versions of the data set based on unobservable parameters, there is no way to empirically test or validate the missing data models used to generate the imputations.

In other words, the imputation models are based on assumptions that cannot be directly verified or tested against real data. The imputation process itself introduces a layer of hypothetical constructs that are removed from the observed data, making it impossible to investigate whether the models accurately reflect the true data-generating process. This lack of empirical grounding makes it difficult to engage in a meaningful scientific dialogue about the validity of the imputation process and the resulting analysis.

### 4. False Sense of Robustness

One of the reasons multiple imputation is so popular is that it is perceived as a robust method for handling missing data. By generating multiple plausible data sets and averaging the results, multiple imputation seems to provide a more reliable estimate than single imputation methods. However, this perception of robustness is illusory.

As discussed earlier, multiple imputation generates estimates based on different missing data models, none of which are directly connected to objective reality. The final result represents an average across these models, but this averaging process does not necessarily lead to a more accurate or reliable estimate. In fact, the pooled result may be misleading, as it is derived from multiple hypothetical models that cannot be verified. The perceived robustness of multiple imputation is, therefore, based on an illusion of model diversity, rather than a meaningful exploration of the uncertainty surrounding the missing data.

### 5. Computational Complexity and Practical Challenges

In addition to the theoretical problems discussed above, multiple imputation also presents practical challenges. The method requires the creation and analysis of multiple imputed data sets, which can be computationally intensive, particularly for large data sets or complex models. Moreover, the process of specifying multiple missing data models and combining the results adds layers of complexity to the analysis, which can make it difficult for researchers to fully understand or interpret the results.

While these practical challenges are not as fundamental as the theoretical issues, they do contribute to the overall difficulty of using multiple imputation effectively. The method's complexity can lead to errors or misinterpretations, particularly for researchers who are not well-versed in the nuances of imputation techniques.

## An Alternative Approach: Single Stochastic Imputation and Deterministic Sensitivity Analysis

Given the numerous problems with multiple imputation, it is worth considering alternative approaches for handling missing data. One such approach is **single stochastic imputation** combined with **deterministic sensitivity analysis**. This method offers several advantages over multiple imputation, particularly in terms of its interpretability, falsifiability, and connection to empirical reality.

### 1. Single Stochastic Imputation

Single stochastic imputation is a method in which missing values are imputed once, using a single statistical model based on the observed data. Unlike multiple imputation, which generates multiple imputed data sets, single stochastic imputation creates just one version of the data set with imputed values.

This approach has a clear interpretation: the missing data model represents an assumption about what the data would look like if no data were missing. The imputation model is based on the observed data, and the imputed values are drawn from a distribution that reflects the uncertainty about the missing values. However, the key difference from multiple imputation is that single stochastic imputation involves just one imputation model, and the imputed values correspond to a single, well-defined assumption about the data.

Because there is only one imputation model, the results of the analysis are directly tied to this model. This makes it easier to interpret the results, as there is no need to average across multiple hypothetical models. The final result reflects the analysis of a single, imputed data set, and the uncertainty about the imputed values is captured within the model itself.

### 2. Deterministic Sensitivity Analysis

One of the criticisms of single imputation methods is that they fail to account for the uncertainty associated with the missing data. However, this issue can be addressed through **deterministic sensitivity analysis**, which involves analyzing the data under multiple competing missing data models.

In deterministic sensitivity analysis, different missing data models are specified, and the data is imputed separately under each model. This generates multiple analysis results, each corresponding to a different assumption about the missing data. Rather than averaging these results, as in multiple imputation, the goal of sensitivity analysis is to examine the range of possible outcomes and assess the robustness of the results to different assumptions.

For example, researchers might specify both a "conservative" missing data model, which assumes that the missing values are more extreme than the observed values, and an "optimistic" model, which assumes that the missing values are more similar to the observed data. By comparing the results under these different models, researchers can quantify the uncertainty associated with the imputation process and make informed decisions about the robustness of their conclusions.

### 3. Interpretability and Falsifiability

One of the key advantages of single stochastic imputation combined with deterministic sensitivity analysis is that it maintains a clear connection to empirical reality. The imputation models are based on assumptions that can be explicitly stated and discussed, and the results of the analysis correspond to these assumptions. This makes it possible to engage in a meaningful scientific dialogue about the validity of the imputation process and the reasonableness of the assumptions.

Moreover, because the imputation models are based on specific, well-defined assumptions, the results of the analysis are falsifiable. If new data becomes available, or if the assumptions of the imputation model are found to be incorrect, the results can be revised or rejected. This stands in contrast to multiple imputation, which generates results based on hypothetical models that cannot be directly tested or falsified.

### 4. Simplicity and Transparency

Another advantage of the single stochastic imputation approach is its simplicity. By focusing on a single imputation model and analyzing the data under that model, researchers can avoid the complexity and computational burden of generating and analyzing multiple imputed data sets. This makes the method more transparent and easier to interpret, particularly for researchers who may not be experts in imputation techniques.

In addition, the use of deterministic sensitivity analysis allows researchers to explore the uncertainty surrounding the missing data in a straightforward and interpretable way. Rather than relying on the averaging of results across multiple imputation models, sensitivity analysis provides a clear picture of how the results might change under different assumptions. This enhances the transparency of the analysis and allows researchers to make more informed decisions about the robustness of their conclusions.

## Conclusion: A Call for Falsifiability in Science

In this article, we have critically examined the fundamental problems with multiple imputation and argued that it is an indefensible approach to handling missing data. While multiple imputation is widely regarded as the gold standard for missing data analysis, it suffers from serious theoretical flaws, particularly in its interpretation and connection to empirical reality.

As an alternative, we have proposed the use of single stochastic imputation combined with deterministic sensitivity analysis. This approach maintains a clear connection to observable reality, preserves the falsifiability of scientific inference, and provides a more transparent and interpretable framework for handling missing data.

By moving away from multiple imputation and adopting methods that are grounded in empirical reality, we can ensure that scientific research remains falsifiable and that the conclusions we draw are based on assumptions that can be tested and revised as new data becomes available. In the end, the goal of science is not to generate results that are merely plausible, but to generate results that are true. And to achieve that goal, we must rely on methods that keep science falsifiable.
