---
author_profile: false
categories:
- Statistics
- Bayesian Inference
classes: wide
date: '2024-11-15'
excerpt: This article critically examines the use of Bayesian posterior distributions as test statistics, highlighting the challenges and implications.
header:
  image: /assets/images/data_science_19.jpg
  og_image: /assets/images/data_science_19.jpg
  overlay_image: /assets/images/data_science_19.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_19.jpg
  twitter_image: /assets/images/data_science_19.jpg
keywords:
- Bayesian Posteriors
- Test Statistics
- Likelihoods
- Bayesian vs Frequentist
seo_description: A critical examination of Bayesian posteriors as test statistics, exploring their utility and limitations in statistical inference.
seo_title: Bayesian Posteriors as Test Statistics
seo_type: article
summary: An in-depth analysis of Bayesian posteriors as test statistics, examining their practical utility, sufficiency, and the challenges in interpreting them.
tags:
- Bayesian Posteriors
- Test Statistics
- Likelihoods
title: A Critical Examination of Bayesian Posteriors as Test Statistics
---

**Abstract:**  
In statistical inference, the Bayesian framework offers a probabilistic approach to updating beliefs in light of new evidence. However, the interpretation and application of Bayesian posteriors as test statistics have been subjects of debate. This article critically examines the use of Bayesian posterior distributions as mere test statistics, highlighting the implications of scaling and normalization, the challenges of interpreting integrated likelihoods, and the importance of sufficient statistics in decision-making. Through this examination, we aim to provide clarity on the practical utility of Bayesian posteriors and offer insights into the ongoing discourse between Bayesian and frequentist methodologies.

## Introduction

Statistical inference is a cornerstone of scientific research, providing tools and methodologies for making sense of data and drawing conclusions about underlying phenomena. Among the various frameworks available, the Bayesian approach has gained prominence for its intuitive probabilistic interpretation of uncertainty and its flexibility in incorporating prior information.

At the heart of Bayesian statistics is the posterior distribution, which represents the updated belief about a parameter after observing data. The posterior combines the prior distribution (representing initial beliefs) and the likelihood function (representing the data's influence) through Bayes' theorem. This approach contrasts with the frequentist perspective, which relies solely on the likelihood and views parameters as fixed but unknown quantities.

Despite its theoretical appeal, the practical application of Bayesian posteriors raises several questions. One critical viewpoint suggests that a Bayesian posterior is merely a test statistic, similar to a likelihood function, and that interpreting areas under its tail or ratios as evidence can be misleading. Furthermore, the scaling and normalization of likelihoods, often required in Bayesian analysis, may not provide meaningful probabilities and could complicate the inference process without offering substantial benefits.

This article delves into these concerns, exploring the nature of likelihoods and Bayesian posteriors, the role of test statistics in statistical inference, and the implications of scaling and normalizing likelihoods. We also discuss the importance of sufficient statistics and the challenges associated with interpreting integrated likelihoods. By critically examining these aspects, we aim to shed light on the limitations and potential pitfalls of treating Bayesian posteriors as test statistics and to provide guidance for practitioners in statistical analysis.

## The Nature of Likelihoods and Bayesian Posteriors

### Understanding Likelihoods

The likelihood function is a fundamental concept in statistical inference, representing the plausibility of different parameter values given observed data. Formally, for a statistical model $$f(x \mid \theta)$$, where $$x$$ denotes the data and $$\theta$$ the parameter(s), the likelihood function $$L(\theta \mid x)$$ is proportional to the probability of observing the data under each possible parameter value:

$$
L(\theta \mid x) = k \cdot f(x \mid \theta),
$$

where $$k$$ is a constant of proportionality that may be ignored when comparing relative likelihoods.

The likelihood function is not a probability distribution over $$\theta$$; rather, it serves as a tool for estimation and hypothesis testing. It allows us to identify parameter values that make the observed data most plausible.

### Bayesian Posterior Distributions

In Bayesian statistics, the posterior distribution represents the updated belief about the parameter $$\theta$$ after observing data $$x$$. It is derived using Bayes' theorem:

$$
p(\theta \mid x) = \frac{p(x \mid \theta) \cdot p(\theta)}{p(x)},
$$

where:

- $$p(\theta \mid x)$$ is the posterior distribution.
- $$p(x \mid \theta)$$ is the likelihood function.
- $$p(\theta)$$ is the prior distribution.
- $$p(x)$$ is the marginal likelihood, ensuring the posterior integrates to one.

The posterior combines the prior information with the likelihood, producing a probability distribution over $$\theta$$ that reflects both prior beliefs and observed data.

### Comparing Likelihoods and Posteriors
While both the likelihood function and the posterior distribution involve $$p(x \mid \theta)$$, they serve different purposes:

- **Likelihood Function:** Used in frequentist inference for parameter estimation and hypothesis testing, focusing on the data's information about $$\theta$$.
- **Posterior Distribution:** Provides a complete probabilistic description of $$\theta$$ given the data and prior beliefs, central to Bayesian inference.

When the prior $$p(\theta)$$ is non-informative or uniform, the posterior is proportional to the likelihood. This similarity has led some to argue that the posterior, in such cases, acts merely as a scaled version of the likelihood function.

### Interpretation and Misinterpretation
A key point of contention arises in interpreting the posterior distribution as a probability distribution over parameters. In frequentist statistics, parameters are fixed but unknown quantities, and probabilities are associated only with data or statistics derived from data. In contrast, Bayesian statistics treat parameters as random variables, allowing for probability statements about them.

Critics argue that when the posterior is viewed as a test statistic, especially in cases with non-informative priors, interpreting the area under its tail or its ratios as probabilities can be misleading. They contend that without meaningful prior information, the posterior does not provide genuine probabilistic evidence about $$\theta$$ but rather serves as a transformed version of the likelihood.

## Test Statistics and Their Role in Statistical Inference

### Definition of Test Statistics
A test statistic is a function of the sample data used in statistical hypothesis testing. It summarizes the data into a single value that can be compared against a theoretical distribution to determine the plausibility of a hypothesis. The choice of test statistic depends on the hypothesis being tested and the underlying statistical model.

### Properties of Good Test Statistics
An effective test statistic should have the following properties:

- **Sufficiency:** Captures all the information in the data relevant to the parameter of interest.
- **Consistency:** Converges to the true parameter value as the sample size increases.
- **Power:** Has a high probability of correctly rejecting a false null hypothesis.
- **Robustness:** Performs well under various conditions, including deviations from model assumptions.

### Sufficient Statistics
A sufficient statistic is a function of the data that contains all the information needed to estimate a parameter. Formally, a statistic $$T(x)$$ is sufficient for parameter $$\theta$$ if the conditional distribution of the data $$x$$ given $$T(x)$$ does not depend on $$\theta$$:

$$
p(x \mid T(x), \theta) = p(x \mid T(x)).
$$

Sufficient statistics are valuable because they reduce data complexity without losing information about the parameter. They play a crucial role in both estimation and hypothesis testing.

### Role in Decision-Making
In hypothesis testing, the decision to reject or fail to reject the null hypothesis is based on the test statistic's value relative to a critical value or significance level. The test statistic's distribution under the null hypothesis determines the probabilities associated with different outcomes.

Critics argue that the long-run performance of a test statistic, driven by the sufficient statistic, is what ultimately matters in statistical inference. Scaling or transforming a test statistic does not change its essential properties or its ability to make accurate decisions in the long run.

## Scaling and Normalization of Likelihoods

### Impact of Scaling on Test Statistics
Scaling and rescaling a test statistic involve multiplying or transforming it by a constant or function. While such transformations can change the numerical values of the statistic, they do not alter its fundamental properties or its distribution under repeated sampling.

For example, if $$Z$$ is a test statistic, then $$c \cdot Z$$ (where $$c$$ is a constant) is a scaled version of $$Z$$. The scaling factor $$c$$ can adjust the magnitude but does not affect the statistic's ability to distinguish between hypotheses.

### Long-Run Performance
The long-run performance of a test statistic refers to its behavior over many repetitions of an experiment. Key considerations include:

- **Type I Error Rate:** The probability of incorrectly rejecting the null hypothesis when it is true.
- **Type II Error Rate:** The probability of failing to reject the null hypothesis when it is false.
- **Power Function:** The probability of correctly rejecting the null hypothesis as a function of the true parameter value.

These properties are inherent to the test statistic's distribution and are not affected by scaling or normalization. Therefore, the focus should be on the statistic's ability to make accurate decisions rather than its scaled values.

### Importance of Sufficient Statistics
Since sufficient statistics capture all relevant information about the parameter, they determine the test statistic's long-run performance. Any transformation that retains sufficiency will preserve the statistic's essential properties.

Scaling and rescaling may be employed for convenience or interpretability but do not enhance the test statistic's efficacy. Consequently, excessive manipulation of the likelihood or posterior may be unnecessary if it does not contribute to better inference.

## Appropriate Lexicon and Notation in Presenting Likelihoods

### Misuse of Bayesian Terminology
Presenting scaled likelihoods or transformed test statistics using Bayesian lexicon and notation, such as invoking Bayes' theorem, can be misleading. This practice may suggest that the resulting quantities are probabilities when they are not.

For instance, integrating a scaled likelihood over a parameter space and interpreting the area as a probability disregards the fact that the likelihood function is not a probability distribution over parameters. Unlike probability densities, likelihoods do not necessarily integrate to one and can take on values greater than one.

### Need for Clarity and Precision
Using appropriate terminology and notation is crucial for clear communication in statistical analysis. Misrepresenting likelihoods as probabilities can lead to incorrect interpretations and conclusions.

Practitioners should:

- **Avoid Ambiguity:** Clearly distinguish between likelihoods, probability densities, and posterior distributions.
- **Use Correct Notation:** Employ notation that reflects the mathematical properties of the functions involved.
- **Provide Context:** Explain the meaning and purpose of scaled or normalized quantities to prevent misunderstandings.

### Emphasizing the Nature of the Likelihood
By presenting the likelihood function in its proper context, analysts can avoid overstating its implications. Recognizing that the area under a likelihood curve is not a probability helps maintain the distinction between likelihood-based inference and probabilistic statements about parameters.

## Challenges with Scaled, Normalized, and Integrated Likelihoods

### Difficulty in Obtaining Standard Distributions
When likelihoods are scaled, normalized, or integrated, the resulting quantities may not follow standard statistical distributions. This lack of standardization presents challenges:

- **Non-Standard Distributions:** The transformed likelihood may not conform to well-known distributions like the normal, chi-squared, or t-distributions.
- **Complexity in Inference:** Without a standard distribution, it becomes difficult to calculate critical values, p-values, or confidence intervals.
- **Analytical Intractability:** The mathematical expressions may be too complex to handle analytically, requiring numerical methods.

### Need for Transformations or Simulations
To make use of scaled or integrated likelihoods, further steps are often necessary:

- **Transformation to Known Distributions:** Applying mathematical transformations to map the likelihood to a standard distribution.
- **Monte Carlo Simulations:** Using computational methods to approximate the distribution of the statistic under repeated sampling.

These additional steps add complexity to the analysis and may not provide sufficient benefits to justify their use.

### Questioning the Practical Utility
Given the challenges associated with scaled and normalized likelihoods, one may question their practicality:

- **Added Complexity Without Clear Benefit:** The effort required to manipulate the likelihood may not yield better inference or understanding.
- **Alternative Methods Available:** Other statistical techniques may provide more straightforward solutions without the need for complicated transformations.
- **Risk of Misinterpretation:** Complex manipulations may lead to misunderstandings or incorrect conclusions if not properly handled.

The critical view suggests that using intractable test statistics complicates the analysis without offering significant advantages.

## The Critique of Bayesian Probability Interpretations

### Over-Interpretation of Bayesian Posteriors
Some critics argue that Bayesian practitioners may overstate the implications of posterior distributions by treating them as definitive probabilities about parameters. This perspective contends that without meaningful prior information, the posterior is merely a transformed likelihood and does not provide genuine probabilistic evidence.

The concern is that the probabilistic interpretation of the posterior may be unwarranted, especially when the prior is non-informative or subjective.

### Reliance on Sufficient Statistics
From a frequentist standpoint, the decision to retain or reject a hypothesis should rely on sufficient statistics derived from the data. The focus is on the long-run frequency properties of the test statistic, which are determined by the sufficient statistic.

The argument is that introducing Bayesian probabilities does not enhance the decision-making process if the sufficient statistic already captures all relevant information.

### Implications for Hypothesis Testing
The critique extends to the practical application of Bayesian methods in hypothesis testing:

- **Evidence vs. Decision:** Bayesian posteriors provide a probability distribution over parameters but may not directly inform the decision to accept or reject a hypothesis.
- **Subjectivity of Priors:** The influence of subjective priors can affect the posterior, potentially leading to conclusions that are not solely data-driven.
- **Complexity Without Added Value:** The additional complexity of Bayesian analysis may not translate into better decisions compared to methods based on sufficient statistics.

### Rebuttals and Counterarguments

#### Defense of Bayesian Methods
Proponents of Bayesian statistics offer several counterarguments:

- **Probabilistic Interpretation:** Bayesian methods provide a coherent probabilistic framework for inference, allowing for direct probability statements about parameters.
- **Incorporation of Prior Information:** The ability to include prior knowledge can enhance inference, especially in cases with limited data.
- **Flexibility and Adaptability:** Bayesian approaches can handle complex models and hierarchical structures more readily than frequentist methods.

#### Value in Decision-Making
Bayesian posteriors can inform decision-making through:

- **Credible Intervals:** Providing intervals within which the parameter lies with a certain probability.
- **Bayes Factors:** Offering a method for model comparison and hypothesis testing based on the ratio of marginal likelihoods.
- **Decision-Theoretic Framework:** Facilitating decision-making by incorporating loss functions and expected utility.

#### Addressing the Critique
- **Objective Priors:** Using objective or reference priors to minimize subjectivity.
- **Emphasis on Posterior Predictive Checks:** Assessing model fit and predictive performance rather than relying solely on the posterior distribution.
- **Recognition of Limitations:** Acknowledging the challenges and working towards methods that address concerns about interpretation and practicality.

## The Bayesian-Frequentist Debate

The debate between Bayesian and frequentist approaches is longstanding, with each offering strengths and weaknesses. Rather than viewing them as mutually exclusive, some suggest adopting a pragmatic stance:

- **Method Selection Based on Context:** Choosing the approach that best suits the problem at hand.
- **Hybrid Methods:** Combining elements of both frameworks to leverage their advantages.
- **Focus on Practical Outcomes:** Prioritizing methods that provide accurate and useful results for decision-making.

## Conclusion

The examination of Bayesian posteriors as test statistics reveals several important considerations in statistical inference:

- **Understanding the Nature of Likelihoods and Posteriors:** Recognizing that while likelihoods and posteriors are related, they serve different purposes and should be interpreted accordingly.
- **Importance of Sufficient Statistics:** Emphasizing that sufficient statistics capture all relevant information for parameter estimation and hypothesis testing, and that scaling or transforming test statistics does not change their inherent properties.
- **Clarity in Presentation:** Using appropriate lexicon and notation to prevent misinterpretation of likelihoods and posteriors, and avoiding the misuse of Bayesian terminology when it is not warranted.
- **Practical Challenges with Complex Transformations:** Acknowledging that scaled, normalized, and integrated likelihoods may introduce unnecessary complexity without providing clear benefits, and that their distributions may not be standard or tractable.
- **Critique of Over-Interpretation:** Considering the argument that Bayesian probabilities, especially in the absence of meaningful priors, may not offer additional value over frequentist methods relying on sufficient statistics.
- **Rebuttals and Balanced Perspectives:** Recognizing the strengths of Bayesian methods, including their probabilistic framework and ability to incorporate prior information, while also acknowledging the importance of context and practical utility.

For practitioners, the key takeaway is to critically assess the methods used in statistical inference, ensuring that they are appropriate for the problem and that their interpretations are valid. Whether adopting Bayesian or frequentist approaches, the focus should remain on making accurate, reliable decisions based on the data and the underlying statistical principles.

By maintaining clarity, precision, and a thorough understanding of the tools at our disposal, we can navigate the complexities of statistical inference and contribute to sound scientific research and decision-making.

## Recommendations for Practitioners

- **Evaluate the Necessity of Complex Transformations:** Before scaling or normalizing likelihoods, consider whether these steps add value to the analysis.
- **Use Appropriate Terminology:** Ensure that the language and notation used accurately reflect the statistical concepts involved.
- **Focus on Sufficient Statistics:** Leverage sufficient statistics to capture all relevant information and base decisions on their properties.
- **Be Mindful of Prior Information:** When using Bayesian methods, carefully select priors and assess their influence on the posterior distribution.
- **Consider the Practical Implications:** Choose statistical methods that are tractable and provide clear, actionable insights.
- **Stay Informed of Methodological Debates:** Engage with the ongoing discourse between Bayesian and frequentist methodologies to enhance understanding and application.

## Appendix

### Python Code for Bayesian Posterior and Test Statistics

```python
# Import necessary libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Define a prior distribution (uniform prior)
def prior(theta):
    return 1 if 0 <= theta <= 1 else 0

# Define the likelihood function
def likelihood(theta, data):
    return np.prod(stats.binom.pmf(data, n=1, p=theta))

# Define the posterior using Bayes' theorem
def posterior(theta, data):
    return likelihood(theta, data) * prior(theta)

# Normalize the posterior to ensure it integrates to 1
def normalized_posterior(data):
    theta_range = np.linspace(0, 1, 100)
    posterior_values = np.array([posterior(theta, data) for theta in theta_range])
    normalization_constant = np.trapz(posterior_values, theta_range)
    return theta_range, posterior_values / normalization_constant

# Plot the posterior distribution
def plot_posterior(data):
    theta_range, norm_posterior = normalized_posterior(data)
    plt.plot(theta_range, norm_posterior, label='Posterior')
    plt.title('Posterior Distribution')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Simulate data (e.g., Bernoulli trials with true parameter 0.7)
data = np.random.binomial(1, 0.7, size=20)

# Plot the posterior for the given data
plot_posterior(data)

# Compute the test statistics (mean, variance, etc.)
mean_posterior = np.trapz(theta_range * norm_posterior, theta_range)
variance_posterior = np.trapz((theta_range - mean_posterior) ** 2 * norm_posterior, theta_range)
credible_interval = np.percentile(theta_range, [2.5, 97.5])

# Print posterior mean, variance, and credible interval
print(f"Posterior Mean: {mean_posterior}")
print(f"Posterior Variance: {variance_posterior}")
print(f"95% Credible Interval: {credible_interval}")

# Frequentist Test Statistics Example: Likelihood Ratio Test
def likelihood_ratio_test(data, theta_null, theta_alt):
    ll_null = np.sum(np.log(stats.binom.pmf(data, n=1, p=theta_null)))
    ll_alt = np.sum(np.log(stats.binom.pmf(data, n=1, p=theta_alt)))
    return 2 * (ll_alt - ll_null)

# Perform a likelihood ratio test for two values of theta
lr_stat = likelihood_ratio_test(data, theta_null=0.5, theta_alt=0.7)
p_value = stats.chi2.sf(lr_stat, df=1)
print(f"Likelihood Ratio Test Statistic: {lr_stat}")
print(f"p-value: {p_value}")
```

### R Code for Bayesian Posterior and Test Statistics

```r
# Load necessary libraries
library(ggplot2)

# Define a uniform prior
prior <- function(theta) {
  ifelse(theta >= 0 & theta <= 1, 1, 0)
}

# Define the likelihood function (Bernoulli trials)
likelihood <- function(theta, data) {
  prod(dbinom(data, size = 1, prob = theta))
}

# Define the posterior function using Bayes' theorem
posterior <- function(theta, data) {
  likelihood(theta, data) * prior(theta)
}

# Normalize the posterior distribution
normalized_posterior <- function(data) {
  theta_range <- seq(0, 1, length.out = 100)
  posterior_values <- sapply(theta_range, posterior, data = data)
  normalization_constant <- sum(posterior_values) * diff(range(theta_range)) / length(theta_range)
  list(theta_range = theta_range, posterior_values = posterior_values / normalization_constant)
}

# Plot the posterior distribution
plot_posterior <- function(data) {
  result <- normalized_posterior(data)
  df <- data.frame(theta = result$theta_range, posterior = result$posterior_values)
  
  ggplot(df, aes(x = theta, y = posterior)) +
    geom_line() +
    labs(title = "Posterior Distribution", x = "Theta", y = "Density") +
    theme_minimal()
}

# Simulate data (e.g., Bernoulli trials with true parameter 0.7)
set.seed(123)
data <- rbinom(20, size = 1, prob = 0.7)

# Plot the posterior for the given data
plot_posterior(data)

# Compute posterior mean, variance, and credible interval
posterior_summary <- function(data) {
  result <- normalized_posterior(data)
  theta_range <- result$theta_range
  posterior_values <- result$posterior_values
  
  mean_posterior <- sum(theta_range * posterior_values) * diff(range(theta_range)) / length(theta_range)
  variance_posterior <- sum((theta_range - mean_posterior)^2 * posterior_values) * diff(range(theta_range)) / length(theta_range)
  credible_interval <- quantile(theta_range, c(0.025, 0.975))
  
  list(mean = mean_posterior, variance = variance_posterior, credible_interval = credible_interval)
}

# Compute and print posterior summary statistics
summary_stats <- posterior_summary(data)
print(paste("Posterior Mean:", summary_stats$mean))
print(paste("Posterior Variance:", summary_stats$variance))
print(paste("95% Credible Interval:", paste(summary_stats$credible_interval, collapse = " - ")))

# Frequentist Test Statistics Example: Likelihood Ratio Test
likelihood_ratio_test <- function(data, theta_null, theta_alt) {
  ll_null <- sum(dbinom(data, size = 1, prob = theta_null, log = TRUE))
  ll_alt <- sum(dbinom(data, size = 1, prob = theta_alt, log = TRUE))
  test_stat <- 2 * (ll_alt - ll_null)
  p_value <- 1 - pchisq(test_stat, df = 1)
  list(test_stat = test_stat, p_value = p_value)
}

# Perform a likelihood ratio test for two values of theta
lr_test_result <- likelihood_ratio_test(data, theta_null = 0.5, theta_alt = 0.7)
print(paste("Likelihood Ratio Test Statistic:", lr_test_result$test_stat))
print(paste("p-value:", lr_test_result$p_value))
```

### Scala Code for Bayesian Posterior and Test Statistics

```scala
// Import necessary libraries
import breeze.stats.distributions._
import breeze.linalg._
import breeze.plot._
import scala.math._

// Define a uniform prior
def prior(theta: Double): Double = {
  if (theta >= 0 && theta <= 1) 1.0 else 0.0
}

// Define the likelihood function (Bernoulli trials)
def likelihood(theta: Double, data: Seq[Int]): Double = {
  data.map(x => pow(theta, x) * pow(1 - theta, 1 - x)).product
}

// Define the posterior function using Bayes' theorem
def posterior(theta: Double, data: Seq[Int]): Double = {
  likelihood(theta, data) * prior(theta)
}

// Normalize the posterior distribution
def normalizedPosterior(data: Seq[Int]): (DenseVector[Double], DenseVector[Double]) = {
  val thetaRange = linspace(0.0, 1.0, 100)
  val posteriorValues = DenseVector(thetaRange.map(posterior(_, data)).toArray)
  val normalizationConstant = sum(posteriorValues) * (thetaRange(1) - thetaRange(0))
  (thetaRange, posteriorValues / normalizationConstant)
}

// Plot the posterior distribution
def plotPosterior(data: Seq[Int]): Unit = {
  val (thetaRange, normPosterior) = normalizedPosterior(data)
  val f = Figure()
  val p = f.subplot(0)
  p += plot(thetaRange, normPosterior)
  p.title = "Posterior Distribution"
  p.xlabel = "Theta"
  p.ylabel = "Density"
  f.saveas("posterior_plot.png")
}

// Simulate data (e.g., Bernoulli trials with true parameter 0.7)
val data = Seq.fill(20)(if (Gaussian(0.7, 0.15).draw() > 0.5) 1 else 0)

// Plot the posterior for the given data
plotPosterior(data)

// Compute posterior mean, variance, and credible interval
def posteriorSummary(data: Seq[Int]): (Double, Double, (Double, Double)) = {
  val (thetaRange, normPosterior) = normalizedPosterior(data)
  val meanPosterior = sum(thetaRange *:* normPosterior) * (thetaRange(1) - thetaRange(0))
  val variancePosterior = sum(pow(thetaRange - meanPosterior, 2) *:* normPosterior) * (thetaRange(1) - thetaRange(0))
  val credibleInterval = (thetaRange(2), thetaRange(97))
  (meanPosterior, variancePosterior, credibleInterval)
}

// Compute and print posterior summary statistics
val (mean, variance, credibleInterval) = posteriorSummary(data)
println(s"Posterior Mean: $mean")
println(s"Posterior Variance: $variance")
println(s"95% Credible Interval: ${credibleInterval._1} - ${credibleInterval._2}")

// Frequentist Test Statistics Example: Likelihood Ratio Test
def likelihoodRatioTest(data: Seq[Int], thetaNull: Double, thetaAlt: Double): (Double, Double) = {
  val logLikelihoodNull = data.map(x => x * log(thetaNull) + (1 - x) * log(1 - thetaNull)).sum
  val logLikelihoodAlt = data.map(x => x * log(thetaAlt) + (1 - x) * log(1 - thetaAlt)).sum
  val testStat = 2 * (logLikelihoodAlt - logLikelihoodNull)
  val pValue = 1 - breeze.stats.distributions.ChiSquared(1).cdf(testStat)
  (testStat, pValue)
}

// Perform a likelihood ratio test for two values of theta
val (lrStat, pValue) = likelihoodRatioTest(data, thetaNull = 0.5, thetaAlt = 0.7)
println(s"Likelihood Ratio Test Statistic: $lrStat")
println(s"p-value: $pValue")
```

### Go Code for Bayesian Posterior and Test Statistics

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Define the prior function (uniform prior)
func prior(theta float64) float64 {
	if theta >= 0 && theta <= 1 {
		return 1
	}
	return 0
}

// Define the likelihood function (Bernoulli trials)
func likelihood(theta float64, data []int) float64 {
	likelihood := 1.0
	for _, x := range data {
		likelihood *= math.Pow(theta, float64(x)) * math.Pow(1-theta, float64(1-x))
	}
	return likelihood
}

// Define the posterior function using Bayes' theorem
func posterior(theta float64, data []int) float64 {
	return likelihood(theta, data) * prior(theta)
}

// Normalize the posterior distribution
func normalizedPosterior(data []int) ([]float64, []float64) {
	thetas := make([]float64, 100)
	posteriors := make([]float64, 100)
	sumPosterior := 0.0

	for i := 0; i < 100; i++ {
		theta := float64(i) / 100
		thetas[i] = theta
		post := posterior(theta, data)
		posteriors[i] = post
		sumPosterior += post
	}

	for i := range posteriors {
		posteriors[i] /= sumPosterior
	}

	return thetas, posteriors
}

// Plot the posterior distribution
func plotPosterior(data []int) {
	thetas, posteriors := normalizedPosterior(data)

	p, _ := plot.New()
	p.Title.Text = "Posterior Distribution"
	p.X.Label.Text = "Theta"
	p.Y.Label.Text = "Density"

	pts := make(plotter.XYs, len(thetas))
	for i := range thetas {
		pts[i].X = thetas[i]
		pts[i].Y = posteriors[i]
	}

	line, _ := plotter.NewLine(pts)
	p.Add(line)
	p.Save(4*vg.Inch, 4*vg.Inch, "posterior.png")
}

// Simulate data (e.g., Bernoulli trials with true parameter 0.7)
func simulateData(size int, prob float64) []int {
	data := make([]int, size)
	for i := range data {
		if rand.Float64() < prob {
			data[i] = 1
		} else {
			data[i] = 0
		}
	}
	return data
}

// Compute posterior mean, variance, and credible interval
func posteriorSummary(data []int) (float64, float64, [2]float64) {
	thetas, posteriors := normalizedPosterior(data)

	meanPosterior := 0.0
	for i := range thetas {
		meanPosterior += thetas[i] * posteriors[i]
	}

	variancePosterior := 0.0
	for i := range thetas {
		variancePosterior += math.Pow(thetas[i]-meanPosterior, 2) * posteriors[i]
	}

	credibleInterval := [2]float64{thetas[2], thetas[97]}
	return meanPosterior, variancePosterior, credibleInterval
}

// Likelihood ratio test
func likelihoodRatioTest(data []int, thetaNull, thetaAlt float64) (float64, float64) {
	llNull := 0.0
	llAlt := 0.0

	for _, x := range data {
		llNull += float64(x)*math.Log(thetaNull) + float64(1-x)*math.Log(1-thetaNull)
		llAlt += float64(x)*math.Log(thetaAlt) + float64(1-x)*math.Log(1-thetaAlt)
	}

	testStat := 2 * (llAlt - llNull)
	pValue := 1 - distuv.ChiSquared{K: 1}.CDF(testStat)
	return testStat, pValue
}

func main() {
	// Simulate data
	data := simulateData(20, 0.7)

	// Plot posterior distribution
	plotPosterior(data)

	// Compute and print posterior summary statistics
	mean, variance, credibleInterval := posteriorSummary(data)
	fmt.Printf("Posterior Mean: %.4f\n", mean)
	fmt.Printf("Posterior Variance: %.4f\n", variance)
	fmt.Printf("95%% Credible Interval: [%.4f, %.4f]\n", credibleInterval[0], credibleInterval[1])

	// Perform likelihood ratio test
	lrStat, pValue := likelihoodRatioTest(data, 0.5, 0.7)
	fmt.Printf("Likelihood Ratio Test Statistic: %.4f\n", lrStat)
	fmt.Printf("p-value: %.4f\n", pValue)
}
```
