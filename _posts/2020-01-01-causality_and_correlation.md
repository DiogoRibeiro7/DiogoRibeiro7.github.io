---
author_profile: false
categories:
- Statistics
classes: wide
date: '2020-01-01'
excerpt: Understand how causal reasoning helps us move beyond correlation, resolving
  paradoxes and leading to more accurate insights from data analysis.
header:
  image: /assets/images/data_science_4.jpg
  og_image: /assets/images/data_science_1.jpg
  overlay_image: /assets/images/data_science_4.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_4.jpg
  twitter_image: /assets/images/data_science_1.jpg
keywords:
- Simpson's paradox
- Causality
- Berkson's paradox
- Correlation
- Data science
seo_description: Explore how causal reasoning, through paradoxes like Simpson's and
  Berkson's, can help us avoid the common pitfalls of interpreting data solely based
  on correlation.
seo_title: 'Causality Beyond Correlation: Understanding Paradoxes and Causal Graphs'
seo_type: article
summary: An in-depth exploration of the limits of correlation in data interpretation,
  highlighting Simpson's and Berkson's paradoxes and introducing causal graphs as
  a tool for uncovering true causal relationships.
tags:
- Simpson's paradox
- Berkson's paradox
- Correlation
- Data science
- Causal inference
title: 'Causality Beyond Correlation: Simpson''s and Berkson''s Paradoxes'
---

In today's data-driven world, we often rely on statistical correlations to make decisions. Whether it’s a business predicting customer behavior or a healthcare study analyzing the impact of a new drug, correlations offer a quick way to extract patterns from data. However, correlations can be misleading. The phrase **"correlation does not imply causation"** is well known, but understanding why this is true and how to move beyond correlation requires deeper insights into causality.

This article is aimed at anyone who works with data and is interested in gaining a more accurate understanding of how to interpret statistical relationships. Here, we will explore how to uncover **causal relationships** in data, how to resolve confusing situations like **Simpson's Paradox** and **Berkson's Paradox**, and how to use **causal graphs** as a tool for making better decisions. The goal is to demonstrate that by understanding causality, we can avoid the pitfalls of over-relying on correlation and make more informed decisions.

---

## Correlation and Causation: Why the Distinction Matters

In statistics, **correlation** measures the strength of a relationship between two variables. For example, if you observe that ice cream sales increase as temperatures rise, you might conclude that warmer weather causes more ice cream to be sold. This conclusion feels intuitive, but what about cases where the data is less obvious? Imagine a study finds a correlation between shark attacks and ice cream sales. Does one cause the other? Clearly not—but the correlation exists because both are influenced by a common factor: hot weather.

This example underscores the central problem: **correlation does not imply causation**. Just because two variables move together doesn’t mean one causes the other. Correlation can arise for several reasons:

- **Direct causality**: One variable causes the other.
- **Reverse causality**: The relationship runs in the opposite direction.
- **Confounding variables**: A third variable influences both.
- **Coincidence**: The relationship is due to chance.

To understand the true nature of relationships in data, we need to go beyond correlation and ask **why** the variables are related. This is where **causal inference** comes in.

---

## The Importance of Causal Inference

Causal inference is the process of determining whether and how one variable causes changes in another. Randomized control trials (RCTs) are the gold standard for establishing causality because they randomly assign treatments to subjects, thus eliminating confounders and biases. However, RCTs can be expensive, time-consuming, and sometimes unethical or impractical to conduct.

In most real-world scenarios, we rely on **observational data**, which is data collected without manipulating any variables. The challenge with observational data is that it’s difficult to establish causality because we can't control for all the confounding factors that might be influencing the relationship.

Fortunately, researchers have developed methods to uncover causal relationships from observational data by combining **statistical reasoning** with a deep understanding of the data's context. This is where **causal graphs** and tools like **Simpson's Paradox** and **Berkson's Paradox** come into play.

---

## Simpson's Paradox: The Danger of Aggregating Data

Simpson's Paradox is a statistical phenomenon in which a trend that appears in different groups of data disappears or reverses when the groups are combined. This paradox occurs because of a **lurking confounder**, a variable that influences both the independent and dependent variables, skewing the relationship between them.

### The Classic Example

Imagine you're analyzing the effectiveness of a new drug across two groups: younger patients and older patients. Within each group, the drug seems to improve health outcomes. However, when you combine the two groups, the overall analysis shows that the drug is **less** effective.

This reversal happens because age, a **confounding variable**, is driving the overall result. If more older patients received the drug and older patients have worse outcomes in general, it can skew the overall data. Thus, the combined analysis gives a misleading result, suggesting the drug is less effective when it actually benefits each group.

### Why Does This Happen?

Simpson’s Paradox occurs because the relationship between variables changes when data is aggregated. In the example above, **age** confounds the relationship between the drug and health outcomes. It’s important to note that combining data from different groups without accounting for confounders can hide the true relationships within each group.

This paradox demonstrates why it’s crucial to understand the **story behind the data**. If we simply relied on the overall correlation, we would draw the wrong conclusion about the drug’s effectiveness.

---

## Berkson's Paradox: The Pitfall of Selection Bias

Berkson's Paradox is another example of how data can mislead us, particularly when we focus on a subset of data and mistakenly infer relationships that don’t exist in the general population.

### The Example of Talent and Attractiveness

Let’s say you’re studying two traits in the general population: talent and attractiveness. You find no significant correlation between the two traits—people’s talent levels seem independent of how attractive they are.

Now, suppose you focus on a particular subset of the population: celebrities. In this subset, you observe a **negative correlation** between talent and attractiveness—celebrities who are highly attractive tend to be less talented, and vice versa.

This is **Berkson's Paradox**. The negative correlation exists in the subset (celebrities) because being a celebrity is contingent on having at least one of these traits—either being very attractive or very talented. If someone is extremely attractive, they might become a celebrity without much talent, and if they are very talented, they can achieve fame without being particularly attractive. However, this negative correlation does not exist in the general population.

### The Problem of Selection Bias

Berkson's Paradox illustrates the problem of **selection bias**—when we restrict our analysis to a specific subset of the population, we may inadvertently introduce spurious correlations that don’t exist in the full dataset. In this case, the act of selecting celebrities introduces a negative correlation between talent and attractiveness that doesn't exist in the general population.

The key takeaway from Berkson’s Paradox is that we need to be careful about **how we select data for analysis**. If we focus only on a specific group without understanding how that group was selected, we can introduce misleading correlations.

---

## Causal Graphs: A Tool for Visualizing Relationships

To avoid falling into the traps of Simpson’s and Berkson’s Paradoxes, it’s helpful to use **causal graphs** to visualize the relationships between variables. These graphs, also known as **Directed Acyclic Graphs (DAGs)**, allow us to represent the causal structure of a system and identify which variables are influencing others.

### What Are Causal Graphs?

A **causal graph** is a diagram that represents variables as **nodes** and the causal relationships between them as **directed edges** (arrows). A directed edge from variable **A** to variable **B** indicates that **A** has a causal influence on **B**.

Causal graphs are powerful because they help us:

1. **Identify confounders**: Variables that influence both the independent and dependent variables.
2. **Clarify causal relationships**: Show which variables are direct causes and which are effects.
3. **Avoid incorrect controls**: Help us decide which variables to control for in statistical analysis.

### Using Causal Graphs to Resolve Simpson's Paradox

Let’s return to the example of the drug trial. A causal graph for this scenario might look like this:

- **Age** influences both **Drug Use** and **Health Outcome**.
- **Drug Use** directly affects **Health Outcome**.

In this case, **Age** is a **confounder** because it influences both the independent variable (**Drug Use**) and the dependent variable (**Health Outcome**). When we control for **Age**, we remove its confounding effect and can properly assess the impact of the drug on health outcomes.

### Using Causal Graphs to Resolve Berkson's Paradox

In the case of celebrities, a causal graph might look like this:

- **Talent** and **Attractiveness** are independent in the general population.
- **Celebrity Status** depends on both **Talent** and **Attractiveness**.

Here, **Celebrity Status** is a **collider**, a variable that is influenced by both **Talent** and **Attractiveness**. When we condition on a collider (i.e., focus only on celebrities), we create a spurious correlation between **Talent** and **Attractiveness**. The key is to recognize that the negative correlation between these variables only exists because we have selected a specific subset of the population (celebrities), not because there is a true relationship between talent and attractiveness.

---

## The Broader Implications of Causality in Data Analysis

Simpson’s and Berkson’s Paradoxes serve as cautionary tales for anyone working with data. They highlight the importance of understanding the causal structure behind data and the potential pitfalls of relying solely on correlation.

By using tools like causal graphs and thinking critically about the relationships between variables, we can:

- **Identify and control for confounders** to avoid drawing incorrect conclusions.
- **Recognize when selection bias is influencing our results**, as in Berkson’s Paradox.
- **Understand when to intervene** in the data collection process, such as designing experiments or deciding which variables to control for in observational studies.

The field of **causal inference** offers powerful tools for moving beyond correlation and gaining deeper insights into the data. It’s not enough to ask, “**What is correlated?**” We need to ask, “**Why are these variables related?**” By doing so, we can make more informed decisions and avoid common pitfalls in data interpretation.

---

## Conclusion

The journey from correlation to causation is not always straightforward, but it is essential for anyone making data-driven decisions. **Simpson's Paradox** and **Berkson's Paradox** illustrate how easily data can mislead us when we don't account for the full causal story behind the numbers. Through tools like **causal graphs**, we can visualize relationships between variables, uncover hidden biases, and make better, more accurate conclusions.

Understanding causality requires more than just statistical analysis—it demands a deeper engagement with the data, including recognizing confounders, colliders, and the true mechanisms that generate the patterns we observe. By asking **"Why?"** instead of just **"What?"**, we can elevate our analysis, avoid common misinterpretation, and ultimately make more informed decisions based on the true nature of relationships in data.

### Further Reading

- Judea Pearl’s “The Book of Why” offers a foundational introduction to causality and causal graphs.
- “Causal Inference in Statistics: A Primer” provides a more technical deep dive into causal inference methods.
- The DoWhy library in Python is an excellent tool for implementing causal inference techniques in practice.
