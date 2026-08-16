---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2021-10-15'
excerpt: Understand how decision tree algorithms split data and how pruning improves
  generalization.
header:
  image: /assets/images/data_science_7.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_7.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_7.jpg
  twitter_image: /assets/images/data_science_7.jpg
keywords:
- Decision trees
- Classification
- Tree pruning
- Machine learning
seo_description: Learn the mechanics of decision tree algorithms, including entropy-based
  splits and pruning techniques that prevent overfitting.
seo_title: How Decision Trees Work and Why Pruning Matters
seo_type: article
summary: This article walks through the basics of decision tree construction and explains
  common pruning methods to create better models.
tags:
- Decision Trees
- Classification
- Regularization
title: Demystifying Decision Tree Algorithms
---

Decision trees are intuitive models that recursively split data into smaller groups based on feature values. Each split aims to maximize homogeneity within branches while separating different classes.

## How a Tree Is Grown

Growth is greedy and recursive. Starting with all training data at the root, the algorithm evaluates every feature and every candidate threshold, scores each resulting partition, takes the best one, and repeats on each child node. It stops when a node is pure, when it holds too few samples to split, or when a depth limit is reached.

Greedy means the algorithm never reconsiders. A split that looks best locally may foreclose a better structure two levels down, and finding the globally optimal tree is NP-hard. This is a practical limitation worth remembering: two trees fitted to slightly different samples can look completely different while performing about the same.

## Choosing the Best Split

Metrics like **Gini impurity** and **entropy** measure how mixed the classes are in each node. The algorithm searches over possible splits and selects the one that yields the largest reduction in impurity.

For a node where class $k$ has proportion $p_k$:

$$
\text{Gini} = 1 - \sum_{k} p_k^2, \qquad
\text{Entropy} = -\sum_{k} p_k \log_2 p_k .
$$

Both are zero when a node contains a single class and maximal when classes are evenly mixed. A split is scored by how much impurity it removes, weighting each child by the share of samples it receives:

$$
\Delta I = I(\text{parent}) - \sum_{j} \frac{n_j}{n} I(\text{child}_j) .
$$

In practice the two criteria almost always choose the same splits. Gini is marginally cheaper because it avoids logarithms, which is why it is the common default. Entropy comes from information theory, where $\Delta I$ is the information gain, and it penalises impurity slightly more aggressively at the extremes.

For regression trees the same scheme applies with variance in place of impurity: each split minimises the sum of squared deviations from the mean within the resulting nodes.

One bias is worth knowing. Impurity-based splitting favours features with many distinct values, because more candidate thresholds mean more chances to find a good-looking split. High-cardinality identifiers can therefore dominate a tree while carrying no real signal.

## Preventing Overfitting

A tree grown until every leaf is pure often memorizes the training data. **Pruning** removes branches that provide little predictive power, leading to a simpler tree that generalizes better to new samples.

There are two moments to intervene. Pre-pruning stops growth early using constraints such as maximum depth, a minimum number of samples required to split, a minimum number of samples per leaf, or a minimum impurity decrease. It is cheap but myopic, since a split that looks worthless can enable a valuable one beneath it.

Post-pruning grows the full tree and then removes subtrees that do not justify their complexity. Cost-complexity pruning formalises this by minimising

$$
R_\alpha(T) = R(T) + \alpha \, |\tilde{T}|,
$$

where $R(T)$ is the training error, $|\tilde{T}|$ is the number of leaves, and $\alpha$ controls the penalty. Increasing $\alpha$ produces a nested sequence of progressively smaller trees, and cross-validation selects the value that generalises best.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)

best_score, best_alpha = -np.inf, 0.0
for alpha in path.ccp_alphas:
    tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    score = cross_val_score(tree, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_score, best_alpha = score, alpha

print(f"best alpha: {best_alpha:.5f}  cv accuracy: {best_score:.3f}")
final = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(X_train, y_train)
print(f"leaves: {final.get_n_leaves()}, depth: {final.get_depth()}")
```

## The Interpretability Question

Trees are usually described as interpretable, and a shallow tree genuinely is: the path from root to leaf is a readable sequence of conditions. That property degrades quickly with depth, and a tree with hundreds of leaves is no easier to reason about than a black box.

Their instability also undercuts naive interpretation. Because splits are chosen greedily, small changes to the training sample can produce a structurally different tree with similar accuracy. A feature appearing at the root is therefore weaker evidence of importance than it appears, and built-in impurity-based feature importances inherit the same high-cardinality bias described above. Permutation importance, computed on held-out data, is the more trustworthy measure.

## When to Use Decision Trees

Decision trees handle both numeric and categorical features and require minimal data preparation. They also serve as the building blocks for powerful ensemble methods like random forests and gradient boosting.

Their practical advantages are concrete. No scaling or normalisation is required, since splits depend only on ordering. Non-linear relationships and interactions are captured without being specified in advance. Missing values can be handled through surrogate splits in some implementations. And the decision logic can be exported as rules that a non-technical audience can audit.

The limitations are equally concrete. A single tree is a high-variance estimator, and its accuracy rarely competes with ensembles. Decision boundaries are axis-aligned, so a diagonal relationship must be approximated by a staircase of many splits. Extrapolation is impossible: predictions outside the training range are flat, because every input eventually lands in a leaf whose value was fixed at training time. That last point rules trees out for genuine extrapolation problems such as trending time series.

The usual conclusion is that a single tree is best treated as an exploratory or explanatory tool rather than a production predictor. Averaging many decorrelated trees, as random forests do, keeps the flexibility while cancelling much of the variance. Fitting trees sequentially to residuals, as gradient boosting does, turns the same weak learner into one of the strongest tabular methods available. Understanding how one tree splits and prunes is what makes the behaviour of both ensembles predictable.

## References

- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.
