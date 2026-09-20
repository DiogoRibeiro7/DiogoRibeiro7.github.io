---
permalink: '/machine-learning/the_graph_is_the_model_in_label_propagation/'
title: 'In Label Propagation, the Graph Is the Model'
categories:
- Machine Learning
- Statistics
tags:
- Semi-Supervised Learning
- Label Propagation
- Graph Learning
- Graph Laplacian
- Manifold Learning
- Unlabelled Data
author_profile: false
seo_title: 'Why the Graph Determines Label Propagation'
seo_description: 'Graph-based semi-supervised learning does not discover labels from unlabelled data alone. Neighbourhood construction, edge weights and graph connectivity determine how labels propagate.'
excerpt: >-
  Label propagation can look almost assumption-free: connect nearby observations,
  fix the known labels and diffuse them through the graph. But the graph already
  defines which observations are allowed to influence each other. Change the
  neighbourhood rule or one important bridge and the harmonic solution can change
  even though the labelled data do not.
summary: >-
  A mathematical treatment of graph-based semi-supervised learning. The article
  derives the harmonic solution from the graph Laplacian, shows its random-walk
  interpretation, and uses small weighted graphs to demonstrate how neighbourhood
  choices and bridge edges control propagated labels. It then develops a validation
  protocol based on graph sensitivity, connectivity, conductance and trusted-label
  holdouts.
keywords:
- graph semi-supervised learning
- label propagation
- graph Laplacian
- harmonic functions
- neighbourhood graph
- semi-supervised learning
- label spreading
classes: wide
date: '2025-08-17'
why_this_exists: >-
  Graph-based semi-supervised learning is often described as if labels simply flow
  through the natural geometry of the data. The geometry is not given. It is built
  by the analyst through a representation, metric, neighbour rule and edge-weight
  function. Those choices are the model.
evidence: >-
  Harmonic-function label propagation, graph-Laplacian regularisation, the
  random-walk interpretation of propagated scores, and exact calculations on small
  weighted graphs where changing one edge changes the inferred labels.
methodology: >-
  Derive the closed-form harmonic solution for unlabelled vertices, interpret it as
  a hitting probability, then perturb graph construction while keeping the labels
  fixed. Separate graph fit, propagation stability and predictive validation on
  held-out trusted labels.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/constellation.jpg
  og_image: /assets/images/headers/constellation.jpg
  overlay_image: /assets/images/headers/constellation.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/constellation.jpg
  twitter_image: /assets/images/headers/constellation.jpg
---

Graph-based semi-supervised learning can look almost assumption-free.

Take a small set of labelled observations and a much larger set of unlabelled ones. Connect similar observations. Fix the known labels. Let information propagate through the network.

The method appears to use the geometry already present in the data.

But the graph is not already present.

We construct it.

We choose:

- the representation,
- the distance,
- the number of neighbours,
- whether the graph is directed or symmetrized,
- how edge weights decay,
- whether weak edges are removed,
- and sometimes whether disconnected components are allowed.

Those choices determine which observations are permitted to influence each other.

So the central object in graph-based semi-supervised learning is not only the labelled sample.

It is the graph.

## The Graph Construction Comes First

Let the observations be

$$
x_1,\ldots,x_n.
$$

A graph is defined by a weight matrix

$$
W=(w_{ij}),
$$

where

$$
w_{ij}\geq0
$$

measures the similarity between observations $i$ and $j$.

A common Gaussian-kernel choice is

$$
w_{ij}
=
\exp
\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right),
$$

possibly after restricting edges to a $k$-nearest-neighbour graph.

The degree of vertex $i$ is

$$
d_i
=
\sum_j w_{ij},
$$

and the degree matrix is

$$
D
=
\operatorname{diag}(d_1,\ldots,d_n).
$$

The unnormalized graph Laplacian is

$$
L
=
D-W.
$$

Once $W$ has been constructed, much of the semi-supervised method is already determined.

The graph says which local differences matter.

## Smoothness on the Graph

Suppose each vertex receives a score

$$
f_i.
$$

For binary classification, one may interpret values near zero as class zero and values near one as class one.

A standard graph smoothness penalty is

$$
\mathcal E(f)
=
\frac12
\sum_{i,j}
w_{ij}
(f_i-f_j)^2.
$$

This can be written as

$$
\mathcal E(f)
=
f^T L f.
$$

Large edge weights penalize large differences in scores.

So if

$$
w_{ij}
$$

is large, the method is encouraged to make

$$
f_i
\approx
f_j.
$$

That is the entire smoothness assumption in one equation.

The graph tells the model which predictions should be similar.

## Label Propagation as an Energy Minimization Problem

Partition the vertices into labelled and unlabelled sets,

$$
V=L\cup U.
$$

Let the labelled scores be fixed,

$$
f_L=y_L.
$$

The unlabelled scores are chosen to minimize graph energy:

$$
\min_{f_U}
f^T L f
$$

subject to

$$
f_L=y_L.
$$

Reorder the vertices so that labelled nodes come first.

Then the Laplacian can be written in block form,

$$
L
=
\begin{pmatrix}
L_{LL} & L_{LU}\\
L_{UL} & L_{UU}
\end{pmatrix}.
$$

The first-order condition for the unlabelled scores is

$$
L_{UU}f_U
+
L_{UL}y_L
=
0.
$$

Therefore,

$$
\boxed{
f_U
=
-
L_{UU}^{-1}
L_{UL}y_L
}
$$

whenever the relevant inverse exists.

This formula is important.

The propagated labels are a deterministic function of the graph Laplacian and the observed labels.

Change the graph.

Change the answer.

## The Harmonic Interpretation

For an unlabelled vertex $i$, the optimality condition can be written as

$$
d_i f_i
=
\sum_j
w_{ij}f_j.
$$

Hence

$$
\boxed{
f_i
=
\frac{
\sum_j w_{ij}f_j
}{
\sum_j w_{ij}
}.
}
$$

Every unlabelled score is a weighted average of neighbouring scores.

This is a discrete harmonic function.

There is no mysterious creation of label information.

Each unknown value is forced to be locally consistent with the graph.

That local consistency is useful only when the graph connects observations in a label-relevant way.

## A Two-Edge Example

Take one unlabelled observation $u$ connected to two labelled observations.

The first has label zero and edge weight

$$
a.
$$

The second has label one and edge weight

$$
b.
$$

The harmonic equation is

$$
(a+b)f_u
=
a\cdot0+b\cdot1.
$$

So

$$
\boxed{
f_u
=
\frac{b}{a+b}.
}
$$

If

$$
b>a,
$$

then

$$
f_u>\frac12
$$

and the point is classified as class one.

If

$$
a>b,
$$

it is classified as class zero.

Nothing else changed.

The label is decided entirely by relative graph weight.

This is trivial mathematically, but it captures the whole method.

## Similarity Is Already a Supervision Assumption

Why should the edge weight

$$
b
$$

be larger than

$$
a
$$

in the first place?

Because the representation and metric made the class-one labelled point look more similar to the unlabelled point.

That is where the assumption enters.

If Euclidean distance is used,

$$
d(x_i,x_j)
=
\|x_i-x_j\|_2,
$$

then the method assumes that Euclidean proximity is relevant to labels.

If cosine distance is used, the assumption changes.

If an embedding is learned first, the labels propagate through the geometry of that embedding.

If features are standardized, the edge weights change.

Graph-based learning does not avoid representation choice.

It amplifies it.

## A Path Graph Gives a Closed-Form Solution

Consider a path with two labelled endpoints.

The left endpoint has label zero.

The right endpoint has label one.

Between them are

$$
m
$$

unlabelled vertices with unit-weight edges.

Write the scores as

$$
f_0=0,
$$

$$
f_{m+1}=1.
$$

For every interior vertex,

$$
f_i
=
\frac{
f_{i-1}+f_{i+1}
}{2}.
$$

This means the second difference is zero:

$$
f_{i+1}
-
2f_i
+
f_{i-1}
=
0.
$$

The unique solution is linear,

$$
\boxed{
f_i
=
\frac{i}{m+1}.
}
$$

Label propagation has interpolated smoothly between the endpoints.

That is exactly what the graph says should happen.

The graph contains no evidence for a sharp class boundary.

So the solution does not create one.

## A Smooth Graph Produces a Smooth Label Function

Suppose the true labels actually change abruptly at one position along the path.

If the graph has uniform edges across that boundary, the harmonic solution still varies smoothly.

The method is behaving correctly relative to its model.

The model is wrong relative to the target.

This distinction matters.

Graph methods can fail even when the optimization is exact.

The source of error can be graph construction rather than numerical fitting.

## One Bridge Can Change the Meaning of the Graph

Now imagine two dense regions that should correspond to different classes.

Within each region, edge weights are strong.

Between them, there should be almost no connection.

If a preprocessing artifact creates one strong cross-region edge, the graph is no longer two nearly separate components.

The bridge permits label influence to cross the boundary.

Its effect depends on the relative conductance of the graph.

The important point is not that one edge always destroys the classifier.

It is that a single bridge can carry much more inferential importance than one extra observation would in an ordinary supervised model.

Graph topology matters.

## Random Walks Give the Same Interpretation

The harmonic solution has a useful probabilistic interpretation.

Normalize the edge weights into transition probabilities:

$$
P_{ij}
=
\frac{w_{ij}}{d_i}.
$$

Imagine a random walk starting at an unlabelled vertex.

Labelled vertices are absorbing states.

For binary labels zero and one,

$$
f_i
$$

can be interpreted as the probability that the random walk reaches a class-one labelled vertex before a class-zero labelled vertex.

Symbolically,

$$
\boxed{
f_i
=
P_i
\left(
T_1<T_0
\right),
}
$$

where

$$
T_1
$$

and

$$
T_0
$$

are hitting times of the labelled classes.

This interpretation makes graph dependence impossible to ignore.

Change an edge.

You change the random walk.

Change the random walk.

You change the hitting probability.

## A Bridge Is a Probability Channel

Under the random-walk interpretation, a cross-class bridge is not merely a visual nuisance.

It is a new route through which probability mass can reach the opposite labelled set.

A high-weight edge increases the probability of taking that route.

A chain of weak edges can also matter if there are many of them.

So graph-based semi-supervised learning is sensitive not only to pairwise distances but to the global connectivity induced by those distances.

That is why inspecting only nearest-neighbour examples is not enough.

The topology of the whole graph matters.

## Disconnected Components Create a Different Problem

Suppose an unlabelled connected component contains no labelled vertices.

Then no label information enters that component.

The harmonic equations do not identify a class from the observed labels alone.

Any implementation has to resolve this somehow.

It may return a default.

It may use class priors.

It may regularize toward a global value.

It may leave the component unresolved.

But the unlabelled geometry itself has not supplied the missing class identity.

This is another form of the identifiability problem.

A graph component can be perfectly well defined and still have no supervision.

## More Labels Do Not Automatically Repair a Bad Graph

Suppose the graph contains many cross-class edges because the chosen representation mixes the classes.

Adding more labelled points helps.

But it does not change the fact that the graph smoothness penalty prefers similar values across those edges.

The classifier is solving a compromise between observed labels and imposed smoothness.

If the graph says

$$
f_i\approx f_j
$$

for many pairs with genuinely different labels, the model is structurally misspecified.

More supervision may reduce the damage.

It does not make the graph assumption correct.

## k-Nearest-Neighbour Graphs Introduce a Scale Parameter

A common graph construction connects each point to its

$$
k
$$

nearest neighbours.

Small

$$
k
$$

produces a sparse and highly local graph.

Large

$$
k
$$

produces a more connected graph.

These are not merely computational choices.

They define the scale at which smoothness is imposed.

If

$$
k
$$

is too small, the graph may fragment.

If

$$
k
$$

is too large, edges may cross class boundaries.

So the parameter controls a bias trade-off:

$$
\text{fragmentation}
\quad\text{versus}\quad
\text{over-smoothing}.
$$

There is no universally correct neighbour count.

## Mutual k-Nearest Neighbours Change the Topology Again

One can symmetrize a directed nearest-neighbour graph in different ways.

Under a union rule, connect $i$ and $j$ if either regards the other as a neighbour.

Under a mutual rule, connect them only if both do.

The union graph is denser.

The mutual graph is more conservative.

That difference can remove weak bridges.

It can also disconnect legitimate low-density transitions.

The semi-supervised predictions can therefore change even when the underlying pairwise distances do not.

Graph topology is a modelling decision.

## Kernel Bandwidth Is Another Structural Parameter

Suppose edge weights are

$$
w_{ij}
=
\exp
\left(
-\frac{
d(x_i,x_j)^2
}{
2\sigma^2
}
\right).
$$

For small

$$
\sigma,
$$

only very close neighbours receive substantial weight.

For large

$$
\sigma,
$$

more distant observations influence one another.

The effective smoothness scale changes.

If

$$
\sigma
$$

becomes very large, many edge weights become similar and local structure is washed out.

If it becomes very small, the graph can behave almost like disconnected local islands.

The graph is therefore conditional on

$$
\sigma.
$$

So is the classifier.

## Local Scaling Can Help Unequal Density

A single global bandwidth can behave poorly when one part of the data is dense and another sparse.

An alternative is local scaling.

For example,

$$
w_{ij}
=
\exp
\left(
-\frac{
d(x_i,x_j)^2
}{
\sigma_i\sigma_j
}
\right),
$$

where

$$
\sigma_i
$$

is determined by a local-neighbour distance around point $i$.

This can make the graph more adaptive.

It also changes the meaning of similarity.

A raw distance of one unit may correspond to a strong edge in a sparse region and a weak edge in a dense region.

That may be desirable.

It is still a modelling assumption.

## Normalized and Unnormalized Laplacians Differ

The unnormalized Laplacian is

$$
L=D-W.
$$

Two common normalized versions are

$$
L_{\text{sym}}
=
I-D^{-1/2}WD^{-1/2}
$$

and

$$
L_{\text{rw}}
=
I-D^{-1}W.
$$

These forms weight degree differently.

In graphs with highly variable density, the distinction can matter substantially.

Using a normalized Laplacian changes the geometry of smoothness.

The choice should not be buried inside a library default.

## The Label Spreading Variant Changes Boundary Conditions

Some methods clamp the labelled vertices.

Their observed labels remain fixed during propagation.

Other methods allow labelled vertices to move slightly under graph regularization.

This is often called label spreading rather than strict label propagation.

That distinction changes the optimization problem.

With hard clamping,

$$
f_L=y_L.
$$

With soft clamping, one might minimize an objective like

$$
\sum_{i\in L}
(f_i-y_i)^2
+
\lambda
f^T L f.
$$

Now labelled values themselves can be smoothed.

This can help when labels are noisy.

It can hurt when labels are trusted but the graph is wrong.

Again, the implementation encodes a statistical position.

## Class Priors Can Be Distorted

Suppose one labelled class is represented by many more seed points than another.

Random walks have more absorbing targets of the majority class.

Depending on graph geometry and algorithm details, that can influence the propagated class balance.

The resulting pseudo-label distribution can drift away from the true class prior.

This is not always corrected by thresholding.

Graph methods can therefore inherit both geometric bias and seed-label imbalance.

## The Graph Can Encode Batch Effects Perfectly

Suppose measurements come from two laboratories.

The true target class is unrelated to laboratory.

But instrument differences dominate the representation.

A nearest-neighbour graph may then connect observations primarily within laboratory.

Label propagation will be smooth along the batch structure because the graph tells it to be.

If the labelled seeds are imbalanced across laboratories, the method can learn the batch rather than the target.

The graph can be internally coherent and scientifically wrong.

This is the graph analogue of a visually beautiful t-SNE batch separation.

## Edge Construction Can Leak Information

Graph-based methods are often transductive.

They use both labelled and unlabelled feature vectors when building the graph.

That is legitimate when the intended task is to infer labels for exactly those unlabelled observations.

But evaluation must respect the transductive setting.

If a future test set is included in graph construction before evaluation, the method has seen test features.

That may or may not be allowed by the deployment problem.

The evaluation protocol must match the intended information boundary.

Otherwise, performance can look stronger than what would be available in a genuinely inductive future-data setting.

## Transductive and Inductive Claims Are Different

A transductive method answers:

> Given these labelled and unlabelled observations together, what labels should be assigned to the current unlabelled set?

An inductive model answers:

> What function should predict labels for new observations not present during training?

Graph label propagation naturally solves the first problem.

Using it for the second requires an extension.

For example, a new point may need to be inserted into the graph and the propagation recomputed.

The distinction should be stated explicitly.

## Holding Out Trusted Labels Is Essential

A useful validation design hides some known labels.

Let the trusted labelled set be split into:

$$
L_{\text{seed}}
$$

and

$$
L_{\text{validation}}.
$$

Build the graph using features as the application allows.

Propagate labels using only

$$
L_{\text{seed}}.
$$

Then evaluate predictions on

$$
L_{\text{validation}}.
$$

This directly tests whether the graph geometry is useful for label inference.

Without such a check, the graph may look reasonable without ever being tested against withheld supervision.

## Leave-One-Label-Out Diagnostics Are Particularly Informative

When labelled data are scarce, one can perform a label-removal diagnostic.

For each labelled observation:

1. temporarily hide its label,
2. propagate from the remaining labels,
3. predict the hidden label,
4. record whether the graph recovers it.

This resembles cross-validation on the seed set.

It is not fully independent if graph construction uses all features.

But it directly probes whether known labels are locally compatible with the graph.

A high failure rate is strong evidence that the graph should not be trusted for unlabelled prediction.

## Measure Edge Purity Where Labels Exist

For edges connecting labelled vertices, compute the fraction that join identical labels.

One simple quantity is

$$
P_{\text{same}}
=
\frac{
\sum_{i,j\in L}
w_{ij}
\mathbb 1\{y_i=y_j\}
}{
\sum_{i,j\in L}
w_{ij}
}.
$$

This asks whether high-weight labelled edges tend to respect class identity.

It is not a proof that unlabelled edges behave similarly.

But it is a direct falsification check.

If high-weight labelled edges frequently cross classes, graph smoothness is already in tension with observed supervision.

## Inspect Cross-Class Edge Mass

A related diagnostic is

$$
M_{\text{cross}}
=
\frac{
\sum_{i,j\in L}
w_{ij}
\mathbb 1\{y_i\neq y_j\}
}{
\sum_{i,j\in L}
w_{ij}
}.
$$

Large cross-class edge mass means the graph strongly connects observations known to have different labels.

That does not automatically invalidate the method.

Some classes genuinely overlap.

But a method based on smooth labels should not ignore the contradiction.

The graph assumption can be measured where labels are available.

## Graph Sensitivity Should Be Part of Model Sensitivity

Suppose a final semi-supervised model performs well for

$$
k=10
$$

nearest neighbours.

What happens at

$$
k=8,
$$

$$
k=12,
$$

or

$$
k=15?
$$

What happens under a mutual-neighbour graph?

What happens if the bandwidth changes by 20 percent?

What happens after standardization or a plausible alternative feature scaling?

If the inferred labels change dramatically, that instability is part of the result.

A single graph is not evidence that graph construction is robust.

## Consensus Across Graphs Can Be More Informative Than One Graph

Suppose several graph specifications are scientifically defensible.

For each one, compute the propagated label score

$$
f_i^{(r)},
$$

where

$$
r=1,\ldots,R.
$$

Then examine the distribution across graph specifications.

For example,

$$
\bar f_i
=
\frac1R
\sum_{r=1}^{R}
f_i^{(r)}.
$$

One can also report the proportion of specifications assigning class one.

This is not a posterior probability unless a probabilistic model justifies it.

But it does reveal graph-construction sensitivity.

An observation classified identically under every plausible graph is different from one whose class depends on whether

$$
k=10
$$

or

$$
k=11.
$$

## Confidence From Propagation Is Not Automatically Calibration

The harmonic score

$$
f_i
$$

often lies between zero and one.

That does not automatically make it a calibrated probability of class membership.

Under the random-walk interpretation, it is a hitting probability on the constructed graph.

Those are different statements.

A value

$$
f_i=0.9
$$

means that, under the graph-defined random walk, class-one seeds are hit first with probability 0.9.

It does not necessarily mean

$$
P(Y_i=1\mid X_i)=0.9.
$$

Calibration requires empirical validation.

The same warning that applies to pseudo-label confidence applies here too.

## More Unlabelled Data Can Change the Graph

A common intuition says that unlabelled observations merely fill in the geometry more accurately.

Often they do.

But adding observations changes the graph.

Nearest-neighbour relations can change.

New paths can appear.

Old edges can disappear under fixed-degree graph construction.

Graph distances and conductance can change.

So the effect of more unlabelled data is not necessarily monotone.

The classifier itself is changing because the discrete approximation to the geometry is changing.

## A New Observation Can Create a Shortcut

Consider two regions that are weakly connected.

Add one observation in the low-density area between them.

That point may connect to both sides.

In a nearest-neighbour graph, it can create a short path that did not previously exist.

The new observation carries no label.

Yet it changes how labels can diffuse.

This is a concrete mechanism through which an unlabelled point can alter the classification of other unlabelled points.

More data is not passive.

It modifies the geometry used by the learner.

## Graph Connectivity Has to Match Label Connectivity

The strongest setting for graph-based semi-supervised learning is when class structure aligns with graph structure.

Roughly,

$$
\text{same-class observations}
\longleftrightarrow
\text{strong graph connectivity},
$$

while class boundaries correspond to weak connectivity.

This is a graph version of the cluster and low-density assumptions.

If that alignment is poor, label propagation has no reason to work.

The method is not discovering the alignment.

It assumes it.

## Conductance Gives a Useful Structural Diagnostic

For a set of vertices

$$
S,
$$

define the cut weight

$$
\operatorname{cut}(S,\bar S)
=
\sum_{i\in S}
\sum_{j\notin S}
w_{ij}.
$$

One form of conductance is

$$
\phi(S)
=
\frac{
\operatorname{cut}(S,\bar S)
}{
\min\{
\operatorname{vol}(S),
\operatorname{vol}(\bar S)
\}
},
$$

where

$$
\operatorname{vol}(S)
=
\sum_{i\in S}d_i.
$$

Low conductance means a set is strongly connected internally relative to its connection outside.

If classes correspond to low-conductance regions, graph propagation has a structural advantage.

If the true classes have high conductance between them, smooth propagation is a poor prior.

This connects semi-supervised learning with graph partitioning directly.

## The Best Graph Need Not Be the Best Visualization Graph

A graph used for t-SNE, UMAP or spectral visualization may be tuned to produce a useful representation.

A graph used for semi-supervised inference has a different objective.

The best neighbourhood scale for visual continuity need not be the best scale for label smoothness.

Reusing one graph everywhere because it already exists can smuggle in assumptions from one task to another.

Graph construction should be justified for the inferential task at hand.

## The Representation and Graph Should Be Validated Together

Suppose the input representation is

$$
z=\phi(x).
$$

The graph is then

$$
W=W\{\phi(X)\}.
$$

So the full semi-supervised model is better written as

$$
\widehat f
=
\mathcal G
\left(
X_L,
Y_L,
X_U;
\phi,
d,
k,
w
\right),
$$

where

- $\phi$ is the representation,
- $d$ is the distance,
- $k$ is the neighbourhood scale,
- $w$ is the edge-weight rule.

This notation prevents the unlabelled sample from appearing to be the only extra ingredient.

The graph pipeline contains several modelling choices.

## A Practical Validation Contract

Before trusting graph-based semi-supervised predictions, I would require several checks.

### 1. A supervised baseline

Train a classifier only on trusted labels.

The graph method should justify its additional complexity by improving something measurable.

### 2. Held-out trusted labels

Hide some real labels and test whether propagation recovers them.

### 3. Graph sensitivity

Vary neighbour count, bandwidth, graph symmetrization and defensible representations.

### 4. Label-edge diagnostics

Measure how often strong edges connect different known classes.

### 5. Connectivity diagnostics

Inspect disconnected components, bridges and low-conductance cuts.

### 6. Class-balance diagnostics

Check whether propagation systematically expands one class.

### 7. Calibration diagnostics

If propagated scores are interpreted probabilistically, test that interpretation empirically.

### 8. Distribution-shift diagnostics

Rebuild or stress-test the graph under plausible changes in the unlabelled population.

## The Method Should Fail Gracefully When Geometry Is Weak

A useful semi-supervised pipeline should be able to decide that unlabelled geometry is not trustworthy enough to improve classification.

That can mean:

- reverting to the supervised model,
- using only high-confidence graph regions,
- abstaining on disconnected components,
- requesting labels near bridges,
- or switching to active learning.

The goal should not be to force every unlabelled observation into the propagation system.

Unlabelled data should earn influence through validated structure.

## Active Learning Can Target Graph Weaknesses

Graph diagnostics also suggest where human labels are most valuable.

Candidate points include:

- bridge vertices,
- observations near low-conductance cuts,
- nodes with conflicting labelled neighbours,
- vertices with unstable propagated scores,
- and points in unlabelled connected components.

This is a more informative strategy than labelling uniformly at random when annotation is expensive.

Semi-supervised and active learning can therefore be combined naturally.

The graph identifies where uncertainty is structural.

Human labels can resolve it.

## Graph-Based Learning Does Not Eliminate the Need for Labels

It can reduce the number of labels required when graph geometry aligns with the target.

That is a conditional statement.

Without such alignment, the graph can propagate mistakes just as efficiently as correct labels.

The relevant trade is

$$
\text{fewer labels}
\quad\text{in exchange for}\quad
\text{stronger geometric assumptions}.
$$

This is the same principle that appears throughout semi-supervised learning.

The graph makes the assumption concrete.

## Conclusion

Label propagation is elegant because the mathematics is simple.

Fix known labels.

Choose the smoothest function on the graph.

The harmonic solution is

$$
\boxed{
f_U
=
-
L_{UU}^{-1}
L_{UL}y_L.
}
$$

But the simplicity of the solution can hide the complexity of the model.

The Laplacian came from a graph.

The graph came from a representation, metric, neighbourhood rule and weighting scheme.

Those choices determine which observations can influence each other.

So the central principle is

$$
\boxed{
\text{in graph-based semi-supervised learning, the graph is the model}.
}
$$

A strong graph can let a few labels supervise a large population.

A weak graph can diffuse the wrong labels with mathematical precision.

The correct validation question is therefore not merely

> Does label propagation converge?

It will.

The useful question is

> Does the graph encode a notion of neighbourhood under which label smoothness is actually true?

That question has to be answered with trusted labels, graph diagnostics, perturbation analysis and domain knowledge.

The propagation equation cannot answer it for us.

## References

- Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. *Journal of Machine Learning Research*, 7, 2399–2434.
- Zhu, X., Ghahramani, Z., & Lafferty, J. (2003). Semi-supervised learning using Gaussian fields and harmonic functions. *Proceedings of the 20th International Conference on Machine Learning*, 912–919.
- Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). Learning with local and global consistency. *Advances in Neural Information Processing Systems*, 16.
