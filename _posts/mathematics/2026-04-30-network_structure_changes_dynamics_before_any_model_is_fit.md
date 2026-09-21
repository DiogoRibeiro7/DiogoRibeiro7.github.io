---
permalink: '/mathematics/network_structure_changes_dynamics_before_any_model_is_fit/'
title: 'Network Structure Changes Dynamics Before Any Model Is Fit'
date: '2026-04-30'
categories:
- Mathematics
tags:
- Network Science
- Graph Theory
- Dynamical Systems
- Complex Systems
- Temporal Networks
author_profile: false
classes: wide
seo_title: 'Network Structure Changes Dynamics Before Any Model Is Fit'
seo_description: 'Two networks can have the same number of nodes, edges and average degree while producing different spreading thresholds, robustness, centrality rankings and temporal reachability.'
seo_type: article
excerpt: >-
  Network topology is part of the data-generating process. The same node-level
  population can spread, fragment, synchronize and transmit information
  differently after only the pattern of edges is changed.
summary: >-
  This article develops network science as the study of how relational structure
  changes dynamics. Two six-node graphs with the same number of edges and the
  same average degree have spectral radii 2 and about 2.514, producing different
  linearized SIS spreading thresholds. The same pair behaves very differently
  under targeted node removal. The article then develops centrality as a
  question-dependent concept, Laplacian diffusion, percolation, communities,
  temporal reachability, directed and weighted networks, and the limits of
  reducing graphs to independent node features.
keywords:
- network science
- spectral radius
- graph dynamics
- epidemic threshold
- percolation
- temporal networks
why_this_exists: >-
  Graphs are often treated as visual summaries or as sources of node features
  for a downstream model. In many systems the graph itself determines which
  interactions are possible, how quickly perturbations spread, whether a giant
  component survives, and which paths exist in time.
evidence: >-
  Exact spectral calculations for two graphs with matched size and edge count,
  linearized SIS dynamics, exact component-size calculations under node removal,
  graph-Laplacian diffusion, centrality counterexamples, percolation theory and
  temporal-network reachability.
methodology: >-
  Hold the node set and edge count fixed while changing topology. Compare
  spectral radius, spreading threshold and robustness; then generalize to
  centrality, diffusion, community structure and temporal ordering.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-network-cables.jpg
  og_image: /assets/images/headers/photo-network-cables.jpg
  overlay_image: /assets/images/headers/photo-network-cables.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-network-cables.jpg
  twitter_image: /assets/images/headers/photo-network-cables.jpg
---

<!--
Development contract
Question: How much can topology change a process when node-level characteristics stay fixed?
Claim: Network topology changes spectral growth rates, reachability, robustness, diffusion and intervention effects, so the graph is part of the data-generating process rather than a decorative representation.
Counterclaim: Detailed topology is not always essential. Under strong mixing, weak interaction effects or targets defined only by coarse aggregate quantities, low-dimensional network summaries can be sufficient.
Evidence object: Two six-node graphs with the same average degree but different spectral radii; exact targeted-removal component sizes; a low-degree high-betweenness bridge example; and a temporal-ordering example where an aggregated path never exists in time.
Failure case: Treating average degree as sufficient, interpreting one centrality as universal importance, inferring causal influence from community structure, or applying static-graph reachability to time-ordered interactions.
Reader payoff: Recognise when topology must enter the model and choose graph summaries according to the dynamics and scientific question.
Exclusions: A catalogue of graph neural networks, a generic introduction to nodes and edges, and repetition of the existing production-network optimization posts.
-->

A network is not merely a table of observations with an extra column saying who is connected to whom. In many systems the connections determine which events are possible. Infection moves along contacts, failures propagate through dependencies, information follows communication links, electricity obeys a physical grid, financial exposures transmit losses across counterparties, and supply disruptions move through supplier relationships. Changing the edge pattern can therefore change the process even when every node retains exactly the same individual attributes.

This is the central idea of network science. Graph theory gives the mathematical language of vertices and edges, but network science asks what the structure does to a dynamical process. Degree, clustering, paths, spectral radius, bottlenecks, community structure and temporal ordering are not interchangeable summaries. They affect different mechanisms, and two graphs that look similar under one statistic can behave very differently under another.

The blog already contains introductions to [graph theory in production systems](/mathematics/networks/) and network optimization. The problem here is different. We will hold node-level information fixed and change only the topology, then ask what happens to spreading, robustness and reachability.

## The same average degree can produce a different spreading threshold

Consider two undirected graphs on six nodes. Both contain six edges, so both have average degree

$$
\bar d
=
\frac{2m}{n}
=
\frac{12}{6}
=
2.
$$

The first graph is the cycle

$$
C_6,
$$

where every node has degree two.

The second graph has one hub connected to all five remaining nodes, plus one additional edge joining two of those leaves. Its degree sequence is

$$
(5,2,2,1,1,1).
$$

The number of nodes is the same. The number of edges is the same. The average degree is the same. The distribution of connectivity is not.

Let $A$ be the adjacency matrix. For the six-cycle,

$$
\rho(A_{C_6})
=
2,
$$

where $\rho(A)$ denotes the spectral radius, the largest eigenvalue in magnitude for this symmetric nonnegative matrix.

For the hub-dominated graph,

$$
\rho(A_H)
\approx
2.5141.
$$

That difference has direct dynamical consequences.

Consider a simple network SIS approximation. Let $p_i(t)$ denote the approximate probability that node $i$ is infected, let $\beta$ be the per-edge transmission rate and let $\gamma$ be the recovery rate. Near the disease-free state, a common linearization has the form

$$
\frac{d p}{dt}
\approx
(\beta A-\gamma I)p.
$$

The dominant exponential growth rate is controlled by

$$
\beta\rho(A)-\gamma.
$$

The disease-free state becomes unstable in this linearized mean-field picture when

$$
\frac{\beta}{\gamma}
\rho(A)
>
1.
$$

Equivalently, the approximate threshold is

$$
\left(
\frac{\beta}{\gamma}
\right)_c
=
\frac{1}{\rho(A)}.
$$

For the cycle,

$$
\left(
\frac{\beta}{\gamma}
\right)_c
=
0.5.
$$

For the hubbed graph,

$$
\left(
\frac{\beta}{\gamma}
\right)_c
\approx
0.3978.
$$

Now choose

$$
\frac{\beta}{\gamma}
=
0.45.
$$

The cycle lies below the linearized threshold,

$$
0.45(2)
=
0.90
<
1,
$$

while the hubbed graph lies above it,

$$
0.45(2.5141)
\approx
1.131
>
1.
$$

The two networks have identical node count, edge count and average degree. A model based only on those summaries would treat them as equally connected. The spectral calculation says that one topology amplifies the infection mode strongly enough for the linearized system to grow while the other does not.

For a finite stochastic SIS process, the all-susceptible state is absorbing, so the statement should not be interpreted as an exact finite-network phase transition with a permanently endemic state. Spectral thresholds arise in deterministic, quenched mean-field and asymptotic approximations, and the precise epidemic threshold depends on the process and approximation being used. The structural message is nevertheless robust: heterogeneity in where edges are placed changes the dominant growth modes even when mean degree is fixed.

This is why average degree is often inadequate. A hub contributes disproportionately to the spectral radius because it concentrates many paths through one location. Two degree sequences with the same mean can have different second moments, different spectra and different dynamical behaviour.

For random-network epidemic approximations, moments of the degree distribution appear naturally. In configuration-model calculations, a branching factor of the form

$$
\frac{
E[D(D-1)]
}{
E[D]
}
$$

controls early propagation. Holding $E[D]$ fixed while increasing degree heterogeneity can therefore change the spreading potential.

Averages do not preserve topology.

## Connectivity and robustness depend on where edges are concentrated

The same two six-node graphs also behave differently under node failure.

Remove any one node from the cycle $C_6$. The remaining graph is a path on five nodes, so the largest connected component has size

$$
5.
$$

Now remove the degree-five hub from the second graph. Five nodes remain, but only the two leaves joined by the extra edge remain connected to each other. The other three leaves become isolated. The largest connected component has size

$$
2.
$$

The two graphs began with the same number of nodes and edges. A single targeted removal leaves one almost entirely connected and fragments the other.

If the removed node is chosen uniformly at random rather than targeted, the contrast becomes weaker. The cycle still leaves a component of size five after every one-node removal. In the hubbed graph, removing the hub leaves largest-component size two, while removing any of the five leaves leaves the remaining five nodes connected. The expected largest-component size is therefore

$$
\frac{
2+5(5)
}{
6
}
=
4.5.
$$

The graph is reasonably robust to random one-node failures and extremely vulnerable to the one specific targeted failure.

This distinction appears repeatedly in empirical networks. Hub-dominated structures can be efficient for communication because many nodes are only a few steps apart. The same concentration creates single points of structural vulnerability. A homogeneous network can require more hops while avoiding such severe dependence on one node.

Percolation theory formalizes the large-network version of this question. Suppose nodes or edges are retained independently with probability $p$. As $p$ changes, the network can undergo a transition from fragmented small components to a giant connected component containing a positive fraction of all nodes.

In an infinite configuration model with degree distribution $D$, a standard giant-component criterion involves

$$
\frac{
E[D(D-1)]
}{
E[D]
}
>
1.
$$

The same degree heterogeneity that can facilitate spreading also changes robustness under random removal.

Targeted removal is different because removal probability is no longer independent of network position. Deleting high-degree or high-betweenness nodes can destroy connectivity much more efficiently than random deletion. The intervention changes the degree distribution and correlation structure adaptively, so a random-percolation threshold is not the appropriate description.

This is why resilience cannot be summarized by the number of redundant links alone. One must ask where the redundancy is located, which nodes share dependencies, whether failures are random or targeted, and whether the process can reroute after disruption.

A supply network with ten nominal suppliers is not diversified if nine of them depend on the same upstream plant. A communication network with many alternative edges is not resilient if every route crosses the same bridge. Graph structure makes common-mode vulnerability visible.

## Centrality is a question, not a universal ranking

Network analysis often compresses structure into a node score called centrality. The word can encourage a false idea that there exists one mathematically correct measure of importance.

Different centralities answer different questions.

Degree centrality counts immediate neighbours,

$$
c_D(i)
=
d_i.
$$

It is natural when the mechanism is direct contact opportunity.

Eigenvector centrality rewards connections to nodes that are themselves central,

$$
Ax
=
\lambda x.
$$

For a connected nonnegative adjacency matrix, the Perron-Frobenius eigenvector associated with the dominant eigenvalue provides a canonical positive solution under standard conditions. It is natural for recursive prestige or amplification processes.

Betweenness centrality counts how often a node lies on shortest paths between other nodes,

$$
c_B(v)
=
\sum_{
s\ne v\ne t
}
\frac{
\sigma_{st}(v)
}{
\sigma_{st}
},
$$

where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$ and $\sigma_{st}(v)$ counts those that pass through $v$.

Closeness centrality is based on shortest-path distances from one node to others. PageRank is a stationary distribution of a random walk with teleportation. Katz centrality counts walks with length-dependent attenuation. These are mathematically different objects because they model different meanings of influence.

A small graph shows why rankings can disagree. Take two four-node cliques and connect them by a two-node corridor:

$$
K_4
-
x
-
y
-
K_4.
$$

The corridor nodes $x$ and $y$ each have degree two. Most clique nodes have degree three, and the clique vertices attached to the corridor have degree four. A ranking by degree therefore places $x$ and $y$ relatively low.

Their betweenness is very different. Every shortest path from a node in the left clique to a node in the right clique must cross both corridor nodes. Removing either corridor node disconnects the two dense regions. The low-degree nodes are globally important because of where they sit.

This is not a paradox. Degree asks how many direct contacts a node has. Betweenness asks whether it mediates geodesic routes between others. They should disagree when the local and global notions of importance differ.

The same issue matters for interventions. If contagion risk is proportional to immediate contacts, high degree can be a useful target. If the goal is to disconnect communities, bridge nodes or bridge edges can matter more. If flow follows weighted capacities rather than shortest paths, ordinary betweenness can be irrelevant. If information can diffuse over all walks rather than shortest routes, eigenvector or communicability measures may better match the process.

A centrality score should therefore be justified from the dynamics. Computing five centralities and choosing whichever produces the most interesting ranking reverses the direction of modelling.

## Diffusion is controlled by the graph Laplacian

Spreading need not be contagious. Many network processes are diffusive: temperatures equilibrate, opinions average, loads redistribute, concentrations move, and consensus algorithms reduce disagreement.

For an undirected weighted graph with adjacency matrix $A$ and degree matrix

$$
D
=
\operatorname{diag}(d_1,\ldots,d_n),
$$

the combinatorial graph Laplacian is

$$
L
=
D-A.
$$

A continuous diffusion or consensus equation can be written as

$$
\frac{dx}{dt}
=
-Lx.
$$

The solution is

$$
x(t)
=
e^{-Lt}x(0).
$$

Because

$$
L\mathbf 1
=
0,
$$

the constant vector is an equilibrium. For a connected graph, the zero eigenvalue is simple and the remaining eigenvalues satisfy

$$
0
=
\lambda_1
<
\lambda_2
\le
\cdots
\le
\lambda_n.
$$

The second-smallest eigenvalue,

$$
\lambda_2,
$$

is the algebraic connectivity. It controls the slowest nontrivial diffusive mode in this linear system. Roughly,

$$
\|x(t)-\bar x\mathbf 1\|
$$

decays at a rate involving

$$
e^{-\lambda_2 t}.
$$

A graph with a narrow bridge between two dense communities can have small $\lambda_2$. Each community mixes internally quickly, but equilibration between communities is slow because little connectivity crosses the bottleneck.

This gives a second example of why density alone is insufficient. Adding many edges inside already dense communities can increase the total number of edges without substantially improving the bottleneck between them. A system can appear highly connected in aggregate and still have one very slow global mode.

The eigenvector associated with $\lambda_2$, the Fiedler vector, is also closely connected to spectral partitioning. Large sign or value changes across a bottleneck help reveal graph cuts. Spectral clustering exploits this geometry.

The adjacency spectrum and Laplacian spectrum therefore answer different dynamical questions. The dominant adjacency eigenvalue is central to multiplicative growth and walk amplification. Small Laplacian eigenvalues are central to diffusion, consensus and bottlenecks. Referring simply to "the graph spectrum" without specifying the operator can conceal the mechanism.

Normalized Laplacians become useful when degree heterogeneity is large, and directed graphs require additional care because the operators need not be symmetric. The correct matrix is part of the model.

## Communities are useful structures, not discovered ground truth

Community detection tries to identify groups with dense internal connection and relatively sparse external connection. In many empirical networks such groups are meaningful: departments in communication graphs, functional modules in biology, interest groups in social networks, or regions in transportation systems.

A common objective is modularity. For an undirected graph with $m$ edges,

$$
Q
=
\frac{
1
}{
2m
}
\sum_{ij}
\left[
A_{ij}
-
\frac{
d_i d_j
}{
2m
}
\right]
\mathbf 1\{g_i=g_j\}.
$$

The term

$$
\frac{
d_i d_j
}{
2m
}
$$

acts as a null-model expectation under a degree-preserving random-graph logic. A partition receives high modularity when it places more edges within communities than this reference would suggest.

The output should not be confused with a uniquely existing categorical truth.

Several partitions can have similar modularity. Large networks can exhibit a resolution limit in which small meaningful groups are merged because the global objective cannot resolve them at the relevant scale. Algorithms can return different partitions after small perturbations or different random initializations. Overlapping and hierarchical communities can violate the assumption that every node belongs to one flat group.

Most importantly, a community is not automatically a causal mechanism. A dense group can arise from homophily, shared environment, organizational rules, spatial proximity, latent common causes or direct influence. The graph alone rarely separates those explanations.

Community structure is still valuable. It can reveal mesoscale organization, suggest stratified interventions and identify bottlenecks between groups. Its interpretation should remain tied to how the network was generated and measured.

The same warning applies to assortativity and clustering. A high clustering coefficient says that neighbours tend to be connected. It does not by itself explain why. A positive degree correlation says high-degree nodes tend to connect to high-degree nodes. It is a structural observation, not a causal story.

Network science gains explanatory power from topology, but topology does not eliminate the need for identification.

## Time ordering can destroy a path that exists in the aggregated graph

Static graphs assume that an edge is available whenever the process needs it. Many interaction networks violate this assumption.

Consider three nodes,

$$
A,\ B,\ C.
$$

Suppose the observed contacts are

$$
A\to B
$$

and

$$
B\to C.
$$

The static aggregated graph contains a directed path

$$
A\to B\to C.
$$

Now attach times.

In network 1,

$$
A\to B
\quad
\text{at }t=1,
$$

followed by

$$
B\to C
\quad
\text{at }t=2.
$$

Information starting at $A$ before time one can reach $B$ at time one and then $C$ at time two. A time-respecting path exists.

In network 2, reverse only the temporal order:

$$
B\to C
\quad
\text{at }t=1,
$$

followed by

$$
A\to B
\quad
\text{at }t=2.
$$

The aggregated graph is identical. The static adjacency matrix is identical. The in-degrees and out-degrees are identical. The same two edges exist.

A signal beginning at $A$ cannot reach $C$. By the time $A$ reaches $B$, the $B\to C$ contact has already happened.

Aggregation has created a path that never existed in time.

This elementary example generalizes to temporal networks with thousands or millions of interactions. Reachability depends on sequences

$$
(v_0,v_1,t_1),
(v_1,v_2,t_2),
\ldots,
(v_{k-1},v_k,t_k)
$$

with nondecreasing event times,

$$
t_1
\le
t_2
\le
\cdots
\le
t_k.
$$

Static shortest paths can therefore be physically impossible when edge availability is time ordered.

Temporal ordering also changes centrality. A node that lies on many static shortest paths may rarely mediate time-respecting paths. Burstiness changes transmission opportunities. Periodic contacts can synchronize or inhibit processes. Memory in edge activation can change epidemic and information-spreading dynamics even when the time-aggregated graph is unchanged.

Window choice becomes a modelling decision. Aggregate over one minute and the graph may be fragmented. Aggregate over a year and every node may appear connected to every relevant region, erasing the timing that constrained actual propagation.

There is no universally correct temporal resolution. It should be chosen relative to the timescale of the process being modelled.

## Direction and weight are part of topology, not decorations

Many real graphs are directed. A citation goes from one paper to another. A payment has a sender and receiver. A supplier relationship has a flow direction. Following someone on a platform need not be reciprocal.

Replacing a directed graph by an undirected graph can invent interactions that are impossible in the original system. A node can have high in-degree and low out-degree, or the reverse. Strongly connected components and reachability depend on direction.

Weights matter similarly. Two hospitals can exchange ten patients per year or ten thousand. Two banks can have exposure of one thousand euros or one billion. Treating both edges as one unweighted link discards the scale through which the dynamical process operates.

For a weighted network, strength is

$$
s_i
=
\sum_j
w_{ij},
$$

but even strength may be insufficient when weights have different meanings. Flow capacity, distance, probability, correlation and interaction frequency require different algebra. A large weight can represent a short path in one application and a high cost in another.

Signed networks introduce positive and negative relations. Multilayer networks contain several edge types simultaneously. A person can be connected to the same colleague through communication, management, friendship and shared projects. Collapsing layers can create a graph that no actual process uses.

The graph representation is therefore already a model. Nodes define what counts as an entity, edges define what counts as interaction, and aggregation rules define which distinctions are retained.

A beautiful network visualization can conceal those modelling choices because once the graph exists on screen, the representation looks factual. It is not. It is a structured abstraction of the underlying system.

## When topology can be simplified

The argument that topology matters should not become the opposite dogma that every individual edge must always be modelled.

If interaction is weak and nearly homogeneous, a mean-field approximation can be accurate enough. In a rapidly rewiring network, the process can experience an effectively well-mixed population even though instantaneous graphs are sparse. If the target is a coarse aggregate with low sensitivity to local structure, degree distribution or community-level summaries may capture what matters.

Large networks can also make full topology computationally expensive or statistically unstable. A graph observed with substantial edge error can make fine structural quantities less trustworthy than robust aggregate summaries.

The right question is sensitivity: does the scientific conclusion change materially when plausible topology changes?

If epidemic risk is nearly identical across graphs preserving the degree distribution, detailed wiring may not matter for that target. If removing one uncertain edge changes reachability between critical subsystems, topology matters enormously.

Graph randomization provides one way to test this. Compare the observed network with null networks preserving selected features such as node count, degree sequence or community sizes. If the dynamical outcome remains unusual after controlling for those properties, finer topology is contributing information.

This approach is more informative than merely computing a long list of network statistics. It asks which structural features are necessary to reproduce the behaviour of interest.

## The graph should enter before the predictive model when the process lives on edges

Network data are sometimes converted immediately into node-level features: degree, centrality, clustering coefficient, community label, neighbour averages. A conventional regression or machine-learning model is then fitted to those rows.

That can be useful when the target is genuinely node-level and network effects are summarized adequately by those features. It can also destroy the dependence structure that produced the observations.

Suppose failure at node $i$ raises failure risk at adjacent nodes. Then the outcome variables are not conditionally independent rows merely because the feature matrix is rectangular. Suppose adoption spreads through exposure. The timing and identity of infected neighbours matter, not only each node's degree. Suppose a supply shock propagates along directed dependencies. The path structure determines who can be affected.

The network is then part of the stochastic model.

This does not imply that graph neural networks are required. Classical network epidemiology, diffusion equations, spatial autoregressive models, point processes on networks, interacting particle systems and stochastic block models all represent network dependence explicitly without deep learning.

The mathematical choice should follow the mechanism.

The six-node example at the beginning is deliberately small because it removes every distraction. Two graphs had the same number of nodes, the same number of edges and the same average degree. Their spectral radii were

$$
2
$$

and approximately

$$
2.514.
$$

Under the same linearized SIS dynamics, their approximate spreading thresholds differed from

$$
0.5
$$

to about

$$
0.398.
$$

Under targeted removal, one retained a connected component of five nodes and the other fragmented to a largest component of two. Nothing about the nodes changed. Only the placement of edges changed.

The temporal example went further. Even the complete static edge set was held fixed. Only the ordering of two interactions changed, and one time-respecting path disappeared.

That is the central lesson of network science. Relationships are not metadata around the observations. When the process acts through those relationships, topology is part of the mechanism.

A model fitted after the graph has been reduced to independent rows may therefore be solving a different problem from the one that generated the data.

## References

Barabási, A.-L. (2016). *Network Science*. Cambridge University Press.

Bollobás, B. (2001). *Random Graphs* (2nd ed.). Cambridge University Press.

Holme, P., & Saramäki, J. (2012). Temporal networks. *Physics Reports*, 519(3), 97–125. https://doi.org/10.1016/j.physrep.2012.03.001

Kivelä, M., Arenas, A., Barthelemy, M., Gleeson, J. P., Moreno, Y., & Porter, M. A. (2014). Multilayer networks. *Journal of Complex Networks*, 2(3), 203–271. https://doi.org/10.1093/comnet/cnu016

Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89, 208701. https://doi.org/10.1103/PhysRevLett.89.208701

Newman, M. E. J. (2003). The structure and function of complex networks. *SIAM Review*, 45(2), 167–256. https://doi.org/10.1137/S003614450342480

Newman, M. E. J. (2006). Modularity and community structure in networks. *Proceedings of the National Academy of Sciences*, 103(23), 8577–8582. https://doi.org/10.1073/pnas.0601602103

Newman, M. E. J. (2018). *Networks* (2nd ed.). Oxford University Press.

Pastor-Satorras, R., Castellano, C., Van Mieghem, P., & Vespignani, A. (2015). Epidemic processes in complex networks. *Reviews of Modern Physics*, 87(3), 925–979. https://doi.org/10.1103/RevModPhys.87.925

Van Mieghem, P. (2011). The N-intertwined SIS epidemic network model. *Computing*, 93, 147–169. https://doi.org/10.1007/s00607-011-0155-y

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440–442. https://doi.org/10.1038/30918
