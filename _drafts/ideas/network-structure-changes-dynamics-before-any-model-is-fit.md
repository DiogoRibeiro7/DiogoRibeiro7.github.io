---
author_profile: false
categories:
- Mathematics
classes: wide
title: 'Network Structure Changes Dynamics Before Any Model Is Fit'
excerpt: 'Degree, clustering, communities and temporal connectivity alter how processes spread through a network. The graph is not merely a visualization layer around independent observations.'
keywords:
- network science
- graph dynamics
- percolation
- contagion
- temporal networks
seo_title: 'Network Structure Changes Dynamics Before Any Model Is Fit'
seo_description: 'A mathematical draft on centrality, contagion, percolation, communities, temporal networks, and why topology changes dynamical behaviour.'
seo_type: article
summary: 'A planned article showing how graph topology changes diffusion, epidemic thresholds, robustness and intervention effects, with explicit spectral and percolation calculations.'
tags:
- Network Science
- Graph Theory
- Dynamical Systems
- Complex Systems
why_this_exists: 'Network data are often reduced to node features before modelling, discarding the interaction structure that determines many collective behaviours.'
evidence: 'Spectral graph calculations, epidemic-threshold examples, percolation simulations and temporal-network counterexamples.'
methodology: 'Compare identical node-level populations placed on different graph topologies, then quantify how spectra and connectivity alter propagation and resilience.'
---

<!--
Development contract
Question: How much can topology change a process when node-level characteristics stay fixed?
Claim: Network structure changes dynamical thresholds, reachability, centrality and robustness, so graph topology is part of the data-generating process rather than an optional representation.
Counterclaim: Some aggregate outcomes are insensitive to detailed topology, particularly under strong mixing or when interaction effects are weak.
Evidence object: Same node set on lattice, random and hub-dominated graphs; spectral epidemic threshold; percolation curve; one temporal-ordering example.
Failure case: Treating centrality as a universal importance score, applying static-graph methods to temporal contacts, or inferring causality from community structure.
Reader payoff: Know when graph structure must enter the model and which network summaries correspond to which dynamical questions.
Exclusions: A catalogue of graph neural networks.
-->

## Mathematical spine

For adjacency matrix $A$, introduce the spectral radius

$$
\rho(A).
$$

In simple SIS-type approximations, epidemic persistence can depend on a threshold involving

$$
\frac{\beta}{\gamma}\rho(A).
$$

Use this to compare graphs with identical node count and mean degree but different spectral structure.

For percolation, examine the giant-component transition and distinguish random node removal from targeted removal of high-degree vertices.

Discuss centralities as different mathematical objects: degree, eigenvector, PageRank, betweenness and closeness answer different questions and need not rank nodes similarly.

## Temporal-network section

Construct contacts $A\to B$ and $B\to C$ in two different temporal orders. The aggregated static graph is identical, but time-respecting paths differ. This gives a compact demonstration that aggregation can create paths that never existed.

## Reproducibility plan

Generate synthetic lattice, Erdős-Rényi and scale-heterogeneous networks with matched size and approximate mean degree. Compare spectral radius, epidemic trajectories and percolation robustness.

## Sources to develop

Newman, M. (2018). *Networks*.

Barabási, A.-L. (2016). *Network Science*.

Pastor-Satorras, R., Castellano, C., Van Mieghem, P., & Vespignani, A. (2015). Epidemic processes in complex networks.

Holme, P., & Saramäki, J. (2012). Temporal networks.
