---
permalink: '/machine-learning/rag_is_a_retrieval_system_before_it_is_an_llm_system/'
title: 'RAG Is a Retrieval System Before It Is an LLM System'
date: '2026-02-12'
categories:
- Machine Learning
tags:
- Retrieval Augmented Generation
- Information Retrieval
- RAG
- Reranking
- Vector Search
- Evaluation
- Large Language Models
author_profile: false
classes: wide
seo_title: 'RAG Is a Retrieval System Before It Is an LLM System'
seo_description: 'Most RAG failures begin before generation. This article treats chunking, retrieval, reranking, context construction and evaluation as the core of a RAG system.'
seo_type: article
excerpt: >-
  A RAG system can fail even when the language model is excellent. If the right
  evidence is not retrieved, ranked, retained and placed into context, generation
  cannot repair the missing information.
summary: >-
  A technical treatment of retrieval-augmented generation as an information
  retrieval pipeline. The article covers chunking, sparse and dense retrieval,
  hybrid search, reranking, context construction, long-context effects, retrieval
  metrics, answer faithfulness and end-to-end error decomposition.
keywords:
- RAG evaluation
- retrieval augmented generation
- vector search
- hybrid retrieval
- BM25
- dense passage retrieval
- reranking
- chunking
- faithfulness
why_this_exists: >-
  RAG is frequently presented as an LLM feature when its dominant engineering
  risks often sit in the retrieval system. A final answer can be wrong because
  evidence was never indexed, was chunked badly, was not retrieved, was ranked
  too low, was dropped from the context, or was ignored by the generator.
evidence: >-
  Foundational RAG and dense passage retrieval work, BEIR retrieval benchmarking,
  evidence on long-context position effects, and published work on decomposed RAG
  evaluation.
methodology: >-
  Decompose the RAG pipeline into corpus construction, chunking, indexing,
  candidate generation, reranking, context construction and generation. Attach
  appropriate metrics to each stage and show why a single end-to-end score cannot
  identify the failure mechanism.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-data-science-network.jpg
  og_image: /assets/images/headers/photo-data-science-network.jpg
  overlay_image: /assets/images/headers/photo-data-science-network.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-network.jpg
  twitter_image: /assets/images/headers/photo-data-science-network.jpg
---

Retrieval-augmented generation is often described as a language-model architecture. That description is incomplete. In practice, a RAG system is an information retrieval system whose final consumer happens to be a language model.

The distinction matters because most of the system's failure modes can occur before generation begins. A document may never enter the corpus. It may be split at the wrong boundary. The relevant chunk may be indexed poorly, retrieved too late, displaced by irrelevant candidates, removed during reranking, or placed into a part of the context that the model uses weakly. The generator may then produce a fluent answer from incomplete evidence, and the final output makes the entire pipeline look like an LLM failure.

It is therefore useful to treat RAG as a composition:

$$
x
\rightarrow
\mathcal C
\rightarrow
\mathcal R_k(x)
\rightarrow
\mathcal S_m(x)
\rightarrow
\mathcal K(x)
\rightarrow
p_\theta(y\mid x,\mathcal K(x)),
$$

where $x$ is the query, $\mathcal C$ the indexed corpus, $\mathcal R_k$ a candidate retriever returning $k$ items, $\mathcal S_m$ a selection or reranking stage retaining $m$ items, $\mathcal K(x)$ the final constructed context, and $p_\theta$ the generator.

![RAG retrieval and evaluation pipeline](/assets/images/articles/machine-learning/rag-retrieval-evaluation-pipeline.svg)

The central engineering principle is simple:

$$
\boxed{
\text{If the required evidence never reaches the model, generation cannot recover it reliably.}
}
$$

That is why retrieval quality must be measured independently from answer quality.

## The corpus is part of the model

A RAG system begins with a corpus, but the word *corpus* hides several design decisions. Which documents are included? Which versions are canonical? Which fields are searchable? How are duplicates handled? What metadata are preserved? Which documents can each user access? How quickly do additions, removals and corrections propagate into the index?

These questions are not peripheral. They define the information available to the system.

Let $D$ denote the set of all source documents that could in principle support a query, and let $\mathcal C\subseteq D$ be the indexed corpus. If the necessary evidence $z^\star$ satisfies

$$
z^\star\notin\mathcal C,
$$

then no retrieval model can recover it.

This sounds obvious, but corpus coverage is frequently ignored because RAG evaluation begins with a fixed benchmark set. In production, missing documents, stale versions and ingestion delays are common. A retrieval benchmark that assumes perfect corpus coverage measures only one layer of the system.

Versioning matters for the same reason. If two policy documents contradict one another and the older one remains indexed, the retriever can return both. The generator may choose either, merge them, or produce a compromise that no source actually supports. This is not fundamentally a hallucination problem. It is a corpus-governance problem.

## Chunking defines the unit of retrieval

Most RAG systems do not retrieve entire documents. They retrieve chunks. Chunking therefore defines the atomic evidence unit.

Suppose a document $d$ is partitioned into

$$
d
=
(c_1,c_2,\ldots,c_J).
$$

A fixed-size chunker may split every $L$ tokens, perhaps with overlap $o$. The method is simple and operationally convenient, but it can sever the relationship between a claim and the qualification that follows it.

Consider a sentence stating that a treatment reduces risk, followed immediately by a sentence restricting that conclusion to a subgroup. If the first sentence lands in $c_j$ and the qualification in $c_{j+1}$, retrieval of only $c_j$ changes the evidential content.

The issue is not that fixed-size chunking is always wrong. It is that chunking introduces an implicit model of locality: information required to answer a query is assumed to be sufficiently contained inside one or a few chunks.

Smaller chunks can increase retrieval specificity but fragment context. Larger chunks preserve local structure but consume more context budget and may dilute embedding similarity. Overlap reduces boundary damage but increases index size and redundancy.

If chunk length is $L$ and overlap is $o$, the approximate number of chunks in a document of length $N$ is

$$
J
\approx
1+
\left\lceil
\frac{N-L}{L-o}
\right\rceil.
$$

As $o$ increases, retrieval redundancy also increases. Several nearly identical chunks may occupy the top-$k$ results, giving the appearance of high relevance while reducing evidence diversity.

A serious chunking evaluation should therefore ask not only whether relevant text is retrieved, but whether the chunk boundaries preserve the evidential unit required by the task.

## Sparse and dense retrieval fail differently

Classical sparse retrieval such as BM25 represents queries and documents primarily through lexical overlap. Dense retrieval represents them in a learned continuous embedding space.

Neither dominates universally.

Sparse retrieval is strong when exact terminology matters. Product codes, legal citations, rare technical terms, names, identifiers and error messages can be retrieved extremely well because lexical matching preserves exact tokens.

Dense retrieval can bridge lexical variation. A query asking about "heart attack" may retrieve a passage containing "myocardial infarction" even when the surface forms differ. Dense Passage Retrieval demonstrated that learned dual encoders could substantially improve open-domain question-answering retrieval over a strong BM25 baseline on several datasets. citeturn221143search1

A dense retriever often computes embeddings

$$
q=f_\eta(x),
\qquad
v_j=g_\eta(c_j),
$$

and ranks chunks using a similarity such as

$$
s(x,c_j)
=
q^\top v_j
$$

or cosine similarity.

The weakness is that semantic similarity is not the same as evidential relevance. Two passages can be semantically close while differing in the precise fact that matters. A dense retriever can also generalize poorly when the deployment domain differs from its training distribution.

BEIR is important here because it evaluates retrieval across heterogeneous domains rather than one narrow benchmark. Its results show that BM25 remains a strong zero-shot baseline, while reranking and late-interaction models often improve effectiveness at greater computational cost. citeturn953962search24

The practical consequence is that "use embeddings" is not a retrieval strategy.

## Hybrid retrieval is often easier to justify than ideological purity

Sparse and dense retrieval have complementary failure modes, so hybrid systems are common.

Let

$$
s_{\mathrm{sparse}}(x,c)
$$

be a lexical score and

$$
s_{\mathrm{dense}}(x,c)
$$

a semantic score. A simple weighted combination is

$$
s_{\mathrm{hybrid}}(x,c)
=
\lambda s_{\mathrm{sparse}}(x,c)
+
(1-\lambda)s_{\mathrm{dense}}(x,c).
$$

In practice, score scales often differ, so direct weighted addition may require calibration or normalization. Rank-fusion methods avoid some of that problem by combining ranked lists rather than raw scores.

The important point is that hybrid retrieval is not inherently more advanced. It is simply an attempt to preserve both exact-match and semantic-recall pathways.

The value should be demonstrated against the deployment query distribution. If dense retrieval already captures all relevant cases, hybrid complexity may add little. If the domain contains many rare identifiers and paraphrases, the combination can be much more robust.

## Candidate generation should optimize recall, not elegance

The first-stage retriever usually returns a candidate set

$$
\mathcal R_k(x)
=
\{c_{(1)},\ldots,c_{(k)}\}.
$$

At this stage the primary concern is usually recall. If the relevant evidence is absent from $\mathcal R_k(x)$, later stages cannot recover it.

For a set of relevant chunks $G(x)$, recall at $k$ is

$$
\mathrm{Recall@}k
=
\frac{
|G(x)\cap\mathcal R_k(x)|
}{
|G(x)|
}.
$$

For single-evidence tasks, hit rate at $k$ is often sufficient:

$$
\mathrm{Hit@}k
=
\mathbf 1
\{G(x)\cap\mathcal R_k(x)\neq\varnothing\}.
$$

These metrics should be computed before generation. Otherwise a good generator can hide mediocre retrieval on easy questions, while a weak generator can make strong retrieval appear poor.

The candidate-set size $k$ creates a trade-off. A larger $k$ increases the probability that relevant evidence is present, but also increases downstream reranking cost and the chance of introducing distracting material.

This is why the retrieval stage and context stage should not be collapsed into one "top-k" parameter.

## Reranking changes the objective from recall to precision

A first-stage retriever must search a large corpus efficiently, so its scoring model is usually cheap. A reranker can afford a more expensive model over a much smaller candidate set.

Let the first stage produce $k$ candidates and a reranker assign scores

$$
r_\phi(x,c_j).
$$

The reranked top $m$ chunks are

$$
\mathcal S_m(x)
=
\operatorname{Top}_m
\{
r_\phi(x,c_j):
c_j\in\mathcal R_k(x)
\}.
$$

This architecture separates two goals:

$$
\text{candidate retrieval}
\rightarrow
\text{high recall},
$$

$$
\text{reranking}
\rightarrow
\text{higher precision}.
$$

A cross-encoder reranker can inspect query and passage jointly, capturing interactions that a dual encoder compresses into independent embeddings. The cost is latency.

Late-interaction methods occupy an intermediate point between full cross-encoding and single-vector dense retrieval. BEIR's results are a useful reminder that retrieval quality and computational cost should be evaluated together rather than treating architecture complexity as free. citeturn953962search24

## Context construction is not a neutral formatting step

After retrieval and reranking, the system must build a prompt. This stage is often treated as concatenation, but it changes the information available to the model.

Suppose the final selected chunks are

$$
c_1,\ldots,c_m
$$

with lengths

$$
\ell_1,\ldots,\ell_m.
$$

Given a context budget $C$,

$$
\sum_{j=1}^{m}\ell_j
+
\ell_{\mathrm{instructions}}
+
\ell_{\mathrm{query}}
\leq C.
$$

If the selected evidence exceeds the budget, something must be removed, compressed or summarized. That creates another ranking decision.

Ordering matters as well. Liu et al. showed that long-context language models can exhibit strong position effects, with performance often degrading when relevant information is placed in the middle of a long context. citeturn221143search0

This means that successful retrieval does not guarantee successful use.

A system can achieve perfect Recall@10 and still answer incorrectly because the one decisive passage appears among nine distractors, is placed unfavourably, or is contradicted by other retrieved chunks.

Context construction should therefore be evaluated as its own stage.

Useful properties include:

- evidence coverage,
- redundancy,
- contradiction rate,
- token efficiency,
- source diversity,
- ordering sensitivity,
- whether citations can be mapped back to exact spans.

The context is a finite decision surface, not a dumping ground.

## More context can make the answer worse

A common response to retrieval uncertainty is to increase $k$ and send more material to the model.

That can fail.

Suppose the probability of retrieving at least one relevant chunk rises with $k$:

$$
P(G(x)\cap\mathcal R_k(x)\neq\varnothing)
\uparrow.
$$

At the same time, the expected number of irrelevant chunks also rises. If the generator's ability to distinguish evidence from distractors does not improve proportionally, answer accuracy can decline.

The resulting performance curve need not be monotone:

$$
\mathrm{Accuracy}(k+1)
<
\mathrm{Accuracy}(k)
$$

for some values of $k$.

This is not paradoxical. Retrieval recall and context usefulness are different quantities.

Longer context windows reduce one hard constraint but do not remove the selection problem. The "lost in the middle" evidence shows that models do not necessarily use all positions equally well even when the entire prompt fits technically. citeturn221143search0

The engineering question is therefore not "How much context can the model accept?" but "What is the smallest context that preserves all evidence required for the answer?"

## Retrieval metrics and ranking metrics answer different questions

Recall at $k$ asks whether relevant evidence entered the candidate set. Precision at $k$ asks how much of the returned set is relevant.

$$
\mathrm{Precision@}k
=
\frac{
|G(x)\cap\mathcal R_k(x)|
}{k}.
$$

Mean reciprocal rank rewards systems that place the first relevant result early:

$$
\mathrm{RR}(x)
=
\frac{1}{
\operatorname{rank}(\text{first relevant result})
}.
$$

Mean reciprocal rank averages this quantity across queries.

Normalized discounted cumulative gain is more appropriate when there are several graded relevance levels. It rewards relevant results near the top while allowing different relevance strengths.

No one metric is sufficient for every RAG system.

A question-answering system that needs one decisive paragraph may care strongly about MRR and Hit@k. A research assistant synthesizing multiple sources may care more about recall, source diversity and graded relevance. A compliance system may care about whether all mandatory evidence is present, making set coverage more important than first-hit ranking.

The metric should reflect what the generator needs.

## Final-answer correctness is not retrieval evaluation

Suppose a question has one correct answer and the language model already knows it parametrically. A RAG system can return irrelevant chunks and still answer correctly.

End-to-end accuracy then gives the retrieval system credit for information it did not provide.

The reverse can also happen. The retriever can return perfect evidence and the generator can misread it. End-to-end accuracy then penalizes retrieval for a generation failure.

The two cases are observationally identical if only the final answer is scored.

A useful decomposition is

$$
P(\text{correct})
=
P(\text{correct}\mid E)P(E)
+
P(\text{correct}\mid E^c)P(E^c),
$$

where $E$ denotes the event that sufficient evidence reaches the final context.

The first term measures generation conditional on evidence availability. The second captures cases where the model answers without sufficient retrieved support.

A grounded system should make that second term visible rather than silently benefiting from parametric memory.

## Faithfulness is different from correctness

An answer can be correct but unsupported by the retrieved evidence. It can also be faithful to the retrieved evidence while the evidence itself is wrong.

Let $A$ denote answer correctness and $F$ answer faithfulness to context. Then all four combinations are possible:

| | Faithful | Unfaithful |
| --- | --- | --- |
| Correct | desired grounded answer | correct for the wrong evidential reason |
| Incorrect | faithfully reproduces bad or incomplete evidence | unsupported failure |

This distinction is central to RAG evaluation.

RAGAS formalized this decomposed perspective by evaluating retrieval relevance, faithfulness and generation quality as separate dimensions rather than collapsing everything into one score. citeturn221143academia49

Automated evaluators are useful for iteration, but they are estimators too. If an LLM judge scores faithfulness, its own calibration and failure modes become part of the measurement process. High-stakes systems still need human-reviewed evaluation sets and explicit evidence labels.

## Citation correctness deserves its own test

Many RAG applications expose citations. A citation is not automatically evidence.

Suppose an answer contains claim $a_j$ and cites document $d_i$. Citation correctness asks whether the cited source actually supports that claim.

A simple support indicator is

$$
S_{ij}
=
\mathbf 1
\{
d_i\text{ entails or directly supports }a_j
\}.
$$

Citation precision can then be defined over cited claim-source pairs. Citation recall asks whether claims requiring external support have citations at all.

This matters because a model can attach a plausible source to a sentence without deriving the sentence from that source. The user sees a citation and infers provenance that may not exist.

For scientific, legal and policy applications, citation support should be tested independently from answer fluency.

## Query rewriting is another model with another failure mode

Many modern RAG systems rewrite the user query before retrieval. This can help when the original prompt is conversational or underspecified.

Let

$$
x'
=
q_\psi(x)
$$

be a rewritten query. Retrieval then uses $x'$ rather than $x$.

The rewrite can improve recall by making the information need explicit, but it can also alter the question. A user asking whether a treatment is "safe in older adults with kidney disease" may receive a rewrite focused only on "treatment safety in older adults," silently dropping the renal condition.

Query rewriting should therefore be evaluated for semantic preservation, not merely downstream retrieval score.

The same applies to decomposition into subqueries for multi-hop questions. Decomposition can improve retrieval, but each generated subquery adds a new point where the information need can drift.

## Multi-hop questions expose single-retrieval assumptions

Some questions cannot be answered from one chunk.

Suppose the answer requires combining

$$
z_1
\land
z_2
\land
z_3.
$$

If each evidence item has retrieval probability $p_j$, then under a crude independence approximation, the probability of retrieving all required evidence is

$$
P(\text{all evidence})
\approx
\prod_{j=1}^{3}p_j.
$$

Even reasonably strong per-hop recall can produce poor joint coverage.

This motivates iterative retrieval, query decomposition and retrieval conditioned on intermediate reasoning. But those methods should be viewed as attempts to solve a coverage problem, not as mystical "agentic RAG."

The relevant metric is whether the required evidence graph was recovered.

## RAG evaluation needs an explicit error taxonomy

A useful production evaluation set should classify failures by stage.

A minimal taxonomy is:

1. **Corpus failure**: the evidence is absent or stale.
2. **Chunking failure**: the evidence is split or stripped of necessary context.
3. **Retrieval failure**: the relevant chunk exists but is not in the candidate set.
4. **Ranking failure**: the relevant chunk is retrieved but ranked too low.
5. **Context failure**: the relevant chunk is selected but dropped, truncated, duplicated or badly ordered.
6. **Generation failure**: sufficient evidence is present but the model answers incorrectly.
7. **Grounding failure**: the answer is correct or plausible but unsupported by the supplied evidence.
8. **Citation failure**: the cited source does not support the attributed claim.

Without this taxonomy, teams often respond to every bad answer by changing prompts or models. That is an expensive way to debug a retrieval problem.

## A useful RAG experiment is factorial

When improving a system, change one layer at a time.

For example, compare:

$$
\begin{array}{c|c|c}
 & \text{BM25} & \text{Dense} \\
\hline
\text{No reranker} & A & B \\
\text{Cross-encoder reranker} & C & D
\end{array}
$$

and evaluate retrieval before generation.

Then vary chunk size, overlap, candidate count and final context count.

A larger experiment might estimate

$$
Y
=
f(
\text{chunk size},
\text{retriever},
\text{reranker},
k,
m,
\text{ordering}
).
$$

The objective should not be one global "RAG score." It should include the metrics tied to each stage.

This is standard experimental design applied to an LLM system.

## The baseline should include no retrieval

A RAG evaluation should always include the base model without retrieval.

If the base model already answers a benchmark correctly, retrieval may add cost without value. Worse, poor retrieval can reduce performance by introducing distractors.

Useful baselines include:

$$
\text{Base model},
$$

$$
\text{Base + oracle evidence},
$$

$$
\text{Base + actual retrieval}.
$$

The oracle-evidence condition is especially informative. It isolates generator capability from retrieval capability.

If the model fails with oracle evidence, improving the retriever will not solve the problem.

If the model succeeds with oracle evidence but fails with actual retrieval, the retrieval pipeline is the obvious target.

This simple three-way comparison often tells more than another prompt-engineering round.

## Retrieval should be evaluated on the deployment distribution

A retriever that performs well on Natural Questions or a generic benchmark may fail in a corporate knowledge base containing acronyms, tables, product identifiers and duplicated documentation.

BEIR's contribution is precisely the demonstration that retrieval systems can behave very differently across domains. citeturn953962search24

Let

$$
P_{\mathrm{bench}}(x,z)
$$

be the benchmark distribution and

$$
P_{\mathrm{deploy}}(x,z)
$$

the real deployment distribution.

If

$$
P_{\mathrm{bench}}
\neq
P_{\mathrm{deploy}},
$$

then benchmark performance is not a direct estimate of deployment retrieval quality.

A useful RAG test set should therefore be sampled from real information needs, including ambiguous queries, rare identifiers, multi-hop questions, stale-document traps, permission boundaries and queries for which no answer exists.

The no-answer cases are particularly important. A retriever will almost always return something. The system must learn when the returned material is insufficient.

## RAG does not eliminate hallucination; it changes the conditions under which hallucination occurs

The original RAG work was motivated partly by limitations of purely parametric memory, including difficulty updating knowledge and providing provenance. citeturn953962search25

Retrieval helps because the model can condition on inspectable external evidence. It does not force the model to obey that evidence.

The generator can still:

- ignore relevant context,
- combine incompatible sources,
- overgeneralize beyond what the sources state,
- invent bridging claims,
- cite the wrong passage,
- answer from parametric memory instead of retrieved evidence.

RAG should therefore be understood as an information-access architecture, not a truth guarantee.

## The right debugging question is "where was the evidence lost?"

When a RAG answer is wrong, the most useful question is not whether the language model hallucinated.

It is:

> At which stage did the evidence required for a correct answer disappear or become unusable?

That question produces an actionable debugging path.

Was the source absent from the corpus? Was the chunk malformed? Did sparse retrieval miss a paraphrase? Did dense retrieval miss an identifier? Did the reranker demote the correct passage? Did context truncation remove it? Was it placed among distractors? Did the generator ignore it? Did the citation point somewhere else?

Each answer implies a different intervention.

That is why a serious RAG system should be observable at every stage. For each query, one should be able to inspect the original query, rewritten query if any, candidate set, scores, reranked set, final context, response and citations.

Without that trace, RAG debugging becomes guesswork.

The language model is the visible part of the system. Retrieval is the part that determines which world the model is allowed to see.

## References

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*, 12, 157–173. https://doi.org/10.1162/tacl_a_00638

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *Advances in Neural Information Processing Systems*, 34.
