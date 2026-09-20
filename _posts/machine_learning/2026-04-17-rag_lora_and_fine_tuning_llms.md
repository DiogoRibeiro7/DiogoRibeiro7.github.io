---
permalink: '/machine-learning/rag_lora_and_fine_tuning_llms/'
title: 'RAG, LoRA, and Fine-Tuning Solve Different LLM Problems'
date: '2026-04-17'
categories:
- Machine Learning
tags:
- Large Language Models
- Retrieval Augmented Generation
- Fine Tuning
- LoRA
- QLoRA
- PEFT
- NLP
author_profile: false
classes: wide
seo_title: 'RAG, LoRA, and Fine-Tuning Solve Different LLM Problems'
seo_description: 'RAG changes the information an LLM sees at inference time. Fine-tuning changes model parameters. LoRA and QLoRA are parameter-efficient ways to fine-tune. This article explains when each approach is appropriate and why they are often complementary.'
seo_type: article
excerpt: >-
  RAG, LoRA and fine-tuning are often presented as competing ways to improve an
  LLM. That framing is wrong. They intervene at different parts of the system,
  solve different failure modes, and are often most useful in combination.
summary: >-
  A systems-level treatment of retrieval-augmented generation, supervised and
  domain-adaptive fine-tuning, LoRA, QLoRA, and preference optimization. The
  article separates inference-time context from parameter updates, derives the
  low-rank LoRA update, explains the operational consequences of each method,
  and gives a decision framework based on knowledge freshness, behavioural
  adaptation, latency, cost, evaluation and governance.
keywords:
- RAG
- retrieval augmented generation
- LoRA
- QLoRA
- fine tuning LLM
- PEFT
- large language models
- supervised fine tuning
- domain adaptive pretraining
why_this_exists: >-
  Discussions of LLM adaptation often place RAG, LoRA and fine-tuning in one
  comparison table as though they were mutually exclusive techniques. They are
  not. RAG changes inference-time context, fine-tuning changes model parameters,
  and LoRA changes how those parameter updates are represented and trained.
evidence: >-
  Foundational work on retrieval-augmented generation, LoRA, QLoRA, domain
  adaptive pretraining and preference optimization, together with current PEFT
  and supervised fine-tuning tooling.
methodology: >-
  Analyse each method by the intervention point in the LLM system, the objective
  it optimises, the information it can change, the operational state it adds,
  and the failure modes it cannot repair. Use the RAG marginalisation and LoRA
  low-rank update equations to make the distinction explicit.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  og_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  overlay_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-ml-pipeline.jpg
  twitter_image: /assets/images/headers/photo-data-science-ml-pipeline.jpg
---

RAG, LoRA and fine-tuning are often introduced as three competing strategies for improving a large language model. The resulting advice usually takes the form of a decision table: use RAG for knowledge, LoRA for cost, and fine-tuning for behaviour. That summary is not entirely wrong, but it hides the most important structural fact.

These methods do not intervene at the same place.

Retrieval-augmented generation changes the information supplied to the model at inference time. Fine-tuning changes the parameters of the model. LoRA does not define a separate objective at all. It is a parameter-efficient way of representing and learning a parameter update. QLoRA goes one step further by keeping the base model quantized while training LoRA adapters.

That distinction is not semantic. It determines what each method can repair, what new state must be operated in production, how rapidly the system can incorporate new information, and how failures should be diagnosed.

![Where RAG, fine-tuning, LoRA and QLoRA intervene in an LLM system](/assets/images/articles/machine-learning/rag-lora-finetuning-system.svg)

A useful starting point is therefore:

$$
\boxed{
\text{RAG changes context}
\qquad
\text{Fine-tuning changes parameters}
\qquad
\text{LoRA changes the parameter update}
}
$$

Once that is clear, many apparent disagreements disappear.

## RAG changes what the model can condition on

A pretrained language model represents information parametrically through its weights. If the model has parameters $\theta$, generation can be written schematically as

$$
p_\theta(y\mid x),
$$

where $x$ is the prompt and $y$ is the generated response.

Retrieval-augmented generation introduces an external information source. A retriever selects documents, chunks, records or other evidence $z$ from a corpus, and the generator conditions on both the original query and the retrieved material. In the formulation of Lewis et al. (2020), the response probability can be written as

$$
p(y\mid x)
=
\sum_{z}
p_\eta(z\mid x)
p_\theta(y\mid x,z),
$$

where $p_\eta(z\mid x)$ is the retrieval distribution and $p_\theta(y\mid x,z)$ is the generator.

The operational meaning is more important than the notation. The model parameters do not need to contain the latest company policy, yesterday's clinical guideline, a newly published paper, or a private customer record. Those facts can remain outside the model and be supplied when needed.

This gives RAG one major property that parameter updates cannot reproduce cheaply: **knowledge can be changed without retraining the model**.

If a document is corrected, removed or superseded, the corpus can be updated. If access control changes, retrieval can respect a new permission boundary. If provenance matters, the system can retain the source that produced the answer. This separation between model and knowledge store is particularly valuable when facts change faster than the model should be retrained.

But RAG is not a mechanism for making the model intrinsically better at reasoning, writing or following instructions. A weak generator remains weak. A model that repeatedly produces an undesirable response style may continue to do so even when the correct facts are retrieved. A model that does not understand a specialised output format will not necessarily learn that format merely because a document describing it appears in the context.

RAG also creates its own statistical and engineering problem. The system now depends on a retrieval pipeline. Errors can arise from chunking, embedding, indexing, filtering, query rewriting, ranking, reranking and context construction before the generator has produced a single token.

For a query $x$, suppose the relevant evidence is $z^\star$. If the retriever does not place $z^\star$ in the candidate set, the generator cannot use it. The overall probability of a correct answer is therefore constrained by retrieval recall:

$$
P(\text{correct})
\leq
P(z^\star \in \mathcal R_k(x)),
$$

for any task in which the answer requires that evidence.

This is why evaluating only the final answer is insufficient. A RAG system should usually be decomposed into retrieval and generation components. Useful quantities include recall at $k$, precision at $k$, ranking quality, answer faithfulness, citation correctness and the sensitivity of the answer to irrelevant retrieved context.

RAG also adds latency and mutable state. The vector index, source documents, metadata, embedding model and access-control rules become part of the deployed system. That complexity is often justified, but it should be counted rather than hidden behind the phrase "just add retrieval."

## Fine-tuning changes the model itself

Fine-tuning starts from a pretrained parameter vector $\theta_0$ and optimizes some objective on new data. In supervised fine-tuning, a dataset of input-output pairs

$$
\mathcal D
=
\{(x_i,y_i)\}_{i=1}^n
$$

may be used to minimize the token-level negative log-likelihood

$$
\mathcal L(\theta)
=
-
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i).
$$

Unlike RAG, the result of this optimization is a different model.

This makes fine-tuning suitable for problems that are genuinely about model behaviour. Examples include learning a stable response schema, adapting to a domain-specific interaction pattern, using specialised terminology consistently, following a narrow instruction format, learning tool-call conventions, or improving performance on a recurring task distribution.

The important word is *behaviour*. Fine-tuning is often misused as a knowledge-loading mechanism because it is tempting to imagine the weights as a database. They are not.

Suppose an organisation has 10,000 policy documents that change every week. Training on those documents does not provide a clean mechanism for replacing one paragraph, revoking one fact, recovering one source or enforcing document-level access control. A model may absorb some patterns from the corpus, but the resulting knowledge is entangled with the parameters. Updating and auditing it becomes difficult.

Fine-tuning can also create forgetting and interference. Improving performance on one distribution may damage another. A domain-adapted model can become more fluent in specialised language while becoming less robust elsewhere. A supervised dataset can teach formatting shortcuts or annotation artefacts rather than the intended behaviour. The training loss may fall while the deployed task deteriorates.

For this reason, "fine-tuning" should itself be decomposed. Several distinct procedures are commonly grouped under that name.

### Continued or domain-adaptive pretraining

The model continues next-token training on a domain corpus. If $t_1,\ldots,t_T$ are tokens, the objective remains approximately

$$
\mathcal L_{\mathrm{LM}}
=
-
\sum_{t=1}^{T}
\log p_\theta(t_t\mid t_{<t}).
$$

This can adapt representation and vocabulary usage to domains such as law, biomedicine, finance or scientific literature. Gururangan et al. (2020) showed that continued pretraining on domain and task distributions can improve downstream performance even after broad pretraining.

The procedure is different from supervised instruction tuning because there need not be input-output instruction pairs. The model is adapting to the statistical structure of the domain.

### Supervised fine-tuning

Supervised fine-tuning teaches a model how to respond to inputs by training on demonstrations. This is the natural mechanism when the main problem is output behaviour.

For example, if a model must transform a clinical note into a strict JSON schema, retrieve tool arguments from a user request, or produce a particular style of technical report, a curated set of examples can directly optimise that behaviour.

### Preference optimization

Methods such as reinforcement learning from human feedback and direct preference optimization operate on preferences between candidate responses rather than merely on demonstration likelihood. A preference dataset may contain triples

$$
(x,y^+,y^-),
$$

where $y^+$ is preferred to $y^-$. DPO optimizes the policy relative to a reference model using these pairwise comparisons without fitting a separate reward model in the original formulation.

This again solves a different problem. Preference optimization can alter helpfulness, verbosity, refusal behaviour, tone or other comparative response properties. It is not a substitute for a current knowledge base.

These distinctions matter because a single phrase, "we need to fine-tune the model," can hide four entirely different projects.

## LoRA is fine-tuning with a constrained update

Full fine-tuning updates a large fraction or all of the parameters in a model. For modern LLMs that can mean billions of trainable parameters, large optimizer states and substantial GPU memory.

LoRA, introduced by Hu et al. (2021), starts from the observation that the update needed for adaptation may lie in a much lower-dimensional subspace than the full parameter space.

Consider a weight matrix

$$
W_0\in\mathbb R^{d\times k}.
$$

Full fine-tuning learns an unrestricted update

$$
\Delta W\in\mathbb R^{d\times k},
$$

so that

$$
W'=W_0+\Delta W.
$$

LoRA instead parameterizes the update as

$$
\Delta W
=
BA,
$$

with

$$
B\in\mathbb R^{d\times r},
\qquad
A\in\mathbb R^{r\times k},
\qquad
r\ll\min(d,k).
$$

The adapted weight is therefore

$$
W'
=
W_0+BA.
$$

Rather than train $dk$ parameters for that matrix, LoRA trains approximately

$$
r(d+k).
$$

When $r$ is small, the reduction can be dramatic.

The base matrix $W_0$ remains frozen. Only the low-rank factors are optimized. In practice, adapters are usually inserted into selected linear projections such as attention or feed-forward layers, although the exact target modules depend on the architecture and task.

This explains why LoRA should not be placed beside fine-tuning as though they were different objectives. LoRA answers the question:

> How should we parameterize the fine-tuning update?

It does not answer:

> What training objective should we use?

LoRA can be used during supervised fine-tuning. It can be used during preference optimization. It can be used for domain adaptation. The objective and the parameterization are separate choices.

This distinction also explains why LoRA does not automatically make fine-tuning safe. If the training data are poor, the adapter learns poor behaviour efficiently. If the validation design leaks, LoRA leaks more cheaply. If the task distribution is wrong, reducing the number of trainable parameters does not repair the mismatch.

Low rank is a structural assumption about the update, not a quality guarantee.

## QLoRA reduces the memory cost further

QLoRA, introduced by Dettmers et al. (2023), combines low-rank adaptation with a quantized frozen base model. Instead of storing the base model in full training precision, QLoRA uses 4-bit quantization for the frozen weights while computing and updating the LoRA parameters at higher precision.

Conceptually,

$$
W_0
\rightarrow
Q_4(W_0),
$$

while the learned update remains

$$
\Delta W=BA.
$$

The resulting forward pass is approximately

$$
h
=
Q_4(W_0)x
+
BAx.
$$

The important point is that the quantized base weights are not being trained in the usual sense. The memory saving comes from freezing and quantizing them while training the adapters.

This makes QLoRA attractive when local or single-GPU adaptation would otherwise be impossible. It does not mean that quantization is free. Numerical approximation, kernel support, training stability and inference configuration still matter. Nor does QLoRA change the conceptual role of adaptation. It remains fine-tuning through a parameter-efficient update.

## RAG and fine-tuning solve different failure modes

A system should be diagnosed before choosing an adaptation method.

Suppose the model answers an internal-policy question incorrectly because the policy changed yesterday. Fine-tuning the model each time a policy changes is operationally awkward and makes provenance difficult. RAG is the natural intervention because the failure is missing or stale context.

Now suppose the model receives the correct policy paragraph but repeatedly fails to output the answer in the company's required structured format. Retrieval is not the main problem. The information is present, but the response behaviour is wrong. Supervised fine-tuning may be appropriate.

Suppose the model understands the format but full fine-tuning is too expensive. That is where LoRA or QLoRA becomes relevant.

The methods can therefore be arranged by failure mechanism:

| Failure | Primary intervention |
| --- | --- |
| Knowledge is current but absent from the prompt | Retrieval / RAG |
| Knowledge changes frequently | Retrieval / RAG |
| Source provenance or document-level permissions matter | Retrieval / RAG |
| Output format is consistently wrong | Supervised fine-tuning |
| Domain language or representation is weak | Domain-adaptive pretraining or fine-tuning |
| Response preferences need systematic adjustment | Preference optimization |
| Full fine-tuning is too expensive | LoRA / other PEFT |
| Fine-tuning memory is still too high | QLoRA |
| Retrieval is poor | Fix retrieval, not the generator |
| Both knowledge and behaviour are inadequate | RAG + fine-tuning |

The final row is important. RAG and fine-tuning are often complementary.

A legal assistant may need RAG because statutes and internal precedents change, while also using a fine-tuned adapter to produce stable legal drafting conventions. A scientific assistant may retrieve papers dynamically while using supervised fine-tuning to produce a strict evidence table. A customer-support model may retrieve the latest product documentation while a LoRA adapter teaches the required response format and escalation policy.

The system can be written schematically as

$$
y
\sim
p_{\theta+\Delta\theta}
\left(
y
\mid
x,
R(x)
\right),
$$

where $R(x)$ is retrieved context and $\Delta\theta$ may be a LoRA update.

Nothing in this expression requires choosing retrieval *or* adaptation.

## Fine-tuning knowledge is harder to govern than retrieving it

One of the strongest reasons to keep volatile knowledge outside the model is governance.

Suppose a source document contains a sentence that later becomes legally invalid. In a RAG system, the document can be removed or corrected and the index rebuilt. The provenance trail can show which version was retrieved.

If the same sentence has been absorbed through parameter training, the relationship between source and output is indirect. There is no simple delete operation equivalent to removing a record from a database. Machine unlearning is an active research area precisely because parameterized knowledge does not behave like a row in a table.

Access control creates a similar issue. RAG can apply permissions before retrieval:

$$
\mathcal R_k(x,u)
\subseteq
\{z: u\text{ is authorized to access }z\}.
$$

The generator therefore only sees documents the user is allowed to retrieve.

Fine-tuning a single model on data belonging to several permission domains does not give the same guarantee. Once information is embedded in shared parameters, enforcing record-level authorization becomes much harder.

This does not make fine-tuning unsuitable for proprietary data. It means the training objective and governance model must match the information being learned. Stable style, terminology and task behaviour are easier to justify in parameters than rapidly changing or permission-sensitive facts.

## RAG has a context budget, fine-tuning has a capacity budget

Both approaches are constrained, but in different ways.

RAG must fit useful evidence into the model's context window. If the retriever returns too much material, irrelevant chunks can dilute the signal. If it returns too little, required evidence may be absent. A larger context window does not remove the ranking problem because attention remains a finite computational resource and long contexts can make evidence selection harder.

Let the context budget be $C$ tokens. If retrieved chunks have lengths $\ell_1,\ldots,\ell_k$, then the construction must satisfy approximately

$$
\sum_{j=1}^{k}\ell_j
+
\ell_{\mathrm{prompt}}
\leq C.
$$

Chunking and ranking therefore determine which information receives that scarce budget.

Fine-tuning has a different capacity problem. Parameter updates compress training examples into a shared model. The model does not retain a transparent copy of every demonstration. It learns a distributed representation that may generalize, interfere or forget.

This is exactly why memorizing a document collection and learning a behaviour are different problems. Behaviour benefits from generalization. A knowledge base often benefits from exact retrieval.

## Evaluation should match the intervention

A common mistake is to compare RAG and fine-tuning using one aggregate benchmark score. That can hide where the improvement came from.

For RAG, evaluation should separate at least three layers:

1. **Retrieval quality**: did the system retrieve the required evidence?
2. **Grounding quality**: did the model use the evidence rather than ignore or contradict it?
3. **Task quality**: was the final response correct and useful?

For fine-tuning, the evaluation should compare the adapted and base models on both target and non-target distributions. The central questions include:

- Did the target behaviour improve?
- Did unrelated capability degrade?
- Did the model become more brittle to prompt variation?
- Did it memorize training examples?
- Is the improvement robust across held-out tasks and users?

For LoRA, there is an additional structural choice: adapter rank $r$, target modules, scaling, dropout and which layers receive adapters. These are hyperparameters. A smaller trainable parameter count does not eliminate model selection.

The evaluation dataset must also be insulated from training data. This sounds obvious, but LLM adaptation workflows make contamination easy because instruction datasets, retrieval corpora and evaluation sets may all be assembled from overlapping document sources.

A useful experimental design compares:

$$
\text{Base},
\quad
\text{Base+RAG},
\quad
\text{Fine-tuned},
\quad
\text{Fine-tuned+RAG}.
$$

If compute permits, the adaptation branch can also compare full fine-tuning, LoRA and QLoRA while holding the training objective constant.

That factorial view reveals interaction effects. RAG may help the base model substantially but help the fine-tuned model less. Fine-tuning may improve formatting without improving factuality. The combination may outperform either alone, or one component may prove unnecessary.

## Latency, cost and maintenance are system properties

Training cost receives most of the attention, but production cost often matters more.

RAG adds retrieval latency, embedding infrastructure, an index, document ingestion and potentially reranking. Fine-tuning adds training jobs, checkpoints, evaluation and model-version management. LoRA reduces storage because multiple tasks can share one base model with small adapters, but adapter loading and routing still need operational design.

Suppose there are $m$ task variants. Full fine-tuning may require storing approximately

$$
mP
$$

parameters for models of size $P$, whereas a shared base plus LoRA adapters may require approximately

$$
P
+
\sum_{j=1}^{m}A_j,
$$

where $A_j\ll P$ is the size of adapter $j$.

This can be a major advantage when many clients or tasks require modest behavioural adaptation.

RAG scales differently. The expensive state is not primarily model copies but the corpus, index and retrieval traffic. A multi-tenant RAG system may therefore be cheaper in model storage but more complex in data governance and indexing.

There is no universal cheaper option because the cost surfaces are different.

## A practical decision rule

A useful sequence is to ask what is wrong before asking which technology to use.

**First, test the base model with the necessary information already in the prompt.** If the model succeeds when the correct evidence is supplied, the problem may be information access rather than model capability. Build retrieval before retraining.

**Second, test whether the information changes frequently.** If it does, keep it external unless there is a strong reason not to. Rapidly changing facts are usually a poor target for parameter updates.

**Third, test whether the failure is systematic behaviour.** If the model repeatedly ignores a schema, uses the wrong terminology, fails a narrow task or responds in an unacceptable style even when the right information is present, supervised adaptation is more plausible.

**Fourth, choose the adaptation mechanism.** Full fine-tuning is not automatically the baseline. If LoRA reaches the required quality, training billions of unnecessary parameters is wasteful. QLoRA is worth considering when memory is the limiting resource.

**Fifth, separate knowledge evaluation from behaviour evaluation.** A model can be perfectly grounded and badly formatted, or perfectly formatted and factually wrong.

This can be summarized as

$$
\boxed{
\begin{aligned}
\text{Missing current facts} &\Rightarrow \text{RAG}\\
\text{Wrong learned behaviour} &\Rightarrow \text{Fine-tuning}\\
\text{Fine-tuning too expensive} &\Rightarrow \text{LoRA}\\
\text{Fine-tuning memory too high} &\Rightarrow \text{QLoRA}\\
\text{Both problems present} &\Rightarrow \text{RAG + adaptation}
\end{aligned}
}
$$

That sequence is more useful than asking whether RAG is "better" than LoRA because the methods do not occupy the same conceptual axis.

## The hardest problems remain data problems

The fashionable part of LLM engineering is choosing architectures and adaptation methods. In practice, the difficult part is often the data.

A RAG system needs a corpus with reliable provenance, useful chunk boundaries, metadata, access controls and a retrieval evaluation set. Fine-tuning needs examples that represent the desired behaviour rather than merely the easiest examples to collect. Preference optimization needs preference data whose annotation process reflects the actual deployment objective.

Poor data can make every method look better in development than it is in production.

If training demonstrations are generated by the same model being evaluated, errors can reinforce themselves. If retrieval evaluation queries are copied from document headings, recall estimates can be artificially high. If fine-tuning examples contain stylistic shortcuts, the model may learn those instead of the intended reasoning procedure.

The relevant question is therefore not only which adaptation algorithm is used, but which statistical population the data represent.

Let the deployment distribution be

$$
P_{\mathrm{deploy}}(x,y),
$$

and the adaptation distribution be

$$
P_{\mathrm{train}}(x,y).
$$

If these differ materially, optimization of training loss does not guarantee deployment improvement. This is ordinary distribution shift appearing inside an LLM workflow.

The same discipline used elsewhere in machine learning still applies: define the target population, split data correctly, state the estimand, measure uncertainty and test on data that represent the actual decision problem.

## The architecture should follow the failure mechanism

RAG is best understood as external memory. Fine-tuning is parameter adaptation. LoRA and QLoRA are efficient parameter-update mechanisms. Preference optimization changes comparative response behaviour. None of these concepts makes the others obsolete.

The most robust LLM systems often use several layers deliberately:

$$
\text{base model}
+
\text{behavioural adaptation}
+
\text{retrieval}
+
\text{tooling}
+
\text{evaluation}.
$$

The mistake is to begin with a technology and search for a problem that justifies it.

If the model is missing information, improve information access. If it has the information but behaves incorrectly, adapt the model. If adaptation is expensive, constrain the update with PEFT methods such as LoRA. If memory is still the bottleneck, quantize the frozen base and train adapters with QLoRA.

The question is not whether RAG, LoRA or fine-tuning wins.

The question is where the error enters the system, and which intervention changes that part of the system with the least unnecessary complexity.

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*, 36.

Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *Proceedings of ACL 2020*, 8342–8360.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *Advances in Neural Information Processing Systems*, 36.
