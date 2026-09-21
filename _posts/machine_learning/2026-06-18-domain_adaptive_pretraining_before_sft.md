---
permalink: '/machine-learning/domain_adaptive_pretraining_before_sft/'
title: 'Domain-Adaptive Pretraining Comes Before Instruction Tuning'
date: '2026-06-18'
categories:
- Machine Learning
tags:
- Continued Pretraining
- Domain Adaptation
- Large Language Models
- Fine Tuning
- Catastrophic Forgetting
- NLP
author_profile: false
classes: wide
seo_title: 'Domain-Adaptive Pretraining Comes Before Instruction Tuning'
seo_description: 'A technical guide to continued and domain-adaptive pretraining for LLMs: corpus design, replay mixing, forgetting, evaluation and the transition to SFT.'
seo_type: article
excerpt: >-
  Feeding domain documents into supervised fine-tuning is not the same as teaching
  a language model the statistical structure of a domain. Continued pretraining
  keeps the next-token objective and adapts representations before instruction
  tuning begins.
summary: >-
  A mathematical and practical treatment of continued pretraining and domain-adaptive
  pretraining for LLMs. The article separates CPT from SFT, covers corpus construction,
  replay mixing, catastrophic forgetting, tokenizer and sequence-length issues,
  checkpoint selection, perplexity, downstream evaluation and the hand-off to
  instruction tuning.
keywords:
- continued pretraining
- domain adaptive pretraining
- DAPT
- task adaptive pretraining
- catastrophic forgetting
- LLM domain adaptation
- causal language modeling
why_this_exists: >-
  Domain adaptation is often reduced to supervised fine-tuning on domain examples.
  That confuses two objectives. Continued pretraining adapts the model to the token
  distribution of a domain using next-token prediction, while SFT teaches response
  behaviour from demonstrations.
evidence: >-
  Domain-adaptive and task-adaptive pretraining results from Gururangan et al.,
  later continual-pretraining research on forgetting and replay, and current
  causal-language-model training practice.
methodology: >-
  Treat continued pretraining as distribution adaptation under a language-model
  objective. Compare the base model, domain-only continued pretraining and a
  domain-plus-general replay mixture on both domain and general held-out sets,
  then evaluate downstream tasks before applying SFT.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-library.jpg
  og_image: /assets/images/headers/photo-library.jpg
  overlay_image: /assets/images/headers/photo-library.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-library.jpg
  twitter_image: /assets/images/headers/photo-library.jpg
---

A company has ten million tokens of legal contracts, or scientific papers, or maintenance logs, or financial filings, and wants its language model to become more competent in that domain.

The first instinct is often to convert the documents into question-answer pairs and run supervised fine-tuning.

That can be useful when the desired change is behavioural. It is not the same as adapting the model to the statistical structure of the domain.

Continued pretraining keeps the language-model objective:

$$
\mathcal L_{\mathrm{LM}}(\theta)
=
-
\sum_{t=1}^{T}
\log
p_\theta(w_t\mid w_{<t}).
$$

The model is still learning to predict the next token. What changes is the distribution of text on which that objective is optimized.

If the original pretraining distribution is

$$
P_0(x),
$$

and the domain corpus follows

$$
P_D(x),
$$

continued pretraining moves the model from parameters

$$
\theta_0
$$

towards parameters better adapted to

$$
P_D.
$$

This is a different statistical problem from supervised fine-tuning.

![Continued pretraining balances domain adaptation against forgetting](/assets/images/articles/machine-learning/continued-pretraining-domain-general.svg)

A useful distinction is therefore:

$$
\boxed{
\text{continued pretraining changes domain representation}
\qquad
\text{SFT changes response behaviour}
}
$$

The two stages can be complementary, but they should not be confused.

## Domain-adaptive pretraining is still language modelling

Suppose a domain corpus contains token sequences

$$
x^{(1)},\ldots,x^{(n)}.
$$

Continued pretraining maximizes

$$
\sum_{i=1}^{n}
\log p_\theta(x^{(i)}),
$$

or equivalently minimizes token-level cross-entropy.

There are no required instruction-response pairs.

A legal corpus can remain legal prose. A biomedical corpus can remain papers and abstracts. A manufacturing corpus can remain maintenance reports, work orders and technical manuals.

This is useful when the model's weakness is not that it ignores a requested output format, but that the domain distribution itself is poorly represented.

Typical symptoms include:

- poor handling of specialist terminology,
- weak modelling of domain-specific syntax,
- poor continuation of technical text,
- unstable use of abbreviations,
- weak performance across several downstream tasks from the same field.

These are plausible representation problems.

## SFT optimizes a different conditional distribution

In SFT, the data usually look like

$$
(x_i,y_i),
$$

where $x_i$ is an instruction or prompt and $y_i$ a desired response.

The objective is approximately

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=
-
\sum_i
\log
p_\theta(y_i\mid x_i).
$$

The model learns how to respond to a class of inputs.

Continued pretraining instead optimizes the probability of the domain text itself.

That distinction explains why converting every domain document into synthetic Q&A can be wasteful. The synthetic transformation throws away part of the original distribution and adds the biases of the model that generated the questions.

If the goal is broad domain adaptation, the raw domain corpus may be the better object for the first stage.

## The historical result is simple but important

Gururangan et al. showed that continued pretraining on domain text could improve downstream task performance after broad pretraining, and that an additional task-adaptive stage could provide further gains in some settings.

The result matters because it demonstrates that broad pretraining is not necessarily the end of representation learning.

A generic model may know language well while still allocating insufficient modelling capacity to the local distribution of a specific domain.

The useful lesson is not that every domain needs DAPT.

It is that domain adaptation should be tested before forcing the problem into an instruction-tuning format.

## The corpus defines what the model will adapt to

A domain corpus is not automatically a good domain corpus.

Suppose the target deployment distribution is

$$
P_{\mathrm{deploy}}(x),
$$

while the collected corpus follows

$$
P_{\mathrm{corpus}}(x).
$$

If these distributions differ substantially, the model adapts to the wrong domain.

A finance model trained mostly on investor-relations prose may become fluent in annual reports while remaining weak on internal accounting notes.

A medical model trained mainly on journal abstracts may not adapt well to clinical notes.

A manufacturing model trained on equipment manuals may still struggle with terse operator logs.

Before training, characterize:

- document sources,
- time periods,
- jurisdictions,
- authorship,
- document types,
- language varieties,
- duplicated documents,
- templated boilerplate,
- formatting artefacts,
- machine-generated text.

Domain adaptation begins with sampling.

## Deduplication matters more than token count

If one boilerplate clause appears in 100,000 contracts, naive token counting treats it as 100,000 observations.

The effective information is much smaller.

Near-duplicate documents can dominate the training signal and make the model appear highly adapted because validation contains the same repeated structures.

Deduplication should therefore operate at several levels:

- exact document duplicates,
- near-duplicate documents,
- repeated headers and footers,
- boilerplate paragraphs,
- templated tables,
- mirrored sources.

A larger raw token count is not automatically a richer training set.

## Temporal leakage also exists in continued pretraining

If the model will later be evaluated on historical forecasting, event prediction or dated benchmarks, continued pretraining can leak future information.

Suppose downstream evaluation is intended to simulate a model available at time

$$
t_0.
$$

Then the domain corpus used for pretraining should usually satisfy

$$
\text{date}(d)\leq t_0.
$$

Otherwise the adapted model may contain information unavailable at the simulated decision time.

The fact that pretraining is unsupervised does not make leakage irrelevant.

## Sequence construction changes the objective seen by the optimizer

Documents must be tokenized and packed into training sequences.

If documents are concatenated indiscriminately, the model can see artificial transitions between unrelated sources.

If every document is padded separately, hardware efficiency can collapse.

A practical pipeline normally balances semantic boundaries and efficient packing.

For each document, preserve explicit end-of-document boundaries.

If several shorter texts are packed into one sequence, the tokenizer should include the model's intended separator or end-of-sequence token.

Sequence construction is part of the data-generating process.

## Context length should follow the domain

If the domain contains long legal clauses, multi-page scientific arguments or long maintenance histories, a very short training context may erase important structure.

Let the training context length be

$$
L.
$$

Before choosing it, inspect the token-length distribution:

$$
Q_{0.50},
Q_{0.90},
Q_{0.95},
Q_{0.99}.
$$

If almost every document is shorter than 2,000 tokens, training at 16,000 tokens may waste memory.

If domain dependencies routinely span 8,000 tokens, a 1,024-token context imposes a structural limitation.

The correct context length is a modelling and resource trade-off.

## Continued pretraining can cause catastrophic forgetting

Domain adaptation is not free.

Optimizing only on

$$
P_D
$$

can reduce performance on the original broader distribution

$$
P_0.
$$

This is catastrophic forgetting.

The model becomes better at the target corpus and worse elsewhere.

Let

$$
S_D(\theta)
$$

denote a domain score and

$$
S_G(\theta)
$$

a general-domain score.

A domain-only update may produce

$$
S_D(\theta_1)>S_D(\theta_0)
$$

while also producing

$$
S_G(\theta_1)<S_G(\theta_0).
$$

Calling the training successful after observing only the first inequality is incomplete.

## General replay is a simple control against forgetting

One practical strategy is to mix domain text with a sample of general text.

Define

$$
P_\lambda
=
\lambda P_D
+
(1-\lambda)P_G,
$$

where

$$
0\leq\lambda\leq1.
$$

When

$$
\lambda=1,
$$

training is domain only.

When

$$
\lambda<1,
$$

general replay remains in the stream.

The mixture weight is a hyperparameter.

A useful experiment might compare

$$
\lambda\in\{1.0,0.9,0.7\}.
$$

The objective is not to maximize domain specialization at any price.

It is to locate the trade-off between domain gain and general forgetting.

## The base model is one of the experimental arms

A sensible continued-pretraining study compares at least:

$$
M_0=\text{base model},
$$

$$
M_1=\text{domain-only CPT},
$$

$$
M_2=\text{domain + general replay CPT}.
$$

All three should be evaluated on the same held-out sets.

This design reveals whether:

- domain adaptation helps,
- replay protects general capability,
- the extra training is unnecessary.

Without the base arm, one cannot tell whether continued pretraining improved anything.

## Perplexity is useful but not sufficient

For a held-out domain sequence

$$
w_1,\ldots,w_T,
$$

the average cross-entropy is

$$
H
=
-
\frac{1}{T}
\sum_{t=1}^{T}
\log p_\theta(w_t\mid w_{<t}).
$$

Perplexity is

$$
\mathrm{PPL}
=
e^H.
$$

Lower domain perplexity means the model assigns higher probability to held-out domain text.

That is relevant evidence of domain adaptation.

It does not prove downstream usefulness.

A model can become much better at predicting legal boilerplate without improving legal reasoning.

Perplexity should therefore be paired with downstream evaluations.

## Evaluate domain and general perplexity together

A useful diagnostic table is:

| Model | Domain PPL | General PPL |
| --- | ---: | ---: |
| Base | $P_D^{(0)}$ | $P_G^{(0)}$ |
| Domain CPT | $P_D^{(1)}$ | $P_G^{(1)}$ |
| Mixed CPT | $P_D^{(2)}$ | $P_G^{(2)}$ |

The desired pattern depends on the deployment objective.

A large reduction in domain perplexity accompanied by a catastrophic increase in general perplexity may be unacceptable for a general assistant.

For a narrow offline specialist model, that trade-off may be acceptable.

The operating context determines the loss function.

## Downstream evaluation should precede SFT

If continued pretraining is immediately followed by supervised fine-tuning and only the final SFT model is evaluated, one cannot tell which stage caused the improvement.

Evaluate after CPT and before SFT.

The sequence should be:

$$
M_0
\rightarrow
M_{\mathrm{CPT}}
\rightarrow
M_{\mathrm{CPT+SFT}}.
$$

Measure after each arrow.

This lets you estimate:

$$
\Delta_{\mathrm{CPT}}
=
S(M_{\mathrm{CPT}})
-
S(M_0),
$$

and

$$
\Delta_{\mathrm{SFT}}
=
S(M_{\mathrm{CPT+SFT}})
-
S(M_{\mathrm{CPT}}).
$$

Without intermediate evaluation, attribution is impossible.

## Continued pretraining can be full-parameter or parameter efficient

Most classical DAPT work updated the full model.

For modern LLMs, full continued pretraining may be expensive.

Parameter-efficient approaches can be used, but the interpretation changes.

If only LoRA adapters are trained during a language-model objective, the model is not undergoing unrestricted domain adaptation. The domain update is constrained to the adapter subspace.

That can still work.

It should be described accurately.

The experiment then tests:

> Is low-rank adaptation sufficient to model this domain shift?

rather than

> Has the whole model been domain-adapted?

## Learning rates should usually be conservative

Continued pretraining starts from an already useful model.

The objective is adaptation, not learning language from scratch.

Large learning rates can erase useful structure rapidly.

A practical sweep should generally begin at smaller values than one might use for pretraining from initialization.

The exact range depends on:

- model size,
- optimizer,
- dataset size,
- full versus parameter-efficient training,
- batch size,
- context length.

There is no universal continued-pretraining learning rate.

Monitor both domain gain and general forgetting.

## Token counts should be reported with optimizer steps

"One epoch" is an ambiguous quantity across corpora.

A domain corpus containing 50 million tokens and one containing 20 billion tokens are not comparable simply because both were trained for one epoch.

Report:

- total training tokens,
- tokens from domain data,
- tokens from general replay,
- optimizer steps,
- effective token batch size.

If the domain mixture is

$$
\lambda,
$$

then the expected domain-token count is approximately

$$
T_D
=
\lambda T.
$$

The training budget should be visible.

## Validation should include source-held-out documents

A random token split from the same documents is too easy.

The validation corpus should preferably contain documents not used in training.

If a company has repeated versions of the same manual, all near-versions should remain in the same split.

The goal is to test whether the model generalized to new domain text, not whether it memorized adjacent chunks.

## Memorization should be tested explicitly

Domain corpora can contain sensitive or copyrighted material.

Continued pretraining can increase memorization risk.

A simple audit can probe rare sequences and unique strings from training data.

If the model reproduces long verbatim spans at unexpectedly high rates, the adaptation process may have overfit.

The risk is higher for:

- small corpora,
- repeated documents,
- high numbers of epochs,
- rare unique strings,
- low-entropy text.

Deduplication and conservative training reduce but do not eliminate this problem.

## Tokenizer mismatch can limit domain adaptation

The tokenizer remains fixed in most continued-pretraining workflows.

If domain terminology is fragmented into many subword pieces, the model can still learn the domain, but efficiency may suffer.

For a domain term $s$, let

$$
\tau(s)
$$

be the number of tokens used to represent it.

Inspect the distribution of

$$
\tau(s)
$$

for important vocabulary.

A biomedical or chemical corpus with extreme fragmentation may motivate tokenizer changes, but changing the tokenizer also changes the embedding matrix and complicates compatibility.

Tokenizer extension is therefore a separate architectural intervention, not a harmless preprocessing tweak.

## Vocabulary extension should be justified by repeated evidence

Adding new tokens is most useful when important strings occur frequently and are represented inefficiently.

If a new token is added for every rare identifier, the vocabulary becomes bloated.

A useful criterion considers both frequency and fragmentation.

For domain string $s$, define approximately

$$
G(s)
=
f(s)
[
\tau(s)-1
],
$$

where $f(s)$ is corpus frequency.

Large values indicate repeated tokenization overhead.

This is only a heuristic, but it is more principled than adding vocabulary because words "look technical."

## Continued pretraining does not replace retrieval

Suppose regulations change every month.

CPT can improve the model's understanding of regulatory language.

It should not be used as the primary mechanism for keeping the exact regulation text current.

The taxonomy remains:

$$
\text{stable domain distribution}
\Rightarrow
\text{continued pretraining},
$$

$$
\text{response behaviour}
\Rightarrow
\text{SFT},
$$

$$
\text{volatile factual knowledge}
\Rightarrow
\text{RAG}.
$$

A strong domain system may use all three.

## A realistic pipeline separates representation, behaviour and knowledge

One coherent architecture is:

$$
\text{base model}
\rightarrow
\text{domain CPT}
\rightarrow
\text{SFT}
\rightarrow
\text{DPO}
$$

with retrieval remaining external at inference time.

Each stage solves a different problem.

Domain CPT adapts the language representation.

SFT teaches task demonstrations.

DPO changes relative preferences.

RAG supplies current external information.

This decomposition is far more useful than calling all four stages "fine-tuning."

## A practical causal-language-model training skeleton

The implementation itself is ordinary causal language modelling.

A simplified workflow with Transformers might look like:

```python
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


model_id = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


dataset = load_dataset(
    "text",
    data_files={
        "train": "domain_train.txt",
        "validation": "domain_validation.txt",
    },
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=False,
        add_special_tokens=True,
    )


tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)


block_size = 2048


def group_texts(batch):
    concatenated = sum(batch["input_ids"], [])
    usable = (len(concatenated) // block_size) * block_size

    chunks = [
        concatenated[i : i + block_size]
        for i in range(0, usable, block_size)
    ]

    return {
        "input_ids": chunks,
        "attention_mask": [
            [1] * block_size
            for _ in chunks
        ],
    }


lm_dataset = tokenized.map(
    group_texts,
    batched=True,
)


collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


args = TrainingArguments(
    output_dir="outputs/domain-cpt",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=20,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=collator,
)


trainer.train()
```

This is only a skeleton.

Production code should preserve explicit document boundaries rather than indiscriminately concatenating unrelated texts.

It should also pin package versions, log corpus hashes and record the exact base-model revision.

## General replay should be implemented as a data-mixture experiment

If the main risk is forgetting, create separate domain and general datasets.

Then construct training mixtures such as:

$$
90\%\text{ domain}
+
10\%\text{ general},
$$

and

$$
70\%\text{ domain}
+
30\%\text{ general}.
$$

Do not mix the corpora once and lose the provenance.

Track source labels so that effective token counts remain measurable.

A mixture without source accounting is difficult to reproduce.

## Checkpoint selection must be multi-objective

Choosing the checkpoint with the lowest domain validation loss can select the point with the most forgetting.

Suppose checkpoint $t$ has domain loss

$$
L_D(t)
$$

and general loss

$$
L_G(t).
$$

A release rule may instead optimize

$$
J(t)
=
L_D(t)
+
\gamma
[
L_G(t)-L_G(0)
]_+,
$$

where the second term penalizes degradation relative to the base model.

The exact form is application-specific.

The principle is not.

If forgetting matters, checkpoint selection must measure it.

## The best domain model may not be the most specialized model

The optimum depends on the deployment boundary.

A model dedicated exclusively to radiology reports can tolerate more specialization than a general assistant that sometimes answers radiology questions.

This can be expressed as a utility:

$$
U(\theta)
=
\omega_D S_D(\theta)
+
\omega_G S_G(\theta),
$$

where

$$
\omega_D+\omega_G=1.
$$

A narrow specialist has large

$$
\omega_D.
$$

A general assistant has substantial

$$
\omega_G.
$$

The correct amount of domain adaptation depends on those weights.

## Continued pretraining should earn its complexity

CPT adds:

- corpus curation,
- extra compute,
- new checkpoints,
- forgetting risk,
- memorization risk,
- another model-selection stage.

It should therefore be compared with cheaper alternatives.

A strong prompt, RAG, terminology injection or narrow SFT may already solve the deployment problem.

The correct question is not

> Can we continue pretraining this model?

It is

> Does continued pretraining improve held-out domain behaviour enough to justify its cost and its regressions?

That is an empirical question.

## References

Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *Proceedings of ACL 2020*, 8342–8360.

Hugging Face. *Transformers documentation: Causal language modeling*. Accessed 21 September 2026.

Ke, Z., Lin, Y., Huang, Y., et al. (2023). Continual Pre-training of Language Models. *arXiv preprint*.

Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DOREMI: Optimizing Data Mixtures Speeds Up Language Model Pretraining. *Advances in Neural Information Processing Systems*, 36.
