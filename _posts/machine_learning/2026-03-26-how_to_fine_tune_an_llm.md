---
permalink: '/machine-learning/how_to_fine_tune_an_llm/'
title: 'How to Fine-Tune an LLM Without Fooling Yourself'
date: '2026-03-26'
categories:
- Machine Learning
tags:
- Large Language Models
- Fine Tuning
- LoRA
- QLoRA
- PEFT
- Supervised Fine Tuning
- NLP
author_profile: false
classes: wide
seo_title: 'How to Fine-Tune an LLM Without Fooling Yourself'
seo_description: 'A practical guide to supervised LLM fine-tuning with LoRA and QLoRA: data design, chat templates, loss masking, training, evaluation, checkpoint selection and deployment.'
seo_type: article
excerpt: >-
  Fine-tuning an LLM is easy to start and surprisingly easy to do badly. The
  difficult parts are defining the behaviour to change, constructing the dataset,
  preventing leakage, formatting tokens correctly, and evaluating behaviour rather
  than celebrating a falling training loss.
summary: >-
  A practical end-to-end guide to supervised fine-tuning of chat LLMs using
  Hugging Face Transformers, TRL and PEFT. It covers dataset design, conversational
  formatting, train-validation-test splitting, QLoRA configuration, LoRA rank and
  target modules, assistant-token loss masking, effective batch size, learning-rate
  selection, checkpointing, behavioural evaluation, regression testing and adapter
  deployment.
keywords:
- how to fine tune LLM
- LLM fine tuning
- LoRA fine tuning
- QLoRA
- PEFT
- SFTTrainer
- supervised fine tuning
- chat templates
- Hugging Face TRL
why_this_exists: >-
  Many LLM fine-tuning tutorials begin with a training script and end when loss
  decreases. That skips the decisions that determine whether the adapted model is
  actually better: target behaviour, data-generating process, split integrity,
  token formatting, loss masking, regression evaluation and deployment criteria.
evidence: >-
  Foundational LoRA and QLoRA papers, current Hugging Face Transformers chat
  templating, PEFT and TRL supervised fine-tuning interfaces, and standard
  statistical principles for train-validation-test separation and model selection.
methodology: >-
  Treat fine-tuning as an experiment on model behaviour. Define the deployment
  target first, construct and audit examples, preserve the model's chat protocol,
  train a parameter-efficient adapter, and evaluate both target improvements and
  regressions on genuinely held-out prompts.
reviewed_at: '2026-09-20'
header:
  image: /assets/images/headers/photo-supercomputer.jpg
  og_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_image: /assets/images/headers/photo-supercomputer.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-supercomputer.jpg
  twitter_image: /assets/images/headers/photo-supercomputer.jpg
---

Fine-tuning a large language model is technically straightforward. A few libraries can load a pretrained model, attach LoRA adapters, quantize the base weights, run supervised training and save the result in a handful of lines.

That convenience creates a problem. It makes it easy to confuse **successfully running a training loop** with **successfully improving a model**.

The training code is rarely the hardest part. The difficult decisions come earlier and later: what behaviour should change, which examples represent that behaviour, how should near-duplicates be split, which tokens should contribute to the loss, whether the chat template matches the base model, what constitutes a regression, and how the resulting adapter will be evaluated on the deployment distribution.

A defensible workflow is therefore

$$
\text{objective}
\rightarrow
\text{dataset}
\rightarrow
\text{split}
\rightarrow
\text{format}
\rightarrow
\text{train}
\rightarrow
\text{evaluate}
\rightarrow
\text{deploy}.
$$

![A defensible LLM fine-tuning workflow](/assets/images/articles/machine-learning/llm-fine-tuning-workflow.svg)

The central principle is:

$$
\boxed{
\text{Fine-tuning is an experiment on model behaviour, not a GPU ritual.}
}
$$

This article develops that workflow using supervised fine-tuning with LoRA or QLoRA because it is currently one of the most accessible ways to adapt an open-weight chat model. The same experimental logic applies to full fine-tuning and preference optimization.

## Start by defining the behaviour that must change

Do not begin with the dataset.

Begin with the failure.

Suppose a base model is given the correct information but repeatedly:

- produces invalid JSON,
- omits mandatory fields,
- uses the wrong terminology,
- fails to follow a domain-specific procedure,
- responds in an inappropriate style,
- generates tool arguments incorrectly,
- or performs poorly on a narrow recurring task.

These are plausible fine-tuning targets because the problem is behavioural.

A useful target is specific enough to be measured. Instead of

> make the model better at finance,

define something like

> given a Portuguese annual report excerpt, extract the requested accounting quantities into a fixed JSON schema and return no commentary outside the schema.

Now the evaluation object is observable.

Let $B$ denote the target behaviour and $X$ the deployment prompt distribution. The quantity of interest is not training loss. It is something closer to

$$
\mathbb E_{x\sim P_{\mathrm{deploy}}}
[
U(B(f_\theta(x)))
],
$$

where $U$ is a task-specific utility or score.

Training is useful only to the extent that it improves this deployment quantity.

## Test the base model before training anything

The base model is the first baseline.

Run it on a fixed set of representative prompts using the intended inference configuration. Record:

- task accuracy,
- schema validity,
- exact-match fields where appropriate,
- style violations,
- unsupported claims,
- refusal or safety behaviour if relevant,
- latency and token use.

Keep these prompts untouched by training.

This baseline matters for two reasons. First, some apparent fine-tuning problems can be solved with better instructions or context. Second, without a base score there is no way to know whether the adapter helped.

A reasonable experiment compares

$$
\text{base}
\qquad\text{vs}\qquad
\text{adapted}.
$$

If retrieval is also involved, compare

$$
\text{base},
\quad
\text{base+RAG},
\quad
\text{adapted},
\quad
\text{adapted+RAG}.
$$

Do not move the benchmark after seeing the results.

## Build examples from the deployment task

Supervised fine-tuning typically starts from demonstrations

$$
\mathcal D
=
\{(x_i,y_i)\}_{i=1}^{n},
$$

where $x_i$ is an input and $y_i$ is the desired response.

For chat models, the data are better represented as conversations. A minimal example is:

```python
example = {
    "messages": [
        {
            "role": "system",
            "content": "Return valid JSON matching the requested schema."
        },
        {
            "role": "user",
            "content": "Extract revenue and operating profit from this passage: ..."
        },
        {
            "role": "assistant",
            "content": '{"revenue_eur": 1200000, "operating_profit_eur": 180000}'
        }
    ]
}
```

The examples should represent the distribution the model will see after deployment.

That sounds obvious, but many datasets are assembled from whatever is easiest to collect. The resulting training distribution

$$
P_{\mathrm{train}}(x,y)
$$

can differ sharply from

$$
P_{\mathrm{deploy}}(x,y).
$$

Fine-tuning then optimizes the wrong population.

If production prompts are short, noisy and ambiguous while training examples are polished synthetic instructions, the model is being trained on a different task.

## Data quality matters more than dataset size slogans

There is no universal number of examples required for fine-tuning.

A narrow deterministic formatting task can improve with relatively few high-quality examples. A broad behavioural shift may require much more variation. The required sample size depends on task diversity, base-model capability, label noise, model size and how far the target behaviour is from the pretrained distribution.

A thousand near-duplicate examples do not provide the same information as a thousand genuinely distinct task situations.

Before training, audit:

- duplicate and near-duplicate prompts,
- contradictory targets,
- malformed outputs,
- impossible examples,
- response-length distribution,
- topic distribution,
- source distribution,
- class or task imbalance,
- synthetic-data artefacts,
- accidental test contamination.

If the same underlying document produces ten lightly paraphrased prompts, those examples should normally remain in the same data split. Otherwise the validation set measures recognition of the source rather than generalization.

## Split by the unit that can leak

Random row splitting is often wrong for LLM datasets.

Suppose several prompts are generated from the same source document. If some enter training and others enter validation, the model may see almost the same content on both sides.

The split unit should follow the dependency structure.

Possible grouping variables include:

- source document,
- customer,
- conversation,
- patient,
- repository,
- product,
- problem template,
- time period.

Let $g(i)$ denote the group associated with example $i$. A leakage-safe split requires

$$
g(i)\in G_{\mathrm{train}}
\implies
g(i)\notin
G_{\mathrm{validation}}
\cup
G_{\mathrm{test}}.
$$

This matters more than whether the split percentages are exactly 80/10/10.

A good workflow keeps three roles distinct:

- **training set** for optimization,
- **validation set** for model selection and hyperparameters,
- **test set** for the final unbiased comparison.

If the test set is inspected repeatedly during development, it becomes another validation set.

## Preserve the model's chat template

Chat models are not trained on abstract roles. They are trained on token sequences with model-specific control tokens.

Two models can represent the same conversation using different templates. One may use tokens resembling

```text
<|user|>
...
<|assistant|>
...
```

while another uses a different set of delimiters.

These tokens are part of the learned protocol.

Hugging Face exposes the model's template through

```python
tokenizer.apply_chat_template(...)
```

and TRL can consume conversational datasets directly. That is safer than inventing a custom string format unless there is a deliberate reason to retrain the protocol.

A useful sanity check is:

```python
rendered = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print(rendered)
```

Inspect the actual text and special tokens before launching training.

A formatting bug repeated across the dataset becomes a training signal.

## Decide which tokens should contribute to the loss

For standard causal language modelling, the token-level objective is

$$
\mathcal L(\theta)
=
-
\sum_{t=1}^{T}
m_t
\log
p_\theta(w_t\mid w_{<t}),
$$

where $m_t\in\{0,1\}$ is a loss mask.

The mask determines which tokens matter.

For instruction tuning, one common choice is to compute loss only on assistant responses. User and system tokens remain in the conditioning context but do not contribute directly to the target loss.

Conceptually,

$$
m_t
=
\begin{cases}
1, & t\text{ belongs to an assistant response},\\
0, & \text{otherwise}.
\end{cases}
$$

This prevents the optimizer from spending capacity learning to reproduce user prompts.

Current TRL supports assistant-only loss for compatible conversational templates. The important qualification is **compatible**. The chat template must expose the assistant-token mask correctly.

Do not trust a configuration flag blindly.

Inspect a tokenized example and its labels. Tokens excluded from loss should normally carry the ignore index, conventionally $-100$.

If the masking is wrong, the training objective is not the objective you think you are optimizing.

## LoRA changes the parameterization, not the learning objective

Suppose a pretrained linear layer has weight matrix

$$
W_0\in\mathbb R^{d\times k}.
$$

Full fine-tuning learns an unrestricted update

$$
\Delta W\in\mathbb R^{d\times k}.
$$

LoRA instead represents

$$
\Delta W=BA,
$$

where

$$
B\in\mathbb R^{d\times r},
\qquad
A\in\mathbb R^{r\times k},
\qquad
r\ll\min(d,k).
$$

The effective adapted layer is

$$
W
=
W_0
+
\frac{\alpha}{r}BA,
$$

under the common scaling convention.

The number of trainable parameters for the update falls from approximately

$$
dk
$$

to

$$
r(d+k).
$$

The base weights remain frozen.

The rank $r$ is therefore a capacity parameter. A larger rank gives the adapter more freedom but increases trainable parameters and memory. There is no universal best rank.

Values such as 8, 16, 32 or 64 are reasonable experimental candidates, not laws of nature.

## QLoRA makes the frozen base cheaper to hold in memory

QLoRA keeps the base model quantized while training LoRA adapters.

A common configuration uses 4-bit NormalFloat (NF4) quantization:

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

NF4 was introduced in the QLoRA work as a datatype designed for approximately normally distributed neural-network weights. Double quantization further reduces memory by quantizing quantization constants.

This can make adaptation of models that would not fit in ordinary training precision feasible on much smaller hardware.

But quantization is a resource strategy, not a modelling argument. If the dataset, loss mask or evaluation is wrong, QLoRA simply trains the wrong thing more economically.

## A current minimal QLoRA configuration

The following is an implementation skeleton rather than a universal recipe.

Install the relevant libraries in a clean environment first:

```bash
python -m pip install -U transformers datasets accelerate peft trl bitsandbytes
```

Pin the resolved versions in the project once the experiment is reproducible. Library APIs move quickly enough that an unpinned fine-tuning notebook is not a durable experiment.

The example assumes:

- a causal chat model,
- a conversational dataset with a `messages` column,
- a GPU that supports the selected compute dtype,
- current versions of Transformers, TRL, PEFT, Datasets and bitsandbytes.

```python
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


model_id = "your-model-id"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)


dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "validation.jsonl",
    },
)


training_args = SFTConfig(
    output_dir="outputs/my-llm-sft",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,
    packing=False,
    assistant_only_loss=True,
    report_to="none",
)


trainer = SFTTrainer(
    model=model_id,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
    quantization_config=quantization_config,
)

trainer.model.print_trainable_parameters()


trainer.train()
trainer.save_model("outputs/my-llm-sft/best-adapter")
tokenizer.save_pretrained("outputs/my-llm-sft/best-adapter")
```

The code is short.

The experiment is not.

Every prominent number in that configuration should be treated as a starting hypothesis:

$$
r=16,
\quad
\alpha=32,
\quad
\eta=2\times10^{-4},
\quad
L=2048.
$$

They require validation for the actual task.

## `target_modules="all-linear"` is useful, but understand what it means

LoRA adapters must be attached to modules.

Hard-coding names such as

```python
["q_proj", "v_proj"]
```

works for some architectures and fails for others.

Current PEFT supports targeting all linear layers through

```python
target_modules="all-linear"
```

which is convenient for QLoRA-style adaptation and avoids architecture-specific module-name lists.

That convenience increases adapter capacity and trainable parameter count relative to adapting only a small subset of projections.

Again, it is an experimental choice.

After constructing the trainer, inspect the number of trainable parameters rather than assuming the configuration is small:

```python
trainer.model.print_trainable_parameters()
```

At that point TRL has wrapped the quantized base model with the PEFT adapter, so the reported count corresponds to the parameters that will actually be optimized.

The ratio

$$
\rho
=
\frac{
\text{trainable parameters}
}{
\text{total parameters}
}
$$

should be known and reported.

## Effective batch size is not the per-device batch size

If

- $b$ is the per-device batch size,
- $g$ is gradient accumulation,
- $n$ is the number of data-parallel devices,

then the approximate effective batch size in sequences is

$$
B_{\mathrm{eff}}
=
bgn.
$$

For

$$
b=2,
\qquad
g=8,
\qquad
n=1,
$$

the optimizer updates after roughly

$$
B_{\mathrm{eff}}=16
$$

sequences.

Token counts can vary substantially across sequences, so even this is only part of the picture.

If one batch contains many long examples and another many short examples, the number of supervised tokens per optimizer step changes.

For that reason, sequence-length distributions should be inspected before choosing batching parameters.

## Truncation can silently delete the answer

Suppose the maximum sequence length is

$$
L_{\max}=2048.
$$

If a formatted example exceeds that length, truncation occurs.

What gets truncated depends on the preprocessing and tokenizer configuration. If the assistant response is at the end of the conversation, a badly designed truncation policy can remove the very target the model is supposed to learn.

Before training, inspect:

- median length,
- 90th percentile,
- 95th percentile,
- 99th percentile,
- maximum length,
- fraction exceeding $L_{\max}$.

For example:

```python
import numpy as np

lengths = []

for row in dataset["train"]:
    ids = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )
    lengths.append(len(ids))

for q in [0.50, 0.90, 0.95, 0.99]:
    print(q, int(np.quantile(lengths, q)))

print("max", max(lengths))
print(
    "fraction > 2048",
    np.mean(np.asarray(lengths) > 2048),
)
```

Choosing `max_length` without looking at this distribution is guesswork.

## Packing changes efficiency, not the statistical population

If examples are short, sequence packing can place several examples into one model-length sequence to reduce padding waste.

This improves hardware utilization.

It does not create new data.

Packing should therefore be viewed as a computational optimization rather than a modelling technique. It can complicate debugging because several examples share one packed sequence, so start without it until formatting and loss masks are verified.

Then benchmark whether

```python
packing=True
```

actually improves throughput for the observed length distribution.

## Learning rate should be tuned for the adaptation regime

LoRA often tolerates learning rates larger than those used for full fine-tuning because only a small adapter parameter set is being optimized.

That does not justify copying one value from a tutorial.

A small sweep might compare

$$
5\times10^{-5},
\quad
10^{-4},
\quad
2\times10^{-4},
\quad
5\times10^{-4}.
$$

The correct range depends on model, adapter configuration, dataset size and objective.

Monitor both training and validation behaviour.

If training loss falls while validation loss rises, the adapter is fitting the training set more aggressively than it generalizes.

But validation loss is still not the final task metric.

## Loss is necessary and insufficient

For token-level SFT, lower validation loss means the model assigns greater probability to held-out target tokens under the validation distribution.

That is useful.

It does not automatically mean:

- JSON validity improved,
- factual accuracy improved,
- instruction following improved,
- hallucination decreased,
- the desired style improved,
- unrelated capabilities were preserved.

The relationship

$$
\Delta \mathrm{eval\_loss}<0
$$

does not imply

$$
\Delta \mathrm{deployment\ utility}>0.
$$

Generated-output evaluation is therefore essential.

## Build behavioural tests before training

For a schema task, define exact validators.

For example:

```python
import json


def valid_output(text: str) -> bool:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False

    required = {
        "revenue_eur",
        "operating_profit_eur",
    }

    return set(obj) == required
```

Then score the held-out test set:

$$
\widehat p
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1
\{
\text{output}_i\text{ is valid}
\}.
$$

If numerical extraction matters, evaluate field-level correctness as well.

For natural-language tasks, define a rubric before training. Human evaluation can remain necessary, but the dimensions should be specified in advance.

Examples include:

- factual correctness,
- completeness,
- unsupported claims,
- style adherence,
- citation support,
- refusal correctness.

Do not invent the rubric after seeing which model looks best.

## Evaluate regressions, not only target improvements

Fine-tuning can improve the target task and damage another.

Create a regression suite containing prompts the base model already handles well.

Then compare

$$
\Delta_j
=
S_j(\text{adapted})
-
S_j(\text{base})
$$

across task families $j$.

A useful release criterion may require

$$
\Delta_{\mathrm{target}}>\delta
$$

for a meaningful target improvement while also enforcing

$$
\Delta_j>-\epsilon_j
$$

for protected behaviours.

This makes the trade-off explicit.

Without regression testing, the fine-tuned model is being judged only where it was expected to improve.

## Compare several random seeds when the dataset is small

Small fine-tuning datasets can produce unstable results.

Different data orders, initialization of adapter parameters and minibatch composition can yield different checkpoints.

If the decision is important, run several seeds:

$$
s\in\{11,29,47,83,101\}.
$$

Report the distribution of the task metric rather than one lucky run.

If model A scores 0.82 and model B 0.83 on one fine-tuning run, there may be no meaningful evidence that B is better.

This is ordinary experimental uncertainty. LLMs do not make it disappear.

## Checkpoint selection is model selection

If validation is performed every 100 steps and the best checkpoint is selected from 30 evaluations, then the final checkpoint was selected after 30 opportunities to look good.

This is why the test set must remain untouched.

Using

```python
load_best_model_at_end=True
```

is convenient, but it turns the validation set into a model-selection instrument.

That is exactly what it is supposed to do.

The final test result should come afterward.

## Watch for catastrophic behavioural shortcuts

The adapted model may learn a shortcut that reduces loss but violates the intended task.

Examples include:

- always producing the same JSON skeleton,
- copying spans without reasoning about them,
- refusing whenever a rare term appears,
- producing excessively short responses because training targets were short,
- overusing a phrase that appears in synthetic examples,
- assuming one source format that happened to dominate training.

These failures are easier to detect with stratified evaluation.

Break the test set into meaningful slices:

$$
S
=
\{
S_{\mathrm{short}},
S_{\mathrm{long}},
S_{\mathrm{rare}},
S_{\mathrm{ambiguous}},
S_{\mathrm{out\ of\ domain}}
\}.
$$

Aggregate accuracy can hide a catastrophic subgroup failure.

## Synthetic training data need their own audit

LLMs make it cheap to generate instruction datasets.

That is useful and dangerous.

If a teacher model systematically produces one reasoning pattern, one tone or one misconception, the student can learn it at scale.

Synthetic examples should therefore be treated as generated measurements, not ground truth.

Audit:

- diversity,
- duplication,
- factual correctness,
- template artefacts,
- teacher-specific phrases,
- agreement with human-written examples,
- distributional similarity to deployment prompts.

A large synthetic corpus can have low effective information if most examples are variations of the same template.

## Save adapters separately unless there is a reason to merge

One advantage of LoRA is that the base model can remain shared.

After training, the adapter can be loaded onto the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM


base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/my-llm-sft/best-adapter",
)
```

This keeps storage small and permits several adapters to share one base model.

For some deployment stacks it is convenient to merge the adapter:

```python
merged = model.merge_and_unload()
merged.save_pretrained("outputs/my-llm-sft/merged")
```

Merging changes packaging, not the learned function in principle.

Keep enough metadata to reproduce the artifact:

- base model identifier and revision,
- tokenizer revision,
- package versions,
- LoRA configuration,
- quantization configuration,
- training arguments,
- dataset version or hash,
- split definition,
- random seed,
- selected checkpoint,
- evaluation results.

An adapter without its base-model identity is incomplete.

## Quantization at training and inference are separate decisions

QLoRA uses quantization to reduce the memory required to hold the frozen base during training.

Deployment can use a different representation.

The adapter may be applied to:

- the same quantized base,
- a higher-precision base,
- or a merged model that is subsequently quantized for inference.

Training quantization and inference quantization should therefore be evaluated separately.

Do not assume that because the model trained under one 4-bit configuration, every 4-bit deployment configuration is equivalent.

## Fine-tuning should have a stopping rule

A project can continue indefinitely because another epoch, rank, target-module choice or dataset expansion might improve the score.

Define release criteria before the final comparison.

For example:

$$
\text{schema validity}\geq 0.99,
$$

$$
\text{field accuracy}\geq 0.95,
$$

$$
\text{critical regression rate}\leq 0.01.
$$

The exact thresholds depend on the application.

The important point is that success is defined in deployment terms rather than by training loss.

## A practical end-to-end checklist

Before training, verify that the base model really fails the target task, the desired behaviour is measurable, examples represent deployment, grouping prevents semantic leakage, and the test set is frozen.

Before the first optimizer step, inspect rendered chat examples, token lengths, truncation, loss masks, trainable-parameter counts and at least one complete tokenized example.

During training, monitor training and validation loss, learning rate, gradient behaviour, throughput and checkpoint size. Do not interpret these as the final product metric.

After training, generate responses on held-out prompts using the exact deployment decoding configuration. Compare the adapted model directly with the base model, evaluate target metrics and regressions, inspect failure slices and repeat seeds when uncertainty matters.

Only then decide whether the fine-tune worked.

## A fine-tune is justified by behaviour, not by the existence of an adapter

Modern tooling has made fine-tuning dramatically more accessible. LoRA reduces the dimensionality of the update. QLoRA reduces the memory cost of carrying the frozen base. TRL and PEFT remove much of the training boilerplate.

Those are engineering improvements.

They do not change the statistical problem.

A fine-tuning experiment still asks whether examples drawn from one finite dataset can produce a parameter update that improves behaviour on future prompts from a target population without unacceptable regressions elsewhere.

The strongest fine-tuning workflow therefore looks less like

$$
\text{load model}
\rightarrow
\text{train}
\rightarrow
\text{save}
$$

and more like

$$
\text{define}
\rightarrow
\text{measure}
\rightarrow
\text{train}
\rightarrow
\text{challenge}
\rightarrow
\text{compare}.
$$

If the model cannot be shown to improve on held-out behaviour that matters, then a successful training run is not yet a successful fine-tune.

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*, 36.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*.

Hugging Face. *Transformers documentation: Chat templates*. Accessed 20 September 2026.

Hugging Face. *PEFT documentation: LoRA*. Accessed 20 September 2026.

Hugging Face. *TRL documentation: SFT Trainer*. Accessed 20 September 2026.
