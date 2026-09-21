---
permalink: '/machine-learning/speculative_decoding_does_not_mean_approximate_generation/'
title: 'Speculative Decoding Does Not Mean Approximate Generation'
date: '2026-08-20'
categories:
- Machine Learning
tags:
- Speculative Decoding
- Large Language Models
- Inference
- Latency
- Draft Models
author_profile: false
classes: wide
seo_title: 'Speculative Decoding Does Not Mean Approximate Generation'
seo_description: 'A technical guide to speculative decoding: draft models, exact target-distribution sampling, acceptance probability, expected speedup, batching, KV cache and failure modes.'
seo_type: article
excerpt: >-
  Speculative decoding uses a cheaper model to propose several future tokens and a
  larger target model to verify them in parallel. In the standard algorithm, the
  draft changes latency, not the target model's output distribution.
summary: >-
  A mathematical and engineering treatment of speculative decoding for LLMs,
  covering the draft-target acceptance rule, rejection correction, expected accepted
  tokens, draft quality, draft length, memory bandwidth, batching, KV-cache handling,
  self-speculative decoding and Medusa-style alternatives.
keywords:
- speculative decoding
- speculative sampling
- draft model
- LLM inference
- LLM latency
- assisted generation
- Medusa
why_this_exists: >-
  Speculative decoding is often described as letting a small model write several
  tokens for a large model. That wording makes the method sound approximate. In
  the standard acceptance-rejection formulation, the target distribution is
  preserved exactly; the draft model only proposes work that may be accepted.
evidence: >-
  Foundational speculative-decoding and speculative-sampling papers, Medusa, and
  current implementation practice in high-performance LLM serving stacks.
methodology: >-
  Start from autoregressive sampling, introduce a draft proposal distribution,
  derive the token-level acceptance and correction rules, then connect acceptance
  rate to expected throughput and the systems constraints that determine whether
  speculation is actually beneficial.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-data-center.jpg
  og_image: /assets/images/headers/photo-data-center.jpg
  overlay_image: /assets/images/headers/photo-data-center.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-center.jpg
  twitter_image: /assets/images/headers/photo-data-center.jpg
---

Autoregressive decoding has a structural bottleneck.

To sample token $y_t$, the model conditions on

$$
y_{<t}.
$$

The next token cannot be sampled until the previous token is known. For a large model, generating $K$ tokens therefore requires approximately $K$ sequential decoding steps.

Speculative decoding attacks that serial dependency by introducing a cheaper proposal model.

The draft model proposes several future tokens quickly. The target model then evaluates those positions in parallel and decides which proposals can be retained.

![Speculative decoding drafts several tokens and verifies them with the target model](/assets/images/articles/machine-learning/speculative-decoding-verification.svg)

The important point is often missed:

$$
\boxed{
\text{standard speculative sampling can preserve the target distribution exactly}
}
$$

The small model does not replace the large model.

It proposes work that the large model may accept.

## Ordinary autoregressive decoding is serial

Let the target model define

$$
p(y_t\mid y_{<t},x).
$$

A sample of length $T$ is generated as

$$
y_1\sim p(\cdot\mid x),
$$

$$
y_2\sim p(\cdot\mid x,y_1),
$$

and so on.

The dependency graph is sequential.

Even if one model forward pass is fast, the latency accumulates token by token.

This is particularly expensive during decoding because model weights are repeatedly read for each generated token.

## A draft model proposes several future tokens

Introduce a cheaper model

$$
q.
$$

Starting from the same prefix, let it propose

$$
\tilde y_1,\ldots,\tilde y_\gamma.
$$

These tokens are sampled autoregressively from $q$.

Because $q$ is cheaper, generating $\gamma$ draft tokens can cost less than generating the same number directly from the target model.

The target then evaluates the full draft block.

Transformer parallelism allows the target to compute probabilities for several drafted positions in one verification pass.

The speedup comes from converting several serial target steps into one larger parallel target computation.

## The draft is a proposal distribution

The right analogy is rejection sampling.

For one drafted token $y$, the target probability is

$$
p(y)
$$

and draft probability is

$$
q(y).
$$

The token is accepted with probability

$$
a(y)
=
\min
\left(
1,
\frac{p(y)}{q(y)}
\right).
$$

If the token is accepted, continue to the next drafted position.

If it is rejected, speculative sampling draws a correction from a residual distribution constructed so that the final emitted token still follows $p$.

This is why the algorithm can be exact.

## The correction distribution repairs proposal bias

When a proposal is rejected, simply drawing again from $p$ would not generally produce the correct joint sampling process under the speculative procedure.

The residual distribution is proportional to

$$
[p(y)-q(y)]_+,
$$

where

$$
[a]_+
=
\max(a,0).
$$

After normalization,

$$
r(y)
=
\frac{
[p(y)-q(y)]_+
}{
\sum_z[p(z)-q(z)]_+
}.
$$

Sampling from this residual after rejection corrects for the mass already represented by the draft proposal.

The resulting emitted token has the same marginal distribution as direct sampling from $p$.

## Exactness depends on the acceptance algorithm

The statement that speculative decoding is lossless is not true for every method that predicts several tokens.

It is true for the acceptance-rejection construction designed to preserve the target distribution.

If an implementation instead accepts tokens heuristically, changes thresholds, prunes candidate trees approximately or modifies the target logits, it may introduce approximation.

"Speculative decoding" is therefore a family label.

The specific acceptance rule matters.

## Draft quality determines acceptance rate

Suppose the draft model closely approximates the target:

$$
q(\cdot\mid h)
\approx
p(\cdot\mid h).
$$

Then for likely draft tokens,

$$
\frac{p(y)}{q(y)}
\approx1,
$$

so acceptance is high.

If the draft distribution differs strongly from the target, many tokens are rejected.

The draft model can then add overhead without saving target decoding steps.

A good draft model is therefore not merely small.

It must be **small and sufficiently aligned with the target**.

## Acceptance probability is linked to distribution overlap

For a single position, the expected acceptance probability under the draft is

$$
\mathbb E_{y\sim q}
\left[
\min
\left(
1,
\frac{p(y)}{q(y)}
\right)
\right].
$$

This simplifies to

$$
\sum_y
\min
(
p(y),
q(y)
).
$$

Using total variation distance,

$$
D_{\mathrm{TV}}(p,q)
=
\frac12
\sum_y
|p(y)-q(y)|,
$$

we obtain

$$
\sum_y
\min(p(y),q(y))
=
1-D_{\mathrm{TV}}(p,q).
$$

So one-token acceptance is directly related to overlap between the draft and target distributions.

This gives a clean interpretation:

$$
\boxed{
\text{better distributional agreement}
\Rightarrow
\text{higher acceptance}
}
$$

## Several accepted tokens can come from one target pass

Let $\gamma$ be the number of draft tokens proposed in one speculative round.

If the first $m$ proposals are accepted before a rejection, then several target-distributed tokens are committed after one verification pass.

The maximum number of accepted draft tokens is $\gamma$.

Some formulations also allow one additional target-sampled token after all proposals are accepted.

The speedup depends on the expected number of committed tokens per target verification.

## A crude expected-acceptance model

Assume, only for intuition, that each draft position has independent acceptance probability $\alpha$.

Then the probability that at least $k$ consecutive draft tokens are accepted is approximately

$$
\alpha^k.
$$

The expected number of accepted draft tokens is

$$
\mathbb E[A]
=
\sum_{k=1}^{\gamma}
\alpha^k.
$$

For $\alpha\neq1$,

$$
\mathbb E[A]
=
\frac{
\alpha(1-\alpha^\gamma)
}{
1-\alpha
}.
$$

If $\alpha$ is low, increasing $\gamma$ adds little.

If $\alpha$ is high, longer drafts can be useful.

The independence assumption is crude, but the qualitative trade-off is real.

## Draft length is a tuning parameter

A large $\gamma$ gives more speculative opportunity.

It also increases:

- draft-model work,
- target verification width,
- wasted computation after early rejection,
- temporary KV-cache complexity.

There is usually an optimal region rather than a universal best draft length.

A useful experiment might compare

$$
\gamma\in\{2,4,6,8\}.
$$

Measure actual latency, not only accepted tokens.

## The cheapest draft is not always the fastest system

Suppose draft model $q_1$ is tiny but has low acceptance.

Draft model $q_2$ is larger but predicts the target much better.

The system latency may satisfy

$$
T(q_2)
<
T(q_1)
$$

despite $q_2$ being slower per draft token.

The relevant quantity is the combined cost:

$$
T
\approx
T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{rejection}}
+
T_{\mathrm{overhead}}.
$$

Draft-model size alone does not optimize this objective.

## Tokenizer compatibility matters

The simplest speculative systems use draft and target models with compatible tokenization.

If token boundaries differ, mapping proposals between models becomes complicated.

Some assisted-generation methods support heterogeneous tokenizers through additional alignment logic, but this adds overhead and edge cases.

A draft from the same model family is operationally convenient because vocabulary and token semantics already align.

## Target and draft calibration need not match perfectly

The draft does not need identical probabilities.

It needs enough overlap to produce useful acceptance.

This distinction matters because a smaller model can be less calibrated or less capable overall while still predicting common local continuations well enough to accelerate decoding.

Speculation exploits **local predictive agreement**, not full capability equivalence.

## Easy tokens are where speculative decoding wins

Many language-model tokens are locally predictable.

Examples include:

- punctuation,
- common syntactic continuations,
- repeated formatting,
- boilerplate code tokens,
- standard phrase completions.

A small draft model may predict these almost identically to the target.

The large model's full capacity is not required at every token.

Speculative decoding exploits this heterogeneity in token difficulty.

## Difficult tokens cause rejection bursts

When the target distribution becomes uncertain or diverges from the draft, rejection increases.

This often happens around:

- factual choices,
- rare names,
- code branch decisions,
- mathematical steps,
- abrupt topic changes.

The system may therefore show variable speedup across prompts.

One average speedup number can hide substantial heterogeneity.

## Greedy decoding is a special case

For deterministic greedy decoding, the target chooses

$$
y^\star
=
\arg\max_y p(y).
$$

A draft token can be accepted whenever its greedy choice agrees with the target under the relevant verification rule.

This removes sampling randomness from the acceptance decision.

The exact algorithm differs from stochastic speculative sampling, but the systems logic is similar.

Agreement produces multi-token progress.

## Temperature changes draft-target agreement

Suppose target sampling temperature is $\tau$.

The target distribution becomes

$$
p_\tau(y)
\propto
\exp
\left(
\frac{z_y}{\tau}
\right).
$$

Higher temperature flattens the distribution.

The draft and target may then disagree more often, depending on their logit structure.

Speculative speedups should therefore be benchmarked under the actual decoding configuration.

A speedup reported for greedy generation may not transfer to high-temperature sampling.

## Top-p and top-k also alter acceptance behaviour

Truncating the target distribution changes which tokens have nonzero sampling probability.

If draft and target use different truncation rules, acceptance can deteriorate or exactness can be compromised unless the algorithm accounts for the transformed distributions correctly.

The decoding policy is part of the method specification.

## Batch size can erase speculative gains

Single-request latency is where speculative decoding is most intuitive.

Large-batch serving changes the arithmetic.

A target GPU with a large batch may already be well utilized.

Adding a draft model and irregular verification can reduce batching efficiency.

The best technique for interactive single-user latency may not maximize datacenter throughput.

Always distinguish:

- latency,
- throughput.

## Memory is another trade-off

A separate draft model requires additional weights in memory.

If target size is

$$
P_T
$$

and draft size is

$$
P_D,
$$

weight memory becomes roughly

$$
M
\approx
M_T+M_D.
$$

For constrained devices, this can eliminate the feasibility advantage.

Quantizing the draft can help, provided draft latency and acceptance remain good.

## KV-cache management is nontrivial

Draft tokens create speculative states.

If only a prefix is accepted, KV entries associated with rejected suffix tokens must be discarded or managed correctly.

The target verification pass also produces KV states for several positions.

High-performance implementations need careful cache bookkeeping to avoid turning algorithmic speedup into memory-copy overhead.

## Speculative decoding mainly targets decode latency

Prefill already processes prompt tokens in parallel.

Speculative decoding attacks the autoregressive output phase.

If a workload has enormous prompts and very short outputs, speculative decoding may have little impact on end-to-end latency.

Let

$$
T_{\mathrm{total}}
=
T_{\mathrm{prefill}}
+
T_{\mathrm{decode}}.
$$

If

$$
T_{\mathrm{prefill}}
\gg
T_{\mathrm{decode}},
$$

even halving decode time changes total latency only modestly.

## Amdahl's law applies

Suppose fraction $f$ of total latency is decoding and speculative decoding speeds that phase by factor $s$.

Overall speedup is

$$
S
=
\frac{
1
}{
(1-f)+f/s
}.
$$

If decode is only 30% of total latency and is accelerated 3x,

$$
S
=
\frac1{0.7+0.3/3}
=
1.25.
$$

A large decoding benchmark speedup can become a modest application-level improvement.

## Self-speculative decoding removes the separate draft model

A separate assistant model is not mandatory.

One can construct a cheaper draft from the target itself using:

- early exiting,
- fewer layers,
- reduced computation,
- auxiliary heads.

This avoids storing a second full model.

The challenge is producing drafts cheaply enough while maintaining good acceptance.

## Early-exit drafting uses partial target computation

Suppose a transformer has $L$ layers.

A draft can use only the first

$$
L_D<L
$$

layers to propose tokens.

The target then completes verification using the full network.

This shares parameters and tokenizer.

The speedup depends on whether the partial network is sufficiently predictive.

## Medusa predicts future tokens with auxiliary heads

Medusa attaches additional decoding heads to the target representation so multiple future-token candidates can be proposed without a separate draft model.

It uses tree-based candidate verification to reduce serial decoding steps.

The Medusa paper reports substantial speedups, with different training regimes trading simplicity, exactness or quality preservation. citeturn805218academia0

This belongs to the broader parallel-decoding family rather than the simplest two-model speculative sampler.

## Multi-head methods change where speculation lives

Two-model speculative decoding has:

$$
\text{draft network}
+
\text{target network}.
$$

Medusa-style decoding has:

$$
\text{target backbone}
+
\text{auxiliary future-token heads}.
$$

The systems trade-off shifts from storing another model to training and serving extra heads.

The common idea is still to propose several future candidates and verify them efficiently.

## Speedup must be measured on end-to-end generation

A useful benchmark reports:

- prompt length,
- output length,
- batch size,
- decoding temperature,
- draft length,
- acceptance rate,
- draft latency,
- target verification latency,
- total tokens/s,
- time to first token,
- end-to-end wall-clock latency.

"2x faster" without these conditions is not reproducible.

## Accepted tokens per target call are a useful diagnostic

Define

$$
A
=
\frac{
\text{committed output tokens}
}{
\text{target verification calls}
}.
$$

Standard autoregressive decoding has approximately

$$
A=1.
$$

Useful speculative decoding should produce

$$
A>1.
$$

But high $A$ alone is insufficient if draft overhead is large.

It is a diagnostic, not the final objective.

## Target calls per output token provide the inverse view

Define

$$
C_T
=
\frac{
\text{target calls}
}{
\text{output tokens}
}.
$$

Ordinary decoding gives roughly

$$
C_T=1.
$$

Speculation aims for

$$
C_T<1.
$$

This metric helps separate algorithmic progress from implementation latency.

## Speedup variance across prompts matters

If some prompts achieve 3x speedup and others 0.8x, the mean can hide an unpleasant tail.

Report distributions:

- median,
- p10,
- p90,
- fraction slower than baseline.

Interactive systems care about worst-case latency as well as mean throughput.

## Draft adaptation can improve acceptance

A draft model can be fine-tuned or distilled toward the target distribution.

This creates a new optimization problem:

$$
\min
D(
p_T
\|
q_D
)
$$

on deployment-like prefixes.

Improved agreement can raise acceptance.

The benefit must exceed the training and maintenance cost of the specialized draft.

## The draft can become stale when the target changes

If the target model is updated, fine-tuned or given a different adapter, the old draft may no longer approximate it well.

Acceptance can drop.

Speculative systems therefore introduce a compatibility relation between draft and target versions.

Version them together.

## LoRA adapters complicate draft-target matching

Suppose the target is

$$
p_{\theta+\Delta\theta}.
$$

A draft approximating the base

$$
p_\theta
$$

may lose acceptance after a strong adapter is applied.

One option is to adapt the draft too.

Another is self-speculation from the target.

This should be benchmarked for each deployed adapter.

## Quantization can help or hurt speculation

A quantized draft reduces memory and may improve latency.

Quantization can also perturb draft probabilities and lower acceptance.

A quantized target changes verification speed.

Therefore the joint system should be benchmarked, not inferred from separate model benchmarks.

## Speculative decoding can be slower

It loses when:

- draft inference is too expensive,
- acceptance is low,
- draft length is poorly chosen,
- batch serving is already saturated,
- output sequences are short,
- cache management overhead is high,
- target verification kernels are inefficient.

The method is not universally beneficial.

## The correct baseline is the same target decoder

Compare speculative decoding against ordinary decoding of the **same target model** under the same:

- precision,
- hardware,
- batch size,
- prompt set,
- sampling configuration.

Otherwise quality and speed changes become entangled.

## Exact speculative sampling should match target statistics

If the algorithm claims exactness, validate it empirically.

On a controlled prompt, generate many samples using:

1. direct target sampling,
2. speculative sampling.

Compare token or sequence statistics.

The distributions should agree within Monte Carlo error.

Performance engineering should not silently change model semantics.

## A practical acceptance-rate study is simple

For each generation, log:

~~~text
drafted_tokens
accepted_tokens
rejected_rounds
target_calls
draft_time_ms
verify_time_ms
total_time_ms
~~~

Then estimate

$$
\widehat\alpha
=
\frac{
\text{accepted draft tokens}
}{
\text{drafted tokens}
}.
$$

Stratify by prompt type and generation position.

This tells you where speculation works and where it fails.

## Draft length should be selected from latency, not acceptance alone

For candidate $\gamma$, measure

$$
T(\gamma).
$$

Choose

$$
\gamma^\star
=
\arg\min_\gamma T(\gamma)
$$

subject to preserving output semantics.

The highest acceptance configuration is not necessarily the fastest.

## Adaptive draft length can exploit changing difficulty

Token difficulty changes during generation.

An adaptive controller can choose shorter speculation when recent acceptance is low and longer speculation when acceptance is high.

This avoids paying for large drafts in difficult regions.

The controller itself adds complexity and should be benchmarked against a fixed-$\gamma$ baseline.

## Speculative decoding changes systems architecture, not model capability

A target model does not become more accurate because a draft model is present.

The draft model does not add knowledge.

The technique attempts to compute the same target distribution more efficiently.

This distinguishes it from:

- RAG,
- fine-tuning,
- DPO,
- distillation.

Those alter information access or model behaviour.

Speculative decoding alters the **execution strategy**.

## The right objective is application-level latency or cost

The optimization target is not acceptance rate.

It is something like

$$
\min
\left[
\lambda_1 T_{\mathrm{latency}}
+
\lambda_2 C_{\mathrm{compute}}
+
\lambda_3 M_{\mathrm{memory}}
\right]
$$

subject to

$$
p_{\mathrm{output}}
=
p_{\mathrm{target}}
$$

for exact speculative sampling.

That is the systems problem.

## References

Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. *Proceedings of ICML 2024*.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. *arXiv:2302.01318*.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of ICML 2023*.
