---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2023-10-02'
excerpt: Modern NLP ranges from classical token-based models to pretrained transformers and language models. The core challenges remain representation, evaluation, domain shift, retrieval, and task definition.
header:
  image: /assets/images/headers/photo-data-science-openalex.jpg
  og_image: /assets/images/headers/photo-data-science-openalex.jpg
  overlay_image: /assets/images/headers/photo-data-science-openalex.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-openalex.jpg
  twitter_image: /assets/images/headers/photo-data-science-openalex.jpg
keywords:
- Natural language processing
- Transformers
- Language models
- Text classification
- Embeddings
- Retrieval
- NLP evaluation
permalink: '/machine-learning/overview_natural_language_processing_data_science/'
redirect_from:
- '/natural language processing/overview_natural_language_processing_data_science/'
- '/machine learning/overview_natural_language_processing_data_science/'
seo_description: A modern overview of NLP covering classical representations, transformers, embeddings, retrieval, language models, evaluation, and domain shift.
seo_title: 'Natural Language Processing: Models, Tasks, and Evaluation'
seo_type: article
tags:
- Natural Language Processing
- Machine Learning
- Data Science
title: Natural Language Processing: Models, Tasks, and Evaluation
---

Natural language processing is the study of computational methods for text and language. The field includes tasks as different as document classification, information extraction, retrieval, translation, summarization, question answering, and open-ended generation. These tasks should not be collapsed into a single notion of "understanding language" because they impose different statistical and operational requirements.

## Text is structured, not merely unstructured

Text has sequence, syntax, discourse, pragmatics, genre, and context. Treating it as unstructured data is a convenient database label, not a description of its statistical structure.

An NLP pipeline therefore begins by deciding what unit carries information: characters, subwords, words, sentences, documents, conversations, or retrieved passages.

## Classical representations still matter

Before transformers, many strong text systems used sparse vector representations such as bag-of-words and TF-IDF.

For term t in document d, a common TF-IDF representation is

$$
\operatorname{tfidf}(t,d)
=
\operatorname{tf}(t,d)
\log\frac{N}{\operatorname{df}(t)}.
$$

Linear classifiers on these representations remain competitive for many supervised text-classification problems, especially when labels are limited, latency matters, or interpretability is useful.

Modern NLP should not be taught as if every problem requires a large language model.

## Embeddings

Dense embeddings map tokens, sentences, or documents into continuous vector spaces. Earlier methods such as word2vec learned distributional word representations. Transformer encoders produce contextual embeddings, so the representation of a token depends on surrounding text.

Embeddings are useful for retrieval, clustering, semantic similarity, and as inputs to downstream models. But geometric proximity is model- and domain-dependent. A cosine similarity score is not a universal semantic truth.

## Transformers

Transformer models replace recurrent computation with attention-based sequence processing. A simplified scaled dot-product attention operation is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Pretraining on large text corpora allows models to learn reusable representations. Fine-tuning or prompting then adapts those representations to downstream tasks.

The practical advantages are transfer learning and flexible conditioning. The costs include substantial compute, sensitivity to data provenance, and evaluation challenges when outputs are open-ended.

## Encoder, decoder, and encoder-decoder models

Encoder-style models are naturally suited to representation and classification tasks. Decoder-only models are autoregressive and generate text token by token. Encoder-decoder architectures are common for conditional generation tasks such as translation and summarization.

These are architectural tendencies rather than absolute boundaries.

## Classification and information extraction

For text classification, the target should be defined independently of the model. A sentiment label, toxicity label, or intent category can be ambiguous, culturally dependent, or poorly measured.

Named entity recognition similarly depends on an annotation schema. Whether "OpenAI" is an organization, product, or nested entity is partly a labeling convention.

Evaluation should therefore report annotation quality and class definitions, not only model scores.

## Retrieval

Many NLP systems first retrieve relevant documents and only then classify, summarize, or generate.

Retrieval quality can be evaluated with metrics such as recall at k, mean reciprocal rank, or normalized discounted cumulative gain, depending on the task.

In retrieval-augmented generation, generation quality is bounded by both retrieval and synthesis. A fluent answer cannot recover evidence that was never retrieved.

## Language-model generation

Autoregressive language models estimate a conditional token distribution

$$
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}
P(x_t\mid x_{<t}).
$$

This training objective rewards predictive fit to text. It does not directly optimize factuality, calibration, causal reasoning, or alignment with a user's underlying goal.

That distinction explains why generated text can be coherent while containing unsupported claims.

## Evaluation must match the task

NLP evaluation is especially vulnerable to using a convenient metric for the wrong target.

- accuracy and F1 can be useful for classification, but may hide calibration and subgroup performance
- BLEU and ROUGE measure forms of lexical overlap, not complete translation or summary quality
- retrieval metrics do not measure final answer correctness
- language-model perplexity does not measure factual reliability
- human evaluation can be informative but needs explicit rubrics and inter-rater design

Benchmark contamination and repeated tuning on public test sets can also make reported scores optimistic.

## Domain shift

Language changes across organizations, time periods, professions, communities, and platforms. A model trained on product reviews may fail on clinical notes even if both tasks are called sentiment or classification.

Vocabulary shift is only one problem. Label prevalence, annotation conventions, document length, and writing style can change too.

External validation or time-based validation is therefore essential for many production systems.

## Preprocessing is model dependent

Lowercasing, stemming, stop-word removal, and aggressive token cleaning were common in sparse classical pipelines. They are not universally beneficial for pretrained transformers, whose tokenizers and pretraining distributions already encode assumptions about casing and punctuation.

Preprocessing should follow the representation and task rather than a fixed checklist.

## Tools

NLTK remains useful for linguistic algorithms and teaching. spaCy provides efficient production-oriented tokenization, tagging, parsing, and entity pipelines. Hugging Face Transformers provides model and tokenizer abstractions for pretrained transformer architectures. Gensim remains useful for topic modeling and vector-space methods.

The library choice is secondary to the statistical design.

## A modern NLP workflow

1. Define the unit of text and target label or retrieval objective.
2. Establish a simple lexical baseline.
3. Split data to reflect future deployment, including time, author, or source grouping where necessary.
4. Compare classical and pretrained representations.
5. Evaluate calibration, subgroup behavior, and domain shift, not just aggregate accuracy.
6. For generative systems, separate retrieval, factuality, instruction following, and style evaluation.
7. Record data provenance and contamination risks.

## Conclusion

NLP has changed dramatically with pretrained transformers and large language models, but the fundamental discipline has not changed. The task must be defined carefully, labels must be meaningful, validation must match deployment, and evaluation must measure the property the system is supposed to deliver.

Modern models broaden what can be built. They do not remove the need for statistical reasoning.

## References

- Jurafsky, D., & Martin, J. H. *Speech and Language Processing*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Vaswani, A., et al. (2017). Attention Is All You Need.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*.
