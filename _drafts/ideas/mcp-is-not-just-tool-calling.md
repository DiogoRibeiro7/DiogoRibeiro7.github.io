---
author_profile: false
categories:
- Programming
classes: wide
excerpt: MCP is often described as a way to let models call tools. The more useful engineering view is that it defines a typed, controlled, observable boundary between AI systems and external capabilities.
header:
  image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  og_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-normal-distribution.jpg
  twitter_image: /assets/images/headers/photo-statistics-normal-distribution.jpg
keywords:
- MCP
- Model Context Protocol
- LLM engineering
- tool calling
- AI systems
seo_description: A systems view of MCP as a typed, authorized, observable capability boundary rather than merely a tool-calling mechanism.
seo_title: MCP Is Not Just Tool Calling
seo_type: article
summary: Why the interesting engineering problem in MCP is capability design, authorization, observability, failure handling, and stable interfaces.
tags:
- AI Engineering
- MCP
- LLMs
title: 'MCP Is Not Just Tool Calling'
---

MCP is often introduced as a way to let a language model call tools.

That description is correct, but incomplete.

The more useful engineering view is that MCP defines a **boundary between a probabilistic model and external capabilities**.

That boundary deserves the same care we would give any other production interface.

## The model should not own authority

A language model can decide that a capability is useful. It should not decide whether the caller is authorized to use it.

Authorization belongs outside the model.

If a tool exposes customer data, account actions, internal documents, or infrastructure operations, permissions must be enforced independently of whatever text the model generates.

The architectural principle is:

$$
\boxed{
\text{model intent} \neq \text{system authority}
}
$$

The model proposes. The system validates and executes.

## Narrow tools are usually better tools

A generic capability such as unrestricted SQL access is flexible, but that flexibility transfers a large amount of responsibility to the model.

A narrower capability such as

```text
get_account_briefing_data(account_id)
```

is easier to authorize, test, observe, version, and use correctly.

The design question is therefore not only

> What can the model call?

It is

> What is the smallest stable capability that represents the business operation we actually want to expose?

## Types are part of the safety model

A tool interface should make invalid states difficult to express.

Inputs should be narrow and typed. Outputs should be structured. Validation should occur before execution. Errors should have predictable semantics.

This reduces ambiguity for both the model and the surrounding application.

A good MCP tool is closer to a small API contract than to an arbitrary function exposed for convenience.

## Observability matters because the caller is probabilistic

Traditional application code can already fail in many ways. An LLM introduces another layer: it may choose the wrong tool, omit required context, construct a valid but inappropriate request, or retry in an unexpected pattern.

That makes observability essential.

For each tool call I want to know at least:

- which capability was selected;
- with which validated inputs;
- how long it took;
- whether authorization succeeded;
- what external dependency was contacted;
- what error class occurred;
- whether the model retried or changed strategy.

Without that trace, debugging becomes anecdotal.

## Failure behaviour is part of the contract

External systems fail. Data can be missing. APIs time out. Credentials expire. Rate limits appear.

An MCP integration should therefore define how failures are represented rather than passing arbitrary exception text back to the model.

The model needs enough information to choose a safe next action, but the application should retain control over retry policy, sensitive error details, and irreversible operations.

## Tool design is product design

The strongest tool interfaces usually reflect meaningful user or business operations rather than underlying infrastructure.

For example, an enterprise briefing assistant may need capabilities such as:

```text
get_account_summary(account_id)
get_recent_customer_interactions(account_id)
get_open_risks(account_id)
get_relevant_documents(account_id)
```

rather than a collection of low-level database and filesystem operations.

That gives the model a smaller and more meaningful action space.

## MCP does not remove ordinary engineering

Using a protocol does not remove the need for:

- authentication and authorization;
- schema evolution;
- versioning;
- monitoring;
- retries and idempotency;
- privacy boundaries;
- integration tests;
- cost and latency control.

If anything, model-driven orchestration makes these concerns more important because the sequence of calls is less deterministic.

## The useful mental model

I would describe MCP as

$$
\boxed{
\text{a standardized capability boundary for AI applications}
}
$$

rather than simply a tool-calling mechanism.

The protocol solves an interoperability problem. The hard engineering work remains deciding which capabilities to expose, how narrowly to define them, who may use them, and how to know what happened when something goes wrong.
