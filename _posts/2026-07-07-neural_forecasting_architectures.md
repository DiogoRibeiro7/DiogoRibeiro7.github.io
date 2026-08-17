---
permalink: '/time-series/neural_forecasting_architectures/'
title: 'Neural Forecasting: What the Architectures Actually Do'
categories:
- Time Series
tags:
- Time Series
- Neural Networks
- Forecasting
- Machine Learning
author_profile: false
seo_title: 'Neural Forecasting Architectures'
seo_description: 'DeepAR, N-BEATS and Temporal Fusion Transformers, what each is built for, and when a gradient boosting baseline still wins.'
excerpt: >-
  Neural forecasting has produced genuinely useful architectures and a great
  deal of noise. The differences between them are more interesting than their
  benchmark scores.
summary: >-
  A practical comparison of the main neural forecasting architectures:
  DeepAR's probabilistic autoregression, N-BEATS's interpretable basis
  expansion, the Temporal Fusion Transformer's handling of covariates, and the
  conditions under which none of them beats gradient boosting.
keywords:
  - neural forecasting
  - DeepAR
  - N-BEATS
  - temporal fusion transformer
  - deep learning forecasting
classes: wide
date: '2026-07-07'
header:
  image: /assets/images/data_science_5.jpg
  og_image: /assets/images/data_science_5.jpg
  overlay_image: /assets/images/data_science_5.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_5.jpg
  twitter_image: /assets/images/data_science_5.jpg
---
Neural forecasting arrived with large claims and a poor early record. The M4 competition in 2018 saw pure deep learning entries beaten by statistical methods; the winner was a hybrid. Since then the architectures have improved substantially, and the useful question is no longer whether they work but what each is actually built to do.

## DeepAR: Probabilistic Autoregression

DeepAR trains a recurrent network across many related series at once, and its defining choice is the output: rather than predicting a value, it predicts the **parameters of a distribution** — the mean and variance of a Gaussian, or the parameters of a negative binomial for counts.

Training maximises the likelihood of the observed data under those predicted parameters. Forecasting proceeds by sampling: draw a value from the predicted distribution, feed it back as input, predict the next step, repeat. Running many such paths gives a full predictive distribution rather than an interval bolted on afterwards.

Two consequences follow. Because it is global, it forecasts new series with little history using learned patterns from related ones. And because the likelihood is explicit, the choice of distribution is a modelling decision you make deliberately — a negative binomial for counts rather than forcing a Gaussian onto data that cannot be negative.

The autoregressive sampling is also its main weakness: errors compound along the path, and generation is sequential and therefore slow at long horizons.

## N-BEATS: Basis Expansion, No Recurrence

N-BEATS discards recurrence entirely. It is a deep stack of fully connected blocks, each producing two outputs: a **backcast** (its reconstruction of the input window) and a **forecast**. Each block subtracts its backcast from the input before passing the residual onward, so successive blocks model what earlier ones could not.

This residual arrangement is the architectural idea, and it produces something unusual for a neural model: interpretability by construction. In the interpretable variant, blocks are constrained to specific basis functions — polynomial for trend, Fourier for seasonality — so the output decomposes into named components much like a classical decomposition.

N-BEATS produces the whole horizon in one forward pass rather than step by step, avoiding compounding error and running considerably faster than autoregressive generation. Its authors report outperforming the M4 competition winner, which was notable because the deep learning entries to M4 itself had not managed that.

Its limitation is that the basic form is univariate and point-valued: it does not natively take covariates, and it does not produce a predictive distribution without additional machinery. NHITS extends it with multi-rate sampling that improves long-horizon performance.

## Temporal Fusion Transformer: Covariates Taken Seriously

The TFT is built around a distinction the other two mostly ignore: different kinds of input carry different information.

It separates **static covariates** (product category, store location), **known-future inputs** (holidays, planned promotions, day of week — things you know in advance), and **observed past inputs** (things measured historically but unknown for the future). Most real forecasting problems have all three, and treating a known holiday calendar identically to an unknown past observation discards genuine information.

Architecturally it combines an LSTM for local processing with attention over longer ranges, and adds variable selection networks that learn which inputs matter. It outputs quantiles directly, trained on pinball loss, so it is probabilistic without sampling.

The attention weights and variable selection weights give some interpretability, though this should be read cautiously — attention weights are suggestive of importance, not a reliable attribution of it.

## When None of Them Wins

The M5 competition is the useful counterweight. It was dominated by **gradient boosting** with careful feature engineering, not by neural architectures. That result is not an anomaly, and the conditions under which it holds are predictable.

Neural models need scale. Their advantage comes from learning shared structure across many series, and with a few dozen series there is not enough signal to justify the parameters. Below a few hundred related series, gradient boosting on lag features is usually both better and cheaper.

They also need the relationships to be genuinely complex. If the structure is trend plus seasonality plus a few covariate effects, a linear model or ETS captures it, and additional capacity fits noise.

And they cost considerably more: tuning, training time, and monitoring. A model that improves accuracy by 2% while adding a GPU dependency and a week of engineering may not be worth it, and that trade-off should be evaluated explicitly rather than assumed away.

## Foundation Models

The current direction is pretrained models — TimeGPT, Chronos, Moirai and others — trained on large collections of series and applied zero-shot to new ones.

The appeal is obvious: no per-dataset training, and immediate forecasts for series with little history. Reported results are genuinely promising, and two cautions are warranted. Evaluation is difficult because the pretraining corpora are large and sometimes undocumented, so contamination with benchmark data is hard to rule out. And zero-shot performance on a specific domain with idiosyncratic structure is frequently worse than a small model trained on that domain.

The reasonable position is to benchmark them against your own baselines rather than against published averages, which is the same discipline that applies to any new method.

## A Sensible Order of Work

Establish the baselines first — seasonal naive, ETS, and gradient boosting on lag features. These are fast, and they set the bar the neural model has to clear.

Reach for neural forecasting when you have many related series, genuine need for probabilistic output, or covariates with the structure the TFT is designed around. Choose the architecture by what the problem needs: DeepAR for probabilistic counts across many series, N-BEATS for fast univariate point forecasts at long horizons, TFT when known-future covariates carry real information.

And evaluate honestly, with rolling-origin validation, against those baselines, on the metric the decision actually depends on. The literature on this subject rewards scepticism: the gap between benchmark results and production performance has been wide, and the methods that survive contact with real data are usually the simpler ones.

## References

- Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191.
- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: neural basis expansion analysis for interpretable time series forecasting. *ICLR*.
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364.
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of AAAI*, 37(9), 11121-11128.
