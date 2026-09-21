---
permalink: '/time-series/a_forecast_distribution_should_be_calibrated_and_sharp/'
title: 'A Forecast Distribution Should Be Calibrated and Sharp'
date: '2025-10-16'
categories:
- Time Series
tags:
- Probabilistic Forecasting
- Forecast Evaluation
- Calibration
- Proper Scoring Rules
- CRPS
author_profile: false
classes: wide
seo_title: 'A Forecast Distribution Should Be Calibrated and Sharp'
seo_description: 'Probabilistic forecasts should be evaluated through calibration, sharpness, PIT diagnostics and proper scoring rules. Coverage alone can reward uselessly wide forecasts.'
seo_type: article
excerpt: >-
  A probabilistic forecast is useful only if its stated uncertainty is credible
  and concentrated enough to support decisions. Calibration without sharpness can
  be achieved by being vague, while sharpness without calibration is simply
  overconfidence.
summary: >-
  This article develops probabilistic forecast evaluation from calibration and
  sharpness. Three Gaussian forecasters with identical point means but different
  variances are compared exactly through interval coverage, interval width,
  expected logarithmic score, CRPS and PIT distributions. A second construction
  shows that a globally uniform PIT can coexist with severe conditional
  miscalibration when a known regime is ignored. The article then connects
  quantile calibration, proper scoring rules, horizon-specific evaluation,
  dependence, ensemble forecasts and operational decisions.
keywords:
- probabilistic forecasting
- forecast calibration
- sharpness
- probability integral transform
- CRPS
- logarithmic score
why_this_exists: >-
  Forecast evaluation is often dominated by point errors or by nominal interval
  coverage. Neither is sufficient for a predictive distribution. A forecaster
  can obtain high coverage by reporting intervals that are far too wide, or pass
  an aggregate calibration diagnostic while remaining systematically wrong in
  important subgroups or regimes.
evidence: >-
  Exact Gaussian coverage and scoring-rule calculations, an analytic PIT density
  under variance misspecification, a regime-mixture counterexample to conditional
  calibration, and classical literature on proper scoring rules and
  probabilistic forecast evaluation.
methodology: >-
  Compare three predictive distributions that share the correct conditional mean
  but use standard deviations 0.5, 1 and 2 when the truth is N(0,1). Derive
  coverage, interval width, expected log score, expected CRPS and PIT behaviour.
  Then construct a two-regime mixture whose unconditional forecast is exactly
  calibrated while its conditional distributions ignore available information.
reviewed_at: '2026-09-21'
header:
  image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  og_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-statistics-confidence-intervals.jpg
  twitter_image: /assets/images/headers/photo-statistics-confidence-intervals.jpg
---

<!--
Development contract
Question: What makes a probabilistic forecast useful rather than merely conservative?
Claim: A useful forecast distribution should be calibrated relative to the information available at forecast time and as sharp as possible subject to that calibration. Proper scoring rules provide principled incentives for reporting the full predictive distribution honestly.
Counterclaim: No single calibration diagnostic is sufficient. Marginal calibration can conceal conditional failure, while proper scores combine several aspects of forecast quality into one number and therefore require decomposition and diagnostic plots for interpretation.
Evidence object: Three exact Gaussian forecasters with equal means and different variances, PIT density under scale misspecification, expected log score and CRPS calculations, and an exact regime-mixture example that passes global PIT calibration while ignoring known regime information.
Failure case: Ranking intervals by coverage alone, treating a uniform PIT as proof of conditional correctness, using point metrics to evaluate distributions, or aggregating scores across horizons and regimes that have different decision relevance.
Reader payoff: Evaluate forecast distributions through calibration, sharpness and proper scores, and recognise when apparently calibrated uncertainty is too wide, too narrow or conditionally wrong.
Exclusions: A catalogue of forecasting architectures, repetition of the existing quantile-and-pinball-loss article, and a full treatment of conformal prediction.
-->

A point forecast can be exactly the same under several probabilistic models that imply very different decisions. Suppose three forecasting systems all predict a mean demand of 100 units for tomorrow. The first says uncertainty is small, the second says it is moderate, and the third says almost anything between 60 and 140 is plausible. If tomorrow's demand is 115, the three systems have identical point error and very different probabilistic performance.

This is why point metrics such as RMSE and MAE cannot evaluate a predictive distribution. They assess one functional of the distribution, usually the mean or median, after most of the uncertainty information has been discarded. Once a forecasting system reports quantiles, intervals, densities or entire simulated paths, evaluation must ask whether that uncertainty is statistically credible and whether it is concentrated enough to be useful.

The standard language for this problem is calibration and sharpness. Calibration concerns statistical consistency between predictive probabilities and observed frequencies. Sharpness concerns the concentration of the predictive distributions and is a property of the forecasts themselves. A forecaster that is too narrow can be sharp and badly calibrated. A forecaster that is extremely wide can achieve high empirical coverage and be practically useless. The objective is therefore not maximum sharpness or maximum coverage in isolation. It is sharpness subject to calibration.

Proper scoring rules turn this principle into an optimization criterion. They reward predictive distributions that assign high probability to what actually happens while penalizing distributions that become diffuse merely to avoid being surprised. Their deeper value is incentive compatibility: in expectation, a strictly proper score is optimized by reporting the forecaster's true predictive distribution rather than a strategically distorted one.

The distinction sounds qualitative until the numbers are written down. A simple Gaussian example shows that calibration, interval width and proper scores can disagree sharply even when every forecasting system has the correct point mean.

## The same mean can hide very different uncertainty models

Assume the true one-step-ahead outcome is

$$
Y\sim N(0,1).
$$

Consider three forecasters. All report mean zero, so their expected squared-error point forecast is identical. They differ only in predictive standard deviation:

$$
F_{0.5}
=
N(0,0.5^2),
$$

$$
F_1
=
N(0,1),
$$

and

$$
F_2
=
N(0,2^2).
$$

The first forecast is underdispersed. It is too confident. The second is correctly specified. The third is overdispersed and too cautious.

A central 90% interval from a normal forecast with standard deviation $s$ is

$$
[-z_{0.95}s,\ z_{0.95}s],
$$

where

$$
z_{0.95}
\approx
1.64485.
$$

The interval width is therefore

$$
W_{0.90}(s)
=
2z_{0.95}s.
$$

Because the actual outcome has unit variance, empirical coverage of this forecast interval is

$$
C_{0.90}(s)
=
P\left(
|Y|
\le
z_{0.95}s
\right)
=
2\Phi(z_{0.95}s)-1.
$$

The three cases are:

| Forecast SD | 90% interval width | True coverage |
| ---: | ---: | ---: |
| 0.5 | 1.645 | 58.9% |
| 1.0 | 3.290 | 90.0% |
| 2.0 | 6.579 | 99.9% |

Coverage alone would favour the widest forecast. It almost never misses. That conclusion is obviously wrong if the stated target is a 90% interval. The forecast with standard deviation two obtains 99.9% coverage because it is reporting far more uncertainty than the data-generating process contains.

Width alone would favour the narrowest forecast. That conclusion is equally wrong because its 90% interval contains the truth only 58.9% of the time.

The correct forecast occupies the middle: it achieves the nominal coverage with substantially less width than the overdispersed system. This is the operational meaning of sharpness subject to calibration.

The same argument applies to asymmetric intervals, quantile forecasts and entire predictive densities. A forecast should not receive credit merely for avoiding errors by becoming vague. The uncertainty statement should match empirical uncertainty at the relevant conditioning level.

## The PIT diagnoses the direction of dispersion error

For a continuous predictive CDF $F_t$ and realized observation $Y_t$, the probability integral transform is

$$
U_t
=
F_t(Y_t).
$$

If the predictive distribution is correctly specified conditionally on the information used to produce it, the PIT values are uniform under suitable regularity conditions:

$$
U_t
\sim
U(0,1).
$$

For the Gaussian variance example, the PIT distribution can be derived exactly rather than described heuristically. Suppose the forecast is

$$
F_s(y)
=
\Phi
\left(
\frac{y}{s}
\right),
$$

while the truth is

$$
Y\sim N(0,1).
$$

Then

$$
U
=
\Phi
\left(
\frac{Y}{s}
\right).
$$

For $0<u<1$,

$$
P(U\le u)
=
P
\left[
Y
\le
s\Phi^{-1}(u)
\right]
=
\Phi
\left[
s\Phi^{-1}(u)
\right].
$$

Differentiating gives the PIT density

$$
g_s(u)
=
s
\exp
\left\{
\frac{
1-s^2
}{2}
\left[
\Phi^{-1}(u)
\right]^2
\right\}.
$$

When

$$
s=1,
$$

this reduces to

$$
g_1(u)=1,
$$

the uniform density.

When

$$
s<1,
$$

the coefficient

$$
1-s^2
$$

is positive. The density increases toward zero and one, producing the familiar U-shaped PIT histogram. Outcomes fall in the predictive tails too often because the forecast is too narrow.

When

$$
s>1,
$$

the exponent is negative away from the centre, so PIT values accumulate near one half. The histogram becomes hump-shaped. The forecast is too broad, placing too much probability in regions that rarely materialize.

This derivation matters because PIT diagnostics are sometimes treated as a pattern-recognition exercise with mnemonic rules. The shapes arise directly from the transformation implied by the predictive CDF.

Other failures create other PIT patterns. Bias pushes PIT mass toward one side. Skewness misspecification produces asymmetric departures. Serial dependence in PIT values can indicate that the marginal predictive distribution is approximately right while temporal dependence remains unmodelled. A histogram should therefore be supplemented by sequence diagnostics rather than treated as the entire calibration analysis.

For discrete outcomes the ordinary PIT is not exactly uniform because the CDF has jumps. Randomized PITs or discrete calibration diagnostics are needed. The principle remains the same: probability statements should be checked against frequencies in a way that respects the support of the predictive distribution.

## Proper scoring rules prevent the infinitely wide forecast from winning

Calibration diagnostics identify particular ways a forecast can fail, but they do not produce one scalar criterion for comparing complete predictive distributions. Proper scoring rules fill that role.

For a forecast density $f$ and observation $y$, the logarithmic score is

$$
S_{\log}(f,y)
=
-\log f(y).
$$

Lower values are better under the negative-log convention used here. The score penalizes forecasts that assign little density to what actually occurs.

For a Gaussian forecast

$$
N(0,s^2)
$$

when the truth is

$$
N(0,1),
$$

the expected logarithmic score is

$$
\mathbb E[
S_{\log}
]
=
\frac12
\log(2\pi s^2)
+
\frac{
1
}{
2s^2
}.
$$

This expression is minimized at

$$
s=1.
$$

For the three forecasters:

| Forecast SD | Expected negative log score |
| ---: | ---: |
| 0.5 | 2.226 |
| 1.0 | 1.419 |
| 2.0 | 1.737 |

The underdispersed forecast performs particularly badly because observations that fall outside its narrow centre receive extremely low density. The overdispersed forecast is safer but still loses probability mass by spreading it too widely. The correct forecast has the best expected score.

The continuous ranked probability score evaluates the entire CDF,

$$
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left[
F(z)
-
\mathbf 1\{y\le z\}
\right]^2
\,dz.
$$

It also has the useful representation

$$
\operatorname{CRPS}(F,y)
=
\mathbb E_F|X-y|
-
\frac12
\mathbb E_F|X-X'|,
$$

where $X$ and $X'$ are independent draws from the forecast distribution.

This representation makes the calibration-sharpness balance visible. The first term penalizes forecast draws that are far from the observation. The second rewards concentration by subtracting half the expected distance between two forecast draws. A diffuse forecast increases both terms and does not obtain a free pass merely by covering the observation.

If the truth is $Y\sim N(0,1)$ and the forecast is $N(0,s^2)$, then

$$
X-Y
\sim
N(0,s^2+1),
$$

so

$$
\mathbb E|X-Y|
=
\sqrt{
\frac{2}{\pi}
}
\sqrt{
s^2+1
}.
$$

Also,

$$
X-X'
\sim
N(0,2s^2),
$$

which gives

$$
\frac12
\mathbb E|X-X'|
=
\frac{s}{\sqrt\pi}.
$$

Hence the expected CRPS is

$$
\mathbb E[
\operatorname{CRPS}
]
=
\sqrt{
\frac{2}{\pi}
}
\sqrt{
s^2+1
}
-
\frac{s}{\sqrt\pi}.
$$

Numerically:

| Forecast SD | Expected CRPS |
| ---: | ---: |
| 0.5 | 0.610 |
| 1.0 | 0.564 |
| 2.0 | 0.656 |

Again the correctly specified distribution performs best.

The log score and CRPS emphasize different aspects of distributional error. The logarithmic score is highly sensitive to assigning very low density to the realized outcome, which makes it particularly unforgiving of tail misspecification. CRPS behaves more like an integrated absolute error over thresholds and is often easier to interpret robustly. Neither should replace diagnostic plots, but both have the crucial property that truthful predictive distributions are optimal in expectation.

This is what makes a score proper. A metric that can be improved systematically by lying about uncertainty is unsuitable for evaluating probabilistic forecasts.

## Quantile calibration is necessary but weaker than distributional correctness

A forecast can also be evaluated one quantile at a time. If

$$
q_\tau(t)
$$

is a forecast of the conditional $\tau$ quantile, calibration requires approximately

$$
P\{
Y_t
\le
q_\tau(t)
\}
=
\tau
$$

at the appropriate conditioning level.

Pinball loss is the proper score associated with a quantile. For error

$$
e=y-q,
$$

the loss is

$$
L_\tau(e)
=
\begin{cases}
\tau e,
&
e\ge0,\\
(\tau-1)e,
&
e<0.
\end{cases}
$$

The existing article [Probabilistic Forecasting: Beyond the Point Estimate](/time-series/probabilistic_forecasting_quantiles_pinball_loss/) develops this connection in detail. The important extension here is that correct coverage at a few selected quantiles does not guarantee that the full predictive distribution is correct.

A model can produce calibrated 90% intervals while being wrong inside the interval. It can match the median and 90th percentile and fail badly in the lower tail. Quantile crossing can make independently estimated quantiles incoherent. Calibration at one horizon can disappear at another.

Full-distribution scores and PIT diagnostics therefore complement quantile-specific evaluation rather than replacing it.

Interval scores make the same point for prediction intervals. For a central $(1-\alpha)$ interval $[l,u]$, the interval score is

$$
IS_\alpha(l,u;y)
=
(u-l)
+
\frac{2}{\alpha}
(l-y)
\mathbf 1\{y<l\}
+
\frac{2}{\alpha}
(y-u)
\mathbf 1\{y>u\}.
$$

The first term rewards narrow intervals. The remaining terms penalize misses, with a penalty scaled by the nominal error probability. An infinitely wide interval receives no miss penalty and pays an enormous width penalty. An extremely narrow interval receives a small width term and frequent large miss penalties.

This is the scalar version of sharpness subject to calibration.

## Global calibration can hide conditional failure

A more difficult problem appears when calibration is evaluated after averaging over information that was available when the forecast was made.

Suppose there are two observable regimes,

$$
R\in\{A,B\},
$$

with equal probability. In regime A,

$$
Y\mid R=A
\sim
N(-2,1),
$$

while in regime B,

$$
Y\mid R=B
\sim
N(2,1).
$$

A competent forecaster that knows the regime should report the corresponding conditional distribution.

Now consider a deliberately inferior forecaster that ignores $R$ and reports the same unconditional mixture for every case,

$$
F(y)
=
\frac12
\Phi(y+2)
+
\frac12
\Phi(y-2).
$$

Because the regimes occur equally often, the unconditional distribution of $Y$ is exactly $F$. Therefore, if we evaluate the PIT without conditioning on regime,

$$
U
=
F(Y)
$$

is exactly uniform:

$$
U\sim U(0,1).
$$

The forecaster passes the global PIT calibration test perfectly.

It is nevertheless wrong for every individual forecast. In regime A, the correct predictive distribution is

$$
N(-2,1),
$$

not the two-component mixture. In regime B, the correct distribution is

$$
N(2,1).
$$

The reported forecast wastes available information and assigns substantial probability to outcomes associated with the wrong regime.

This construction demonstrates an important limitation of marginal calibration. A forecast can be calibrated after averaging over a population and remain conditionally miscalibrated given variables known at forecast time.

The correct requirement is stronger. If $\mathcal I_t$ denotes the information available when the forecast is issued, then the predictive distribution should approximate

$$
P(Y_t\le y\mid\mathcal I_t),
$$

not merely the unconditional distribution of $Y_t$.

In practice, conditional calibration cannot be checked exhaustively because $\mathcal I_t$ can be high-dimensional. The solution is not to abandon calibration diagnostics. It is to stratify them across variables that matter operationally: forecast horizon, season, demand regime, geography, customer segment, volatility state, model confidence, weather regime or any other information that can plausibly expose systematic error.

This is analogous to model evaluation in supervised learning. Good aggregate accuracy can coexist with severe subgroup failure. A global PIT histogram can look excellent while regime-specific PIT histograms are badly distorted.

## Time series add dependence and horizon structure

The preceding calculations treat forecast cases as if they were independent. Time-series forecasts usually are not.

One-step-ahead PIT values from a correctly specified dynamic model should not only have an approximately uniform marginal distribution. They should also contain no predictable temporal structure relative to the information set. Serial correlation in PIT values, squared transformed PITs or tail indicators can reveal that the marginal forecast distribution is plausible while temporal dependence has been missed.

A common transformation is

$$
Z_t
=
\Phi^{-1}(U_t).
$$

Under a correctly specified continuous forecast model, these transformed PIT values should behave approximately like standard normal innovations under suitable conditions. Autocorrelation, volatility clustering or predictable tail episodes in $Z_t$ indicate remaining structure.

Multi-step forecasting introduces another dimension. A model can be well calibrated one step ahead and poorly calibrated twelve steps ahead. Prediction intervals should generally widen with horizon when uncertainty accumulates. Scores should therefore be reported by horizon rather than averaged into one number that can be dominated by the easiest short-term predictions.

If the forecast is a full path rather than independent marginal distributions at each horizon, evaluating marginal CRPS separately at every horizon still ignores dependence across horizons. Energy scores, variogram scores and pathwise diagnostics can be used for multivariate predictive distributions, although each has its own sensitivities. A forecast of total weekly demand, maximum daily demand and the temporal sequence of demand cannot be evaluated completely from the seven one-day marginals alone.

This distinction matters operationally. Storage, staffing, maintenance and financial decisions often depend on the joint path. Two models can have identical marginal distributions at each horizon and different correlations across horizons, producing different distributions for cumulative or peak quantities.

Probabilistic forecasting is therefore not only about attaching a separate interval to each point prediction. The predictive object should correspond to the decision object.

## Proper scores should be decomposed rather than worshipped

Proper scoring rules are powerful because they make probabilistic forecast comparison coherent. They can also become another leaderboard number if used without diagnosis.

Suppose model A has lower average CRPS than model B. That establishes a preference under the chosen score over the evaluated sample. It does not explain whether A improved because of better calibration, sharper distributions, better tail behaviour or superior performance in one dominant regime.

Score decomposition can help. Reliability-calibration decompositions of CRPS and Brier-type scores separate components associated with calibration and resolution. Quantile-level score plots can show whether one model improves the centre while another improves the tail. Horizon-specific scores can expose where ranking changes with forecast distance.

The choice of score also expresses a loss function. Log score strongly penalizes assigning very low density to realized outcomes. CRPS is less locally sensitive and integrates performance over thresholds. Weighted versions of CRPS can emphasize upper or lower tails when those regions matter more to decisions.

A score should therefore be selected before inspecting which model it favours and should be connected to the forecast quantity of interest. If an organization cares primarily about high-demand events, a tail-weighted proper score can be more informative than an unweighted average dominated by ordinary days.

This is not a licence to invent a metric that crowns a preferred model. The score still needs to be proper or otherwise justified relative to the decision problem. The point is that probabilistic forecast quality is multidimensional, and the evaluation criterion should reflect which dimensions matter.

## Ensemble forecasts need the same discipline

Many probabilistic forecasting systems produce distributions through ensembles rather than explicit densities. Weather prediction is a canonical example, but the same structure appears in bootstrap forecasts, Bayesian posterior predictive simulation and machine-learning ensembles.

If

$$
x_1,\ldots,x_M
$$

are ensemble members, the empirical predictive CDF is

$$
\hat F_M(y)
=
\frac1M
\sum_{m=1}^M
\mathbf 1\{x_m\le y\}.
$$

CRPS can be computed directly from the ensemble representation,

$$
\operatorname{CRPS}
=
\frac1M
\sum_{m=1}^M
|x_m-y|
-
\frac{
1
}{
2M^2
}
\sum_{m=1}^M
\sum_{m'=1}^M
|x_m-x_{m'}|.
$$

The first term measures distance from the observation. The second rewards ensemble concentration. Again, an ensemble does not improve merely by spreading members over a wider range.

Rank histograms provide an ensemble analogue of PIT diagnostics. A U-shaped rank histogram indicates underdispersion, while a dome-shaped histogram suggests overdispersion, under exchangeability assumptions. Bias produces asymmetry.

Finite ensemble size introduces sampling noise into these diagnostics. Members can also be non-exchangeable if they come from structurally different models. The interpretation should therefore follow the ensemble construction rather than relying only on the visual shape.

## Forecast uncertainty should be evaluated where decisions are made

The practical endpoint of probabilistic forecasting is not the PIT histogram or CRPS table. It is a decision made under uncertainty.

Suppose shortage costs are much larger than excess inventory costs. The relevant upper quantiles matter more than median accuracy. Suppose capacity decisions are based on the maximum load over the next week. The joint distribution of the path matters. Suppose a safety rule requires a 99% upper bound. Tail calibration matters more than average interval coverage.

A forecast distribution can be globally respectable and operationally poor if its failures are concentrated exactly where the decision is sensitive.

Evaluation should therefore be conditional not only on statistical regimes but also on decision regimes. Report calibration and scores at relevant horizons. Examine upper and lower quantiles separately. Check rare-event coverage with uncertainty intervals rather than trusting a handful of misses. Compare interval width only after nominal coverage is credible. Evaluate cumulative and path-dependent quantities when those are what the downstream decision uses.

The central principle remains simple. A forecast should say what it knows and no more. Underdispersion claims more precision than the data support. Overdispersion avoids commitment by spreading probability too widely. Marginal calibration can hide information the forecast ignored. A proper score rewards a distribution that concentrates probability where outcomes actually occur without gaining an advantage from strategic vagueness.

In the exact Gaussian example, all three forecasting systems had the correct mean. Point evaluation declared them equivalent. Distributional evaluation did not. The underdispersed forecast covered only 58.9% of outcomes with a nominal 90% interval. The overdispersed forecast covered 99.9% but used an interval twice as wide as necessary. Expected log score and CRPS both selected the correctly calibrated forecast.

The regime-mixture example added a harder lesson. Even a perfectly uniform global PIT did not imply that the forecast used all available information correctly. Calibration has to be judged relative to the conditioning information that defines the forecast problem.

A good probabilistic forecast is therefore not merely one that contains the truth often enough. It is one whose probabilities are credible, whose uncertainty is no wider than necessary, and whose distribution remains reliable in the regimes and horizons that matter.

Calibration tells us whether the probabilities can be trusted. Sharpness tells us whether the forecast is informative. Proper scoring rules force the two to be considered together.

## References

Dawid, A. P. (1984). Present position and potential developments: Some personal views. Statistical theory. The prequential approach. *Journal of the Royal Statistical Society: Series A*, 147(2), 278–292.

Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. *International Economic Review*, 39(4), 863–883.

Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B*, 69(2), 243–268. https://doi.org/10.1111/j.1467-9868.2007.00587.x

Gneiting, T., & Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of Statistics and Its Application*, 1, 125–151. https://doi.org/10.1146/annurev-statistics-062713-085831

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437

Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570.

Laio, F., & Tamea, S. (2007). Verification tools for probabilistic forecasts of continuous hydrological variables. *Hydrology and Earth System Sciences*, 11, 1267–1277.
