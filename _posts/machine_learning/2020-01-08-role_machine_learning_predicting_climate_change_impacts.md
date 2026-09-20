---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2020-01-08'
excerpt: Machine learning can emulate expensive climate-model components, extract structure from observations, and improve some forecasts, but it does not replace physical climate models or eliminate extrapolation and uncertainty problems.
header:
  image: /assets/images/headers/photo-climate-satellite.jpg
  og_image: /assets/images/headers/photo-climate-satellite.jpg
  overlay_image: /assets/images/headers/photo-climate-satellite.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-climate-satellite.jpg
  twitter_image: /assets/images/headers/photo-climate-satellite.jpg
keywords:
- climate machine learning
- Earth system science
- hybrid modeling
- climate emulation
- weather forecasting
seo_description: A rigorous look at where machine learning helps climate and Earth-system science, where physics remains essential, and why distribution shift and uncertainty dominate long-horizon prediction.
seo_title: 'Machine Learning in Climate Science: Where It Helps and Where It Fails'
seo_type: article
summary: Machine learning is useful in climate science as an emulator, downscaler, pattern detector, and hybrid-model component. This article separates those roles from the stronger claim that data-driven models can replace physical climate models.
tags:
- Climate Science
- Machine Learning
- Scientific Computing
title: 'Machine Learning in Climate Science: Where It Helps and Where It Fails'
---

Machine learning is useful in climate science, but the useful claim is narrower than the popular one.

Climate is governed by physical processes: fluid dynamics, radiation, thermodynamics, phase changes, ocean circulation, land-surface exchange, chemistry, and biology. General circulation models encode these processes numerically.

Machine learning does not make those equations obsolete.

Its strongest role is often to approximate expensive components, infer unresolved structure from observations, or combine physical constraints with flexible statistical models.

The distinction is

$$
\boxed{
\text{physics model}
\neq
\text{machine-learning replacement}
}
$$

and, increasingly,

$$
\boxed{
\text{physics}
+
\text{data-driven approximation}
=
\text{hybrid model}.
}
$$

## Weather and climate are different prediction problems

Weather forecasting asks for the future atmospheric state from the current state over days to weeks.

Climate projection asks about distributions of future states under forcing scenarios over decades or longer.

The two tasks share equations but differ statistically.

A weather model is evaluated on trajectory accuracy over forecast lead time.

A climate model is often evaluated on quantities such as:

- climatological means;
- variability;
- extremes;
- teleconnections;
- energy balance;
- regional distributions;
- response to external forcing.

A model that predicts tomorrow's temperature well is not automatically a valid climate model.

## Emulation

One direct use of machine learning is **emulation**.

Suppose an expensive simulator maps inputs $x$ to output

$$
y=f(x).
$$

An emulator learns

$$
\hat f(x)
\approx
f(x)
$$

at much lower computational cost.

This can be useful for:

- parameter sweeps;
- uncertainty propagation;
- sensitivity analysis;
- ensemble generation;
- inverse problems.

The emulator approximates the simulator.

It does not add physical validity that the simulator lacks.

And it is reliable only over the input region represented in its training data or supported by strong structural assumptions.

## Parameterization of unresolved processes

Climate models cannot resolve every spatial and temporal scale.

Cloud microphysics, convection, turbulence, and land-surface processes may occur below the grid scale.

Traditional models use parameterizations:

$$
\text{resolved state}
\rightarrow
\text{subgrid tendency}.
$$

Machine learning can estimate such mappings from:

- high-resolution simulations;
- observations;
- process models.

This can improve computational efficiency or local fidelity.

But a learned parameterization must remain stable when coupled back into the dynamical model.

A small one-step prediction error can create a large long-term climate bias after repeated feedback.

Offline accuracy is therefore not enough.

## Conservation laws and physical constraints

A generic neural network can violate basic physical constraints.

For example, a model may predict tendencies that do not conserve mass or energy.

Hybrid approaches can constrain outputs so that

$$
\sum_i \Delta E_i = 0
$$

for an energy-conserving subsystem, or incorporate conservation directly into the architecture or loss.

Physical constraints reduce the hypothesis space.

That can improve extrapolation and make failures easier to diagnose.

The goal is not to make the network look more “scientific.”

It is to encode information already known to be true.

## Downscaling

Global climate models operate on spatial grids that are often coarser than the scale required for local impact studies.

Statistical downscaling learns a relationship such as

$$
Y_{\mathrm{local}}
=
g(
X_{\mathrm{large\ scale}}
)
+
\varepsilon.
$$

Machine learning can make $g$ highly flexible.

The central difficulty is stationarity.

A relationship learned under historical climate conditions may not remain valid under a substantially warmer climate.

This is a distribution-shift problem.

Training and deployment distributions differ:

$$
P_{\mathrm{train}}(X,Y)
\neq
P_{\mathrm{future}}(X,Y).
$$

Cross-validation within the historical period cannot prove future-climate validity.

## Extreme events

Extremes are statistically difficult because the events of greatest interest are rare.

A model trained to minimize ordinary mean-squared error can perform well overall while smoothing the tails.

For an extreme threshold $u$, the relevant quantity may be

$$
P(Y>u\mid X),
$$

not the conditional mean.

Evaluation should therefore include:

- tail calibration;
- threshold exceedance rates;
- return-level behavior;
- spatial extent;
- event duration;
- compound extremes.

A high average forecast score can coexist with poor extreme-event performance.

## Sea-level rise

Machine learning can help with components of sea-level analysis such as:

- altimetry gap filling;
- regional pattern extraction;
- ice-sheet emulation;
- surrogate models for expensive simulations;
- bias correction.

But long-term sea-level projection remains a physical inference problem involving thermal expansion, glaciers, ice sheets, land-water storage, and vertical land motion.

A purely historical regression cannot reliably learn responses to forcing states it has never observed.

This is a general lesson:

$$
\boxed{
\text{interpolation}
\neq
\text{climate extrapolation}.
}
$$

## Biodiversity and ecological impacts

Remote sensing and machine learning can map land cover, vegetation traits, habitat fragmentation, species distributions, and disturbance.

These tasks are valuable for climate-impact science.

But ecological response is not identified by image classification alone.

Species distributions depend on:

- climate;
- land use;
- dispersal;
- competition;
- observation bias;
- adaptation;
- human intervention.

A model predicting current species presence from current climate does not automatically estimate the causal effect of future climate change.

## Distribution shift is central

Climate applications routinely violate the standard machine-learning assumption that future data resemble training data.

The system changes because:

- greenhouse-gas concentrations change;
- land use changes;
- observation networks change;
- climate regimes shift;
- extreme-event frequency changes;
- feedbacks enter regions not represented historically.

This makes out-of-distribution behavior a first-order problem.

A random historical train-test split mostly measures interpolation.

It does not test the scenario the model is built for.

## Validation should follow the deployment question

Useful validation designs include:

- training on earlier decades and testing on later decades;
- holding out geographic regions;
- training on some climate-model simulations and testing on others;
- testing across forcing scenarios;
- evaluating coupled long-run stability rather than one-step error.

The split must be scientifically adversarial enough to expose the extrapolation problem.

## Uncertainty has several sources

A climate prediction contains more than ordinary statistical uncertainty.

One useful decomposition is

$$
\text{uncertainty}
=
\text{internal variability}
+
\text{model uncertainty}
+
\text{scenario uncertainty}
+
\text{statistical approximation error}.
$$

Machine learning can reduce one component while leaving the others untouched.

An emulator with very small prediction error does not remove scenario uncertainty.

A calibrated weather model does not remove structural uncertainty in long-run feedbacks.

The uncertainty statement must match the model's role.

## Interpretability is not the main issue

Climate ML discussions often focus on whether neural networks are interpretable.

Interpretability matters, but physical validity matters more.

A model can be easy to visualize and still violate conservation.

A complex model can be scientifically useful if it is:

- stable;
- calibrated;
- physically constrained;
- validated across regimes;
- accompanied by uncertainty;
- reproducible.

The key question is not whether every neuron has a human-readable meaning.

It is whether the model supports the scientific claim being made.

## Hybrid Earth-system modeling

Reichstein and colleagues argued for combining deep learning with process understanding rather than treating the two as competing paradigms.

That direction is still the most convincing framing.

A hybrid model may use:

$$
\frac{dx}{dt}
=
f_{\mathrm{physics}}(x)
+
f_{\mathrm{ML}}(x;\theta),
$$

where the data-driven term represents unresolved or poorly known processes.

The decomposition makes assumptions explicit.

It also allows the known physics to control behavior outside the densest regions of training data.

## Conclusion

Machine learning can make climate science faster and, in some tasks, more accurate.

Its strongest uses are not magical prediction of the future.

They are:

- emulation;
- parameterization;
- downscaling;
- data assimilation support;
- remote-sensing inference;
- hybrid modeling.

The hard problems remain physical and statistical:

$$
\boxed{
\text{distribution shift}
+
\text{rare extremes}
+
\text{feedback}
+
\text{uncertainty}
+
\text{physical constraints}.
}
$$

A climate ML model should be judged by how well it handles those problems, not by whether it beats a baseline on an in-distribution test set.

## References

- Reichstein, M., Camps-Valls, G., Stevens, B., et al. (2019). Deep learning and process understanding for data-driven Earth system science. *Nature*, 566, 195–204. https://doi.org/10.1038/s41586-019-0912-1
- Rolnick, D., Donti, P. L., Kaack, L. H., et al. (2022). Tackling climate change with machine learning. *ACM Computing Surveys*, 55(2), Article 42.
- Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., & Thuerey, N. (2020). WeatherBench: A benchmark data set for data-driven weather forecasting. *Journal of Advances in Modeling Earth Systems*, 12(11), e2020MS002203.
