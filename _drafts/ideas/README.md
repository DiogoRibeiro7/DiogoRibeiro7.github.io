# Editorial Backlog

This file is the **single source of truth for unpublished article ideas**.

Do not create a new topic-list file when an idea occurs. Add or refine a seed here. A separate Markdown file under `_drafts/ideas/` should exist only when the idea has become a real article draft.

## Current full drafts

These articles exist as complete drafts. Their review status is recorded individually below. Publication is intentionally paused until a separate editorial decision is made.

| Article | File | State | Next decision |
| --- | --- | --- | --- |
| Twenty Seeds Are Not a Phase Diagram | `twenty-seeds-are-not-a-phase-diagram.md` | technical review complete | final editorial pass / publish or hold |
| When the Data Refuse a Point Estimate | `when-the-data-refuse-a-point-estimate.md` | technical review complete | final editorial pass / publish or hold |
| Before Deep Learning, Look at the Geometry | `before-deep-learning-look-at-the-geometry.md` | technical review complete | final editorial pass / publish or hold |
| MCP Is Not Just Tool Calling | `mcp-is-not-just-tool-calling.md` | technical review complete | final editorial pass / publish or hold |
| Reproducibility Is Local | `reproducibility-is-local.md` | technical review complete | final editorial pass / publish or hold |
| The Baseline Is Not a Straw Man | `the-baseline-is-not-a-straw-man.md` | technical review complete | final editorial pass / publish or hold |
| Imputation Accuracy Is Not Inferential Validity | [imputation-accuracy-is-not-inferential-validity.md](imputation-accuracy-is-not-inferential-validity.md) | full draft | technical review / editorial pass |
| Nonparametric Does Not Mean Assumption-Free | [nonparametric-does-not-mean-assumption-free.md](nonparametric-does-not-mean-assumption-free.md) | full draft | technical review / editorial pass |
| Power Is a Surface, Not a Number | [power-is-a-surface-not-a-number.md](power-is-a-surface-not-a-number.md) | full draft; examples verified | technical review / editorial pass |
| The Estimand Comes Before the Test Menu | [the-estimand-comes-before-the-test-menu.md](the-estimand-comes-before-the-test-menu.md) | full draft; examples verified | technical review / editorial pass |
| Stability Is Not Truth | [stability-is-not-truth.md](stability-is-not-truth.md) | full draft; examples verified | technical review / editorial pass |

## Articles prepared for the 2026 publication queue

These articles have dated source files under `_posts/`. Each becomes eligible for the normal site build when its date is reached; appearing on the live site also requires a build and deployment.

| Date | Article | Source |
| --- | --- | --- |
| 2026-09-21 | More Subjects and Longer Trajectories Solve Different Problems | [Article](../../_posts/statistics/2026-09-21-more_subjects_and_longer_trajectories.md) |
| 2026-09-23 | Monitoring Without Labels: What Is Actually Identifiable? | [Article](../../_posts/machine_learning/2026-09-23-monitoring_without_labels_identifiability.md) |
| 2026-09-25 | Numerical Verification Comes Before Optimization | [Article](../../_posts/programming/2026-09-25-numerical_verification_before_optimization.md) |
| 2026-09-28 | What a Wearable Heart Alert Can Actually Tell You | [Article](../../_posts/healthcare/2026-09-28-what_a_wearable_heart_alert_can_tell_you.md) |
| 2026-09-30 | Microwaves Heat Food Without Making It Radioactive | [Article](../../_posts/science_communication/2026-09-30-microwaves_heat_food_without_making_it_radioactive.md) |
| 2026-10-02 | Economic Data Have Two Dates | [Article](../../_posts/economics/2026-10-02-economic_data_have_two_dates.md) |

The final three entries begin focused coverage of health technology, public science communication, and economic/financial data. Their seven-part development contracts are preserved in their source files. The wearable article examines denominators and confirmation timing; the science article addresses the claim that microwave cooking makes food radioactive; the economic-data article reconstructs historical information availability from archived GDP releases.

## Science communication archive additions

These six articles were written and source-checked on 18 September 2026. Their earlier dates place two articles in each of the 2024, 2025, and 2026 archives; each article includes a visible note distinguishing its archive date from its preparation date. Because these dates are in the past, the articles are eligible for the normal build as soon as their source files are included in a deployment.

| Archive date | Article | Source |
| --- | --- | --- |
| 2024-02-15 | Cold Days Still Belong in a Warming Climate | [Article](../../_posts/science_communication/2024-02-15-cold_days_in_a_warming_climate.md) |
| 2024-07-11 | Why Summer Follows Earth’s Tilt | [Article](../../_posts/science_communication/2024-07-11-why_summer_follows_earths_tilt.md) |
| 2025-03-20 | How Antibiotic Resistance Spreads Through Bacteria | [Article](../../_posts/science_communication/2025-03-20-how_antibiotic_resistance_spreads.md) |
| 2025-10-09 | Randomness Does Not Owe Us a Reversal | [Article](../../_posts/science_communication/2025-10-09-randomness_does_not_owe_a_reversal.md) |
| 2026-02-12 | Natural Origin Does Not Establish Safety | [Article](../../_posts/science_communication/2026-02-12-natural_origin_does_not_establish_safety.md) |
| 2026-06-18 | Read the Starting Risk Before the Percentage | [Article](../../_posts/science_communication/2026-06-18-read_the_starting_risk_before_the_percentage.md) |

Each source contains a development contract. The original figures and calculations are reproduced by `assets/viz/generate_science_communication_figures.py`; numerical geometry, hemisphere symmetry, conditional probabilities, and selection behaviour have independent checks in `tests/test_science_communication_figures.py`.

---

# Priority A — development queue

These topics have a precise question, a clear argument, and a natural connection to existing research or software work. Entries with full drafts retain their original development requirements for review; the remaining seeds are candidates to outline next.

## Statistics, inference, and experimental design

### A1. Imputation accuracy is not inferential validity

**State:** [full draft](imputation-accuracy-is-not-inferential-validity.md) — technical review pending

**Core question:** When does reconstructing missing values accurately fail to preserve the downstream estimand or uncertainty?  
**Thesis:** RMSE/MAE on artificially masked values evaluates reconstruction, not whether regression coefficients, uncertainty intervals, treatment effects, or other inferential targets remain valid.  
**Development requirements:** distinguish prediction from inference; compare single imputation, multiple imputation, likelihood-based methods, and sensitivity analysis on at least one controlled example; connect to `imputation-methods`.  
**Avoid:** declaring one imputation method universally best.

### A2. Nonparametric does not mean assumption-free

**State:** [full draft](nonparametric-does-not-mean-assumption-free.md) — technical review pending

**Core question:** What assumptions are hidden when analysts switch from a t-test or ANOVA to a rank-based alternative?  
**Thesis:** changing the test often changes the estimand or null hypothesis as well as the distributional assumptions.  
**Development requirements:** paired and independent examples; Wilcoxon/Mann–Whitney/Kruskal–Wallis interpretation; explicit estimands; simulation showing how conclusions change under unequal shapes or heteroskedasticity.  
**Avoid:** a decision tree based only on normality-test p-values.

### A3. Power is a surface, not a number

**State:** [full draft](power-is-a-surface-not-a-number.md) — examples verified; technical review pending

**Core question:** Why is “80% power” incomplete without specifying an effect, variance, design, decision threshold, and analysis?  
**Thesis:** power should be treated as a function

\[
\pi(\theta,n,\alpha,\mathcal D,\mathcal A)
\]

rather than a property of a study in isolation.  
**Development requirements:** effect-size curves; minimum detectable effect; fixed- versus sequential-design discussion; simulation or analytic examples.  
**Avoid:** reverse-engineering a single sample size without exposing the assumptions.

### A4. The estimand comes before the test menu

**State:** [full draft](the-estimand-comes-before-the-test-menu.md) — examples verified; technical review pending

**Core question:** Why do many “which statistical test should I use?” guides start one step too late?  
**Thesis:** the scientific quantity of interest, sampling design, dependence structure, and loss from error should determine the analysis before a named test is chosen.  
**Development requirements:** examples where apparently interchangeable tests answer different questions; paired versus independent data; mean versus distributional effects; repeated measures.  
**Avoid:** a catalogue of named tests.

### A5. Stability is not truth

**State:** [full draft](stability-is-not-truth.md) — examples verified; technical review pending

**Core question:** Can a clustering solution be highly reproducible even when there are no discrete latent classes?  
**Thesis:** stability measures reproducibility of an algorithmic partition under perturbation; it does not by itself identify a discrete data-generating structure.  
**Development requirements:** continuous-latent negative control; true discrete-class positive control; representation sensitivity; uncertainty over partitions.  
**Research link:** longitudinal clustering work.

## Time series and longitudinal data

### A6. Missing data in time series is an observation-process problem

**State:** seed  
**Core question:** Why is temporal imputation often more than filling holes in a vector?  
**Thesis:** missingness can depend on system state, sensor failure, sampling policy, or user behavior; the observation process can therefore be part of the model.  
**Development requirements:** regular versus irregular sampling; informative missingness; interpolation versus state-space filtering; effect on downstream forecasting and inference.  
**Research link:** `imputation-methods`, sensor work.

### A7. Anomaly, change point, and drift are different hypotheses

**State:** seed  
**Core question:** What exactly changed when a monitoring system raises an alert?  
**Thesis:** a point anomaly, structural break, gradual distribution shift, and performance degradation are different objects and should not share one generic “drift” detector.  
**Development requirements:** formal definitions; toy data with identical-looking alerts but different causes; consequences for retraining and intervention.  
**Research link:** change-point and monitoring work.

### A8. State-space models are a language for uncertainty, not just forecasting

**State:** seed  
**Core question:** Why are latent-state models useful even when the main goal is not prediction?  
**Thesis:** state-space formulation separates latent dynamics, observation noise, missingness, and updating in a way that makes assumptions auditable.  
**Development requirements:** simple local-level model; Kalman recursion; contrast with smoothing/interpolation and black-box sequence models.  
**Avoid:** treating the Kalman filter as a generic imputation trick.

### A9. More subjects and longer trajectories solve different problems

**State:** [dated article, 2026-09-21](../../_posts/statistics/2026-09-21-more_subjects_and_longer_trajectories.md) — calculations, simulation, and figure complete

**Core question:** In longitudinal studies, what does increasing \(n\) solve that increasing \(T\) does not, and vice versa?  
**Thesis:** subject count improves population/mixture estimation while trajectory length improves estimation of subject-level dynamics or features; neither is a substitute for the other.  
**Development requirements:** hierarchical or feature-noise model; simulations; consequences for clustering and personalization.  
**Research link:** longitudinal clustering work.

## Model monitoring and production systems

### A10. Data drift is not model drift

**State:** seed  
**Core question:** What can distribution shift tell us when labels are delayed or absent?  
**Thesis:** a changed input distribution does not imply degraded predictive performance, and unchanged marginal distributions do not guarantee stable conditional relationships.  
**Development requirements:** covariate shift, label shift, concept/conditional shift; examples where drift detector and performance move in opposite directions; monitoring decision table.  
**Avoid:** one-number drift scores.

### A11. Monitoring without labels: what is actually identifiable?

**State:** [dated article, 2026-09-23](../../_posts/machine_learning/2026-09-23-monitoring_without_labels_identifiability.md) — exact examples and figure complete

**Core question:** Which claims about production model quality can be supported before outcomes arrive?  
**Thesis:** unlabeled monitoring can detect changes in observables and invariants, but it cannot generally identify performance degradation without additional assumptions or proxy signals.  
**Development requirements:** formalize observables versus unobservables; delayed-label setting; proxy metrics; bounds or warning states rather than invented performance estimates.

### A12. Retraining is a decision problem, not a cron job

**State:** seed  
**Core question:** When should a model actually be retrained?  
**Thesis:** retraining should compare expected benefit against data, validation, deployment, stability, and regression risk rather than trigger on elapsed time or any detected drift.  
**Development requirements:** cost/loss formulation; challenger evaluation; rollback; minimum evidence gate; examples with false-positive drift alarms.

## Scientific computing and modelling

### A13. Numerical verification comes before optimization

**State:** [dated article, 2026-09-25](../../_posts/programming/2026-09-25-numerical_verification_before_optimization.md) — convergence study and figure complete

**Core question:** Why is a fast solver with no verification story not yet an engineering result?  
**Thesis:** numerical software should be checked against analytic solutions, conservation laws, convergence rates, invariants, or trusted reference implementations before performance claims matter.  
**Development requirements:** one ODE/PDE example; grid refinement or order-of-accuracy test; distinction between unit tests and numerical verification.  
**Research link:** modern Fortran / numerical repos.

### A14. Do not rewrite a numerical kernel until you understand the boundary

**State:** seed  
**Core question:** When is Python–Fortran/C interoperability better than a rewrite?  
**Thesis:** language boundaries should be chosen around stable computational kernels; performance is often dominated by algorithms, memory movement, and vectorized library calls rather than source-language identity.  
**Development requirements:** small benchmark; FFI/f2py/ISO_C_BINDING concepts; maintenance and reproducibility trade-offs.

### A15. Predictive maintenance is a decision problem, not a failure classifier

**State:** seed  
**Core question:** Why can a model with better classification accuracy produce a worse maintenance policy?  
**Thesis:** maintenance decisions depend on asymmetric failure costs, intervention cost, lead time, uncertainty, censoring, and asset state—not merely discrimination metrics.  
**Development requirements:** expected-cost formulation; precision/recall counterexample; remaining-useful-life or hazard perspective; decision threshold.  
**Research link:** predictive-maintenance work.

---

# Priority B — strong research-linked seeds

These are worth keeping but need either data, a sharper theorem/experiment, or literature work before outlining.

## Health, sensing, and ageing

### B1. Personal baseline versus population model in passive health monitoring

When should a health-monitoring system learn an individual’s normal state rather than compare them with a population distribution? Develop around hierarchical models, calibration, adaptation, and false alarms.

### B2. Passive sensing is a measurement model before it is a prediction model

Use RSSI, wearables, or smart-home sensing to show that sensor geometry, missingness, calibration, and observation noise determine what can be inferred before ML enters the picture.

### B3. Frailty is not a label waiting to be predicted

Examine competing operational definitions of frailty and the consequences of turning a multidimensional clinical construct into a binary ML target.

### B4. Smart-home activity recognition under privacy constraints

Frame privacy as an information-design problem: what minimum signals are sufficient for useful inference without reconstructing unnecessarily detailed behavior?

### B5. Survival analysis belongs in more ML systems than we admit

Show how censoring and time-to-event structure arise in failure prediction, churn, disease progression, and remaining useful life; contrast survival estimands with binary classification.

## Economics and public finance

### B6. A Gini coefficient is not a wealth distribution

Compare Lorenz/Gini summaries with tail indices, quantile shares, mobility, and distributional models. Explain how societies with similar Gini values can have materially different structures.

### B7. Pareto tails: where the model starts and where the story usually goes too far

Develop tail estimation, threshold choice, finite-sample uncertainty, and the difference between describing an upper tail and claiming a universal mechanism of inequality.

### B8. GDP–wage decoupling is a time-series identification problem

Develop trend, cointegration, structural breaks, real versus nominal measurement, labor-share definitions, and state-space/ECM approaches. Avoid plotting two indexed series and declaring a structural break by eye.

### B9. Government debt dynamics: start with the accounting identity

Build from the debt accumulation identity before stochastic or causal modelling. Separate arithmetic sustainability, behavioral responses, and policy counterfactuals.

### B10. Overlapping-generations models as an accounting device for intergenerational claims

Use a minimal OLG model to make assumptions about pensions, transfers, debt, and cohort incidence explicit before discussing policy conclusions.

### B11. Tax incidence is not the statutory tax rate

Article seed on behavioral response, pass-through, elasticities, general-equilibrium effects, and why “who pays the tax” is an empirical/model question.

## Climate, energy, and environment

### B12. Better prediction does not remove structural climate uncertainty

Separate parameter uncertainty, scenario uncertainty, model discrepancy, and forecast error. Challenge the vague claim that ML “reduces climate uncertainty.”

### B13. Energy transition as an intertemporal control problem

Develop generation mix, storage, capital turnover, emissions stock, reliability constraints, and discounting in a transparent optimal-control model.

### B14. Digital twins versus scenario models

Clarify what extra empirical coupling is required before a simulation becomes a useful “digital twin”; discuss calibration, state updating, validation, and decision use.

### B15. Precision agriculture under distribution shift

Focus on spatial/temporal transfer, sensor and weather covariate shift, and the limits of models trained on one season/region.

## Traffic and physical systems

### B16. Traffic prediction should respect conservation before adding architecture

Start from flow conservation and classical traffic models, then ask what GNNs/PINNs/learned components add and where physical constraints help or hurt.

### B17. Sensor fusion is uncertainty propagation, not feature concatenation

Use traffic or IoT sensors to develop asynchronous measurement, different noise models, calibration, missingness, and state estimation.

### B18. Adaptive traffic control: prediction and control are different problems

Separate forecasting traffic from choosing interventions; include feedback, delayed effects, counterfactual evaluation, and sim-to-real limitations.

## NLP and representation

### B19. An embedding is a measurement instrument

Treat text embeddings as a representation map with invariances and information loss. Ask which distinctions survive the representation before clustering, regression, or retrieval.

### B20. Topic-model stability before topic-model interpretation

Investigate how preprocessing, random seed, number of topics, representation, and sampling affect apparent themes. Require stability and negative controls before attaching substantive labels.

### B21. Sentiment is not a scalar ground truth

Examine annotation disagreement, domain dependence, pragmatics, class definition, and calibration rather than comparing another list of sentiment classifiers.

## Software and engineering

### B22. The performance bottleneck is often data movement, not arithmetic

Use a numerical kernel to connect cache behavior, memory layout, vectorization, copying, and algorithmic complexity across Python/C/Fortran implementations.

### B23. A benchmark is an experiment

Develop benchmark design around warm-up, variance, hardware, compiler flags, dataset regimes, statistical summaries, and claims that actually follow from measurements.

### B24. A reproducible research package needs failure cases, not only examples

Show why examples demonstrate possibility while tests, negative controls, and adversarial cases define a trustworthy boundary.

---

# Priority C — reserve series

These are useful directions, but they should be developed selectively rather than as generic survey articles.

## Mathematical biographies with a technical hook

A biography enters the queue only when there is a mathematical idea to teach through the person’s work.

- **Mary Cartwright:** nonlinear oscillations, wartime radio modelling, and an early route into chaos.
- **Ingrid Daubechies:** wavelets as a bridge from pure analysis to signal representation.
- **Évariste Galois:** why solvability of equations became a question about symmetry.
- **Georg Cantor:** cardinality, infinity, and what changed in the language of mathematics.
- **Carl Friedrich Gauss:** one technical theme per article rather than a cradle-to-grave biography.
- **Srinivasa Ramanujan:** intuition, proof, and how mathematical claims become durable knowledge.

## Mathematics through modelling

- Fourier analysis as a modelling decision: basis choice, resolution, leakage, and interpretation.
- Differential equations as assumptions about mechanism, not just equations to solve.
- Optimization: objective functions encode values and trade-offs before algorithms begin.
- Probability models as deliberate approximations rather than labels discovered in data.
- Linear algebra as geometry of representation, conditioning, and numerical error.

## Business and decision systems

- BI, prediction, and decision support answer different questions.
- A dashboard is not a decision system.
- Prescriptive analytics requires an intervention model, constraints, and a loss function.
- ROI for analytics should include error costs, adoption, intervention capacity, and maintenance—not model accuracy alone.

---

# Seeds already absorbed or superseded

Do not recreate these as generic articles unless there is a substantially new angle.

- **Correlation versus causation / Granger causality:** the published causal-inference article was substantively revised in September 2026.
- **Shapiro–Wilk versus Anderson–Darling by sample-size threshold:** the published normality article was corrected in September 2026.
- **“Multiple imputation is indefensible”:** the published missing-data article was corrected in September 2026; future work should focus on a narrower inferential question such as A1 or A6.
- **Generic deep learning versus traditional model comparisons:** superseded by the stronger representation/geometry and baseline-first drafts.
- **Generic MCP/tool-calling overview:** superseded by `mcp-is-not-just-tool-calling.md`.
- **Generic reproducibility checklist:** superseded by `reproducibility-is-local.md`.

---

# Development contract for a seed

Before a seed becomes its own file, write down:

1. **Question** — one sentence that can actually be answered.
2. **Claim** — what the article will argue or demonstrate.
3. **Counterclaim** — the strongest reasonable alternative interpretation.
4. **Evidence object** — theorem, derivation, simulation, dataset, benchmark, literature synthesis, or worked example.
5. **Failure case** — where the argument or method stops working.
6. **Reader payoff** — what decision or understanding should change after reading it.
7. **Exclusions** — what tempting adjacent topics will not be covered.

A seed without those pieces stays in this backlog.

---

# Consolidation map

The legacy topic-list files were useful scratchpads, but they mixed duplicate, generic, dated, and high-value ideas. Their useful content has been absorbed here as follows:

| Legacy pool | Canonical destination |
| --- | --- |
| Statistical-test title lists + generic statistics list | A2–A4; B5; development contract |
| Mathematics list | Priority C: Mathematics through modelling |
| Time-series list | A6–A9; B5; B19 where representation is relevant |
| Data/model drift list | A10–A12 |
| Predictive-maintenance list | A15; A7; A12; B5 |
| Health + elderly-care + epidemiology lists | B1–B5 |
| Climate-change list | B12–B15 |
| Traffic-modelling list | B16–B18 |
| Economics + wealth + macroeconomics lists | B6–B11; B13 |
| NLP list | B19–B21 |
| BI/data-science/programming lists | Priority C business topics; B22–B24 |
| C + Fortran lists | A13–A14; B22–B23 |
| Biography + female-mathematician lists | Priority C biography series |

The goal of consolidation is not to preserve every old title. It is to preserve the **best questions** and discard topic-list noise.
