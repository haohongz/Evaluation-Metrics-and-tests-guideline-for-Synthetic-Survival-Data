# SynSurv_Bench: Chapter 1 Evaluation Metrics Review

> **Author**: Haohong Zheng (郑浩宏)  
> **Group**: KECC, Prof. Kevin He, University of Michigan  
> **Date**: May 2026  
> **Purpose**: Presentation for group meeting — Chapter 1 review + literature survey + proposed new metrics

---

## Table of Contents

- [Part 1: Chapter 1 Overview + My Comments](#part-1-chapter-1-overview--my-comments)
- [Part 2: Literature Review — Evaluation Metrics in Synthetic Survival Papers](#part-2-literature-review--evaluation-metrics-in-synthetic-survival-papers)
- [Part 3: Code Demo — Running All Metrics in Python](#part-3-code-demo--running-all-metrics-in-python)

---

## Part 1: Chapter 1 Overview + My Comments

### What Chapter 1 Covers

Chapter 1 of the Survival Notes introduces survival analysis under the homogeneous independent population assumption. The logical flow is:

**1.1 Failure Time Distributions** — Four equivalent ways to describe "when does an individual experience an event":

| Function | Definition | Intuition |
|----------|-----------|-----------|
| Survival function S(t) | P(D > t) | Probability of surviving beyond time t |
| Density f(t) | -dS(t)/dt | Probability density of event at exactly time t |
| Hazard h(t) | f(t)/S(t) | Instantaneous event rate, given survival to t |
| Cumulative hazard Λ(t) | ∫₀ᵗ h(u)du = -log S(t) | Accumulated risk up to time t |

The key relationship connecting all four: **S(t) = exp(-Λ(t))**. Knowing any one function lets you derive the other three.

For the discrete case: S(t) = ∏(1 - λₖ) — this product structure is exactly the foundation of the Kaplan-Meier estimator.

**1.2 Censoring** — What makes survival data unique. Some individuals leave the study before the event is observed.

- **Observed data**: Xᵢ = min(Dᵢ, Cᵢ), δᵢ = I(Dᵢ ≤ Cᵢ)
- **Right censoring** is the most common type (administrative end, loss to follow-up, withdrawal)
- **Independent censoring assumption**: h(t | at risk) = h(t | at risk AND uncensored). This is the foundation of KM and Cox — and it is generally **unverifiable** from observed data.
- **Likelihood**: L = ∏ [f(Xᵢ)]^δᵢ · [S(Xᵢ)]^(1-δᵢ). Events contribute density; censored observations contribute survival.

**1.3 Truncation** — Even more extreme than censoring: you don't even know the individual exists.

- **Left truncation (delayed entry)**: Individual enters observation only if they survive past a threshold. Common in registry data.
- Left truncation + right censoring is the most common real-world structure.

**1.4 Kaplan-Meier Estimator** — The nonparametric MLE for S(t):

- At each event time tₖ: dₖ events among nₖ at risk
- **KM formula**: Ŝ(t) = ∏_{tₖ≤t} (nₖ - dₖ)/nₖ
- **Greenwood's variance**: Var[Ŝ(t)] = Ŝ(t)² · Σ dₖ/[nₖ(nₖ-dₖ)]
- Log-log transform CI guarantees confidence intervals stay within [0,1]

**1.5 Nelson-Aalen Estimator** — Estimates cumulative hazard Λ(t) directly:

- Λ̂(t) = Σ dₖ/nₖ
- Breslow estimator: S̃(t) = exp(-Λ̂(t)), approximately equals KM in large samples
- More numerically stable than KM in small samples (summation vs multiplication)

**1.6 Comparison of Survival Curves** — Five tests for H₀: S₁(t) = S₂(t):

| Test | Statistic | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| **1. Log-Rank** | Weighted sum of (Observed - Expected) | Most powerful under proportional hazards | Fails when curves cross (+ and - cancel out) |
| **2. Stratified Log-Rank** | Stratified (O-E), combined across strata | Controls for confounders | Requires defining meaningful strata |
| **3. Weighted Log-Rank (Fleming-Harrington)** | Weighted (O-E) with G(ρ,γ) weights | Flexible: ρ=1 emphasizes early, γ=1 emphasizes late | Needs prior knowledge of where differences occur |
| **4. Kolmogorov-Smirnov** | sup\|Ŝ₁(t) - Ŝ₂(t)\| | No PH assumption needed | Sensitive to single-point anomalies |
| **5. Lin & Xu** | ∫\|Ŝ₁(t) - Ŝ₂(t)\|dt | Immune to curve crossing; global measure | Doesn't localize where differences occur |

**Key point for our project**: In synthetic data evaluation, the interpretation is **reversed** — larger p-value = synthetic matches real = GOOD.

### My Comments and Suggestions for Chapter 1

1. **Add a comparison table for the five tests** (like the one above). Currently they are described sequentially, making it hard to see trade-offs at a glance.

2. **Add a numerical example.** A 10-patient toy dataset with hand-calculated KM and log-rank would help readers build intuition, especially newer group members.

3. **Explicitly note the reversed p-value interpretation.** When these tests are used for synthetic data evaluation rather than clinical comparison, the logic flips — this is non-obvious and should be stated clearly.

4. **Gap between §1.2 and §1.6.** Section 1.2 defines censoring as a core concept, but Section 1.6's five tests all focus exclusively on event-time survival curves. There is no test specifically for whether the censoring mechanism is preserved. This is a structural gap that I address in Part 2.

5. **Nelson-Aalen as an alternative.** Consider adding cumulative hazard comparison (Nelson-Aalen based) as a parallel option to KM-based survival curve comparison.

---

## Part 2: Literature Review — Evaluation Metrics in Synthetic Survival Papers

### Overview

I surveyed the main papers on synthetic survival data generation to answer four questions from Prof. He:
1. What evaluation metrics are used in other synthetic survival papers?
2. Do these connect to Chapter 1's classical tests?
3. What metrics does Chapter 1 not include?
4. Can we improve them?

### Papers Surveyed

| Paper | Venue | Link |
|-------|-------|------|
| **SurvivalGAN** (Norcliffe et al.) | AISTATS 2023 | [Paper](https://proceedings.mlr.press/v206/norcliffe23a/norcliffe23a.pdf) |
| **SurvDiff** (Brockschmidt et al.) | arXiv 2025 | [Paper](https://arxiv.org/abs/2509.22352) |
| **Ashhad & Henao** | MLHC 2025 | [Paper](https://arxiv.org/abs/2405.17333) |
| **Synthcity** (Qian et al.) | arXiv 2024 | [Paper](https://arxiv.org/abs/2301.07573) / [GitHub](https://github.com/vanderschaarlab/synthcity) |
| **Heart Failure Synthetic Data** (Puttanawarut et al.) | arXiv 2025 | [Paper](https://arxiv.org/abs/2509.04245) |
| **SynthEval** (Poulsen et al.) | arXiv 2024 | [Paper](https://arxiv.org/abs/2404.15821) |

### All Metrics Found, Organized by Category

#### Category 1: Covariate Fidelity

These measure whether the marginal and joint distributions of patient covariates (age, sex, lab values, etc.) in synthetic data match the real data. **None of these appear in Chapter 1** because Chapter 1 focuses on survival curves only.

| Metric | What It Measures | Used In | Ch.1 Connection |
|--------|-----------------|---------|-----------------|
| **Jensen-Shannon Distance** | Symmetric divergence between per-feature marginal distributions | SurvDiff; Synthcity | ❌ Not in Ch.1 |
| **Wasserstein Distance** | "Earth mover's distance" for overall multivariate distribution | SurvDiff; Synthcity | ❌ Not in Ch.1 |
| **Max Mean Discrepancy (MMD)** | Kernel-based two-sample test in RKHS | Synthcity; SynthEval | ❌ Not in Ch.1 |
| **Detection Score** | Train classifier to distinguish real vs synthetic; 50% accuracy = good | Synthcity | ❌ Not in Ch.1 |
| **t-SNE / PCA Visualization** | Visual overlap in reduced-dimension space | SurvDiff; HF study | ❌ Not in Ch.1 |
| **KS Test (per-feature)** | Tests whether each feature's marginal distribution matches | SynthEval; Synthcity | 🔶 Same method as Ch.1 Test 4, but applied to covariates not survival curves |

#### Category 2: Survival-Specific Fidelity

These directly evaluate whether the survival structure matches between real and synthetic data.

| Metric | What It Measures | Used In | Ch.1 Connection |
|--------|-----------------|---------|-----------------|
| **KM Divergence** | L1 distance between KM curves, normalized by area under real KM | SurvivalGAN (2023) | 🔶 Closely related to **Lin & Xu (Test 5)** — both measure area between curves. KM-Div adds normalization. |
| **Optimism** | Whether synthetic KM is systematically above (optimistic) or below (pessimistic) real KM | SurvivalGAN (2023) | ❌ Novel directional metric. Ch.1 tests are all **non-directional** (they detect difference but not direction). |
| **Short-Sightedness** | Whether synthetic data generates event times shorter than real (KM drops to 0 too early) | SurvivalGAN (2023) | ❌ Captures tail behavior. No Ch.1 test focuses on the right tail. |
| **Log-Rank Test** | Tests H₀: real KM = synthetic KM | Ashhad & Henao (2025) | ✅ **Directly = Ch.1 Test 1** |
| **Censoring Rate Match** | Compare proportion of censored observations | SurvDiff; HF study | ❌ Not in Ch.1 |

#### Category 3: Downstream Utility (TSTR Paradigm)

Train a survival model on synthetic data, test on real data. If synthetic data is good, performance should match training on real data.

| Metric | What It Measures | Used In | Ch.1 Connection |
|--------|-----------------|---------|-----------------|
| **C-index** (Concordance Index) | Rank correlation between predicted risk and observed event times. 0.5=random, 1.0=perfect | Harrell (1982); used in SurvDiff, SurvivalGAN, Ashhad, Synthcity, HF study | ❌ Not in Ch.1 |
| **Integrated Brier Score (IBS)** | Average squared difference between predicted survival probability and observed outcome, integrated over time | Brier (1950); Graf et al. (1999); used in SurvDiff, Ashhad, Synthcity | ❌ Not in Ch.1 |
| **Time-dependent AUC** | At each time t, can the model distinguish who has/hasn't had the event? | SurvBenchmark (Herrmann et al., 2022) | ❌ Not in Ch.1 |

Typical downstream models used in TSTR: CoxPH (linear), XGBoost-Survival (tree-based), DeepHit (neural network). SurvDiff tested across five models, 10 seeds.

#### Category 4: Privacy

| Metric | Used In |
|--------|---------|
| Nearest Neighbor Distance | Synthcity; SynthEval |
| Membership Inference Attack | Synthcity; SurvDiff |
| Differential Privacy (DP-SGD) | SurvDiff — first DP method for synthetic survival data |

> **Note**: Prof. He mentioned sending privacy papers for further review. This section will be expanded.

### Summary: What Chapter 1 Covers vs What It Misses

| Dimension | Chapter 1 Coverage | Gap? |
|-----------|-------------------|------|
| **Survival curve shape** | ✅ All 5 tests | Covered, but no time-localized version |
| **Clinical interpretability** | ❌ Only p-values, no effect size | **GAP** — no metric says "how much" difference |
| **Covariate-outcome relationship** | ❌ Not in scope | **GAP** — doesn't check if "smoking → higher risk" is preserved |
| **Censoring mechanism** | ❌ Defined in §1.2 but not tested in §1.6 | **GAP** — no censoring-specific test |
| **Covariate distributions** | ❌ Not in scope | Covered by literature (JS, Wasserstein, etc.) |
| **Downstream utility** | ❌ Not in scope | Covered by literature (C-index, IBS) |
| **Privacy** | ❌ Not in scope | Partially covered; pending Prof. He's papers |

### My Four Proposed New Metrics

Based on the gaps identified above, I propose four new evaluation metrics:

#### Proposed Metric 1: RMST Difference

**What gap it fills**: Clinical interpretability — Chapter 1's tests only give p-values, not effect sizes.

**Definition**: RMST(τ) = ∫₀^τ S(t)dt (area under KM curve up to τ).

```
RMST_diff = |RMST_real(τ) - RMST_synth(τ)|
```

**Why it's needed**: A p-value of 0.3 tells you "no significant difference," but not "the synthetic data overestimates mean survival by 0.3 years." RMST_diff gives a clinically interpretable effect size in units of time.

**Advantages**: No proportional hazards assumption. Works when curves cross. Clinicians can directly interpret the result.

**Connection to Chapter 1**: Complementary to Lin & Xu (Test 5). Lin & Xu measures ∫|S₁-S₂| (discrepancy area); RMST measures ∫S(t) (average survival time). One tells you "how different," the other tells you "how much time difference."

**Applicable to**: Any survival data with a meaningful truncation time τ — ICU (τ=30 days), cancer (τ=5 years), transplant (τ=10 years).

**Reference**: Royston & Parmar (2013). "Restricted mean survival time: an alternative to the hazard ratio." BMC Med Res Methodol. [Link](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-152)

#### Proposed Metric 2: Hazard Ratio Calibration

**What gap it fills**: Covariate-outcome relationship — Chapter 1's tests only compare overall curves, not whether variable effects are preserved.

**Definition**: Fit Cox model on real → get HR_real per covariate. Fit same Cox on synthetic → get HR_synth.

```
HR_calibration = RMSE(log(HR_real), log(HR_synth))
```

**Why it's needed**: Synthetic data could match the real KM curve perfectly but completely lose the relationship "smoking → 1.5x higher death risk." A model trained on such data would give wrong clinical conclusions.

**Connection to Chapter 1**: Extends beyond Chapter 1 entirely, bridging to Chapter 2 (Cox model). This upgrades evaluation from "distribution comparison" to "model-level comparison."

**Applicable to**: Any survival data with covariates that have clinical meaning (age, sex, treatment, lab values).

**Reference**: Cox (1972) for the model itself. Comparing covariate effects across real/synthetic is used implicitly in Ashhad & Henao (MLHC 2025) but never formalized as a standalone metric.

#### Proposed Metric 3: Censoring Pattern Fidelity

**What gap it fills**: Censoring mechanism — §1.2 defines it, §1.6 doesn't test it.

**Definition**:

```
CensorFidelity = KS(C_real, C_synth) + |CensorRate_real - CensorRate_synth|
```

where C_real and C_synth are the censoring time distributions.

**Why it's needed**: If censoring patterns differ, KM estimates have different precision across time, and downstream methods using IPCW (inverse probability of censoring weighting) will be biased. SurvDiff (2025) explicitly emphasized "faithfully reproducing the censoring mechanism." SurvivalGAN's Short-Sightedness captures one aspect (early truncation); our metric is more comprehensive.

**Connection to Chapter 1**: Fills the gap between §1.2 (defines censoring) and §1.6 (ignores censoring in evaluation).

**Applicable to**: All survival data with censoring (essentially all survival data). Especially important when censoring rate is high (>50%, common in clinical trials).

**Reference**: SurvDiff (Brockschmidt et al., 2025) discusses censoring preservation. SurvivalGAN's Short-Sightedness is a partial version. Our metric is a more complete formalization.

#### Proposed Metric 4: Time-Stratified Lin & Xu

**What gap it fills**: Temporal granularity — the original Lin & Xu gives one number for the entire follow-up.

**Definition**: Partition [0, τ] into K intervals, compute Lin & Xu within each:

```
T_k = ∫_{t_{k-1}}^{t_k} |Ŝ₁(t) - Ŝ₂(t)| dt,  k = 1,...,K
```

**Why it's needed**: Different clinical applications care about different time windows. ICU cares about 0-30 days. Cancer prognosis cares about 5-year survival. A global metric might hide that synthetic data is excellent early but terrible late (or vice versa).

**Connection to Chapter 1**: Direct generalization of Test 5 (Lin & Xu). The original test is the special case K=1.

**Practical output**: A heatmap or bar chart showing fidelity across time windows — immediately reveals where synthetic data is strong or weak.

**Reference**: Lin & Xu (2010) for the original test. The time-stratified extension is my proposal. Inspiration from Fleming-Harrington's weighted approach (different time emphasis), but applied to the Lin & Xu framework instead of log-rank.

### Complete Framework Summary

| Dimension | Ch.1 Tests | Literature Metrics | My Proposals |
|-----------|-----------|-------------------|--------------|
| **Survival curve shape** | Log-Rank, Stratified, Weighted, KS, Lin&Xu | KM Divergence, Optimism, Short-Sightedness | **Time-Stratified Lin&Xu** |
| **Clinical interpretability** | *(only p-values)* | *(none)* | **RMST Difference** |
| **Covariate-outcome relationship** | *(none)* | Implicit via TSTR (C-index, IBS) | **HR Calibration** |
| **Censoring mechanism** | *(none)* | Short-Sightedness (partial) | **Censoring Pattern Fidelity** |
| **Covariate distributions** | *(not in scope)* | JS, Wasserstein, MMD, Detection | *(defer to literature)* |
| **Downstream utility** | *(not in scope)* | C-index, IBS, Time-dep AUC | *(defer to literature)* |
| **Privacy** | *(not in scope)* | NN distance, MIA, DP | *(pending Prof. He's papers)* |

---

## Part 3: Code Demo — Running All Metrics in Python

### Quick Start

```bash
pip install lifelines matplotlib pandas numpy scipy
python synsurvbench_metrics_demo.py
```

### What the Script Does

1. **Simulates** real + synthetic survival data (500 patients each, with age/sex/treatment covariates)
2. **Runs Chapter 1.6's five classical tests** (Log-Rank, Stratified, Weighted, KS, Lin & Xu)
3. **Runs SurvivalGAN's three metrics** (KM Divergence, Optimism, Short-Sightedness)
4. **Runs my four proposed metrics** (RMST Diff, HR Calibration, Censoring Fidelity, Time-Stratified Lin&Xu)
5. **Generates six-panel figure** + summary comparison table

### Demo Results

#### Chapter 1.6: Five Classical Tests

| Test | Statistic | p-value | Quality |
|------|-----------|---------|---------|
| Log-Rank | 0.202 | 0.6527 | ✅ GOOD |
| Stratified Log-Rank (by sex) | 0.341 | 0.8434 | ✅ GOOD |
| Weighted Log-Rank (Wilcoxon) | 0.031 | 0.8599 | ✅ GOOD |
| KS Test | 0.052 | 0.7193 | ✅ GOOD |
| Lin & Xu | area=0.129 | 0.7740 | ✅ GOOD |

Interpretation: all p-values > 0.05, meaning synthetic and real survival curves are not significantly different. All five classical tests pass.

#### Literature Metrics (SurvivalGAN)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| KM Divergence | 0.031 | ✅ Close to 0 = good match |
| Optimism | -0.007 | ✅ Close to 0 = no systematic bias (slightly pessimistic) |
| Short-Sightedness | 0.000 | ✅ Synthetic follow-up is not shorter than real |

#### My Proposed Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| RMST Diff | 0.029 time units | ✅ Synthetic underestimates mean survival by only 0.03 units |
| HR Calibration (RMSE) | 0.063 | ✅ Covariate effects are well preserved |
| Censoring Fidelity | 0.145 | 🔶 Moderate — censoring distribution has some discrepancy |
| Time-Stratified Lin&Xu | see below | 🔶 Reveals hidden temporal pattern |

**HR Calibration Detail:**

| Covariate | HR_real | HR_synth | |log diff| |
|-----------|---------|----------|-----------|
| age | 1.030 | 1.022 | 0.008 |
| sex | 1.455 | 1.489 | 0.023 |
| treatment | 0.604 | 0.672 | 0.106 |

The treatment effect shows the largest discrepancy — the synthetic data underestimates the protective effect of treatment.

**Time-Stratified Lin & Xu Detail:**

| Interval | Area (discrepancy) |
|----------|-------------------|
| [0.0, 2.1) | 0.0217 |
| **[2.1, 4.1)** | **0.0509** ← worst |
| [4.1, 6.2) | 0.0359 |
| [6.2, 8.3) | 0.0207 |

The overall Lin & Xu area (0.129) looks fine, but the stratified view reveals that the [2.1, 4.1) interval has the most discrepancy — more than double the best intervals. If we only looked at the global number, we would miss this.

### Output Figure

The script generates a six-panel figure:

| Panel | Content |
|-------|---------|
| Top-left | KM curves with shaded area between them |
| Top-center | Bar chart of p-values for five classical tests |
| Top-right | SurvivalGAN metrics (KM-Div, Optimism, Short-Sightedness) |
| Bottom-left | HR calibration: real vs synthetic hazard ratios per covariate |
| Bottom-center | Censoring time distributions (real vs synthetic histograms) |
| Bottom-right | Time-Stratified Lin & Xu bar chart showing per-interval discrepancy |

### Code Structure

```
synsurvbench_metrics_demo.py
├── Section 0: Simulate real + synthetic data
├── Part A: Chapter 1.6 five classical tests
│   ├── Log-Rank (lifelines.statistics.logrank_test)
│   ├── Stratified Log-Rank (per-stratum + combine)
│   ├── Weighted Log-Rank / Wilcoxon (lifelines weightings='wilcoxon')
│   ├── KS Test (scipy.stats.ks_2samp on event times)
│   └── Lin & Xu (numerical integration + bootstrap p-value)
├── Part B: Literature metrics
│   ├── KM Divergence (L1 / area under real KM)
│   ├── Optimism (signed area: synthetic - real)
│   └── Short-Sightedness (max event time comparison)
├── Part C: My proposed metrics
│   ├── RMST Difference (area under KM comparison)
│   ├── HR Calibration (Cox regression + RMSE of log HR)
│   ├── Censoring Fidelity (KS on censor times + rate diff)
│   └── Time-Stratified Lin & Xu (per-interval area)
├── Part D: Figure generation (6 panels)
└── Part E: Summary table (CSV output)
```

### Dependencies

```
lifelines >= 0.27    # KM, Cox, log-rank
matplotlib >= 3.5    # Plotting
pandas >= 1.4        # Data handling
numpy >= 1.21        # Numerical
scipy >= 1.7         # KS test, chi2
```

---

## References

1. Norcliffe, A., Cebere, B., Imrie, F., Lio, P., & van der Schaar, M. (2023). [SurvivalGAN: Generating Time-to-Event Data for Survival Analysis.](https://proceedings.mlr.press/v206/norcliffe23a/norcliffe23a.pdf) AISTATS 2023.
2. Brockschmidt, M., et al. (2025). [SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis.](https://arxiv.org/abs/2509.22352) arXiv.
3. Ashhad, M. & Henao, R. (2025). [Generating Accurate Synthetic Survival Data by Conditioning on Outcomes.](https://arxiv.org/abs/2405.17333) MLHC 2025.
4. Qian, Z., et al. (2024). [Synthcity: Facilitating Innovative Use Cases of Synthetic Data.](https://arxiv.org/abs/2301.07573) arXiv.
5. Puttanawarut, C., et al. (2025). [Synthetic Survival Data Generation for Heart Failure Prognosis.](https://arxiv.org/abs/2509.04245) arXiv.
6. Poulsen, A., et al. (2024). [SynthEval: A Framework for Detailed Utility and Privacy Evaluation.](https://arxiv.org/abs/2404.15821) arXiv.
7. Royston, P. & Parmar, M.K. (2013). [Restricted Mean Survival Time: An Alternative to the Hazard Ratio.](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-152) BMC Med Res Methodol.
8. Harrell, F.E. (1982). Evaluating the Yield of Medical Tests. JAMA. *(C-index)*
9. Brier, G. (1950). Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review. *(Brier Score)*
10. Graf, E., et al. (1999). Assessment and Comparison of Prognostic Classification Schemes for Survival Data. Statistics in Medicine. *(Integrated Brier Score)*
11. Herrmann, M., et al. (2022). [SurvBenchmark: Comprehensive Benchmarking Study of Survival Analysis Methods.](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac071/6652188) GigaScience.
12. Lin, D.Y. & Xu, J. (2010). On the Area Between Two Survival Curves.
