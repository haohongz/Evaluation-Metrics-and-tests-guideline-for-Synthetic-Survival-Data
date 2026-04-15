# Evaluation Metrics for Synthetic Survival Data

**Author:** Haohong  
**Last updated:** April 2026  
**Source:** Section 1.6 of *Statistical Learning and Knowledge Distillation for Survival Analysis* (He, Spring 2026), SurvivalGAN paper, and synthcity library.

---

## What this document covers

We need a way to measure how good synthetic survival data is. This document collects the metrics we plan to use. 

The metrics fall into two groups. The first group comes from Section 1.6 of Professor He's draft. These are classical statistical tests that compare two survival curves. They were originally designed for comparing treatment vs. control groups. We use them to compare real data vs. synthetic data instead. 

The second group comes from the SurvivalGAN paper and the synthcity library. These are metrics that other researchers have actually used when evaluating synthetic data. Some of them measure fidelity (does the synthetic data look like the real data). Others measure utility (can we train a useful model on the synthetic data).

**Dependencies:** `pip install lifelines scipy`

---

## Part 1: Statistical Tests from Section 1.6

All five tests share the same logic. You feed in two sets of (time, event) data. The test returns a p-value. A large p-value means the two groups have no significant difference, so the synthetic data is similar to the real data. A small p-value means there is a significant difference, so the synthetic data does not match.

---

### 1. Log-Rank Test

**Source:** Section 1.6.1. Reference: Mantel, 1966 [11].

This is the most common method for comparing survival curves. Every survival analysis textbook covers it. 

The test works like this. At each time point where someone dies, it compares the actual number of deaths to the expected number of deaths under the assumption that both groups are the same. It adds up these differences across all time points to get a test statistic.

One important assumption is proportional hazards. This means the hazard ratio between the two groups stays constant over time. If the two survival curves cross each other (group A is better early on, group B is better later), the early and late differences cancel out. The log-rank test may fail to detect any difference in that case.

**Python (lifelines):**

```python
from lifelines.statistics import logrank_test

result = logrank_test(
    real_time, syn_time,
    event_observed_A=real_event,
    event_observed_B=syn_event
)
print(result.test_statistic, result.p_value)
```

**R (survival):**

```r
survdiff(Surv(time, event) ~ group, data = df)
```

---

### 2. Stratified Log-Rank Test

**Source:** Section 1.6.2. Reference: [12].

This is the stratified version of the log-rank test. 

Here is an example. Male and female patients may have different survival curves by default. If you run a regular log-rank test on the combined data, the gender effect will interfere with your comparison of real vs. synthetic data. The stratified version splits the data by gender first. It runs a separate log-rank test within each gender group. Then it combines the results. This controls for the confounding effect of gender.

In our project, we use this when we want to check whether the synthetic data matches the real data after controlling for some variable.

**Python:** lifelines does not have a built-in stratified interface. You need to manually split by strata, run logrank_test on each stratum, and combine the results yourself.

**R:**

```r
survdiff(Surv(time, event) ~ group + strata(gender), data = df)
```

R handles this much more easily than Python. You just add `strata()` and it works.

---

### 3. Weighted Log-Rank Test

**Source:** Section 1.6.3.

The standard log-rank test treats all time points equally. But sometimes you care more about differences at certain time periods. The weighted version lets you choose which time period to emphasize.

Common weight functions:

- **Wilcoxon (Gehan-Breslow):** g(t) = n(t). This uses the number of people at risk as the weight. Early time points have more people at risk, so they get more weight. This emphasizes early differences.
- **Fleming-Harrington:** g(t) = S(t)^p \* (1-S(t))^q. This is the most flexible option. A larger p emphasizes early differences. A larger q emphasizes late differences. You can tune these based on what you need.

This test is relevant to our project. Feiyang's slides mention that existing fidelity metrics "can disagree and mislead (early/late/crossing)." The weighted log-rank test is designed to handle exactly this problem.

**Python (lifelines):**

```python
# Wilcoxon
logrank_test(real_time, syn_time, real_event, syn_event,
            weightings='wilcoxon')

# Fleming-Harrington, q=1 emphasizes late differences
logrank_test(real_time, syn_time, real_event, syn_event,
            weightings='fleming-harrington', p=0, q=1)
```

**R:**

```r
# rho=1 is Wilcoxon, rho=0 is standard log-rank
survdiff(Surv(time, event) ~ group, data = df, rho = 1)
```

---

### 4. Kolmogorov-Smirnov Test

**Source:** Section 1.6.4. Reference: [13].

This test takes a completely different approach from log-rank. It does not assume proportional hazards. It just looks at the maximum gap between the two KM curves: D = max|S1(t) - S0(t)|. Whichever time point has the biggest difference, that difference becomes the test statistic.

The KS test is much more sensitive to curve crossing than log-rank. But it only looks at the single point of maximum difference. It does not consider how long that difference lasts.

**Python (scipy):**

```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(real_time, syn_time)
```

**R:**

```r
ks.test(real_time, syn_time)
```

**Note:** scipy's ks_2samp compares the raw time distributions directly. It does not account for censoring. Strictly speaking, you should compute KM estimates first and then compare those. But for a quick check, ks_2samp is good enough.

---

### 5. Lin and Xu Test

**Source:** Section 1.6.5. Reference: Lin and Xu [14].

This test goes further than KS. The KS test only looks at the single point of maximum difference. Lin and Xu look at the total area between the two KM curves: Delta = integral of |S1(t) - S0(t)| dt. This captures both the size of the difference and how long it lasts.

This test was specifically designed for the case where survival curves cross. Log-rank fails in that situation. Lin and Xu does not.

There is no ready-made Python package for this. You have to implement it yourself based on the formula.

**Python (manual implementation):**

```python
from lifelines import KaplanMeierFitter

kmf_real = KaplanMeierFitter()
kmf_real.fit(real_time, event_observed=real_event)
kmf_syn = KaplanMeierFitter()
kmf_syn.fit(syn_time, event_observed=syn_event)

all_times = sorted(set(list(real_time) + list(syn_time)))
tau = min(max(real_time), max(syn_time))

delta = 0
for i in range(len(all_times) - 1):
    t = all_times[i]
    t_next = all_times[i + 1]
    if t >= tau:
        break
    s_real = kmf_real.predict(t)
    s_syn = kmf_syn.predict(t)
    delta += abs(s_real - s_syn) * (t_next - t)

# smaller delta = more similar curves
```

**R:** No ready-made package. Same logic can be implemented in R.

---

## Part 2: Additional Metrics from Papers and synthcity

These metrics come from the SurvivalGAN paper and the synthcity library. Other researchers have used these when evaluating synthetic survival data. Feiyang specifically asked us to include KM divergence.

---

### 6. KM Divergence

**Source:** SurvivalGAN paper, synthcity library.

Feiyang specifically mentioned this one. It measures the distance between the real and synthetic KM curves. The synthcity library calls it `survival_km_distance`. It computes three sub-metrics:

- **optimism:** directional difference. Is the synthetic data more optimistic or more pessimistic than the real data?
- **abs_optimism:** absolute difference. How far apart are the two curves, regardless of direction?
- **sightedness:** temporal shift. Is the synthetic data's time scale shifted compared to the real data?

**Python:** synthcity's `Benchmarks.evaluate()` computes this automatically. Look for `stats.survival_km_distance.optimism`, `stats.survival_km_distance.abs_optimism`, and `stats.survival_km_distance.sightedness` in the results.

---

### 7. C-index (Concordance Index)

**Source:** SurvivalGAN paper, synthcity library.

This metric is different from all the previous ones. The previous metrics measure fidelity (does the synthetic data look like the real data). C-index measures utility (can we actually use the synthetic data for something useful).

The procedure is: train a Cox model on the synthetic data, then test it on the real data. Check whether the model's predicted risk ranking matches the actual survival order. A C-index of 0.5 means random guessing. A C-index of 1.0 means perfect prediction. Anything above 0.7 is generally considered acceptable.

This metric directly addresses the core question of our project. Feiyang's slides call it "task-aligned evaluation."

**Python (lifelines):**

```python
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# train on synthetic data
cph = CoxPHFitter()
cph.fit(syn_df, duration_col='time', event_col='event')

# test on real data
pred = -cph.predict_partial_hazard(real_df)
c = concordance_index(real_df['time'], pred, real_df['event'])
```

---

### 8. Brier Score

**Source:** SurvivalGAN paper, synthcity library.

This is another utility metric, like C-index. It measures the gap between the model's predicted survival probabilities and the actual outcomes. It combines calibration (if the model says 80% of people survive, do roughly 80% actually survive?) and discrimination (can the model tell high-risk and low-risk patients apart?). Lower is better.

**Python:** synthcity's `Benchmarks.evaluate()` computes this automatically. Look for `performance.linear_model.syn_ood.brier_score` in the results.

---

### 9. Jensen-Shannon Distance

**Source:** synthcity benchmark.

This measures the distance between two distributions. It is a symmetric version of KL divergence. KL divergence has a problem: the distance from A to B is not the same as the distance from B to A. JS distance fixes that. You can use it on any variable, not just survival time. In synthcity, JS distance is computed for each feature separately and then averaged. It is a purely marginal metric.

**Python (scipy):**

```python
from scipy.spatial.distance import jensenshannon
import numpy as np

bins = np.linspace(min(min(real_time), min(syn_time)),
                   max(max(real_time), max(syn_time)), 50)
hist_r, _ = np.histogram(real_time, bins=bins, density=True)
hist_s, _ = np.histogram(syn_time, bins=bins, density=True)
js = jensenshannon(hist_r, hist_s)
# smaller is better, 0 means identical
```

---

### 10. Maximum Mean Discrepancy (MMD)

**Source:** synthcity benchmark.

This is different from the metrics above. It is a joint metric. It looks at all variables at the same time, not one by one. It maps the data into a high-dimensional space and compares the mean of the two groups in that space.

KS test and JS distance can only compare one variable at a time. MMD compares the entire joint distribution at once. The downside is that it is hard to interpret. You know the two datasets are different overall, but you do not know which specific variable is causing the difference.

**Python:** synthcity's `Benchmarks.evaluate()` computes this automatically. Look for `stats.max_mean_discrepancy.joint` in the results.

---

### 11. Correlation Matrix Difference

**Source:** synthcity benchmark.

This one is straightforward. Compute the correlation matrix for the real data. Compute the correlation matrix for the synthetic data. Compare the two matrices element by element. If the real data has a correlation of 0.6 between age and creatinine, and the synthetic data has 0.1, then the synthetic method did not preserve the relationship between these two variables.

This metric can directly show a situation where every single variable's distribution looks correct, but the relationships between variables are completely wrong.

**Python:**

```python
import numpy as np
corr_real = real_df.corr().values
corr_syn = syn_df.corr().values
diff = np.mean(np.abs(corr_real - corr_syn))
# smaller is better
```

---

## Summary Table

### Statistical Tests (from Section 1.6)

| # | Method | Strength | Python | R |
|---|--------|----------|--------|---|
| 1 | Log-Rank Test | Classic, best under proportional hazards | lifelines | survival::survdiff |
| 2 | Stratified Log-Rank | Controls for confounders | lifelines (manual split) | survdiff + strata() |
| 3 | Weighted Log-Rank | Can emphasize early or late differences | lifelines | survdiff(rho=) |
| 4 | KS Test | Sensitive to curve crossing | scipy.stats | ks.test |
| 5 | Lin and Xu Test | Area difference, best for curve crossing | Manual implementation | Manual implementation |

### Additional Metrics (from papers and synthcity)

| # | Method | Type | Python |
|---|--------|------|--------|
| 6 | KM Divergence | Fidelity | synthcity built-in |
| 7 | C-index | Utility | lifelines |
| 8 | Brier Score | Utility | synthcity built-in |
| 9 | JS Distance | Fidelity (marginal) | scipy |
| 10 | MMD | Fidelity (joint) | synthcity built-in |
| 11 | Corr Matrix Diff | Fidelity (correlation) | numpy |

### Metric categories

Metrics 1 through 6, 9, and 11 are **fidelity metrics**. They check whether the synthetic data looks like the real data. Metrics 7 and 8 are **utility metrics**. They check whether the synthetic data is useful for downstream tasks. Metric 10 is a **joint metric**. It checks the overall distribution.

Feiyang's slides argue that fidelity and utility can disagree. A synthetic dataset might score well on fidelity but poorly on utility, or the other way around. That is why we need both types.
