# 合成生存数据评估指标整理

**整理人:** haohong Zheng  
**更新时间:** 2026年4月  
**来源:** Section 1.6, SurvivalGAN论文, synthetic库

---

## 这份文档是干什么的

我们生成了一份合成的生存数据之后, 需要判断它好不好。这份文档整理的就是用什么方法去判断。

指标分两组。第一组是Section 1.6里的5个统计检验。这些检验原来是用来比较治疗组和对照组的生存曲线的, 我们拿来比较真实数据和合成数据。第二组是从SurvivalGAN论文和synthcity库里找到的。别人在评估合成数据的时候实际用过这些指标。有些看的是像不像(fidelity), 有些看的是能不能用(utility)。

有生成数据的code模板后, 这些metrics可以直接接上去跑。

**依赖库:** `pip install lifelines scipy`

---

## 第一部分: Section 1.6的统计检验

这5个检验的逻辑都一样。输入两组(time, event)数据, 输出一个p-value。p大说明两组没有显著差异, 合成数据和真实数据很像。p小说明有显著差异, 合成数据不行。

---

### 1. Log-Rank Test

**出处:** Section 1.6.1, Mantel 1966 [11]

最常见的生存曲线比较方法。生存分析课本里都会讲这个。

它的做法是这样的。在每个有人死的时间点, 它比较实际死亡数和"假如两组一样的话期望多少人死"之间的偏差。把所有时间点的偏差加起来得到一个统计量。

这个检验有一个前提假设叫proportional hazards。意思是两组的危险比一直是固定的, 不随时间变。如果两条生存曲线交叉了, 就是早期A组好晚期B组好那种情况, 早期和晚期的偏差会互相抵消。log-rank在这种情况下可能检测不到差异。

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

**出处:** Section 1.6.2, [12]

log-rank的分层版本。

举个例子。男女的生存曲线本来就不一样。你直接拿全部数据做log-rank的话, 性别会干扰结果。分层版本先把男女分开, 各自做一次log-rank, 再合并。这样就控制了性别这个混杂因素。

在我们的场景里, 如果想看控制了某个变量之后合成数据和真实数据的生存曲线还像不像, 就用这个。

**Python:** lifelines没有直接的分层接口。得手动按变量分组, 各自跑logrank_test, 再自己合并结果。

**R:**

```r
survdiff(Surv(time, event) ~ group + strata(gender), data = df)
```

R里面加个strata()就行, 比Python方便很多。

---

### 3. Weighted Log-Rank Test

**出处:** Section 1.6.3

标准log-rank对所有时间点一视同仁。但有时候你更关心某个时间段的差异。加权版本让你自己选强调哪个时间段。

几种权重:

- **Wilcoxon (Gehan-Breslow):** g(t) = n(t), 用当时的风险人数做权重。早期人多权重大, 等于强调早期差异。
- **Fleming-Harrington:** g(t) = S(t)^p * (1-S(t))^q, 最灵活。p大强调早期, q大强调晚期, 可以自己调。

这个检验跟我们的项目比较相关。slides里提到过, 现有的fidelity指标会在early/late/crossing的时候mislead。加权log-rank就是应对这种情况的。

**Python (lifelines):**

```python
# Wilcoxon, 强调早期
logrank_test(real_time, syn_time, real_event, syn_event,
            weightings='wilcoxon')

# Fleming-Harrington, q=1强调晚期
logrank_test(real_time, syn_time, real_event, syn_event,
            weightings='fleming-harrington', p=0, q=1)
```

**R:**

```r
# rho=1是Wilcoxon, rho=0是标准log-rank
survdiff(Surv(time, event) ~ group, data = df, rho = 1)
```

---

### 4. Kolmogorov-Smirnov Test

**出处:** Section 1.6.4, [13]

跟log-rank完全不同的思路。KS不管什么proportional hazards假设。它就看两条KM曲线之间的最大差距, D = max|S1(t) - S0(t)|。哪个时间点差得最多, 就用那个点的差距做统计量。

KS对曲线交叉的情况比log-rank敏感很多。但它只看最大差异这一个点, 不管这个差异持续多久。

**Python (scipy):**

```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(real_time, syn_time)
```

**R:**

```r
ks.test(real_time, syn_time)
```

注意: scipy的ks_2samp直接比的是time的原始分布, 没考虑censoring。严格来说应该先算KM估计再比。但做快速筛查够用了。

---

### 5. Lin and Xu Test

**出处:** Section 1.6.5, Lin and Xu [14]

比KS更进一步。KS只看最大差异点。Lin and Xu看的是两条KM曲线之间夹的整个面积, Delta = ∫|S1(t) - S0(t)| dt。不仅看差异有多大, 还看差异持续了多久。

这个检验专门为曲线交叉的情况设计的。log-rank在那种情况下会失效, 这个不会。没有现成的Python库, 得按公式自己写。

**Python (手动实现):**

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

# delta越小越好
```

**R:** 没有现成的包, 用同样的逻辑在R里写就行。

---

## 第二部分: 论文和synthetic里的额外指标

这些是SurvivalGAN论文和synthcity tutorial里别人实际用过的。

---

### 6. KM Divergence

**来源:** SurvivalGAN论文, synthcity库

跟Lin and Xu有点像但不完全一样。synthcity里叫survival_km_distance, 算的是真实和合成数据的KM曲线之间的距离。包含三个子指标:

- **optimism:** 有方向的差异。合成数据偏乐观还是偏悲观。
- **abs_optimism:** 绝对差异。不管方向, 只看差多少。
- **sightedness:** 时间轴上的偏移。合成数据的时间尺度有没有偏。

**Python:** synthcity的Benchmarks.evaluate()会自动算。结果里找stats.survival_km_distance.optimism这些字段。

---

### 7. C-index

**来源:** SurvivalGAN论文, synthcity库

这个和前面那些不一样。前面的都是看合成数据和真实数据像不像(fidelity)。C-index看的是合成数据用起来行不行(utility)。

做法: 用合成数据训练一个Cox模型, 然后在真实数据上测试, 看预测的风险排序和实际生存顺序是不是一致。0.5是随机猜, 1.0是完美。超过0.7一般算还行。

这个指标直接对应slides里说的task-aligned evaluation, 就是我们项目的核心问题。

**Python (lifelines):**

```python
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# 在合成数据上训练
cph = CoxPHFitter()
cph.fit(syn_df, duration_col='time', event_col='event')

# 在真实数据上测试
pred = -cph.predict_partial_hazard(real_df)
c = concordance_index(real_df['time'], pred, real_df['event'])
```

---

### 8. Brier Score

**来源:** SurvivalGAN论文, synthcity库

和C-index一样是utility metric。衡量的是模型预测的生存概率和实际结果之间的偏差。包含了校准度(模型说80%的人活, 实际是不是差不多80%)和区分度(能不能把高风险和低风险的人分开)。越低越好。

**Python:** synthcity的Benchmarks.evaluate()自动算。结果里找performance.linear_model.syn_ood.brier_score。

---

### 9. Jensen-Shannon Distance

**来源:** synthcity benchmark

比较两个分布的距离。是KL散度的对称版本。KL有个问题, A到B的距离不等于B到A的。JS解决了这个。可以用在任何变量上, 不只是生存时间。synthcity里对每个feature都会算一个JS distance然后取平均。纯粹的marginal指标。

**Python (scipy):**

```python
from scipy.spatial.distance import jensenshannon
import numpy as np

bins = np.linspace(min(min(real_time), min(syn_time)),
                   max(max(real_time), max(syn_time)), 50)
hist_r, _ = np.histogram(real_time, bins=bins, density=True)
hist_s, _ = np.histogram(syn_time, bins=bins, density=True)
js = jensenshannon(hist_r, hist_s)
# 越小越好, 0表示完全一样
```

---

### 10. Maximum Mean Discrepancy (MMD)

**来源:** synthetic benchmark

这个和前面不一样。它是joint metric, 同时看所有变量, 不是一个一个单独看。把数据映射到一个高维空间, 然后比较两组数据在这个空间里的均值差异。

KS和JS只能一个变量一个变量地比。MMD可以一次比整个数据的联合分布。缺点是不容易解释。你只知道整体不一样, 但不知道具体哪个变量不一样。

**Python:** synthcity的Benchmarks.evaluate()自动算。结果里找stats.max_mean_discrepancy.joint。

---

### 11. Correlation Matrix Difference

**来源:** synthetic benchmark

这个很直观。分别算真实数据和合成数据的相关矩阵, 看两个矩阵差多少。如果真实数据里age和creatinine的相关系数是0.6, 合成数据里变成0.1了, 那就说明合成方法没保留变量之间的关系。

这个指标可以直接证明我们之前说的那种情况: 每个变量的分布都很像, 但变量之间的相关性全乱了。

**Python:**

```python
import numpy as np
corr_real = real_df.corr().values
corr_syn = syn_df.corr().values
diff = np.mean(np.abs(corr_real - corr_syn))
# 越小越好
```

---

## 汇总表

### 来自Section 1.6的统计检验

| 编号 | 方法 | 特点 | Python | R |
|------|------|------|--------|---|
| 1 | Log-Rank Test | 经典, 比例危险下最优 | lifelines | survival::survdiff |
| 2 | Stratified Log-Rank | 控制混杂因素 | lifelines(手动分层) | survdiff + strata() |
| 3 | Weighted Log-Rank | 可强调早期或晚期差异 | lifelines | survdiff(rho=) |
| 4 | KS Test | 对曲线交叉敏感 | scipy.stats | ks.test |
| 5 | Lin and Xu Test | 面积差异, 曲线交叉最佳 | 手动实现 | 手动实现 |

### 来自论文和synthetic的额外指标

| 编号 | 方法 | 类型 | Python |
|------|------|------|--------|
| 6 | KM Divergence | fidelity | synthcity内置 |
| 7 | C-index | utility | lifelines |
| 8 | Brier Score | utility | synthcity内置 |
| 9 | JS Distance | fidelity(marginal) | scipy |
| 10 | MMD | fidelity(joint) | synthcity内置 |
| 11 | Corr Matrix Diff | fidelity(correlation) | numpy |

### 指标分类

1到6, 9, 11是fidelity指标, 看合成数据和真实数据像不像。7和8是utility指标, 看合成数据能不能用来训练模型。10是joint指标, 看整体分布像不像。

slides里的核心论点是fidelity和utility可以不一致。一份合成数据可能fidelity分数很高但utility很差, 也可能反过来。所以两类指标都得有。
