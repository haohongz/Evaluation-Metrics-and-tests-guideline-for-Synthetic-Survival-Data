# 三个新Evaluation方向 完整解析

> 每个方向都按统一框架讲：问题→想法→本质→做法→好处→局限→传统文献→合成数据用法→novelty

---

# Direction 1: 比较平均生存时间（用RMST）

## 发现了什么问题

Chapter 1.6的五个test跑完，每个给你一个p值。比如Log-Rank p=0.3。

何老师问你："所以呢？合成数据到底好不好？差了多少？"

你说不出来。因为p值只回答"有没有显著差异"这一个是/否问题，不回答"差了多少"。

而且p值有一个很大的问题：**受样本量影响**。1000个人的时候，两条KM曲线差了一点点也能给出p=0.01（显著）。100个人的时候，差了很多也可能p=0.2（不显著）。所以p值不能跨数据集比较。

## 怎么想到的

我需要一个**不受样本量影响、有临床意义、直接说"差了多少"**的量。

在传统survival analysis里，RMST（Restricted Mean Survival Time）就是这样一个东西。它在2013年被Royston & Parmar正式推广，作为hazard ratio的替代品——因为HR也需要PH假设，RMST不需要。

RMST本身是一个成熟的、被广泛使用的传统survival概念。但**从来没有人把它用在合成数据评估上**。

## 本质是什么

画一条KM曲线，从时间0到截断时间τ。曲线下面的面积就是RMST。

临床意思：**"在τ时间内，平均还能活多久"**。比如τ=5年，RMST=3.8年，意思是"平均在5年内能活3.8年"。

然后在真实数据和合成数据上各算一次RMST，比差：

```
RMST_diff = |RMST_real(τ) - RMST_synth(τ)|
```

底层工具：KM估计（Chapter 1.4已有） + 数值积分（梯形法则）。没有任何新的统计方法。

## 具体怎么做

```python
# 伪代码
kmf_real = KaplanMeierFitter().fit(real_time, real_event)
kmf_synth = KaplanMeierFitter().fit(synth_time, synth_event)

tau = min(real_time.quantile(0.9), synth_time.quantile(0.9))
t_grid = np.linspace(0, tau, 1000)

rmst_real = np.trapz(kmf_real.predict(t_grid), t_grid)
rmst_synth = np.trapz(kmf_synth.predict(t_grid), t_grid)

rmst_diff = abs(rmst_real - rmst_synth)
```

输出：一个数字。比如"合成数据平均多估了0.3年的生存时间"。

## 好处

1. **临床可解释**："差了0.3年"比"p=0.15"有意义得多。医生一听就懂。
2. **不依赖PH假设**：曲线交叉也能用。Log-Rank在交叉时废掉，RMST不怕。
3. **不受样本量影响**：差0.3年就是差0.3年，不管100人还是10000人。
4. **传统文献大量支撑**：Royston & Parmar (2013)被引用2000+次，RMST是被广泛认可的quantity。

## 局限

1. **需要选τ**：τ太大，KM尾巴不稳定会影响结果。τ太小，丢失晚期信息。建议用两组观测时间的90th percentile的较小值。
2. **交叉时可能misleading**：两条曲线来回交叉，上面多出来的面积和下面少的面积互相抵消，RMST_diff可能很小。但实际形状差异很大。所以**不能单独用，要跟Lin & Xu配合**。
3. **只看整体平均**：不告诉你哪个时间段差距大。需要Time-Stratified Lin & Xu补充。

## 传统文献

| Paper | Year | 说了什么 |
|---|---|---|
| Royston & Parmar, BMC Med Res Methodol | 2013 | RMST作为HR的替代，不需PH假设。Link: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-152 |
| Huang & Tian, Comput Math Methods Med | 2022 | RMST vs 传统方法的对比，PH不成立时RMST更好。Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC9812622/ |
| Tian et al., PMC | 2020 | τ的选取讨论。Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8078843/ |
| Huang & Kuan, Am J Epidemiol | 2026 | τ影响power和Type I error。Link: https://academic.oup.com/aje/article/195/1/32/8019555 |

## 在合成数据evaluation上有没有人用过

**没有。** 检索了SurvivalGAN (2023)、SurvDiff (2025)、Ashhad & Henao (2025)、Synthcity (2024)——全部只用p值或divergence score，没有一篇用RMST。

**这就是novelty**：把传统survival里成熟的RMST概念，应用到合成数据evaluation这个新场景。

---

# Direction 2: 比较变量效应（用Cox HR）

## 发现了什么问题

五个test只比较两条**整体**KM曲线。但整体匹配不等于变量关系保留。

极端例子：真实数据里"年龄大→死得快"、"女性→死得慢"。合成数据里恰好反过来——"年龄大→死得慢"、"女性→死得快"。但因为两个效应互相抵消，整体KM曲线跟真实的**一模一样**。

五个test全pass。KM Divergence接近0。SurvivalGAN的三个metric全绿。一切看起来完美。

但用这种数据训练一个模型，模型会告诉医生"年龄大的人风险低"——完全相反的临床结论。

## 怎么想到的

我想：有没有一种方法可以直接检查"每个变量的效应有没有保留"？

Cox模型就是做这件事的——它输出每个变量的Hazard Ratio，量化"这个变量让风险高了多少/低了多少"。

那我就在两边各跑一次Cox，比较HR。如果两边的HR一致，说明变量效应保留了。不一致，说明丢了。

## 本质是什么

Cox回归就是一个模型：h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

每个βᵢ对应一个变量的效应。exp(βᵢ) = Hazard Ratio：
- HR > 1：这个变量让风险升高（风险因素）
- HR < 1：这个变量让风险降低（保护因素）  
- HR = 1：这个变量没有影响

两边各跑一次Cox，得到HR_real和HR_synth，然后算：

```
RMSE(log(HR_real), log(HR_synth))
```

用log是因为HR是乘性的——HR=2（风险翻倍）和HR=0.5（风险减半）应该对称。取log后变成+0.69和-0.69，算RMSE才公平。

底层工具：Cox模型（Cox 1972，人人都会跑） + RMSE（基础数学）。没有任何新方法。

## 具体怎么做

```python
# 伪代码
from lifelines import CoxPHFitter

covs = ['age', 'sex', 'treatment']

cox_real = CoxPHFitter().fit(df_real[['time','event']+covs], 'time', 'event')
cox_synth = CoxPHFitter().fit(df_synth[['time','event']+covs], 'time', 'event')

hr_real = cox_real.hazard_ratios_
hr_synth = cox_synth.hazard_ratios_

# 逐变量对比
for cov in covs:
    print(f"{cov}: HR_real={hr_real[cov]:.3f}, HR_synth={hr_synth[cov]:.3f}")

# 总体RMSE
rmse = np.sqrt(np.mean((np.log(hr_real) - np.log(hr_synth))**2))
```

输出：一个HR对比表 + 一个总体RMSE数字。

| 变量 | HR_real | HR_synth | 差多少 |
|---|---|---|---|
| 年龄 | 1.03 | 1.02 | 很小 ✅ |
| 性别(男) | 1.50 | 1.48 | 很小 ✅ |
| 治疗 | 0.60 | 0.90 | **很大** ❌ |
| **总体RMSE** | | | 0.25 |

一眼看出：治疗的保护效应从0.6（风险降40%）变成了0.9（风险只降10%）——合成数据严重低估了治疗效果。

## 好处

1. **抓hidden failure**：整体KM匹配但变量关系丢失——这是五个test、KM Divergence、RMST全都抓不到的。只有直接比较HR才能发现。
2. **可解释**：每个变量的HR差异一目了然，知道哪个变量的效应丢了。
3. **工具成熟**：Cox模型是survival analysis的基础工具，所有软件都支持。
4. **对下游影响直接**：如果HR不对，用合成数据训练的模型给出的临床结论就是错的。

## 局限

1. **依赖Cox的PH假设**：如果PH不成立，两边的HR都不准确，比较不准确的HR意义有限。
2. **只能检查线性效应**：Cox假设log-hazard是covariates的线性函数。非线性效应抓不到。
3. **变量太多时不稳定**：MIMIC有598个feature，直接跑Cox可能不稳定。可以只选top几个临床重要的变量（年龄、性别、主要诊断）。
4. **未来改进方向**：用random survival forest或time-varying coefficient model替代Cox，去掉PH限制。

## 传统文献

| Paper | Year | 说了什么 |
|---|---|---|
| Cox, JRSS-B | 1972 | Cox模型原始论文 |
| Austin, Statistics in Medicine | 2012 | 用Cox生成模拟数据，评估时也可用Cox。Link: https://onlinelibrary.wiley.com/doi/10.1002/sim.5452 |

## 在合成数据evaluation上有没有人用过

**隐含使用但没有显式做**。Ashhad & Henao (2025)在TSTR框架里用CoxPH训练再测试——间接比较了covariate effects。但他们没有**显式地**把两边的HR拿出来逐变量对比。

SurvDiff (2025)也没有。SurvivalGAN (2023)也没有。

**novelty**：把"比较两边Cox HR"从隐含变成显式，作为一个独立的evaluation方向。

---

# Direction 3: 比较删失分布（Censoring Comparison）

## 发现了什么问题

读Chapter 1时发现一个结构性的矛盾：

- Section 1.2花了一整节讲censoring多重要——它是生存数据的独特特征
- Section 1.6的五个test全部只比较event time的生存曲线——没有一个检查censoring

在传统临床研究里这合理——censoring是噪音，你要克服它不是比较它。

但在合成数据里不一样。合成数据必须同时生成event time **和** censoring time。如果所有合成病人都出事了、没有一个被删失的，一看就是假的。**Censoring是数据生成过程的一部分，必须被验证。**

## 怎么想到的

我问自己：如果censoring pattern不对，会出什么问题？

1. **KM精度不对**：KM在censoring多的时间段精度差（risk set小、Greenwood variance大）。如果合成数据的censoring集中在不同的时间段，KM的精度分布就不同。

2. **IPCW出偏差**：很多survival方法用IPCW（inverse probability of censoring weighting）来调整censoring的影响。IPCW需要估计censoring的分布。如果合成数据的censoring分布跟真实不同，IPCW的权重就是错的。

3. **模型训练受影响**：在合成数据上训练的模型学到的是错误的censoring pattern，拿到真实数据上用就会出偏差。

然后我想：检查censoring分布的工具是什么？**KS检验**——本来就是比较两个分布的标准工具。我只是把比较对象从event time换成了censoring time。

另外从Huang & Tian (2022)那篇传统论文借鉴了一个做法：**把censored当event来跑Log-Rank**——传统survival里就有人这么做来检验censoring mechanism的同质性。

## 本质是什么

三部分加起来：

1. **Censoring rate差异**：|CensorRate_real - CensorRate_synth|。真实30%删失，合成50%删失，差了20个百分点——最直观的检查。

2. **Censoring时间分布的KS检验**：把两边被删失的人的删失时间拿出来，用KS检验比较分布。比例可能一样（都是30%），但分布不同（一个集中在早期，一个均匀分布）。

3. **Censoring的Log-Rank**（从Huang & Tian 2022借鉴）：把原始数据里的event indicator反转——把censored (δ=0)当成event (δ=1)，把event (δ=1)当成censored (δ=0)。然后跑Log-Rank。这相当于"把censoring当成我们关心的事件来比较"。

底层工具：KS检验（标准） + 比例差（简单算术） + Log-Rank（Chapter 1.6 Test 1，换了比较对象）。没有任何新方法。

## 具体怎么做

```python
# 伪代码

# 1. Censoring rate差异
rate_real = 1 - df_real['event'].mean()  # e.g., 0.30
rate_synth = 1 - df_synth['event'].mean()  # e.g., 0.35
rate_diff = abs(rate_real - rate_synth)  # 0.05

# 2. Censoring时间分布KS
censor_times_real = df_real.loc[df_real['event']==0, 'time']
censor_times_synth = df_synth.loc[df_synth['event']==0, 'time']
ks_stat, ks_p = ks_2samp(censor_times_real, censor_times_synth)

# 3. Censoring的Log-Rank（反转event indicator）
lr_censor = logrank_test(
    df_real['time'], df_synth['time'],
    event_observed_A = 1 - df_real['event'],   # 反转！
    event_observed_B = 1 - df_synth['event']
)
```

输出三个数字：
- Censoring rate差异：5个百分点
- Censoring时间KS：D=0.08, p=0.15
- Censoring Log-Rank：p=0.25

## 好处

1. **填补Chapter 1的结构性gap**：1.2定义了censoring但1.6不测它。这个direction直接填补。
2. **Censoring是合成数据生成过程的一部分**：不像传统里只是噪音，在合成数据里必须验证。
3. **传统文献有支撑**：Huang & Tian (2022)的"反转Log-Rank"做法是现有的传统方法。
4. **简单但之前没人做**：工具全是现成的，但在合成survival evaluation领域没有人把它们组合起来用。

## 局限

1. **只检查marginal censoring分布**：不检查censoring跟covariates的关系。比如"年轻人更容易被删失"这种conditional pattern抓不到。
2. **如果censoring是informative的**：marginal比较可能不够。未来可以按subgroup分别检查。
3. **三个数字怎么合并成一个score**：目前是加法（rate差 + KS），权重怎么选没有理论依据。

## 传统文献

| Paper | Year | 说了什么 |
|---|---|---|
| Huang & Tian, Comput Math Methods Med | 2022 | **把censored当event跑Log-Rank来检验censoring同质性**。原文："The homogeneity of censoring mechanism assumption was evaluated by KM curve or log-rank test, in which the censoring parameter was specified as 1 instead of 0." Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC9812622/ |
| Lagakos, Biometrics | 1979 | censoring mechanism对survival estimation的影响 |

## 在合成数据evaluation上有没有人用过

**部分使用**：
- SurvivalGAN (2023)：Short-Sightedness只看"合成数据follow-up是不是太短了"——只是censoring的一个方面
- SurvDiff (2025)：提到"preserving the censoring mechanism"但没给具体metric

**没有人做完整的censoring分布比较**。特别是"反转Log-Rank"这个做法，从来没有人用在合成数据evaluation上。

**novelty**：把传统survival里的censoring检验方法（Huang & Tian 2022），应用到合成数据evaluation。

---

# 三个方向的整体逻辑

```
一份好的合成生存数据应该满足什么？

1. 生存曲线形状匹配 → Chapter 1的五个test已覆盖 ✅
2. 差了多少要可量化 → RMST比较（Direction 1）🆕
3. 变量效应要保留   → Cox HR比较（Direction 2）🆕  
4. 删失模式要逼真   → Censoring比较（Direction 3）🆕
5. 哪个时间段有问题 → 分段Lin & Xu ✅（已在slides里）
```

这四个🆕方向（包括分段Lin & Xu）跟五个已有test互补，不重叠：

| 五个test告诉你 | 新方向告诉你 |
|---|---|
| "有没有差异"（是/否） | "差了多少时间"（RMST） |
| "整体曲线匹配吗" | "每个变量的效应保留吗"（HR） |
| "event time的分布对吗" | "censoring的分布对吗"（Censoring） |
| "整体差异大吗" | "哪个时间段差距大"（分段Lin & Xu） |

**核心卖点（跟何老师说的时候）**：底层工具全是传统survival里成熟的——RMST、Cox、KS、Log-Rank。没有发明新方法。我做的是**把已有工具组合起来，应用到合成数据evaluation这个新场景**。传统文献有大量支撑，但在合成survival领域目前没有人这么做。
