# Q-RMEA L5 Round37：分类对齐不等于词责任，新的线索在类别 margin

本轮读三篇原始多实例学习工作，并重算已有完整报告中的25个模式×seed条件、50张混淆矩阵。得到一个具体探索线索：Q-RMEA 的固定上下文权重置换后 Macro 高于 real-overlap，但 dephased 很接近 Q；这还不是新的独立确认、相位机制或新方法。限定 L4 维持，L5 尚未接受。未访问私有数据、模型、checkpoint、COS 或执行训练。

## 原文给出的任务条件

| 原始工作与选读范围 | 原文事实 | 当前模型的边界 |
|---|---|---|
| [Wang et al.,1810.09050v3](https://arxiv.org/abs/1810.09050v3)，pp1–3 | Section2/Table1 已给出 weighted pooling 导数；负袋中 attention 可集中到低 frame probability，袋级成功与定位可能分离 | 本模型是40类联合 CE＋带符号 utility，不是每类独立 BCE 的[0,1]实例概率；“加权均值的 sensitivity”已有直接先例 |
| [Li,1811.12346v2](https://arxiv.org/abs/1811.12346v2)，pp1–4、8 | 类集合标签、背景类别、位置 categorical emissions；Assumption1 为给定网络内部状态的发射条件独立，Proposition1 用乘积与 inclusion-exclusion 算精确袋似然 | 当前每词40个独立二元 measurement 不是 categorical class posterior，词权重也不是该 OR-set 似然的责任后验；只读了部分 AppendixA |
| [Azaklı & Stelzer,2508.07114v2](https://arxiv.org/abs/2508.07114v2)，pp1–6 | Eq3 是 iid 碰撞事件的密度乘积；Eq7 的 sqrt(N) SNR 依赖独立同分布/同质条件；Eq15 讨论学习误差下的有效信息 | 原文明确研究跨所有实例分布的弱信号，不是找一个关键词。同句上下文化词不构成已认证 iid 事件；有效信息近似和误差模型未独立复核，不能拿来承诺 Macro 提升 |

三篇 exact version/PDF/parse/integrity 身份及9页图见[阅读记录](reading-receipts.json)。只在指定范围内读原文，没有复现代码或完整实验。检索共24个不同 metadata，21篇未选记录仍只属 metadata；搜索服务404后改用官方 arXiv API，三次请求均200，本轮不需要浏览器，不声称穷尽近邻。

## 从负类别梯度得到的范围修正

对当前独立 q 坐标，记 S_r=sum_valid(q_rt+epsilon)、bar_u_r=sum alpha_rt*u_rt、g_r=softmax(z)_r−1[r=y]，有

    dCE/dq_rt = g_r*(u_rt−bar_u_r)/S_r。

正确类别的 g_y<0，错误类别的 g_r>0。仅在自由 q 的局部坐标下降视角，前者偏向提高高 utility 位置的权重，后者可偏向低 utility 位置。因此局部 CE 对齐不能自动解释成所有关系分支都在定位“表达该关系”的词。这个导数是已知 normalized pooling/CE 链式法则，不是新理论，也不代表共享 prepare、电路拉回和 clipped AdamW 的实际更新。39个负类别的数量也不等于39倍梯度优势：其 g_r 总和恰等于1−p_y，抵消正确类梯度的总量。

精确袋似然需要定义潜在发射变量及监督连接；不能因为权重非负且归一化，就把 Born 权重认作证据责任后验。旧 L4 支持的是固定模型 weight–value 对齐与任务效用的限定合取，维持这个范围。

## 旧完整报告中的可观测新线索

先冻结[计算范围](source-verification.json)：5模式、5seed、全test13418个样本、所有40类，无子集筛选。Macro 和 accuracy 从实际/固定置换各一张40×40混淆矩阵重算，共50张；CE 沿用已审计汇总值并核对差值方向，未由缺失的逐样本概率假称重算 CE。结果全部保留在[endpoint 重分析](endpoint-reanalysis.json)。

| 模式 | 实际 Macro 均值 | 固定置换后 Macro 均值 | 实际−置换 Macro | 置换−实际 CE |
|---|---:|---:|---:|---:|
| uniform | .2149121913 | .2149121913 | 0 | 约6.25e-9，数值 null |
| relation_softmax | .5182790362 | .3543919013 | .1638871349 | .2710060105 |
| real_overlap | .5158315007 | .3454114928 | .1704200078 | .2764112803 |
| quantum | .5207533288 | .3791560149 | .1415973140 | .2020138258 |
| dephased | .5172695949 | .3750607627 | .1422088323 | .2344815864 |

四个非均匀模式的 Macro 与 accuracy 在每个 seed 都因这个置换下降；Q 的 Macro下降幅度在五个 seed 均小于 real-overlap。它们是已暴露、同一固定置换方案上的探索性结果，没有新统计门、独立确认或新评级。各模式 value/decoder 是各自训练的，较小下降本身不能区分更稳健的权重、不同 readout 或不同效用尺度。dephased 的置换后 Macro 接近 Q，禁止把这个现象直接归因于相位。

统一正温度缩放全部40 logits 不改变实际或置换后的 argmax，因此纯全局温度差异不能单独解释 Macro 的变化差别；它仍能改变 CE。该已知不变性仅排除很窄的 confidence-only 解释，不排除不同 margins、类别 bias、coadaptation、特征容量或共享权重–value 的协方差结构。

## 唯一下一步

从“注意力找到了关键词”转向可观察的类别间 margin：核查 finite permutation covariance 与 attention margin stability 的原始近邻，推导当前**所有关系共享同一 eligible-context 置换**时，固定权重和带符号 utilities 的联合扰动协方差。先判断是否存在超出通用有限总体恒等式的 Born 约束设计或可区分预测；只写一个 variance/Cantelli/链式法则公式仍不算新贡献。

更分散的单行权重未必是正确解释：分类使用 logits 之差，共同变化可能在 margin 中抵消。这个方向尚无选定方法、非平凡新定理或独立验证；不读取私有数组来追逐未经限定的故事，不补可选训练 seed，不重开已停家族。新正式全量实验仍交师弟。
