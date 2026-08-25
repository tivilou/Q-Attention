# Q-Attention / Q-LASS 理想理论框架

> 面向博士后面试 PPT agent 的理论 handoff。本文档选取 Q-LASS 作为本项目最适合展示的主理论，并把已验证结果、理想化命题和未完成验证严格分开。

## 1. 推荐主线

### 一句话版本

**Q-LASS 是一个 label-free、standalone 的量子注意力 score field：它从 query、key 和关系上下文中估计每个 token 对当前注意力的作用，生成有界、零和的 attention-logit residual，从而重新分配注意力，而不是无约束地放大整行注意力。**

### 为什么选择这条理论

这条主线同时容纳了项目的三个核心成果：

1. 原始 `key-space steering` 思想：通过修改 key 或等价地修改 attention score 来改变模型关注位置。
2. standalone quantum module：量子特征映射和量子核独立产生 score field，不依赖 classical projector 才能工作。
3. 可审计的注意力干预：centered residual、gain bound、padding mask 和 matched classical control 都能写成明确的数学约束。

Q-SRPA（query-conditioned soft-role pair attention）应作为第二代扩展放在主线之后：它让关系锚点随 query 改变，但目前更适合作为“理论上自然、机制筛查已通过、自然任务尚未证实”的后续方向，而不是主成果标题。

## 2. 问题定义：从注意力分数到可控的注意力场

对第 `l` 层第 `h` 个 head，标准 attention logit 为

\[
 A_{ij}^{(l,h)} = \frac{q_i^{\top} k_j}{\sqrt{d_k}},
 \qquad q_i,k_j\in\mathbb{R}^{d_k}.
\]

其中 `i` 是 query token，`j` 是 key token。关系抽取或 prompt highlighting 的共同问题是：模型已经产生了基础注意力，但我们希望在不改动 frozen backbone 的前提下，把注意力质量向与当前 query/关系有关的 token 重新分配。

因此定义干预目标：

\[
 A'_{ij}=A_{ij}+\Delta_{ij},
\]

其中 `Δ` 不是任意 bias，而是由一个 label-free quantum score field 产生的、受约束的 token-level residual。

## 3. Q-LASS 的四层结构

### 3.1 关系锚点：把任务条件注入 token 表示

先由冻结的 `Q/K` 和 padding mask 构造关系锚点 `r`。当前实现支持以下层次：

- `global_context`：所有有效 key 的 masked mean；
- `soft_role_pair`：用两个 soft role router 得到 source/target context；
- `query_conditioned_soft_role_pair`：让 role distribution 随每个 query 改变；
- `entity_pair`：历史实体对锚点，仅作为 legacy/control 路径，不能代表 label-free 主张。

Q-LASS 主理论推荐使用 `global_context` 作为最干净的 baseline 版本；Q-SRPA 再引入 query-conditioned role pair。

### 3.2 量子特征映射

对每个 query/key 和关系锚点拼接后的输入，使用可训练的量子特征映射：

\[
 z_i^q = \phi_\theta(q_i,r),\qquad
 z_j^k = \phi_\theta(k_j,r),
\]

其中 `φθ` 可以理解为“投影 + angle encoding + data re-uploading + entangling circuit”的整体。当前代码在 statevector 上实现可微模拟；它证明的是可训练的函数机制，不等于当前硬件上的速度或能耗优势。

量子核的最直观读出是 fidelity：

\[
 \kappa_Q(i,j)=\left|\langle z_i^q,z_j^k\rangle\right|^2.
\]

项目还保留 interference、observable、continuous measurement 等 readout。它们都遵守同一个抽象接口：输出一个 query-key score field，再经过中心化和增益控制。

### 3.3 中心化和有界 residual

对每个有效 query 行，在 key 轴上做 masked centering：

\[
 \bar\kappa_{ij}
 = \kappa_{ij}
 - \frac{1}{|K_i|}\sum_{j'\in K_i}\kappa_{ij'}.
\]

然后使用有界 gain：

\[
 g_{l,h}=g_{\max}\tanh(\rho_{l,h}),
 \qquad
 \Delta_{ij}=g_{l,h}\,\bar\kappa_{ij}
\]

必要时再乘以 label-free evidence modulation。由于中心化，所有有效 key 上满足：

\[
 \sum_{j\in K_i}\Delta_{ij}=0.
\]

这意味着 Q-LASS 做的是 **attention transport**：它不施加统一的 row-wise logit 偏移，而是把相对偏好从部分 token 移向另一部分 token；softmax 后的概率仍然由归一化决定。

### 3.4 注意力更新

最终分数为：

\[
 A'_{ij}=\frac{q_i^{\top}k_j}{\sqrt{d_k}}+\Delta_{ij},
 \qquad
 P'_{i,:}=\operatorname{softmax}(A'_{i,:}).
\]

模型主体保持 frozen；训练只更新量子 score kernel、gain 以及被显式允许的 role/readout 参数。所有 action path 必须只读取 query、key 和 padding mask，不能读取 gold label 或实体 span。

## 4. 最漂亮的理论命题

这些命题是 PPT 中应当突出的“理论骨架”。它们是由定义直接推出的结构性质，不应写成未经条件说明的实验结论。

### 命题 A：零和注意力传输

若 `Δ` 按有效 key 集合中心化，则每个有效 query 行的 residual 总和为零。于是干预保持 row-wise conservation，只改变 token 间相对竞争关系。

**直观解释：** Q-LASS 不是另加一层启发式 bias，而是在注意力 simplex 上做受约束的质量搬运。

### 命题 B：有界性和数值稳定性

若 `|g_{l,h}|<gmax`，且量子 readout 经过有限范数归一化，则 residual 的幅度受到显式上界控制。训练中的 gain 不会因单个 head 或单个 token 无界爆炸。

### 命题 C：score intervention 与 key steering 的精确对偶

对于非零 query，定义 query-aligned key delta：

\[
 \delta k_{ij}
 = \frac{\sqrt{d_k}\,\Delta_{ij}}{\|q_i\|_2^2}\,q_i.
\]

则有严格恒等式：

\[
 \frac{q_i^{\top}(k_j+\delta k_{ij})}{\sqrt{d_k}}
 = \frac{q_i^{\top}k_j}{\sqrt{d_k}}+\Delta_{ij}.
\]

所以 Q-LASS 可以被看成两种等价表述：

- **score view：** 直接给 attention logits 加上量子 residual；
- **key view：** 沿 query 方向编辑 key，使内积产生同样的 logit 增量。

这里的 `δk_{ij}` 是针对 `(query i, key j)` 的局部对偶表示；要把它实现成对所有 query 共享的单一 key 编辑，还需要额外的低秩/可分解条件。PPT 中不要把这一步误写成“任何 score residual 都自动等价于一个全局 key 矩阵”。

### 命题 D：padding 与 token permutation 不变性

只要 anchor、centering 和 evidence modulation 都使用同一个 padding mask，padding token 的权重为零；对有效 token 做同样的置换，输出 score field 只发生对应置换。这给出 label-free routing 的基本结构保证。

### 命题 E：量子性必须通过 matched control 才能归因

Q-LASS 的 quantum kernel 与 classical control 应使用相同的输入、训练预算和可训练参数量。即使 Q-LASS 任务效果更好，也只能先说明“该量子参数化机制在此协议下有效”；只有在预注册的 matched-control gate 通过后，才可以讨论量子特有归因。

## 5. Q-SRPA：最自然的第二代扩展

Q-SRPA 将单一关系锚点改为两个 soft role context：

\[
 s=\sum_j w_j^{(s)}k_j,\qquad
 t=\sum_j w_j^{(t)}k_j,
\]

并构造：

\[
 r=[s,\ t,\ s-t,\ s\odot t].
\]

在 query-conditioned 版本中，role logits 同时由 key 和 query 决定。例如：

\[
 \ell_{iqr}
 =\frac{(q_i^{\top}u_r)(k_q^{\top}v_r)+b_r}{\tau},
 \qquad
 w_{iqr}=\operatorname{softmax}_{q}(\ell_{iqr}).
\]

这使模型能够在 query 改变时反转 source/target role emphasis，同时仍只读取 `Q/K + padding mask`。它是对“关系不是一个固定全局向量，而是 query-dependent field”的漂亮理论表达。

当前机制 toy screen 已观察到 query-role reversal 和 zero-sum action 等结构现象；matched classical router 产生同一类 role routing，因此该结果不能支持量子优势。Q-SRPA 的正式 Re-TACRED gate 未通过，不应在面试中说成已证实的自然任务提升。

## 6. 训练目标和控制组

推荐 PPT 只展示抽象目标：

\[
 \mathcal L
 =\mathcal L_{task}
 +\lambda_H\mathcal L_{entropy}
 +\lambda_O\mathcal L_{overlap}
 +\lambda_D\mathcal L_{diversity}.
\]

- `L_task`：关系抽取或冻结 synthetic task 的任务损失；
- `L_entropy`：避免 role router 退化到单 token；
- `L_overlap`：鼓励 source/target role 分布不要完全重合；
- `L_diversity`：减少不同 heads 的 score field 完全相同。

实验上至少保留：disabled baseline、matched classical kernel、query shuffle/magnitude control，以及必要的 dephased/product-state control。控制组的目的不是增加图表，而是把“任务有效”“参数化有效”“量子特有”三件事拆开。

## 7. 当前证据：可以展示什么

### 已完成且适合展示的证据

冻结 synthetic held-out attention-alignment audit 使用 seeds `7, 11, 13, 17, 23`，并通过 source/config replay gate。Q-LASS 在 valid/test 上五个 seed 都呈现同一主方向：

| 指标 | valid | test |
| --- | ---: | ---: |
| evidence-minus-distractor margin delta | `+0.08851` | `+0.08233` |
| evidence top-2 recall delta | `+0.03770` | `+0.03887` |
| harmful movement（全部 query） | `2.50%` | `2.46%` |
| harmful movement（active actions） | `12.82%` | `13.10%` |

这支持的表述是：**Q-LASS 在冻结 synthetic held-out task 上具有可重复的 evidence localization / attention redistribution 现象。**

### 必须同时说明的限制

1. evidence 位置来自 synthetic gold annotation，不是自然语言 rationale。
2. product/classical control 也改善了 alignment 指标，quantum-specific attribution gate 失败（冻结协议的 exact paired sign-flip `p=0.0625`）。
3. 大约 `13%` 的 active actions 会把 margin 推向有害方向，aggregate positive 不等于逐样本安全。
4. 当前没有 finite-shot、噪声、真实硬件速度或能耗结论。
5. Q-LASS 的 Re-TACRED 正式 single-seed 结果没有优于 matched classical control；Q-SRPA 的正式 test practical-gain gate 也没有达到预注册阈值。因此不能声称 natural-task improvement 或 multi-seed replication 已成立。

## 8. 建议 PPT 结构（10 页）

1. **问题：** 传统 attention 只能被动计算，如何在不改 backbone 的情况下让模型关注关系相关 token？
2. **几何起点：** `k' = k + gPk`，把 key-space steering 解释为 attention-logit control。
3. **核心想法：** 用量子特征映射生成 relation-conditioned score field。
4. **结构约束：** masked centering + bounded gain，得到 zero-sum attention transport。
5. **关键理论：** score intervention 与 query-aligned key update 的精确恒等式。
6. **关系条件：** global context → soft role pair → query-conditioned Q-SRPA。
7. **实验协议：** frozen backbone、label-free action path、matched classical control。
8. **最强结果：** 五 seed synthetic attention alignment 表格或 evidence/distractor heatmap。
9. **诚实边界：** natural transfer、quantum attribution、finite-shot/hardware resource 尚未建立。
10. **研究展望：** 带自然 evidence annotation 的 transfer protocol、active-action harm gate、有限 shots 和硬件资源审计。

## 9. 推荐的面试表述

### 可以直接使用

> 我们不是把量子电路当作一个黑箱分类器，而是把它约束成一个 label-free attention score field。这个 field 在每个 query 行上做零和、有限幅度的注意力质量搬运，因此既保留了 key-space steering 的几何解释，又能用 matched classical control 检验量子结构到底贡献了什么。

> 在冻结的 synthetic held-out task 上，Q-LASS 跨五个 seed 一致提高 evidence localization；但 matched classical control 也有改善，所以我们把结果报告为可重复的注意力机制证据，而不是量子优势证明。

### 不要使用

- “Q-LASS 已经证明量子优越性”；
- “Q-LASS 在 Re-TACRED 上稳定提升”；
- “当前硬件上具有速度或能耗优势”；
- “所有 score residual 都等价于一个共享的 key 矩阵编辑”；
- “Q-SRPA 已完成自然任务验证”。

## 10. PPT agent 的素材索引

- 原始 key-space steering 理论：`paper_extract.txt`，以及项目根目录的 `2603.01281v1.pdf`。
- 当前 score-kernel 实现：`src/q_attention/plugins/attention_score_kernel.py`。
- 当前 Q-LASS 正式配置：`configs/retacred_qlass_formal_single_seed.json`。
- Q-SRPA plugin plan：`configs/query_conditioned_soft_role_pair.plugin-plan.json`。
- synthetic Q-LASS alignment evidence：`.ai-progress/workstreams/quantum-projector-v2/refs/2026-08-21-qlass-attention-alignment-evidence.json`。
- ideal resource audit：`.ai-progress/workstreams/quantum-projector-v2/refs/2026-08-20-ideal-quantum-resource-audit.json`。

## 11. 最终定位

本项目当前最理想、最适合面试展示的理论不是“量子模型替代 Transformer”，而是：

> **用一个可解释、可约束、可对偶到 key-space steering 的 standalone quantum score field，完成关系条件下的注意力重分配。**

它的科学价值首先在于统一了 attention steering、量子特征映射和结构化证据定位；量子优势、自然任务增益和硬件资源收益都应作为后续待验证问题，而不是当前结论。
