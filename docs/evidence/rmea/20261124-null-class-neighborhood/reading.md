# 无关系类别：近邻阅读与原模型对应

当前事实仅来自 TRAIN 元数据：no_relation 38,761/58,465 行，覆盖 25 种类型组合。它不证明分类空间里存在相应数量的模态，也不证明量子测量纯度是原因。

## 已核对的原文范围

- Sabo、Elazar、Goldberg、Dagan，Revisiting Few-shot Relation Classification: Evaluation Data and Classification Schemes，arXiv:2104.08481v1。核对首页题名/作者；阅读 §5.1–5.4（PDF pp.7–9），并目视核对 p.9 的 NAV/MNAV 与训练式。该文在 few-shot episodic 分类中把无目标关系的实例纳入 NOTA，比较固定阈值、学习的单个 NOTA 向量以及多个 NOTA 向量，讨论它们对嵌入空间的约束。不能把其中 few-shot NOTA 率或性能移植到我们的监督 Re-TACRED。
  原文：https://arxiv.org/abs/2104.08481v1
- Zhou、Huang、Ma、Huang，Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling，arXiv:2010.11304v3。核对首页题名/作者；阅读 PDF p.3 的 Adaptive Thresholding、两项损失和 p.4 的 Localized Context Pooling 开头，目视核对 p.3 的公式。该文的输入依赖 TH 分数区分每个实体对的正负关系；专用训练损失与普通多类 CE 有区别。其任务是文档级、多标签 RE，不是当前单标签句子级任务。
  原文：https://arxiv.org/abs/2010.11304v3

这只是上述局部原文阅读；不是两篇全文深审、系统综述或新颖性审查通过。PDF 与解析文本留本机研究缓存；Git 仅保存本说明与题录/哈希/阅读范围。

## 对 Q-RMEA 的直接约束

当前模型为每类输出 z_r(x)，不是统一的固定拒识阈值。类别 0 为 no_relation，故其推断等价于：仅在 max_(r>0)[z_r(x)-z_0(x)]>0 时输出最大正关系，否则输出 0（平分时遵从当前索引次序）。z_0(x) 已随输入变化。因此，把 z_0 改称“自适应阈值”没有新增推断机制；但不能由此声称已经实现 ATLOP 的训练损失或 NAV/MNAV 模型。

此外，Q-RMEA 的 query/effect 控制 token 注意力，类别分数是加权有符号 value。它不是对整句嵌入做 nearest-prototype 分类。把“单一 NOTA 原型不够”的论述直接套到无关系 effect，缺少实现对应；增加多个 null effect 也需同输入/容量的经典控制和真正任务贡献。

历史边界继续保留：native-mass-evidence 的冻结接受率读出失败；null-relative-filter 的固定 128 次续训失败；类型/全局偏置、硬筛及 shared-null differential 的具体配置已关闭。这里没有重新开放这些配置或选择新的量子插件。

下一项有用证据是完整预算四个非均匀头在 TRAIN 的匹配错误结构：Q 单独多出的错误是否集中在与无关系竞争，或主要是正关系之间区分。先取得同一输入、同一标签、同一 selected checkpoint 的数据，再定机制；不使用 TEST 挑样本/类别/阈值。
