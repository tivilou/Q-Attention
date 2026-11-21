# 实体端点散射：一次完整负向实验

## 一句话理解

把原来对词证据加权求和的读出换成两个实体首词之间的传播强度；完整seed13实验严重退步，停止这版配置。

## 原任务与动机

输入为句子和两个已知实体，输出40类关系。原模型在每个词上形成关系utility并按关系查询加权。本次希望用上下文中的多路径干涉代替最终汇总，让实体间证据组合有更强结构。此为当时假设，实测没有支持任务收益。

## 一步步怎么做

1. 读取冻结RoBERTa-large词缓存h（每词1024实数）及4维实体位置特征，padding由mask排除；有效词数随句子变化。
2. 原关系头把输入准备为三量子比特、8维复状态psi；40个关系查询phi形成归一化路由q。每词的utility来自可训练value映射与关系decoder。
3. 对每个关系构造词图：两词连接是sqrt(q_i q_j)乘状态内积；对角在位势为tanh(utility)。选主、客体跨度各自首个有效词为两个端点；缺失/重叠会报错。
4. 固定能量0、阻尼0.1、端点引线强度1，在复数矩阵逆中取两端点元素。它的模平方加1e-12后取log，再加原关系bias，得到40个logit。
5. softmax交叉熵反馈到关系头全部81,028参数。测试时只前向取类别，标签用于评价；没有文本生成、检索、量子硬件或额外encoder训练。

## 原理与公式

直觉是不同上下文词可形成增强或抵消的传播路径。设G_ij=psi_i†psi_j，D=diag(sqrt(q))，e_i=tanh(utility_i)。实际H=D(G-I)D+diag(e)，R=(0I-H+iB)^-1，B=0.1I+(P_s+P_o)/2。关系分数log(|R_so|²+1e-12)+bias。复数方案用8维Woodbury恒等式计算，精确对照检查通过；real_magnitude仅把最后G替换为|G|，采用稠密求解。

q在这里是上游路由，不是新的最终池化权重。Green函数、被动传输、Woodbury、相位回路等都是已有工具；未声明量子独占性或加速。

## 哪些参数会学习

实际训练对象是原头的prepare线性映射、关系effect参数、value映射、decoder与bias等全部参数，AdamW(head.parameters())，总数81,028，三条件一致。RoBERTa缓存固定，非本次optimizer成员。能量、阻尼、引线、端点规则、控制类型及epsilon固定。训练标签只用于CE；验证/测试标签只用于评价。

## 训练与推理

完整训练58465条，固定seed13、4轮、batch32、micro4梯度累积。每次本地保存best/latest；COS按全局10%协调屏障做临时恢复。best由valid保留，但报告固定最终第4轮，不选择best。三路各7312次更新，三张3060并行；复用完成的linear，不重训基线。最终valid19584、test13418全量评价。

## 帮助理解的例子

未物化原句和全数值Case Study；本条不编造样本。九个最终回放锚点只是各条件×三个split首2样本的紧凑接口证据。该探索配置现已因负结果停止，未要求后续Case Study；不将锚点或理论合成fixture当成真实语言样本解释。

## 实验设置与对照

complex为预声明候选；real_magnitude保留上游q/e，仅改最终Gram；constant_onsite把tanh utility取有效词无权均值再广播，各参数仍训练。linear_coherent为已完成同seed、初始权重、训练顺序、数据缓存、AdamW及参数预算参考。新fit的microbatch核与旧核不保证逐比特相同。联合门槛：complex测试positive39 Macro高于三参考且CE不高于linear。只一个反复使用的官方split/seed，不是独立确认。

## 实际效果

| 条件 | 测试 positive39 Macro-F1 | 测试 micro-F1 | 测试 CE（越低越好） |
|---|---:|---:|---:|
| linear_reference | 0.462882468 | 0.720766894 | 0.586970119 |
| complex | 0.111801607 | 0.542171429 | 1.010669309 |
| real_magnitude | 0.118826117 | 0.572305668 | 0.989814600 |
| constant_onsite | 0.153024914 | 0.586340206 | 0.942552644 |


positive39 Macro固定平均39个正关系的F1（零除取0）；micro-F1是benchmark排除no_relation定义；CE为逐样本交叉熵。独立混淆矩阵重算一致。complex对linear下降35.108个百分点，且低于两个新对照，冻结门槛失败。

## 成功与失败原因

[已测事实] 本轮无任务收益，执行和数值校验通过。三路替代读出均低于linear，训练CE均逐epoch下降；这排除了“任务未跑完”，没有证明训练最优或唯一根因。

[尚未证明] 单实体首词传输可能丢失分布式词证据，也可能是该损失几何、阻尼、固定预算的组合。尚无相应诊断。若以后作为全新明确假设研究，应在训练前定义保留原读出的对照来区分；当前停止方案不执行额外冻结模型诊断或调参补救。

[机制解释] 复数相位容量、自由态抵消边界和漂亮的逆矩阵恒等式无法保证语言任务有效。此结果仅约束本配方，不排除全部量子/输运模型。历史限定L4没有被本次新配方覆盖。

## 一分钟口述

我们想让关系分类不只对词证据做加权平均，而是利用词状态之间的复数连接，让两个实体之间的传播路径表达上下文。实现保留原81,028参数，固定读出常数，用三张卡分别训练复数、实数幅值和常数在位势三个条件。完整数据的固定第4轮结果显示复数方案Macro-F1只有11.18%，原线性参考46.29%，两个新对照也更好，所以这条配方停止。数学上多路径确实能干涉，但这次没有把这种容量转成任务收益，更不能据此称L5。

## 面试追问

- 是否全量？是，58465/19584/13418，单seed13；单seed不等于独立重复确认。
- 相位有用吗？该读出下complex未胜过幅值对照；后者还保留上游相位，不能作全部相位的纯因果结论。
- 为什么失败？当前只有聚合证据和实现检查，具体根因待区分；不编造Case Study。
- 还要多跑种子吗？这版不晋级，无补救种子/epoch/checkpoint选择。

## 证据与版本

执行代码8b5eca869517b873c89e12369a6389355d3845d8；source bf3e086677aa71b28a9fb60f7ef0344f0f0ffff715c5fd9a6b4b442b7b7ebcc7；config a3000139e14526e303a020f056f21d6c28da692d0b1a205a38cb8f535aabed8a；summary SHA256 3047a831b7b72c45fb5a4dd224dc8e65595e74d20a45216dccb9c702a6ce509c。报告、审计与代码身份分别保留。

- [正式结果](../../../reports/q_rmea_entity_scattering/20261122-seed13-92ba0e8a/summary.json)、[配置和来源](../../../reports/q_rmea_entity_scattering/20261122-seed13-92ba0e8a/manifest.json)、[审计](../../evidence/rmea/20261122-entity-scattering/formal-audit.json)。
- [研究日志](../../research-log/20261121T-231433Z-entity-scattering-full-seed13.md)、[结果表](/apps/experiment-results/?family=family-q-rmea-entity-scattering-seed13)。
- [回放锚点清单](../../../reports/q_rmea_entity_scattering/20261122-seed13-92ba0e8a/analysis-readiness.json)：未物化sample-trace.v2；解释文档完整度partial，实验complete，VALID_NEGATIVE，L5 NOT_ACCEPTED。
