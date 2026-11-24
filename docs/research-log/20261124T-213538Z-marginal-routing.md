# 两个局部量子视图分别路由取值：固定四路筛选

原模型每个关系使用一组注意力汇总整个64维取值。本候选保留81028注册参数，原3qubit态按2×2×2拆成AB/BC两个约化视图，分别加EPS/掩码/归一化，池化偶数/奇数取值坐标；共享value/decoder/bias。实数12维按2×3×2作同样处理。

四路各512更新、seed13、TRAIN4192fit/4000assessment、b128/FP32/同AdamW与顺序。全部fit完成后仅final/all40一次评估，并在同一candidate终态执行预声明的冻结共用路由干预，不拟合或选择。四原参考复用，不重跑baseline。

| 条件 | all40 Macro F1 | CE | 正类漏判null | null误报正类 | 来源 |
|---|---:|---:|---:|---:|---|
| quantum_marginal_split | 0.121397256 | 1.068608880 | 829 | 227 | 本轮 |
| population_marginal_split | 0.188187782 | 0.874360323 | 720 | 204 | 本轮 |
| quantum_marginal_tied | 0.145386159 | 0.961994290 | 808 | 172 | 本轮 |
| real12_marginal_split | 0.188483187 | 0.873105228 | 635 | 230 | 本轮 |
| quantum_native | 0.172509408 | 1.076011419 | 791 | 206 | 复用 |
| real12 | 0.211989007 | 0.922038913 | 592 | 277 | 复用 |
| softmax12 | 0.201206552 | 0.909885466 | 599 | 272 | 复用 |
| rbf12 | 0.353977485 | 0.882513046 | 517 | 308 | 复用 |

预声明七项Macro＋一项冻结margin条件通过0/8：CLOSE_EXACT_MARGINAL_ROUTING_SCREEN。CE等描述量不代替主门。

冻结candidate终态，把两组已经归一化的注意力同时替换为平均图；各自重新计算最大对手。全部1377条正例的完整减共用平均gold-largest-rival margin为-0.160623670；原logit重放最大误差0。干预不等于重新训练，独立tied训练对照另列。

- macro_beats_population_marginal_split: FAIL
- macro_beats_quantum_marginal_tied: FAIL
- macro_beats_real12_marginal_split: FAIL
- macro_beats_reused/quantum_native: FAIL
- macro_beats_reused/real12: FAIL
- macro_beats_reused/softmax12: FAIL
- macro_beats_reused/rbf12: FAIL
- positive_margin_beats_frozen_tying: FAIL

关闭本确切局部视图/坐标拆分/归一化/512配置，不改视图、坐标、增益、seed、预算、指标或对照救援；不判整个家族不可能。

分组路由确实能超出固定单一凸池化，已构造单图最大1而双图约2的例子；GHZ正负态却有相同两个局部视图，故也会丢掉原完整态可区分的信息。这些是已知partial trace/multihead工具与代数，不是新T或文本收益定理。当前结果只检验本具体实现，不把可模拟性当作量子机制否决条件。

population对照只删局部重叠的非对角项，仍保留相干制备；81028注册参数不代表相同有效容量，实数局部维数也不同。固定参数读出恒等式和干预只能说明本拟合解的依赖，不能排除重新学习的共享路由。

科学运行137.001秒/2048新更新，含1.498秒冻结回放；全部八CM独立重算Macro误差≤2.78e-17，正类margin干预与三组修复/损伤从远端私有数组独立核对。真实8行、独立振幅收缩/NumPy、梯度、padding/单词/相位、factory加载通过。corev6/host不改，不重做续训/COS矩阵。

没有新官方VALID/TEST或COS写入，没有向Windows下载模型、特征或逐例数组。全部正负结果/旧限定L4保留，新L3/L4-T未接受。

[协议、源码和全部结果](../evidence/rmea/20261125-marginal-routing/summary.json)。
