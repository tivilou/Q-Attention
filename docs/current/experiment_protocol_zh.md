# Dual Q-RES 正式实验协议

## 阶段 A：full 单 seed 预跑

- 数据：完整 Re-TACRED train/valid。
- Seed：`13`。
- 比较：baseline、quantum、local classical、strong classical。
- 目的：确认完整流程、输出结构、显存和运行时间。
- 运行方式：[Re-TACRED full 运行指南](retacred_dual_qres_full_run_zh.md)。

这一阶段属于主实验配置预跑，不作为最终统计结论。

## 阶段 B：full 五 seed

固定阶段 A 的配置后，分别从头运行：

```text
7, 11, 13, 17, 23
```

每个 seed 必须重新训练 baseline、quantum/classical core 和三个 selector。不同 seed 不共享 checkpoint。

报告以下统计量：

- Macro-F1：均值、标准差、逐 seed 值。
- Valid loss：均值、标准差、逐 seed 值。
- Correct-label margin：均值、标准差、逐 seed 值。
- Selectivity pass rate。
- Quantum 与两个 classical control 的配对差值和显著性检验。

## 阶段 C：机制消融

至少包括：

- no-entanglement/separable selector；
- shared readout 对比 dual readout；
- 不使用 `context_budget`；
- 仅 core、不附加 selector；
- 参数量和资源统计。

优先在 seed 13 完成消融；关键消融再扩展到五 seed。

## 阶段 D：blind test

只有在以下条件全部满足后才能运行 test：

1. train/valid 配置已经冻结；
2. 五 seed checkpoint 已经确定；
3. 不再根据 test 调整模型、超参数或 checkpoint；
4. 记录 test evaluation 开始时间和对应 commit。

test 只运行一次统一评估，不进入训练或模型选择。

## 固定约束

- 当前最小代码基线：commit `b8d794f`。
- Core 使用 `selection_metric=valid_loss`。
- full screening 使用 `diagnostic_batches=64`，task metrics 仍遍历完整 valid。
- Quantum/local classical/strong classical 的 selector 参数量必须匹配。
- 不把 `runs/`、数据或模型权重提交到公开仓库。
