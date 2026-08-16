# Q-VRES 正式实验协议

## 阶段 A：扩大版 task-transfer screen

- 数据：Re-TACRED 的 train/valid/test 子集。
- 比较：disabled、Q-VRES causal transport、classical causal transport、Q-VRES key-only。
- 结果：screen 出现单 seed 正向信号，但不能作为最终结论。

## 阶段 B：full seed 13 pilot

- baseline 只训练一次，三个 selector 在三张 GPU 上并行。
- validation macro-F1：baseline `0.217253`，Q causal `0.215264`，classical `0.217354`，Q key-only `0.218519`。
- Q causal 未通过 validation gate，因此不启动原方案五 seed。
- test 指标只用于报告，不用于修改方法或选择超参数。

## 阶段 C：validation 机制诊断

从 seed 13 raw run 读取 baseline 和 selector checkpoint，生成以下聚合结果：

- 按关系类别的 support、precision、recall 和 F1。
- Q causal 相对 baseline 的 recall/F1 变化及关系频次分桶。
- quantum readout、leave-one-out value influence 和组合 evidence 的排序相关性。
- 干预前后 attention、gold-margin 梯度代理和预测翻转统计。

只输出 `diagnostic_summary.json` 和 `diagnostic_summary.md`。不输出逐样本预测、原句、token、checkpoint 或梯度张量。

## 阶段 D：决定下一版机制

- 如果正式诊断确认 value influence 只表示影响大小、缺少有利/有害方向，应先修正或门控 Q causal。
- 如果 Q key-only 的任务表现和方向对齐更稳定，只将其升级为下一轮受控 pilot 候选。
- 如果两者都没有有效证据对齐，停止当前 evidence 定义。
- 新机制通过单 seed validation gate 后，才能恢复五 seed 实验。

## 结果交接

1. 合作者只向 `1.1` 提交本轮 `reports/` 下的聚合报告。
2. 禁止提交 `runs/`、`data/`、checkpoint、逐样本预测、JSONL、梯度张量或完整日志。
3. 负责人审查后将报告合并到 `main`。
