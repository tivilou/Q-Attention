# Q-VRES 正式实验协议

## 阶段 A：扩大版 task-transfer screen

- 数据：Re-TACRED 的 train/valid/test 子集。
- 训练：冻结 baseline，只训练 score-level intervention。
- 比较：disabled、Q-VRES causal transport、classical causal transport、Q-VRES key-only。
- 目的：确认任务可见信号、控制组、输出结构和运行稳定性。
- 结果：Q-VRES 在单 seed screen 上相对 baseline 和 classical control 有小幅正向信号，但不构成最终结论。

## 阶段 B：full 五 seed

固定当前配置后，从头运行：

```text
7, 11, 13, 17, 23
```

每个 seed 独立训练 baseline 和三个 intervention selector，不共享 checkpoint。多 GPU 只做 seed-level 并行，不做 DDP 梯度同步。

报告以下统计量：

- 每个 selector 的 valid/test macro-F1 均值、标准差和逐 seed 数值。
- 相对 disabled baseline 的 paired delta。
- Q-VRES 与 classical causal transport 的逐 seed 配对差值。
- attention geometry、context mass error、residual RMS。
- trainable parameter count、Git commit、数据 hash、GPU 和运行时间。

## 阶段 C：结果交接

1. 先完成所有 seed 并确认每个 seed 有 `RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md`。
2. 使用 exporter 生成只包含公开统计结果的报告目录。
3. 只把报告目录提交到 `1.1`，不提交 `runs/`、`data/`、checkpoint、预测文件、JSONL 或完整日志。
4. 负责人审查后再合并到 `main`。

## 固定控制

- `disabled`：冻结 baseline，不加 intervention。
- `q_causal_transport`：主 Q-VRES。
- `classical_causal_transport`：相同接口和参数规模的 classical evidence control。
- `q_causal_key_only`：量子机制只使用 key-only evidence 的控制。
