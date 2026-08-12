# Q-NESS Toy Gate

在项目根目录执行：

```bash
python experiments/run_qness_toy_gate.py --seeds 7,11,13,17,23 --device auto
```

脚本会在 `runs/qness_toy_gate/` 下按当前时间自动创建目录，并写入：

```text
run_summary.json
run_summary.md
```

本阶段只验证机制，不使用 Re-TACRED，也不需要上传 `runs/`。结果必须同时包含：

- Q-NESS 五 seed 的 necessity/sufficiency 恢复误差；
- complement error、两种 evidence 的 overlap/cosine；
- commutator、off-diagonal density norm、mutual information；
- commuting、separable、phase-scrambled、dephased 和 classical 控制。

只有 `gate_pass: true` 且五个 seed 全部通过，才进入 proportional gate。Toy gate 通过不等于任务效果提升。
