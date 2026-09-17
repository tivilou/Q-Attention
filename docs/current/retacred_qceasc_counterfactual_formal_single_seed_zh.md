# Q-CEASC per-key counterfactual influence：Re-TACRED 正式单 seed

## 实验目的

本实验验证新的 Q-CEASC per-key counterfactual influence 机制在完整 Re-TACRED 上是否相较同一冻结 baseline 提升关系抽取效果。量子候选与参数匹配的 classical counterfactual 控制必须同时运行。

- seed：13
- 数据：`train=58465`、`valid=19584`、`test=13418`
- candidate：`q_ceasc_counterfactual`
- matched control：`classical_counterfactual`
- primary metric：held-out test macro-F1
- 成本：精确 leave-one-out 的评估量记录在 metadata 和 summary 中；成本高本身不否决效果验证，OOM、非有限值、掩码错误、断点契约错误和证据缺失仍是硬失败。

## 合作者运行

在合作者服务器的项目目录执行：

```bash
git fetch origin --prune
git checkout 1.1
git merge --ff-only origin/1.1
git merge --no-edit origin/main
git status --short --branch
bash scripts/run_retacred_qceasc_counterfactual_formal_single_seed.sh --gpu auto --hardware-profile adaptive
```

脚本会自动：

1. 预检 GPU、代码入口和配置；
2. 从最高吞吐档位启动：logical/physical batch 256、无 chunk、无 activation checkpointing；OOM/压力事件后先降 physical micro-batch 并用 accumulation 保持 logical batch，再从 all-query chunk 逐级减半；
3. baseline 只运行一次，然后把两个 selector 放入动态 GPU 队列；
4. 每个 selector 保存 batch checkpoint、RNG、游标和自适应显存状态；
5. 完成后自动运行报告导出器。

如果中断，使用同一个 run 目录恢复：

```bash
bash scripts/run_retacred_qceasc_counterfactual_formal_single_seed.sh \
  --gpu auto --hardware-profile adaptive \
  --resume runs/retacred_qceasc_counterfactual_formal_single_seed/<原时间戳>_seed13
```

只有明确允许 GPU 拓扑改变时才加 `--allow-gpu-topology-change`。不要改 seed、数据、batch、epoch、selector、baseline 或 kernel 参数；不要删除 checkpoint 后假装续跑。

## Case Study 证据

配置冻结 train/valid/test 各 3 个样本，并在 `initial_or_pre_training`、`best_valid_or_declared_selection_checkpoint`、`final` 三个 checkpoint 重放。每个 selector 的 raw run 会写：

- `case_study.json`：安全投影，包含原句、tokens/token IDs、attention mask、subject/object 文本与 span、gold relation、disabled/selector logits/probabilities、预测和正确性；
- `sample_trace.json`：结构化 `sample-trace.v1` 链路；
- `case_study_tensors/`：仅保留在 raw run 的 detached tensor captures，带 shape、dtype、axis semantics、SHA-256 和字节数。

报告导出只复制安全投影、结构化 trace、metrics、summary、配置和 provenance，不复制 checkpoint、数据、预测全集、tensor 原文件或完整日志。

## 完成后的回传

导出器会在合作者的 `1.1` 分支生成：

```text
reports/retacred_qceasc_counterfactual_formal_single_seed/<时间戳>_seed13/
```

合作者只提交该 `reports/` 子目录：

```bash
git status --short --branch
git diff --cached --check
git add reports/retacred_qceasc_counterfactual_formal_single_seed/<时间戳>_seed13
git commit -m "Add Q-CEASC counterfactual formal single-seed report"
git push origin 1.1
```

项目方审计报告完整性、配置/代码 provenance、Case Study 语义覆盖和 candidate 相对 disabled/matched control 的指标后，才决定是否进入多 seed；本单 seed 本身不授权多 seed 复制。
