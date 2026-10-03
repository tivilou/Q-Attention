# Re-TACRED 核心 baseline 完整评估

## 目的

本入口补齐本项目当前关系识别模型的 `Disabled / baseline` 评估。它与量子或经典 selector 使用相同的 Re-TACRED 数据划分和标签定义，输出统一的可审计指标，但不把未完成的外部 SOTA 复现写成已完成结果。

正式全量训练和评估由合作者执行；项目方只负责代码、契约、toy/preflight 和报告审计。

## 输出指标

对 `train`、`valid`、`test` 三个 split 都记录：

- micro precision、micro recall、micro F1；
- accuracy；
- macro precision、macro recall、macro F1；
- 平均交叉熵 loss；
- 每个关系类别的 precision、recall、F1、support、predicted support；
- 混淆矩阵、数据文件 SHA-256、checkpoint SHA-256 和 Git provenance。

`valid` 的 macro-F1 再以 loss 作为 tie-breaker 选择 checkpoint；`test` 只做最终评估，不参与训练或 checkpoint 选择。单标签多分类下 micro-F1 与 accuracy 数值相同是定义结果，不是漏记指标。

## 合作者执行

合作者在自己的 `1.1` 工作区同步 `origin/main` 后，先运行：

```bash
git fetch origin --prune
git merge origin/1.1
git merge origin/main
bash scripts/check_retacred_baseline_complete.sh
```

随后执行 seed 13 的正式核心 baseline：

```bash
bash scripts/run_retacred_baseline_complete.sh \
  --seed 13 \
  --gpu 0
```

可以通过 `PYTHON_BIN=/path/to/python` 指定环境解释器。若已有经过审计的冻结 baseline checkpoint，可使用 `--model-dir` 跳过训练，仅重新评估三个 split。runner 会在创建新 run 之前验证 checkpoint 文件、seed 和冻结训练参数是否匹配：

```bash
bash scripts/run_retacred_baseline_complete.sh \
  --seed 13 \
  --gpu 0 \
  --model-dir runs/<existing-baseline-dir>
```

runner 在完整评估成功后自动生成 `RUN_COMPLETE`，写出 `run_summary.data`，调用 exporter 生成安全报告，并只暂存对应报告目录后提交、推送到 `origin/1.1`。发布由 `scripts/publish_retacred_baseline_report.sh` 执行；如果 commit 或 push 失败，修复原因后用同一个 raw run 重新调用该脚本，它会重新检查暂存范围或复用已完成的 commit，不会重跑训练：

```bash
bash scripts/publish_retacred_baseline_report.sh \
  --run-dir runs/retacred_baseline_complete/<timestamp>_seed13
```

导出、staged diff 检查、commit 和 push 任一步失败都会停止后续动作。训练和评估不会因为推送失败而重跑；报告、已完成的本地 commit 和状态历史会保留。`EXPORT_COMPLETE`、`COMMIT_COMPLETE` 和 `PUSH_COMPLETE` 分别记录在 raw run 中，便于定位中断位置。

诊断模式：`--skip-export` 只完成 raw run；`--no-push` 完成导出和本地 commit，但不联网推送。独立审计或历史 run 仍可直接调用：

```bash
bash scripts/export_retacred_baseline_report.sh \
  runs/retacred_baseline_complete/<timestamp>_seed13
```

不要提交 `runs/`、checkpoint、预测文件、数据集或完整日志。exporter 会拒绝缺失/空的 `run_summary.data`、不完整指标、失败运行和私有文件；runner 只会暂存 `reports/retacred_baseline_complete/<timestamp>_seed13/`。

## 报告文件

报告目录包含 `baseline_metrics.json`、`run_summary.json`、`run_summary.data`、`run_summary.md`、`run_config.json`、`data_counts.txt`、`data.sha256`、`export_commit.txt` 和日志 tail。`run_summary.data` 是身份凭证，不是训练数据；其中记录 seed、checkpoint 摘要、三个 split 的记录数、数据哈希和评估批次数。

## 外部 SOTA/reference

外部 reference baseline 尚未在本协议中宣称已复现。候选、任务可比性、官方来源、待锁定 source commit、checkpoint、适配边界和 fidelity gate 统一登记在 [`configs/retacred_reference_baselines.json`](../../configs/retacred_reference_baselines.json)。只有完成 source/paper fidelity 审计并通过共同协议，才能进入正式比较表；没有实测分数的条目必须保持 `planned` 或 `not_verifiable`。

## 状态门槛

该 baseline 报告用于补齐相对提升比较的参照指标，不改变量子方法的 L1/L2 判定规则。报告审计完成前不发布绝对 SOTA 结论，也不因外部 reference 尚未复现而伪造对比结果。
