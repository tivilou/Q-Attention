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

可以通过 `PYTHON_BIN=/path/to/python` 指定环境解释器。若已有经过审计的冻结 baseline checkpoint，可使用 `--model-dir` 跳过训练，仅重新评估三个 split：

```bash
bash scripts/run_retacred_baseline_complete.sh \
  --seed 13 \
  --gpu 0 \
  --model-dir runs/<existing-baseline-dir>
```

runner 在完整评估成功后自动生成 `RUN_COMPLETE`，并写出 `run_summary.data`。报告导出只允许安全摘要文件：

```bash
bash scripts/export_retacred_baseline_report.sh \
  runs/retacred_baseline_complete/<timestamp>_seed13
git add reports/retacred_baseline_complete/<timestamp>_seed13
git diff --cached --check
git commit -m "report: add complete Re-TACRED baseline evaluation"
git push origin 1.1
```

不要提交 `runs/`、checkpoint、预测文件、数据集或完整日志。exporter 会拒绝缺失/空的 `run_summary.data`、不完整指标、失败运行和私有文件。

## 报告文件

报告目录包含 `baseline_metrics.json`、`run_summary.json`、`run_summary.data`、`run_summary.md`、`run_config.json`、`data_counts.txt`、`data.sha256`、`export_commit.txt` 和日志 tail。`run_summary.data` 是身份凭证，不是训练数据；其中记录 seed、checkpoint 摘要、三个 split 的记录数、数据哈希和评估批次数。

## 外部 SOTA/reference

外部 reference baseline 尚未在本协议中宣称已复现。候选、任务可比性、官方来源、待锁定 source commit、checkpoint、适配边界和 fidelity gate 统一登记在 [`configs/retacred_reference_baselines.json`](../../configs/retacred_reference_baselines.json)。只有完成 source/paper fidelity 审计并通过共同协议，才能进入正式比较表；没有实测分数的条目必须保持 `planned` 或 `not_verifiable`。

## 状态门槛

该 baseline 报告用于补齐相对提升比较的参照指标，不改变量子方法的 L1/L2 判定规则。报告审计完成前不发布绝对 SOTA 结论，也不因外部 reference 尚未复现而伪造对比结果。
