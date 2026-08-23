# Q-SRPA Re-TACRED 正式单 seed 实验

固定 seed 13、单 GPU、串行执行完整 Re-TACRED：disabled baseline、现有 Q-LASS `global_context`、其参数匹配 classical control、Q-SRPA `soft_role_pair` 以及参数匹配 classical SRPA。Q-SRPA role router 只读取冻结 Q/K 和 padding mask；subject/object spans 只能用于离线审计，不能进入 action path。所有 kernel 只在 valid 选择 checkpoint，test 只在最后评估。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
bash scripts/check_retacred_qsrpa_formal_single_seed.sh
bash scripts/run_retacred_qsrpa_formal_single_seed.sh --gpu 0
```

合作者的实验和报告准备始终在 `1.1` 分支进行；`main` 只作为负责人发布的代码来源，不在其上开发或提交实验结果。若合并出现冲突，立即停止并反馈。

运行目录在 `runs/retacred_qsrpa_formal_single_seed/`。完成后必须存在 `RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md`。交付时只提交审计后的 report-only 文件，不提交 `runs/`、数据、checkpoint、预测或完整日志。

在 `1.1` 分支合并 `origin/main` 后执行：

```bash
bash scripts/export_retacred_qsrpa_formal_single_seed_report.sh \
  --run-dir runs/retacred_qsrpa_formal_single_seed/<timestamp>_seed13
```

导出脚本只提交 `reports/retacred_qsrpa_formal_single_seed/` 下的审计文件并推送 `origin/1.1`。

判定门槛：Q-SRPA 相对 disabled 的 valid/test macro-F1 至少 `+0.001`，且相对 classical SRPA 不低于 `-0.0005`。未达到门槛时停止；只有正式单 seed 通过后才预声明五 seed replication。
