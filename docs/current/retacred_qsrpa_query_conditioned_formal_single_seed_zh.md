# Query-conditioned Q-SRPA Re-TACRED 正式单 seed 实验

固定 seed 13、单 GPU、串行执行完整 Re-TACRED。比较 disabled/global-context、legacy soft-role pair、query-conditioned soft-role pair 及其参数匹配 classical controls。query-conditioned role router 只读取 Q/K 和 padding mask；subject/object spans 不得进入 action path。

协作者必须在 `1.1` 分支执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
bash scripts/check_retacred_qsrpa_query_conditioned_formal_single_seed.sh
bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh --gpu 0
```

完成后必须存在 `RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md`。只导出审计后的 `reports/retacred_qsrpa_query_conditioned_formal_single_seed/` 文件，不提交 runs、数据、checkpoint、预测或完整日志：

```bash
bash scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh \
  --run-dir runs/retacred_qsrpa_query_conditioned_formal_single_seed/<timestamp>_seed13
```

candidate 相对 disabled 的 valid/test macro-F1 必须均达到 `+0.001`，且相对 matched classical 不低于 `-0.0005`。任一门槛未达到即停止，不启动 multi-seed replication。
