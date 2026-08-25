# Query-conditioned Q-SRPA Re-TACRED 正式单 seed 实验

固定 seed 13、单 GPU、串行执行完整 Re-TACRED。比较 disabled/global-context、legacy soft-role pair、query-conditioned soft-role pair 及其参数匹配 classical controls。query-conditioned role router 只读取 Q/K 和 padding mask；subject/object spans 不得进入 action path。

脚本会使用当前环境中的 `python`，找不到时自动尝试 `python3`。如果环境中有多个解释器，可在命令前用 `PYTHON_BIN=...` 临时指定；文档不假定任何固定绝对路径。协作者必须在 `1.1` 分支执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
bash scripts/check_retacred_qsrpa_query_conditioned_formal_single_seed.sh
bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh --gpu 0
```

runner 完成全部训练和评估后，会先生成 `RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md`，再自动调用 exporter。默认报告目录为 `reports/retacred_qsrpa_query_conditioned_formal_single_seed/`；如需指定目录，直接把 `--report-dir reports/retacred_qsrpa_query_conditioned_formal_single_seed/<name>` 传给 runner。exporter 会再次检查完整性、分支、clean 状态和私有文件，随后只提交并 push 审计后的报告子集，不提交 runs、数据、checkpoint、预测或完整日志。

例如：

```bash
bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh \
  --gpu 0 \
  --report-dir reports/retacred_qsrpa_query_conditioned_formal_single_seed/seed13-fixed
```

如果 runner 已经完成但自动导出因网络或 GitHub 临时失败，可在确认 `RUN_COMPLETE` 和汇总文件齐全后单独重试 exporter：

```bash
bash scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh \
  --run-dir runs/retacred_qsrpa_query_conditioned_formal_single_seed/<timestamp>_seed13
```

candidate 相对 disabled 的 valid/test macro-F1 必须均达到 `+0.001`，且相对 matched classical 不低于 `-0.0005`。任一门槛未达到即停止，不启动 multi-seed replication。
