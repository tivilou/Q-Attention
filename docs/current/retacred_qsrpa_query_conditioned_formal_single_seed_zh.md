# Query-conditioned Q-SRPA Re-TACRED 正式单 seed 实验

固定 seed 13、单 GPU、串行执行完整 Re-TACRED。比较 disabled/global-context、legacy soft-role pair、query-conditioned soft-role pair 及其参数匹配 classical controls。query-conditioned role router 只读取 Q/K 和 padding mask；subject/object spans 不得进入 action path。

脚本会使用当前环境中的 `python`，找不到时自动尝试 `python3`。如果环境中有多个解释器，可在命令前用 `PYTHON_BIN=...` 临时指定；文档不假定任何固定绝对路径。协作者必须在 `1.1` 分支执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
bash scripts/check_retacred_qsrpa_query_conditioned_formal_single_seed.sh
bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh --gpu 0
```

runner 启动时只记录一次 UTC 时间戳，并用它命名运行目录和默认报告目录；在首个训练阶段前还会把执行 commit、分支和 Python 解释器写入 raw run 的 `provenance.env`。完成全部训练和评估后，会先生成 `RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md`，再自动调用 exporter。未指定时，报告目录为 `reports/retacred_qsrpa_query_conditioned_formal_single_seed/<启动时间戳>_seed13/`。exporter 会再次检查完整性、`origin/1.1` 与 `origin/main` ancestry、分支、clean 状态、执行 commit provenance 和私有文件，随后只提交并 push 审计后的报告子集，不提交 runs、数据、checkpoint、预测或完整日志。显式目录覆盖只作为例外，见后文。

标准运行命令不需要指定报告目录：

```bash
bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh --gpu 0
```

runner 会把默认报告目录传给 exporter；如果 runner 已经完成但自动导出因网络或 GitHub 临时失败，可在确认 `RUN_COMPLETE` 和汇总文件齐全后，使用相同 raw run 的目录名重试 exporter。exporter 默认会从 raw run 的 `<timestamp>_seed13` 目录名推导同名报告目录，显式 `--report-dir` 只用于确有需要时的例外覆盖：

```bash
RUN_DIR=$(ls -dt runs/retacred_qsrpa_query_conditioned_formal_single_seed/*_seed13 | head -n 1)
bash scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh \
  --run-dir "${RUN_DIR}"
```

实验完成并自动导出后，先核对当前分支、工作树和最近的报告提交：

```bash
RUN_DIR=$(ls -dt runs/retacred_qsrpa_query_conditioned_formal_single_seed/*_seed13 | head -n 1)
REPORT_DIR="reports/retacred_qsrpa_query_conditioned_formal_single_seed/$(basename "${RUN_DIR}")"
test -f "${REPORT_DIR}/run_summary.json"
test -f "${REPORT_DIR}/run_summary.md"
git branch --show-current
git status --short
git log -1 --oneline --decorate
git show --stat --oneline --summary HEAD
```

`git status --short` 应为空，当前分支应为 `1.1`，且当前 `HEAD` 应同时包含 `origin/1.1` 和 `origin/main`。runner 自动导出时已经执行 `git add`、报告提交和 `git push origin 1.1`；如果使用 `--no-commit` 或自动导出失败，则在确认报告目录只含审计允许文件后执行：

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add query-conditioned Q-SRPA formal single-seed report"
git push origin 1.1
```

`git diff --cached --name-only` 只能包含 `reports/retacred_qsrpa_query_conditioned_formal_single_seed/<timestamp>_seed13/` 下的审计报告文件；不得提交 `runs/`、数据、checkpoint、预测或完整日志。

candidate 相对 disabled 的 valid/test macro-F1 必须均达到 `+0.001`，且相对 matched classical 不低于 `-0.0005`。任一门槛未达到即停止，不启动 multi-seed replication。
