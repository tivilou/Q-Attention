# Q-TRIAD Re-TACRED 正式单 seed 实验

这是完整 Re-TACRED 的 seed 13、单 GPU、串行正式实验。Q-TRIAD candidate 使用 label-free 的 query、key 和 subject/object relation anchor；`classical_density_tensor` 是参数和输入匹配的经典密度控制，`quantum_product` 是无三方纠缠的量子 product control，`disabled` 是基线。完整数据正式运行必须由合作者在 `1.1` 分支执行。

## 1. 同步并执行

在项目根目录执行。下面的默认命令会检查分支、数据、环境和测试，运行 baseline 及三个 selector，写入摘要，并在完成标记生成后自动导出报告、提交并推送到 `origin/1.1`。不要在标准命令中加入 `--report-dir`。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
bash scripts/run_retacred_qtriad_formal_single_seed.sh --gpu 0
```

`git status --short` 在启动前必须没有输出。脚本在启动时记录一个 UTC 时间戳 `YYYYMMDDTHHMMSSZ`，raw run 默认写入：

```text
runs/retacred_qtriad_formal_single_seed/<timestamp>_seed13/
```

脚本从当前环境解析 Python：优先使用 `PYTHON_BIN`，否则依次查找 `python`、`python3`；不得写入或假定负责人服务器的解释器路径。默认报告目录由同一个启动时间戳确定：

```text
reports/retacred_qtriad_formal_single_seed/<timestamp>_seed13/
```

运行期间不要改动 seed、数据、selector、epoch、batch size、控制组或代码，也不要并行启动第二个完整 run。

## 2. 完成检查

运行成功后 raw run 必须包含：

```text
RUN_COMPLETE
run_summary.json
run_summary.md
baseline/metrics.json
selectors/q_triad/metrics.json
selectors/classical_density_tensor/metrics.json
selectors/quantum_product/metrics.json
```

摘要必须声明 seed 13、完整数据计数 `58465/19584/13418`，且 test 未用于训练或 valid checkpoint 选择。报告 exporter 还会检查 branch ancestry、clean tree、三份数据行数、完整 selector 指标、provenance 和私有文件禁带。

## 3. 报告提交

一键脚本已经调用 exporter。exporter 只复制 `reports/retacred_qtriad_formal_single_seed/<timestamp>_seed13/` 下的审计子集，并自动执行 `git add`、`git diff --cached --check`、commit 和 `git push origin 1.1`。不得提交 `runs/`、`data/`、checkpoint、预测、JSONL 或完整日志。

完成后检查提交结果：

```bash
git status --short
git diff --cached --name-only
git log -1 --oneline
git ls-remote --heads origin 1.1
```

正常情况下前两条没有输出，最后两条应显示本次报告提交和更新后的 `origin/1.1`。如果自动 push 失败，保留 raw run 和已生成的报告目录，先记录错误；不要重跑、改写报告、强推或切换到 `main`。修复网络或分支同步后，只能在 clean `1.1` 上重新执行 exporter。

只有在需要审计而不提交时，才使用 exporter 的 `--no-commit`；显式 `--report-dir` 仅用于经负责人确认的例外目录，不是标准路径。

## 4. 停止门禁

负责人审计报告中的 valid/test 指标、控制组、数据 hash、provenance、test isolation 和 staged 文件后，才形成项目结论。单 seed 结果未通过预声明的 candidate-vs-disabled 实用增益门禁或 matched classical comparator 门禁时，立即停止；即使中间日志看起来有提升，也不得启动 multi-seed。只有负责人完成审计并明确授权后，才会发布后续 multi-seed handoff。
