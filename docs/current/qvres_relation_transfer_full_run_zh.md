# Q-VRES Re-TACRED 正式实验

诊断优先由负责人执行；如果正式 raw run 只在合作者机器上，则由合作者在原目录执行下面的只读诊断。不要重新训练 seed 13，也不要启动五 seed 实验。

## 1. 同步代码

在项目根目录执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
```

最后一条命令必须没有输出。

## 2. 定位 seed 13 raw run

`PILOT_DIR` 是已经完成的 raw 输出目录，不是 GitHub 中的 `reports/` 目录，也不要求位于当前代码仓库下。先查找它：

```bash
pwd
find "$HOME" -type d -name '*_seed13_selector_parallel' -print | sort
```

找到后使用绝对路径，例如：

```bash
PILOT_DIR=/root/projects/Q-Attention/runs/q_vres_relation_transfer_full/20260815T235926Z_seed13_selector_parallel
echo "PILOT_DIR=${PILOT_DIR}"
test -f "${PILOT_DIR}/RUN_COMPLETE"
test -f "${PILOT_DIR}/stages/baseline/baseline/model.pt"
test -f "${PILOT_DIR}/stages/q_causal_transport/selectors/q_causal_transport/best_kernel.pt"
```

如果 `find` 没有输出，说明 raw run 不在当前用户的 home 目录中，不能仅凭 GitHub 上的报告进行正式诊断。不要重新创建空目录，也不要立即重跑训练，先确认原实验实际保存位置。

不要删除或移动这个目录。诊断需要读取其中的 baseline 和 selector checkpoint，但不会把它们复制到报告中。

## 3. 运行 validation 诊断

持有 raw run 的一方在对应项目根目录激活 Conda 环境，然后执行：

```bash
PYTHON_BIN=python bash scripts/run_qvres_validation_diagnostic.sh \
  "${PILOT_DIR}" \
  --gpu 0 \
  --batch-size 8 \
  --log-every-batches 50
```

这是只读诊断，不重新训练模型。脚本依次检查 baseline、Q causal、classical control 和 Q key-only，并在终端显示 batch、速度和 ETA。

完成后终端会打印：

```text
REPORT_DIR=reports/q_vres_relation_transfer/时间戳-validation-diagnostic
```

## 4. 查看并提交诊断结果

```bash
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/*-validation-diagnostic | head -n 1)
cat "${REPORT_DIR}/diagnostic_summary.md"

cat "${REPORT_DIR}/diagnostic_summary.json"

git add \
  "${REPORT_DIR}/diagnostic_summary.json" \
  "${REPORT_DIR}/diagnostic_summary.md"

git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES seed 13 validation diagnostics"
git push origin 1.1
```

`git diff --cached --name-only` 必须只有上述两个文件。负责人负责审查和归档诊断报告。

不要提交 `data/`、`runs/`、checkpoint、逐样本预测、JSONL、梯度张量或完整日志。

## 5. 后续实验

负责人审查诊断结果前，合作者不要运行：

```text
scripts/run_qvres_relation_transfer_multi_seed.sh
```

以后复用已有 seed 时，将比较实验有效代码、配置、数据 hash、seed 和关键参数的指纹，不再只比较整个 Git `HEAD`。该复用逻辑完成前，不要自行使用 `--reuse-seed`。
