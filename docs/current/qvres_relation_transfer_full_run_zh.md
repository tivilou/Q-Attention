# Q-VRES Re-TACRED 正式实验

当前任务是诊断已经完成的 seed 13 pilot。不要重新训练 seed 13，也不要启动五 seed 实验。

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

```bash
PILOT_DIR=$(ls -dt runs/q_vres_relation_transfer_full/*_seed13_selector_parallel | head -n 1)
echo "PILOT_DIR=${PILOT_DIR}"
test -f "${PILOT_DIR}/RUN_COMPLETE"
test -f "${PILOT_DIR}/run_summary.json"
```

不要删除或移动这个目录。诊断需要读取其中的 baseline 和 selector checkpoint，但不会把它们复制到报告中。

## 3. 运行 validation 诊断

先激活自己的 Conda 环境，然后执行：

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

## 4. 查看并提交结果

```bash
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/*-validation-diagnostic | head -n 1)
cat "${REPORT_DIR}/diagnostic_summary.md"

git add \
  "${REPORT_DIR}/diagnostic_summary.json" \
  "${REPORT_DIR}/diagnostic_summary.md"

git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES seed 13 validation diagnostics"
git push origin 1.1
```

`git diff --cached --name-only` 必须只有上述两个文件。

不要提交 `data/`、`runs/`、checkpoint、逐样本预测、JSONL、梯度张量或完整日志。

## 5. 后续实验

负责人审查诊断结果前，不要运行：

```text
scripts/run_qvres_relation_transfer_multi_seed.sh
```

以后复用已有 seed 时，将比较实验有效代码、配置、数据 hash、seed 和关键参数的指纹，不再只比较整个 Git `HEAD`。该复用逻辑完成前，不要自行使用 `--reuse-seed`。
