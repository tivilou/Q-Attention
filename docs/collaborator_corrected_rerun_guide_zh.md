# 本次 Re-TACRED 重跑与提交

本次直接使用现有项目、conda 环境和数据，不重新 clone，不修改源码和配置。

## 1. 同步代码

```bash
cd ~/projects/Q-Attention
git fetch origin --prune
git switch 1.1
git status --short
git merge origin/main
git status --short --branch
```

运行前 `git status --short` 必须没有输出。merge 有冲突时停止并报告。

## 2. 运行实验

```bash
conda activate <你的实验环境>
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
mkdir -p runs/handoff_logs
set -o pipefail
```

### Full

```bash
FULL_LOG=runs/handoff_logs/retacred_full_gpu_$(date +%Y%m%d-%H%M%S).log

python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda \
  2>&1 | tee "${FULL_LOG}"

FULL_RUN=$(ls -dt runs/retacred_full_gpu/*/ | head -n 1)
echo "FULL_RUN=${FULL_RUN}"
```

### Low-resource

```bash
LOW_LOG=runs/handoff_logs/retacred_low_resource_gpu_$(date +%Y%m%d-%H%M%S).log

python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee "${LOW_LOG}"

LOW_RUN=$(ls -dt runs/retacred_low_resource_gpu/*/ | head -n 1)
echo "LOW_RUN=${LOW_RUN}"
```

不要添加 `--output_dir` 或 `--max_valid_records`。脚本会自动创建：

```text
runs/retacred_full_gpu/<日期-时间>/
runs/retacred_low_resource_gpu/<日期-时间>/
```

每次运行都会使用新目录，不会覆盖上一次结果。实验成功结束时，脚本会自动生成 `run_summary.json` 和 `run_summary.md`，不需要再手动运行汇总命令。

检查结果：

```bash
test -f "${FULL_RUN}/pipeline_summary.json"
test -f "${FULL_RUN}/run_summary.json"
test -f "${FULL_RUN}/run_summary.md"
test -f "${LOW_RUN}/pipeline_summary.json"
test -f "${LOW_RUN}/run_summary.json"
test -f "${LOW_RUN}/run_summary.md"
```

如果实验已成功但 summary 文件缺失，再执行：

```bash
python experiments/summarize_relation_run.py --run_dir "${FULL_RUN}"
python experiments/summarize_relation_run.py --run_dir "${LOW_RUN}"
```

## 3. 整理报告

不要使用个人 `copy.sh`。full 和 low-resource 都成功后执行：

```bash
git status --short
REPORT_TAG=$(date +%Y%m%d-%H%M%S)
REPORT_DIR=reports/retacred/${REPORT_TAG}

python scripts/export_retacred_report.py \
  --full-run "${FULL_RUN}" \
  --low-resource-run "${LOW_RUN}" \
  --full-log "${FULL_LOG}" \
  --low-resource-log "${LOW_LOG}" \
  --report-tag "${REPORT_TAG}"
```

运行前 `git status --short` 必须为空；成功时应输出 `Files exported: 20`。脚本报错时停止并把报错发回，不要手工绕过。

## 4. 提交到 1.1

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git status --short
git commit -m "Add Re-TACRED ${REPORT_TAG} results"
git push origin 1.1
git rev-parse HEAD
```

只提交本次报告目录中的 20 个文件，不要使用 `git add .`。不要提交 `data/`、`runs/`、模型权重、predictions、JSONL 或完整日志。
