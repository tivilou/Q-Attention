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

```bash
REPORT_TAG=$(date +%Y%m%d-%H%M%S)-corrected
REPORT_DIR=reports/retacred/${REPORT_TAG}
mkdir -p "${REPORT_DIR}"/{configs,full,low_resource,logs}

cp configs/retacred_full_gpu.json "${REPORT_DIR}/configs/"
cp configs/retacred_low_resource_gpu.json "${REPORT_DIR}/configs/"
python -m json.tool "${REPORT_DIR}/configs/retacred_full_gpu.json" >/dev/null
python -m json.tool "${REPORT_DIR}/configs/retacred_low_resource_gpu.json" >/dev/null
```

```bash
cp "${FULL_RUN}/pipeline_summary.json" "${REPORT_DIR}/full/"
cp "${FULL_RUN}/run_summary.json" "${REPORT_DIR}/full/"
cp "${FULL_RUN}/run_summary.md" "${REPORT_DIR}/full/"
cp "${FULL_RUN}/baseline/metrics.json" "${REPORT_DIR}/full/baseline_metrics.json"
cp "${FULL_RUN}/classical_steering_eval/metrics.json" "${REPORT_DIR}/full/classical_steering_metrics.json"
cp "${FULL_RUN}/quantum_steering_eval/metrics.json" "${REPORT_DIR}/full/quantum_steering_metrics.json"
cp "${FULL_RUN}/spectral_filter_sweep/summary.json" "${REPORT_DIR}/full/spectral_filter_summary.json"
cp "${FULL_RUN}/relation_routing_eval/metrics.json" "${REPORT_DIR}/full/routing_metrics.json"
tail -n 1000 "${FULL_LOG}" > "${REPORT_DIR}/logs/retacred_full_gpu.tail.txt"
```

```bash
cp "${LOW_RUN}/pipeline_summary.json" "${REPORT_DIR}/low_resource/"
cp "${LOW_RUN}/run_summary.json" "${REPORT_DIR}/low_resource/"
cp "${LOW_RUN}/run_summary.md" "${REPORT_DIR}/low_resource/"
cp "${LOW_RUN}/baseline/metrics.json" "${REPORT_DIR}/low_resource/baseline_metrics.json"
cp "${LOW_RUN}/classical_steering_eval/metrics.json" "${REPORT_DIR}/low_resource/classical_steering_metrics.json"
cp "${LOW_RUN}/quantum_steering_eval/metrics.json" "${REPORT_DIR}/low_resource/quantum_steering_metrics.json"
cp "${LOW_RUN}/spectral_filter_sweep/summary.json" "${REPORT_DIR}/low_resource/spectral_filter_summary.json"
cp "${LOW_RUN}/relation_routing_eval/metrics.json" "${REPORT_DIR}/low_resource/routing_metrics.json"
tail -n 1000 "${LOW_LOG}" > "${REPORT_DIR}/logs/retacred_low_resource_gpu.tail.txt"
```

## 4. 提交到 1.1

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add corrected Re-TACRED results ${REPORT_TAG}"
git push origin 1.1
```

暂存区只能包含本次 `reports/retacred/${REPORT_TAG}/`。不要提交 `data/`、`runs/`、模型权重、predictions、JSONL 或完整日志。推送后把 commit hash 发回。
