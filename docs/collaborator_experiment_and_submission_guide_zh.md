# Re-TACRED 实验与提交步骤

基于当前 `main`。使用 `1.1` 分支跑实验和提交报告，不修改源码和配置。

## 1. 同步代码

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short --branch
```

确认工作树干净后再继续。merge 有冲突时停止并报告。

## 2. 检查环境

```bash
python -m pytest -q
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
ls data/relation/retacred/*.jsonl
```

CUDA 不可用、pytest 失败或数据文件缺失时停止。

## 3. 运行实验

```bash
mkdir -p runs/handoff_logs
set -o pipefail
```

### Debug

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --device cuda 2>&1 | tee runs/handoff_logs/retacred_debug_gpu_$(date +%Y%m%d-%H%M%S).log
```

### Low-resource

```bash
LOW_LOG=runs/handoff_logs/retacred_low_resource_gpu_$(date +%Y%m%d-%H%M%S).log
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda 2>&1 | tee "${LOW_LOG}"
LOW_RUN=$(ls -dt runs/retacred_low_resource_gpu/*/ | head -n 1)
echo "LOW_RUN=${LOW_RUN}"
```

不要手动增加 low-resource 的 `max_valid_records`。

### Full

```bash
FULL_LOG=runs/handoff_logs/retacred_full_gpu_$(date +%Y%m%d-%H%M%S).log
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda 2>&1 | tee "${FULL_LOG}"
FULL_RUN=$(ls -dt runs/retacred_full_gpu/*/ | head -n 1)
echo "FULL_RUN=${FULL_RUN}"
```

不传 `--output_dir` 时，脚本会自动创建 `runs/<配置名>/<日期-时间>/`，并在成功结束时自动生成 `pipeline_summary.json`、`run_summary.json` 和 `run_summary.md`。只有 summary 缺失时，才手动运行 `summarize_relation_run.py`。

## 4. 使用正式脚本整理报告

不要使用个人 `copy.sh`，也不要手工复制文件。保持第 3 节生成的 `FULL_RUN`、`LOW_RUN`、`FULL_LOG` 和 `LOW_LOG` 变量，然后执行：

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

运行前 `git status --short` 必须为空。脚本成功后会生成本次唯一报告目录，并输出 `Files exported: 20`。报错时停止，不要手工绕过。

## 5. 提交命令

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git status --short
git commit -m "Add Re-TACRED ${REPORT_TAG} results"
git push origin 1.1
git rev-parse HEAD
```

暂存区只能包含本次报告目录中的 20 个文件。不要使用 `git add .`，也不要提交 `data/`、`runs/`、权重、predictions、JSONL、源码或配置修改。

## 6. 诊断说明

日常结果诊断只需要 GitHub 上提交的 `reports/` 内容，尤其是 `pipeline_summary.json`、`run_summary.json`、`run_summary.md`、配置文件和日志尾部。

只有在需要重新运行 gain selection、复现模型结果，或发现报告缺文件/commit 不一致时，才需要私下提供模型权重、projector、完整日志或数据。私有材料不提交 GitHub。
