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
  --device cuda 2>&1 | tee runs/handoff_logs/retacred_debug_gpu.log
python experiments/summarize_relation_run.py --run_dir runs/retacred_debug_gpu
```

### Low-resource

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda 2>&1 | tee runs/handoff_logs/retacred_low_resource_gpu.log
python experiments/summarize_relation_run.py --run_dir runs/retacred_low_resource_gpu
```

不要手动增加 low-resource 的 `max_valid_records`。

### Full

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda 2>&1 | tee runs/handoff_logs/retacred_full_gpu.log
python experiments/summarize_relation_run.py --run_dir runs/retacred_full_gpu
```

## 4. 提交到 GitHub 的文件

每次使用新的日期目录，例如：

```text
reports/retacred/2026-07-20-corrected/
├── configs/retacred_full_gpu.json
├── configs/retacred_low_resource_gpu.json
├── full/pipeline_summary.json
├── full/run_summary.json
├── full/run_summary.md
├── full/baseline_metrics.json
├── full/classical_steering_metrics.json
├── full/quantum_steering_metrics.json
├── full/spectral_filter_summary.json
├── full/routing_metrics.json
├── low_resource/pipeline_summary.json
├── low_resource/run_summary.json
├── low_resource/run_summary.md
├── low_resource/baseline_metrics.json
├── low_resource/classical_steering_metrics.json
├── low_resource/quantum_steering_metrics.json
├── low_resource/spectral_filter_summary.json
├── low_resource/routing_metrics.json
└── logs/retacred_*.tail.txt
```

配置文件必须从 `configs/` 目录复制，并确认是合法 JSON：

```bash
python -m json.tool reports/retacred/<date>/configs/retacred_full_gpu.json >/dev/null
python -m json.tool reports/retacred/<date>/configs/retacred_low_resource_gpu.json >/dev/null
```

## 5. 提交命令

```bash
git add reports/retacred/<date>
git diff --cached --check
git status --short
git commit -m "Add Re-TACRED <date> results"
git push origin 1.1
```

只提交新的 `reports/` 目录。不要提交：

```text
data/
runs/
*.pt
*.pth
*.ckpt
*.jsonl
predictions.jsonl
源码或配置源码修改
```

代码侧审核后再把 `1.1` 合并到 `main`。

## 6. 诊断说明

日常结果诊断只需要 GitHub 上提交的 `reports/` 内容，尤其是 `pipeline_summary.json`、`run_summary.json`、`run_summary.md`、配置文件和日志尾部。

只有在需要重新运行 gain selection、复现模型结果，或发现报告缺文件/commit 不一致时，才需要私下提供模型权重、projector、完整日志或数据。私有材料不提交 GitHub。
