# Re-TACRED Experiment Handoff

This handoff is for `dzy958`. The coding side has already verified that Re-TACRED can be built from licensed TACRED, converted to Q-Attention JSONL, and run through the full prototype chain on a small real-data subset.

If you need a Chinese overview of the project background and how this differs from the previous reference-paper experiments, read [retacred_collaborator_overview_zh.md](retacred_collaborator_overview_zh.md) first.

## Current Gate Status

```text
Dataset: Re-TACRED canonical JSONL exists locally under data/relation/retacred
Smoke gate: passed on 256 train / 128 valid examples
Formal GPU handoff: ready for debug run, then low-resource/full runs
Do not commit: any files under data/ or runs/
```

## Required Inputs

Before running, confirm these files exist on the experiment machine:

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
data/relation/retacred/data_config.json
```

### Option A: Use The Prepared Data Package

If you receive `retacred_q_attention_data.tar.gz`, place it at the Q-Attention repository root and verify the checksum if `retacred_q_attention_data.sha256` is provided:

```bash
sha256sum retacred_q_attention_data.tar.gz
cat retacred_q_attention_data.sha256
```

Expected SHA256:

```text
c06b9647b5977a3a06fd2a4f338b50931567def231d77a37c8fe6bf93a36a64c
```

Then extract it from the repository root:

```bash
tar -xzf retacred_q_attention_data.tar.gz
```

After extraction, verify the canonical files:

```bash
ls -lh data/relation/retacred
python - <<'PY'
from pathlib import Path
for name in ['train.jsonl', 'valid.jsonl', 'test.jsonl', 'data_config.json']:
    path = Path('data/relation/retacred') / name
    print(name, path.exists(), path.stat().st_size if path.exists() else 'missing')
PY
```

Do not commit the package or extracted `data/` directory; `data/` is intentionally ignored.

### Option B: Rebuild From Licensed TACRED

If the prepared package is not available, build the data from licensed TACRED plus the public Re-TACRED patches:

```bash
python experiments/build_retacred_from_tacred.py \
  --tacred_dir data/raw/tacred/data/json \
  --patch_dir data/raw/Re-TACRED-source/Re-TACRED \
  --output_dir data/raw/Re-TACRED-patched-jsonl

python experiments/prepare_relation_data.py \
  --format tacred_jsonl \
  --dataset_name retacred \
  --train_path data/raw/Re-TACRED-patched-jsonl/train.jsonl \
  --valid_path data/raw/Re-TACRED-patched-jsonl/dev.jsonl \
  --test_path data/raw/Re-TACRED-patched-jsonl/test.jsonl \
  --output_dir data/relation/retacred
```

## Environment Log

Run these before experiments and include the outputs in the report:

```bash
git rev-parse HEAD
which python
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi
python -m pytest -q
```

## Step 1: GPU Debug Run

This verifies the collaborator environment without committing to a long run.

```bash
mkdir -p runs/handoff_logs
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_debug_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_debug_gpu
```

Expected summary files:

```text
runs/retacred_debug_gpu/run_summary.json
runs/retacred_debug_gpu/run_summary.md
```

## Step 2: Low-Resource Run

This is useful for the paper's low-resource ablation and should be run after the debug run passes.

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_low_resource_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_low_resource_gpu
```

## Step 3: Full Validation Run

Run this only after the debug and low-resource commands are clean.

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_full_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_full_gpu
```

## Required Artifacts To Return

For every run, return or summarize:

```text
runs/<run_name>/pipeline_summary.json
runs/<run_name>/baseline/metrics.json
runs/<run_name>/classical_steering_eval/metrics.json
runs/<run_name>/quantum_steering_eval/metrics.json
runs/<run_name>/supervised_quantum_gain_selection/gain_selection.json
runs/<run_name>/supervised_quantum_gain_selection/metrics.json
runs/<run_name>/supervised_quantum_gain_selection/run_info.json
runs/<run_name>/supervised_quantum_gain_selection/predictions.jsonl
runs/<run_name>/spectral_filter_sweep/summary.json
runs/<run_name>/relation_routing_eval/metrics.json
runs/<run_name>/run_summary.md
runs/handoff_logs/<run_name>.log
```

## Failure Report

If any command fails, stop and report:

```text
commit hash
exact command
GPU model
last 100 lines of the log
full traceback
whether the failure happens again after rerun
```

Do not change source code or configs while debugging the experiment environment. Report the failure to the coding side instead.
