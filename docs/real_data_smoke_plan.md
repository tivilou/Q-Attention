# Real-Data Smoke Plan

## Purpose

This stage moves Q-Attention from toy-only prototypes toward a formal experiment handoff. The goal is not to report paper results yet. The goal is to prove that real relation extraction data can enter the same frozen-backbone steering chain:

```text
canonical real data -> baseline -> classical projector -> quantum projector -> spectral sweep -> routing eval
```

## Supported Input Formats

The converter currently supports:

```text
project_jsonl          # current canonical Q-Attention JSONL
TACRED-style JSON      # --format tacred_json
TACRED-style JSONL     # --format tacred_jsonl
SemEval-2010 Task 8    # --format semeval2010_task8
```

Canonical JSONL uses zero-based, end-exclusive spans:

```json
{"tokens": ["The", "company", "acquired", "startup"], "subject": [1, 2], "object": [3, 4], "label": "org:acquired"}
```

## Dataset Preparation

Example for TACRED/Re-TACRED style JSON files:

```bash
python experiments/prepare_relation_data.py \
  --format tacred_json \
  --dataset_name retacred \
  --train_path data/raw/retacred/train.json \
  --valid_path data/raw/retacred/dev.json \
  --test_path data/raw/retacred/test.json \
  --output_dir data/relation/retacred
```

Example for SemEval-2010 Task 8:

```bash
python experiments/prepare_relation_data.py \
  --format semeval2010_task8 \
  --dataset_name semeval2010_task8 \
  --train_path data/raw/semeval2010/TRAIN_FILE.TXT \
  --valid_path data/raw/semeval2010/TEST_FILE_FULL.TXT \
  --output_dir data/relation/semeval2010_task8
```

The converter writes:

```text
data/relation/<dataset>/train.jsonl
data/relation/<dataset>/valid.jsonl
data/relation/<dataset>/test.jsonl       # when --test_path is provided
data/relation/<dataset>/data_config.json
```

## Real-Data Smoke Pipeline

After conversion, run a tiny smoke chain first:

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config data/relation/retacred/data_config.json \
  --output_dir runs/retacred_real_smoke \
  --device cpu \
  --max_train_records 256 \
  --max_valid_records 128
```

For command generation only:

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config data/relation/retacred/data_config.json \
  --output_dir runs/retacred_real_smoke \
  --device cpu \
  --dry_run
```

Expected outputs include:

```text
runs/<name>/baseline/model.pt
runs/<name>/baseline/relation_projector.pt
runs/<name>/baseline/relation_quantum_projector.pt
runs/<name>/classical_steering_eval/metrics.json
runs/<name>/quantum_steering_eval/metrics.json
runs/<name>/spectral_filter_sweep/results.jsonl
runs/<name>/relation_routing_eval/metrics.json
runs/<name>/pipeline_summary.json
```

## Handoff Gate

Do not hand this to the experiment runner for full GPU benchmarking until all checks pass:

```text
converter works on the selected real dataset
small real-data smoke pipeline completes
pipeline_summary.json records exact commands
pytest passes
result files contain baseline, steering, sweep, and routing metrics
no placeholder paths remain in the selected run config
```

## Current Division of Work

Coding side:

```text
maintain converters, configs, smoke pipeline, tests, and debugging fixes
```

Experiment runner:

```text
prepare candidate datasets, record environment/GPU details, and run only the exact commands provided after the handoff gate is met
```