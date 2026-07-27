# Re-TACRED Dual Q-RES Full 运行指南

本指南用于完整 train/valid 上的 seed 13 主实验预跑。不要使用旧 `run_relation_smoke_pipeline.py` 或 `run_relation_attention_transfer_screen.py`，不要运行 test。

## 1. 运行前检查

在项目根目录执行：

```bash
git status --short
git merge-base --is-ancestor b8d794f HEAD && echo "code version OK"
python -m pytest -q
wc -l data/relation/retacred/train.jsonl
wc -l data/relation/retacred/valid.jsonl
```

要求：

- `git status --short` 没有输出；
- 测试通过；
- train/valid 分别为 `58465` 和 `19584`。

建议在 `tmux` 中运行：

```bash
tmux new -s qattention_full_seed13
```

## 2. 创建本次输出目录

```bash
set -o pipefail
SEED=13
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="runs/retacred_dual_projector_full_${STAMP}_seed${SEED}"
mkdir -p "${RUN_DIR}/logs"
echo "RUN_DIR=${RUN_DIR}"
```

同一终端中的后续命令继续使用这个 `RUN_DIR`。

## 3. 训练 baseline

```bash
python experiments/train_relation_baseline.py \
  --train_path data/relation/retacred/train.jsonl \
  --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${RUN_DIR}/baseline" \
  --epochs 12 --batch_size 128 --lr 0.0005 \
  --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256 \
  --dropout 0.1 --max_length 128 \
  --selection_metric valid_loss --seed "${SEED}" --device cuda \
  2>&1 | tee "${RUN_DIR}/logs/baseline.log"
```

完成后必须存在：

```bash
test -f "${RUN_DIR}/baseline/metrics.json"
test -f "${RUN_DIR}/baseline/model.pt"
```

## 4. 训练 quantum/classical core

```bash
for FAMILY in quantum classical; do
  python experiments/train_relation_attention_score_kernel.py \
    --model_dir "${RUN_DIR}/baseline" \
    --train_path data/relation/retacred/train.jsonl \
    --valid_path data/relation/retacred/valid.jsonl \
    --output_dir "${RUN_DIR}/core/${FAMILY}" \
    --kernel_type "${FAMILY}" \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --score_readout observable --input_encoding factorized_shared \
    --query_scope all --epochs 4 --batch_size 128 --lr 0.001 \
    --selection_metric valid_loss --diagnostic_batches 64 \
    --seed "${SEED}" --device cuda \
    2>&1 | tee "${RUN_DIR}/logs/core_${FAMILY}.log"
done
```

检查：

```bash
test -f "${RUN_DIR}/core/quantum/metrics.json"
test -f "${RUN_DIR}/core/classical/metrics.json"
test -f "${RUN_DIR}/core/quantum/attention_score_kernel.pt"
test -f "${RUN_DIR}/core/classical/attention_score_kernel.pt"
```

## 5. 训练三个 selector

```bash
for METHOD in quantum classical classical_strong; do
  if [ "${METHOD}" = quantum ]; then CORE=quantum; else CORE=classical; fi

  python experiments/train_relation_counterfactual_evidence.py \
    --model_dir "${RUN_DIR}/baseline" \
    --core_checkpoint "${RUN_DIR}/core/${CORE}/attention_score_kernel.pt" \
    --train_path data/relation/retacred/train.jsonl \
    --valid_path data/relation/retacred/valid.jsonl \
    --output_dir "${RUN_DIR}/selector/${METHOD}" \
    --evidence_type "${METHOD}" \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --evidence_gate_calibration context_budget \
    --evidence_view_score_mode positive \
    --evidence_task_readout dual \
    --evidence_readout connected_relation_token \
    --evidence_correlation_mode phase_selective \
    --evidence_weight_mode signed_centered_l1 \
    --evidence_measurement_mode entanglement_phase_offset \
    --intervention_mode direct_bias --direct_bias_mode centered \
    --evidence_budget 0.35 --diagnostic_batches 64 \
    --epochs 10 --batch_size 64 --lr 0.01 \
    --seed "${SEED}" --device cuda \
    2>&1 | tee "${RUN_DIR}/logs/selector_${METHOD}.log"
done
```

检查：

```bash
for METHOD in quantum classical classical_strong; do
  test -f "${RUN_DIR}/selector/${METHOD}/metrics.json"
  test -f "${RUN_DIR}/selector/${METHOD}/diagnostics.json"
done
```

## 6. 整理提交报告

不要把整个 `runs/` 或 checkpoint 提交到 GitHub。只复制 metrics、diagnostics 和日志尾部：

```bash
RUN_COMMIT=$(git rev-parse HEAD)
RUN_GIT_STATUS=$(git status --porcelain)
test -z "${RUN_GIT_STATUS}"

REPORT_TAG=$(date +%Y%m%d-%H%M%S)-dual-qres-full-seed13
REPORT_DIR="reports/retacred/${REPORT_TAG}"
mkdir -p "${REPORT_DIR}"/{baseline,core/quantum,core/classical,selector/quantum,selector/classical,selector/classical_strong,logs}

printf '%s\n' "${RUN_COMMIT}" > "${REPORT_DIR}/commit.txt"
printf '%s' "${RUN_GIT_STATUS}" > "${REPORT_DIR}/git_status.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl > "${REPORT_DIR}/data_counts.txt"

cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/baseline/"

for FAMILY in quantum classical; do
  cp "${RUN_DIR}/core/${FAMILY}/metrics.json" "${REPORT_DIR}/core/${FAMILY}/"
  cp "${RUN_DIR}/core/${FAMILY}/diagnostics.json" "${REPORT_DIR}/core/${FAMILY}/"
done

for METHOD in quantum classical classical_strong; do
  cp "${RUN_DIR}/selector/${METHOD}/metrics.json" "${REPORT_DIR}/selector/${METHOD}/"
  cp "${RUN_DIR}/selector/${METHOD}/diagnostics.json" "${REPORT_DIR}/selector/${METHOD}/"
done

for LOG in "${RUN_DIR}"/logs/*.log; do
  tail -n 1000 "${LOG}" > "${REPORT_DIR}/logs/$(basename "${LOG}").tail.txt"
done

echo "REPORT_DIR=${REPORT_DIR}"
```

`git_status.txt` 应为空。然后按照 [合作者 Git 工作流](collaborator_git_workflow_zh.md) 只提交这个报告目录到 `1.1`。

## 7. 需要返回的信息

- 报告目录；
- `1.1` 分支 commit hash；
- 实际运行时长和 GPU 型号；
- 失败时对应 stage 和日志最后 100 行。
