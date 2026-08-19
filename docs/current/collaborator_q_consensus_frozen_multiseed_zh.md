# Q-LASS 共识量子估计器冻结实验

本轮是 synthetic 验证，不是 Re-TACRED。固定 seed 为 `7,11,13,17,23`，五个 selector 和全部训练参数均已冻结，禁止改参数、换 seed 或只报告通过结果。

单 seed 使用一张 GPU；五 seed 按 seed 分配 GPU。同一 seed 内的五个 selector 顺序执行，不做 DDP 或 selector 多 GPU 并行。

## 1. 同步代码

在项目根目录执行：

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
git status --short
```

最后一条命令必须没有输出。然后检查环境和脚本：

```bash
conda activate py310
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
python -m pytest -q \
  tests/test_q_consensus_error_witness_prescreen.py \
  tests/test_q_consensus_quantum_estimator.py \
  tests/test_q_consensus_quantum_estimator_single_seed.py \
  tests/test_q_consensus_quantum_estimator_frozen_multiseed.py
```

## 2. 运行 seed 7

先检查单 GPU 计划：

```bash
python experiments/run_q_consensus_quantum_estimator_single_seed.py \
  --gpu auto \
  --dry-run
```

确认输出为 `single_seed_single_gpu` 后正式运行：

```bash
python experiments/run_q_consensus_quantum_estimator_single_seed.py \
  --gpu auto
```

检查结果：

```bash
PREFLIGHT_DIR=$(ls -dt runs/q_consensus_quantum_estimator_single_seed/*/ | head -n 1)
PREFLIGHT_SUMMARY=${PREFLIGHT_DIR}/run_summary.json
test -f "${PREFLIGHT_DIR}/SINGLE_SEED_COMPLETE"
python -c "import json; p=json.load(open('${PREFLIGHT_SUMMARY}')); assert p['gate']['status']=='pass'; print(p['runtime'], p['parallelism'])"
```

gate 失败时停止，不得运行五 seed。

## 3. 运行五 seed

五 seed 由合作者执行。每张 GPU 同时最多运行一个 seed；卡内串行，卡间并行：

```bash
python experiments/run_q_consensus_quantum_estimator_frozen_multiseed.py \
  --preflight-summary "${PREFLIGHT_SUMMARY}" \
  --gpus auto
```

完成后检查：

```bash
GROUP_DIR=$(ls -dt runs/q_consensus_quantum_estimator_frozen_multiseed/*/ | head -n 1)
test -f "${GROUP_DIR}/MULTI_SEED_COMPLETE"
cat "${GROUP_DIR}/aggregate_summary.md"
```

## 4. 导出并提交报告

```bash
REPORT_DIR="reports/q_consensus_quantum_estimator/$(date -u +%Y%m%d-%H%M%S)-frozen-multiseed"
python scripts/export_q_consensus_quantum_estimator_frozen_multiseed_report.py \
  --group-dir "${GROUP_DIR}" \
  --report-dir "${REPORT_DIR}"

git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-LASS frozen multi-seed report"
git push origin main
```

只提交导出的 report。不要提交 `data/`、`runs/`、checkpoint、逐样本预测、JSONL 或完整日志。

## 5. 失败处理

- 环境、测试、GPU 或配置检查失败：停止并反馈错误。
- seed 进程失败：保留输出，不改参数重跑。
- scientific gate 失败：保留并报告，不换 seed、不筛选结果。
- aggregate gate 失败：停止后续真实数据和硬件实验。
