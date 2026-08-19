# Q-LASS 共识误差见证器：冻结五 seed 合作者执行包

## 实验边界

本轮是 synthetic dynamic-address consensus error-witness v2 的冻结 robustness validation，不是 Re-TACRED，不是真实数据实验。正式方法是 standalone quantum candidate estimator；`classical_consensus_control` 仅为等参数 product-state 对照。必须保留以下五个 selector：

```text
disabled
q_consensus_quantum
classical_consensus_control
q_consensus_shuffled_query
q_consensus_magnitude
```

固定 seed 为 `7,11,13,17,23`。训练步数、学习率、量子线路、witness、动作预算、数据规模和 gate 均已写入配置，禁止通过命令行覆盖、扫参、替换 seed、改变 threshold/gain/hard rate 或删除任何 control。任何 seed gate 失败都要保留并汇总，不能只报告通过的 seed。

本轮不授权：真实数据、Re-TACRED、finite-shot、硬件运行时间、当前 NISQ 优势、quantum speedup 或 quantum-advantage claim。

## 1. 同步与环境

以下操作在已授权 GPU 机器执行。不得把 raw dataset、`runs/`、checkpoint、逐样本 predictions、credentials 或完整日志提交回仓库。

```bash
cd /home/Q-Attention/Q-Attention-qvres-diagnostics
git fetch origin --prune
git switch codex/qvres-diagnostics
git status --short --branch
git rev-parse HEAD
```

运行前必须确认工作树 clean，并把 `git rev-parse HEAD` 记录到交付信息。若分支不是负责人指定的冻结 commit，停止并反馈，不要自行 merge 或修改源码。

```bash
conda activate py310
python -c "import sys,torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
python -m pytest -q \
  tests/test_q_consensus_error_witness_prescreen.py \
  tests/test_q_consensus_quantum_estimator.py \
  tests/test_q_consensus_quantum_estimator_single_seed.py \
  tests/test_q_consensus_quantum_estimator_frozen_multiseed.py
```

## 2. 前置门禁：单 seed 单 GPU

先在一张空闲 GPU 上完成固定 seed 7。单 seed 不使用 DDP，不拆分 quantum/classical stage，也不在同一张 GPU 上启动多个 worker。先做一次不执行实验的计划检查：

```bash
python experiments/run_q_consensus_quantum_estimator_single_seed.py \
  --config configs/q_consensus_quantum_estimator_frozen_multiseed.json \
  --gpu auto \
  --dry-run
```

确认输出中的 `formal_preflight_started` 为 `false`、`parallelism` 为 `single_seed_single_gpu`。然后运行固定 seed 7 的正式 preflight：

```bash
set -o pipefail
python experiments/run_q_consensus_quantum_estimator_single_seed.py \
  --config configs/q_consensus_quantum_estimator_frozen_multiseed.json \
  --gpu auto \
  2>&1 | tee runs/q_consensus_quantum_estimator_single_seed.log
```

该 runner 在一个进程中顺序训练 quantum estimator 与等参数 product-state control，并评估全部五个 selector。输出记录完整 seed 的端到端时间、物理 GPU ID、`workers_on_gpu=1`、配置 hash 和 clean commit provenance。

```bash
PREFLIGHT_DIR=$(ls -dt runs/q_consensus_quantum_estimator_single_seed/*/ | head -n 1)
PREFLIGHT_SUMMARY=${PREFLIGHT_DIR}/run_summary.json
test -f "${PREFLIGHT_DIR}/SINGLE_SEED_COMPLETE"
python -c "import json; p=json.load(open('${PREFLIGHT_SUMMARY}')); assert p['gate']['status']=='pass'; assert p['gate']['next_multi_seed_authorized'] is True; assert p['parallelism']['type']=='single_seed_single_gpu'; print(p['runtime'], p['parallelism'])"
```

如果 preflight gate 失败，停止，不得运行五 seed，不得换 seed 或调参。

## 3. 五 seed 唯一正式命令

先验证五 seed 调度计划和 preflight 绑定：

```bash
python experiments/run_q_consensus_quantum_estimator_frozen_multiseed.py \
  --config configs/q_consensus_quantum_estimator_frozen_multiseed.json \
  --preflight-summary "${PREFLIGHT_SUMMARY}" \
  --gpus auto \
  --dry-run
```

正式五 seed 只允许使用以下命令。每张 GPU 同一时刻只运行一个独立 seed；同一 GPU 分到多个 seed 时按固定顺序串行，GPU 队列之间并行。该阶段仍不使用 DDP，也不在单卡内并发多个 worker。

```bash
python experiments/run_q_consensus_quantum_estimator_frozen_multiseed.py \
  --config configs/q_consensus_quantum_estimator_frozen_multiseed.json \
  --preflight-summary "${PREFLIGHT_SUMMARY}" \
  --gpus auto \
  2>&1 | tee runs/q_consensus_quantum_estimator_frozen_multiseed.log
```

脚本会在 `runs/q_consensus_quantum_estimator_frozen_multiseed/<UTC时间>/` 创建独立目录，并把 preflight 路径与 SHA-256 锁入 `multi_seed_manifest.json`。只有五个 seed 都有 `SEED_COMPLETE` 且执行成功时才会出现 `MULTI_SEED_COMPLETE`。

## 4. 结果检查与交付

```bash
GROUP_DIR=$(ls -dt runs/q_consensus_quantum_estimator_frozen_multiseed/*/ | head -n 1)
test -f "${GROUP_DIR}/MULTI_SEED_COMPLETE"
test -f "${GROUP_DIR}/aggregate_summary.json"
test -f "${GROUP_DIR}/aggregate_summary.md"
python scripts/summarize_q_consensus_quantum_estimator_frozen_multiseed.py \
  --group-dir "${GROUP_DIR}"
```

只能导出 report-only 文件。导出前仓库必须 clean，且 HEAD 必须等于 `multi_seed_manifest.json` 中的 commit：

```bash
python scripts/export_q_consensus_quantum_estimator_frozen_multiseed_report.py \
  --group-dir "${GROUP_DIR}" \
  --report-dir "reports/q_consensus_quantum_estimator/$(date -u +%Y%m%d-%H%M%S)-frozen-multiseed"
```

报告目录只允许包含 manifest、aggregate summary、preflight `run_summary.json`、每个 seed 的 `run_summary.json` 和完成标记。不要使用 `git add .`；只提交 report 目录中的白名单文件。如果项目负责人要求 push，先回传以下信息再等待确认：冻结 commit、preflight 的单卡 ID 与端到端耗时、五个 seed 的完成状态、aggregate gate 状态、report 路径及 `git diff --cached --name-only`。

## 5. 失败处理

- 环境、测试、配置 hash、commit、GPU 检查失败：停止并反馈原始错误摘要。
- 单 seed preflight GPU 不可用、worker 数不是 1、执行失败或 gate 失败：停止，不得启动五 seed。
- 单个 seed 进程失败：不要改参数重跑；保留失败目录和 `MULTI_SEED_FAILED`，反馈 seed、GPU、return code。
- 单 seed scientific gate 失败：不要删除、替换或重跑该 seed；仍完成剩余冻结 seed，并在汇总中保留 failed status。
- aggregate gate 失败：这不是调参邀请。停止进入真实数据、finite-shot 或硬件实验，回传 `aggregate_summary.md`。
