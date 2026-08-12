# Q-NESS 比例实验

在项目根目录、激活自己的 conda 环境后执行。代码由 `main` 维护，实验在 `1.1` 分支进行。

## 运行

先确认工作树干净：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
git log -1 --oneline
```

执行比例 gate。脚本会自动在 `runs/` 下创建新的时间戳目录，不复用旧结果。单卡命令保持兼容：

```bash
bash scripts/run_retacred_qness_proportional.sh --gpu 0 --seed 13
```

多卡时使用 `--gpus`。baseline 和 quantum core 只在第一张卡上各运行一次；随后六个 selector 由每张 GPU 的 worker 动态领取，先完成的 GPU 自动继续运行下一个 selector，不使用 DDP：

```bash
bash scripts/run_retacred_qness_proportional.sh --gpus 0,1,2 --seed 13
```

使用服务器上的全部可见 GPU：

```bash
bash scripts/run_retacred_qness_proportional.sh --gpus auto --seed 13
```

运行时主终端每 10 秒打印一次六个 selector 的统一进度：完成数、运行数、待分配数、失败数，以及每个 GPU 当前运行的 selector、epoch、batch 和 ETA。完整子进程日志仍写入 `logs/<stage>.log`。如需调整刷新间隔，可加：

```bash
bash scripts/run_retacred_qness_proportional.sh \
  --gpus auto --seed 13 --dashboard-interval-seconds 30
```

正常情况下，启动脚本只有在所有 stage 完成、所有 selector 的 `metrics.json` 和 `diagnostics.json` 存在、summary 生成且 `RUN_COMPLETE` 写入后才会退出。中断或 watchdog 超时会清理整个子进程组；失败时终端会列出失败或未完成的 stage，并返回非零状态。

需要先检查命令但不创建文件时：

```bash
bash scripts/run_retacred_qness_proportional.sh --gpus 0,1,2 --dry-run
```

只做很小的真实数据 canary 时：

```bash
python experiments/run_relation_qness_proportional_gate.py \
  --baseline_train_limit 256 --train_limit 256 --valid_limit 128 \
  --baseline_epochs 1 --core_epochs 1 --selector_epochs 1 \
  --diagnostic_batches 2 --random_repeats 1 --run_controls never \
  --output_dir runs/retacred_qness_canary_$(date -u +%Y%m%dT%H%M%SZ)_seed13 \
  --device cuda --gpus 0
```

## 查看结果

运行过程和每个阶段的输出在自动生成目录中：

```bash
RUN_DIR=$(ls -dt runs/retacred_qness_proportional_*_seed13/ | head -n 1)
cat "${RUN_DIR}/run_summary.md"
cat "${RUN_DIR}/run_summary.json"
tail -n 50 "${RUN_DIR}/logs/selector_qness.log"
cat "${RUN_DIR}/gpu_assignments.json"
cat "${RUN_DIR}/status/selector_qness.env"
```

需要单独查看 selector 状态时：

```bash
grep -H -E 'STATUS=|GPU_ID=|PID=|COMPLETED_AT=|FAILED_AT=|EXIT_CODE=' \
  "${RUN_DIR}"/status/selector_*.env
```

并行 selector 的终端输出由统一 dashboard 负责，阶段日志仍分别保存在 `logs/<stage>.log`。`gpu_assignments.json` 记录实际 GPU、状态和耗时，`status/*.env` 记录每个阶段的 GPU、PID、开始时间、结束时间和 heartbeat 路径。

训练结束后导出可提交报告：

```bash
REPORT_DIR=$(bash scripts/export_retacred_qness_proportional_report.sh "${RUN_DIR}" \
  | sed -n 's/^REPORT_DIR=//p')
echo "REPORT_DIR=${REPORT_DIR}"
```

脚本会打印 `REPORT_DIR`。只提交该 `reports/retacred/...` 目录；不要提交 `private_subsets/`、完整 `logs/`、checkpoint 或整个 `runs/` 目录。导出的报告已经包含 summary、GPU 分配、阶段状态、各阶段 metrics/diagnostics 和日志尾部，可以直接用于诊断。

提交到 `1.1`：

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-NESS proportional gate results"
git push origin 1.1
```

比例 gate 只用于筛选路线，不代表显著性、任务最终提升或量子优势。只有 gate 通过后，才进入 full 多 seed 实验。
