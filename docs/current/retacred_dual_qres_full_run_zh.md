# Re-TACRED Dual Q-RES Full 运行指南

本指南用于完整 train/valid 上的单 seed 主实验和多 seed 正式实验。运行、GPU 调度和报告整理已经封装为脚本，不要再手工逐条输入训练命令。

不要使用旧 `run_relation_smoke_pipeline.py` 或 `run_relation_attention_transfer_screen.py`，不要运行 test。

## 1. 同步并检查代码

在 `1.1` 分支合并最新 `main` 后执行：

```bash
git status --short
git log -1 --oneline
```

`git status --short` 必须没有输出。

## 2. 运行预检

激活自己的 Conda 环境，然后执行：

```bash
conda activate YOUR_ENV_NAME
bash scripts/check_retacred_dual_qres_full.sh
```

预检会检查：

- Git commit 和 clean 状态；
- 当前 Python、PyTorch 和 CUDA；
- GPU 型号与显存；
- Re-TACRED train/valid 文件及行数；
- 完整 pytest 回归。

## 3. 运行 full seed 13

建议先进入 `tmux`：

```bash
tmux new -s qattention_full_seed13
```

单 GPU 串行运行时执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13
```

两张或更多 GPU 对单个 seed 做阶段并行时执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh \
  --seed 13 \
  --gpus auto \
  --parallel-mode stages \
  --stage-timeout-hours 36
```

两张 GPU 时，quantum/classical core 会分别运行；quantum selector 独占一张 GPU，两个经典 selector 在另一张 GPU 排队。单 GPU 默认命令保持原来的串行行为。

训练日志默认每 50 个 batch 写入一次当前 phase、epoch、完成比例、已用时间、ETA 和预计完成时间。终端同时显示可读进度条，JSON 日志仍然保留。需要更频繁的进度时可以执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13 --log-every-batches 25
```

只保留 JSON 进度时增加：

```bash
--progress-format json
```

完整 valid task metric 始终使用全部验证集；`diagnostic_batches=64` 只限制 selectivity 和 alignment 等诊断，不缩减主指标评估。

只验证真实数据和量子路径、不启动正式全量训练时执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13 --canary-only
```

脚本会自动完成：

1. 再次运行预检；
2. 创建带日期时间的 `runs/retacred_dual_projector_full_*_seed13/`；
3. 先用真实 Re-TACRED 的 256 条 train、128 条 valid 做量子路径 canary；
4. canary 通过后训练正式 baseline、quantum/classical core 和三个 selector；
5. 每个阶段保存日志、心跳和状态文件；
6. 检查所有 metrics、diagnostics 和 checkpoint；
7. 成功写入 `RUN_COMPLETE`，异常写入 `RUN_FAILED`。

canary 只检查真实数据上的前向、反向和量子路径数值是否有限，不作为正式结果，也不会改变正式实验数据。

只查看命令、不启动训练时执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13 --dry-run
```

## 4. 双 GPU 多 seed 运行

seed 13 的修复后单 seed 正式实验通过后，再运行五 seed：

```bash
bash scripts/run_retacred_dual_qres_multi_seed.sh \
  --seeds 7,11,13,17,23 \
  --gpus auto \
  --stage-timeout-hours 36
```

脚本先集中运行一次预检，再为每张 GPU 启动一个 worker。两张 GPU 同时运行两个独立 seed，任一 GPU 完成后会自动领取下一个 seed。每个 seed 都使用独立目录，单个 seed 内保持单 GPU 训练，不改变有效 batch size。

只验证双 GPU 调度和量子 canary 时执行：

```bash
bash scripts/run_retacred_dual_qres_multi_seed.sh \
  --seeds 7,11 \
  --gpus auto \
  --canary-only
```

多 seed 运行目录为：

```text
runs/retacred_dual_projector_multiseed_YYYYMMDD_HHMMSS/
  seed_7/
  seed_11/
  multi_seed_manifest.json
  multi_seed_summary.json
```

## 5. 获取运行目录

训练过程中查看最新目录和当前阶段：

```bash
RUN_DIR=$(ls -dt runs/retacred_dual_projector_full_*_seed*/ | head -n 1)
ls -lht "${RUN_DIR}/logs"
tail -f "${RUN_DIR}/logs/selector_quantum.log"

# 查看阶段状态和最近一次心跳
cat "${RUN_DIR}/status/selector_quantum.env"
cat "${RUN_DIR}/status/selector_quantum.heartbeat"
```

多 seed 运行时查看汇总：

```bash
GROUP_DIR=$(ls -dt runs/retacred_dual_projector_multiseed_*/ | head -n 1)
cat "${GROUP_DIR}/multi_seed_summary.json"
nvidia-smi
```

训练完成后：

```bash
RUN_DIR=$(cat runs/latest_dual_qres_full_run.txt)
echo "${RUN_DIR}"
test -f "${RUN_DIR}/RUN_COMPLETE"

# 若中途失败，查看失败阶段
cat "${RUN_DIR}/RUN_FAILED"
```

## 6. 导出提交报告

```bash
bash scripts/export_retacred_dual_qres_report.sh "${RUN_DIR}"
```

导出脚本只复制：

- baseline metrics；
- quantum/classical core metrics 和 diagnostics；
- 三个 selector 的 metrics 和 diagnostics；
- 运行 manifest、数据行数和日志最后 1000 行。

它不会复制 checkpoint、完整日志、数据或 predictions。命令结束时会输出 `REPORT_DIR`。

多 seed 全部完成后执行：

```bash
bash scripts/export_retacred_dual_qres_multi_seed_report.sh "${GROUP_DIR}"
```

该脚本会一次性导出所有 seed 的白名单结果，不会导出 checkpoint、数据或 predictions。

## 7. 提交到 1.1

使用导出脚本打印的报告目录：

```bash
REPORT_DIR=reports/retacred/REPLACE_WITH_EXPORTED_DIRECTORY
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add dual Q-RES full seed 13 results"
git push origin 1.1
```

暂存区只能包含本次报告目录。推送后返回 commit hash、运行时间和 GPU 型号。

## 8. 失败反馈

脚本会打印失败的 stage。反馈以下内容即可：

```bash
FAILED_STAGE=selector_quantum
tail -n 100 "${RUN_DIR}/logs/${FAILED_STAGE}.log"
git rev-parse HEAD
git status --short
```
