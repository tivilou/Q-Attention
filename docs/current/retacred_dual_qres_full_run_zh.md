# Re-TACRED Dual Q-RES Full 运行指南

本指南用于完整 train/valid 上的 seed 13 主实验预跑。运行和报告整理已经封装为脚本，不要再手工逐条输入训练命令。

不要使用旧 `run_relation_smoke_pipeline.py` 或 `run_relation_attention_transfer_screen.py`，不要运行 test。

## 1. 同步并检查代码

在 `1.1` 分支合并最新 `main` 后执行：

```bash
git status --short
git merge-base --is-ancestor b8d794f HEAD && echo "code version OK"
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

然后只需执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13
```

训练日志默认每 50 个 batch 写入一次当前 phase、epoch、完成比例、已用时间和 ETA。需要更频繁的进度时可以执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13 --log-every-batches 25
```

完整 valid task metric 始终使用全部验证集；`diagnostic_batches=64` 只限制 selectivity 和 alignment 等诊断，不缩减主指标评估。

脚本会自动完成：

1. 再次运行预检；
2. 创建带日期时间的 `runs/retacred_dual_projector_full_*_seed13/`；
3. 训练 baseline；
4. 训练 quantum/classical core；
5. 训练 quantum、classical、classical_strong selector；
6. 保存每个阶段的日志；
7. 检查所有 metrics、diagnostics 和 checkpoint；
8. 写入 `RUN_COMPLETE` 和最新运行目录记录。

只查看命令、不启动训练时执行：

```bash
bash scripts/run_retacred_dual_qres_full.sh --seed 13 --dry-run
```

## 4. 获取运行目录

训练过程中查看最新目录和当前阶段：

```bash
RUN_DIR=$(ls -dt runs/retacred_dual_projector_full_*_seed*/ | head -n 1)
ls -lht "${RUN_DIR}/logs"
tail -f "${RUN_DIR}/logs/selector_quantum.log"
```

训练完成后：

```bash
RUN_DIR=$(cat runs/latest_dual_qres_full_run.txt)
echo "${RUN_DIR}"
test -f "${RUN_DIR}/RUN_COMPLETE"
```

## 5. 导出提交报告

```bash
bash scripts/export_retacred_dual_qres_report.sh "${RUN_DIR}"
```

导出脚本只复制：

- baseline metrics；
- quantum/classical core metrics 和 diagnostics；
- 三个 selector 的 metrics 和 diagnostics；
- 运行 manifest、数据行数和日志最后 1000 行。

它不会复制 checkpoint、完整日志、数据或 predictions。命令结束时会输出 `REPORT_DIR`。

## 6. 提交到 1.1

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

## 7. 失败反馈

脚本会打印失败的 stage。反馈以下内容即可：

```bash
FAILED_STAGE=selector_quantum
tail -n 100 "${RUN_DIR}/logs/${FAILED_STAGE}.log"
git rev-parse HEAD
git status --short
```
