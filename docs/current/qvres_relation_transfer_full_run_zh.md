# Q-VRES Re-TACRED 正式实验

## 1. 同步代码

在项目根目录执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
```

最后一条命令必须没有输出。私有数据应位于：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
```

## 2. 先跑 seed 13 pilot

三张 GPU 并行运行三个 selector：

```bash
bash scripts/run_qvres_relation_transfer_full.sh \
  --parallel-mode selectors \
  --gpus auto \
  --seed 13
```

脚本先在 GPU 0 训练一次 baseline；baseline 完成后，三张 GPU 自动领取：

```text
q_causal_transport
classical_causal_transport
q_causal_key_only
```

终端会持续显示每个阶段的 GPU、状态、epoch、batch、速度和 ETA，并等待所有任务完成后才退出。

## 3. 查看进度

另开一个终端，在项目根目录执行：

```bash
PILOT_DIR=$(ls -dt runs/q_vres_relation_transfer_full/*_seed13_selector_parallel | head -n 1)
cat "${PILOT_DIR}/status/selector_parallel_status.json"
grep -H -E 'STATUS=|GPU_ID=|PID=|EXIT_CODE=' "${PILOT_DIR}"/status/*.env
watch -n 2 nvidia-smi
```

查看某个 selector 的实时日志：

```bash
tail -f "${PILOT_DIR}/logs/q_causal_transport.log"
```

## 4. 导出并提交 pilot 结果

命令正常结束后执行：

```bash
cat "${PILOT_DIR}/run_summary.md"
bash scripts/export_qvres_relation_transfer_pilot_report.sh "${PILOT_DIR}"
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/*-full-pilot-seed13 | head -n 1)
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES seed 13 full pilot results"
git push origin 1.1
```

只提交本次 `REPORT_DIR`。不要提交 `data/`、`runs/`、checkpoint、预测文件、JSONL 或完整日志。

## 5. pilot 通过后运行五 seed

只有在代码和正式配置没有变化时，seed 13 才能复用：

```bash
bash scripts/run_qvres_relation_transfer_multi_seed.sh \
  --gpus auto \
  --seeds 7,11,13,17,23 \
  --reuse-seed 13="${PILOT_DIR}"
```

如果代码或 `configs/q_vres_relation_transfer_full.json` 已改变，不要使用 `--reuse-seed`，应重新运行 seed 13。

五 seed 完成后执行：

```bash
GROUP_DIR=$(ls -dt runs/q_vres_relation_transfer_full_multiseed_* | head -n 1)
bash scripts/export_qvres_relation_transfer_multi_seed_report.sh "${GROUP_DIR}"
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/*-full-multiseed | head -n 1)
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES formal multi-seed results"
git push origin 1.1
```
