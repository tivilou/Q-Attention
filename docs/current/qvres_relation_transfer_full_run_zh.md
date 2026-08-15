# Q-VRES Re-TACRED 正式实验

## 运行

在项目根目录执行。先确认私有数据位于：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
```

激活你自己的 Conda 环境后执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short

bash scripts/run_qvres_relation_transfer_multi_seed.sh \
  --gpus auto \
  --seeds 7,11,13,17,23
```

`--gpus auto` 会发现所有 GPU，每张 GPU 同时负责一个 seed；某张 GPU 完成后会自动领取下一个未完成 seed。只有所有 seed 完成后命令才退出。

## 查看进度

找到本次实验目录：

```bash
GROUP_DIR=$(ls -dt runs/q_vres_relation_transfer_full_multiseed_* | head -n 1)
cat "${GROUP_DIR}/multi_seed_status.json"
```

单个 seed 的实时日志在：

```text
${GROUP_DIR}/seed_7/logs/run.log
```

状态文件在：

```text
${GROUP_DIR}/seed_7/status/run.env
```

查看所有 GPU 的占用：

```bash
watch -n 2 nvidia-smi
```

## 导出和提交结果

实验完成后执行：

```bash
bash scripts/export_qvres_relation_transfer_multi_seed_report.sh "${GROUP_DIR}"
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/* | head -n 1)
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES formal multi-seed results"
git push origin 1.1
```

只提交 `reports/q_vres_relation_transfer/` 下本次报告目录。不要提交 `data/`、`runs/`、checkpoint、预测文件、JSONL 或完整日志。
