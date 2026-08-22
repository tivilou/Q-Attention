# Q-LASS Re-TACRED 正式单 seed 实验

本实验只跑完整 Re-TACRED 的一个声明 seed（13），单 GPU 串行执行 baseline、Q-LASS `global_context` 和参数匹配的 classical control。两种 kernel 都冻结 baseline，只在 valid 上选择最佳 checkpoint；test 只在最后评估。

## 1. 同步并检查

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
bash scripts/check_retacred_qlass_formal_single_seed.sh
```

检查必须确认三份数据行数为 `58465/19584/13418`，CUDA 可用，且工作树干净。不要提交数据、`runs/`、checkpoint、predictions 或完整日志。

## 2. 启动

```bash
bash scripts/run_retacred_qlass_formal_single_seed.sh --gpu 0
```

运行目录默认写入 `runs/retacred_qlass_formal_single_seed/`。指定其他单卡时只改 `--gpu`；本入口不接受多 GPU 或并行参数。

完成标志和摘要：

```text
RUN_COMPLETE
run_summary.json
run_summary.md
```

## 3. 导出并提交到 1.1

raw `runs/` 不提交。合作者先把 `main` 合入自己的 `1.1`，再执行：

```bash
git fetch origin --prune
git switch 1.1
git pull --ff-only origin 1.1
git merge origin/main
bash scripts/export_retacred_qlass_formal_single_seed_report.sh \
  --run-dir runs/retacred_qlass_formal_single_seed/<时间戳>_seed13
```

这里允许生成一个 merge commit：1.1 通常已经包含合作者的实验报告提交，和 main 会发生分叉。若 git merge origin/main 出现冲突，立即停止并反馈，不要删除源码、使用 git add -f runs/... 或强行继续。
脚本会检查 `RUN_COMPLETE`、`RUN_FAILED`、seed、test isolation、valid/test 指标和私有文件；然后只把精简报告写入 `reports/retacred_qlass_formal_single_seed/`，执行 `git add`、`git diff --cached --check`、提交，并推送到 `origin/1.1`。如需先检查而不提交：

```bash
bash scripts/export_retacred_qlass_formal_single_seed_report.sh \
  --run-dir runs/retacred_qlass_formal_single_seed/<时间戳>_seed13 \
  --no-commit
```

不要使用 `git add -f runs/...`，也不要执行 `git push origin main`。

## 4. 判定

先看 `run_summary.md` 的 valid/test `macro-F1` 和 Q-LASS minus classical。若单 seed 没有实际提升，停止；若有提升，再预声明独立多 seed 复制。不得用 test 选择模型或调参。
