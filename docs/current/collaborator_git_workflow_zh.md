# 合作者 Git 工作流

## 分支约定

- `main`：负责人维护的最新代码。
- `1.1`：合作者实验分支，只提交 `reports/` 下的结果。

不要直接向 `main` 提交实验结果，不要使用 GitHub ZIP 作为持续开发方式。

## 第一次 clone

```bash
git clone https://github.com/tivilou/Q-Attention.git
cd Q-Attention
git fetch origin --prune
git switch --track -c 1.1 origin/1.1
git merge origin/main
```

如果本地已经有 `1.1`：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
```

出现冲突时停止并反馈，不要删除源码。

## 实验前检查

```bash
git status --short
conda activate YOUR_ENV_NAME
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
python -m pytest -q
```

`git status --short` 必须没有输出。不要使用固定的 Conda 环境名。

数据必须位于：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
```

三者行数应为 `58465`、`19584`、`13418`。

## 本轮实验

按照 [Q-VRES Re-TACRED 正式实验](qvres_relation_transfer_full_run_zh.md) 执行：

```bash
bash scripts/run_qvres_relation_transfer_multi_seed.sh --gpus auto --seeds 7,11,13,17,23
```

## 提交报告

实验完成后：

```bash
GROUP_DIR=$(ls -dt runs/q_vres_relation_transfer_full_multiseed_* | head -n 1)
bash scripts/export_qvres_relation_transfer_multi_seed_report.sh "${GROUP_DIR}"
REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/* | head -n 1)
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES formal multi-seed results"
git push origin 1.1
```

暂存区只能包含本次报告目录。禁止提交 `data/`、`runs/`、checkpoint、预测文件、JSONL、完整日志或环境文件。

负责人更新 `main` 后，先执行 `git fetch origin --prune`、`git switch 1.1`、`git merge origin/main`，确认 clean 后再开始下一轮。
