# 合作者 Git 工作流

## 分支约定

- `main`：负责人维护的最新代码。
- `1.1`：合作者实验分支，只提交 `reports/` 下的结果。

不要直接向 `main` 提交结果，不要使用 GitHub ZIP 作为持续开发方式。

## 第一次 clone

```bash
git clone https://github.com/tivilou/Q-Attention.git
cd Q-Attention
git fetch origin --prune
git switch --track -c 1.1 origin/1.1
git merge origin/main
```

## 每次实验前同步

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
```

`git status --short` 必须没有输出。出现冲突时停止并反馈，不要删除源码。

激活自己的 Conda 环境并检查 GPU：

```bash
conda activate YOUR_ENV_NAME
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

不要使用文档中的示例名称代替自己的 Conda 环境名。

## 本轮任务

本轮不重新训练，也不运行五 seed。由于 seed 13 raw run 在你的机器上，请执行只读 validation 诊断：

```bash
find "$HOME" -type d -name '*_seed13_selector_parallel' -print | sort

PILOT_DIR=/root/projects/Q-Attention/runs/q_vres_relation_transfer_full/20260815T235926Z_seed13_selector_parallel
test -f "${PILOT_DIR}/RUN_COMPLETE"
test -f "${PILOT_DIR}/stages/baseline/baseline/model.pt"
test -f "${PILOT_DIR}/stages/q_causal_transport/selectors/q_causal_transport/best_kernel.pt"

PYTHON_BIN=python bash scripts/run_qvres_validation_diagnostic.sh \
  "${PILOT_DIR}" \
  --gpu 0 \
  --batch-size 8 \
  --log-every-batches 50
```

诊断不会重新训练，也不会修改 raw run。完成后提交两个聚合文件：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short

REPORT_DIR=$(ls -dt reports/q_vres_relation_transfer/*-validation-diagnostic | head -n 1)

git add \
  "${REPORT_DIR}/diagnostic_summary.json" \
  "${REPORT_DIR}/diagnostic_summary.md"

git diff --cached --check
git diff --cached --name-only
git commit -m "Add Q-VRES seed 13 validation diagnostics"
git push origin 1.1
```

`git diff --cached --name-only` 必须只有上述两个文件。禁止提交 `data/`、`runs/`、checkpoint、逐样本预测、JSONL、梯度张量或完整日志。

负责人更新 `main` 后，下一轮开始前仍先同步 `origin/main` 并确认工作树 clean。
