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

本轮不重新训练，也不运行五 seed。诊断由负责人在服务器执行。你只需要：

1. 保留已经完成的 `runs/q_vres_relation_transfer_full/*_seed13_selector_parallel` raw run。
2. 不删除 baseline 或 selector checkpoint。
3. 等负责人根据 validation 诊断结果确认下一轮 full-data 或 multi-seed 命令。

本轮不提交诊断报告。后续正式 full 实验开始前，仍要先同步 `origin/main` 并确认工作树 clean。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
git status --short
```

后续 full 实验只提交负责人指定的聚合 `reports/` 目录。禁止提交 `data/`、`runs/`、checkpoint、逐样本预测、JSONL、梯度张量或完整日志。

负责人更新 `main` 后，下一轮开始前仍先同步 `origin/main` 并确认工作树 clean。
