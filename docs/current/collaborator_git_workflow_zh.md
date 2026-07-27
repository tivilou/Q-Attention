# 合作者 Git 工作流

## 分支约定

- `main`：代码主线，由负责人维护。
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
```

出现冲突时停止操作并反馈，不要自行删除源码。

## 每次实验前

```bash
git status --short
git merge-base --is-ancestor b8d794f HEAD && echo "code version OK"
```

`git status --short` 必须没有输出。然后激活自己的 Conda 环境：

```bash
conda activate <你的实验环境>
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
python -m pytest -q
```

不要使用固定的环境名；只要当前环境满足项目依赖并能使用 CUDA 即可。

## 数据检查

数据应位于：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
```

当前 full train/valid 预期：

```bash
wc -l data/relation/retacred/train.jsonl
wc -l data/relation/retacred/valid.jsonl
```

输出应为 `58465` 和 `19584`。本轮不要读取或运行 test。

## 运行实验

按照 [Re-TACRED full 运行指南](retacred_dual_qres_full_run_zh.md) 执行。长时间任务建议放在 `tmux` 中。

## 提交报告

实验完成后只整理 `reports/retacred/<报告目录>/`，然后执行：

```bash
git status --short
git add reports/retacred/<报告目录>
git diff --cached --check
git diff --cached --name-only
git commit -m "Add dual Q-RES full seed 13 results"
git push origin 1.1
```

暂存区只能包含本次报告目录。禁止提交：

- `data/`
- `runs/`
- `*.pt`、`*.pth`、`*.ckpt`
- predictions、JSONL、完整日志
- Conda 环境、缓存或压缩数据包

推送后返回 commit hash。

## 下一轮同步

负责人更新 `main` 后：

```bash
git fetch origin --prune
git switch 1.1
git status --short
git merge origin/main
git status --short --branch
```

仓库 clean 后再开始下一轮实验。
