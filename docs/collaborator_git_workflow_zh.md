# 合作者 Git 实验工作流（1.1 分支）

本文面向负责 GPU 实验的合作者。目标是把实验流程从“下载 GitHub zip、手动上传下载结果”改成可追踪的 Git 工作流：代码从 `main` 来，实验结果提交到 `1.1` 分支，负责人再把结果合并回 `main`。

请优先遵守下面三条规则：

1. 不再使用 GitHub 的 Download ZIP 跑实验。ZIP 目录没有 Git 历史，后续同步代码和提交结果都会很麻烦。
2. 不直接向 `main` 分支提交。`main` 是代码主线，实验结果先提交到 `1.1`。
3. 不提交原始数据、模型权重和大体积中间产物。只提交整理后的 `reports/` 目录。

## 1. 分支约定

本项目采用下面的协作方式：

```text
main  -> 代码主线，由负责人维护和合并
1.1   -> 实验分支，由合作者运行实验并提交报告文件
```

常规循环是：

```text
负责人更新 main
        -> 合作者把 main 合并进本地 1.1
        -> 合作者在 1.1 上跑实验
        -> 合作者把 reports/ 结果 push 到 GitHub 的 1.1
        -> 负责人检查后合并 1.1 到 main
```

## 2. 第一次 clone 项目

在 GPU 服务器上选择一个固定工作目录，例如：

```bash
mkdir -p ~/projects
cd ~/projects
```

从 GitHub clone 项目：

```bash
git clone https://github.com/tivilou/Q-Attention.git
cd Q-Attention
```

确认远端地址和当前分支：

```bash
git remote -v
git branch --show-current
git status --short --branch
```

正常情况下刚 clone 下来是在 `main` 分支。

## 3. 第一次把本地 1.1 对齐到 main

这一步只在“第一次规范化工作流”时做。它的作用是让本地 `1.1` 从干净的 `main` 开始。

请先确认没有需要保留的本地改动：

```bash
git status --short
```

如果输出为空，再执行：

```bash
git fetch origin
git switch -C 1.1 origin/main
```

此时检查：

```bash
git branch --show-current
git log --oneline -3
```

`git branch --show-current` 应该输出：

```text
1.1
```

如果负责人明确要求把 GitHub 上的 `1.1` 也同步成当前状态，再执行：

```bash
git push --force-with-lease origin 1.1
```

注意：`--force-with-lease` 只用于这一次分支对齐。后续不要反复 reset 或 force push；后续同步代码用 `git merge origin/main`。

## 4. 准备 Python 环境和数据

进入项目根目录后，先激活你自己服务器上已经能跑实验的 Python/conda 环境。不要照抄别人的环境名，例如不一定叫 `py310`。

示例：

```bash
conda activate <your_env_name>
```

确认 Python 和 CUDA 状态：

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

安装项目为 editable 包：

```bash
python -m pip install -e ".[dev]"
```

运行测试：

```bash
python -m pytest -q
```

如果测试不能通过，先不要跑正式实验，把完整报错发回来。

数据目录需要自己从负责人提供的数据包解压得到。公开 GitHub 仓库不放 TACRED/Re-TACRED 原始数据。项目中实验默认读取：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
```

## 5. 在 1.1 分支上跑实验

每次跑实验前都确认自己在 `1.1` 分支：

```bash
git branch --show-current
git status --short --branch
```

创建日志目录：

```bash
mkdir -p runs/handoff_logs
```

### 5.1 Debug run

先跑 debug，确认环境没有问题：

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_debug_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_debug_gpu
```

期望生成：

```text
runs/retacred_debug_gpu/pipeline_summary.json
runs/retacred_debug_gpu/run_summary.json
runs/retacred_debug_gpu/run_summary.md
```

### 5.2 Low-resource run

Debug 通过后再跑：

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_low_resource_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_low_resource_gpu
```

### 5.3 Full run

最后跑 full：

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_full_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_full_gpu
```

## 6. 如果 run_summary 没有生成

`summarize_relation_run.py` 正常执行成功时，会打印类似：

```json
{"output_json": "runs/retacred_full_gpu/run_summary.json", "output_markdown": "runs/retacred_full_gpu/run_summary.md", "rows": 5}
```

如果目录里没有 `run_summary.json` 或 `run_summary.md`，先不要重新下载 zip，先检查路径：

```bash
pwd
git branch --show-current
ls -la runs/retacred_full_gpu
python experiments/summarize_relation_run.py --run_dir runs/retacred_full_gpu
find . -path '*retacred_full_gpu/run_summary.*' -print
```

常见原因是：命令不是在项目根目录执行，或者查看的是另一个旧目录。

## 7. 整理要提交的结果

不要直接提交 `runs/`。`runs/` 里面有模型权重、中间文件和可能很大的日志。请把需要交付的结果复制到 `reports/`。

建议按日期建目录，例如：

```bash
DATE=2026-06-13
mkdir -p reports/retacred/${DATE}/full
mkdir -p reports/retacred/${DATE}/low_resource
mkdir -p reports/retacred/${DATE}/debug
mkdir -p reports/retacred/${DATE}/logs
```

以 full run 为例，复制 summary 和核心 metrics：

```bash
RUN=runs/retacred_full_gpu
OUT=reports/retacred/${DATE}/full

mkdir -p ${OUT}/baseline
mkdir -p ${OUT}/classical_steering_eval
mkdir -p ${OUT}/quantum_steering_eval
mkdir -p ${OUT}/spectral_filter_sweep
mkdir -p ${OUT}/relation_routing_eval

cp ${RUN}/pipeline_summary.json ${OUT}/
cp ${RUN}/run_summary.json ${OUT}/
cp ${RUN}/run_summary.md ${OUT}/
cp ${RUN}/baseline/metrics.json ${OUT}/baseline/
cp ${RUN}/classical_steering_eval/metrics.json ${OUT}/classical_steering_eval/
cp ${RUN}/quantum_steering_eval/metrics.json ${OUT}/quantum_steering_eval/
cp ${RUN}/spectral_filter_sweep/summary.json ${OUT}/spectral_filter_sweep/
cp ${RUN}/relation_routing_eval/metrics.json ${OUT}/relation_routing_eval/
cp runs/handoff_logs/retacred_full_gpu.log reports/retacred/${DATE}/logs/
```

Low-resource 和 debug 按同样方式复制到对应目录。

不要提交这些文件：

```text
data/
runs/
*.pt
*.pth
*.ckpt
predictions.jsonl
routing.jsonl
原始 TACRED/Re-TACRED 数据
```

## 8. 提交结果到 1.1

提交前先检查状态：

```bash
git status --short
```

理想情况下，只应该看到 `reports/retacred/<日期>/...`。如果看到 `data/`、`runs/`、`*.pt`，不要提交。

只 add 报告目录，不要用 `git add .`：

```bash
git add reports/retacred/${DATE}
git diff --cached --name-only
```

确认 staged 文件都在 `reports/` 下，再 commit：

```bash
git commit -m "Add Re-TACRED ${DATE} results"
git push origin 1.1
```

推送后，把 GitHub 上 `1.1` 分支链接或 commit hash 发给负责人。负责人会检查结果，并决定是否合并到 `main`。

## 9. 负责人更新 main 后，下一轮如何继续

下一轮实验开始前，不要重新下载 zip，也不要重新 clone。直接在现有仓库里同步代码：

```bash
cd ~/projects/Q-Attention
git fetch origin
git switch 1.1
git status --short
git merge origin/main
```

如果 `git status --short` 显示有未提交改动，先处理这些改动，再 merge。常见情况是你已经复制了报告但还没 commit，这时先完成 commit 或把文件移走。

如果 merge 出现 conflict，不要随便删除文件。执行：

```bash
git status
```

把冲突文件列表发给负责人。

同步完成后，再跑新一轮实验，并重复“整理 reports -> commit -> push origin 1.1”的流程。

## 10. 如果之前已经用 zip 跑完实验

不用把 zip 目录继续当工作目录。建议这样迁移：

1. 保留旧 zip 目录，不要删除，因为里面可能有 `runs/` 结果。
2. 按本文第 2 节重新 `git clone` 一个干净仓库。
3. 在干净仓库中切到 `1.1`。
4. 从旧 zip 目录把需要的 summary、metrics、log 复制到新仓库的 `reports/` 目录。
5. 在新仓库里 `git add reports/...`、`git commit`、`git push origin 1.1`。

如果旧 zip 目录里缺少 `run_summary.json`，可以在旧 zip 目录先执行：

```bash
python experiments/summarize_relation_run.py --run_dir runs/retacred_full_gpu
```

然后再复制结果。

## 11. 每次提交前的最终自检

提交前请确认：

```bash
git branch --show-current
git status --short
git diff --cached --name-only
```

应该满足：

```text
当前分支是 1.1
staged 文件只在 reports/ 下
没有 data/、runs/、模型权重或原始数据
run_summary.json 和 run_summary.md 已生成
```

这套流程的核心是让实验可复现、结果可追踪、代码更新可同步。只要固定使用 `git clone` + `1.1` 分支，后续每轮实验都不需要再靠 zip 手动搬项目。