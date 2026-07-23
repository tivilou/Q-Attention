# 合作者 Git 实验工作流（1.1 分支）

本文面向负责 GPU 实验的合作者。运行时以最新 `origin/main` 为准，并确保代码至少包含 `20bc225` 的 validation sampling 修复。目标是把实验流程从“下载 GitHub zip、手动上传下载结果”改成可追踪的 Git 工作流：代码从 `main` 来，实验结果提交到 `1.1` 分支，负责人再把结果合并回 `main`。

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

这一步只在“第一次规范化工作流”时做。不要用 reset 或 force push 覆盖远端已有的报告。

请先确认没有需要保留的本地改动：

```bash
git status --short
```

如果输出为空，再执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
```

如果本地还没有 `1.1` 分支，执行：

```bash
git switch --track -c 1.1 origin/1.1
git merge origin/main
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

如果 merge 出现冲突，停止并把冲突文件列表发给负责人，不要删除文件或 force push。后续同步代码统一使用：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/main
```

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

当前正式实验必须使用 CUDA；实验前还要确认：

```bash
git rev-parse HEAD
git status --short
```

`git status --short` 必须为空。不要手动修改源码或配置后直接跑正式实验。

`git.dirty=false` 的含义是：实验运行时 Git 管理的源码和配置没有未提交修改，当前 commit 可以复现本次运行。运行目录中的 ignored 文件不影响这个字段。检查方式：

```bash
git status --short
```

没有输出时才是 clean；如果有输出，先停止并处理，不要继续提交该实验结果。

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
set -o pipefail
```

### 5.1 Debug run

先跑 debug，确认环境没有问题：

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_debug_gpu_$(date +%Y%m%d-%H%M%S).log
```

脚本会自动创建带时间戳的目录，期望生成：

```text
runs/retacred_debug_gpu/<日期-时间>/pipeline_summary.json
runs/retacred_debug_gpu/<日期-时间>/run_summary.json
runs/retacred_debug_gpu/<日期-时间>/run_summary.md
runs/retacred_debug_gpu/<日期-时间>/supervised_quantum_gain_selection/gain_selection.json
```

### 5.2 Low-resource run

Debug 通过后再跑：

```bash
LOW_LOG=runs/handoff_logs/retacred_low_resource_gpu_$(date +%Y%m%d-%H%M%S).log

python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee "${LOW_LOG}"

LOW_RUN=$(ls -dt runs/retacred_low_resource_gpu/*/ | head -n 1)
echo "LOW_RUN=${LOW_RUN}"
```

不要添加 `--output_dir` 或 `--max_valid_records`。

### 5.3 Full run

最后跑 full：

```bash
FULL_LOG=runs/handoff_logs/retacred_full_gpu_$(date +%Y%m%d-%H%M%S).log

python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda \
  2>&1 | tee "${FULL_LOG}"

FULL_RUN=$(ls -dt runs/retacred_full_gpu/*/ | head -n 1)
echo "FULL_RUN=${FULL_RUN}"
```

不传 `--output_dir` 时，每次运行都会自动创建 `runs/<配置名>/<日期-时间>/`，不会覆盖之前的正式结果。

## 6. 如果 run_summary 没有生成

pipeline 正常完成时会自动调用 `summarize_relation_run.py`，不需要再手动执行。汇总成功时会打印类似：

```json
{"output_json": "runs/retacred_full_gpu/<日期-时间>/run_summary.json", "output_markdown": "runs/retacred_full_gpu/<日期-时间>/run_summary.md", "rows": 6}
```

`rows` 数量应与实际运行阶段一致，不要把固定的旧数量当作成功条件。

如果目录里没有 `run_summary.json` 或 `run_summary.md`，先不要重新下载 zip，先检查路径：

```bash
pwd
git branch --show-current
FULL_RUN=$(ls -dt runs/retacred_full_gpu/*/ | head -n 1)
echo "FULL_RUN=${FULL_RUN}"
ls -la "${FULL_RUN}"
python experiments/summarize_relation_run.py --run_dir "${FULL_RUN}"
```

常见原因是：命令不是在项目根目录执行，或者查看的是另一个旧目录。

## 7. 整理要提交的结果

不要再使用个人 `copy.sh`，也不要手工逐项复制。full 和 low-resource 都成功后，确认下面四个变量仍指向本轮运行：

```bash
echo "FULL_RUN=${FULL_RUN}"
echo "LOW_RUN=${LOW_RUN}"
echo "FULL_LOG=${FULL_LOG}"
echo "LOW_LOG=${LOW_LOG}"
git status --short
```

此时 `git status --short` 必须为空。然后运行仓库内正式导出脚本：

```bash
REPORT_TAG=$(date +%Y%m%d-%H%M%S)
REPORT_DIR=reports/retacred/${REPORT_TAG}

python scripts/export_retacred_report.py \
  --full-run "${FULL_RUN}" \
  --low-resource-run "${LOW_RUN}" \
  --full-log "${FULL_LOG}" \
  --low-resource-log "${LOW_LOG}" \
  --report-tag "${REPORT_TAG}"
```

成功时应输出：

```text
Report exported: reports/retacred/<日期-时间>
Files exported: 20
```

脚本会检查工作树、运行 commit、`git.dirty=false`、CUDA、配置哈希、test 隔离、阶段返回码和必需文件，并且只导出 20 个允许公开提交的配置、摘要、指标和日志尾部。任何检查失败都不要手工绕过，直接把报错发给负责人。

不要提交 `data/`、`runs/`、模型权重、predictions、JSONL 或完整日志。`supervised_quantum_gain_selection/` 和 projector metadata 只在负责人明确要求排错时私下提供。

## 8. 提交结果到 1.1

```bash
git add "${REPORT_DIR}"
git diff --cached --check
git diff --cached --name-only
git status --short
```

确认暂存区只有本次 `reports/retacred/${REPORT_TAG}/` 下的 20 个文件，再执行：

```bash
git commit -m "Add Re-TACRED ${REPORT_TAG} results"
git push origin 1.1
git rev-parse HEAD
```

把最后输出的 commit hash 发给负责人。不要使用 `git add .`。

## 9. 负责人更新 main 后，下一轮如何继续

下一轮实验开始前，不要重新下载 zip，也不要重新 clone。直接在现有仓库里同步代码：

```bash
cd ~/projects/Q-Attention
git fetch origin --prune
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
git diff --check
git diff --cached --name-only
```

应该满足：

```text
当前分支是 1.1
staged 文件只在 reports/ 下
没有 data/、runs/、模型权重或原始数据
run_summary.json 和 run_summary.md 已生成
pipeline_summary.json 中 `git.dirty` 为 false
pipeline_summary.json 中 validation/test 使用 proportional sampling 或 source
```

这套流程的核心是让实验可复现、结果可追踪、代码更新可同步。只要固定使用 `git clone` + `1.1` 分支，后续每轮实验都不需要再靠 zip 手动搬项目。
