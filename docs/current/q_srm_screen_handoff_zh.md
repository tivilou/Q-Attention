# Q-SRM 有界筛选实验：交付与回报

## 这次执行什么

默认命令先在所选 GPU 上执行合成 canary，再运行固定 Re-TACRED 小样本筛选，最后自动生成七文件白名单报告：

```bash
bash scripts/run_q_srm_handoff.sh --data-dir "$DATA_DIR" --gpu auto
```

这次是开发筛选，不是全量正式实验，也不是正式多 seed 复现。固定训练/验证/测试为 630/631/587 条，seed 为 13、29、53、71、97；每个 seed 先共享八轮 warm-start，再依次运行 disabled、quantum、fidelity、diagonal、key_mismatch 各六轮。factual CE 是唯一训练目标，验证集 Macro-F1 选 checkpoint，测试集不选优。历史 Q-CFRA 的 disabled 不作为同组对照。

Q-SRM 直接用集合条件化 square-root POVM 的 token 概率加权 V，保留零值 null，不二次 softmax 或重新归一化。它具有精确经典表达式；本次没有新颖性或量子优势结论。完整私有 trace 使用 sample-trace.v2，覆盖三个 split 的 initial/best/final 和训练 update0，独立 NumPy 重放必须通过。GitHub 仅接收聚合指标与追踪核验结果。

## 同步分支

先确认没有本地未保存修改；有修改则停止反馈，不要覆盖或强制清理。当前有 `1.1` 时：

```bash
git status --short
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
git status --short
```

若首次 clone 尚无本地 `1.1`，用 `git switch --track origin/1.1` 后再 merge `origin/main`。发生冲突则停止反馈。预检要求分支为 `1.1`，`origin/main` 和 `origin/1.1` 均是 HEAD 祖先，工作区干净。运行期间不要修改或切换代码。脚本不自动 fetch、提交或 push。

## 环境与 GPU 预检

使用现有兼容 conda 环境；要求 Python 3.10、PyTorch 2.7.1、NumPy 1.26.4。推荐 PyTorch CUDA12.6 构建，与已验证环境一致。不符合时返回版本和错误，不要自行换配方。脚本优先使用 `PYTHON_BIN`，否则依次找 `python`、`python3`：

```bash
conda activate qattn-cfra
export PYTHON_BIN="$(command -v python)"
export DATA_DIR="$PWD/data/relation/retacred"
"$PYTHON_BIN" -c 'import sys,torch,numpy; print(sys.version); print(torch.__version__,numpy.__version__,torch.cuda.is_available())'
nvidia-smi
bash scripts/check_q_srm_handoff.sh --data-dir "$DATA_DIR" --gpu auto
```

数据目录内需有 train.jsonl、valid.jsonl、test.jsonl，使用已有获授权的私有副本。代码检查三个源文件哈希与筛选数量；不自动下载，也不接受替代版本。预检不会训练。`--gpu auto` 尊重当前可见 GPU，只选一张空闲显存至少 8GiB 的卡，记录 `low_memory_serial` 配置（microbatch2、evaluation2、串行条件）。其他卡不占用；逻辑 batch16、轮数、seed 和科学门槛不随显存改变。显存不足请排队，勿改 batch/精度/预算。

我们已验证 POSIX CPU 真实入口和失败重试；本次交付时自己的服务器无可见 GPU。师弟必须由默认命令先通过 CUDA canary，才能进入 screen；CUDA 故障会阻断，不会悄悄回退 CPU。单独诊断可用：

```bash
bash scripts/run_q_srm_handoff.sh --canary-only --gpu auto
```

## 运行、恢复与停止

默认命令打印 `PRIVATE_RUN=/.../<timestamp>_seed13`，请保存该路径。该目录是五 seed 筛选的共同批次目录；seed13 后缀沿用项目批次命名，绝不表示只跑 seed13。所有原始数据、checkpoint、完整案例和日志留在 checkout 外。不要因为终端断开就新建实验；先确认旧进程已经退出，不能并发使用同一目录。

```bash
# RUN 改成首次打印的 PRIVATE_RUN；恢复不改变路径、数据、环境或 GPU 可见顺序
export RUN='/首次打印的完整私有路径/20261025T000000Z_seed13'
bash scripts/run_q_srm_handoff.sh --data-dir "$DATA_DIR" --gpu auto --run-dir "$RUN" --resume
```

恢复严格绑定代码哈希、数据、配置、环境和 GPU 选择。从已提交 minibatch 的模型、Adam、RNG、实际游标恢复；未提交 batch 重放。已完成阶段先验 hash/trace 后跳过。auto 选卡变化会拒绝恢复，此时用首次记录的可见 index，例如 `--gpu 0`，且保留原 `CUDA_VISIBLE_DEVICES`。换机器/设备或环境不可直接续跑，先反馈。损坏 checkpoint 不静默回退。

若训练已完成但报告导出失败，保留同一原始目录，仅重试导出（不会训练）：

```bash
bash scripts/export_q_srm_handoff.sh --run-dir "$RUN"
```

代码、环境、数据不符，OOM/nonfinite、追踪/哈希失败、缺完成标记、分支脏或同步失败，均应停止并反馈错误文本；不要改代码、seed、数据、budget、precision、控制组、gate 或重复启动救结果。未认证断电后的全文件持久性；不要把服务器本地磁盘当作长期备份。

## 只提交报告

成功后自动打印 `report=reports/q_srm_screen/<timestamp>_seed13`；常规运行不指定 `--report-dir`。精确七个文件为 summary.json、manifest.json、trace_validation.json、receipt.json、run_summary.json、data.sha256、data_counts.txt。`run_summary.data` 非空且包含源数量/筛选数量/哈希。默认报告目录不含原始句子、逐条预测、权重、张量、源选样索引、数据文件或完整日志。

将 REPORT 换成成功输出的相对路径，然后逐条检查：

```bash
export REPORT='reports/q_srm_screen/实际时间戳_seed13'
git status --short
git add -- "$REPORT"
"$PYTHON_BIN" experiments/q_srm_handoff.py check-staged --report-dir "$REPORT" --run-dir "$RUN"
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git commit -m "reports: Q-SRM bounded screen"
git push origin HEAD:1.1
git status --short
```

禁止 `git add .`、提交原始 runs/data/checkpoints/predictions/logs 或 force push。`check-staged` 必须显示 STAGED_REPORT_OK；若远端报告分支已更新，先 fetch，按前述顺序同步，保留原完成态再导出/检查。已提交的同一报告不需重复提交。push 被拒绝时停止反馈，不能重跑训练作为修复。

## 结果由我们审计

先验门槛不变：quantum 相对同 seed disabled 的测试 Macro-F1 平均差 >0、至少4/5 seed 正差、配对差值样本标准差不高于 disabled 样本标准差。输出绝对差及相对差（baseline=0 时相对差为空）。key_mismatch 接近限制机制归因，不替代任务效用判定；fidelity/diagonal 是机制对照，不是有效容量等价证明。

无严重缺陷的短筛选零增益归为 inconclusive，不据此声称方法自然任务失败。即便显示 `promote_to_formal_preparation`，也只提交报告等我们审计：不自动启动全量、不增加 seed。之后若发正式交付，先运行全数据单 seed13；我们审核后再决定正式多 seed。当前五 seed 仅属于冻结的小样本筛选。
