# Q-Attention 项目说明给实验合作者

面向：`dzy958`

这份文档用于说明当前 Q-Attention 项目和原论文实验之间的关系，以及接下来需要如何运行 Re-TACRED 实验、需要返回哪些结果。不需要先读完整个代码库；先理解项目目标、数据位置、运行顺序和结果文件即可。

## 1. 概括

本项目并不是复现实验性质的 prompt highlighting，而是在原论文“谱投影注意力 steering”思想基础上，做一个面向真实 NLP 信息抽取任务的量子增强版本：

```text
Q-Attention = spectral key steering + quantum-inspired projector + spectral filtering + adaptive routing
```

当前第一批真实数据实验使用：

```text
Re-TACRED relation extraction
```

当前阶段的目标是先验证这个机制在真实关系抽取数据上是否能跑通、是否相对 baseline 有稳定收益，尤其是在 low-resource 场景下是否更明显。

## 2. 和原论文实验的关系

你之前跑过的原论文实验可以理解为：

```text
在注意力 key space 里学一个 projector P
推理时把 attention key 从 k 改成 k' = k + gPk
模型权重不更新，只通过 key steering 改变注意力行为
```

本项目保留的是这个核心机制：

```text
learn P offline
apply k' = k + gPk during inference/evaluation
keep backbone weights frozen during steering
```

但要改变研究目标：

| 项目 | 原论文实验 | 当前 Q-Attention |
|---|---|---|
| 任务 | prompt highlighting / attention steering 验证 | 真实 NLP 信息抽取，当前是 Re-TACRED 关系抽取 |
| 数据 | 原论文任务数据 | Re-TACRED canonical JSONL |
| 模型 | 原论文代码中的模型/LLM 实验流程 | 当前先用项目内自写 Transformer RE baseline |
| projector | classical spectral projector 为主 | classical + quantum-inspired + spectral filters + routing |
| 评价 | 注意力/高亮效果 | macro-F1、accuracy、precision、recall、loss、routing entropy 等 |
| 代码 | 原论文源码 | 本仓库从零写的公开代码，不包含原论文源码 |

所以现在的项目可以理解为：

```text
用原论文的 key-space spectral steering 原理，迁移到真实 relation extraction，并加入量子机器学习创新模块。
```

## 3. 当前已经完成的内容

代码侧已经完成：

```text
1. tensor-level key steering
2. Transformer encoder key steering adapter
3. relation extraction baseline
4. Re-TACRED 数据构建和转换脚本
5. classical spectral projector
6. quantum-inspired projector
7. spectral filtering sweep
8. adaptive projector routing
9. 实验结果汇总工具
10. Re-TACRED debug / low-resource / full GPU 配置
```

当前 smoke gate 已通过：

```text
Re-TACRED 256 train / 128 valid 小样本 CPU smoke 已跑通
baseline -> classical steering -> quantum steering -> spectral sweep -> routing 全链路可执行
```

当前 GitHub 仓库：

```text
https://github.com/tivilou/Q-Attention
```

注意：数据不在 GitHub 上，因为 TACRED/Re-TACRED 涉及 LDC 授权和数据分发限制。

## 4. 数据包使用

我会发你两个文件：

```text
retacred_q_attention_data.tar.gz
retacred_q_attention_data.sha256
```

数据包只包含处理好的 canonical Re-TACRED 数据：

```text
data/relation/retacred/train.jsonl
data/relation/retacred/valid.jsonl
data/relation/retacred/test.jsonl
data/relation/retacred/data_config.json
```

收到后，把它放到 Q-Attention 仓库根目录，然后执行：

```bash
sha256sum retacred_q_attention_data.tar.gz
cat retacred_q_attention_data.sha256

tar -xzf retacred_q_attention_data.tar.gz
ls -lh data/relation/retacred
```

当前期望 SHA256 是：

```text
c06b9647b5977a3a06fd2a4f338b50931567def231d77a37c8fe6bf93a36a64c
```

不要把 `data/` 或这个压缩包提交到 GitHub。

## 5. 你需要先确认环境

进入仓库后，先记录环境：

```bash
git rev-parse HEAD
which python
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi
python -m pytest -q
```

如果 `python -m pytest -q` 不能通过，先不要跑正式实验，把错误日志发回来。

## 6. 推荐运行顺序

### 第一步：GPU debug run

这是环境验证，不作为论文结果。

```bash
mkdir -p runs/handoff_logs

python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_debug_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_debug_gpu
```

如果这一步失败，请先停止，不要继续跑 low-resource 或 full。

### 第二步：low-resource run

这是比较重要的一组实验，用来观察少样本情况下 projector / quantum / routing 是否有收益。

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_low_resource_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_low_resource_gpu
```

### 第三步：full run

这是第一版完整 Re-TACRED 验证实验。

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_full_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_full_gpu.log

python experiments/summarize_relation_run.py \
  --run_dir runs/retacred_full_gpu
```

## 7. 每个实验会产生什么文件

以 `runs/retacred_full_gpu/` 为例：

```text
runs/retacred_full_gpu/
  pipeline_summary.json
  run_summary.json
  run_summary.md

  baseline/
    model.pt
    metrics.json
    vocab.json
    labels.json

  classical_steering_eval/
    metrics.json
    predictions.jsonl
    run_info.json

  quantum_steering_eval/
    metrics.json
    predictions.jsonl
    run_info.json

  spectral_filter_sweep/
    results.jsonl
    summary.json

  relation_routing_eval/
    metrics.json
    predictions.jsonl
    routing.jsonl
    run_info.json
```

你最应该先看的文件是：

```text
runs/<run_name>/run_summary.md
```

它会汇总：

```text
baseline
classical_steering
quantum_steering
spectral_sweep_best
adaptive_routing
```

以及这些指标：

```text
macro_f1
accuracy
macro_precision
macro_recall
loss
delta_macro_f1
```

## 8. log 文件记录什么

例如：

```text
runs/handoff_logs/retacred_full_gpu.log
```

会记录终端中打印出来的完整过程，包括：

```text
每个 epoch 的 train_loss 和 valid metrics
projector 构建路径、shape、采样 key vector 数量
classical / quantum steering 指标
spectral sweep 每组参数结果
routing entropy 和 expert 分布
每一步实际执行的命令
如果失败，会包含 traceback
```

如果实验失败，log 是最重要的排错文件。

## 9. 如何理解结果

我们最关心的是：

```text
quantum_steering 是否优于 baseline
spectral_sweep_best 是否优于 baseline
adaptive_routing 是否优于 baseline
low-resource 下提升是否比 full-data 下更明显
```

理想结果形态：

```text
baseline < classical_steering < quantum_steering / spectral_sweep_best < adaptive_routing
```

但第一轮实验的目标仍然是机制验证，不是直接宣称 SOTA。当前 baseline 是项目内自写的小型 Transformer，不是 BERT/RoBERTa 预训练 encoder。若这轮结果显示机制有效，下一步会把同样的 projector / steering 机制迁移到预训练 encoder 上。

## 10. 你需要返回什么

每个 run 完成后，请返回或汇总以下文件内容：

```text
runs/<run_name>/run_summary.md
runs/<run_name>/pipeline_summary.json
runs/<run_name>/baseline/metrics.json
runs/<run_name>/classical_steering_eval/metrics.json
runs/<run_name>/quantum_steering_eval/metrics.json
runs/<run_name>/spectral_filter_sweep/summary.json
runs/<run_name>/relation_routing_eval/metrics.json
runs/handoff_logs/<run_name>.log
```

如果文件太大，优先发：

```text
run_summary.md
pipeline_summary.json
各阶段 metrics.json / summary.json
log 最后 100 行
```

## 11. 如果失败，请这样反馈

请不要自己改源码或配置文件。直接反馈：

```text
1. git commit hash
2. exact command
3. GPU model
4. Python / torch version
5. full traceback
6. log last 100 lines
7. 是否重跑仍然失败
```

这能让代码侧快速定位问题。

## 12. 当前代码中你可能会接触的关键文件

```text
configs/retacred_debug_gpu.json
configs/retacred_low_resource_gpu.json
configs/retacred_full_gpu.json
experiments/run_relation_smoke_pipeline.py
experiments/summarize_relation_run.py
experiments/build_retacred_from_tacred.py
experiments/prepare_relation_data.py
docs/retacred_experiment_handoff.md
```

核心方法代码在：

```text
src/q_attention/steering.py
src/q_attention/projectors.py
src/q_attention/quantum.py
src/q_attention/routing.py
src/q_attention/experiments/relation_steering.py
```

你运行实验时通常不需要修改这些文件。

## 13. 最重要的注意事项

```text
不要提交 data/
不要提交 runs/
不要改 configs 后直接跑正式结果
不要只发截图，尽量发纯文本 log 和 JSON/Markdown 结果
先 debug，再 low-resource，最后 full
```