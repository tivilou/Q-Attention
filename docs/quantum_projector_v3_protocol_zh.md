# Q-LASS v3 实验协议

## 核心改动

当前正式方法不是把所有 Transformer 层的 key 混在一起学习一个投影器，而是对每个可干预层分别学习独立的监督量子投影器：

```text
第 l 层关系 key 样本 -> 第 l 层量子特征图 -> P_q^(l)
```

推理时，第 l 层只使用自己的 `P_q^(l)`，仍然保持原论文的干预形式：

```text
k'_l = k_l + g P_q^(l) k_l
```

单矩阵 projector 格式仍然保留，用于 classical、untrained quantum 和共享 projector 消融实验。classical-plus-quantum residual 仍然只是对比项，不是主方法。

## 严格的数据划分

`train` 只用于训练 baseline 和收集量子 projector 样本；`valid` 内部固定拆成 selection 和 acceptance 两部分，前者选择 steering gain，后者做 bootstrap 接受检验；`test` 只在 gain 冻结后做一次最终评估。

候选 gain 默认包含 `0.0`。如果所有 steering 都不能改善 validation macro-F1，流程可以选择零干预，避免为了得到正增益而人为调参。

新版正式配置使用 `strategy=coordinate`：按 key layer 逐层扫描 gain，未被选中的层使用 0 gain。`strategy=shared` 是旧的所有层共享一个 gain 的对照，`strategy=best_layer` 是只启用一个 validation 最优层的对照。

为避免 validation 上极小的偶然提升被当成有效信号，正式配置还使用 paired bootstrap。只有 acceptance 子集上的 macro-F1 增益 95% 置信区间下界大于 0，才接受非零 layer gain；否则 `selection_accepted=false`，最终测试自动回退为零干预。该判断不读取 test 标签。

正式阶段由配置中的 `stages` 自动启用；手动传入 `--stages` 时，以命令行参数为准。正式配置会运行：

```text
baseline
classical_projector / classical_steering
quantum_projector / quantum_steering
supervised_quantum_projector (layerwise)
supervised_quantum_gain_selection
spectral_sweep
routing
```

## 关键输出

```text
baseline/relation_supervised_quantum_projector.pt
baseline/relation_supervised_quantum_projector_metadata.json
supervised_quantum_gain_selection/gain_selection.json
supervised_quantum_gain_selection/metrics.json
supervised_quantum_gain_selection/run_info.json
supervised_quantum_gain_selection/predictions.jsonl
```

`gain_selection.json` 必须记录候选 gain、每层 gain、validation 指标、bootstrap 区间、接受/拒绝状态和选择指标。`metrics.json` 只记录冻结 gain 在 test 上的 baseline、steered 和 delta 指标。

## 运行命令

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda \
  2>&1 | tee runs/handoff_logs/retacred_low_resource_gpu.log
```

不要把 `test_path` 改成 `valid_path`，也不要在 test 上扫描 gain、rank、filter 或其他超参数。实验失败时保留完整日志和 `pipeline_summary.json`，不要直接修改源码重跑。
