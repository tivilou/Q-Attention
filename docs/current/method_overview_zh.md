# 当前方法概览

## 当前主线

当前主方法是附加到关系抽取 Transformer 上的 standalone quantum attention intervention：

1. 冻结 relation baseline。
2. Q-VRES 根据 query、key、value 和关系条件计算 token 的因果 value evidence。
3. 量子模块读取关系-token 的 fidelity evidence，并把 evidence 转换为非负、保持 context mass 的 attention score transport。
4. score residual 仍保留参考论文的 query-aligned key-update 解释，同时直接作用在 attention score 上。
5. 正式任务比较 Q-VRES、classical causal transport、quantum key-only 和 disabled baseline。

## 当前代码入口

- 关系迁移 runner：`experiments/run_q_causal_value_evidence_relation_transfer.py`
- Q-VRES kernel：`src/q_attention/plugins/q_causal_value_evidence.py`
- Attention score adapter：`src/q_attention/adapters/attention_scores.py`
- 关系模型：`src/q_attention/models/relation_transformer.py`
- 正式单 seed：`scripts/run_qvres_relation_transfer_full.sh`
- 正式多 GPU：`scripts/run_qvres_relation_transfer_multi_seed.sh`

## 证据边界

- bounded real-data screen 只验证实现、数值稳定性、机制诊断和 task-transfer 信号。
- 当前扩大版 screen 中 Q-VRES test macro-F1 高于 disabled baseline 和 classical control，但只有单 seed，不能作为最终论文结论。
- 正式结论必须来自完整 Re-TACRED、五个 seed、独立 test 和聚合报告。
- 当前使用 PennyLane/经典模拟时，只能报告功能、任务效果和资源账本；不能据此声称当前硬件速度优势。
- 理想通用量子计算机的资源优势必须单独说明 state preparation、oracle/query、gate、depth、shots 和 readout 假设。
