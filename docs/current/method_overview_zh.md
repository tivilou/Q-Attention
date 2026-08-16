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
- validation 诊断：`experiments/diagnose_qvres_relation_transfer.py`
- Q-VRES kernel：`src/q_attention/plugins/q_causal_value_evidence.py`
- Attention score adapter：`src/q_attention/adapters/attention_scores.py`
- 关系模型：`src/q_attention/models/relation_transformer.py`
- 已完成的正式单 seed runner：`scripts/run_qvres_relation_transfer_full.sh`
- 当前诊断入口：`scripts/run_qvres_validation_diagnostic.sh`

## 证据边界

- bounded real-data screen 只验证实现、数值稳定性、机制诊断和 task-transfer 信号。
- seed 13 full pilot 的 Q causal validation gate 未通过，当前不启动五 seed。
- 当前先对已完成 raw run 做 validation 机制诊断；正式结论仍需后续受控实验和独立 test，当前没有最终论文结论。
- 当前使用 PennyLane/经典模拟时，只能报告功能、任务效果和资源账本；不能据此声称当前硬件速度优势。
- 理想通用量子计算机的资源优势必须单独说明 state preparation、oracle/query、gate、depth、shots 和 readout 假设。
