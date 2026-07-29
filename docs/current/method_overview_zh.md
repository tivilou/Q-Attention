# 当前方法概览

## 当前主线

当前主方法是一个可附加到关系抽取 Transformer 上的 standalone quantum attention intervention：

1. 冻结 relation baseline。
2. Quantum Attention Score Kernel 根据 query、key 和关系条件产生 attention-score residual。
3. Dual Q-RES 使用共享 PQC 的两种读出：
   - signed phase-sensitive readout 负责 attention steering；
   - positive connected-projector readout 负责 counterfactual sufficiency。
4. `context_budget` 将 sufficiency evidence mass 固定为 `0.35`。
5. 干预可以直接作用于 attention score，并具有 query-aligned key update 的等价解释。

## 当前主实验比较

| 名称 | Core | Selector | 作用 |
| --- | --- | --- | --- |
| Baseline | 无 | 无 | 冻结任务模型 |
| Quantum | quantum | quantum dual Q-RES | 主方法 |
| Local classical | classical | classical | 参数匹配的局部经典控制 |
| Strong classical | classical | classical_strong | 更强的经典控制 |
| Separable | quantum | quantum，无 cross entanglement | 量子纠缠消融 |

当前 full seed 13 预跑要求前三个 selector 比较。Separable control 在流程确认后补充；正式结论使用五 seed。

## 代码入口

- Attention score hook：`src/q_attention/models/relation_transformer.py`
- Score adapter：`src/q_attention/adapters/attention_scores.py`
- Quantum/classical core：`src/q_attention/plugins/attention_score_kernel.py`
- Q-RES selector：`src/q_attention/plugins/attention_evidence.py`
- Expert routing 扩展：`src/q_attention/plugins/attention_routing.py`
- Core 训练：`experiments/train_relation_attention_score_kernel.py`
- Selector 训练：`experiments/train_relation_counterfactual_evidence.py`
- 机制诊断：`src/q_attention/experiments/attention_score_training.py` 和 `attention_evidence_training.py`

## 当前证据边界

- Toy 五 seed 和 proportional Re-TACRED 五 seed 已通过候选筛选。
- full baseline 与 score core 已做开发侧检查。
- full selector、多 seed 和 blind test 尚未形成正式结论。
- 不应把 proportional subset 的正向结果表述为量子优势。

## 非当前主入口

- Legacy key-steering/projector 代码保留用于对照和历史兼容。
- Observable expert routing 已实现，但不是本轮 full seed 13 的必跑阶段。
- 旧 smoke pipeline 和 transfer-screen wrapper 不代表当前 dual Q-RES 配置。
