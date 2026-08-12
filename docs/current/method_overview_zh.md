# 当前方法概览

## 当前主线

当前主方法是附加到关系抽取 Transformer 上的 standalone quantum attention intervention：

1. 冻结 relation baseline。
2. Quantum Attention Score Kernel 根据 query、key 和关系条件产生 attention-score residual。
3. Q-NESS 在同一个 relation-token quantum state 上独立测量两组非互补证据：
   - necessity：`O_N = connected Z_relation Z_token`；
   - sufficiency：`O_S = X_relation Z_token`。
4. 两个观测量不对易：`[O_N, O_S] != 0`。它们有独立 observable bank，不再使用 `drop = 1 - keep` 或固定互补预算。
5. 对每个 token 使用加权中心化残差：

   `delta_j = tau_N (n_j - sum_k a_k n_k) + tau_S (s_j - sum_k a_k s_k)`。

   该 score residual 仍可转换为 query-aligned key update，保留参考论文的 key steering 原理。

## Q-NESS 控制

| Selector ID | 作用 |
| --- | --- |
| `qness` | 主方法，非对易 necessity/sufficiency 读出 |
| `qness_commuting` | 将 sufficiency 观测量替换为 commuting Z-basis 控制 |
| `qness_separable` | 去掉 relation-token cross entanglement |
| `qness_phase_scrambled` | 保留幅度但打乱相位符号 |
| `qness_dephased` | 去掉依赖密度矩阵非对角项的 sufficiency 信号 |
| `qness_classical` | 参数匹配的 classical shared-trunk dual-head 控制 |

## 代码入口

- Attention score hook：`src/q_attention/models/relation_transformer.py`
- Score adapter：`src/q_attention/adapters/attention_scores.py`
- Quantum/classical core：`src/q_attention/plugins/attention_score_kernel.py`
- Q-NESS 和历史 Q-RES selector：`src/q_attention/plugins/attention_evidence.py`
- Selector 训练：`experiments/train_relation_counterfactual_evidence.py`
- Q-NESS toy gate：`experiments/run_qness_toy_gate.py`
- 机制诊断：`src/q_attention/experiments/attention_evidence_training.py`

## 当前证据边界

- Q-NESS 五 seed toy mechanism gate 已通过，输出只包含机制指标和量子资源诊断。
- Toy gate 不包含任务 validation loss、macro-F1 或 blind test，不能支持任务提升或量子优势结论。
- proportional Re-TACRED 和 full Re-TACRED 尚未运行 Q-NESS，必须在代码合并后另行执行。

## 非当前主入口

- Legacy key-steering/projector、dual Q-RES 和 observable expert routing 保留用于对照和历史兼容。
- 旧 smoke pipeline 和 transfer-screen wrapper 不代表当前 Q-NESS 实验协议。
