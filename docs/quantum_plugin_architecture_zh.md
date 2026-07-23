# 可组合量子注意力插件架构

## 目标

量子模块不写死在关系抽取模型中。基础模型保持不变，插件通过 key projection 的 forward hook 接入；基础模型参数可以全部冻结，插件参数独立训练、保存和加载。

核心机制仍为：

\[
k' = k + \Delta k
\]

其中每个 operator 插件产生 \(gP_qk\)，evidence gate 对 operator delta 做 token 级调制。多个 operator 同时启用时默认取平均，避免插件数量直接放大 steering 强度。

## 三个插件

### Headwise Quantum Projector

`HeadwiseQuantumProjectorPlugin` 为每层、每个注意力头学习独立的实振幅量子线路：

\[
P_{l,h}=U_{l,h}(\theta)D_rU_{l,h}(\theta)^T
\]

该构造天然满足对称性和幂等性。projector 只在单个 head 内作用，不混合不同注意力头的坐标空间。当前正式模型的 head dimension 为 16，正好对应 4 qubit 状态空间。

### Quantum Evidence Gate

`QuantumEvidenceGatePlugin` 使用有序 subject-object 和当前 token 的联合特征进行量子数据重上传，通过 Pauli-Z 期望值得到 \([-1,1]\) 内的 token/head gate。

与 operator 插件组合时，它调制 operator delta；单独启用时，以 identity operator 形成量子 token scaling 插件。

### Quantum Expert Bank

`QuantumExpertBankPlugin` 包含多个严格量子 projector。量子 router 根据有序 subject-object 表征生成 Born probability，并对 projector 专家进行输入相关的软路由。

每个专家仍是严格 projector；动态混合用于处理不同关系模式对 key 子空间的不同需求。

## 组合方式

以下八种模式都由同一个 `ComposableQuantumSteering` 执行：

```text
baseline
headwise_projector
evidence_gate
expert_bank
headwise_projector + evidence_gate
headwise_projector + expert_bank
evidence_gate + expert_bank
headwise_projector + evidence_gate + expert_bank
```

插件栈、基础模型和 checkpoint 相互独立。checkpoint 只保存插件参数、插件配置以及基础模型引用信息，不复制基础模型权重。

## 关系任务原型入口

训练时冻结基础模型，只优化插件：

```bash
python experiments/train_relation_quantum_plugins.py \
  --model_dir <baseline_dir> \
  --train_path <train.jsonl> \
  --valid_path <valid.jsonl> \
  --output_dir <plugin_output_dir> \
  --plugins headwise_projector,evidence_gate,expert_bank \
  --steering_anchor all_tokens \
  --device cuda
```

独立加载插件并评估：

```bash
python experiments/eval_relation_quantum_plugins.py \
  --model_dir <baseline_dir> \
  --checkpoint <plugin_output_dir>/quantum_plugins.pt \
  --data_path <test.jsonl> \
  --output_dir <eval_output_dir> \
  --device cuda
```

目前这些入口用于 toy 和小规模真实数据诊断，尚未加入正式 Re-TACRED pipeline。只有在小规模诊断显示稳定改善后，才更新正式 GPU 配置并交给合作者。
