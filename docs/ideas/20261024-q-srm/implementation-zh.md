# Q-SRM 实现与验证

本实现把第二层 attention 的 context 替换为集合条件化的 square-root POVM 加权值；第一层、Q/K/V 基础投影和 Transformer 主干冻结，分类头继续学习。查询与键分别由输入条件化的三比特线路编码。两个映射共计 144 个名义参数，跨头共享；参数数量相同不代表有效容量相同。

给定有效键态 `k_j`，先构造 `S = Σ k_j k_j† + 0.01 I`，再取 `G = S^(-1/2)`。token 权重为 `|k_j† G q_i|²`，另保留 `0.01 q_i† S^(-1) q_i` 的 null 概率。context 直接等于 token 权重乘 V，null 的值为零，不额外 softmax，也不把 token 权重重新归一化。

谱分解前向使用 complex128/float64。逆平方根反向使用 Loewner 差商的闭式表达式，避免重复特征值处的特征向量梯度奇异；仅支持一阶梯度。矩阵、概率和线路中间结果可以独立用 NumPy 重放。此有限维实现有精确经典等价表达式，不构成量子资源优势证据。

五种条件在同一 seed 下使用同一个 warm-start、同一个 factual CE 目标、相同选样和更新次数：

| 条件 | 行为 |
|---|---|
| disabled | 原始 softmax，适配器返回原始路径；插件参数为零 |
| quantum | 完整集合条件化 POVM 与显式 null |
| fidelity | 相同线路，逐对 fidelity 加 null 归一化，不使用集合 Gram |
| diagonal | 构造 Gram 前去掉键密度矩阵的非对角项 |
| key_mismatch | 只循环错配有效键态，V 的位置不变 |

代码只提供合成 canary 和有界 screen。canary 使用 12/6/6 条训练/验证/测试合成记录、seed 13、每阶段两轮。screen 固定使用 630/631/587 条 Re-TACRED 记录、五个 seed、八轮 warm-start 和六轮干预。历史 Q-CFRA 的三视图损失和历史 disabled 结果均不能作为本轮的同组对照。screen 仍是复用过的开发子集，不能当作正式独立证据。

运行前要求 `docs/validation/q_srm_review.json` 对当前源代码哈希给出 scope=both 的可接受审查。源代码、数据、配置、运行环境任一变化都会拒绝续跑。原始输出必须位于 checkout 之外。

在已配置环境中验证小型 canary：

```bash
cd /home/Q-Attention/q-srm-implementation
export PYTHON_BIN=/usr/local/miniconda3/envs/qattn-cfra/bin/python
bash scripts/run_q_srm_screen.sh \
  --profile canary --device cpu \
  --output-dir /home/Q-Attention/q-srm-private/canary_NEW_ID
```

同一命令追加 `--resume` 可从已提交的 minibatch 边界继续。每批 checkpoint 包含模型、Adam、随机数状态、下一批索引和验证集选优游标；未提交批次会重放。已完成条件先验证文件和数值追踪再跳过。保留最近两代 checkpoint，但损坏会停止并报告，不静默回退。仅重新导出时同时追加 `--resume --export-only`，不会启动训练子进程。

成功后自动生成输出目录下的 `export/`。其白名单只有 `summary.json`、`manifest.json`、`trace_validation.json`、`receipt.json`。原始句子、预测明细、checkpoint 和全量张量留在私有输出目录。导出采用临时目录及原子改名，失败清理后可从同一个完成态重试。

每个条件保存十份完整 `sample-trace.v2`：每个 split 的 initial/best/final，加训练集固定样本的真实 update0。训练通道的 loss/gradient 对应逻辑 minibatch，前向展示固定的首个样本；诊断重放不执行 optimizer step。验证器检查结构、语义、逐文件哈希、任意坐标读取、线路、POVM、mask、context 和参数差分。

已通过 29 项机制/追踪测试、31 项已有模型与数据回归测试，以及 1 项真实父进程集成测试，共 61 项。集成测试确认连续运行与批次断点续跑的模型、Adam、随机数状态完全一致，并确认导出失败清理、原结果重试和 checkpoint 篡改拒绝。五种条件共验证 50 份完整追踪、1810 个完整数值数组。独立审查为 scope=both、`PASS_WITH_RISK`；外部审查请求超时后使用同一代码包的受限子代理审查。

完整记录见 `docs/validation/q_srm_validation.json`。保留的干净 canary 是其中 `lifecycle.clean_source_run` 指向的 `direct/`；`resumed/` 在通过恢复及导出检查后，被测试故意破坏一个 checkpoint，用于验证拒绝篡改，不能作为后续运行源。

目前只验证了 Linux/POSIX 的 CPU 路径，服务器没有可见 NVIDIA 设备。GPU 可用后仍需 CUDA canary；强制进程终止、断电持久性及原生 Windows 执行尚未认证。本文件不是师弟的全量正式实验交接包。本轮未执行真实数据五 seed screen 或任何全量训练，未产生任务效果结论。
