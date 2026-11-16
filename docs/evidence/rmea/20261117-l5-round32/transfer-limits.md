# Q-RMEA：优化隐式偏置原文与实际实现的对应（Round32）

本轮核对了三个精确版本原文的指定范围。结论是这些工具目前没有给出当前Q-RMEA的优化优势定理；这不改变原来限定L4，也没有排除所有最大间隔分析。没有新模型、训练、数据集读取或L5验收。

## 原文到底证明了什么

1. [Transformers as Support Vector Machines](https://arxiv.org/abs/2308.16898v3)：p2脚注明确固定head。p4定义GD与全局约束regularization path（RP）；p6 Theorem2区分直接W的Frobenius范数与分解KQ的核范数偏置，依赖唯一最优词和可行分离。p8 Theorem3及p11 Theorem5的GD结论还有support/初始局部锥等条件。范数增长、全局RP和有限训练轨迹不是同一事实。
2. [Optimizing Attention with Mirror Descent](https://arxiv.org/abs/2410.14581v5)：pp5-10为二分类、固定head的显式ℓp镜像下降与局部收敛条件。pp11-12确实给出了attention/head共同更新，不能把全部先前理论说成只处理固定value。pp49-50 AssumptionB/Theorem31的联合结论针对全局JointRP最优解，要求可行attention SVM、选词与标签间隔损失之间的几何关系，以及r(R)=exp(Θ(R))、R→∞。pp50-52用指数衰减的非最优词质量作比较；它不是对任意有限AdamW的共同训练证明。
3. [Quantum Natural Gradient](https://arxiv.org/abs/1909.02108v3)：p3式7/10-13明确要求Fubini–Study/QGT度量预条件，更新需解gΔθ=−η∇L或使用伪逆。量子状态参数化本身不等于执行了这一算法。p7也明确区分该度量与Adam的梯度历史机制。

阅读范围、PDF/解析哈希和图像页在[reading-receipts.json](reading-receipts.json)。没有声明全文证明审计、独立科学评审或实验复现。

## 原生模型与协议

缓存源码a4dfe1f的rmea.py:55-65：πtanh角度→三量子比特纯态→40个独立二元关系效应→q=|φ†ψ|²→(q+ε)在每个关系有效词集内归一化→带符号utility加权→40类logit。frozen_feature_head.py的value为1024→64仿射映射，decoder为40×64，另有40维输出bias。encoder冻结不表示这些head参数或上下文化输入在编辑后冻结。

现有公开审计/protocol/seed_configs的五种子均为20epochs、batch128、AdamW、lr0.001、weight_decay0.01、clip_norm1.0；历史执行源SHA为494c9e3e…，与本轮重开的开发快照分开保留。协议记录复核不是逐步重放每次优化更新。精确来源和适用检查见[source-verification.json](source-verification.json)。

## 有效分数有界不等于输出有界

在一个含T个有效词的非空集合上，ε=10⁻⁶固定，

    α_t = (q_t+ε)/Σ_j(q_j+ε) = softmax(log(q+ε))_t.

理想归一化纯态下0≤q_t≤1，因此有效log分数最大间隔至多log((1+ε)/ε)，且

    α_t ≤ (1+ε)/(1+Tε) < 1，T>1。

这只是整个原生类的保守上界，不保证端点在浅电路中同时可达；浮点舍入也不是严格实数不等式。该界可以非常接近1，不能解释当前实测增益或宣称实际attention足够平滑。它说明固定ε下有效分数间隔不沿训练无限增大；并不说明原始参数范数有界，更不证明所有最大间隔方法都失效。若改变ε或加入温度发散，就是另一个方法/极限条件。

输出有独立的放大途径。固定attention，令value(h)=Ah+c，decoder行为D_r，最终bias为b_r，则

    z_r = Σ_t α_rt D_r·(Ah_t+c) + b_r。

对任意s，同时将A,c,D乘exp(s/2)，b乘exp(s)，attention参数保持原值，恰好得到z'_r=exp(s)z_r。因此仅有分数有界不足以推出logit有界；若固定符号间隔已可分，缩放可继续降低分类CE而不改变选词。这里没有证明当前样本已可分、训练会遵循此路径、AdamW极限等于任何最优解或量子优势。即使可把value和decoder写成有效线性权重，分解参数的更新几何也没有自动变成直接优化独立权重。

## 决定与下一条问题

当前已核查的SVM、ℓp-MD及QNG结论均不直接认证原生有限AdamW的Born注意力训练。已知上限、缩放恒等式和coadaptation观察只是工具，不作为新理论或L5贡献。旧Q对uniform的Macro与固定值局部CE对齐证据保持，未变成优化路径或量子必要性证据。

下一步只核查原始quasi-homogeneous／分解分类器隐式偏置理论：是否能同时容纳零缩放率的有界可学习attention与正缩放率value/decoder/bias；若只在gradient-flow等理想化条件成立，必须与实际AdamW分开，并要求新增的可区分预测。尚未选新方法、改optimizer或启动训练，不重开已停实验家族；新全量正式实验仍由师弟执行。
