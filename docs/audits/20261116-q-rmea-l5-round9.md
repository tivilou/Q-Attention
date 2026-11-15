# Round9：Macro-F1 决策理论与量子检测的接口审查

结论为 `revision_required / L5_NOT_ACCEPTED`。本轮完成目标与代码对应性的审查，排除一条未经论证的最优性迁移；未选择新模型。原条件性 ordinary scoped L4 及其统计假设、原始运行、历史卡均保持不变。

## 1. 三个对象分别定义

总体混淆矩阵为 `C_ij=P(Y=i,d(X)=j)`，其中 `d` 是类别决策。记 `pi_j=sum_k C_jk`、`q_j=sum_i C_ij`、`t_j=C_jj`。在正分母范围，

`MacroF1(C) = (1/K) sum_j 2 t_j/(pi_j+q_j)`。

有限 cohort 使用经验计数代入上述式子；它不同于“随机有限批次 F1 的期望”。缺失类的 `0/0 -> 0` 计分约定另行处理，不在正分母的可微结论内。本轮 1024 exposed-valid 只有39个实际标签，因此 `.975` 只是固定40类评分下的样本指标上界，不是总体上界或可达收益。

实际模型执行 `d(x)=argmax_r z_r(x)`；训练 CE 是对 logits softmax 的另一目标。CE、硬 argmax 准确率与 MacroF1 排序不同并不矛盾。当前数据没有证明 softmax 是真实总体后验 `eta(x)`，也没有证明校准差异。

## 2. 多分类原文给出的已知结构及其限制

Ramaswamy/Narasimhan/Agarwal，版本化原文 [1501.00287v1](https://arxiv.org/pdf/1501.00287v1)，实际阅读 pp1–8、13–14，视觉核对 pp4、7。p4表1把 MacroF1 标为非凹；该表分子有2，随后正文公式漏2，本审查采用标准/表1归一化。p7定理2要求后验分布对单纯形基准测度绝对连续，并要求指标的连续、可微、有界和单调条件。它在未知总体最优混淆矩阵 `C*` 处构造 `G*=gradient psi(C*)`，不是把当前经验 CM 直接代入就得到最优分类器。有限原子经验分布也不是该绝对连续假设的直接实例。

固定真实类别边缘 `pi` 的可行方向上，可使用约化 gain

`G_ij = (2/K) [ 1{i=j}/(pi_j+q_j) - t_j/(pi_j+q_j)^2 ]`。

这是已知混淆指标的微分，不是新定理。若从所有 CM 元素独立变动的完整梯度开始，还存在对每一行固定的项 `-(2/K)t_i/(pi_i+q_i)^2`；在保持行和的方向或类别决策 argmax 比较中，该项抵消。相应加权后验决策比较

`eta_j(x)/(pi_j+q_j) - t_j/(pi_j+q_j)^2`。

这里的 `t,q` 应处于所声称的总体最优对象；它们未知。对非凹 MacroF1，一般驻点/一次局部线性化不提供全局最优证书。原文 p13定理10 的 BayesCG 一致性明确要求凹且光滑，不能把那条保证泛化到表中标为非凹的 MacroF1。未核证全篇附录、原代码或实证结果。

## 3. 量子检测原文优化的目标

Eldar/Megretski/Verghese，[quant-ph/0205178v1](https://arxiv.org/pdf/quant-ph/0205178v1)，实际阅读 pp1–11、视觉核对 p10。给定先验 `pi_i` 与类条件量子态 `rho_i`，选择联合 POVM `M_i >= 0, sum_i M_i=I`，最小错误检测等价于

`max_M sum_i pi_i Tr(M_i rho_i)`。

这是正确分类概率的线性函数。p10式15–20 的 dual 为 `min_X Tr(X), X >= pi_i rho_i`，以及互补条件；p11继续给出该目标的最优性条件。已知线性 SDP 证书不是 MacroF1 的证书。

即使另造一个按 POVM outcome 随机输出类别的分类器，`C_ij=pi_i Tr(M_j rho_i)`，MacroF1 仍是多个比值的和。把线性检测器中的 `pi_i rho_i` 换为依赖当前 CM 的梯度算子，仅能构成待证明的优化步骤；不能继承全局最优性。原文后续秩结果与数值实现未审查。

## 4. 精确反例：最佳错误率不蕴含最佳 MacroF1

这是复核已知目标区别的二维 commuting fixture，不是新研究定理或 Re-TACRED 证据。令两正交输入 `a,b` 的联合质量如下：

| 真实类别 | a | b | 先验 |
|---|---:|---:|---:|
| 1 | .72 | .11 | .83 |
| 2 | .08 | .09 | .17 |

对应 `rho1=diag(72/83,11/83)`，`rho2=diag(8/17,9/17)`。

`M1=I,M2=0` 的 accuracy 为 `.83`。取 dual `X=diag(.72,.11)`，则 `X-.83rho1=0`、`X-.17rho2=diag(.64,.02)>=0`，primal/dual 同为 `.83`，故它是全局最小错误测量。其 MacroF1 为 `83/183 = .453551912568306`。

另一合法测量 `M1=|a><a|,M2=|b><b|` 的 accuracy 为 `.81`，MacroF1 却为 `4131/6031 = .6849610346542861`。后者不在此被声称为 F1 全局最优。精确 Fraction 及 PSD 对角、归一化、零 dual gap 均由 `decision_counterexample.py` 检查。此例否定普遍的错误率→F1最优性迁移；它没有否定其他量子理论或推断真实 cohort 的改进。

## 5. 回到实际 Q-RMEA：缺失的接口

原模型 `RMEAHead.forward`（round3私有源镜像 `source/legacy/rmea.py`，SHA256 `9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0`）实际执行：

`psi_t = circuit(pi*tanh(W[h_t,roles_t]+a))`；

`E_r=|phi_r><phi_r|`，40个关系是独立二元 effect，未要求 `sum_r E_r=I`；

`alpha_rt = mask_t (Tr(E_r |psi_t><psi_t|)+1e-6) / sum_s mask_s(Tr(E_r |psi_s><psi_s|)+1e-6)`；

`u_rt = decoder_r^T value(h_t)`；`z_r = sum_t alpha_rt u_rt+b_r`；最终为40类硬 argmax。

因此 token Born 数不是类别后验，不是抽样类别；归一化轴是 token，不是 relation；类条件平均态也没有替代独立训练的 token utilities。把模型改为一个 joint POVM outcome 分类器会改变核心机制，原 L4 不能自动转移给它。本轮没有进行这种修改。

round8 的已有观察仅是：selected full 与 selected lambda-disabled 在1024例硬预测一致；lambda-disabled 相对从头 constrained control 有28比22例独有正确，CE更差；30个低支持类 TP/FN 不变，部分 F1 变化来自 FP。它们不提供总体后验、罕见类恢复、校准因果或 `state -> utility -> margin -> hard FP` 联合规律。

## 6. 研究决策

“为 MacroF1 使用代价敏感规则”已有直接近邻；“量子最小错误测量有 SDP 最优证书”也已有直接近邻。两者相接还缺实际决策过程和非凹目标的论证，当前不选作 L5 创新或训练候选。独立候选盲态咨询同样提出零条支持充分路线。

下一轮应离开单纯指标重加权：从关系抽取的任务定义和原始方法文献寻找独立、可证伪的 task/state/utility 联合结构，再决定是否存在对测量设计有区别性预测的问题。当前实体 roles 或 pair-marked encoder 不能自行充当语义正确性/label-preserving nuisance 的证明。新任务前提必须有独立来源，不能由当前 exposed-valid 的错误类型倒推。此前停止的 ordered/joint/complement/spectrum 家族保持停止；无阈值拟合、额外 seed/epoch 或新全量运行。

本轮仅原文获取/阅读、精确算术和文档研究；零训练、零模型/encoder forward、零调度实验、零 test-set 访问、零 COS restore/上传、零新资源或师弟服务器访问。L5 继续研究，原条件性 scoped L4 不变。
