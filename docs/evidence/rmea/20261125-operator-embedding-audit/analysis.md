# 算符值核嵌入审计：精确对应不等于新机制，集中界须保留范围因子

本轮核读 [2605.25146v1](https://arxiv.org/abs/2605.25146v1)《Kernel Embedding for Operator-Valued Measures and Its Application to Quantum Tomography》。得到原生token注意力的精确条件POVM表示、40效应的全混合态识别边界，并发现其印刷版Theorem4.25的一个有限反例。三者均不建立Q-RMEA的新效用、非平凡新T或L4-T；原量子目标继续，gen98–100及原限定L4保持。

## 实际做了什么

官方Atom/PDF题名、Philipp N. Mayer/Ho Yun与v1版本栏一致；PDF SHA `9f789d5cf2ed631e86ed12682597a3c6ea9694060d79698b32b6d0828b321949`，46页均可提取文本。实际范围阅读pp1–9、11–16、18–21、27、35、45共22页，视觉核p1/7/21。读取了嵌入定义、测量设计及统计条件、相关公式与集中界证明；没有完整审核全部无限维论证、minimax证明或作者代码，不称全篇定理认证。

原文将给定测量装置的POVM嵌入核空间，并用同一未知态的独立重复测量做tomography（pp4–8,11–16）。原生Q-RMEA直接模拟可训词态/效应，沿词归一化后读带符号utility；这不是该统计采样问题。以下是针对精确对应和具体公式的本项目核验。

## 1. 保留原分数的条件POVM对应

固定一个输入，定义有效词的 `B_t=ρ_t+εI`、`C=Σ_t B_t>0`。令

`F_t=C^(-1/2) B_t C^(-1/2)`，

`χ_r=C^(1/2)φ_r / sqrt(φ_r†Cφ_r)`。

则 `F_t≥0, Σ_t F_t=I`，并且

`χ_r†F_tχ_r = φ_r†B_tφ_r/(φ_r†Cφ_r) = α_rt`。

在有限token集合采用Dirac kernel，源Definition3.3的条件嵌入就是 `diag(α_r)`。signed utility是该分布上函数 `u_rt` 的期望：

`z_r-b_r=χ_r†(Σ_t u_rt F_t)χ_r`。

因此signed A_r本身虽不是正测度，仍可作为POVM的带符号矩；无需丢失key/value配对。这里是坐标表示恒等式，未改变实际模型。只改F而不补偿χ才会改变函数，这正是不能借本源重开旧word-frame/whitening配置的原因。输入和训练参数改变时F及χ都变，不能继承“固定装置+未知态”的统计保证。

本次以原生3qubit/2层circuit源 `9f4bfc57…` 的40个合法effect角度、10个key角度检查。主实现用谱平方根；另一实现用完整8×8 RY/RZ+CNOT矩阵及Cholesky gauge，独立重建条件概率和signed读出，最大误差2.78e-16、4.44e-16。没有任务h或训练head，合成utility固定为sin((r+1)(t+1))。

## 2. 40个二元效应不是全混合态tomography设计

这与上一节的token POVM是两个不同方向：此处将40个关系效应当测量设置，作用于任意8维key密度。

8×8 Hermitian trace-one矩阵有63个实自由度；每个binary setting提供至多一个独立实数。原生40个setting至多约束40维，因此至少23个traceless方向不可见。增加同设置的shots不会补足这些方向。原生raw40个rank-one效应也不构成一个联合40结果POVM（其trace和为40而I的trace为8）。

在固定合成合法角度中测量矩阵恰为rank40。构造Hermitian H使tr(H)=0、tr(E_r H)=0、||H||op=1，令 `ρ±=I/8±H/16`。两者均为合法密度，最小特征值.0715482/.0625，Frobenius距离.204299；40个测量概率最大差2.78e-17。独立全矩阵重建差4.61e-17，basis-free Gram也确认rank40。

此结论只排除对任意混合态的全识别。它不证明原生纯态族有碰撞，不要求分类一定先做tomography，也不解释真实错误或证明增加测量会改善任务。不能以此扫query数、换基、设计正则或重开已关闭shared-measurement方案。

## 3. 原文Theorem4.25字面范围的反例

原文p21对任意Hermitian方向S给出方差项 `v=M/r` 的上界 `exp(-v h(t/v))`，其中 `h(x)=(1+x)log(1+x)-x`。p45证明使用了以1为尺度的mgf界，却没有约束对应独立增量≤1。以下例子满足该式所引用的complete-design和正定kernel条件：

- q=2；三个Pauli X/Y/Z测量；每设置r=1；有限谱{−1,+1}上的Dirac kernel。
- 真实态 `ρ=diag(999/1000,1/1000)`；方向 `S=−10σ_z`；t=10。
- 该QUARK估计等于通常未投影Pauli线性估计 `ρhat=(I+Xhat σx+Yhat σy+Zhat σz)/2`。原Theorem4.20采用的正是未投影估计。

随机方向误差 `tr((ρhat−ρ)S)` 以1/1000概率为19.98，否则为−.02。因此精确tail为 **.001**，均值0、方差 **.3996**。代入该印刷式却得 **4.201873323×10^-11**，与tail矛盾。

主实现用有理数枚举8种Pauli结果；独立实现直接解6×4测量矩阵的最小二乘，并从源中的H_K/A_K/S_K算出M=.3996000000000112，tail=.001，重现矛盾。没有Monte Carlo误差、参数搜索或任务样本。

保留独立中心增量上界b时，通常Bennett形式为 `exp(-(v/b²) h(bt/v))`。本例b=19.98得到.07301161116，与精确tail相容。这个已知范围因子不是本项目新T。这里只否定v1的该任意方向印刷界，不评价其他结论、后续版本或整个研究；没有联系作者或发送外部消息。

## 与历史及主线的关系

原native C/A分式与自由query端点已在routing-headroom审计；此次POVM对应不把旧代数重命名。旧word-frame、gen99 relation-frame、state/attention Fisher及shared-Pauli测量均有独立停止记录。Theorem4.25的纠错也不构成任务性能机制，更不解释gen98失败。

本轮0任务行/训练headforward/updates/GPU/COS/API、原生源未改。只有本地生成的角度、矩阵与8种合成测量结果，不是训练参数、任务预测或硬件实验。独立数值重算是另一实现，不冒称外部科学评审。

直接把此源变成新Q机制或任务保证的路线停止。下一步复用已发现的 `2403.10859v1`《Neural-Kernel Conditional Mean Embeddings》，只核其学习目标/样本单位/一致性条件，判断是否能作用于原生sentence-level带符号配对读出；不得把sentence标签当token真值，也不把有限label kernel下可能等价的普通损失称作量子理论。仍需实质机制、适用命题和Q/D/G判别预测，才进入有限设计；不承诺新fit或以经典衍生替换目标。

证据：`source-receipt.json`、`reading-record.json`、`scope.md`、`audit.py`、`audit-results.json`、`execution.json`、`verify.py`、`verification.json`、`history-comparison.json`、`next-direction.json`。原PDF和渲染仅留本地复核，不发布副本。
