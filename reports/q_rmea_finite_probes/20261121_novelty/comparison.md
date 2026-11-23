# 有限 value 探针：近邻核查与满秩 decoder 下的最少次数

## 当前判断

当前 checkpoint 的端口、九维子式和17组探针通过，是正向的接口可观测性结果。它没有建立自然文本任务上的新收益。本文只作有界原文比较和数学推导，不接受 L5，不改历史 L4，不启动新训练。

## 原文比较

| 原文 | 已读的精确对象 | 与当前命题的关系 | 仍有区别 |
|---|---|---|---|
| Brunner et al., On Identifiability in Transformers, arXiv:1908.04211v4, Section3 | 固定 value 下 attention 到 head 输出的线性映射及其零空间；归一化约束；effective attention | 高秩 attention 不等于任务输出必要性，早有明确框架 | 本项目另加固定 routing 的独立 value 干预，以及 relation-specific decoder 和一个共享 softmax |
| Heinosaari et al., Quantum Tomography under Prior Information, arXiv:1109.5478v3, pp3-4, Proposition2 | 受限状态集合的可识别性等价于差集合与观测核仅交于零 | 有限干预识别是受限线性观测的一例；不能把核判据本身当新理论 | 我们的参数是归一化 routing，而不是密度算子；decoder 和 softmax 共同平移需具体处理 |
| Fawzi et al., Positive semidefinite rank, arXiv:1407.4095v1, pp2-4 | PSD 因子分解、非负因子分解及秩下界 | Born score 的 PSD 分解与经典非负分解对比属于已有工具 | 本项目的实际电路/端口证书是实例化证据，不自动产生新一般定理；实/复秩界不可混用 |
| Konda, Identifiability and Observability of Deep Normalized Attention, arXiv:2610.09620v1, Sections2/4/5 and AppendixC.9 | 深层单头无 mask 归一化 dot-product attention 的参数函数可识别性、有限 jet、有限普通输入 bank、接近 collapse 的敏感性 | 有限探针、函数可识别性和数值可观测性已是明确研究问题；C.9 也用有限维函数空间与逐步消核构造 bank | 它恢复未知 Q/K/V 的有效参数，输入同时影响路由和值；我们固定 routing 并自由操控 value，decoder 已知。因此它没有直接证明我们的计数，我们也没有解决它的高效原始输入 bank 设计问题 |

这是有界查重，不是穷尽式首创证明。Konda 是2026-10-07的 arXiv v1 预印本，不声称已同行评审。所有具体命题以所读页为范围，不把全文下载等同全文审查。

## 对已有16次上界的补充：满行秩时的精确计数

设 R>=2、T>=2，q=T-1。每个关系的 attention 行 alpha_r 位于 T 维概率单纯形内部；各行独立。路由在所有干预中固定，且不能读被干预的 value。固定已知 decoder D∈R^(R×d)，共同 bias 固定；各 token 的 value x_t∈R^d 可自由且独立指定。只有一个 R 类 softmax。本文比较任意两组 alpha/beta 在同一组预定探针上的概率，或从已知 bias 的概率观测恢复 alpha。

若 rank(D)=R，则识别全部 alpha 所需的最少非基线探针数为

    K* = ceil(R(T-1)/(R-1)) = q + ceil(q/(R-1)).

这里的最少是：在完整单纯形内部识别任意固定 routing、预先选定 value 赋值、精确无噪输出的情形。不是为排除某一个低秩对照所需的最少次数，也不是噪声下的统计样本复杂度。若两模型的未知 bias 不同，则可另加一次全零 value 基线抵消差值；这个额外基线不计入以上数字。

### 下界

alpha 有 Rq 个自由度。每次探针的 anchored log-odds 为 logits 相对第一个关系的 R-1 个差，是 alpha 的仿射线性观测。K(R-1)<Rq 时线性观测有非零核。从均匀 attention 沿核作足够小的非零扰动，可保持全部行严格为正、行和为1，并使所有 log-odds 和概率完全相同。因此预定探针少于 K* 不足以识别全部 alpha。

### 达到下界的构造

令 s=ceil(q/(R-1))，K=q+s，故 Rs>=K。对每个关系 r=0,...,R-1，设

    E_r = { (rs+j) mod K : j=0,...,s-1 }.

每个 E_r 有 s 个不同位置，所有 E_r 的并集覆盖 {0,...,K-1}。令 U_r∈R^(K×q) 由不在 E_r 中的 q 个标准坐标列组成。

对第 j 次探针，第 t<T 个 token，要求 relation utilities 满足

    (D x_jt)_r = (U_r)_(j,t),

最后一个 token 的 value 为0。rank(D)=R 保证每个上述方程可解；各 token 独立可控保证能同时实现。

令 Delta=beta-alpha，删去最后一列后视每个 Delta_r∈R^q。所有探针概率匹配，当且仅当存在共同 c∈R^K 使

    U_r Delta_r = c    对全部 r。

由于 U_r 在 E_r 的行恒为0，c 在每个 E_r 上为0。覆盖性给出 c=0；U_r 满列秩再给出每个 Delta_r=0。最后一列由行和确定，故 beta=alpha。证毕。

R=40、T=9 时，这个条件计数是9次，原二维 decoder 假设下的最坏情形上界是16次。两者假设不同，没有推翻旧的16次最坏情形锐性。原 checkpoint 只验证了二维可观测 decoder 条件，没有验证 D 的满行秩；因此9次只是条件推导，不是已在 checkpoint 执行的结论。本轮不为此追加 GPU 或模型探针实验。

## 贡献判断与关闭条件

原17组 checkpoint 探针（零基线+16）结果保留，现有条件定理保留。新增满行秩精确次数是一个小型具体引理，证明依靠维数计数和覆盖子空间，不能仅凭推导正确、计数更小或外部模型赞同便接受其首创性、顶会创新性或 L5。

当前有限探针路线作为独立 L5 主创新暂不提名；保留为方法解释/可观测性工具。此判断不是说某篇原文已经逐字给出新公式，也不是说 quantum-inspired 机制毫无价值。当前缺口是：在实际可达输入和真实任务目标上，一个与已有方法有实质区别的同一机制具有额外作用。既有 synthetic rank9 并不能填这个缺口。

## 下一步的单一研究问题

回到已有实际训练证据：完整 classical routing+readout 较强，而替换固定 Q readout 的 routing 效果并不一致。先从现有代码、报告和已有 checkpoint 的理论表达中，分离一个此前负向路线未检验的“路由与 readout 配合”机制；要求一个明确干预能区分它与普通共同适配。没有具体机制和相应预测前，不追加种子、重复全量训练或扩展 rank/探针审计。该问题尚非新方法，不继承 L4，也不承诺会产生 L5。


## 原文与数学咨询

- [Brunner et al. 原文 v4](https://arxiv.org/pdf/1908.04211v4)，Section3。
- [Heinosaari et al. 原文 v3](https://arxiv.org/pdf/1109.5478v3)，Proposition2。
- [Fawzi et al. 原文 v1](https://arxiv.org/pdf/1407.4095v1)，Definitions2.2/2.4、Proposition2.5。
- [Konda 原文 v1](https://arxiv.org/pdf/2610.09620v1)，Sections2/4/5、AppendixC.9。

A6API gpt-6.2-astra，high，数学与文献作用域咨询：PASS_WITH_SCOPE，must_fix为空。独立核查认可维数下界和循环遗漏集合构造，确认9次与16次的假设区别，并认可当前仅作为工具的贡献判断。该咨询不是穷尽式文献查新。若两模型存在不同的未知类别偏置，最坏情形总观测数为K*+1；零value基线给出一种实现。实际checkpoint尚未核查满行秩D，仍只声称原17组执行结果。机器可读记录见review.json、reading-receipt.json。
