# Q-RMEA L5 Round38：共享置换的 margin 协方差与实际非恒等规则

本轮把上一轮的探索线索变成了可检查的解析问题：Q 的固定置换后 Macro 较高，是否来自特殊的类别间稳定性？得到的是一个已有统计恒等式、实际代码的分布边界，以及一个反例。目前没有 Born 专属的稳定性优势结论，也没有形成 L5。没有模型执行、训练、私有数组恢复或正式实验。

## 1. 固定模型、固定样本下的对象

令 E 为实际可置换位置：valid 且非两个实体跨度，n=|E|。所有关系分支使用同一个置换。每个关系 r 的 attention 已在全部 valid 词上归一化，其在 E 上的总质量 m_r 不必为1。令

    c_r = bias_r + sum_{t not in E} alpha_rt * u_rt
    z_r(pi) = c_r + sum_{t in E} alpha_r,pi(t) * u_rt
    a_r = alpha_r|E - mean_E(alpha_r)
    v_r = u_r|E - mean_E(u_r)

c_r、weights、utilities 均固定；这不是对原句的语义变换，也不是重新前向得到的新词表示。以下均匀分布是明确引入的参考模型，不能从固定 SHA 排序自动推得。

## 2. 均匀参考下的均值与协方差

当 n>=2、pi 在所有 n! 个置换上均匀时，

    mu_r = E[z_r(pi)] = c_r + m_r * mean_E(u_r)
    Cov(z_r(pi), z_s(pi)) = <a_r,a_s> * <v_r,v_s> / (n-1)

证明：中心化后 E[a_r,pi(t)*a_s,pi(l)] 在 t=l 时为 <a_r,a_s>/n；在 t!=l 时为 -<a_r,a_s>/(n*(n-1))。对 t,l 加权求和，并使用 sum(v_r)=sum(v_s)=0，得到上式。固定项 c 不影响协方差，但影响 margin 的均值。

因此类别对的 margin M_rs=z_r-z_s 满足

    Var(M_rs) = ||a_r tensor v_r - a_s tensor v_s||_F^2 / (n-1)

等价地，logit 协方差矩阵是 attention 行 Gram 与 utility 行 Gram 的 Hadamard 乘积再除以 n-1。每行权重的集中度、每行 utility 的波动量只给出对角项；margin 还需要跨类别项。把40类当成独立置换会丢掉这些项，定义的是另一种干预。

这不是新定理。Goldstein 与 Islak 的 [1304.5001v3](https://arxiv.org/abs/1304.5001v3) Section3.1/Eq10 已明确给出均匀置换的双中心数组方差。上式可由其 rank-one 数组情形及极化直接得到。本轮阅读了该文 pp1、4–5，未完整审核其一般集中不等式证明，也未取得 Hoeffding1951 原文。

## 3. 当前实现没有均匀地抽“非恒等置换”

正式源码 rmea.py:73–84 按 sample_id 与词位置的 SHA256 摘要排序，若排序恰等于原顺序，则把它左循环一次。这是确定性函数。即使额外假设摘要产生均匀随机排序且无并列，变换后的分布也是：identity 的概率变0，指定左循环 rho 的概率变2/n!，其他置换仍为1/n!。它不是所有非恒等置换上的均匀分布。

在该额外参考假设下，令 z_I、z_rho 分别为恒等与左循环的 logit 向量，M=E[zz^T]，则精确修正为

    mu* = mu + (z_rho-z_I)/n!
    M* = M + (z_rho*z_rho^T-z_I*z_I^T)/n!
    Cov* = M* - mu*mu*^T

n=0或1时不做非恒等修正、方差为0；n=2时两个参考排序都变成 swap，方差也为0。任何有界事件概率与均匀参考的差异最多1/n!，但这个界只适用于所述随机排序参考模型；固定 SHA 干预本身不带这个概率保证。旧报告没有逐样本 eligible 数量和 margin，本轮没有估计实际 cohort 受此修正的影响，也没有改动旧协议。

## 4. 对应到当前 Born 权重

量子模式的 rho_t=|psi_t><psi_t|、E_r=|phi_r><phi_r|，q_rt=Tr(E_r rho_t)，S_r=sum_valid(q_rt+epsilon)。在 eligible 位置中心化后，

    a_rt = Tr[E_r * (rho_t-mean_E(rho))] / S_r
    v_rt = decoder_r^T * V * (h_t-mean_E(h))

epsilon 在分子中心化中消失，但仍留在全部 valid 词的 S_r 中；value 的 affine bias 在 v_r 中消失。entity 固定贡献、分类 bias、eligible mean 仍影响 margin 均值。对应位置为正式 rmea.py:55–65、73–98 以及共享 value/decoder 定义，四份精确正式源码均重新验哈希。

dephased 也有对应的对角表示；real-overlap 是归一化实向量平方重叠，亦可进入相同固定权重统计。这里没有证明三者的 tensor-Gram 或 margin 方差存在统一大小关系，也没有把自由 weight 反例声称为原生 ansatz 可达状态。

## 5. 集中度相同，margin 可以完全不同

冻结两个类别、三个 eligible 位置，两个类别均取 alpha=(1/2,1/3,1/6)，固定贡献 c=(1/4,0)。第一类 utility 为 (1,0,-1)。

| 第二类 utility | 每行权重平方和 | 每行中心 utility 平方和 | 均匀参考 mean margin | margin<=0 的比例 | identity修正后比例 |
|---|---|---|---|---|---|
| (1,0,-1) | 两行均7/18 | 两行均2 | 1/4 | 0 | 0 |
| (-1,0,1) | 两行均7/18 | 两行均2 | 1/4 | 1/2 | 2/3 |

第一组两个 logits 的置换变化完全共同，margin 恒为1/4；第二组变化方向相反。它们具有相同的逐行集中度、utility 波动和均匀参考 mean margin，却给出不同的类别比较。这个合成反例排除了“只看逐行集中度就能决定 margin 稳定性”的一般断言；不证明实际 Q 模型正是依靠这个机制，也不直接比较两个真实训练模型。

精确 Fraction 枚举覆盖 n=0、1、2、3、4，逐一检查全部置换、协方差、tensor-margin、identity修正和错误独立行对照，无浮点容差、无结果筛选。24是最大单个 fixture 的置换数。输入、脚本、literal命令和输出见[source-verification.json](source-verification.json)，完整结果见[enumeration-result.json](enumeration-result.json)。

## 6. 原有 attention 文献已经区分局部干预与模型解释

[Jain & Wallace,1902.10186v3](https://arxiv.org/abs/1902.10186v3) Section4.2.1/Algorithm2 已使用固定 h、100次 attention permutation，并将 median output-TVD 与 max attention 对照。读了 pp1、4、7–8，p3仅部分文本；没有复现其代码或补充实验。Q-RMEA 的多关系共享 eligible 置换、CE/Macro 端点与之不同，但这些区别本身还不是实质理论创新。

[Wiegreffe & Pinter,1908.04626v2](https://arxiv.org/abs/1908.04626v2) pp1–3/Section2.1 指出，attention 与模型其余部分共同训练；单独替换 attention 不自动产生一个参数一致的替代模型。本文没有完整阅读其后续协议、结果和Section5。这个边界不否定局部干预的诊断价值，但限制了“局部置换影响输出→证明量子结构必要”的推断。

## 7. 本轮判定与下一步

共享 weight–utility 几何是理解当前干预所需的统计对象，但其协方差恒等式与局部置换诊断都有原始先例。较小方差还可能来自均匀 attention、类别共同项抵消；这些都不保证任务效用。旧25个 aggregate 模式×seed 端点没有逐样本 margins、eligible counts 或 Gram，不能从 Macro 差别回填这些中间量。

因此关闭“以这个通用方差公式本身作为 L5 理论”的路线；保留它作为以后诊断的正确工具。原有 Q-RMEA 的限定 L4 不变，本轮没有新 L4、独立实证、相位结论或已选新方法。下一步只做 normalized Born score 家族与实二次/对角控制的确切约束及近邻原文审计：寻找任务相关的构造性区别或明确等价，再判断是否有值得检验的新核心贡献。既有 L4 不自动转移，新正式实验仍由师弟执行。
