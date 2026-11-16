# Q-RMEA L5 Round39：新近 Born-attention 原文与精确实数对应

本轮找到并选读了直接相关的新预印本《Reading Why》v4，同时核对两篇 PSD 因子化原文。结论是：按行归一化的复数平方重叠已经进入直接的 attention 文献，复数评分实数化和正行缩放也有明确先例；Q-RMEA 的实质创新需要落在更具体的任务机制上。对方的有限阶精确词归因依赖特定多项式设计，不适用于当前头。L5 仍未接受，旧限定 L4 保持。

当前实际模型为：Re-TACRED 原句与有序实体条件→冻结 RoBERTa 词特征→pi*tanh(affine) 产生三比特两层 RY/RZ/CNOT 状态→40个独立二元关系 effect→有效词上 epsilon 平滑归一化→带符号64维 value/decoder 形成40类 logits；训练采用 CE/裁剪 AdamW，valid选模型，完整test的40类Macro衡量任务效用。固定头上下文权重置换 CE 检验局部 weight–value 对齐。下面仅做原文与源码/解析对应，没有执行该训练、前向、私有恢复或新的秩演示。

## 1. 直接近邻：Reading Why

[Tongloy et al., arXiv:2611.05355v4](https://arxiv.org/abs/2611.05355v4)，2026年11月12日更新，40页。本轮读 pp1、2、5–7、34、36，实际看了 p1、5、7、34 的页图；没有复现模型、e-SNLI 结果、代码、设备或光学分析，也没有完整认证其数学/资源结论。

Section2.1/Eq2 写出按行归一化的 squared complex overlap attention。Section2.2/Eq5 则先把 attention row sums、句均值分母、contrast 系数及最终强度分母固定在被解释输入上，得到关于词 amplitude 的 polynomial，再作从零起点的 integrated gradients。文中给出的 degree 上界200对应100点 Gauss–Legendre 精确积分；这是该设计与固定分母下的结论。作者明确承认经典 polynomial 模型也可有这种精确 completeness，并区分“加和精确”与“排序有用”。这些是原文陈述，不是本项目复现所得。

AppendixB/p34 对自由分母路径给出了不同的零点极限：reference slot 未随 amplitude 一起缩小，故其起点不是固定的 no-information share。本文并非遗漏了这个区别；不能把其 held-object 当作自由模型沿路径的同一函数。本项目也不据此声称其论文错误。

| 对象 | Reading Why 原文所述 | 当前 Q-RMEA |
|---|---|---|
| query/key | 每个 slot 的线性复数场 q=Qh、k=Kh，16 modes、两个 heads | 词状态由三比特 PQC 产生；40个独立关系 effect，状态均归一化 |
| 归一化 | squared overlap 除以 row sum；Eq2未把 q、k 都显式归一成单位纯态 | q属于[0,1]，每个有效词加1e-6后归一化 |
| value与主体 | 两块 transformer，quadratic measured feed-forward、残差与reference slot | 冻结上下文化特征，线性value/每类decoder汇聚 |
| readout/解释对象 | 3类 Born camera probability；held polynomial IG | 40类 logits/CE-softmax；旧L4为效用与局部对齐的限定合取 |

两者共享评分代数模式，不是整个函数相同。尤其对非零场，query norm在行归一化中抵消，但 key norm 不抵消：

    |q^H k_t|^2 / sum_j |q^H k_j|^2
      = ||k_t||^2 * |qhat^H khat_t|^2 /
        sum_j ||k_j||^2 * |qhat^H khat_j|^2。

因此线性场的 key norm 仍可充当逐词因子。纯态 kernel 与一般未归一化场不能直接划等号。反过来，架构差异本身也不证明实质创新、优先权、任务收益或量子必要性。

该文 p5、p36/ref9 把 Born attention 的先例指向 CNet/arXiv:2610.08592。本轮尚未取得并核对这篇被引原文，故这里只登记引用线索，不把二手引用当成已验证的优先权结论。下一步优先补这条原始链。

## 2. 复数平方重叠的精确实数对应是已知工具

写 phi=u+i*v、psi=x+i*y，则逐项展开即得

    |phi^H psi|^2 = (u.x+v.y)^2 + (u.y-v.x)^2。

对实数化状态 s=[x;y]、a=[u;v]、Ja=[-v;u]，评分为

    s^T [a*a^T + (Ja)*(Ja)^T] s。

单位 phi 对应的两轴正交且范数1，因此这里是 J 不变的 rank-two 实投影；complex8 对应16个实坐标，不是16个量子比特。使用同一 preparation 的实数运算，分数、epsilon 平滑、归一化和下游 logits 都可逐点一致。这只是精确经典表示，并不否定量子模块的功能或应用价值，也不建立硬件速度或参数效率结论。

[Fawzi et al.,1407.4095v1](https://arxiv.org/abs/1407.4095v1) Section2.2/Eq6 明确给出 Hermitian→real PSD 的保内积映射与维数倍增关系。[Gribling et al.,1605.00988v3](https://arxiv.org/abs/1605.00988v3) p4/Section2 也明确称该实数化为已知事实。这里没有新原理或新定理。

当前 real_overlap 对照在 real8 中计算一个平方内积，其 key 来自归一化 affine map；它不是上述“同一 nonlinear feature map＋real16两个正交轴”的精确模拟器。不能因为存在实数化就说旧对照与Q完全等价，也不能因为这个对照不同就说已证明Q具有量子必要性。

## 3. 正分母与 epsilon 不会创造新的 PSD 原理

对一个固定有效词 bag，rho_t=|psi_t><psi_t|、E_r=|phi_r><phi_r|，有

    alpha_rt = Tr[((E_r+epsilon*I)/S_r) * rho_t]
    S_r = sum_valid(Tr[E_r*rho_t]+epsilon) > 0。

这是一般 PSD 交叉因子化的一种表示。Fawzi et al. Theorem2.9(ii) 的正对角缩放不变性及其 p7 证明已经涵盖“把正行分母吸收进因子”的操作。它不把 E_r+epsilon*I 变成原生可训练纯态 effect，也不取消 unit trace、两层 ansatz 或共同 affine prepare 的限制。S_r 依赖当前 bag，不能把数学缩放因子当作跨所有句子固定的另一个 effect 参数。

Gribling et al. 的 CPSD 对象是同一组 PSD 因子的方阵对称 Gram；本项目类别×词的矩形交叉表不自动属于同一对象。本文没有读取其指数下界/Clifford构造证明，故不移植这些下界、维度优势或量子通信结论。

## 4. 为什么不能把 held-polynomial 精确归因移入当前头

给出一个仅在当前 frozen-feature 接口上的解析构造。第一层全部角度为0，第二层仅 wire0 的 RY 为 pi*tanh(x)，其他角度为0；effect 所有角度为0。prepare 的对应 affine 行只选择特征坐标 x，bias为0，全部为有限合法参数。最后 CNOT 环不改变 |000> 分量，得到

    q(x) = cos^2((pi/2)*tanh(x))。

q(0)=1，而 |x|→infinity 时 q(x)→0；q 是实轴上的有界、非恒定、real-analytic 函数，所以不是多项式。若它在任一开区间与某个多项式相同，解析恒等性会延伸到整个实轴，仍与“有界非恒定”矛盾。即使把一个 attention 分母冻结为正数，q(x)+epsilon 这一分子仍非多项式。当前 class probability 还涉及 exp-softmax，也没有该原文 quadratic-only 的结构前提。

这个构造不声称 RoBERTa 在自然句子上产生这条连续特征曲线；没有执行模型、数值积分或重新训练。它只反驳“冻结分母就必然让当前头成为有限阶多项式”这一一般迁移断言，不是新理论、实际病例解释或对原文正确性的否定。如果把完整 upstream 全部固定、只给已算好的 scalar contribution 加外部门控，当然可以解释那个新门控对象；它仍不等于当前模型的完整词级归因，不能自动继承同一实证主张。

## 5. 面向 L5 的判定

R19 已停止“高秩→实际任务必要”的自动推进，本轮在执行前读取并遵守该边界，没有新增rank39/40演示、投影参数变更或同病例拟合。固定 utilities 下，delta_alpha 的行和与对 u 的内积同时为0，就不改变对应 logit；这些自由权重变化也不必由原生 ansatz 可达。表示约束与任务贡献仍需要明确连接。

本轮新增的是一篇直接近邻及清楚的迁移边界：generic Born 归一化、realification、正行缩放、held-polynomial completeness 都不能单独承担本项目的新颖性。旧限定L4不变，五项同核心L5要求仍未全部建立，没有新方法被选择。

唯一推荐下一步是取得并核对 CNet 的 exact original/version/date/code，以实际 normalization、feature map、value、readout 合同进一步界定最近邻，再寻找任务上可观察的新机制。若仍只是标准平方重叠、PSD重写或已有归因形式，则关闭其作为核心创新的表述；不靠改名、补种子或重开停止家族推进评级。
