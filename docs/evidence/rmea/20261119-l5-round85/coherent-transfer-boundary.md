# 相干 token 擦除的迁移边界（分析草案）

状态：`ANALYTICAL_DRAFT_REVIEW_PENDING`。本节研究一个新的探索假设的必要条件，不选择正式方法，不运行模型，不继承旧 L4。非平凡新理论与任务收益均未建立。

## 问题与来源

R84 的固定检查点交叉回放中，经典路由接量子读出后 Macro 三升一降、CE 四组变差。这只说明整体头的双指标优势没有作为独立路由优势迁移；不能唯一归因于共同适配。

一个来自弱测量的探索问题是：把 token 值作为量子可观测量，并通过相干后选择读出，能否改变独立路由／读出这条因果链？弱值本身已有精确定义和异常谱外值先例：Farinholt、Ghazarians、Troupe，arXiv:1512.02113v2，第3页 §2、eq(1)–(2)。此处不把弱值、谱外值或相干叠加当作新理论。

以下先检查两个便宜的淘汰条件：保持经典 token 标记时是否只是旧头的精确重写；引入相干擦除后是否新增旧头没有依赖的跨 token 相位。它们尚未构成一个可实施的方法候选。

## A. 保留经典 token 标记，弱条件读出精确退回旧头

固定一个样本与关系 r，n 个有效 token，d=8。令 rho_t=|psi_t><psi_t|、M=|phi_r><phi_r|、p_t=Tr(M rho_t)，u_t 为原完整 value/decoder 给出的实数 utility，epsilon=10^-6。旧 logit 去 bias 后为

    mu = sum_t (p_t+epsilon) u_t / sum_t (p_t+epsilon).

在 token 寄存器上定义经典标记态、utility 算符与后选择投影

    rho_flag = sum_t |t><t| tensor (rho_t+epsilon I_d) / [n(1+d epsilon)],
    O = sum_t u_t |t><t| tensor I_d,
    P = |+_n><+_n| tensor M,
    |+_n> = sum_t |t>/sqrt(n).

rho_flag 是正的、迹一；O 为 Hermitian；P 是合法投影。广义弱条件值是 Tr(P O rho_flag)/Tr(P rho_flag)。逐 token 展开得到

    Tr(P rho_flag) = sum_t(p_t+epsilon) / [n^2(1+d epsilon)],
    Tr(P O rho_flag) = sum_t u_t(p_t+epsilon) / [n^2(1+d epsilon)].

所以该值恰为 mu，包含原来的 epsilon。n 个有效 token 保证分母正。尽管 P 在 token 基底有非对角项，输入态的经典标记消除了所有跨 token 项；O 与 rho_flag 也对易。仅给这一路径换成“弱值读出”的名称不改变原函数。

这不是说所有弱值读出都等价于旧头；等价性只覆盖上面明示的标记态、算符与投影。

## B. 相干擦除增加了受控制备的相位合同

换成相干纯态（此探索分支暂不加入 epsilon 噪声支路）

    |Psi> = sum_t |t> tensor |psi_t> / sqrt(n),
    |chi> = |+_n> tensor |phi_r>,
    c_t = <phi_r|psi_t>.

在 sum_t c_t 非零时，同一个 O 的弱值及后选择概率为

    W = <chi|O|Psi>/<chi|Psi> = sum_t u_t c_t / sum_t c_t,
    p_success = |sum_t c_t|^2/n^2.

如果使用 Re(W) 作实 logit，则有效实权重 sum_t Re(c_t/sum_s c_s)=1，但权重可为负。它不再是旧 Born 非负权重的简单读出重写。

同时，单独制备每个 token 态时无关的相位 psi_t -> exp(i theta_t) psi_t，会在上述受控相干制备中成为跨 token 相对相位，改变 W 和成功概率。因此只指定每个 rho_t 并不足以定义这条相干路径。需要额外指定受控制备电路、各分支相位参考与后选择／估计成本。

这没有否定相干模型的物理合法性。受控制备完全可以把这些相位作为模型的一部分；它说明的是新模型的数据访问与实现合同变了，旧头的函数或 L4 证据不能无条件迁移。

## C. 原三比特两层电路内的有限参数见证

以下是两个参数点和一个合法两 token 输入的解析构造，不是实际 R81 权重或自然语言样本的实验结果。取 h_1=e_0、h_2=e_1，mask 均真，角色分别为主体／客体；原 Linear prepare 可独立指定两个点的12个 raw angles，忽略其余输入列。

令所有关系 effect angles=0，所以 phi_r=|000>。先考虑 token1：第一层与第二层 q0 的 RZ 都为 +pi/2，其余角为0。token2 同样设置两个 q0 RZ=+pi/2，另把第二层 q1 的 RY 设为 pi/3，其余为0。记末尾公共 CNOT 环为 U；首层 CNOT 环作用于 |000> 不变，故

    psi_1 = -i U|000>,
    psi_2 = -i U (|0> tensor [cos(pi/6)|0>+sin(pi/6)|1>] tensor |0>).

第二个参数点仅将 token2 的上述两个 q0 RZ 变为 -pi/2。得到 psi_2'=-psi_2，而 psi_1 不变。所有角绝对值都小于 pi，可由实际 pi*tanh(raw) 在有限 raw 上精确表示：+/-pi/2 对应 +/-atanh(1/2)，pi/3 对应 atanh(1/3)。prepare 的线性列可以实现两 token 的这些指定值。整个构造也可嵌入 B=0 的 ECQP；不是其实际拟合权重。

值与解码器取一个关系 u_1=0、u_2=1，其余关系 utility/bias=0。这可由 value 的一个坐标读取 h 的 e_1 列、该关系 decoder 读取该坐标实现。

令 b=cos(pi/6)=sqrt(3)/2。两参数点的每个 token rho_t、全部40个 Born 分数、attention 和 logits 都相同；选定关系的旧 logit 为

    mu = (3/4+epsilon)/(7/4+2epsilon).

相干分支却有

    before: (c_1,c_2)=(-i,-ib),
    after:  (c_1,c_2')=(-i,+ib),
    W_before=b/(1+b)=2 sqrt(3)-3,
    W_after=-b/(1-b)=-3-2 sqrt(3),
    p_before=(7+4 sqrt(3))/16,
    p_after=(7-4 sqrt(3))/16.

两个分母和成功概率都严格正；异常值不来自除以零。后者小于0，超出 utility 谱[0,1]，但同时改变了后选择资源。此处不声称无成本放大、稳定性、学习收益、量子必要性或新的弱值定理。

## 淘汰条件与尚存问题

1. 保留 rho_flag 的方案以 A 为精确等价性淘汰检查，不能仅靠命名改变作为新贡献。
2. 声称“仍只使用原 token projectors、不新增相位合同”的相干方案被 C 反例否定。
3. 显式固定受控制备与相位的相干方案仍属探索问题。需要解释该相位如何由任务学习而不是产生任意分支选择；明确成功概率与估计成本；与精确复数经典对照、非负 Born 池化和同预算非线性读出比较。
4. 上述反例是结构判别：旧头预言该 phase flip 下输出严格不变，明示相干路径预言输出改变。它没有利用 R84 的972个输出生成或确认任何新的效用预测。
5. 这不是已停止的实体 partial-swap/commutator 融合；但相干机制的新颖性、收益和必要假设均未通过。当前不选定方法、不实现、不训练、不发正式任务。L5 NOT_ACCEPTED。
