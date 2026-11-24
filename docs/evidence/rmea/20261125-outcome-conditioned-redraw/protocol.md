# 测量结果条件化的保留／重抽：固定探索

任务假设：词选择与词的分类效用显式耦合，可能比只按相似度选词更有用。既有上下文交换均损害接收者，与共同适应相容，但不证明具体语义根因。这里测试新的前向函数，不是原光学或 MEG 配置调参。

对每关系 r，保持原三qubit两层词态 ψ_t、关系线路 U_r、value/decoder/bias。计算八个测量概率 p_tk=|<k|U_r†ψ_t>|²，原提议 a_t=(p_t0+EPS)/sum(p_j0+EPS)，u_t=decoder_r·value(h_t)，μ=sum a_t u_t。c_k=sum a_t p_tk(u_t−μ)。测量结果 k 后，当 c_k>0 时保留提议词，否则从 a 重新抽一次并无条件接受。不使用标签，执行确定性期望：z_r=b_r+μ+sum ReLU(c_k)。等价最终权重 a_t(1+m_t−sum a_j m_j)，m_t=sum p_tk 1[c_k>0]。非负且和为1。掩码外词不参与；空序列拒绝，单词/常数效用恒等，无新可训练参数。

这是经典“观察信号后作二元决定”的已知最优性应用，不是新T认证。固定状态时增益 G=sum ReLU(c)=.5 sum|c|，不大于知道确切词效用的 G*=sum a_t ReLU(u_t−μ)。每一类都提高自己的logit可能损害分类；实际CE变化为log sum P_old(r) exp(G_r)−G_y，必须报告反例。两固定词类型按不同比例出现时，G=s(1−s)|u_A−u_B|TV(p_A,p_B)；在原提议每份质量w≠v且非退化时，输出差有二阶极点，独立重参数的两个逐词归一化一元行之差只有一阶极点。这只区分固定输入tuple的一元预测族，不分离耦合经典、实际RoBERTa上下文重编码或证明风险收益。理论优先性与独立内容审查尚未建立。

四个新训练条件：quantum_record（候选）；dephased_record（ψ及U在计算基取概率，p_tk=sum_d |U_dk|²|ψ_td|²，其他规则相同）；utility_oracle（同原量子提议，但能直接决定保留u_t>μ，z=b+μ+G*；名称仅指模型内部效用，不读取gold）；classical_record（原real12提议，八结果由每关系仿射[key12,u_t−μ]的softmax给出，额外4480参数，显式披露）。候选与全部对照均独立训练。量子及exact-utility无额外参数，经典多参数不声称等有效容量或硬件优势。

复用已接受screen v6/core/hostwrapper/输入、原TRAIN整句4192fit/4000assessment、seed13/b128/FP32 AdamW .001/.01/betas .9,.999/eps1e-8/clip1，每新路512更新、final512一次全部4000/all40评估；所有fit完成后才assessment。原生quantum/real12/softmax12只复用heralded-token-prior-20261125已完成的紧凑结果和原配置：相同core、数据、split、init factory、order及训练协议，固定Macro .17250940811644527/.2119890069947711/.20120655247669292。它们不是新训练条件，不加载旧checkpoint/逐例数组、不引入跨运行resume或CLI路径读取。冻结原summary与design哈希和三行CM；summarize须检查新order及已知参考一致。若复用可比性不成立则不判门，不静默补跑或修补历史。

固定门：quantum_record全40Macro严格超过三个新对照及三个复用原生锚点，六项全部通过；CE仅描述。失败关闭此确切八结果/一次重抽/512配置，不改结果数、抽取次数、seed、预算、metric或控制救援；短负结果不代表家族不可能。通过仅允许新完整协议准备，不是L3/L4-T。全量另冻结完整预算和独立确认，既有完整负结果不抹除。

只检查新公式与同算式独立枚举、mask/singleton/常数、实际原提议、NumPy完整U与有限梯度/一次真实8例克隆更新。不重测训练/恢复/COS生命周期。两GPU共用既有core队列、每新路600秒/总2400秒，可重启廉价screen只本地latest/best/final，无COS；Windows只收集紧凑汇总。受影响源码一次Astra-high审查；记录固有反例与近邻不明，不在pilot阶段做新T/评级审计。

与R93已知utility-MEG的区别：R93更新测量效应exp(log E0+A/N)，本方法固定关系测量基，只优化测量结果上的一次保留/重抽策略；没有矩阵指数、MEG步长或熵近端。与旧DPP、全局词叠加及weak-value读出在公式上均不同。通用utility-aware思想已有，不作首次提出的声明。
