# 原生测量的共享词先验：固定 TRAIN512 筛选

动机是gen70已完成七路的RBF12 Macro .353977高于原生Q .172509；Q6 .322467又低于去相干 .344355。不能归因为相干/维度或文本原因。新问题：原生归一化测量默认各词先验均匀，输入可计算的共享词先验是否有独立任务效用？RBF分解为 key·query/12−||key||²/24−||query||²/24，最后一项在词softmax消去；需要另设同温度、无energy的RBF对照来区分energy与温度。

本轮只改变词先验，保留原3qubit两层prepare/tanh/circuit/effect、原概率、value/decoder/bias和81028参数。令a_t为prepare([h,roles])的12维tanh前输出，e_t=mean(a_t²)/2。Q的词log权重为log(p_rt+1e−6)−e_t，masked softmax；D同原生去相干概率。Real12先验同样使用其key的12维线性tanh前范数，log(cos²+eps)−e；softmax12先验为key·query/sqrt(12)−e。RBF-noenergy为key·query/12，除此以外保持原RBF初始化及共享value路径。没有可调lambda、阈值、门、额外参数或warmup。

可以用成功概率w_t=exp(−e_t)的辅助位衰减和独立二元效应(P_r+eps I)/(1+eps)表示Q路径，条件词概率即w_t(p_rt+eps)/sum。每个关系是单独二元测量，不是40关系共同POVM。此表示为已知后选择/核加权代数，不是T，不声称大范数词是无关词或不可经典模拟。数值实现使用log域以免exp下溢。与旧的关系级normalizer-mass、signed-null、跨词相干/读出以及Q6角重分配不同，旧配置全部保持关闭。

同批拟合九路：quantum_prior/dephased_prior/real12_prior/softmax12_prior/rbf12_no_energy，以及原quantum_native/real12/softmax12/rbf12。四个短对照沿用gen70完全相同实现和初始化，重新执行固定512更新；不是重跑历史全量baseline。本轮舍弃新增的跨运行复用接口，避免为几分钟短试验扩展数据与信任边界。原gen70结果保持不动。全部九路结束后仅final512一次比较，无独立L3含义。

复用screen v6不改训练/保存/双GPU队列。seed13、原8192整句分组4192fit/4000assessment、all40、FP32、b128、AdamW、每路512更新，总4608，单路600秒/全局1800秒。每路仅final512；不读官方VALID/TEST，不根据中间结果选状态。Qprior主指标all40 Macro须严格胜过其余8路；CE/全部40类/修复与损害/时间完整报告为次指标。八门全过才进入新完整协议准备，不自动接受L3/T。任何失败关闭本确切energy定义/系数/归一化/预算配置；不改norm/lambda/温度/eps/初始化/LR/seed/split/预算/metric或控制救援，不宣称家族不可能。

只检验改变部分：先验条件概率及梯度/零energy原生归一化、RBF分解/无energy温度、mask与参数schema、first8真实克隆一步。同批共享原有输入、分组、顺序和value初始化；不重复核心生命周期/COS能力矩阵。新短状态和逐例数组只在授权GPU，Windows/Git只有代码与聚合；本轮无新COS。通过后全量仍须预声明独立评估与最小COS续训，失败不保存新checkpoint到COS。

近邻已核对Tsai等 arXiv:1908.11775v4 (2019), Transformer Dissection，PDF第1–3页/2.2节Definition1明确核归一化注意力。共享先验乘核与RBF分解本身是已知工具，不能充当新理论。此轮只检验可行性，未完成查新或理论创新审核。

实现修正保留：拒绝python -O；Q/D沿用原生utility-first的FP32读出顺序，fixture独立重算先验/likelihood及梯度，并在克隆模型上中和先验后实际调用生产forward核对最终logits。v1-v3的跨运行复用提案及审查保留为撤回历史，不进入启动代码。固定同批全部九路数据来源和训练/保存/COS行为复用既有稳定核心；无新增CLI或跨运行复用承诺。
