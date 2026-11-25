# Q-RMEA：Pauli/Lego原文与直接迁移资格审计

本轮沿原Q-RMEA主线推进L4-T，仍在“失败后寻找任务机制”阶段，未启动新训练。gen98五臂十三门STOP、gen99–100及全部已关闭profile保持。上一轮排除了把自由学习的12个角度重排包装成时间结构。本轮核读2311.13552v1，新增的是局部可观测量选择的真实定义、理论适用条件，以及它与非负注意力质量的差别。

官方v1题名、三作者、ID和PDF SHA c9612638已核，27页解析；范围读1–11、13–15、20–27共22页，视觉1/6/7/20。作者代码、完整引用证明及数值实验未复现。源把可观测量期望乘积作为Lego，用非负权重组合成PSD核；Pauli H-body恰好只含H体项，S子系统核则带identity和所有低体项。源的实验是8qubit、PCA Fashion-MNIST二元SVM，主要比较局部核与全局量子核；不是本项目的NLP、40类输出或强经典对照证据。

**已核事实一：PSD并不等于非负路由。** 两个原生合法角度端口可得|+++>与|--->，一体Pauli核分别为1和−1。直接代入native EPS归一化会产生负α；合成utilities=(1,−1)时理想输出10^6，而原生Q同端口约0.999998。该例针对“直接替换注意力概率”的操作，不是训练失败或文本案例，也不反驳源的合法SVM核。正的子系统HS均值保留了常数和低体项；选择它、补常数或再缩放均属新的定义，不能沿用原生分母或已知源界而自动晋级。

**已核事实二：一般Mercer基不等于一个物理Pauli基变换。** 源p7用同一U0联系Mercer算符与Pauli。上述正交状态在不等权两点分布下有唯一rank1正本征算符；规范化三qubit Pauli全部rank8，共轭保持rank，所以这种映射不具有一般保证。对应Q/D积分算符交换子范数平方精确为1/1296，也展示一般Pauli重加权可以改变本征函数。这个范围结论不否定抽象Lego PSD构造或在真正Mercer基中的谱缩放。

源的margin复杂度界还依赖固定base kernels、读出范数约束和二元样本单位。原生联合学习prepare/effect、signed value和token分母，并不直接满足这套论证；p数小不能自动推出40类Macro提升。源S核的平方根求和尺度和GTQK权重l2约束也需明确处理重叠项：n3,S1/S2的源权重范数平方为3/2。本轮保留公式边界，未把这些已知线性代数/核工具计为新T。

事前固定24随机key+8完整X基key、40query、全部H=0..3/S=1..3，1280交叉端口均保留。原生Pauli全局Q/D回收与S子系统均值误差不超过3.33e−16；主审计约0.016秒。独立NumPy稠密门、按bit计算Pauli期望、显式索引partialtrace及Fraction理想反例约0.281秒，最大误差1.78e−15，精确负路由输出10^6、交换子平方1/1296；字面命令、输出、exit0和hash已核。它是独立实现复核，不是独立任务确认或外部科学内容评审。

本轮任务行/任务feature/checkpoint读取、完整headforward、optimizer更新、GPU/申请、serverwrites、COS及模型API均0；另有72个合成原生circuit状态和独立72个稠密门状态。原生源副本未改，PDF/渲染仅本地。没有实际文本case或sample-trace，不从这些合成端口编造语义机制。

决定：直接Pauli/Lego迁移为NONE，未选局部核或新plugin。旧uniform-shadow估计器失败与改变kernel不是同一件事，不能据此否定全部局部核；本轮停止的是没有任务对应及正确保证的直接移植。原L4、全部正负历史和量子目标保持；新L3/L4-T仍未接受，goal ACTIVE。

下一步返回任务相关机制发现，用现有官方采集器执行两条事前固定top6查询：量子attention与关系抽取/上下文，以及量子kernel与feature learning/trainable embedding。先与旧role/order/sharedquery/query-alignment/frame/ansatz/bandwidth/VAT和已读原文排重；优先实际NLP结构或联合学习机制。只有原文、实际对照和关系抽取的Q/D/强经典可区分预测齐备后才设计。若没有新候选，记录NONE，不调query、补seed或重复泛化咨询来强凑结果。

原文：[2311.13552v1](https://arxiv.org/abs/2311.13552v1)。本轮没有新增任务效用、硬件优势或非平凡理论贡献。

[独立稠密门与有理反例核验](../evidence/rmea/20261126-trace-kernel-audit/verification.json)；[方程对应及边界](../evidence/rmea/20261126-trace-kernel-audit/derivation.md)。
