# 关系查询框架测量：小筛选候选，尚未实现

[已测事实] gen94/95中Q在全部正关系排序上的错误多于D/G；它是本候选的任务动机，不证明查询冗余是原因，也不给出预期提升。现有slot-style竞争与上下文word-frame已经审查/测试；不重开旧配置。题材来自已知平方根测量，不能计为新T。

## 本次实质差异

对40个原生查询定义F_r=phi_r phi_r†+EPS I，S=sum_r F_r，E_r=S^(-1/2) F_r S^(-1/2)。对未经滤波的原词态psi_t读p_rt=psi_t† E_r psi_t；按原规则在词维归一化，用原signed value、decoder和bias分类。sum_r p_rt=1是代数性质，不是句子标签的校准概率。

这不同于旧竞争q_rt/sum_s q_st：后者等价于同E作用于S^(1/2)psi归一化后的滤波词态。本次不滤波。也不同于按句子key构造的S_word框架；本次S只依赖40个关系查询，跨句共享。algebra_check.py已复算显式二维反例，差值>0.12；它不是实际三比特原电路可达性或任务效用证明。

若S是标量单位阵，则按词行归一化后退化原读出。EPS=0时，每个效应仍rank1，可写成变换后的查询；所以不是对一般自由query的表达力分离。实际有限线路约束、全查询耦合与优化偏置才是待检验变量，不能继承PGM集合判别的近优保证来承诺关系分类收益。

## 拟冻结的有限比较

复用gen94—98相同TRAIN8192fit/4096assessment、40类、sentence-group无重叠、seed13、b128/micro4、FP32/TF32off、原AdamW。三新臂各1024更新：Q relation-frame、对所有F/S一致取对角的dephased relation-frame、12维实数归一化向量relation-frame。保留各输入、value初始化与批次顺序；参数数目/有效容量差异必须实际报告，不能仅同参宣称公平。

强Gaussian baseline及Gaussian covariance、native Q/D四旧参照只能复用同协议已封存终态；不重跑、不删除更强旧控制。Q须Macro严格胜两个新控制和四旧参照，共六门，才设计完整单seed；任何失败暂缓确切EPS/线路/框架/初始化/1024配置，不扫正则化、秩、query scale、seed、budget或对照救援。不接受L3/T或自动全量。三臂先全部训练结束再统一最终评估，无新VALID/TEST。

三fit共3072更新；每fit最多600训练秒、总1800训练秒，不购容量；等待/实际占用另记。仅经共享调度获得新allocation后执行；原gen98不重启。当前没有allocation、实现或运行。

实现前需要固定源码/data/order/reference SHA及实际参数账，使用已有InvSqrt导数与稳定screen/host；仅验新E公式、近简并谱导数、梯度穿过全部query、S标量退化、对角定义、padding与clone actual8。一次精确变更审核后才进入有界小筛选，不重做生命周期矩阵。若已有历史同函数研究被发现，先停下排重。若技术审核失败，保留首次记录，修正后精确复审，不改科学门。

## 近邻阅读与边界

复用并核SHA的Barnum/Knill quant-ph/0004088v2原文p10 Section V/Eq37：平方根测量是已知原语；集合平均判别保证不自动迁移到文本效用。只读该范围，非全文阅读。source: https://arxiv.org/pdf/quant-ph/0004088v2 。本轮未建立文献穷尽性、理论创新或硬件收益。现有fair controls失败仍保留真正支持的普通效用；经典衍生独立评级，不替换量子L4-T目标。

数值说明：训练参数及激活接口为FP32；全部frame代数先提升振幅到FP64/complex128，包括对角平方概率；最终p再转回FP32。内部精度与额外计算如实报告，不声称与原生FP32等算力。
