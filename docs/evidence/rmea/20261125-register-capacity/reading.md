# 物理寄存器分配与经典可模拟边界

发现问题来自当前项目原生完整13模型/逐类结果、固定12角和40关系测量结构；既有实验积累而非文献单向套名。抽象问题为有限维测量query的packing与乘积核表达/集中风险。普通web搜索本轮仍返回404；随后只沿官方arXiv精确ID获取，非全面查新。

实际读取：A. J. Scott, *Tight informationally complete quantum measurements*, [quant-ph/0604049v6](https://arxiv.org/abs/quant-ph/0604049v6)。PDF20页/372365bytes/SHA256 0fcb26ff3253d639112a8ca41f59b81a4fe257a471231806079e02ea2e63164b，首页明确v6/11Oct2006、标题作者匹配。读取PDF1、3、4页（第3页原版图也核对），非全文；Theorem6/Eq12及Eq13–14证明给纯态分布的frame-potential界，第4页明确其为Welch界推广。

直接t=1推导：S=sum_r |q_r><q_r|，TrS=R，rank(S)≤d，所以TrS²≥R²/d；减去R个自重叠，再除R(R−1)，得平均非对角|<q_r|q_s>|²≥max(0,(R−d)/(d(R−1)))。R40/d8给0.102564。不要求query是同一联合POVM（原模型40个独立binary测量并非POVM）；也没有任何本任务分类必要性结论。query多于维数依然可以被分类，value/bias、词归一化和输入分布都不受这个数直接限制。

6个qubit乘积态用相同12角，Hilbert维64，可以用每位RY=±π/2、RZ=0的有限effect编码40个正交态。Q6内积平方等于6个单qubit fidelity相乘；D6是6个计算基重叠相乘，共同末端CNOT置换均消去。这些都是已知张量乘积/测量工具。以Bloch向量写单因子(1+n(q)·n(x))/2即可完全用经典实数计算；没有量子计算加速或不可模拟性。实数乘积/RBF12和未经改动的最强经典输入/预算对照同时保留。

相关风险来源：[Thanasilp等, *Exponential concentration in quantum kernel methods*, 2208.11060](https://arxiv.org/abs/2208.11060)。本轮只读取官方摘要；PDF响应读取超时，receipt记录未取得全文，不援引其具体定理/界。摘要指出若干条件下核值可随qubit数集中，并讨论测量资源问题；不能自动把这些条件套到本项目6qubit可训练/精确经典计算上。乘积会压低概率、EPS可使注意力更均匀、梯度有乘积风险，是本构造直接可见的失败可能。

迁移假设是将固定角预算从较深3qubit改为较宽乘积寄存器，是否改善同一自然文本任务；这是新的受控效用检验，不是由packing数推出收益。与旧3qubit幅度/相位分离、K2测量混合、跨词循环、零gate经典锚点不同，原配置和失败门保持闭合。已知frame/乘积态/核等式不算T；若效用通过，仍须完整独立L3和同核心实质近邻/非平凡理论，不能把这份已知推导包装L4-T。
