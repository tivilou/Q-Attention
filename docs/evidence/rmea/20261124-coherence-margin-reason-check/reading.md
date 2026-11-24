# 训练约束近邻：Right for the Right Reasons

Ross、Hughes、Doshi-Velez，Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations，arXiv:1703.03717v1。官方PDF题名/作者核对；本次只读PDF pp.1–4，重点§2.1/2.2，不是全文深审或系统新颖性通过。PDF SHA1e6f3e05f25fbdc88f30795476ea075f38e17b1b41aa4ad20507bc7f8699cada，14页，1,940,774 bytes。

原文：https://arxiv.org/pdf/1703.03717v1

该文在监督分类损失中，按外部给出的无关输入mask惩罚输入梯度；没有标注时，另行寻找依赖不同特征的模型供审查。故“限制模型因错误理由答题”与梯度正则是已知思路，不能因套在相干项上就当新理论。

当前任务没有证明哪些相干项是语义无关的，也没有输入层的无关词mask。训练标签不自动提供这类解释监督；不能把去相干参考称为天然正确的语义教师。另一方面，RRR是输入梯度约束，不等于本次同权重去相干margin增量，这个差别仍不证明新颖性或有效性。

原生读出的小反例已发现：允许参考分支共同训练时，Q仍错、D变得更差，即可把相干增量惩罚降至0。该朴素方案在模型实现前被拒绝；stop-gradient只阻止一次反传的直接方向，不能自动建立整个共享参数训练轨迹的不作弊保证。
