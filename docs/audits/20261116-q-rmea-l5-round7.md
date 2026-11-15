# Q-RMEA L5 第七轮：排除相位训练对照的门参数化混杂

**L5未接受，原限定L4不变。**本轮只训练一个新对照，判定`RETAIN_EXPLORATORY_TRAINING_COMPONENT_SIGNAL`。旧四组结果没有重跑，没有正式全量实验或师弟服务器访问。

## 为什么补这个对照

第六轮ordinary mixture使用kappa=tanh(g)，fusion使用(kappa,lambda)=(g1,g2)/sqrt(1+g1²+g2²)。因而旧fusion比mixture略高，既不能归因phase，也不能排除参数化/优化差异。旧所选fusion把lambda置0后Macro/accuracy不变、CE略好，此不利结果保留；相同汇总指标不等于所有预测或logits相同。

这次使用原FusionHead(fusion)，仅在**训练全程**设置phase_off=True。保留原kappa映射、原初始化/同seed307、相同batch与AdamW设置、4epochs、2361train/已暴露1024valid。g2从0出发，梯度及更新后值每步均须严格0。80628个存储参数中40坐标仅沿本次零约束训练轨迹不活动；全参数域里g2仍在kappa分母，不声称与完整fusion等表达能力。旧fusion及输入身份直接复用Git中固定报告，不重复其训练。

## 冻结方向与结果

两组各自strict最大valid full40 Macro选模、同分最早epoch；主方向事前固定为旧fusion Macro−新同门对照Macro>0。CE/accuracy为描述性结果，保留全部epoch。即使主方向为正也只是后续独立测试的线索。

| 条件 | 所选epoch | valid Macro | accuracy | CE |
|---|---:|---:|---:|---:|
| 旧完整fusion（复用） | 4 | 0.170059088175 | 0.679687500000 | 1.240799655694 |
| 同门映射、训练全程关闭phase | 4 | 0.163012314633 | 0.673828125000 | 1.231495794490 |
| 旧完整fusion所选模型仅评估时关闭phase（复用） | 4 | 0.170059088175 | 0.679687500000 | 1.240795537266 |

主Macro差=`+0.007046773542`；control CE−fusion CE=`-0.009303861205`。One exposed-development training-component signal retained; not independent confirmation or proof of phase necessity.。这一比较改变了预定训练通道、相应梯度/优化轨迹和选模结果；不能区分所有容量、正则与随机训练解释，也不是独立确认或量子必要性检验。[逐epoch](../evidence/rmea/20261116-l5-round7/epoch-metrics.json)。

## 原文与理论边界

[CNM，NAACL2019](https://aclanthology.org/N19-1420/)已使用复数词态、局部projector混合及可训练测量方向。本轮只读PDF pp1–6（印刷4139–4144），查看p3图像。它对局部窗口的测量概率做max pooling再匹配问答，不能声称与当前关系归一化/value pooling完整相同；但密度矩阵与学习测量本身不是新概念。

[Q-Interference v1](https://arxiv.org/abs/2608.17288v1)已使用幅度/相对相位打分；只读pp1–5并查看p4图像。eqs3–5通过cos差角恒等式写成两个矩阵乘积，仍保留dense T×T score。它不是当前实体跨度partial-swap效应；结构不同也不自动意味着有用新颖性。附录证明、代码和完整实验未审核。此前partial-swap算符与循环相位见证的直接先例继续有效。[版本/阅读范围](../evidence/rmea/20261116-l5-round7/reading-receipts.json)。

普通mixture可通过g=atanh(kappa)表示所选phase-free模型的相同函数，但优化坐标与惩罚不同。这是简单重参数化，不是L5理论。本轮unit及验证集回放检查该数值对应，既不把经典可模拟作为否决，也不把精确重参数化算作新贡献。独立盲态咨询未找到支持充分的新理论路线；当前缺独立任务或学习机制前提，审计结果不能凭空补出它。

## 执行与恢复

source`05e792a51fe45524ca074e16d41a30f56788f493`，sourceSHA`edb88c05d9d79f29fc70eef190e97de79d8259eb5d04bca846bdf7e103ddbc91`，contract`e1644bf3c79f17827515a681e59bb95ab41af84cf73f1518c1a8d6b1d89688e1`。共享CPU task`a340e5e9b97c45b9b97e17b60efa2777`/attempt`48007c77adaf444da3a62ae079adfcbf`，科学执行31.686s、peak645180KiB、退出0。新head训练1、旧head重复训练0、encoder前向/正式实验/test/COS/新资源/师弟访问均0。

同一公共初始化hash、输入完整identity与依赖源码hash核对；零g2/梯度每步检查、selected checkpoint/logits回读复算、atanh普通mixture twin回放均通过。worker闭包7文件/640457字节核验，权重/logits未下载；只回聚合记录。临时owner/worker绑定恢复，逆补丁夹具通过。此短任务仅声明从既有固定输入重新开始，不声明优化器续训或冷恢复。

下一步：Seek an independently justified learning/task premise and audit nearest optimization/auxiliary-coordinate work before selecting a new theory or independent test. No extra seed/epoch rescue and no L5 from this screen.

记录时间：2026-11-15T19:45:17.574327+00:00。
