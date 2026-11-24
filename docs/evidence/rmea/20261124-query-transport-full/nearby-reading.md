# 共同条件酉查询：近邻阅读边界

官方原文：https://arxiv.org/html/2509.11390 ，实际页面标识2509.11390v1、2025-09-14。Arthur M. Faria、Mehdi Djellabi、Igor O. Sokolov、Savvas Varsamopoulos，Quantum Graph Attention Networks: Trainable Quantum Encoders for Inductive Graph Learning。原HTML227094bytes/SHA5ca9724271ea8bfe498986b0ea9fa632772eddac22d7c79df3521b5aaaff7961。已读题录、摘要、引言部分、section4完整及section5开头；未核代码/全部实验/全部引用。HTML文本保留MathML与TeX重复表示，未把文本长度当覆盖率。

Section4.1、式6–8把每个邻居特征乘可训练注意力系数，再用输入相关酉U_(alpha*x)准备零态；邻居寄存器做张量积。Section4.2式9–14进入量子卷积/池化；section4.3式15–19取可观测量期望、与节点自身embedding拼接并监督训练。原文已直接使用input-dependent/trainable unitary来承担attention，因此“输入相关酉即注意力”不是新贡献。

当前候选相对这一具体公式的差别：输入实体摘要控制同一个3-qubit酉，作用于40个预先学习的关系query；相对固定词态做Born分数并按词归一化，decoder仍用词值。没有邻居寄存器张量积/量子卷积池化；Gram守恒是共同酉带来的已知恒等式。该文没直接写出此查询输运公式，不等于证明没有其他近邻，也不等于非平凡新理论。NLP Born投影/CNM等既有近邻仍然适用。

复用上一轮已实际读取的2501.15630v2：Pauli-Z实特征点积及全局范数外积相位项，与本式不同；仍不足以完成新颖性清查。历史R99单类utility构造生成元的双对易旋转不是当前共享上下文电路，但unitary orbit、量子核、共同旋转及其守恒均是已知工具。

最初搜索API404、arXiv查询429记录保留，未重试同一受限查询或绕过访问限制；本次成功的是已知文献原始HTML的正常独立获取。下载原文前未安装额外解析包；bs4不可用后用标准库HTMLParser，未改环境。

## 训练等待期间补读：section5、table1/2及section6

在同一已保存原HTML解析文本中，补读了完整section5.1/5.2、table1/2、algorithm1及section6；未读取外部代码、补充材料或全部被引论文。该阅读没有调用本项目模型、改动正在训练的源或读取提前测试结果。

原文table1和section5.1明确single模型用30个、multi用40个QM9分子，8输入/量子位，量子与经典模型参数约209/218或1675/1700；table2和图5/6报告的是这些小样本上的训练loss/R²。section5.2称随机取样并比较平均训练表现；在本次已读范围内没有找到另设独立held-out测试的协议。不能把作者关于generalization的措辞替代为已核实的独立泛化证据。

table2中single经典GAT的R²在四个大小区间均高于QGAT（如<=9为1.0000 vs .9504，<=25为.9967 vs .8793）；multi多数模型接近训练拟合饱和，<=25的经典GAT .9999、QGAT .9892。论文对QGAT相对QGNN的收益可以是它自己的源事实，但没有提供Q相对公平经典替换的普遍任务优势，不能搬来填补本项目L3。

对当前研究的作用仅是进一步收紧近邻边界：输入相关酉、attention与浅层模块组合已出现；我们的完整Re-TACRED训练/验证/测试与它的30/40分子训练拟合是不同证据范围。数据规模或评估更严格本身也不是实质理论创新。若当前三路完整资格通过，仍需针对共同酉query机制提出并独立检验更具体的新理论，而非引用该文措辞申请T。

## 更直接的近邻：QKSAN，复用已有原文

Crossref公开元数据本次检索确认QKSAN: A Quantum Kernel Self-Attention Network，Zhao/Shi/Li，DOI10.1109/TPAMI.2024.3434974，2024年12月；元数据只是身份/发现信息，正式刊本内容未读。arXiv公开普通搜索定位2308.13422v2（2023-10-12）；随后发现项目已有原始HTML，故没有再下载同一原文。旧缓存relative-return-training-20261122/qksan-source.html实算SHA034021c7eca4b0746da36b84f05cb8e2afe47235083ff0e5c761f60cf154ffe4，307102bytes，与既有receipt一致。原始获取URL https://arxiv.org/html/2308.13422v2 。

本次重新实际阅读旧原文sectionIII中式14–24及其相邻定义、论证/机制解释；摘要/部分引言也读，未读全部相关工作、sectionIV–VI、实验表、代码或参考文献。式16/17分别用不同可训练变换准备query/key；式21明确为两终态的模平方重叠；式22–24把测量结果与另一个value寄存器的条件操作连接。不能把“量子query/key/value、Born相似度、条件酉或测量控制value”单独作为新理论。

当前Q-RMEA与这份具体公式的区别在40个关系查询、同一实体摘要控制共同query输运、在原词维归一化后汇聚经典signed values。该区别描述实现和约束，不自动形成一般函数类分离：若允许QKSAN任意重定义输入/复合feature map，把(c,r)编码进query准备即可吸收U(c)q_r这类复合状态。反之，也没有核定它的固定具体ansatz可在相同预算精确覆盖当前全参数族。后续T只能依靠新且有内容的约束、设计原则或可区分预测；“我们也是量子核attention”或不同应用任务不足以过T。当前完整效用实验的固定门与结论保持独立。
