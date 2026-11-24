# 共同条件酉查询：近邻阅读边界

官方原文：https://arxiv.org/html/2509.11390 ，实际页面标识2509.11390v1、2025-09-14。Arthur M. Faria、Mehdi Djellabi、Igor O. Sokolov、Savvas Varsamopoulos，Quantum Graph Attention Networks: Trainable Quantum Encoders for Inductive Graph Learning。原HTML227094bytes/SHA5ca9724271ea8bfe498986b0ea9fa632772eddac22d7c79df3521b5aaaff7961。已读题录、摘要、引言部分、section4完整及section5开头；未核代码/全部实验/全部引用。HTML文本保留MathML与TeX重复表示，未把文本长度当覆盖率。

Section4.1、式6–8把每个邻居特征乘可训练注意力系数，再用输入相关酉U_(alpha*x)准备零态；邻居寄存器做张量积。Section4.2式9–14进入量子卷积/池化；section4.3式15–19取可观测量期望、与节点自身embedding拼接并监督训练。原文已直接使用input-dependent/trainable unitary来承担attention，因此“输入相关酉即注意力”不是新贡献。

当前候选相对这一具体公式的差别：输入实体摘要控制同一个3-qubit酉，作用于40个预先学习的关系query；相对固定词态做Born分数并按词归一化，decoder仍用词值。没有邻居寄存器张量积/量子卷积池化；Gram守恒是共同酉带来的已知恒等式。该文没直接写出此查询输运公式，不等于证明没有其他近邻，也不等于非平凡新理论。NLP Born投影/CNM等既有近邻仍然适用。

复用上一轮已实际读取的2501.15630v2：Pauli-Z实特征点积及全局范数外积相位项，与本式不同；仍不足以完成新颖性清查。历史R99单类utility构造生成元的双对易旋转不是当前共享上下文电路，但unitary orbit、量子核、共同旋转及其守恒均是已知工具。

最初搜索API404、arXiv查询429记录保留，未重试同一受限查询或绕过访问限制；本次成功的是已知文献原始HTML的正常独立获取。下载原文前未安装额外解析包；bs4不可用后用标准库HTMLParser，未改环境。
