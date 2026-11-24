# Quantum Natural Gradient：原文阅读和迁移边界

原始来源：https://arxiv.org/pdf/1909.02108 ，实际arXiv:1909.02108v3 [quant-ph] 14May2020，PDF注明Quantum接受2020-05-08。作者James Stokes、Josh Izaac、Nathan Killoran、Giuseppe Carleo。15页/831991bytes/SHA40480340a3d3709d2e4a0b72a2857623b3b499cc7d52fdec7964880ba82abc77。单次官方GET200，身份已核，全文/图像不发布到Git。

实际阅读：p1摘要/引言、p2主要论述、p3全文§2.1–2.2/式3–13，p4§2.3–2.4和§2.5开头，p5§2.5余下/§2.6。p3已渲染核查Fisher、Fubini–Study/QGT及伪逆更新公式。尚未读完整后续实验、附录、全部参考文献和代码；不作全面近邻检索清白证明。

原文p3式7是标准QGT，式10–13是FS度量下的最速下降/解线性方程/伪逆更新。p4§2.3明确指出，对一般量子优化目标，量子Fisher与目标曲率没有直接关系；其变分期望值目标不自动等于我们的归一化词注意力加40类CE。p5给出同层可交换门生成元协方差的块度量估计。普通FS预条件或这一协方差公式不是本项目新理论。

本次新诊断的B是每条输入固定word ket加EPS构成的矩阵；它属于归一化Rayleigh读出，不等于查询电路态的QGT，也不等于最终softmax分类分布的Fisher。这是不同对象的区分，不是已经证明训练错配是主因或改优化器能有效。

决定：不把普通QNG作为“新T”推进或启动对应全量实验。转向明确的共享测量容量假设，K2与K1有直接前瞻训练对照；rank2/PSD/混合恒等式同样是已知工具，仍不接受T。新插件尚未运行，见shared-rank2-measurement-20261124/protocol.md。
