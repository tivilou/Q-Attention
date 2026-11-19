# Q-RMEA R99：保留固定谱查询旋转候选

本轮新增一个结构候选：效用通过反Hermitian交换子旋转query，保持effect原有特征值，
不增加可训练参数或调节系数。推导得到固定步长Gibbs在纯态极限的响应衰减，
以及该旋转在明确非退化条件下保留正单类增益的闭式结果。
两命题及分类反例由指定gpt-6.1-sol/medium一次独立数学审核通过，must_fix=0。

结果只支持继续筛选：实际epsilon=1e-6下的误差界、任务收益、近邻新颖性均未确立。
没有运行新的科学数值任务、训练或正式实验，没有发布新模型权重。
原限定L4、R96开发正向点和全部停止路线不变，L5 NOT_ACCEPTED。

读取Gluza2024官方原文的相关页面并核对公式，确认double-bracket旋转是已有工具，
不将其包装为新量子原理。本文只提出其与实际效用归一化及有限步抑制的待验证联系。

R98执行准备保留，补充准入审核BLOCK：配置读取比较后再rename并不保证与未协调的
并发写入原子互斥。科学任务提交0、live配置变更0。先解决真实运行时边界；
不覆盖其他项目的配置、不反复审核同一未修复方案、不绕过调度执行模型。

[数学推导](../evidence/rmea/20261119-l5-round99/isospectral-query-rotation-v001.md)

[候选比较与止损](../evidence/rmea/20261119-l5-round99/candidate-triage.md)

[原文阅读范围](../evidence/rmea/20261119-l5-round99/primary-reading-receipt.json)

[独立数学审核](../evidence/rmea/20261119-l5-round99/math-review-record.json)

[准入阻塞的独立审核](../evidence/rmea/20261119-l5-round98/admission-supplement-review-record.json)
