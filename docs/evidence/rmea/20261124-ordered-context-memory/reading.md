# 有序记忆近邻阅读范围

2026-11-24。通用web搜索两query请求返回HTTP404 Invalid URL POST/v1/alpha/search，没有可用搜索结果；不是无近邻证据。随后直接读取官方arXiv三条abstract页面，核对2006.14619、1511.06464、1602.06662真实题名。1602.06662是Henaff/Szlam/LeCun的Recurrent Orthogonal Networks and Long-Memory Tasks，不能误记为Wisdom的full-capacity uRNN；仅摘要，不依赖其正文。

## Bausch, Recurrent Quantum Neural Networks

官方2006.14619v1（PDF第1页ID与作者/题名）；29页/3269035bytes/SHA256 ba739f6480fc42a0cfff5536cc895b29c46faac581ece7db3a8d11e79739ea4f。实际读第1–6页：摘要/引言、2.1、2.2开头及Eq1、neuron图的抽取文字。图只读caption/抽取文字，未视觉核查，未读2.3完整cell、后续实验/appendix/code。因此仅采用已知量子循环memory框架这一事实；不复现其neuron/amplitude amplification或借用其梯度/实验结论。其cell包含input/work/output stage、postselection和amplitude amplification，不等于本候选的直接数据依赖unitary前缀测量。

## Arjovsky/Shah/Bengio, Unitary Evolution Recurrent Neural Networks

官方1511.06464v4（PDF第1页ID、三作者、题名/ICML2016）；9页/1300482bytes/SHA256 6d1cece68d88b7fd8c2198c8277066f8975c37998360e1a0fc101c1cb11b8751。实际读第1–3页：引言、Section2 Eq1–4梯度上界、Section3 Eq5–6结构化unitary、Section4开头。其讨论包含非线性和ReLU条件，不能从酉转移推全部参数梯度不消失。结构化unitary组合和norm preservation是已知；我们的12维Givens控制不是其Eq6复现，也不是整个经典RNN家族。

## 迁移假设与边界

已有RMEA逐词独立prepare，原冻结RoBERTa输入已含上下文。候选改变的是量子路由内部的跨词状态依赖，不是发现原模型没有词序。最小检验为同输入原生Q、逐步去相干Markov、两种同12角的正交memory经典控制和Q memory。零角归一化U(theta)U(0)^dagger避免原CNOT固定置换在无输入时传播；1/sqrt(length)为本次前瞻固定约定，不是由论文证明的最优值。原生最强经典仍是正式实验的必要对照。

该领域与量子/酉RNN直接碰撞；不声称循环、非交换或范数保持本身创新。本轮先判断任务效用，若不通过固定筛选则关闭确切配置。尚未全面检索量子automata、QNLP sequence models或近年全部相关工作，不能出新颖性结论；若效用通过，需扩展这些近邻并另建非平凡同核心理论。

原文：https://arxiv.org/abs/2006.14619v1 ；https://arxiv.org/abs/1511.06464v4 。PDF/parse/身份receipt见papers/。仅小型论文和代码报告在Windows，无实验特征或checkpoint下载。
