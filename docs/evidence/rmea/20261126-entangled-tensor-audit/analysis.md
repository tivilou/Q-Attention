# 纠缠张量核可精确表示原生Q/D，但这一对应没有带来新干预

项目流程：Re-TACRED输入为标有subject/object的句子，冻结encoder产生token特征；原生Q-RMEA把特征及角色映到三qubit态，用40个relation query分别计算Born重叠，在词上归一化后配对带符号value/decoder，输出40类logits。D使用去相干后的population重叠，Gaussian等为公平经典路由对照。训练用交叉熵，晋级由全部类Macro-F1与已冻结门槛决定。gen98共享query/distinct-token五臂已完成并按十三门停止，本轮不改它，也没有新训练或任务数据推理。

本轮工作是原文/原生方程的适用性审计：同一电路能否被源的纠缠张量核表示，以及该表示是否真的引入不同的学习机制。先固定来源、合成输入和数值门限，再作精确对应及另一实现核验。

| 与上一检查点比较 | 前轮CME目标诊断 | 本轮ETK结构审计 |
| --- | --- | --- |
| 问题 | 有限label是否只是Brier | 固定电路是否只是张量重表达 |
| 实际任务数据 | 复用4096预测/gold | 0任务行；固定合成输入 |
| 新结果 | Q风险仍输G | 概率、归一化、全梯度均精确对应 |
| 新插件/效用证据 | 无 | 无 |

## 原文身份与已读范围

[Shin/Sweke/Jeong的2503.20683v3](https://arxiv.org/abs/2503.20683v3)已核exact ID、作者、v3栏和HTTP200/SHAff22b475。Atom保留带“New perspectives…”的旧题名，PDF题名是“Quantum kernels through the lens of entangled tensor kernels”，两者分别记录。首次helper按短题名核验报metadata-title mismatch，保留题名差异后按唯一ID/版本/作者核验通过，没有重取PDF或静默改名。

25页均解析，实际读1–17/21–23共20页，视觉核1/11/16。源pp7–11给PSD core与局部三维三角特征的ETK表示；p4允许固定训练参数后作表示分析。p16的具体Mercer例子限单层编码，pp22–23的通用构造还需相对输入分布正交化。没有审核全部引用定理、代码、未读尾页，也没有把源的硬件复杂性条件直接用作本项目的量子效用否决条件。

## 已测事实与适用性

[已测事实] 用64维实Pauli密度坐标，原生12角度得到`v=B⊗(1,cos,sin)`；Q/D的core分别是B^T B和B^T Π_Z B。只用虚维64收缩，不展开531441平方矩阵；core矩阵rank上界64/8，一种MPO bond上界4096，未求最小bond。完整推导见derivation.md。它是已知ETK/PTM工具，不是新T。

固定96合法角度的density坐标误差3.33e−16，Q/D Gram误差2.33e−15/5.55e−16；数值rank64/8。Q−D的coherence Gram也为PSD，但非对角分数有正有负，不产生signed readout的单调收益承诺。

固定一个新随机native初始化（未训练）、3个合成sentence×7token、40queries、有效长度7/5/1：两次原生head forward覆盖Q/D，均81028参数、参数SHA未动。Q的probability/alpha/logit误差1.44e−15/2.50e−16/6.25e−17；全部7参数tensor及合成h梯度最大5.42e−18。D也通过同门限。主计算0.0780秒，无optimizer。

另一NumPy稠密8×8门实现独立核对Q/D全部40relation、mask、EPS、signed utility和logits；最大读出误差8.50e−17。三档双侧角度方向差分最大误差1.04e−13，独立重算0.1560秒。输入、方向、literal输出、环境、exit0、source/code/protocol SHA均记录；没有重跑旧实验。

case/trace：audit-results.json保存固定合成角度、utilities、mask、完整40类port及方向。不是自然语言case；本轮没有训练轨迹或encoder语义trace，因此实际错误原因、语义机制和泛化case均不适用，不以合成等价性替代它们。没有独立样本确认或外部科学评审。

[机制解释] 此表示在现有参数下保持logits和精确导数；不能仅换实现声称任务改进。可训自由MPO core会换函数类，是另一个模型，源自身的参数固定分析/单层谱例子也不保证原生联合训练或Macro效用。本轮不借此重开word/query-frame、shared measurement、angle/qubit/depth等已关闭配置。

[尚未证明] 新Q机制、新L3、非平凡新理论及L4-T仍未建立。没有新任务行、训练更新、GPU、调度申请、服务端写入、模型API或COS；只有本地CPU合成方程审核。三qubit可经典模拟不是拒绝理由；函数不变的已知重表达不足以形成新贡献。原限定L4和全部正负历史保持，量子目标ACTIVE。

## 唯一推荐下一步

复用既有metadata线索[2609.13360v1 Structured Quantum Kernels for Chaotic Forecasting](https://arxiv.org/abs/2609.13360v1)，先取得/核原文，读实际结构编码、经典控制、消融与机制定位协议。它的摘要声称经典数据上的结构归纳偏置；本轮尚未核这些实证，不能移用源的数值或把时序输入类比成NLP证据。

最小工作是一个原文的CPU/网络范围读和任务结构对应，GPU预算0，既有持续科研授权覆盖。成功门是源证据及实质输入结构支持新的原生机制与Q/D/G判别预测后再设计；若控制不匹配、机制只是事后proxy，或迁移等于旧role/grouping/angle/bandwidth救援，就该迁移记NONE并停止，不自动fit。自由经典MPO与纯ETK换实现均不作为当前量子目标的替代。
