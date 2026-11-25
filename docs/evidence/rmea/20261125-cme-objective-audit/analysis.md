# CME没有给出新的量子机制；五臂平方风险诊断仍由G领先

项目流程：输入为Re-TACRED带有已标记subject/object的句子，既有冻结encoder产生token特征和角色信息。原Q-RMEA把token映到三qubit态，以40个relation query分别计算Born重叠并在句内归一化，再与带符号value/decoder配对给出40维类别logits；D使用population重叠，G使用同信息的Gaussian路由。训练用交叉熵，评估以全部40类Macro-F1及冻结消融/经典对照决定晋级。gen98在此骨架上使用已冻结的共享query和distinct-token二阶读出，五臂8192 FIT/4096 assessment、每臂1024更新的筛选已在先前完成。

本轮不改变该模型或重训；只读五个最终logits，按温度1softmax计算平方概率风险及其输出层导数。Macro-F1仍是原主指标；Brier是新的development诊断。没有本轮训练、encoder推理或参数更新轨迹。case/trace边界：服务器既有NPZ包含4096行final logits/gold/indices，SHA3677d55f；本轮未新增逐token态、attention或反向轨迹，因此训练因果与内部路由case均unavailable，不重建；按全部行和预声明分组聚合，不挑选例子。

| 对照前一检查点 | 前轮算符值源审计 | 本轮CME目标审计 |
| --- | --- | --- |
| 对象 | 测量/embedding坐标和集中界适用性 | 条件均值目标的有限label等价性 |
| 实际任务计算 | 无任务行 | 复用4096行五臂输出及FIT计数 |
| 新训练或量子机制 | 无 | 无；风险诊断不构成方法 |

[已测事实] 本轮推进的是一个明确的迁移问题：条件均值嵌入的学习目标，能否为原Q-RMEA的带符号key/value配对提供此前没有的机制或预测。结果是有限类别下先落到已知损失，已有五臂输出也没有出现有利于Q的风险排名反转。原gen98十三门10PASS、停止该确切配置的决定保持；没有据此启动Brier重训。

## 先核原文在做什么

[Shimizu等2403.10859v1](https://arxiv.org/abs/2403.10859v1)使用一个完整(X,Y)观察作为样本，神经网络给固定原子η的系数w，估计条件分布的RKHS均值。p3 Eq3是核平方误差去掉固定自核项；并未普遍要求系数构成概率单纯形。p4 Theorem3.1及pp12–13证明比较相同系数在Gaussian density kernels σ和√2σ下的RKHS范数，用于连续密度带宽目标，不是40类Macro-F1、泛化或量子收益保证。p7–8有限原子RL中使用softmax；那是不同的任务/损失，不提供本项目新机制。

原文17页全部解析，实际范围读10页（1–8/12–13），视觉核p1/3；作者、版本和PDF SHA独立记录。没有把未读的参考文献/完整实验附录或作者代码记为审核完成。

## 对应到本任务后剩下什么

令40个label是固定原子，Gram矩阵K固定。保留被省略的常数后，目标为

`L_K(w,y)=(w-e_y)^T K(w-e_y)`。

条件真分布为q时，

`E[L_K|X]=(w-q)^T K(w-q)+Σ_y q_y K_yy-q^T Kq`。

[机制解释：已知代数] K=I时，裸w是普通平方one-hot回归；若w=softmax(z)，就是Brier损失。K在单纯形切空间严格正定，可给出相应的proper概率估计目标，但这不保证受限函数类、有限优化或argmax的Macro效用。若K本身对任意label置换不变，则K=aI+b11^T，在单纯形上仅是a倍Brier。这说的是K的置换不变条件，不冒称排除了所有使损失相同的仿射规范自由度。按任意label整数放Gaussian核会另加未支持的顺序；若用语义K，公平经典控制也应取得相同信息。

原生骨架中alpha_r(t)按一个relation在句内词上归一化，value/decoder可带符号；它不是P(label|sentence)。一个sentence的gold不能复制为每个token的真值。gen98另含distinct-token二阶项，本轮只在其既有最终40维logits上计算目标，不把该head误写成原生一阶加权和。上述对应属于已知有限核损失代数，不是新T或新插件。

## 已有预测的固定诊断

协议先于本轮新指标冻结；gen98既有输出已暴露，因此都是development证据。4096行、40类、五臂全部保留；稳定FP64、温度1softmax，无拟合校准。Brier越低越好；“类均Brier”是分别按gold类求均值后对40类等权，和原Macro-F1不同。

| 臂 | 行均Brier↓ | 类均Brier↓ | CE↓ | 原Macro-F1↑ |
| --- | ---: | ---: | ---: | ---: |
| Q distinct | 0.331545620 | 0.826482963 | 0.789705653 | 0.500925142 |
| D distinct | 0.347543609 | 0.908258514 | 0.798684148 | 0.436654901 |
| G distinct | 0.315657972 | 0.758019867 | 0.725160110 | 0.510930540 |
| Q query-permuted | 0.358101553 | 0.904496253 | 0.859582852 | 0.453901740 |
| Q collision-mass | 0.338250252 | 0.858399298 | 0.842055663 | 0.481717994 |

Q−G行均Brier为+0.0158876473942。恒等分解为平方概率和差+0.0122350821599，以及−2倍真类概率差+0.00365256523429，两项均正。它是有限预测的代数分解，不是reliability/校准/方差分解；一输入一标签也不识别条件真分布或训练失败原因。Q虽优于D和两个消融，仍落后于G，等权40类后也一样。

八个固定配对分组覆盖全部4096行：null的both-wrong/Qwrong-Gcorrect/Qcorrect-Gwrong/both-correct分别233/158/145/2195，positive分别375/110/74/806。没有从某个有利小组取代全体结论。

输出层梯度为`g_CE=p-e_y`和`g_B=2p⊙[(p-e_y)-p^T(p-e_y)1]`。各模型单独定义的max(p)>=.9错误集有Q310、D317、G271、Qperm347、Qmass350行；平均||g_B||/||g_CE||分别.08377/.09819/.09813/.09162/.07794。各模型集合不同，不能把这作为配对比较；冻结logits的梯度衰减也不识别参数Jacobian、重训轨迹或改善收益。因此不据此把换Brier当已验证的修复。

## 核验与下一步边界

第一实现NumPy向量化，另一实现只用NumPy读NPZ、用Python math逐行重算概率、损失、梯度、全部40类及8组。2715个标量叶最大误差4.44e−15，Brier恒等式最大残差2.22e−16；五个40类混淆矩阵SHA和原Macro完全一致。literal stdout、exit0、输入/代码/协议SHA均保留。它是数值实现交叉核验，不是新样本独立确认或外部科学评审。

读了既有4096预测/gold及FIT标签计数，没有新head forward、update、GPU或资源申请；没有读正文/特征/权重、改核心、打开新VALID/TEST文件、COS或模型API。原生源副本SHA9f4bfc57保持。全部原始数组留服务器，仅导出聚合。原限定L4、原full负结果和gen98–100等历史保持，新L3/L4-T仍未接受，量子主目标ACTIVE。

[尚未证明] 本轮未建立新量子机制、实际错误因果或重训收益。固定两查询中attention表达性请求超时，kernel inductive-bias返回6篇metadata/abstract；这不是穷尽搜索。2503.20683v3等4条ID早在common-utility-router银行收过摘要，本轮明确复用，不称新读原文。

唯一推荐下一步：取得并核[2503.20683v3纠缠张量核原文](https://arxiv.org/abs/2503.20683v3)，读清定义及适用命题，再同原生可训12角度/三qubit overlap、归一化signed readout及旧Pauli/shared-effect/frame排重。假设仅是“该视角可能揭示不同于换损失/坐标的任务结构限制”，并非已发现。最小工作为一个原文的CPU/网络范围读与原生方程对照，GPU预算0、无需新资源申请；持续科研授权已覆盖。成功门是找出实质结构对应、正确假设和Q/D/G可区分预测后才设计候选；若只是已知张量改写、固定KRR结论不适用或旧profile救援，就记该迁移NONE并停止，不自动fit。具体入口与门见next-direction.json。
