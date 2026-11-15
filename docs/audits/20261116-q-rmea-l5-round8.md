# Q-RMEA L5第八轮：已有预测的配对任务审计

**L5未接受，原限定L4不变。**本轮不训练、不前向，只读取三个既有已选模型logits数组及同一开发cache中的标签/角色/训练支持，覆盖全部1024valid。所选checkpoint与数据均已暴露，不是独立确认。

## 原文先例

[Arora/Cohen/Hazan1802.06509v2](https://arxiv.org/abs/1802.06509v2)实际读pp1–5并检查p4原图：线性矩阵因子化会改变优化；Theorem1的连续梯度和平衡Gram前提不适用于本项目非线性态制备/归一化及AdamW。因此不移植其加速结论。

[ExpandNets1811.10495v5](https://arxiv.org/abs/1811.10495v5)实际读pp1–4并检查p4原图：训练时线性展开、推理时精确矩阵收缩已有先例。它与关闭一个非线性融合评分项不同，不声称整模型同构。两篇剩余证明、实验和代码未认证；“训练额外自由度有益、推理可去除”本身不是新理论。

## 配对结果

`phase_free`内部键只表示关闭融合lambda-commutator项，**不是移除复数量子态/Born特征中的全部相位**。与完整所选fusion相比：预测改变0例，完整独有正确0、lambda0独有正确0；最大logit差0.0546898841858。相同汇总指标不能事先推出相同逐样本预测，此处由既有数组直接检查。

lambda0所选模型对比训练全程lambda0的control：预测改变75例，双方正确668，前者独有正确28，control独有正确22，双方错误306。两者选模Macro差+0.007046773542，CE差+0.009299742776；正CE差表示前者更差。保留两个方向，不解释为全面收益。

三个边际分层在本轮结果产生前冻结：实体中点顺序；距离<=32/>32使用既有角色截断尺度；训练支持<=16/>16使用既有overlay补齐目标。各分层轴完整分割1024，不搜索交叉组合；全40类F1/支持及转移均在JSON中保留。

| 分层 | N | lambda0独有正确 | control独有正确 | 预测改变 | CE(lambda0-control) |
|---|---:|---:|---:|---:|---:|
| all | 1024 | 28 | 22 | 75 | +0.009299743 |
| subject_before_object | 531 | 14 | 10 | 39 | -0.007120956 |
| same_midpoint | 0 | — | — | — | — |
| subject_after_object | 493 | 14 | 12 | 36 | +0.026986134 |
| midpoint_gap_le32 | 973 | 28 | 20 | 70 | +0.009342493 |
| midpoint_gap_gt32 | 51 | 0 | 2 | 5 | +0.008484133 |
| training_support_le16 | 151 | 0 | 0 | 17 | -0.058071271 |
| training_support_gt16 | 873 | 28 | 22 | 58 | +0.020952690 |

true-label margin定义为z_y−max其他logit，原始logit尺度敏感，仅描述，不是任务机制或校准证据。按结果挑选的类别不能用于确认解释；本轮没有挑选单样本故事、p值、显著性或总体推断。

另用已公开的两份完整CM做事后**整数核算**，不是新增配对任务或调阈值：全40类TP/FP/FN及精确分数均保留在decision-accounting.json。低训练支持组共有30类、151条valid，所有类TP/FN逐类不变；其中F1发生变化的类索引为[16, 20, 35]，由FP变化产生。因此不能把这批低频类F1变化称为新增找回罕见关系。此为已暴露cohort内的算术事实，不解释误报为何变化，也不推出总体规律。

## 执行与边界

三份已有完整CM/逐类F1/Macro/accuracy/CE均重新计算并逐字段匹配已发表汇总。输入closure/数组/cache哈希执行前后检查；源树干净。任务`f4384a7e269c4cc6a504d033fb1b2ab5`/attempt`53510c871bd04f4b9555fb47bf2e0e02`，1.475613秒，exit0，1CPU/1536MiB/0GPU；两端原调度绑定逐字恢复。4文件/67438字节闭包在worker验证，仅紧凑汇总取回。

收集器完成所有远端哈希/闭包核验和本地写入后，末尾展示引用旧字段screen_disposition触发KeyError。第一个本地收尾版本又把deployment中的worker_repo误从owner元数据读取，触发KeyError；失败源码存recovery-v1，两次失败命令均保留。修正后只重开已取回的receipt和5个输出，逐hash/size及deployment路径复核，无科学重跑或重复远端取回。见collection-finalization.json，不把收尾错误写成科学失败或隐藏为全命令成功。

源`96ff953afd3414398a4f76025b7a79b20b03cdd3`；sourceSHA`fffbaa3b99432243940cd9ff7a21bf7d05328e541b6f9e83578a53110ad5c454`；contractSHA`59ff29de88b727bf1b6d992c4857631b063d478b336909433e20eb776875ff03`。精确代码/生命周期both审查通过；新目录逆补丁fixture通过。新训练/模型前向/编码器前向/正式全量实验/test/COS读取/新资源/师弟访问均0。原数组、模型、cache留在既有worker，不下载也不删。

本轮只定位既有训练比较的决定变化。它没有建立某个独立任务要求、新学习规律、相位必要性或L5。下一步：Audit the decision-rule/false-positive explanation against nearest nondecomposable-F1/calibration work and independently justified task requirements before selecting a new theory. Low-support151valid has no changed true-positive counts; do not recast its F1 precision changes as rare-relation recovery. Retain all adverse results, no threshold fitting to exposed valid, optional seed/epoch or owner formal run; no L5 from this audit.
