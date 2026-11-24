# 按归一化attention Fisher修正query方向：固定五路结果

既有TRAIN整句4192/4000、seed13、原生FP32/b128/每路512更新；五路总2560更新，实际179.345850秒。所有fit结束后只评估final512，全部4000/all40；零新VALID/TEST/COS。

| 条件 | all40 Macro | CE | 正→null | null→正 | 正关系错分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quantum_adam | 0.172509408 | 1.076011419 | 791 | 206 | 136 |
| quantum_fisher | 0.157106227 | 1.096374631 | 773 | 199 | 152 |
| dephased_fisher | 0.146655039 | 1.054304600 | 758 | 248 | 188 |
| real12_fisher | 0.201875547 | 0.915886939 | 613 | 262 | 140 |
| softmax12_fisher | 0.185407614 | 0.940431058 | 615 | 286 | 165 |

固定8门通过1/8，结果CLOSE_EXACT_FISHER_QUERY_SCREEN。

| Q Fisher相对 | Macro差/百分点 | CE差 | 正确修复 | 新损害 | 净正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quantum_adam | -1.540318 | +0.020363212 | 131 | 122 | 9 |
| dephased_fisher | +1.045119 | +0.042070031 | 266 | 196 | 70 |
| real12_fisher | -4.476932 | +0.180487692 | 227 | 336 | -109 |
| softmax12_fisher | -2.830139 | +0.155943573 | 260 | 318 | -58 |

保留全部40类支持、零类、配对修复与新损害。正→null减少不是等量正确修复。详细每金标类配对在paired-analysis.json；拟合39类/评估38类，仍使用40分母。暴露TRAIN单seed短筛选不构成独立L3，也不排除全家族完整训练。
原生AdamW锚点对上轮相同原生K1、同初始/顺序/分区：{'same_partition': True, 'logits_exact': True, 'maximum_logit_difference': 0.0}；重复确定性锚点不是独立重复实验。

独立重算全部5×4000×40预测，混淆精确一致/Macro误差0/CE误差6.47e-08、8门一致；FP32 final/latest及参数/更新数/共享initial和顺序一致。

新参数链式导数/词归一化Fisher/padding/real零及sub-epsilon边界对原生autograd验证；每路clone真实首8更新，NumPy独立更新方向/全body原AdamW步完全一致，精确修订Astra-high PASS。两个代码检查BLOCK均在科学运行前修复并保留，不是实验重跑。稳定core仅新增默认兼容的优化器工厂入口，未重做训练/队列/恢复/COS矩阵。
算法是Fisher形状修正的AdamW方向，不是坐标不变自然梯度或新T。额外Fisher计算不同，不能声称相同计算量。已知null-direction例说明Adam元素缩放可产生不可观测分量，阻尼逆矩阵可放大它；例中步范数不变但null分量从10%升95.714%。这是算法风险，不是本次各模型的已证原因。即使短筛选过门，正式研究也必须包括未经修正的经典对照，不能靠损害新经典组建立贡献。
未达到新L3/L4-T，原限定L4和全部历史结果保留。局部checkpoint/逐例数组留GPU节点，Windows仅紧凑报告；短负结果无后续二进制消费者时仅Git保存。

关闭本确切阻尼/逆平方根/每行步长匹配/seed13/512配置，不改阻尼、指数、频率、步长、seed、预算、split、metric或对照救援，不晋级全量。
