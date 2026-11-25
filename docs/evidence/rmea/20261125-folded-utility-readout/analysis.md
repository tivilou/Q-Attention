# 等价单层词评分能否改善量子路由共同训练？

## 实际流程与问题

Q-RMEA由固定词表示和角色距离产生原生量子注意力，原value层Vh+c与40行decoder D
共同定义词评分。上一轮共同冻结utility时Q略胜D/G，但所有绝对表现低，未证成共同
训练就是历史差距原因。本轮检验另一个明确干预：去掉冗余的两层线性读出参数化。

令M=DV，beta=b+Dc；直接输出sum_i alpha_ri M_r h_i+beta_r。
alpha在有效词上归一，因此每行bias只贡献一次。反向用D=[I40,0]、V=[M;0]、c=0、
b=beta即可表示任何direct M，所以在同一个router下函数类别相同，没有额外非线性。
初始M/beta从每路原生参数映射，FP32存在有界舍入差，不能称字节相同。

三路从step1联合训练，无均匀warmup/冻结；不是延长或调整已关闭256/768配置。
原value/decoder/bias仅作固定buffer保留原初始化身份，live读出只用新的M/beta。
不存在独立经典分类器与Q混合残差。新臂各53828可训练参数，原生81028；
原固定buffer仍在state中，不能从参数减少推导checkpoint空间或硬件加速。

## 全部七路结果

| 条件 | 全40类Macro-F1 | CE | 注册可训练参数 |
|---|---:|---:|---:|
| folded/quantum | 0.386360237326 | 0.739693880081 | 53828 |
| folded/dephased | 0.402286076622 | 0.679531216621 | 53828 |
| folded/gaussian_base | 0.404411922317 | 0.685384273529 | 53828 |
| reused/quantum | 0.389300602574 | 0.750126183033 | 81028 |
| reused/dephased | 0.472501085293 | 0.707974433899 | 81028 |
| reused/gaussian_base | 0.484228946800 | 0.705439209938 | 81028 |
| reused/gaussian_covariance | 0.512054050292 | 0.695091843605 | 82308 |

原TRAIN8192fit/4096assessment、同句组不重叠/完整40类、seed13、1024updates、
b128/micro4、FP32/complex64、TF32off和原AdamW保持；三新fit全部完成后才评最终状态。
复用四路精确同cohort/顺序/初始化方法/预算结果，不重训。Gcov是较强函数类别的
实用参照，不是容量匹配的折叠消融。新旧同名条件单独命名，没有覆盖任何结果。

## 所有固定对比与结论

- quantum_folded_minus_native: -0.294037个百分点。
- dephased_folded_minus_native: -7.021501个百分点。
- gaussian_base_folded_minus_native: -7.981702个百分点。
- Qfold_minus_Dfold: -1.592584个百分点。
- Qfold_minus_Gfold: -1.805168个百分点。
- Qfold_minus_Gcov_practical: -12.569381个百分点。

六门0/6通过。六门未全部通过，暂缓本确切折叠参数化配置，不投入全量或额外seed。
完整40类、confusion、support、CE及loss趋势均保留。它是开发筛选，不是独立确认。
不改LR、尺度、rank、初始化、seed、split、预算、指标或控制救援；不判整个家族不可能。

## 理论边界

折叠恒等式是已知线性代数，Nov17 R14/R56已有函数类与参数化优化轨迹不等价的
相关结果，本轮没有把旧证明当新发现。新的仅是固定真实小数据训练检验。
AdamW、clip和weight decay换到了不同坐标，所得差异不能隔离条件数、隐式正则或
唯一机制原因；不声称核范数等价或量子专属理论。形式高级或参数更少都不自动达到T。

## 执行事实

172.891秒、3072实际更新，双2080Ti；gen93可信归还，
exit0且相关进程终态。稳定screen50ce858f/host238f7e68字节未改。
仅新增折叠FP32/FP64端点、独立梯度、NumPy Gaussian、padding和克隆真实8行更新检查，
Astra/high确切新增语义PASS、实际fixtures PASS。全部七路40类CM独立重算误差
1.11e-16。没有额外训练/恢复/COS生命周期矩阵。
0新VALID/TEST、0COS；Windows没有特征、模型或逐例数组，只有代码和紧凑报告。

新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE；原限定L4及全部历史保留。
