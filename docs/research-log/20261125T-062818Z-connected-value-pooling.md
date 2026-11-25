# 同词线索共现：小数据单 seed 读出筛选

## 实际流程与假设

原生Q-RMEA由固定词表示及主宾语角色距离产生量子词态，关系态给出词注意力，
注意力聚合64维signed value，最后经decoder分类。本候选保留路由、线性均值、
偏置和训练核心，只让同一个路由额外读取32对相邻value通道的加权协方差。
零初始化的40×32个系数使初始logit与各自原生模型完全一致。

想检验的是“线索在同一个词上一起出现”能否带来超过“各线索均值相乘”的作用。
固定attention和value时，两组均值都为零的两词数据可以有+1/-1协方差；
这是已知统计量的区分见证，不是端到端模型类分离，也没有证成真实错误的成因。
冻结encoder本身可能已经编码词级共现。所有新臂都是同一个router的扩展读出，
不是独立经典分类器与Q的混合残差。

## 全部八路结果

| 条件 | 全40类Macro-F1 | CE | 来源 |
|---|---:|---:|---|
| quantum_covariance | 0.428488140718 | 0.781959116459 | 本轮训练 |
| gaussian_covariance | 0.512054050292 | 0.695091843605 | 本轮训练 |
| quantum_mean_product | 0.402680485626 | 0.905063271523 | 本轮训练 |
| gaussian_mean_product | 0.511174688570 | 0.839876890182 | 本轮训练 |
| dephased_covariance | 0.473625962137 | 0.707378268242 | 本轮训练 |
| reused/quantum | 0.389300602574 | 0.750126183033 | 同协议旧控制复用 |
| reused/dephased | 0.472501085293 | 0.707974433899 | 同协议旧控制复用 |
| reused/gaussian_base | 0.484228946800 | 0.705439209938 | 同协议旧控制复用 |

8192fit/4096assessment均来自原TRAIN，同句组不重叠且覆盖全部40类。
五新fit统一seed13、1024updates、b128/micro4、原AdamW、FP32/complex64；
全部新fit结束后才评最终状态。各新臂82308个注册参数，三原生控制81028个；
新增臂之间计数一致不代表有效容量相等，不能把对原生的所有增益都归因于统计量。
旧控制身份、cohort、原初始化方法、batch顺序、预算及依赖固定后复用，不重训。

## 所有预定对比与决策

- Q_covariance_minus_native_Q: +3.918754 个百分点。
- Q_covariance_minus_Q_mean_product: +2.580766 个百分点。
- Q_covariance_minus_Gaussian_covariance: -8.356591 个百分点。
- Q_covariance_minus_dephased_covariance: -4.513782 个百分点。
- Gaussian_covariance_minus_native_Gaussian: +2.782510 个百分点。
- Gaussian_covariance_minus_Gaussian_mean_product: +0.087936 个百分点。
- dephased_covariance_minus_native_dephased: +0.112488 个百分点。

固定七门2/7通过。七个固定门未全部通过，本确切配置暂缓，不进入完整单seed或多seed。
全部40类F1和support见class-analysis.json，所有混淆矩阵保留在summary.json。
上述差值是开发集描述，不是显著性或独立确认；不能把不同模型的优化差值称为因果交互。
不更换通道配对、幅度、seed、cohort、预算、主指标或对照救援；也不据此否定整个家族。

## 本轮新增认知

相对于Q的原生读出，同词统计有局部收益；但公平population与Gaussian读出更强，
所以“给Q加二阶统计就能形成独立贡献”在本固定预算中没有通过。
Gaussian协方差与均值乘积控制的Macro差值只有
0.087936个百分点；
两者都超过原生Gaussian。因此经典路由的增益不要求特定的中心化同词统计；
这只是本开发筛选中的比较，不能证明两者总体等效或给出唯一因果解释。
下一项假设需要解释相对于强经典对照的剩余差距，而非继续堆叠已知读出功能，
或把原生Q的局部改善包装为量子独立贡献。没有据这些结果选定新的插件或全量任务。

## 理论与证据边界

C=E[ab]-E[a]E[b]=1/2 E[(a_i-a_j)(b_i-b_j)]。
同词共现与独立两次抽样的区别、平移不变性、singleton为零均为已知协方差性质。
实际阅读ICCV2015 Bilinear CNN Models原文PDF第2–3页：局部外积后聚合及反向传播
已属先例；我们的关系加权、中心化和32对压缩不自动构成新理论。
本轮不申报新的量子资源、硬件加速、L3或L4-T。

## 执行与成本

共5120次新更新，291.961秒，双2080Ti稳定ready queue。
只验证新增读出的独立pair-sum/double-draw公式和全部参数梯度、零系数原生端点、
NumPy Gaussian、padding/singleton/平移，以及五臂克隆真实8行更新。
针对确切新增代码的Astra/high审查PASS；稳定screen及host wrapper不变，未跑生命周期矩阵。
全部八路CM/40类独立重算误差2.22e-16。
0新VALID/TEST、0COS；Windows只存代码与紧凑报告，没有checkpoint、特征或逐例数组。
本轮进程终态/exit0，allocation gen91可信归还，其他项目分配保持。

idea-maturity1新L3/L4-T仍NOT_ACCEPTED；目标ACTIVE，原限定L4及全部历史保留。

[完整结果、源码与固定协议](../evidence/rmea/20261125-connected-value-pooling/summary.json)。
