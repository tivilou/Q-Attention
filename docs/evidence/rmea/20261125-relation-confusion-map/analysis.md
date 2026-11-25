# 原生模型的关系混淆与稀有类拟合诊断

## 问题与证据范围

gen94已知prior修正关闭：漏检下降但误报大增、主Macro降低。本轮先区分关系类别排序
与null判断，再检查原固定8192fit上的表现，避免把“少样本所以过拟合”当既定事实。
模型、source、最终checkpoint和原训练预算都未改，没有新增训练或学习规则。
该诊断只定位下一问题；不晋级任何候选，不接受新L3/L4-T。

## 四路完整结果

| 模型 | 原fit全40Macro | 原assessment全40Macro | fit强制正类排序正确 | assessment强制正类排序正确 |
|---|---:|---:|---:|---:|
| quantum | 0.806097 | 0.389301 | 2685 / 2792 | 1064 / 1365 |
| dephased | 0.787402 | 0.472501 | 2673 / 2792 | 1129 / 1365 |
| gaussian_base | 0.875209 | 0.484229 | 2727 / 2792 | 1125 / 1365 |
| gaussian_covariance | 0.921118 | 0.512054 | 2749 / 2792 | 1147 / 1365 |

强制正类排序只分析真实正关系，使用gold标签，不能部署为分类器或替代all40主指标。
assess来自已有4096输出，fit为同一已完成1024-update模型的8192原训练例，无更新。
训练/评估差距是描述，不隔离优化、数据难度、统计方差、query、readout或量子因果。

## 全39类按训练支持排序的三等分

固定按fit_count、classID排序，每13类一组，仅作事后描述；跨边界相同count按ID划分。
完整class-analysis保留全部40类，不能只用下面的稀有组或最高负差类决定成功。

| 支持分组（每类fit数量） | assessment总数 | Q fit / assess | D fit / assess | G fit / assess | Gcov fit / assess |
|---|---:|---:|---:|---:|---:|
| fit_support_tertile_1 (1–13) | 59 | 0.6428 / 0.2035 | 0.6258 / 0.2987 | 0.8056 / 0.3454 | 0.8848 / 0.3875 |
| fit_support_tertile_2 (13–41) | 157 | 0.8591 / 0.4008 | 0.8239 / 0.5018 | 0.8983 / 0.4844 | 0.9315 / 0.5190 |
| fit_support_tertile_3 (43–754) | 1149 | 0.9037 / 0.5275 | 0.8986 / 0.5871 | 0.9143 / 0.5937 | 0.9424 / 0.6020 |

最稀少13类只有59条assessment，它们的离散F1波动不能当独立机制确认或总体理论。
这些是已有固定小筛选的性质；没有因结果换样本、补seed或为旧配置放宽晋级门。

## Q相对Gaussian的最大主Macro负贡献类

按全40主指标逐类F1差/40排序，下面只用于解释，完整40类均已保存。

| 关系 | fit / assessment支持 | Q fit / assess F1 | D | G | Gcov |
|---|---:|---:|---:|---:|---:|
| per:date_of_birth | 8 / 6 | 0.8235 / 0.4615 | 0.8750 / 0.9231 | 0.9412 / 0.9231 | 1.0000 / 1.0000 |
| per:date_of_death | 26 / 11 | 0.8936 / 0.1538 | 0.8750 / 0.6667 | 0.9200 / 0.5556 | 0.9804 / 0.6667 |
| org:founded_by | 13 / 8 | 0.9231 / 0.0000 | 0.8462 / 0.5000 | 0.9231 / 0.4000 | 0.9231 / 0.4615 |
| per:stateorprovince_of_birth | 7 / 3 | 0.4444 / 0.0000 | 0.7692 / 0.4000 | 0.8333 / 0.4000 | 0.9231 / 0.4000 |
| per:stateorprovince_of_death | 9 / 3 | 0.7500 / 0.0000 | 0.8000 / 0.0000 | 0.9412 / 0.4000 | 1.0000 / 0.3333 |
| org:number_of_employees/members | 10 / 3 | 0.8889 / 0.0000 | 0.9000 / 0.2857 | 0.9000 / 0.2500 | 0.9524 / 0.4444 |
| org:stateorprovince_of_branch | 43 / 22 | 0.8642 / 0.4103 | 0.8049 / 0.6286 | 0.8966 / 0.6316 | 0.8864 / 0.6512 |
| org:political/religious_affiliation | 27 / 13 | 0.9615 / 0.5000 | 0.9615 / 0.7273 | 0.9818 / 0.6957 | 0.9818 / 0.6957 |

## 条件化关系排序的常见混淆

下列按Q在已知正类中的错误数量排序，不是按主Macro贡献排序，不能混为同一问题。
完整39×39及全部观测到的有向错误边、Q独对/对照独对配对均保留。

| 真关系 → 误判关系 | Q | D | G | Gcov |
|---|---:|---:|---:|---:|
| per:identity → per:employee_of | 10 | 7 | 13 | 6 |
| per:title → per:identity | 10 | 7 | 4 | 5 |
| per:identity → per:spouse | 9 | 2 | 4 | 8 |
| per:origin → per:employee_of | 9 | 5 | 5 | 5 |
| per:title → per:employee_of | 9 | 5 | 3 | 4 |
| org:alternate_names → org:members | 7 | 6 | 3 | 2 |
| org:members → org:alternate_names | 7 | 3 | 3 | 3 |
| org:top_members/employees → per:employee_of | 7 | 8 | 9 | 4 |
| per:employee_of → org:top_members/employees | 7 | 3 | 8 | 9 |
| org:alternate_names → per:employee_of | 6 | 5 | 5 | 3 |

## 身份与执行

关系ID从原source的q_rmea_entity_labels.json按原loader的sorted规则恢复：
40类/no_relation在0，contract SHA 6e822509a07c5c865130ebe315122a9d83e97b1b11496b9e86c6b2a600f8a429。
三个已有private npz SHA、同4096indices和原cohort8fb8393b已核对；四路主CM精确等于
已有公共报告，逐类Macro误差≤0。
fit回放先复现每路原assessment前128条输出与argmax，再按原batch128/micro4、FP32、
TF32off回放8192fit。全部checkpoint SHA前后保持、fit CM独立核对，最大Macro误差
2.22e-16；没有新模型语义、optimizer或checkpoint格式变化。

运行29.595秒、33280条forward、0更新，gen95可信归还。
复用原factory和host238f7e68，检查仅针对新窗口的身份、stored-output parity与CM，
没有调用外部代码审核，也没有将本机检查冒称独立实证/内容审核；没有生命周期矩阵。
0新VALID/TEST、0COS；Windows仅代码、类名、聚合计数与指标，无模型/特征/逐例数组。

原限定L4与全部历史保留，新L3/L4-T未接受，目标ACTIVE。只在明确任务缺口及其
区分预测后选下一插件；不重开已关闭null-bias、prior-strength或旧完整配置。

## 当前可支持的任务认知

量子Q在原fit上全40Macro .80610，高于去相干D .78740；三个支持分组中fit的Q
也均略高于D，但assessment全部低于D。这反对“Q只是比D更学不会训练例”的
单一解释，不证明量子coherence导致过拟合，更不隔离query与readout的贡献。
Gaussian/Gcov在fit与assessment均更强，因此也不能把所有差距都叫方差。

例如出生日期Q的fit强制正类召回8/8、死亡日期26/26，但assessment分别只有
3/6和4/11；创办人Q fit主F1 .92308、assessment为0。省份出生地在fit上也弱，
所以不是全部稀有类都属于同一种失败。主Macro最差类别和最常见条件化错误边
也不相同。下一机制必须同时保留40类全目标，不能只修几个事后挑出的类别。

按fit支持独立于模型结果的排序，最低13类（每类1–13fit）对Q−Gaussian总
all40Macro差贡献−4.61218pp，另两组−2.71612/−2.15310pp，null另计。
最稀少组59条assessment的局限保留；该事后描述不构成新的确认样本或可达增益。

接下来的候选应检验稀有关系的结构共享/适当归纳偏置能否改变这种fit-assessment
差距，并给所有经典/去相干对照相同标签结构与训练输入。当前没有证据证明
semantic query sharing有量子独立收益，更没有以此建立新理论评级。


# 下一轮选择：共享关系查询的固定小筛选

本轮任务证据：Q比D拟合更好但评估更差，稀有类差距大但不是全部欠拟合/过拟合。
Astra/high给出NONE，只建议D注意力替换Q读出；该组件回放不推进当前机制选择，未选。
主agent独立选择一个新的、尚未证成的共享结构假设，不把外部建议写成支持结论。

所有39个公开关系名按同一正则拆词，以固定binary/sqrt-count矩阵做top7 SVD，
逐行L2归一；null单独占第8坐标。40x8固定code乘8x12训练系数，取代40x12自由
query raw坐标。Q/D再通过原pi*tanh和原3qubit浅层circuit；G仍用原Gaussian query。
不增加词输入或独立经典分类器。仅96而非480 query参数，预计各80644 trainables。
使用原seed13自由query到固定code的最小二乘初始化；不声称router初值与旧模型相同。
shared value/decoder/bias和原key保持同样初始化/训练方法。

四新臂Qshared/Dshared/Gshared/Qpermuted：所有结构臂有相同name信息、code秩和相同
共享参数化；Qpermuted沿固定public-name hash顺序循环移位39行，null不动，完整
Gram谱/conditioning不变而语义对应打乱。code不使用训练文本、label目标、支持数或
模型结果，精确数组/词表/奇异值及SHA固定。SVD退化子空间不保证跨库bitwise重建。

原8192fit/4096assessment/全部40类/同组不重叠/seed13/1024updates/b128/micro4，
原AdamW/FP32/TF32off，四fit全结束后最终评估。复用已有nativeQ/D/Gbase/Gcov，
不重跑baseline；Qshared要胜Qpermuted、Dshared、Gshared以及四旧参照，七门全过
才允许完整单seed协议设计。差距可能来自已知共享/优化归纳偏置，不能直接归因
coherence、统计方差或量子理论。新输入结构对控制公平，registered count不等于
有效容量公平。小筛选与L3/理论接受分开，不声明新理论或novelty。

失败后暂缓这一确切rank7/name-token/初始化/1024配置，不扫rank/词法/阈值/LR/seed/
预算/样本/控制救援；不以低支持组或强制正类指标替代主Macro。下一动作是实现此
插件和对实际新语义做一次有界检查，然后由调度器运行四路小筛选。尚未运行。
