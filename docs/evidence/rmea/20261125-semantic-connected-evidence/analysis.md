# 共享关系查询与中心化证据：固定五路小筛选

## 实际工作流与这次改动

本项目从句子及指定实体对识别40种关系（包括no_relation）。使用服务器上已有的
冻结RoBERTa词特征、实体角色及词mask，训练关系条件注意力和分类读出，以全40类
Macro-F1衡量各类等权表现。当前阶段仅使用原TRAIN中固定8192条拟合、另4096条评估，
句子组不重叠；没有新读取VALID/TEST。这是开发筛选，不能代替全量或独立确认。

[已测事实] 上一轮语义共享使Q优于其原生版本，但仍输给同样共享的D/G。
本轮把该固定共享查询与先前的固定二阶读出合并，检验一个联合假设：是否能保留
公平经典/去相干对照下的收益。原先两轮确切配置仍关闭，没有改rank、分词、seed或预算。

| 部分 | 上一轮 | 本轮 |
|---|---|---|
| 关系查询 | 固定40×8 label-name code乘8×12参数 | 原样复用 |
| 读出 | 注意力加权value均值线性读出 | 加32对相邻value通道的中心化二阶统计 |
| 新参数 | 无moment读出 | 40×32 moment_decoder，全0初始化 |
| 训练核心/优化器/数据 | 固定版本 | 原样复用 |

同一attention权重alpha下，mu=Σalpha*v，新增C_j=Σalpha*v[2j]*v[2j+1]
−mu[2j]*mu[2j+1]，最后以relation-specific系数线性读出。mean-product对照固定用
正号mu[2j]*mu[2j+1]替代C；code置换对照打乱39正类对应、保持null行。Q/D/G同样
获得公开关系名结构及二阶统计，均81924 trainables；不声称有效容量相同。
0初始化使各模型初始输出精确等于对应gen96共享端点。

## 全部固定结果

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
| new/quantum_shared_covariance | 0.478786491123 | 0.775347114 | 434 | 297 |
| new/dephased_shared_covariance | 0.483777932823 | 0.744908094 | 454 | 300 |
| new/gaussian_shared_covariance | 0.498421796471 | 0.692677498 | 361 | 343 |
| new/quantum_shared_mean_product | 0.504045671350 | 0.983658075 | 316 | 497 |
| new/quantum_permuted_covariance | 0.454453570942 | 0.760431051 | 514 | 303 |
| reused/shared_quantum | 0.468022177009 | 0.750726461 | 433 | 316 |
| reused/native_quantum_covariance | 0.428488140718 | 0.781959116 | 418 | 396 |
| reused/native_gaussian_covariance | 0.512054050292 | 0.695091844 | 372 | 336 |
| reused/shared_gaussian | 0.500862151478 | 0.699699640 | 372 | 351 |

- Qjoint_minus_Djoint：-0.499144个百分点。
- Qjoint_minus_Gjoint：-1.963531个百分点。
- Qjoint_minus_Qshared：+1.076431个百分点。
- Qjoint_minus_Qmeanproduct：-2.525918个百分点。
- Qjoint_minus_Qpermuted：+2.433292个百分点。
- Qjoint_minus_QnativeCov：+5.029835个百分点。
- Qjoint_minus_GnativeCov：-3.326756个百分点。
- Djoint_minus_Dshared：-0.902622个百分点。
- Gjoint_minus_Gshared：-0.244036个百分点。
- sharing_cov_gain_Q_minus_D：+1.979053个百分点。
- sharing_cov_gain_Q_minus_G：+1.320467个百分点。

[已测事实] 固定八门3/8通过。这个确切联合配置暂缓，不进入全量或多seed；不挑类别、换主指标或扫描参数救援。
Q必须严格优于四个新对照及Qshared/QnativeCov/GnativeCov/Gshared四个固定参照；
仅优于自身Q基线不足以确认量子独立贡献。所有九路及40类CM/F1/support完整保存。
新增公式核对使用真实小批量，记录于fixtures.json；案例级预测、attention及优化轨迹
未纳入此轮聚合收尾，不补写或从CM反推个例。机制实例解释仍有这一限制。

## 已获得与尚未获得的认识

[已测事实] 各对照差值和完整类级结果见contrast-analysis.json与class-analysis.json。
共享前后增益差只是同一开发集上的描述，不能据此声称相干导致互补、统计交互被识别、
显著性或泛化规律。CE和null错分帮助描述行为，不能替换冻结的Macro-F1门槛。

[尚未证明] 小筛选尚不支持新L3/L4-T。低秩语义共享和二阶pooling是已知工具，组合
本身未建立实质理论创新、量子硬件优势或相干机制的独立贡献。原限定L4和全部历史保留。

按照新skill分别处理贡献：量子方案相对自身基线的效用、相干归因、经典衍生候选独立
记录。经典组合即使优于Q，也只成为自身协议的候选，仍需强基线、关键消融、公平经典
替代和分级投入后的适当确认，不能自动获得L3/L4或替代当前量子L4-T目标。

## 执行和投入

seed13，每路1024updates/b128/micro4，原AdamW/FP32-complex64/TF32off。
五fit全部完成后才统一评估，5120实际更新，耗时312.867秒；双GPU动态队列，
gen97已可信归还。四个旧参照完全复用，无重复baseline。
稳定核心50ce858f/host238f7e68未改，只检查新增组合端点、独立公式/梯度、padding和
clone actual8更新；A6API Astra6.2/high静态审核通过，实际fixture另记，二者不混称。
九路CM独立重算误差≤1.67e-16。0新COS，Windows
0权重/特征/逐例数组；短且可重启的失败筛选仅Git保留代码、配置和聚合报告。

## 下一步

唯一推荐方向在next-direction.json冻结：检验删除均值乘积里的同token自乘项，
U=M−Σalpha²*v_even*v_odd，而非继续调covariance。新增Q/D/G/置换U及
Q集中度乘M五臂，后者区分纯注意力集中度效应；保留本轮全部九个旧参照，固定
十三个全40类Macro比较。原8192/4096、seed13、每臂1024updates；十三门全过
才设计完整单seed，任何门失败就暂缓该确切配置。尚未实现或启动这一后继实验。
U身份是已知非对角二次统计，不是新T，也尚未证明多个真实关系线索存在互补。
外部Astra/high仅提供一个建议，主模型加入集中度对照并独立冻结方案；未获文献新颖性。
具体已知二次读出等价和观察边界见mechanism-note.md；经典衍生路线未自动替换目标。
若八门未全过，不为本确切profile追加rank、token、pair、通道、scale、threshold、LR、
seed、cohort、budget或删除对照救援，也不据一次失败否定整个方法族。
