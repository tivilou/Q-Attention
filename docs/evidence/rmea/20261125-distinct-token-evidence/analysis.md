# 删除同token自乘：固定五路小规模筛选

## 实际流程与本轮差异

项目输入是句子、指定实体对、服务器已有冻结RoBERTa词特征及实体角色；目标为40类
关系识别，包括no_relation。关系条件attention对value加权，输出40个分类分数，
以全40类Macro-F1（每类F1等权平均）为主指标。当前只在原TRAIN8192条拟合、
另4096条评估，句子组隔离；没有新读取VALID/TEST，也不是全量或独立确认。

[已测事实] 上一轮Q均值乘积 .504045671350好于Q中心化 .478786491123，
仍低于强GnativeCov .512054050292。该控制结果未被重命名为成功候选。
本轮冻结一个新开发假设：删除均值乘积中的同token位置自乘项是否有帮助。

| 部分 | 上一轮 | 本轮 |
|---|---|---|
| 查询/基础分类 | 固定语义共享查询和原读出 | 原样复用 |
| 二次项 | M=mu_even*mu_odd，或中心化covariance | U=M−Σalpha²*v_even*v_odd |
| 额外控制 | 无 | QcollisionMass=(1−Σalpha²)*M |
| 初始化/核心/预算 | 零moment decoder、原稳定核心 | 原样复用，各81924 trainables |

mu=Σalpha*v，U等于Σ_{t!=u}alpha_t*alpha_u*v_t,even*v_u,odd；不归一、不
调整温度或通道pair。Q/D/G及置换Q都使用相同U，QcollisionMass用于区分去自乘
与单纯注意力集中度调制。五个fit全部结束后才统一评估，旧九个参照直接复用。

## 全部结果与固定判定

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
| new/quantum_distinct | 0.500925142309 | 0.789705634 | 379 | 391 |
| new/dephased_distinct | 0.436654900537 | 0.798684120 | 450 | 351 |
| new/gaussian_distinct | 0.510930539751 | 0.725160122 | 329 | 378 |
| new/quantum_permuted_distinct | 0.453901740475 | 0.859582841 | 433 | 406 |
| new/quantum_collision_mass | 0.481717994286 | 0.842055619 | 392 | 373 |
| reused/previous/quantum_shared_covariance | 0.478786491123 | 0.775347114 | 434 | 297 |
| reused/previous/dephased_shared_covariance | 0.483777932823 | 0.744908094 | 454 | 300 |
| reused/previous/gaussian_shared_covariance | 0.498421796471 | 0.692677498 | 361 | 343 |
| reused/previous/quantum_shared_mean_product | 0.504045671350 | 0.983658075 | 316 | 497 |
| reused/previous/quantum_permuted_covariance | 0.454453570942 | 0.760431051 | 514 | 303 |
| reused/earlier/shared_quantum | 0.468022177009 | 0.750726461 | 433 | 316 |
| reused/earlier/native_quantum_covariance | 0.428488140718 | 0.781959116 | 418 | 396 |
| reused/earlier/native_gaussian_covariance | 0.512054050292 | 0.695091844 | 372 | 336 |
| reused/earlier/shared_gaussian | 0.500862151478 | 0.699699640 | 372 | 351 |

[已测事实] 固定十三门10/13通过。十三门未全过，本确切配置暂缓，不进全量或多seed。 差值和全部40类CM/F1/support
见contrast-analysis.json与class-analysis.json，不选类别、换主指标或删除不利对照。

## 认识与边界

[机制解释] U是已知非对角二次统计，适用于任意归一attention；不能仅凭公式或成绩
称其为新T、语义subject/object绑定、token顺序建模或量子独占资源。Q相对去相干及
Gaussian对照的结果才涉及本轮相对贡献，经典衍生另记自己的方法身份和证据等级。
[尚未证明] 本轮仍是暴露开发集的单seed筛选，不接受新L3/L4-T，不转移原限定L4。
逐例语义、attention和训练轨迹没有纳入聚合报告；不从CM反推机制案例。

新增U/mass独立双token公式、全参梯度、非零query差分、单token=0、同value恒等式、
padding、零moment共享端点和clone actual8检查记录于fixtures.json。首轮Astra/high
BLOCK指出差分可在两边均0时假通过；已保留原记录，加入两边均有限且绝对值>1e-10
约束，继续要求误差<1e-7；第二轮PASS。静态审核与实际执行证据分别保存。
代码修正仅涉及fixture，没有改变科学公式、数据、预算或十三门。

seed13，每路1024updates/b128/micro4，原AdamW/FP32-complex64/TF32off；
五fit5120物理更新，耗时318.537秒，双GPU动态队列，gen98
已可信归还。核心50ce858f/host238f7e68未改，不做续训/COS生命周期矩阵。
十四路CM独立误差≤1.67e-16，0新COS，Windows0模型/
特征/逐例数组。短筛选仅Git保留代码/配置/聚合报告，未自动归档checkpoints。

## 下一步

唯一后继决策见next-direction.json。若任一门失败，不对本确切profile追加归一化、
集中度、通道pair、rank、token、scale、LR、seed、budget、cohort或删除对照救援。
所有先前负结果保持原状，不能据本轮否定整个方法族或接受量子/经典新L3/T。

### 本轮收尾的唯一建议（仅设计，不启动fit）

[已测事实] 对现有shared、covariance、distinct三组Q/G的40类F1逐类复核，15类三组均Q<G。全部40类、support与差值存于next-direction.json；低support类别不作稳定总体归因，相关性不证明共同机制。

[尚未证明] 后继只审计“有序、实体角色绑定的交互读出”是否构成实质不同的任务假设，尚未选定新插件。当前二次聚合可交换，并不代表整个模型丢失顺序：冻结RoBERTa特征已有上下文和位置信息。这一边界必须在任何实现前明确。

最低下一步为原始文献与历史order/entity失败核查、当前聚合的精确不变性见证和可区分反例。零新训练、零任务forward、零GPU申请；若等同旧配置、仅为U/mass调参或依赖未经证明的整体顺序丢失，则设计阶段停止。仅通过该设计门才冻结小筛选及公平经典/去相干对照，不自动全量或接受L3/T。原始文献检索工具本轮HTTP404，未声称已阅读或新颖性成立。
