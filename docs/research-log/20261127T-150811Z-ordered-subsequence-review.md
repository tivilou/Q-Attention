# 20261127T-150811Z | 有序子序列独立资格与条件预算

## 原任务流程与本轮改变

原Q-RMEA以一句话和有序主客体为输入：固定骨干产生1024维词特征，加4维角色坐标；独立量子模块产生每类对各词的注意力，池化全部原词的64维value，输出原40类关系。目标仍为同一量子核心的L3独立任务贡献及L4-T理论、区别性预测、实现一致性与适当独立审核。原限定L4和全部失败、无效与正结果保留。

当前候选固定k=min(3,n)，对含标记词t的递增、无重复k词子集，先相干求和其有序门积，再做Born读出。它区别于旧gen98的value矩及旧history的经典硬币、全长度路径混合。区别足以进入一次有界资格工作，不等于文献新颖性、非平凡理论或任务收益。

| 项目 | 上一步 | 本轮 |
| --- | --- | --- |
| 数学核验 | 单次人工CPU已通过 | 原结果/源码哈希复用，未重跑 |
| 实际新源码 | 数学函数 | 暂存完整Q/D/C/Gcov工厂、固定Q两种干预与显式注册分支 |
| 独立意见 | 待审 | A6API gpt-6.2-astra/high返回PASS_WITH_RISK |
| 实验状态 | 未准入 | 候选身份合格；实际适配/原环境资格、成本仍待核，科学筛未准入 |

本轮没有任务数组读取、模型创建/前向、更新、GPU分配、科学fit或新增Macro。真实训练病例/sample-trace不适用：没有运行相应阶段，旧partial记录未重建。源码只在暂存/证据目录，生产core和注册器未替换。

## [已测事实] 独立审查到底核了什么

外部审查使用精确模型ID、原生Responses和high努力档；模型/完成状态/努力回显均核验。ai-model-world元数据请求超时，未据此编造能力排名；模型选择沿用业主明确的code-review顺序。直接网络超时后静态代理返回有效catalog及审查；首模型已给实质意见，因此未另找模型覆盖结论。

审查覆盖实际[数学函数](/docs/evidence/rmea/20261127-ordered-subsequence-review/ordered_subsequence.py)、[完整插件](/docs/evidence/rmea/20261127-ordered-subsequence-review/rmea_ordered_subsequence.py)、注册副本、旧原生helper和上一轮完整checker/结果。它认可递增子集DP与物理路径D的C^-2缩放、完整64维词值/40读出及独立signed-real16经典对照的映射。物理实现仍须说明一致controlled相位、子集准备/擦除、成功率和读出代价，既不是免费oracle，也没有速度结论。[结构化审查](/docs/evidence/rmea/20261127-ordered-subsequence-review/implementation-review.json)。

**审查是供给源码的静态独立意见，不是独立运行。**原checker没有调用新工厂、注册分支、固定Q方法或Gcov，不能把上轮绿色数学检查写成这些新接口已经执行。Q/D81028、C85748、Gcov82308是raw参数数，不是有效容量匹配。

## [机制解释] 两个去相干问题必须分开

1. **路径D：**在相同子集准备/投影过程里去掉子集标签之间的相干，保留路径内量子记忆，质量为sum_path |psi_path><psi_path|/C²。独立训练的D用于效用比较；在同一选定Q参数上调用路径D用于机制干预。
2. **末态双D：**固定Q的相干子集和不变，将最终记忆和query效应在固定计算基去相干，等价于用diag(E)读出。它与路径D不是一回事。

两个固定Q的资格集CE差都须各自>=0.005，不能相加，也不能代替Q对独立D/C/Gcov的主指标优势。统一用原TRAIN类先验重加权CE；本轮只冻结定义，未测CE。固定EPS使去相干既能改变相干项也能改变均匀支路占比，CE差不自动等于EPS=0下形状效应。

## [已测事实] 旧消耗未清零，整段成本已补齐

只读查询原gen128分配，确认released。以原请求创建1795781441.4973128到已归还更新1795781503.8186426的**整个区间**为单GPU保守上界，62.3213298秒，向上预留**62.322 GPU秒**，包含该分配内等待、初始化、CPU参考、任务及归还开销。原更新计时31.1380969秒和wrapper49.9420922秒仍保留，不当整段支出。

原profile字节不变，按独立campaign解释；当前新候选明确分账，旧4技术更新/512暴露仍已耗，新0。新候选未来至多4次技术clone（每臂一次）和2560科学更新；跨两者条件上限为8技术、2568总更新、328704暴露，不是把旧4次恢复成未使用。旧history仍技术PASS/成本DEFER，不获得追加资格测试。

此外采用更严的跨两campaign累计3600GPU秒帽，扣旧整段预留后，新候选总可用上限**3537.678 GPU秒**；其全部技术、训练、检查点评估、两干预、启动/保存/归还和测量开销均须计入。新native campaign总elapsed<=2400秒、最多既有2GPU、每GPU Torch峰值4GiB；闲卡不扩大帽。实测完整预测还需至少10%不确定开销预留。此处是条件资源包，不是已验证可行或当前启动许可。[成本凭据](/docs/evidence/rmea/20261127-ordered-subsequence-review/historical-cost-reserve.json)、[主审裁定](/docs/evidence/rmea/20261127-ordered-subsequence-review/primary-resolution.json)。

## [尚未证明] 执行前仍须通过的实际门

- legacy实际为8192 FIT/4096评估/1024更新/final-only；需要最小任务/阶段适配，不能只改配置冒充4096 FIT/1024选模/2048资格、640步及160/320/480/640四检查点。每臂独立选最佳、同分取最早；Q在480与640两点均须不低于各强控制，混合即defer。资格集只在所有fit/选择冻结后每个选定状态一次。
- group隔离、固定类权重、全40类混淆矩阵、paired group bootstrap500、两次独立fixed-Q调用须对实际适配代码测试和范围审查。新canary必须用新FIT4096固定order首128，而不是冒用旧cohort/首批hash；共享调度/优化器/恢复核心复用。
- 原Torch2.4.1环境须核实际工厂保存初态、Q/D同原始状态、C共享张量和独立存储、Gcov完整keyset/状态及输出。合并字典后strict-load并不能排除Gcov漏覆盖key。
- 已冻结C非退化必要门：首128中n>3有效行至少16；其中至少80%行的mean mass/EPS>=10；逐行class-mean attention TV的中位数>=0.001；C算子和末端梯度norm各>1e-10、routing/all梯度norm比>=1e-8，并检查有限/实际更新。它们只排数值性弱对照，不认证经典最优或有效容量。失败不换人群、阈值、EPS、归一化或seed救援。
- 每臂仅一次discardable clone更新，所有相关前向/梯度/干预检查计成本，不能warmstart。完整四fit、24576常规评估row-forwards、4096额外干预row-forwards及开销的实测预测都在帽内后，才可冻结唯一科学筛选。

详见保留r1原件后生成的[前瞻合同r2](/docs/evidence/rmea/20261127-ordered-subsequence-review/prospective-entry-r2.json)。审查的四项required_fixes有明确解决位置；实际adapter、native检查与成本预测仍PENDING，未伪写成完成。

## 唯一推荐与停止条件

**下一步SEARCH-35：**实现并范围审查上述最小阶段/数据适配，核原环境实际源码/初态，然后仅在前置门通过时执行新候选每臂一次的有界技术clone；原人工数学不重跑。必要数值门失败或成本预测超帽，defer确切实现/比较，不追加技术次数或科学预算，不重启旧serialhistory。只有完整科学小筛通过，才进入完整单seed及预声明独立确认；理论/近邻区别继续保留未解决状态。

原量子L4-T目标ACTIVE，新L3/L4-T NOT_ACCEPTED。门户仅同步Plan41/SEARCH35与第40日志；143IDs/136真实科学行/237指标/历史5120终态和旧39日志保持，不新增科研成绩，不以经典衍生成果替换量子目标。

## [已测事实] 补充：新注册分支已实际导入

原注册器对新ID返回预期未注册，新暂存注册器成功导入四臂插件并暴露两种fixed-Q接口；两个独立子进程均exit0，未创建模型、未任务前向/更新、CUDA未初始化。这只是实际Python路由/依赖导入，不替代factory初态、前向/梯度、控制非退化或阶段适配检查。[字面命令与输出](/docs/evidence/rmea/20261127-ordered-subsequence-review/source-wiring-execution.json)。
