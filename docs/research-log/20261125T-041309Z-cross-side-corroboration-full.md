# 跨侧线索印证：五路全量单种子结果

## 实际工作流程与本轮变化

[已测事实] 任务是 Re-TACRED 关系分类：输入句子及指定主语、宾语，由固定的 RoBERTa 特征缓存提供每个词的1024维表示和角色距离；训练可学习的 key、40个 relation query、value 和分类 decoder。模型给每种关系分配词注意力，聚合 value 后输出40类分数，含无关系类。VALID 选择 checkpoint；TEST 报告所选模型的完整分类结果。全40类 Macro-F1 是各类 F1 的等权平均，是本轮唯一主指标；CE 是交叉熵次指标。

本轮让主语侧词的权重乘以它与宾语侧所有词的平均量子态重叠；宾语侧不变，无宾语侧则关闭门控。共享门控与原有 query-key Gaussian 亲和度相乘，逐词加 EPS 后归一化。value/decoder 不变，不加参数或经典残差。

| 项目 | 上一完整数相干光学方案 | 本轮 |
|---|---|---|
| 额外信息 | 单个 query-key 数态核 | 同句宾语侧的态均值 |
| 作用点 | query-key 相似度 | 主语侧共享词先验 |
| 原始工具 | 有限 Fock 叠加核 | density mean / fidelity kernel |
| 核心执行 | 固定训练、队列、选模、COS | 全部复用，仅科学插件/配置改变 |

[已测事实] 先前固定8192/4096小数据筛选七路中，本候选主指标胜全部六对照；固定权重4096条宾语置换诊断也通过两门。其 batch32 首次回放失败是技术无效结果，batch128 重试严格复现父结果后才解释。此次按预声明进入全量单seed，不抹去前两步正证据，也不将它们当作全量保证。

## 冻结设置与结果

五路新训练均 TRAIN58465 / VALID19584 / TEST13418、seed13、20epochs、batch128、原 AdamW，各9140更新，总45700更新；严格最佳VALID全40类Macro选模，平手最早。动态两卡队列执行，9个原生/数相干完整对照只复用，不重训。81028为注册参数数量，不是有效容量等价证明。

既有 worker 在各路完成后评所选状态的TEST；主 agent 按冻结规则等全部五路完成才读取科学效果，无中途改条件/预算/门槛。seed13及TEST已经多次暴露，本轮是开发资格实验，不是独立确认。

| 条件 | 选中轮 | VALID全40宏F1 | TEST全40宏F1 | TEST CE | 候选减该条件 |
|---|---:|---:|---:|---:|---:|
| quantum_corroboration | 15 | 0.600880631 | 0.527658179 | 0.601487806 | — |
| dephased_corroboration | 12 | 0.588731555 | 0.529276830 | 0.600215144 | -0.001618651 |
| normalized_dephased_corroboration | 20 | 0.587227898 | 0.524333529 | 0.665832662 | +0.003324651 |
| gaussian_corroboration | 15 | 0.586097650 | 0.518741936 | 0.614011984 | +0.008916243 |
| gaussian_base | 15 | 0.596403658 | 0.551057980 | 0.582219524 | -0.023399801 |
| native/uniform | 17 | 0.242704894 | 0.221166094 | 1.150371873 | +0.306492085 |
| native/quantum | 17 | 0.579813636 | 0.495323642 | 0.650666541 | +0.032334537 |
| native/dephased | 9 | 0.575895939 | 0.504767888 | 0.587277266 | +0.022890291 |
| native/softmax12 | 19 | 0.586704986 | 0.534296111 | 0.606533580 | -0.006637932 |
| native/real12 | 17 | 0.577730988 | 0.546447526 | 0.570518212 | -0.018789347 |
| optical/number_coherent02 | 17 | 0.595994599 | 0.530943407 | 0.615253624 | -0.003285228 |
| optical/number_mixed02 | 15 | 0.607889786 | 0.542285124 | 0.573055887 | -0.014626944 |
| optical/number_mixed02_purity_normalized | 15 | 0.607111719 | 0.542001062 | 0.573054319 | -0.014342883 |
| optical/curvature_matched_gaussian02 | 15 | 0.596403658 | 0.551057980 | 0.582219515 | -0.023399801 |

**固定13门通过5/13，未通过资格门。关闭此确切插件、分侧规则、mean gate、seed及预算配置，不补seed或重选主指标救援。**

| 对照 | 修正其错误 | 引入错误 | 两者均正确 | 两者均错误 |
|---|---:|---:|---:|---:|
| dephased_corroboration | 508 | 506 | 10348 | 2056 |
| normalized_dephased_corroboration | 558 | 535 | 10298 | 2027 |
| gaussian_corroboration | 515 | 525 | 10341 | 2037 |
| gaussian_base | 404 | 533 | 10452 | 2029 |
| native/uniform | 2898 | 800 | 7958 | 1762 |
| native/quantum | 950 | 765 | 9906 | 1797 |
| native/dephased | 867 | 699 | 9989 | 1863 |
| native/softmax12 | 706 | 729 | 10150 | 1833 |
| native/real12 | 782 | 962 | 10074 | 1600 |
| optical/number_coherent02 | 633 | 699 | 10223 | 1863 |
| optical/number_mixed02 | 588 | 632 | 10268 | 1930 |
| optical/number_mixed02_purity_normalized | 589 | 634 | 10267 | 1928 |
| optical/curvature_matched_gaussian02 | 404 | 533 | 10452 | 2029 |

## 学到了什么

[已测事实] 候选比原生 quantum 高 .032334537，但比 Gaussian base 低 .023399801；raw dephased 又比候选高 .001618651。候选胜 Gaussian pair 的 .008916243 是有用的正差，但两种跨侧门控都低于关闭门控的 Gaussian base，不能解释成跨侧印证已贡献独立增益。

[已测事实] Q 对 raw dephased 是508修正/506损伤，却略输宏F1；对 Gaussian pair 是515修正/525损伤，却胜宏F1。宏F1关注各类别，而正确数关注逐例总量，不能互相替代。对 Gaussian base 是404修正/533损伤，净少129个正确；保留全部40类和全部13组配对，不挑正类重新评级。

[已测事实] 新 Gaussian base 与旧 curvature Gaussian 的所选TEST混淆矩阵相同，Macro相同；CE相差约8.8e-9，两条独立训练/来源记录仍保留，不能据此宣称所有浮点执行或权重完全一致。

[机制解释] 当前 g 是所有relation共享的 kernel-mean affinity，不直接根据各 relation 的 decoded value 或竞争 margin 判断词的用途。因此“与宾语相似”并不等于“对正确关系有用”。这是与负结果相容的解释，不是这轮训练已证实的唯一原因。

[尚未证明] 对象均值稀释、共享先验交比、含EPS的读出协方差等式已做代数核验，但都是已知工具/结构边界，不构成非平凡新T。ConQAR/QMSAN 已有密度矩阵和跨对象/HS注意力；原文阅读只覆盖所列方法页，无新颖性清场或硬件加速证明。见 theory-neighbors/structural-limits.md、readout-criterion.md。

[尚未证明] sample-trace 未物化，本报告不是逐文本语义机制或因果归因。选中checkpoint可回放；不从最终混淆矩阵捏造中间轨迹。一次小筛选的晋级失败也不证明所有小数据分级投入无效，或所有相干/跨侧家族无效。

## 唯一推荐下一步

先做当前 selected Q 的固定参数、全TRAIN gate-on/off 只读诊断，而不是换温度/相位再训练。假设：共享印证门与真正的关系判别 value 不对齐，可能在同一模型内压低正确关系 margin。用既有58465条TRAIN、原batch128、同一checkpoint/source/cache，关闭门控但保留同一Gaussian base、value/decoder，统计全部预测修正/损伤、40类F1、gold-v-max-rival margin，并核对含EPS的精确协方差分解。它与分别训练的 Gaussian base 对比不同。

最低实验：一次双输出固定参数回放，预计既有节点一张GPU不足2分钟、无训练/新seed/TEST/COS上传；仍须调度分配。成功门为完整行数、原Q回放一致与分解残差通过，科学正负全部保留；不能从局部logit提高直接宣布分类贡献。若回放不匹配则技术无效停止，若没有预期干预证据则放弃该原因，不改样本/预算/指标救援。之后才选择真正不同的任务问题与新机制。持续科研授权覆盖这一有界诊断；本页不表示它已运行。

## 执行、存储与成熟度

[已测事实] 运行exit0/进程scope清空/monitor errors0，历时3210.804秒。11个COS节点/21对象/58878373字节，版本、回读和私有索引按既有publisher验证；只收回约1.47MB七文件报告，Windows无权重/特征/逐例数组。调度allocation已可信归还，恢复对象不在本次收尾删除。

按 idea-maturity1，本轮新L3与L4-T均NOT_ACCEPTED；原限定L4与全部历史保留。没有进入多seed确认，不改变原目标。证据：七文件报告、result-analysis.json、class-analysis.json、completed-delivery.json、运行前protocol.md和prospective.json。

[完整七文件报告](../../reports/q_rmea_corroboration/20261125T031939Z_seed13/summary.json)；[全部分析](../evidence/rmea/20261125-cross-side-corroboration-full/analysis.md)。
