# Q-RMEA 配对混叠预测：真实数据检验未通过

gen105 完成既定4096例TRAIN assessment四路无更新重放，按此前冻结门槛停止这个确切预测。
主统计未超过.55，且低于最强指定诊断控制；未选出新量子方法或取得新L3/L4-T。

| 无标签诊断分数 | 主人群AUC |
|---|---:|
| S：Q与强Gcov的输入/词态pair距离关联之差 | 0.509391424620 |
| 同Q词态去相位后与Gcov之差 | 0.513609958506 |
| 原词数 | 0.541583679115 |
| Q平均off-diagonal fidelity/标量纯度代理 | 0.524882434302 |

相对最强指定控制的优势为−0.032192254495，冻结要求为至少+0.02。
541个Qdistinct/Gcov discordants中，Gcov独对300、Qdistinct独对241；另有共同正确2979、共同错误576。
主统计和所有控制均无undefined，全40类及固定长度35两组的80个表项完整报告。
四门仅技术与样本量两门通过，不按子组、距离、核、阈值、seed或控制选择救援。
精确独立pair计数给出主AUC36829/72300，优势−931/28920，见[independent-verification.json](../evidence/rmea/20261126-context-pair-diagnostic/independent-verification.json)。
这仅停止当前代理距离/核/人群/方向下的观察性预测，不证明所有量子表示问题都不存在。

本轮使用既有完成物：gen98 Qdistinct/Ddistinct/Gdistinct及gen91最强声明Gcov final checkpoints。
共48个小文件/源码/checkpoint哈希先行匹配，随后分配内验证全部50个输入，包括完整4.43GB h缓存。
四路封存预测全部重现，argmax差异0、最大scaled logit误差2.2309e−5、40类Macro误差0：
Q .500925142309、D .436654900537、Gdistinct .510930539751、Gcov .512054050292。
保留81924对82308参数与结构差异，没有替换强控制或把它当参数严格等同。

原状态和checkpoint字节保持，所有512次head及512次batched circuit调用完成，0updates，耗时11.292636秒。
exact real16 pair核最大差1.11e−15，四个固定首例的另一路NumPy全pair核/距离/统计误差2.11e−15。
同一agent的不同实现不算外部审核。私有population及四例完整cached-input/kernel/logit轨迹留服务器，
上游contextual encoder与训练轨迹缺失范围保留，未伪称端到端自然语言或训练过程审计。
全部数值与四路40类混淆矩阵见[summary.json](../evidence/rmea/20261126-context-pair-diagnostic/summary.json)、[verification.json](../evidence/rmea/20261126-context-pair-diagnostic/verification.json)。

共享调度gen105分配已可信归还，原owner PID已不存在；无新容量、COS对象或model API。
首次收集把封存LF文本写成Windows换行导致本地字节hash失败；原失败回执/转换副本保留，
仅将write_text改为write_bytes后重新读取同一封存结果，未重跑任务、改门槛或修改任何模型。
见[collection-repair.diff](collection-repair.diff)、[resource-verification.json](../evidence/rmea/20261126-context-pair-diagnostic/resource-verification.json)。
执行配置的LF版本及原冻结版本均保留，JSON语义逐字段相同；三个科学源码字节哈希与实际执行完全一致。
实际命令、literal输出及exit0见[执行核验](../evidence/rmea/20261126-context-pair-diagnostic/execution-verification.json)，不存在按结果改配置。

下一步只做一次明确的源级范围检查：gen102固定teacher value替换失败，是否覆盖共同训练词态与bounded observable utility。
这不是有效性证据；须先排除已做Q-BAM、connected-value和operator-valued定义，
给出真正不同的机制/匹配强经典控制/可证伪预测，否则记录NONE，不因“联合学习”字样重开旧profile。
原entity-conditioned query完整研究也已因强经典控制失败关闭，不再重新命名尝试。
下一步源码pins和0GPU/0taskforward预算见[next-direction.json](../evidence/rmea/20261126-context-pair-diagnostic/next-direction.json)。

原限定L4与全部正负历史、gen98/gen102/gen104及新增gen105 STOP保持；新L3/L4-T未接受，原量子目标ACTIVE。
