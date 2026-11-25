# 关系查询联合框架测量：固定三路小筛选未通过

## 实际任务与干预

从既有冻结RoBERTa词特征及主客体角色预测40类关系，包含no_relation。保持词态不滤波；由全部40关系查询构造S及E_r=S^(-1/2)(phi_r phi_r†+EPS I)S^(-1/2)，沿词归一化Born质量后池化原signed values。区别于旧按词scalar competition及word-frame，不重开这些旧实验。

三臂为量子、对F与S一致取对角的去相干、12维实数归一向量框架。各81028注册参数不等于有效容量或算力相同。FP32训练参数、frame内部FP64/complex128如实披露。平方根测量为已知工具，没有迁移其集合判别保证或声称新T。

## 固定设置及全部结果

同TRAIN8192fit/4096assessment/40类/句子组无重叠、seed13、每臂1024更新、b128/micro4、原AdamW；三fit全结束才最终评估，四旧参照复用。未读取新VALID/TEST或写COS，未下载模型、特征或逐例数组到Windows。

| 条件 | all40 Macro-F1 | CE |
|---|---:|---:|
| quantum_relation_frame | 0.224937200 | 0.958279 |
| dephased_relation_frame | 0.246706471 | 0.913766 |
| real12_relation_frame | 0.331089897 | 0.801166 |
| reused/quantum | 0.389300603 | 0.750126 |
| reused/dephased | 0.472501085 | 0.707974 |
| reused/gaussian_base | 0.484228947 | 0.705439 |
| reused/gaussian_covariance | 0.512054050 | 0.695092 |

[已测事实] 六门0PASS，DEFER_EXACT_RELATION_FRAME_SCREEN。三fit3072更新，运行189.958秒。全部七路40类CM/F1/support保留，独立重算误差1.11e-16。原生Q .389300603仍高于frameQ .224937200；不能将表中的不同控制换成主结果。

## 验证与边界

实际first8fit克隆检查全部三臂通过，全部40查询梯度有限非零；不改原模型。静态首轮BLOCK两项与runtime配置文字BLOCK均保留并在执行前修复，精确复审PASS。稳定screen50ce858f/host238f7e68不变。当前attempt4c68f5f227544b31aba87a712c380467进程已退出、exit0、终态封印/源码/数据及预算身份通过；gen99 allocatione0d075bf已可信归还。

[机制解释] 联合frame改变了查询几何与优化耦合，但此次成绩未支持该配置的效用。没有任务中间轨迹证明是白化、条件数、标签竞争或相干造成失败。量子可模拟性也不是本次淘汰理由。

[尚未证明] 新L3/L4-T、理论创新或硬件收益；暴露TRAIN小筛选不是独立确认。当前未产出新的文本sample-trace.v2/训练CaseStudy，不从CM反推真实案例。原限定L4和全部正负历史保留。

## 处置与唯一下一步

暂缓本确切EPS/查询线路/框架/初始化/1024配置，不扫描正则、秩、query scale、LR、seed、budget、cohort或删除对照救援；不证明整个联合测量家族不可能。下一步完成本轮代码/聚合结果及中文日志的Git发布收尾，再依据既有完整错误证据选择实质不同的任务机制。当前没有后继实验、排队任务或已选新插件；文档发布待完成，原L4-T目标继续active。

[完整代码、协议与聚合结果](../evidence/rmea/20261125-relation-query-frame/summary.json)。
