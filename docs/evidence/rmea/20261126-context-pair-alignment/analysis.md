# Q-RMEA 长context混叠假设：源级排重与诊断限定

本轮保留一个暂定、可直接计算的输入/词态配对诊断，没有选定新插件或建立新T。
六个指定源码定义及依赖已核验18个hash，见[source-exclusion.json](source-exclusion.json)。
去同位置乘积改value二阶读出，词帧与关系帧改Gram白化，Fisher改查询步方向，
semantic sharing改query坐标；本诊断不改它们的参数、输出或任何旧筛选门槛。
另纠正两处范围：原context-frame固定记录是mixed Float/Double的exit1，非有效负筛选；
native shared-query是全TRAIN零更新梯度诊断，非另一条训练方法。
这只排除与指定六个定义的直接同一性，未做全领域新颖性清关。

唯一主统计 S 是Q词态相似 pair 内的平均 contextual-input 距离与既定强Gaussian相应量之差，
再按冻结定义处理所有词对。它不依赖gold、关系query或选定长度子群。
原gen98短句正确差+1/长句−50仅是动机，仍有类混合、任务readout等混杂，未证明混叠原因。

[两个固定人工例](algebra-trace.json)的 native prepare 角、词态、密度纯度、
输入距离多重集合、谱、alpha、value和全部40输出都按构造一致；
统计AQ分别−1/3和1/6，差1/2，Gaussian对应量相等。
它确实不是纯度/长度改名，但输出仍完全相同，所以它也不足以证明可修复任务错误。
[推导](derivation.md)给出精确分数和23/50正定下界；
[核验](verification.json)重开完整输入，用单独标量实现与Fraction检查。
这是同一agent的独立实现核验，未做外部审核。

[冻结预测](hypothesis.json)：若后续现有checkpoint/缓存/封存预测精确绑定通过，
只在原gen98已暴露4096 TRAIN assessment做一次无更新诊断。
预定高S应辨别Gcov独对/Qdistinct独错，主AUC至少.55并超长度、纯度及同Q去相位控制中最强者至少.02；
所有40类、两长度段及失败/undefined计数完整保留，任何失败即停止这个确切预测。
未执行该真实任务检验；不把人工例当自然任务证据，也不把gen98附加readout当原完整Q因果。
强Gcov的参数/结构差异与旧数据暴露限制保留。real16为同函数技术对照，非独立训练优势证据。

本轮CPU人工代数.015625秒，0新task/native/circuit/GPU/optimizer/modelAPI/server/COS或购量。
定义脚本保留字语法错误及其下游缺文件两份exit1回执已保存，修正仅字段名；最终定义和审计exit0。
原限定L4、全部正负历史、gen98/gen102/gen104 STOP保持；新L3/L4-T NOT_ACCEPTED，持续目标ACTIVE。

唯一下一步是定位并核验现有gen98四个Q/D/Gdistinct与最强Gcov实际final checkpoint、
输入和封存预测映射，冻结一次无更新诊断的确切调用/资源上限。
若缺失或身份不符，记录技术未准入，不重训/重启旧配置补齐，也不删强对照。
