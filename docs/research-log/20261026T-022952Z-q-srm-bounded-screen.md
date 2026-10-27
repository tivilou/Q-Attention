# 20261026T-022952Z | q-srm-bounded-screen | Q-SRM 五 seed 筛选未晋级

时间为run目录记录的启动时间，非推测的结束时间。提交于2026-10-26；本日志对应报告`0206a48`。

## 项目工作流总览

输入是带主客体实体跨度的Re-TACRED句子。先按固定选择规则保留630/631/587条train/valid/test，词表和关系标签映射只由train建立；输入是token、位置、padding/主客体mask。模型是两层32维、4头、FF64、无dropout的Transformer，从头完成8轮factual CE warmup，以验证集Macro-F1选共同checkpoint。

各seed的五个条件从相同checkpoint开始。冻结词嵌入、主干及Q/K/V，继续训练分类头；有插件时再训练144个共享线路参数。第二层查询和键经3比特线路编码为8维态；构造`S=Σkk†+0.01I`及逆平方根，以`|k†S^(-1/2)q|²`加权V，保留零值null且不重归一化token。拼接主客体/全句池化表示，经分类头给出40类关系logits。

仅train标签进入训练CE，6轮、每条件240步；valid仅选checkpoint。test只对选定checkpoint评估（固定案例replay不更新模型）。输出关系类别、混淆矩阵与Macro-F1；后者等权平均各关系F1，用于预先约定的配对效用门槛。

## 一句话结论

[已测事实] 测试均值disabled0.051639758、Q-SRM0.051309126，差-0.000330632（相对-0.6403%），2/5seed正向，未通过预设门槛。保持短预算inconclusive，不晋级全量；不是SRM家族的正式否定。

## 想法与上一轮的变化

Q-CFRA以反事实fidelity重分配attention；Q-SRM改为由整个键集合构造square-root measurement，保留显式null，本轮所有条件统一factual CE。上一轮三视图损失或历史disabled结果不能充当本轮配对对照。预期机制是集合条件化改善证据竞争，[尚未证明] 当前测试结果未提供稳定收益。

## 验证与结果

冻结配置、精确source哈希、环境、数据与比较门槛见[审计](../audits/20261026-q-srm-bounded-screen.md)。seeds13/29/53/71/97，五模式。平均差和至少4/5正向两项失败，SD一项通过；不能挑seed13的正结果代表整批。

| 条件 | 测试平均Macro-F1 |
| --- | ---: |
| disabled | 0.051639758 |
| quantum | 0.051309126 |
| fidelity | 0.050641578 |
| diagonal | 0.052041424 |
| key_mismatch | 0.049958605 |

## Case Study与完整性

固定规则为每个split首个已选记录的initial/best/final，以及train真实update0。生产端报告250份`sample-trace.v2`及9050数组通过检查；接收端只有验证摘要，未获得固定sample ID、真实句子、张量或checkpoint。训练、推理和批量正/负机制案例均 **未提供可读artifact**；trace审计partial。不能用最终指标逆推样本过程。独立idea解释及门户真实案例/backlink仍pending，本条是带缺口的审计日志，不声称完整文档closeout。

## 学到什么与证据判断

[已测事实] 报告自动交付成功，七文件和source哈希一致，50个混淆矩阵可复算。验证集平均有收益而测试没有，当前收益不稳定。

[机制解释] 这提示泛化或干预分配尚需诊断，但尚不能归因于训练不足、null、代码bug或硬件环境。diagonal略高是对照事实，不等于找到可直接晋级的新候选。

[尚未证明] 自然任务效用、注意力对齐、机制独特性、量子优势均未建立。当前L0 diagnostic；五seed不自动构成L2；精确经典等价限制量子归因，不是本轮L0的理由，负主指标才阻断L1。

## 下一步（只推荐一个方向）

先审读已有warmup指标和固定配对trace。假设是能定位验证收益未转化测试收益的具体机制问题；最小工作0新增训练步，只读已有工件。成功要求source/config/fingerprint一致且现象可重放；停止条件为缺工件或仍无法区分假设，此时不追加全量/seed或改门槛。导出时间和字节量未测；具体合作者导出、代码或新实验待另行安排。本轮只审计，未训练。

证据：[原始report](../../reports/q_srm_screen/20261026T022952Z_seed13/summary.json)、[审计](../audits/20261026-q-srm-bounded-screen.md)、[evidence card](../evidence/q-srm/20261026-bounded-screen/evidence-card.json)。
