# 完整输出一致性四臂：定向VAT未通过冻结门槛

## 研究流程与本轮改变

本项目输入冻结词元特征h、四个主客体角色坐标及有效位置mask；原生三量子位Q-RMEA用可训h→门角映射与40个关系效应计算Born相似度，逐关系沿词元归一化注意力，再将词元value与关系decoder汇成40类logits。D臂用同线路的去相干概率，G臂用既有Gaussian经典路由。训练只在固定TRAIN拟合侧进行，四个固定末态均完成后才统一评估TRAIN句组隔离的assessment侧。主要指标为包含no_relation的全部40类Macro-F1；CE及null误判仅辅助描述。没有新VALID/TEST读取，也没有独立确认或新全量研究。

| 与上一轮gen99相比 | gen99 | 本轮gen100 |
|---|---|---|
| 改变位置 | 关系查询frame读出 | 原生模型的显式训练损失 |
| 不变项 | 词特征、value/decoder、同cohort | 不复用失败frame；恢复原native Q/D/G |
| 区分控制 | D/real12 frame及旧native | 同VAT D/G、Q随机一致性及四旧native |

本轮采用已知VAT：clean输出分布detach；对有效h施加无标签KL扰动，xi=.001、epsilon=.01倍每例有效h范数，定向臂一次power iteration；训练CE+1.0 KL，方向detach。Qrandom同半径/权重/初始随机序列但跳过power。角色不扰动，padding扰动为零，评估clean logits。特征扰动并不保证文本语义不变，VAT不是本项目新理论。

## 冻结设置及资格检查

8192fit/4096assessment/all40/seed13，各1024更新，合计4096。实际dense batch128；历史microbatch_size=4字段不在dense路径生效，与旧参照一致，纠正简称micro4的歧义。原AdamW/FP32-TF32off，共享value及原模型初始化，内部KL/范数FP64；各81028参数，不声称等容量/等算力。每臂600秒fit+assessment、campaign2400秒，四fit先结束才评估。

[已测事实] 关闭损失时四臂loss/全参数梯度/clip+AdamW更新逐位等于原dense CE；独立NumPy stopped-gradient误差4.84e-11，power方向误差0。显式loss callback与objective元数据，无logits隐藏梯度或全局CE猴补。回滚已实际执行并恢复修改版，见execution-verification.json及screen.diff/rollback.ps1。

[已测事实] 实际首8fit克隆的四臂一次更新、全部40行query梯度、padding和扰动范数检查通过，原models及generator未变；最大相对范数误差<4.87e-8，padding误差0。双GPU seed13的8/128/8方向流逐位一致。static及精确runtime-v2 Astra6.2/high PASS，旧runtime BLOCK保留：旧审查包把节录误标为完整design；完整实际设计一直含source/support，不曾借此改科学配置。

本小筛选的实际8例只记录聚合资格，未物化完整sample-trace.v2逐词/逐类/优化器轨迹；不宣称Case Study或案例因果归因。训练、评估阶段实际存在；量子硬件采样和生成阶段不适用。GPU私有diagnostic保存逐例logits，未下载Windows；公开文件只有代码及聚合数值。

## 全部八路结果

| 条件 | 全40Macro-F1 | clean CE |
|---|---:|---:|
| quantum_vat | 0.346787784326 | 0.672876477242 |
| dephased_vat | 0.361381219609 | 0.698345065117 |
| gaussian_vat | 0.302461279959 | 0.719641268253 |
| quantum_random_consistency | 0.412731650388 | 0.756615459919 |
| reused/quantum | 0.389300602574 | 0.750126183033 |
| reused/dephased | 0.472501085293 | 0.707974433899 |
| reused/gaussian_base | 0.484228946800 | 0.705439209938 |
| reused/gaussian_covariance | 0.512054050292 | 0.695091843605 |

[已测事实] 七门1PASS，唯一通过是QVAT胜GVAT。判定 `DEFER_EXACT_OUTPUT_CONSISTENCY_SCREEN`；不晋级全量/多seed，不调epsilon、xi、alpha、power次数、seed、LR、预算或删强对照救援。

[已测事实] QVAT相对原生Q Macro下降而clean CE改善，不能把CE改善改称Macro成功。Qrandom高于原生Q，但仍低于原生D/G及强Gcov，不晋级Qrandom为新胜者。全部三路VAT都低于各自native Macro；这是确切一次性配置的结果，不推定VAT/量子家族整体无效。

[机制解释] 定向平滑和随机噪声控制给出了不同结果，说明该训练干预改变了行为；尚未隔离梯度方向、离流形扰动、类别不均衡或正则强度中的因果因素。不能凭fit-assessment差或无标签扰动就宣称找到了量子过拟合原因。没有理论、语义保持、独立泛化或相干必要性结论。

## 独立重算、预算与收尾

八路40类CM用独立纯Python重算，最大误差1.11e-16；七门逐项一致。四旧row与远端SHA固定report逐字段相同，源码/cohort/批次顺序、预算和先全部fit再评估时间核对通过。四fit4096updates，训练秒合计613.426122，fit+assessment合计619.279688（两lane并行，不等于墙时）。attempt61462ed4d26a4e9896bf4b7277f7524f，gen100 allocation09b6a1e终态exit0/两进程消失、可信归还。0新VALID/TEST/COS，0Windows模型/特征/逐例数组。

## 评级与唯一下一步

idea-maturity1：新L3/L4-T仍NOT_ACCEPTED。原限定L4及gen98/gen99和全部正负历史保留。已知VAT不是新T，即使未来有正效用也仍须同核心完整L3、实质近邻区别、非平凡正确理论、可区分预测、实现一致性和独立审核；不以更好的经典衍生替换量子目标。

唯一推荐：读取已审计概念近邻2211.05225v2所引用的原始Quantum Embeddings for Machine Learning（2001.03622），核对任务特定量子embedding的训练目标与决策规则，并与已关闭sharedquery/Fisher/frame记录作公式级排重。假设只是“可能存在未试过的共享关系几何约束”，未选择新插件。最小检查是原文范围阅读、明确转移条件和可区分预测；0GPU/0更新。若仅得到普通metric learning、已知后选择恒等式、等价重参数化或关闭配置的参数救援，则停止该方向、不分配GPU。已有持续科研授权覆盖此只读检查；未知是否能形成合格机制或正结果。

证据：[冻结协议](protocol-draft.md)、[完整设计](design.json)、[实际资格](fixtures.json)、[全部结果](summary.json)、[独立重算](result-verification.json)、[全部类别比较](postmortem.json)、[下一步边界](next-direction.json)。
