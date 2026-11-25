# 完整输出一致性：不同于frame/ansatz的训练假设（草案）

## 动机及已知边界

已审计gen95中native Q在fit Macro.806而assessment.389，D为.787/.473；这是描述，不证明敏感性或过拟合是原因。gen99 frame六门0PASS不重开。测试新训练目标是否改善原生量子映射相对自身及公平经典/去相干的任务效用，不改量子线路或frame参数。

依据Miyato等1704.03976v2原文pp3–5 Eq5–15：detach原输入40类softmax分布，用局部KL变化构造不含标签的扰动，再训练有标签CE+一致性KL。此原语为已知VAT，不是新T。Caro等2111.05292v2固定非训练encoding定理不直接适用可训h→angles，故不声称按12门得到泛化保证。后续若实证成功仍需独立实质理论/近邻审核，不以通用正则替代原L4-T目标。

## 拟议机制与公平条件

冻结h和角色输入来源，训练时仅扰动有效token的h，角色及padding不扰动。epsilon=0.01倍当前样本有效h整体L2范数，xi=0.001倍同范数；这些为一次性待检查设计值，不声称语义不变。每样本单位L2随机方向，零范数方向保留零并记录；VAT一次power iteration，方向及clean目标detach。用FP64 logsoftmax/KL累计数值量，训练参数仍FP32；评估无噪声、无正则、原始logits。

四新臂：native Q+VAT、native D+VAT、原生Gaussian+VAT、native Q+随机方向KL（与VAT相同epsilon、权重、初始随机方向；不做power）。全部alpha=1、同信息/训练目标定义，分别生成各模型最敏感方向，不用标签选方向，不把相同注册参数称等算力，额外前后向及时间如实报告。随机方向控制用于区分定向敏感性与普通特征噪声；三种模型均获得同级优化机会。

延用既有TRAIN8192fit/4096assessment、句子组隔离/全部40类、seed13、各1024更新、b128/micro4、原AdamW与共享初始化/value/批次顺序。四fit共4096更新，按每臂600训练秒/最多2400训练秒设计；最终实际预算与运行身份在实现审核后另行冻结。四fit全部结束才统一末态评估，无新增VALID/TEST。QVAT须严格胜Qrandom、DVAT、GVAT以及原native Q/D/Gbase/Gcov四强参照，共七门；通过只允许设计完整单seed，失败暂缓确切扰动/权重/算法预算配置，不扫eps/xi/alpha/seed/budget/control救援。不把本开发筛选当独立确认。

## 实现入口及检查

已只读核验rmea_fair_reference_runtime.backward/rmea_task_runtime.backward确实为原始CE路径，无额外loss回调。拟仅在screen的科学backward分派增加显式可选plugin.backward接口；无callback时完全沿用旧函数，不修改原baseline模块，不猴补全局F.cross_entropy，不伪造logits承载KL。已有core/host/数据/调度/保存语义复用，不做生命周期矩阵。新hook需独立范围审核及关闭时输出/全参梯度/实际更新一致性。

其余最低检查：完整loss的独立KL与梯度、alpha=0端点、clean target/direction停止梯度、padding/随机方向mask范数、每样本扰动界、方向分离、非零power梯度及全参数finite、克隆真实8例；任何资格失败在科学执行前停止。不按结果调eps/xi或换统计口径救援。

## 实现核验更新

已实现显式可选backward与native Q/D/G + Qrandom四臂，尚无任务执行/分配。CPU合成测试确认四臂alpha=0损失/全参梯度/clip+AdamW更新逐位等于原dense CE；独立NumPy固定target/direction全loss数值梯度误差4.84e-11；target梯度为空、power步骤不积累参数梯度、方向detach、padding/范数/零方向分支通过。独立审核及真实8例仍待完成。

实际执行精度与批次必须从源码解释：继承dense_batches=true时每次使用完整128例，microbatch_size=4仅为保留旧字段，未参与dense路径；本轮与四旧参照保持该实际行为一致，不声称micro4。模型为FP32，KL/logsoftmax和范数为FP64；每臂600秒沿用既有fit+assessment预算语义。四臂每个模型私有generator(seed13)，避免两个GPU线程竞争全局随机数；QVAT/Qrandom初始方向序列相同，VAT另作一次power。没有新T、任务提升或L3接受声明。
