# R96：效用耦合测量的同预算联合训练筛选

R95关闭的是“旧模型训练完成后，直接追加效用更新”的路线；它没有优化新计算图。
本轮仅测试另一个明确假设：让准备态、查询态、value与decoder从一开始通过该更新
共同接受原始分类CE训练后，是否比同初始参数的原生读出更有效。

三个条件native/full/simplex均从seed20261118的同一原生初始化开始，统一FP64/complex128，
共享既定3124fit/972eval实体对分组和四个epoch排列。AdamW lr.001/weight_decay.01、
betas(.9,.999)/eps1e-8、batch32、clip1、4epochs；各392步，总1176步，不搜索。
原生FP64对照是新计算图检验所需的匹配对照，不替换或修复已有FP32历史baseline。

仅fit标签用于CE梯度；attention推断从不接受标签。首次fitbatch先检查完整梯度而不更新，
清空梯度后按固定顺序训练。只评估各final4，不挑epoch，不以eval改变训练或预算。
主指标为full-native的positive39 Macro差；任何正点估计保留，CE和simplex比较单列。
本轮好结果仍只是重复使用训练开发分区的探索，不能授予正式utility/独立确认或L5。

私有仅保存3份final state用于真实weights_only重载与第一evalbatch逐值回放，合计<=4MiB。
没有每样本预测/中间状态归档，没有COS请求，也不声称GPU实例掉线可续训或模型已受云保护。
失败/停止探索按现有Git代码与紧凑报告保留原则处理，清理旧字节另行授权。

精确BOTH审核后，统一调度CPU1/RAM8192/GPU0/1800秒，单attempt/no replay。
任何非有限值、梯度/输入/回放身份失败终止，保留现场；不额外seed/epoch/系数救结果。
完整正式实验仍只由师弟按独立前瞻合同执行，本轮不访问师弟服务器。
