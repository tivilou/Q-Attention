# R95：效用耦合测量的固定模型探索筛选

复用原始 train 前4096条缓存中既定972条 evaluation、866个实体对组，以及
R81 native_quantum-final.pt（4 epoch固定权重）。不训练，不触碰正式 valid/test。
所有条件同一权重、同一输入、batch4、float64/complex128。FP32历史聚合门不在本轮修改。
只读取已在节点的固定输入；不恢复 COS、不复制原始数据/权重，不新增 checkpoint。

比较原生 Born、完整效用更新、单纯形效用倾斜、共享尺度 pinching、仅测量构造路径
置换效用。eta=1与尺度固定；没有搜索。主指标 positive39 Macro-F1，CE为次指标。
full的Macro delta>0且CE delta<=0仅记录探索点估计正向，否则固定checkpoint路径不正向。
全部对照和伤害均保留；不以对照输赢改写原生utility结论，也不从本轮授予新等级。

本组样本已多次探索且缺6类；任何好结果都还需要独立、适当完整的证据。
固定权重的负结果只关闭直接postfit更新推进，不证明联合训练结构普遍无效。
但它也不自动授权追加训练、种子、系数或预算来救本次路径。

真正运行前，精确源码/启动/输出/资源/重试合同需独立BOTH审核，再走共享CPU调度。
零更新须在每批以2e-10绝对容差复现同批原生注意力与logits；不改旧门。
检查非有限值、padding、归一化及逐类own-score gain数值界，失败即保留该单次失败。
这不是Case Study、float32/GPU训练验收、最近工作新颖性结论或L5验收。
