# 逐例可达与共享query学习：固定完整TRAIN局部诊断

对象为已经完成的原生Q/real12，共同TRAIN58465/全部40类，各seed13初始和原VALID-selected epoch17，共四个状态；同源cache、标签、原checkpoints保持不变。不是重训baseline，不读新VALID/TEST，不应用任何梯度或优化器，不选新checkpoint/类/样本/阈值。每路两状态串行、Q/real两GPU并行，复用既有host wrapper/分配及进度接口。

固定问题：逐例自由query可以改变大多数错例，是否在当前真实共享480个query坐标里存在明显的样本/类别学习信号抵消？所有样本都计入，包括原正确/错误、null/正及全部40类。不是检验另一种query几何优化器，不重开Fisher/MGDA/类平衡/curriculum/全量失败配置。

对每例i/关系r，p为原Born或real归一化重叠平方，a=(p+EPS)/sum_t(p+EPS)，u为原线性value/decoder工具值。导数为 d z_ir/d theta_rj =sum_t a_irt (u_irt−sum_s a_irs u_irs) d p_irt/d theta_rj /(p_irt+EPS)。乘以原native logits的CE系数softmax(z)_r−1[r=y_i]，得到每例480维query-CE梯度。Q的dq由Pauli旋转参数平移pi/2因子加tanh链（state角平移pi）计算；real用球面归一化导数/原epsilon分支。词归一化及中心化必需，不能使用未中心化的value原点梯度范数。

原生logits/p/key/u以原FP32计算，导数和聚合FP64；这是浮点原生量处的实算术链式近似，非机器舍入导数或全局不变量。first3真实样本对每例全480坐标原autograd做FP64误差<1e-8、FP32缩放误差<2e-4；同时核对native readout/中心化常数不变性，实际误差全报告。共享fixture不更新模型，状态身份前后应相同。

主报告固定：全体/每个gold类/原正确/原错误的平均梯度范数、平方范数均值、平均向量平方范数及coherence=||mean(g)||²/mean(||g||²)（零分母报告0）；全部40类均值间余弦矩阵（零范数余弦报告0）、每类均值对原样本加权总均值/等类均值的点积。另给所有样本对这两个均值方向的正/负/零点积数量与分位数，不按结果挑样本。原生selected完整TRAIN指标/混淆矩阵必须复原已固定结果；初始仅描述。

这些坐标统计可揭示局部抵消，不能证明同一更新可修复模型、共同下降不存在、失败因果、全局不可学、最佳训练方向、预期Macro收益或新理论。CE不是Macro-F1，类别均值也不是所有个例约束。Q/real参数坐标不同，参数数相同不使原始范数跨模型可比；不据大小宣称量子固有困难。常见正负梯度冲突和NTK思想是已知工具，不算T。

固定600秒总计算预算，每状态180秒；所有四状态/全部40类完整才分析，不基于半程选择。输出只聚合JSON/fixture/完成标记，原生gradients仅GPU节点进程内存，不写逐例二进制或下载到Windows；零训练/heldout/COS。失败保留现场、先看现有attempt，禁止因断连或计时不明重启。
