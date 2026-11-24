# 固定角预算下的寄存器分配：七路 TRAIN 可行性试验

动机：当前40个关系的原生3qubit/8维query测量头，在完整比较中整体低于real12，且差距分散31类。Welch/frame恒等式给任意40个8维单位query的平均非对角平方重叠下界(40−8)/(8×39)=0.102564。它只限制query packing，不证明分类需要40个正交态，不是已发现错误原因或分类误差下界；readout还使用value/bias，词注意力也另有归一化。

新假设：同样12个输入角/关系query坐标，改用6个各RY/RZ的qubit乘积态，能够给测量分配更大空间，可能改善关系路由。原3qubit两层和新6qubit一层的参数数均81028，prepare仍1028→12、effect仍40×2×3×2存储但flatten成6×2解释；不增参数/新输入/预训练。原生初始prepare/effect/value/decoder逐位相同。新方法不是给已关闭的K2/循环/零gate组合调参，所有原配置不动。

单qubit态s(y,z)=(exp(−iz/2)cos(y/2),exp(iz/2)sin(y/2))。Q6 raw probability为六个|<s_rk|s_tk>|²之积。D6在完整6qubit计算基上去相干后重叠，等于六个sum_b|s_rkb|²|s_tkb|²之积。共同末端CNOT置换对两种重叠均消去，生产不模拟64维state。正常词权重仍(p+1e−6)/sum_valid(p+1e−6)，原value/decoder/bias不变。无跨word相干/记忆/实体条件query、signed readout、额外gate或优化器。

固定七路：quantum_native（原3q），quantum_register6，dephased_register6，real_product6（原12维线性key/query拆为6对，每对单位实向量重叠平方后相乘），rbf12（原12维线性key/query，score=−mean((key−query)²)/2后masked softmax），real12，softmax12。所有同原h/roles/shared初始化/数据/优化预算、81028注册参数；Q/D共用完全相同初始化。D6的6414个RZ相关标量解析不影响概率，生产直接用RY概率，避免相位舍入噪声；D-only差距不能单独证明量子贡献。经典参数化及不变量也不同，不宣称有效容量相同。RBF与实乘积控制防止把乘法非线性/更大核特征空间本身归因于量子特有。Q6等价于经典可高效计算的Bloch球乘积核，不声称硬件优势或量子不可模拟。

复用稳定screen v6、原整句TRAIN8192→4192fit/4000assessment、全部40类、seed13、b128、FP32、原AdamW/clip1。每路512更新，共3584，两张既有GPU动态队列，全部拟合完成后仅final512统一评估一次。总预算1800秒、每路600秒。固定主指标all40 Macro：Q6必须严格高于其余六路，六门全过才允许准备新的完整独立协议；CE/逐类/修复损害/耗时同时报告为次指标。本轮选择与用户当前成熟度目标一致的主指标，不倒改任何先前闭合实验的双指标门。

任何主门失败关闭本确切6qubit×1层/乘积/eps/12角/512配置，不改qubit数、角分配、深度、Born幂、温度、eps、初始化、LR、seed、split、预算、metric或控制救援；不是整个量子寄存器家族不可能。短TRAIN不能接受独立L3/T，过门也仅进入完整协议准备。

只核对新概率分解与梯度、独立NumPy64维tensor态及共同CNOT置换、固定有限参数40态正交构造、native3q packing下界、mask/归一化/参数和初始化、每路克隆first8真实一步更新。有限误差需直接拒绝NaN，避免max掩盖；不污染CUDA随机态。core/调度/best-latest/COS不改，不重做能力矩阵。私有逐例数组和短状态仅在授权GPU，本机/Git只紧凑代码/聚合；零新VALID/TEST/COS。

理论边界：packing下界和乘积态计算均为已知数学，不是T。64维允许40个正交态（六个RY角取±π/2，有限effect=atanh(±1/2)）不证明现有文本能学习这些编码，也不保证泛化；纯乘积核会有相似性集中、EPS平滑和梯度乘积风险。本轮检验任务效用，理论创新仍待同核心新结果和实质近邻区别。
