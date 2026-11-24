# 幅度／相位分离：方法、控制和理论边界

每词原prepare线性层与每关系原effect均输出12个tanh有界角。前6角作两层实RY及定向CNOT环，生成实幅度a；后6角作可交换Z0、Z1、Z2、Z0Z1、Z1Z2、Z2Z0相位，生成psi_i=a_i exp(-i f_i·theta/2)。查询同规则生成phi_i=b_i exp(-i f_i·eta/2)，仍为三qubit单位纯态；每关系独立二元Born测量，40个查询不是联合40结果POVM。EPS=1e-6后词维归一化与64维有符号value/decoder保持原样。

在角变量层，概率等于 sum_i a_i² b_i² + 2 sum_i<j a_i a_j b_i b_j cos[(f_i-f_j)·(theta-eta)/2]。这是已知对角量子特征映射的Fourier表示，不是新定理。共同移动key/query的后6角不改变重叠；完全去相干只留下第一项。后者的6*1029+40*6=6414个相位注册参数对读出不变，不能宣称同有效容量。

实数控制使用同一个前6角幅度电路，后6角改为两层RY/CNOT环。量子条件是6RY+3RZ+3RZZ，每个RZZ标准分解为2CNOT+1RZ；在三线完全连接假设下量子/实数均12旋转+12CNOT，每个非uniform模型81028注册参数。控制不保证等函数类或等优化难度。零后6角时，实数新增CNOT是key/query共同的同一置换，因此两种Born读出相同；此为实际fixture之一，而非完整模型所有角下相同。

所有这些恒等式都只在明确的潜在角/状态域成立。tanh前真实词特征的平移未必诱导相同角平移，不能称为文本语义不变性、实际扰动鲁棒性或方法收益。Schuld/Sweke/Meyer arXiv2008.08605v2 pp1–4此前已读，编码决定Fourier频谱是已知近邻；见qattn-l5-round16-20261116/reading-receipts.json。CNM2019复状态/Born测量/实消融已知。尚无实质新颖性清关、非平凡新理论或独立机制证据，所以不接受T。

这轮只检验不同编码下相干贡献的经验假设。4192fit/4000已暴露TRAIN assessment、seed13、每组1024次原CE/AdamW更新，五个原始同协议完成基线复用，三路新训练，全部训练结束再统一评估。主指标all40Macro，必须候选严格胜全部七个控制才准备新完整实验。任何失败不改门、角、rank、初始化、维度、epsilon、预算、seed、控制、checkpoint或指标挽救；短筛选不是全家族不可能。
