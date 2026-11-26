# 本轮审查采纳范围

审查gpt-6.2-astra/high仅PASS_WITH_RISK，候选PROVISIONAL。允许下面固定的无数据数学检验，不准入task plugin、任务forward或训练。不改变冻结hypothesis.md或被审代码；本文件补充其真实有效容量和测试端点。

精确写W2=C (tensor RZ(z2)RY(y2))，有<0|C=<0|。故<0|W2是exp(-i sum z2/2)乘仅依赖RY的实测试向量。p_mid Q/D对末段三个词RZ严格无关，prepare对应三行从route得到零任务梯度（weight decay仍能移动参数）。D在W1后去相干，也消去首段三个词RZ，词角依赖最多6个RY；Q至多9个词角方向。名义12角和相同registered参数不能掩盖这项差别。class完整Q矩阵又可能激活原末端纯query未使用的方向。没有声称有效容量相等。

原生zero-angle块是C，zero-angle两层query是C²；不把它们写成I。固定commuting端点用W2角全0、Q角全0，允许W1变化：C与(C²)†交换。另用全角0验证返回0态。raw h=0和preparer有bias时不是同一个端点。

按审查给定构造，W1两词均角0；W2仅wire0分别(y,z)=(pi/2,pi/2)、(-pi/2,-pi/2)。row0 query仅首层wire0 RY=-pi/2，其余角0。先验证最终rho相同，再验证p_mid=(0,1)、D=(.5,.5)和原概率逐类相同。utilities row0=(+1,-1)、其余39行0，bias0，验证native logits全0、mid row0=-1/(1+2EPS)及all40 softmax差。固定input h=e0/e1可由prepare两列atanh(theta/pi)生成这些角，value第0通道h0-h1、decoder row0选择该通道；这是人工可达性，不是cached encoder或训练checkpoint可达性，更不是自由encoder/decoder的family分离。

另外32组角固定为sin/cos确定序列，不按输出选择：word=1.2*sin((index+1)*j*.173)，query=.9*cos((index+2)*j*.137)，j=1..12。独立原生Torch circuit/保留layer_matrix代码、NumPy dense链、complex amplitude、real16、归一vec和显式density-dephase核对。所有代数残差冻结1e-11，不改EPS或容差；数值失配保留失败证据后仅技术修复。

必须继续披露：固定preparer+utility反例不能排除原生全重参数化；process/Choi lift、非交换和已知门结构不是新T。任务控制的real8层顺序/角映射、Q/D初始化与旧强参考完整身份仍待下一步实际plugin规格确定，当前不训练。所有旧STOP及原L4/完整目标保留。
