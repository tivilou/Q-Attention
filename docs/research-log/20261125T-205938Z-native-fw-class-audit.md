# 原量子核心与实际 FastWeights：旧相位构造不足以排除近邻

原Q-RMEA在冻结词特征/角色特征上制备3量子位、2层线路的词态和40个查询态，用重叠概率加EPS归一化为词权重，再池化共同的64维经典value输出40类分数。目标仍是同一量子核心取得有公平经典/关键消融支持的L3，并满足实质近邻区别、非平凡新T、可区分预测、实现一致性与独立审核的L4-T。本轮只审计函数表示和人工例，不训练、不访问新的任务样本/标签/模型checkpoint。

上一轮真实TRAIN固定C方向诊断已按冻结门STOP；本轮没有执行该查询更新。以前的人工构造使用真实原生线路可达态，原生Q在两个context的输出均0；历史guided0/.9640来自已停方案，不能写成原量子分类器的优势。本轮复用其已保存态，检查实际FastWeights是否同样可分，及一般复数态的精确经典控制维数。

| 对象 | 上一步范围 | 本轮新增核验 |
| --- | --- | --- |
| 近邻模型 | 已读Ba原文，但旧matched-alpha例没有排除实际FW | 完整非负hidden历史、LN/ReLU/持续输入的2写1读轨迹 |
| 经典表示 | real8只在人工实态例精确 | 一般复数态real16双内积/Hermitian64，保留EPS |
| 任务方法 | gen104冻结门STOP | 没有新方法准入，只有源绑定边界结果 |

## 实际轨迹案例：先说明观察了什么

[已测事实] 原文使用已缓存的官方会议版Using Fast Weights to Attend to the Recent Past，PDF SHA028b1b47…，本轮重读第3/4页Eq1–5并视觉复核第4页；此前完整读文收据保留。其A=lambda Aold+eta hh^T；先h0=ReLU(b)，再h1=ReLU(LN(b+A h0))，b在内循环持续存在。此处S=1、eta=.5、lambda=.95、W=0、C=I、LN bias0、共同gain=sqrt(1+1e-5)/sqrt7；LN稳定常数1e-5是本轮明确的数值实例，不冒称论文默认值。不是上游代码复现或训练性能重现。

两个预声明人工序列从零memory开始：依照同一公式先写入e0，第二步分别写入e2或e0；为生成指定非负hidden而代数求出不同写入输入。第三步共同输入q=(e0+e2)/sqrt2。所有慢权重/增益/衰减/学习率/读输入均一致，完整过程实际运行；没有把带负分量的量子态直接塞进ReLU历史。固定Hadamard把旧native plus/minus变到e0/e2，query同变；它是一个共同经典表示映射，不声称整个原生ansatz对该基变换闭合。

| 固定人工序列 | 实际写入hidden | 读取前memory | 共同读出h2-h0 |
| --- | --- | --- | ---: |
| distinct | e0,e2 | .5(.95 e0e0T+e2e2T) | 0.014669139485 |
| repeated | e0,e0 | .5(1.95 e0e0T) | -0.535091374063 |

这里读出h2-h0指最后hidden的坐标2减坐标0，不是时间步之差。LN前对应差的公因子去除后精确为1/40与-39/40；两坐标均高于均值，ReLU不裁去它们，所以符号保持。写入状态/独立解析读取误差<=2.23e-16，Eq3全历史重算误差0，旧native态经共同基变换的Born概率误差2.23e-16。[derivation.md](../evidence/rmea/20261126-native-fw-class-audit/derivation.md)给出完整证明，[results.json](../evidence/rmea/20261126-native-fw-class-audit/results.json)与[verification.json](../evidence/rmea/20261126-native-fw-class-audit/verification.json)给出数值/分数凭据。

新轨迹[sample-trace.v2.json](../evidence/rmea/20261126-native-fw-class-audit/sample-trace.v2.json)，SHA `1ba2b3f06f04da0fc43cf6fbdc210797a6b2e2306fecd883f3f967379742a1ff`，2个完整预定序列、26数值表示、1192元素、74坐标读取，无结构error/warning。输入、初始memory、每步preliminary/preLN/中心化/variance/hidden/memory前后与共同读出完整保存于 `distinct.values.json` / `repeated.values.json`，`fw_trace.py read_value`检验hash并提供任意坐标读取。data/scoring/generation/diagnosis均实际记录；训练、任务gold、NLP评估不适用。该两例全部是人工机制验证，不能推及自然任务频率或效用。

## 已知恒等式及数值范围

[已测事实] 原Q池化可写为Tr[(E_r+EPS I)V_r]/Tr[(E_r+EPS I)M]，M=sum rho_t、V_r=sum u_rt rho_t；V_r可带符号，EPS保留。Hermitian64实坐标对点积精确表达；复数8维态写为real16后，概率为两个内积的平方和。此前real8只针对实态例有效。明确原生角度的复相位例给出Born概率0，而丢弃虚部后概率.25（再归一实部则1），所以该简单real8化对一般原生态不成立。这不是R8任意非线性编码的维数下界。

冻结24组(T=1/2/7、实/复各4构造、每组40关系)检查全部通过：稳定operator误差7.701e-15，Hermitian64误差3.811e-14，real16概率误差2.841e-16，池化误差2.920e-16。另4个signedutility/EPS精确Fraction读出证书与FW符号分数证书通过。CPU process_time=0.0625秒，QT NumPy1.26.4，没有安装。

直接未中心化的矩阵迹比值在T=1实态第0例有7.77085e-12缩放舍入误差，最初执行exit1及加诊断消息的exit1都保留。改为完全等价的c+Tr(Q sum(u-c)rho)/Tr(Q M)，c仅取已定义的逐关系词效用均值，消去常数项造成的浮点抵消；EPS、输入、normalizer与门槛均未调整。最终exit0。未用失败改科学假设或重开真实任务尝试。

[机制解释] 量子态的方向/交叉几何可以被实际关联memory保留，旧的同p/同value/同population约束不排除这一完整FW实例。[尚未证明] 一般FW包含全部Q函数、FW在本任务更好、当前FW与原头公平容量、自然任务量子优势或非平凡新T，均未建立。线性无LN矩阵幂不是实际FW；经典可模拟性也不是否决量子机制的准则，它只要求诚实的比较对象。

## 任务证据与可改善空间的复用边界

已审计既有报告，[headroom-reuse-audit.json](../evidence/rmea/20261126-native-fw-class-audit/headroom-reuse-audit.json)保留各口径。完整五路seed13/20epoch历史测试：原Q all40 Macro=0.495323641671，real12=0.546447526183，可观测差0.051123884513；这是其他同输入头已达到的分数，未证明量子参数族也能达到。gen98同冻结TRAIN筛选Q=.500925142309，最佳当轮G=.510930539751，最强声明复用Gcov=0.512054050292，Q距其0.011128907984；复用Gcov结构/参数量不同的限制照保留，不偷偷删掉强控制。不同数据人口/预算不合并。

旧top2 accuracy .9755和共享offset .8835来自旧checkpoint2da52b1c…、4000assessment、accuracy/positive39口径；不挪到当前checkpoint8ec179f4…或4096/all40 Macro。旧dataset qualification的qualified标签和限定source-copy范围保持，不写成新缓存端到端认证。超越最强经典控制的可达上界与量子可改进幅度仍unresolved，不把名义1当可达最优，也不要求精确最优才允许一切有用探索。

## 决定与恰好一项推荐

**NO_QUALIFIED_NEW_METHOD_FROM_THIS_AUDIT。** 本轮原生head/circuit调用0、任务/标签/checkpoint访问0、GPU/网络/API/服务器/COS写入0，仅2次既有独立NumPy稠密电路函数用于人工角度例。原限定L4、所有正负历史与gen98/gen102/gen104 STOP保持；新L3/L4-T **NOT_ACCEPTED**，目标ACTIVE。

推荐下一步只做原核心的一个输入可计算假设选择，尚不选插件。待证假设是长context中3量子位编码可能混叠任务相关词，作用超出普通浓度/方差；动机仅是既有固定lengthcut35观察Q-v-G短句net+1、长句net-50，尚无因果证据。最小动作先逐定义排重已停distinct-token covariance、frame/whitening、Fisher/embedding/shared-query，不重新审所有history；若结构不同，才能写一个无标签输入统计、匹配强Gaussian/dephased/real16控制与可证伪预测/STOP。若只是旧闭族或purity/variance/length改名，就记录NONE。预算0GPU/0新taskforward/0fit/0新容量，现持续目标已授权只读证据与假设选择；后续诊断需另冻而不是此处自动启动。[next-direction.json](../evidence/rmea/20261126-native-fw-class-audit/next-direction.json)。
