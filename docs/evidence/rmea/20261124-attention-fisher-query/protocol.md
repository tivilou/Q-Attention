# 归一化attention Fisher修正query更新方向：固定五路探索筛选

前瞻冻结：未科学运行。原生K1表达式不变；不是K2/输入条件query/角度尺度救援。假设是实际注意力几何的各向异性能改善原生query的优化方向；既有几何反例不证明训练根因或预期提升。

## 唯一算法

原生key/value/decoder/bias/单query及所有81028注册参数不变。五路为原生quantum AdamW锚点、quantum/dephased/real12/softmax12各采用同一Fisher修正规则。共享seed13原生初始value/decoder、各既有router初始化、原精确整句4192fit/4000assessment（既有8192TRAIN）划分及更新顺序。

每次训练forward仅使用当前128个输入、所有40行/有效词，固定key计算query12维块：g_t=grad_query log(raw_t)，F=mean_examples sum_t alpha_t (g_t-mean_alpha_g)(g_t-mean_alpha_g)^T。Q/D/real raw=p+1e-6，softmax raw=exp(score)。EPS、mask、真实query参数链式导数均计入。用中心化Gram在FP64聚合F；无标签、VALID/TEST或跨batch统计进入F。

已有全模型AdamW .001/betas(.9,.999)/eps1e-8/wd.01和全局gradclip1保持。先得到该步原AdamW更新；value/key/decoder/bias原步保持。只对去除decoupled weight decay后的query行步delta计算v=(F+(.01*trace(F)/12+1e-8)*I)^(-1/2)*delta，并把每行v的欧氏长度精确匹配到原delta长度，然后用同一decoupled decay更新query。每批一次，零初始/末端选择。F为零或delta为零时规则有定义；非持久cache当步用完清空，下步重算。

这是保留AdamW动量与步长的Fisher形状修正，**不是**坐标不变自然梯度、目标Hessian或新T。相同参数数不等于相同有效容量。方向变换在FP64中匹配每行步长；FP32存储后范数绝对误差须小于5e-7，并报告，不能宣称浮点字节级精确相等。额外计算更多，不声称等计算公平。

## 固定执行和判断

- 每路512逻辑更新/b128/nativeFP32复数complex64（仅F聚合/特征分解FP64）。双GPU原有队列；每路训练+评估最多600秒、整批3000秒。零新VALID/TEST/COS；local best/latest/final仍用原序列化器，评估只用final512。
- 全五路训练结束后各4000/all40一次评估，主报Macro、CE、所有40支持/F1/混淆和FN/FP/错正关系，配对修复/损害。
- Q Fisher须Macro严格高于且CE严格低于原Q AdamW和同路径D/real12/softmax12全部四对照（8门全过），才准备另行前瞻全量资格研究。否则关闭此确切Fisher形状/阻尼/步长匹配/seed/512配置。不调阻尼、平方根指数、频率、步长、seed、预算、split、metric或删对照救援。短筛选不证明全家族全量不可行。
- 拟合39/评估38类、暴露TRAIN划分和单seed的既有限制保留，不能直接定L3。即使筛选过门，仍需完整预声明独立L3和同贡献全部T门。原限定L4及关闭完整结果不改。

## 受影响检查

仅新数学/优化器：FP64真实首例前3词、all40×12 query坐标与原生autograd JVP对照，归一化切空间/padding/Fisher；clone真实首8标签各一次更新、步长相等/参数有限/原forward parity。科学initial不被fixture改变。精确新增代码审查包含稳定screen唯一新增make_optimizer工厂入口的默认行为。复用accepted双GPU/host/checkpoint/COS能力，不重复生命周期矩阵。
