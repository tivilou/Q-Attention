# 实体条件酉输运查询：前瞻候选

冻结时间：本文件及插件写入后、任何该插件模型执行前。之前实体查询完整配置的失败保持关闭，不是换种子、增预算或改指标救援。

已知事实：旧完整条件查询 Q TEST all40Macro .521912，比原 Q .495324 好但低于经典 .546448；VALID下降。固定TRAIN128、五个已选模型、无标签诊断发现：Q查询平均迹距离 .81447→.57046，关系间词路由TV .34884→.29318；softmax路由TV .32109→.18129，压缩更多而成绩更高。因此不能宣称压缩是失败原因。诊断原forward逐值相等、0更新、2.868秒、gen60已释放。

待检验假设：实体上下文可能有用，但把它加进关系参数的tanh前会同时改变关系间几何。改成在已准备的关系态上施加所有关系共享的实体条件酉电路，分离“共同条件变换”和“关系态准备”，有可能改善任务效用。并不预测保持距离必然改善分类。

精确方法：x=concat(h,roles)，c为与上一轮完全相同的主客体masked-mean平均，含相同角色输入；key=U(pi*tanh(prepare(x)))|000>，q_r=U(pi*tanh(effect_r))|000>，q_r(c)=U(pi*tanh(prepare(c)))q_r。prepare含偏置、与key共享，不新增参数。Q使用模平方Born重叠；D在完整query transport之后对key/query取计算基概率再内积；实控制为4层RY/CNOT的key/basequery/context-unitary。value/decoder/CE/优化器不变。不把该顺序当作词间相干或全局POVM。

可直接检验的结构预测：每条样本40个query的复Gram在conditioned开关前后保持相同；这与上一轮tanh前加法不同，但不是Q相对实酉控制的独有性质。attention行归一化不保持Gram/迹距离，也不推出分类或鲁棒性界。unitary、Born、共同旋转、Gram守恒都是已知工具，不申请理论创新定级。旧R99是按每类自身utility构造的双对易梯度旋转，本方法无utility反馈、无每类生成元/步长，二者不是同一模型。

仅三路新训练：quantum_transport、dephased_transport、real_circuit_transport，均81028注册参数；有效容量不宣称相等。复query执行base6RY6RZ/6CNOT再context6RY6RZ/6CNOT，key仍6RY6RZ/6CNOT。实query为base12RY/12CNOT再context12RY/12CNOT，key仍12RY/12CNOT；额外门数明确披露。

技术准入只测新插件：从|000>开始的transport与已有原电路/实电路parity、归一化、query Gram不变、padding、无条件原native Q/D parity、有限梯度、一次真实TRAIN更新。复用上轮实体标记可用性和稳定核心/多GPU/恢复能力，不扫描全split、不重哈希4.4GB特征、不重做生命周期矩阵。新代码需一次精确源机制/代码审查。

完整资格沿用原协议：seed13、TRAIN58465/VALID19584/TEST13418、20epochs、b128/FP32/native AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1、Torch randperm(seed+epoch)、严格VALID all40最好/最早并列、选定TEST一次。三路各9140更新=27420；动态双卡队列。原生五路和上轮实体条件五路直接复用，输入/初始/顺序/训练预算/选模身份核对，不重训它们。新Q必须严格胜过自身D、实酉控制及全部10个历史完整模型的TEST all40，才进入另行前瞻冻结的独立多seed确认。失败关闭确切配置；不救seed/指标/预算/控制/选模。不把单seed资格当L3，也不以Gram恒等式当T。

持久化仍用稳定best/latest及每10%总进度COS、同一已固定输入对象、Git源与紧凑报告，不在Windows落地数据/模型，不购买资源。当前是未验证方法，完整实验尚未启动；仅全部L3+T通过才接受L4-T。
