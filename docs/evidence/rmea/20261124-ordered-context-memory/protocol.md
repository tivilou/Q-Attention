# 有序上下文记忆：固定五路探索筛选

动机来自累计错误与已关闭路线，不声称已证Q的任务根因：现有RMEA每个词各自制备后测量，量子模块本身不跨词传播状态；冻结RoBERTa词特征已经含上下文/位置，不能称整个模型是词袋。新假设仅为增加词序敏感的有限量子记忆后，关系路由可能受益；固定内容32未证多线索或干扰实体为Q特长。

在原native3qubit/2layer线路上构造完整8x8酉矩阵U(theta)，词角theta_t=pi*tanh(prepare([h_t,roles_t]))/sqrt(有效长度)。V_t=U(theta_t)U(0)^dagger保证零角是恒等步；按原词序产生psi_t=V_t...V_1|000>。padding是恒等转移。40个原生query态、正测量质量+原EPS、逐关系词归一化、value和decoder均保留。并行前缀积必须对顺序串行积/导数一致，不改稳定训练核心。

五路：quantum_native原AdamW锚点；quantum_seq；dephased_seq在每词转移后完全计算基去相干（概率转移|V_t|^2，非仅末端去相干）；real12_seq和softmax12_seq使用相同12维线性输入投影/角缩放，12维实正交状态有序传播。其正交转移为两层相邻及错位不交Givens对，每层6角，零角恒等；分别平方归一化重叠与scaled-dot-softmax读出。它们不是同有效容量证明，需报告表达式和参数量；均81028参数，无额外可训练记忆。所有shared value/decoder和量子native initial复用原seed13，经典初始化复用原factory。

固定既有TRAIN整句4192fit/4000assessment，seed13/b128/每路512原生FP32 AdamW/.001/.01/betas.9,.999/eps1e-8/globalclip1；零VALID/TEST。只有全部fit完成后的final512、全部4000/all40一次评估。Qseq必须Macro严格高、CE严格低于另4路（全部8门）才进入新的完整协议准备；否则关闭本确切模型/缩放/512配置，不变长度缩放/线路深度/方向/seed/预算/指标或对照救援。短筛选不判全家族不可能。

每路600秒、总3000秒原compute预算；共享一次两GPU分配/原runner自动队列。最小新增检验：矩阵首列对nativecircuit、零角/padding/范数、前缀对串行输出及梯度、每路一次真实8例更新、受影响精确源码review。复用已接受训练/序列化/调度/COS能力，不做生命周期矩阵。短可重启筛选仅本地latest/best/final，不写COS/Windows模型数据。

已知近邻：Bausch QRNN 2006.14619v1、Arjovsky等uRNN1511.06464v4；循环量子记忆、酉范数保持与非交换顺序均是已知工具，不作新T。这里改变每词key的信息依赖，区别于已关闭的对全部query共同酉旋转和图上entity-scattering resolvent；未继承任何效用、创新或理论评级。若短筛选通过，正式设计仍须包含未经新转移改动的最强原生经典、恰当消融及独立重复，再单独审查同核心实质理论。
