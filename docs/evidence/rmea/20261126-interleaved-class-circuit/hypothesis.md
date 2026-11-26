# 原生关系电路插入词态演化：待核查的单一任务假设

状态：PROVISIONAL_DESIGN，未实施任务插件、未调用任务forward、未训练、未分配GPU。既有数据/基线资格和训练core继续复用，gen98/102/104/105及全部确切停止配置保持。

原生词电路为 W2(h,roles) W1(h,roles)，每段是一层逐wire RY/RZ及同一CNOT ring；关系电路为 Q_r，两层共12角。原生概率为

`p_out(r,t) = |<0| Q_r† W2(t) W1(t) |0>|²`。

唯一拟议变动是关系电路的位置：

`p_mid(r,t) = |<0| W2(t) Q_r† W1(t) |0>|²`。

保持原prepare/effect参数、角度pi*tanh、输入、EPS=1e-6、词归一化、独立h-linear64 value、40类decoder/bias及优化预算。不是增加深度、重上传次数、寄存器维数、query数、额外输入或新参数。Q_r†必须是实际原生完整电路的共轭转置，不能把角度取负后仍按原顺序执行。

任务假设：原生关系只比较已完成的词态。插入后的关系操作可在词态的第二段演化之前改变其分量，形成关系与词第二段参数的非交换响应。它可能更容易学习哪些词证据对各关系有用。这是受原生Q及已关闭改型不及强经典结果启发的假设，没有确认过相应错误原因，也没有效用正证据。

该式与gen76 interlayer-channel不同：gen76在两段之间混合相干态/计算基去相干态，class effect仍在末端。它与共享entity-query transport不同：后者以句子实体anchor生成C(x)，对关系query作同一酉输运，再与完整词态重叠；这里关系电路在每个词自己的两段演化之间。旧6qubit-register是六个单qubit重叠概率相乘，输入角拆成6对，没有中间class操作；它与此处64维vec表示的关系须区分实际特征与query约束，不能单凭维数排重。一般data reupload、process kernel仍是需要核查的近邻，不能凭名称宣布新颖。

在固定参数下若W2(t)与Q_r†交换，则两概率相同；若所有词W2相同，一个不受限的query family可吸收这个公共酉变换。无论概率怎样变化，只有一个有效词时alpha=1，logit等于原value+bias，不能凭单词概率差证明任务差。真正的输出检验至少要有两个词及非恒定utility，并保留可训练encoder/decoder的解释限制。

存在已知的静态lift：令a_t=W2(t)†|0>、b_t=W1(t)|0>，amplitude=a_t†Q_r†b_t。向量化Q可把它写成64维process特征上的双线性形式/平方重叠。这个Choi/vec恒等式不是新T，也不证明与原生8维全参数化任务family严格分离。复数8维各块还可精确realify为16维实矩阵链，这是技术同函数控制，不是需要输给量子模型的独立训练baseline。

若数学/近邻排重通过，拟在既有8192FIT/4096assessment/cohort8fb8393b、seed13、batch128、原AdamW、1024更新和末次all40Macro上冻结一次小筛选：interleaved Q、在每个W1/Q†/W2块后去相干的interleaved D、同12参数四层RY实电路按两层+关系四层+两层插入的real8。保留同预算native Q、native real12/Gaussian及最强声明Gcov参考；只有输入/初始/顺序/预算完全匹配才复用旧参考，不重训旧尝试。新三个臂共享value/decoder/bias及可比初始参数。Q/D使用同一全矩阵，不用已各自训练的value作固定checkpoint路由归因。

暂拟效用门：Q末次all40Macro严格高于每个新控制及所声明旧强参考；CE同时报告，不替换Macro。所有类/零support保留。任一关键比较失败，停止这个确切插入点/ansatz/预算profile，不改位置、depth、seed、cohort、温度、budget或控制救援；没有通过前不完整训练或补seed。该门须在任何新任务计算前以实际控制身份/代码/budget落定，这里尚未准入。

研究边界：同registered参数不等于同有效函数容量；插入使Q矩阵的更多列影响输出，有效不变方向可能改变。RZ删去后的real对照不是完整复杂度匹配，须用强实/经典参考判断实际价值。已知非交换性/vec lift/简单输出反例均不是L4-T所需新理论。若只有位置/训练速度解释，或实际被旧profile公式覆盖，则返回NONE。完整L3、独立确认、实质近邻区别、非平凡正确理论、预测及实现/独立审核仍未满足。
