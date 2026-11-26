# 一项可实施的新信息路径：关系操作进入词电路中间

本轮保留一个 **PROVISIONAL** 任务假设：把原生关系电路从词电路末端移入两段词演化之间。公式及原生源码排重、一次外部机制审查和固定无数据数学检验已完成；没有新任务forward、fit或GPU占用。它不是效用结果，也没有接受新L3或L4-T。

## 具体改变和近邻

原式为 `|<0| Q_r† W2(t) W1(t)|0>|²`，拟议式为 `|<0| W2(t) Q_r† W1(t)|0>|²`。仍用原3qubit、两段词RY/RZ+CNOT、两层关系电路、原12角/prepare/effect、EPS词归一化和独立h-linear64值。区别是关系参数在词信息继续演化前参与作用，不是增加深度、query数、寄存器宽度或改训练优化器。

核对的具体旧实现：gen76在中间混合相干态/去相干态而保留末端class effect；query-transport先用句子实体anchor共同输运关系query；6qubit-register将六个单qubit重叠相乘。这些代码与新式不同。6qubit表示尤其接近：新token的vec特征经固定置换也是六个乘积因素，但单位酉Q的归一vec在两半间Schmidt秩8，旧query跨六wire完全乘积。这里只区分已实现的表示及约束，没有证明允许自由重参数化后两种任务family严格分离。

一般data reupload和process kernel仍是直接文献近邻，尚未核原文或清除新颖性问题。已知的64维vec lift、16维实矩阵表示及非交换性不作为新T。既有完整/小筛选负结果也没有确认“末端位置是错误原因”。

## 实际新增证据

一次A6API `gpt-6.2-astra/high`机制审查返回 **PASS_WITH_RISK**，仅允许有界数学检验。精确model/effort回显符合要求，101.916秒，37650 tokens；没有第二次派发。模型元数据先查询ai-model-world，固定commit `6d17c876ce0761a7c39d333589a4c2af6a02e76c`，再核A6精确catalog；选择沿用owner代码审查偏好，不声称benchmark排名。审查、主模型采纳边界分别见 [review.json](review.json)、[primary-adoption.md](primary-adoption.md)。

固定32组角度的dense complex amplitude、原生Torch circuit和保留layer_matrix、real16、归一vec以及显式density-dephase全部一致，最大残差 **1.443290100671058e-15**，小于冻结1e-11。共64次原生vector-circuit及128次保留layer-matrix调用，均为人工角度CPU检验，0任务样本；计时0.558728秒。错误地仅取负角而不倒转顺序的query逆操作会产生最大amplitude差0.566458885，因此没有采用该近似。全部检查含singleton、padding及真正的zero-angle C/C² commuting端点。

审查给出的两词构造已通过原生门约定验证：两词W1角均0，W2仅wire0取(y,z)=(pi/2,pi/2)和(-pi/2,-pi/2)。它们最终rho相同，数值差2.47e-32；任何固定末端query都给两词相同概率。row0 query仅首层wire0 RY=-pi/2，新式概率分别0和1，块边界去相干式均为1/2。

取row0 utilities=(+1,-1)，其余39行及bias为0，原all40 logits为0，新row0为 **-500000/500001 = -0.999998000004**。class0 softmax从0.025变成0.00934467833361204，最大概率差0.015655321666387962。这个输出变化没有指定gold标签，不解释成改善。单词情况下alpha仍为1，位置变化不影响logit。

第二份实现用Gaussian有理数重建两密度矩阵的64项，用Fraction Markov矩阵核对D=1/2、原概率1/4及精确logit，再重开全部40行trace核池化。结果见 [math-results.json](math-results.json)、[independent-verification.json](independent-verification.json)。这是主模型的独立计算加外部审查构造，不是独立任务确认。

## 必须公开的容量损失

新式最后一段的三个词RZ只给出整体相位，对Q/D概率严格不可见，原prepare相应行从route得到零任务梯度。D还消去首段三个词RZ。因此Q的词角依赖至多9个方向，D至多6个，不能把registered12角当成有效容量相等。32组有限变化核验Q/D末RZ不变；原末端式对相同RZ变化的最大概率差0.09309076553。class完整酉矩阵又可能激活原query首列未使用的方向。该风险没有通过改门顺序或移动插入点隐去。

两词反例固定了preparer和utility。人工h=e0/e1可以实现它的角和值，但没有证明缓存encoder产生相应h，也没有排除自由native encoder/decoder重新学出相同最终预测。这个边界对后续理论与任务解释都有效。

## 下一步

只实现同一插入式的独立科学plugin与紧凑小筛选配置，明确Q/D/real8的角映射、adjoint顺序、无效方向和共同初始tensor。绑定既有8192FIT/4096assessment、seed13、batch128、1024更新及最强Gaussian/real/Gcov参考的实际输入/初始/顺序/预算；不满足同口径的旧分数不作为已匹配对照。复用稳定core/资格/调度，不建新runner框架。

在新任务计算前冻结所有必要比较：Q末次all40Macro严格胜每个新关键控制及最强声明旧参考，平局即失败；CE同时报告。任何失败停止这个确切位置/ansatz/预算profile，不补seed、位置、深度、预算或控制救援。数学检查通过只支持进入plugin实现和受影响验证，不准入任务训练。完整L3/适当独立确认、实质近邻区别、非平凡正确理论、预测及实现/独立审核继续欠缺。

原Q-RMEA持续科研goal ACTIVE；原限定L4、gen98/102/104/105及全部确切旧STOP和正负历史保持。没有以经典衍生成果替换量子目标。
