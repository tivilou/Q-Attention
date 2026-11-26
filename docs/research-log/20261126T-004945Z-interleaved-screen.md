# 中间关系电路固定小筛选：STOP_EXACT_MIDDLE_PLACEMENT_PROFILE

本项目用固定文本编码器产生每个原词的1024维特征，加4维实体角色信息，经12角词电路和40个类别关系操作计算词权重。词权重加EPS后在句内归一化，池化h-only线性64维值，经40类decoder与bias得到logits，以交叉熵训练。当前筛选只使用既有TRAIN中8192条拟合、4096条评估，最终1024更新计算包含全部40类的宏平均F1；VALID/TEST未读。本轮保留同一原量子核心，不引入新的编码输入、深度、词值分支或训练优化器。

| 项目 | 原生位置消融 | 本轮中间关系位置 |
|---|---|---|
| 概率 | abs(<0|Q† W2 W1|0>)² | abs(<0|W2 Q† W1|0>)² |
| 共同项 | 原prepare/effect、pi*tanh、线性值、EPS1e-6、decoder/bias | 同左；完整初始tensor逐项相同 |
| 控制 | 已完成native Q/D、real12、Gaussian及强矩参考 | 新D块边界去相干、新real8同位置，固定映射 |

[已测事实] 插件先经一次A6API gpt-6.2-astra/high受影响机制审查PASS_WITH_RISK（49.576s）。按审查要求补上FP32实际参数变化断言及缺失参考源pins；科学公式未改。56项源码/输入/参考文件全hash通过，八个参考原始初始化摘要复现、共享value/decoder/bias逐tensor相同；native Q全状态与新三臂一致。旧参考均未重训。

[已测事实] 原Linux训练环境的CPU独立dense算子/相位还原复振幅/显式density-D/real8有序实矩阵检查通过。32词角×40关系角，FP64概率最大误差1.1102230246251565e-15，FP32最大5.779827247920366e-7，低于预设1e-11/2e-6。三个头各用固定前8条真实FIT样本及labels执行丢弃的FP64与FP32 clone一步更新：梯度有限、有限差分通过、参数确实变化，原初始模型未动。Windows QT的首次初始摘要核对失败被原样保留；训练环境复现原冻结摘要，未替换期望摘要或重新选择初始化。

任务样本逐层训练轨迹/完整sample-trace.v2未采集；本轮是开发效用筛选，不声称trace-complete机制归因。已保存的真实8条clone有限差分/零梯度/更新记录见validation.json；全4096三臂logits、完整checkpoint仅留原服务器，私有diagnostic摘要见summary.json。人工角独立oracle不代表自然文本或编码器训练轨迹。

## 固定结果

seed13，AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1，batch128，FP32/complex64、deterministic、TF32off。三臂各1024更新，共3072；全部拟合完成后才做最终评估。microbatch配置继承4，但复用的dense_batches路径实际整batch执行，与旧参考路径一致。强Gcov是82308注册参数，新三臂81028，差异保留。

| 新臂 | all40 Macro-F1 | CE | 注册参数 |
|---|---:|---:|---:|
| quantum_middle | 0.350218485036 | 0.804061532 | 81028 |
| dephased_middle | 0.444409337067 | 0.650071621 | 81028 |
| real8_middle | 0.350500416677 | 0.871053576 | 81028 |

| 复用参考 | all40 Macro-F1 | 注册参数 |
|---|---:|---:|
| context-spectral-flow-20261125/quantum | 0.389300602574 | 81028 |
| context-spectral-flow-20261125/dephased | 0.472501085293 | 81028 |
| staged-bottleneck-design-20261125/real12 | 0.382766614258 | 81028 |
| staged-bottleneck-design-20261125/gaussian_base | 0.484228946800 | 81028 |
| connected-value-pooling-20261125/gaussian_covariance | 0.512054050292 | 82308 |
| connected-value-pooling-20261125/gaussian_mean_product | 0.511174688570 | 82308 |
| distinct-token-evidence-20261125/gaussian_distinct | 0.510930539751 | 81924 |
| semantic-connected-evidence-20261125/quantum_shared_mean_product | 0.504045671350 | 81924 |

[已测事实] 冻结十门通过0/10；失败门：Q_beats_dephased_middle、Q_beats_real8_middle、Q_beats_reference/context-spectral-flow-20261125/quantum、Q_beats_reference/context-spectral-flow-20261125/dephased、Q_beats_reference/staged-bottleneck-design-20261125/real12、Q_beats_reference/staged-bottleneck-design-20261125/gaussian_base、Q_beats_reference/connected-value-pooling-20261125/gaussian_covariance、Q_beats_reference/connected-value-pooling-20261125/gaussian_mean_product、Q_beats_reference/distinct-token-evidence-20261125/gaussian_distinct、Q_beats_reference/semantic-connected-evidence-20261125/quantum_shared_mean_product。独立用40×40整数混淆矩阵及Fraction重新计算每类和总Macro，十个严格大于判定全部重现；平局视为失败。CE为次指标，全部类及零F1保留，无择seed或择指标补救。[verification.json](../evidence/rmea/20261126-interleaved-screen/verification.json)

[机制解释] Q/D末词段三个RZ只带整体相位，路由梯度精确0；D还失去首词段三个RZ及query末层三个RZ。AdamW仍可衰减这些已注册参数。real8采用两段各2层RY+ring、关系4层RY+ring，门数与几何不同。注册数相同不能推出有效容量或复杂度相同。完整Q矩阵的真实adjoint及D块边界含义见plugin.py与review.json。

[尚未证明] 合成终态相同而中置响应不同，只是固定preparer/utility可分性，未证明自由重参数化模型族、自然任务误差原因或非平凡新理论。旧vec/Choi/realification等价仍成立，无计算优势结论。当前没有新L3/L4-T，原限定L4保留。

## 决策与唯一下一步

STOP_EXACT_MIDDLE_PLACEMENT_PROFILE。只审计gen106固定最终Q/D/real8三臂的逐类与配对错误：复用已存4096 TRAIN logits，0新模型前向/0训练/0容量；报告所有40类及Q独有修复/损害，检验此前合成路径可分性是否存在任务证据。不得改变插入位置、深度、初始化、温度、seed、数据或控制来恢复此已关闭profile；审计后再择至多一个实质不同的原量子核心假设或NONE。

成功判据及kill gate保持事先冻结：Q必须严格高于新D/real8和全部八个参考；任一失败关闭本确切位置/ansatz/预算，不位置-depth-seed-temperature-cohort-budget-control救援。下一步0GPU/新forward/fit/容量，仅分析或前置协议；后续任何新实验仍走同一共享调度。gen106 allocation已trusted released，旧gen98/102/104/105及全部正负历史保留。

证据：[design.json](../evidence/rmea/20261126-interleaved-screen/design.json)、[summary.json](../evidence/rmea/20261126-interleaved-screen/summary.json)、[validation.json](../evidence/rmea/20261126-interleaved-screen/validation.json)、[input-binding.json](../evidence/rmea/20261126-interleaved-screen/input-binding.json)、[review.json](../evidence/rmea/20261126-interleaved-screen/review.json)、[execution.json](../evidence/rmea/20261126-interleaved-screen/execution.json)、[next-direction.json](../evidence/rmea/20261126-interleaved-screen/next-direction.json)。原公式数学证据沿用上一checkpoint，未作为新任务正效应。
