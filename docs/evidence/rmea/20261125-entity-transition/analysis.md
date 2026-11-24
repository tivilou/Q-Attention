# 主体经词到客体的量子转移：固定四路筛选

原词/关系两层线路的完整酉矩阵保留。主体和客体各由实际实体词态的投影平均构成rho_S/rho_O；每个词转移T_t作用于rho_S，关系转移V_r作用于rho_O，以两者重叠作为注意力质量，再沿词归一化并池化原signed value/decoder。全部81028注册参数，无新经典主头。不同于只旋转query：词对主体态的作用也改变。

四路各512更新、seed13、TRAIN4192fit/4000assessment、b128/FP32/同原生AdamW与顺序；全部fit结束后final/all40一次评估。去相干对照只在完整转移端点使用人口概率；real12使用相同两层12角Givens转移，参数数目相同不表示同有效容量。drop_subject只去掉主体输入，客体保留。四个旧参考只复用同协议紧凑指标，未重训baseline或读取其checkpoint。

| 条件 | all40 Macro F1 | CE | 正类漏判null | null误报正类 | 来源 |
|---|---:|---:|---:|---:|---|
| quantum_transition | 0.160428723 | 1.321205378 | 666 | 283 | 本轮 |
| population_transition | 0.141503151 | 1.025422812 | 554 | 241 | 本轮 |
| quantum_drop_subject | 0.165307028 | 1.261187315 | 553 | 318 | 本轮 |
| real12_transition | 0.126943398 | 1.188985348 | 672 | 304 | 本轮 |
| quantum_native | 0.172509408 | 1.076011419 | 791 | 206 | 复用 |
| real12 | 0.211989007 | 0.922038913 | 592 | 277 | 复用 |
| softmax12 | 0.201206552 | 0.909885466 | 599 | 272 | 复用 |
| rbf12 | 0.353977485 | 0.882513046 | 517 | 308 | 复用 |

固定七项Macro门通过2/7：CLOSE_EXACT_ENTITY_TRANSITION_SCREEN。CE与错误计数只描述，不替换主门。

- macro_beats_population_transition: PASS
- macro_beats_quantum_drop_subject: FAIL
- macro_beats_real12_transition: PASS
- macro_beats_reused/quantum_native: FAIL
- macro_beats_reused/real12: FAIL
- macro_beats_reused/softmax12: FAIL
- macro_beats_reused/rbf12: FAIL

| 对照 | 修复 | 新损害 | 共同正确 | 共同错误 |
|---|---:|---:|---:|---:|
| population_transition | 251 | 347 | 2635 | 767 |
| quantum_drop_subject | 325 | 364 | 2561 | 750 |
| real12_transition | 371 | 317 | 2515 | 797 |

修复/损害仅对有对齐预测的三个新对照计算，四个旧CM不能推出逐例交集。
关闭本确切主体/客体混合、词/关系转移、顺序、归一化和512配置；不变source/operator/gain/init/seed/预算/指标/对照救门，不判整个转移家族不可能。

代数反例：I和Z制备相同真空态密度，但在plus输入/接收态时转移概率为1/0；固定值赋给被抑制词又会降低gold logit。前者是已知完整酉访问区别，后者排除了普遍收益保证。两者均不是新T/文本上限或不可经典计算定理。
科学运行142.181秒/2048更新。全部八个40类CM独立重算Macro误差≤2.78e-17，三组paired从远端私有数组独立核对。真实8行更新、独立纯态振幅/NumPy、梯度、原生相干/去相干真空退化、padding/空输入/实体和factory严格加载通过。

稳定corev6/host与原训练/恢复接口未改，未重做单卡/多卡/COS续训矩阵。无新官方VALID/TEST、COS写入，Windows未下载特征/模型/逐例数组。旧限定L4和全部历史保留，新L3/L4-T未接受。
