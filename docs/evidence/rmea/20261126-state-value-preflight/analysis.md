# 同态 attention/value：两条停止，一项待核的信息前提

本轮完成上一检查点冻结的读值代数检查和四项既有机制排重。结果没有选出任务方法，也没有取得新 L3/L4-T。D 组件仍非退化，值得先问一个可直接核验的信息问题：固定量子词态是否保留了原生独立 signed value 所需的信息。

| 配置 | 实际结论 | 后续 |
|---|---|---|
| A：秩一类投影后读 V | EρE/p=E，池化输出仅由类查询和值算符决定 | 此配置因输入无关而停止 |
| B：n/p 弱值，沿用原 p+EPS 池化 | EPS留下未相消项；固定 ||V||=1、另有概率1的anchor时仍可发散 | 此配置因极点停止，不改EPS救援 |
| C：另一定义 n/(p+EPS) | 等于一阶密度和的有符号算符比；单位范数例可达约499.99975 | 只记录恒等式，不恢复B、不计新T |
| D：同一词态上 v=Tr(Vρ) | 保持observable范围，依赖一阶/二阶密度矩，一般非退化 | 保留组件，信息前提尚未建立 |

[完整推导](derivation.md) 保留了零概率条件事件、EPS和每个函数的差别。80组固定随机算符检查覆盖560个词密度矩阵，主实现最大误差 `1.7545481366772505e-13`；另一份纯Fraction矩阵实现完成48个正概率条件事件的精确检查，12个零概率事件保持未定义，并精确交叉核验同密度和反例、四种router/value去相干组合与弱值极点表。稠密实现最后一行弱值池化约499999，而去掉EPS的同一构造趋于0。[主结果](results.json) 与 [精确核验](independent-results.json) 均保存 literal exit0；这是独立计算实现，非外部理论审核。

D可精确改写为64维实Hermitian特征的一、二阶矩模型，因此后续必须包括这个经典控制。这个事实本身不否决量子组件，也不证明任务收益。构造中的两组词态拥有相同密度和，却给D不同读出；该二维反例未认证为固定原生ansatz的可达输入，更不作为自然任务证据。

[历史比较](history-comparison.json) 核对了 Q-BAM、gen97、gen92及已读operator-valued审计。D不等于gen97的32对自由经典value协方差，也不是gen92路由warmup或已知POVM的纯重写；它限制了原生h-linear独立value的函数族。这样的公式差异尚不足以建立实质新机制或新理论，量子value近邻文献也尚未在本轮核读。

下一项已冻结为固定表示的信息可表示性诊断：[协议](readout-span-protocol.md)。既有完整native Q checkpoint、缓存、cohort的本地回执相互匹配；cohort索引基于缓存id_hash顺序，须在服务器核对精确metadata/target文件SHA后再访问。**本轮未验证服务器文件可用性，未执行这项任务诊断。** 后续只用原TRAIN8192/4096，无新VALID/TEST，不以gold label拟合或计算新F1；以一次固定线性解检查宽松Hermitian span，并报告全部40类信息误差和teacher预测保持率。它只检验固定表示的必要条件，正面结果仍不晋级任务训练。[输入预览](task-input-preview.json) 与 [下一步](next-direction.json)。

本轮0新任务行、feature/checkpoint读取、native head/circuit执行、更新、GPU申请、server写入、COS或model API。此前gen98 STOP、原限定L4和全部正负历史保留，量子主目标 ACTIVE，新L3/L4-T NOT_ACCEPTED。
