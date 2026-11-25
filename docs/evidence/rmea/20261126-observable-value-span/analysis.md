# 固定 native-Q 表示的 value-span 诊断：停止指定替换假设

本轮沿原量子研究主线执行上一轮冻结的诊断。generation 102 已完成并释放，三个冻结门均失败；停止用当前固定表示的 Hermitian observable 拟合旧逐词 utility 的这项替换方案。该结论不覆盖联合重训表示、直接优化池化 logits 或其他量子研究方向。新 L3/L4-T 未接受，持续目标 ACTIVE。

## 冻结输入与执行

使用已有 seed13、epoch17 的 native-Q checkpoint（SHA `8ec179f4605a901ea1aeeeaefaacc20d78a0b202f236a2d41c1a37b06220cf3c`），沿用 TRAIN 8192/4096 句组互斥 cohort。索引按缓存 metadata 的 id_hash 顺序，未重建分组。全部 51 个输入/源码 hash 通过，teacher、router、原 utility 及 bias 固定。checkpoint 参数转 float64 后的有效状态在前后完全一致。首次控制 Python 缺 torch 的环境探针在任务前结束，随后使用原项目环境，零安装。

每句总权重为 1，逐词使用 1/T 权重；只在 FIT 一次求 64 维实 Hermitian 系数及 8 维 diagonal 系数，固定 rcond=1e-10。assessment 前锁定系数 SHA，锁定时 assessment forward 为 0。另有 FIT 常数控制。保留原 alpha 与 EPS，没有调整 rank、阈值、checkpoint 或 seed。

## 全 40 类结果

| assessment 控制 | 平均 normalized MSE | 最差类 normalized MSE | teacher argmax 保持率 |
|---|---:|---:|---:|
| Hermitian span | 0.9096102994 | 0.9826075065 | 0.67626953125 |
| diagonal span | 0.9615988699 | 1.0089978233 | 0.67626953125 |
| FIT 常数 | 0.9917518201 | 1.0194991520 | 0.67626953125 |
| 冻结门 | ≤0.01 | 每类≤0.04 | ≥0.99 |

主方案三个门均失败；另外两个控制也均未达门。相同的 agreement 计数不证明预测向量相同。FIT 的 Hermitian 平均 normalized MSE 也达 0.9164820662，因此问题已经出现在拟合集。求解 rank 为完整 64/8，没有被截断的方向。拟合的 40 个算符范数在 8.1078–23.3895，属于宽松无范数约束诊断，未宣称满足 unit-norm 值域。

该最小二乘目标是 FIT 逐词 utility MSE，并非 pooled-logit 或分类保持目标；assessment MSE 也不是对所有 V 的最优下界。因此负结果关闭的是这份预声明协议及固定逐词替换假设，不能推出所有固定表示分类器、联合训练或量子 value 方法失败。teacher 曾训练于完整 TRAIN、由旧 VALID 选模，本轮分组只隔离当前线性求解，非独立任务确认；没有读取新 gold/F1 或官方 VALID/TEST。

## 数值与资源核验

运行 19.0532 秒，320 次 native-head forward、640 次原生 circuit 调用，零 optimizer/backprop；8192 FIT 累积与重放、4096 assessment，共 20480 example-pass。两个已访问样本的独立 NumPy 算符/重叠/池化核验，最大误差 2.31e-14；该检查不等于独立 circuit 实现。释放后只读取私有充分统计量，对同一解作 CPU SVD 复核，系数相对误差≤5.11e-14、normal residual≤1.48e-15，二遍 FIT MSE 与充分统计量结果误差≤1.04e-13，没有第二个任务拟合或调参。

共享调度 allocation `a05f5131ad684808bda8aa2c092e6289`、generation 102 已 released，原 owner 进程退出，资源可复用。特征、checkpoint、系数及充分统计数组均留服务器；本地仅收集汇总和 hash。未购容量、重启旧尝试、补 seed 或调用外部模型。

## 下一步

只做一个未选定机制的代数及历史去重前置检查：保持原有 signed utility，用其与密度矩共同构成输入相关的 class-effect 更新，检查是否有任务相关、可区分且区别于关闭历史的预测。候选表达式和 kill 条件见 [next-direction.json](next-direction.json)。须同时比较强经典 scalar-value exponential tilt、diagonal/实向量算符更新；单类 score 上升不构成 margin、Macro-F1 或量子专属收益。已知 Rayleigh/谱倾斜恒等式不计新理论。该前置检查无需新任务行或 GPU，不把更换生成元自动认作创新。

原限定 L4、gen98 STOP 与全部正负历史保持，经典精确表示仍作为公平控制，量子目标持续推进。
