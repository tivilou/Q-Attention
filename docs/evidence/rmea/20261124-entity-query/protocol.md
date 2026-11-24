# 实体对条件查询：前瞻冻结协议

本文件在任何本方法模型执行前冻结。旧五路完整预算及所有已停止筛选配置保持原状。本方法不是它们的增预算、换种子或重定指标。

假设：当前词特征已经编码实体标记；另外用主客体词平均形成查询条件，可能帮助关系查询定位这对实体的证据。未证明固定查询是既有错误原因。所有新条件使用完全相同的输入：拼接 h 与四维 roles，分别按 mask 和最后两维二值实体标记求均值，再对两个实体均值取平均。Q/D 将共享 prepare(anchor) 加到 effect 的 tanh 前；real12/softmax12 将共享 key(anchor) 加到 query。key/value/decoder、CE、优化器和数据顺序不变。不增加参数。

新五路：quantum_context、dephased_context、real12_context、softmax12_context、real_circuit_context。实电路四层 RY/CNOT，12RY/12CNOT；原复电路6RY+6RZ/6CNOT。相同81028注册参数不代表相同有效容量；实电路更多纠缠门的差异明确报告。

技术门：逐 split 只检查输入实体标记非空/二值，不读取TEST标签、不进行TEST模型推理；新模型有限输出、padding不变、禁用条件后的原模型parity（实电路不冒充原量子parity）、有限梯度及一次真实TRAIN更新。失败只修机械bug并保留失败，科学配置不得看效用后改变。通过后可以进入以下完整资格实验，无4k效用筛选、无旧生命周期矩阵。

完整资格：seed13；TRAIN58465/VALID19584/TEST13418；20epochs，每epoch457次更新，batch128、FP32；原AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1；Torch randperm(seed+epoch)；严格VALID all40Macro最好且最早并列；仅选定checkpoint对TEST一次。五路新条件各9140更新，通过已有一键入口、双卡动态队列、localbest/latest和10% aggregate COS节点执行。不得重复原五路训练。

复用原生五路：source4479550ffdb061d686f5308bfd51ed5d3478c350、原完整输出及封印、相同数据/初始化/顺序/优化器/预算/选模/指标身份。新Q必须严格超过自身D、同输入real12、softmax12、实电路，以及全部原生uniform/Q/D/softmax12/real12，才通过单seed资格门。无种子、预算、checkpoint、metric救结果。所有十个结果一并报告，旧结果不改写。

第一完整seed13通过仍不接受L3。随后另行在看到结果前冻结独立多seed确认；之后才处理同一核心的近邻区别、非平凡正确理论、可区分预测、实现一致性和独立实证/内容审核。条件查询及Born/复振幅本身均不自动构成理论新颖性。新L3/L4-T均未接受。

持久化：复用已接受恢复实现和已固定输入对象；新增仅必要源/配置/输入引用、当前恢复best/latest/RNG/optimizer/游标，每10% aggregate封印COS，Git携带代码与紧凑报告；不在Windows落地模型/数据，不额外购买资源。
