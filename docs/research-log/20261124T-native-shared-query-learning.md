# 共享学习诊断：抵消不是量子独有解释

完整 TRAIN 四状态诊断在73.36秒完成。训练后Q/real12的梯度coherence分别约0.000160/0.000191；初始Q反而高于real12。这项检查没有找到量子头特有的学习信号抵消，不能据此再调优化器或类权重。

逐例自由解不等于同一模型能学到的规则。后续保留这一差距，另立可证伪方法假设；本轮无训练、无新heldout和COS，GPU已归还。
证据：[完整结果与边界](../evidence/rmea/20261124-native-shared-query-learning/analysis.md)、[固定协议](../evidence/rmea/20261124-native-shared-query-learning/protocol.md)。
