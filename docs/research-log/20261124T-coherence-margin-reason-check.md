# 不先跑：相干margin约束存在参考作弊漏洞

同权重诊断给出了有价值的问题：相干修复很多正关系，也会误触发无关系样本。我们考虑约束相干相对去相干参考的margin增量，但先做了一个原生Born+EPS读出反例。

Q的gold-v-rival margin可一直为−1、始终答错，同时把参考margin从0降到−2，就能把“相干增量应非负”的惩罚从1降到0。两词、同一有效纯态/有符号value读出数值残差2.22e-16；这不是训练模型的实测下降方向，但足以说明约束本身不保证修复。

因此没有把这项朴素loss写进训练脚本或占GPU。stop-gradient只堵单步直接参考梯度，不会自动提供全程保证。后续先定位输入可识别、内容支持的相干误触发结构，或找到不会恶化参考的明确约束，再选择插件。

同时读了Right for the Right Reasons原文pp.1–4：输入梯度/解释约束是已有方法，当前没有语义无关mask，不能包装成新的理论。新的L3和L4-T仍未通过。

[反例与可复算代码](../evidence/rmea/20261124-coherence-margin-reason-check/result.json) · [近邻阅读范围](../evidence/rmea/20261124-coherence-margin-reason-check/reading.md)
