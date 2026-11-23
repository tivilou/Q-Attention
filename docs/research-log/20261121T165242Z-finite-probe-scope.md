# 有限探针路线收束：保留正向结构证据，回到任务机制

时间：2026-11-21T16:52:42.081489+00:00。主线：Q-RMEA L5；当前 L5 NOT_ACCEPTED，历史限定 L4 不变。

## 这轮得到什么

前一轮真实checkpoint的76行独立端口、decoder二维可观测性、九维区间证书和17组探针全部通过。这轮没有重跑它们，而是用4篇精确版本原文检查这些命题的贡献范围。注意力零空间、受限观测的核判据、PSD与非负分解，以及有限输入bank都有现成理论基础；组合到本接口仍需处理decoder与softmax，但不足以单独支撑L5。

有一个具体数学补充：固定routing、独立value、满行秩decoder下，识别全部attention的最少非基线探针数恰为ceil(R(T-1)/(R-1))。40类、9词时为9次；较弱decoder条件的16次最坏情形锐界仍然成立。Astra6.2 high独立核查PASS_WITH_SCOPE，无must_fix。当前checkpoint没有验证decoder满行秩，也没有执行9次方案，因此只记条件数学引理。

## 对研究方向的影响

正向结构结果留作机制解释工具；有限探针路线不再作为独立L5主创新提名。这不是撤回历史实验效果，也不是认定某篇论文已给出这个精确公式。真正待补的是实际可达输入与任务目标上、具有实质区别的同一机制贡献。

下一步从已有任务训练证据中选一条与已停路线不同的可区分机制假设，先明确它与普通共同适配的区别。继续扩展秩证书或探针数本身不再是主线，也不以追加种子或更换checkpoint救回旧结论。

## 执行与证据

用户确认3060曾重启；本轮一次只读检查已看到三卡在线且Q项目无非终态任务。没有新训练、GPU预约、COS写入或模型/数据/checkpoint下载到本机。

- [原文比较与完整证明](../../reports/q_rmea_finite_probes/20261121_novelty/comparison.md)
- [精确版本和实际阅读范围](../../reports/q_rmea_finite_probes/20261121_novelty/reading-receipt.json)
- [数学咨询](../../reports/q_rmea_finite_probes/20261121_novelty/review.json)
- [机器可读结论](../../reports/q_rmea_finite_probes/20261121_novelty/summary.json)
- [前一轮已执行checkpoint证据](../../reports/q_rmea_finite_probes/20261121_checkpoint/summary.json)
