# Q-RMEA L5 第六轮：已知 partial-swap 融合的有界效用筛查

原限定L4保持；**L5未接受**。本轮结论为 `RETAIN_EXPLORATORY_UTILITY_LEAD`，只适用于下面固定开发预算。没有新增全量实验或师弟服务器访问。

## 先例与改变

[Audenaert/Datta/Ozols《Entropy power inequalities for qudits》v3](https://arxiv.org/abs/1503.04213v3)的eq3.1及4.10–4.17已给出partial-swap quantum addition和对应ellipse。只阅读pp1–10并检查p9图像；PDF页脚18Apr2016、内部日期19Oct2018分别保留，不称出版社最终版。融合算符、正性与简单圆盘条件均不计作新理论。

[Fernandes等《Unitary-invariant witnesses of quantum imaginarity》v1](https://arxiv.org/abs/2403.15066v1)只阅读pp1–4并检查p2图像；PRL133190201元数据与预印本版本分开。三态pairwise-overlap结论不能外推为四态以上都无法见证imaginarity。两篇的附录/完整证明均未审核；[版本与阅读范围](../evidence/rmea/20261116-l5-round6/reading-receipts.json)不是新颖性放行。

新设计把主、客体跨度内**已制备词态投影的平均**A、B直接融合成关系二元测量E_r=((1-kappa)A+(1+kappa)B+lambda*i[A,B])/2，kappa²+lambda²≤1。删除全局关系phi，保留原制备、epsilon归一化和value/decoder；不是制备平均词特征，也不是以前已停止的实体unitary链或六量子比特joint分支。原失败结果仍保留。经典复数运算可精确实现此模型；没有硬件加速、必要性或经典不可模拟主张。

## 固定开发比较

CPU重建缓存2048train加313个train-only覆盖样本，总2361train/40类/每类至少6，已有1024valid；这些样本已暴露，不是独立确认数据。一个固定seed307，四个epoch，batch16，AdamW lr.001/weight_decay.0001/clip1，全部共享初始化和batch顺序。各自按strict最大valid full40 Macro、同分最早epoch选模，保留全部epoch结果。没有测试集、encoder重新前向、正式baseline重跑或新COS读取。

| 条件 | 所选epoch | 参数 | valid Macro | accuracy | CE |
|---|---:|---:|---:|---:|---:|
| global | 4 | 81028 | 0.163523108 | 0.620117188 | 1.318084015 |
| even | 4 | 80548 | 0.166858968 | 0.670898438 | 1.231775572 |
| mixture | 4 | 80588 | 0.165523380 | 0.672851562 | 1.231031350 |
| fusion | 4 | 80628 | 0.170059088 | 0.679687500 | 1.240799656 |

fusion−mixture Macro=`+0.004535708`；fusion−globalQ=`+0.006535980`。冻结规则要求两者都>0才保留探索线索，否则HOLD_NO_PROMOTION。所选fusion固定其余参数把lambda置0后，Macro=`0.170059088`、accuracy=`0.679687500`、CE=`1.240795537`；on−off Macro=`+0.000000000`。这是局部依赖诊断，不是总增益归因；不同gate参数化、优化和选模仍是解释边界。

固定同一所选模型把相位系数lambda置0，Macro及accuracy均不变，CE从1.240799655694134下降到1.240795537266044。因此本次没有观察到相位项改善这些判别指标；完整fusion训练略优不能归为相位收益，gate参数化、共同训练和选模路径仍是其他解释。所有条件四个epoch的Macro仍在上升，本轮不是收敛后效果比较。冻结的保留规则通过仅指探索性utility lead，不代表相位机制得到支持，也不产生L1/L4/L5等级。

## 实现与证据边界

共享调度task `6efa849c64aa46cbbe1fbae1fe89cd71` / attempt `877f3cf0fd7d4661b11cbc5c2a3579dc`，source `4f799cbf3b0f4d9d5d586552cd93975bce0816a9`，source SHA `8458d724c09deeb151bb60b8e4f6f65d4c493d7a6eb7c43022dbde13c84cf7b5`，contract SHA `f4d32b7b0b4b2a837e9da1c9cd9a592c50444a82c296b882c4ab9a76f47901fa`。运行 `112.696`秒、peak RSS `679984`KiB、退出0。独立BOTH审核先发现辅助文件未与review完整绑定，已在执行前修正为freeze原始SHA绑定；审核通过后才部署/运行。

单位检查先于训练：独立d²partial-swap unitary与显式partial trace对照24组混合态，检查正性、越界反例、归一化抵消、梯度/mask/padding，以及原globalQ前向和梯度逐项一致。所选checkpoint/validation logits在worker逐一回读复算一致，输入未变。所有训练、开发结果和不利对照均保留，[逐epoch](../evidence/rmea/20261116-l5-round6/epoch-metrics.json)不隐藏。

worker输出闭包 `14`文件/`2682507`字节已核验；仅compact聚合报告回本机，私有checkpoint、logits和特征均未下载/上传Git或COS。CPU短筛查只声明从既有输入重新开始，不声明训练断点恢复或冷恢复保障。owner/worker临时绑定均恢复。独立逆补丁夹具通过，原checkout、main和1.1保留。

[理论边界](../evidence/rmea/20261116-l5-round6/theory-boundary.md)说明非零commutator可被归一化或value读出消掉；这些商法则恒等式也不是非平凡L5理论。即便开发筛查为正，新模型也不能继承旧Q-RMEA的L4，仍缺直接注意力近邻区别、强经典对照、独立实证和同一贡献上的实质理论。正式全量实验继续只交师弟，当前没有正式交付授权扩展或准备完成主张。

记录时间：2026-11-15T18:47:25.920182+00:00。
