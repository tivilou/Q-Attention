# Q-RMEA L5 持续研究：第一轮

本轮完成 L5 技能更新并开始理论研究；**尚未达到 L5**。现有普通、限定 L4 原样保留。

## 已完成的技能更新

共享证据技能新增独立版本 `theory-empirical-contribution/1`，并接入构思和研究入口。
它要求同一贡献的 L4、实质近邻区别、非平凡理论、可区分验证和实现对应同时成立。
旧 `advantage-evidence/1`、L0–L4、历史卡片、统计假设及实验契约没有改变。
独立审查 PASS，8 个行为案例、10 个验证命令及精确补丁/回退检查通过。
代码与评审结果已在共享 GitHub 分支 `codex/l5-theory-empirical-20261115` 留存。

## 已验收的实证基础与边界

正式结果仍仅来自 `20261029T021229Z_fresh5`，五个完整训练 seed29/53/71/97/113，
固定官方 Re-TACRED train58465/valid19584/test13418，冻结 RoBERTa-large，只训练输出头。
Q 对 uniform 的 Macro 平均差 +.305841；同一选定模型上下文置换 CE 损失增加 +.202014。
条件性配对 t 支持原合取门禁；独立近似正态 seed 差、固定数据集和 n5 限制均保留。
双侧 sign-flip .0625 未达到 .05。Q 对 softmax/dephased 的差异仍不确定。
详见 [原 L4 验收](20261115-q-rmea-ordinary-l4-acceptance.md)。

本轮没有训练、真实数据 forward、GPU 分配、COS 恢复或师弟服务器访问。
以下结果属于代数与文献审计，不是新的全量实验，也不升级实证证据。

## 第一候选为什么停止晋级

从实际代码的 α_t=(|φ†ψ_t|²+ε)/Σ_t(|φ†ψ_t|²+ε) 推导固定头的
共享测量态兼容性界。该界描述一个关系共用的 φ 能否同时达到各句子的独立最优。
200 组代数 fixture、每组25个 query 检查通过；独立复核使用另一求根路线也通过。

但是独立审查对 L5 接受给出 **BLOCK**：其主要内容是标准广义 Rayleigh 商与
PSD 核交推论，实质新颖性未建立。更关键的是，反例显示“证书为零但 shuffle CE
明显受损”和“证书为正但 shuffle 完全不改变输出”都存在。它既非已验收局部对齐
现象的必要条件，也非充分条件。d8 的 ε 嵌入还会让这个界明显变松。
因此先停止这条路线的真实诊断和晋级，保留它作为普通数学工具，不包装成新理论。

- [冻结候选与证明](../evidence/rmea/20261115-l5-round1/theory-candidate-v1.md)
- [独立科学审查与反例](../evidence/rmea/20261115-l5-round1/theory-acceptance-review.json)

## 近邻与跨领域比对

十篇原文按指定页面阅读并记录版本/哈希；这是有界近邻审查。
[QSANN](https://arxiv.org/abs/2205.05625v2) p4 已讨论平方内积注意力，
[Schuld](https://arxiv.org/abs/2101.11020v2) p6 已建立密度矩阵特征映射。
“使用 Born 权重”和 Tr(Eρ) 恒等式本身不足以构成这里的新理论。

跨领域查阅 [Barnum–Knill](https://arxiv.org/abs/quant-ph/0004088v2) p10 Eq37 后，
确认 token frame 的白化形式恰为已知 pretty-good measurement。
其平均态判别保证针对指定态集合与先验；当前任意学习 query 的语义选择/分类收益
并不由此推出。40 个关系效果是分别二元测量，代码也没有实施联合40结果 POVM。
见 [映射与迁移边界](../evidence/rmea/20261115-l5-round1/pgm-transfer-boundary.md)。

[Attention is not Explanation](https://aclanthology.org/N19-1357/) §4.2 和
[Attention is not not Explanation](https://aclanthology.org/D19-1002/) §§2.1/3.2–3.4
已系统讨论置换、非唯一解释、uniform 和训练种子变异。新的解释性主张须与这些近邻区别。
此外，对高阶量子注意力近邻中一项泛化的“softmax 单层只有二阶”直觉，构造了精确
三变量三阶系数 −1/18 的反例；这只纠正该直觉，不否定其整篇复杂度定理。
全部 [阅读收据](../evidence/rmea/20261115-l5-round1/reading-receipts.json) 保留覆盖不足。

## 正在重设的问题

已知恒等式给出 CE(z′)−CE(z)=(p−e_y)·(z′−z)+KL(p||p′)。
前项与任务标签方向有关，后项非负。一个固定、标签无关的三 token 置换反例中，
平均 CE 损失增加 .004991689，但平均方向项为零，增量全部来自 KL；accuracy 保持 .5。
这说明单看正 CE 损失不足以认定正确任务方向选择；不撤销旧 L4 的干预依赖解释。
该公式是已知凸分析工具，本身也不作为 L5 创新。

下一步只推荐一个方向：先明确 Q-RMEA 的哪种可区分机制性质超出这些通用工具，
把正确任务方向、扰动曲率和具体测量约束分开，完成近邻与反例检查，再决定是否值得
做冻结开发集上的小规模固定头诊断。成功门槛是出现实质不同且可证伪的机制命题；
若仍只是已知分解或与任务证据无关的几何性质，就停止晋级。当前没有新的正式实验指令。
见 [新问题与反例](../evidence/rmea/20261115-l5-round1/alignment-question-v2.md)。
