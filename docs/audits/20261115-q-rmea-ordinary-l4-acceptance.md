# Q-RMEA：普通 L4 证据验收（2026-11-15）

当前 `advantage-evidence` 1 下，**限定的普通 L4 目标已验收**：在官方 Re-TACRED 固定测试集上，训练后的 Q-RMEA 头相对 uniform 有 Macro 效用，且同一选定模型的非实体上下文权重与词值对齐，在预声明置换干预下具有 CE 功能作用。统计结论以独立、近似正态的训练种子差值为假设。

这是对已经完成的正式全量实验 `20261029T021229Z_fresh5` 的后续分级审计。五个原计划种子为29、53、71、97、113；每个使用58465训练、19584验证、13418测试样本。此次新增训练、推理、师弟服务器访问和 COS 读取均为0。

| 原冻结端点 | 五种子均值差（正值有利） | 原95%区间 | 原 paired-t 门槛对应双侧 p |
|---|---:|---|---:|
| Macro：Q − uniform | +0.305841（30.58个百分点） | [0.281372, 0.330310] | 4.1141e−6 |
| CE：同模型置换 − 实际 | +0.202014 | [0.178473, 0.225554] | 1.8401e−5 |

两项均5/5同向，原至少4/5同向、区间下界大于0及干预不变量门槛通过。CE覆盖全部13418例，13414例可置换；uniform不变性和重构恒等式残差各满足2e−5容差。重复记录的 uniform ablation 与主比较是同一对照，不能算独立第三次确认。

**必须保留的统计限制：**两项原描述性精确双侧 sign-flip p均为 **0.0625，未通过0.05**。五个种子不足以验证分布形状，因此这里只接受原参数假设下的统计支持。原始 AND 门槛解释为单一联合主张的 IUT；这不是事后宣称预注册了校正方法，也不给出两个区间的同时95%覆盖率或各端点独立校正发现。

uniform少了打分路径参数，故总增益仍有容量混杂；同一 checkpoint 的匹配置换仅支持局部对齐的功能作用。一种置换干预也可能落在学习分布之外。量子独有效果、全部增益由对齐产生、关系条件或相位的必要性、语义证据正确性、SOTA、效率、硬件优势及跨数据集泛化均未建立。原 classical/dephased 不利或不稳定结果继续保留。辅助 shuffled Macro 仍是事后一致性分析，没有升格为冻结确认端点。

## 为什么现在可以登记

原 v3 将所有控制绑定为主指标，不能忠实表示原来的 Macro 效用与 CE 机制门槛。此次增加了逐端点 v4 格式，仍使用同一个 policy1：各端点保留原指标、方向、原值和门槛；五个选定模型以原 closed-run inventory digest 绑定（这不是 checkpoint 文件 hash），群体和选择规则由源码及七文件报告审计共同核实。旧 L2/L3 卡和原合同未改。

新格式明确区分原 paired-t gate 与描述性 sign-flip sensitivity。所有关键对照仍须通过；新格式不能让漏种子、换指标、换模型、漏负结果、失败 gate 由 sensitivity 替代或 OR 决策过关。Schema/CLI 的低等级一致性问题在首轮被独立审查阻断，修复后重新冻结并复审。

精确复审结论 **PASS_WITH_RISK，must_fix=[]**。44项测试、原13项 legacy 测试、实际卡与v3/v4全部10个等级样例的 Schema、独立构造的正反例、独立46个拒绝用例及服务器实际校验通过。项目技能入口已核对，doctor36/36；无需全球安装或新算力。

共享技能的审核代码已通过独立分支发布并回读：[`be3cfac311cfea0efecf7322a8cabb906de8d6ce`](https://github.com/tivilou/ai-skills/commit/be3cfac311cfea0efecf7322a8cabb906de8d6ce)。它包含目标技能所需的既有 v3 依赖与本次 v4，排除了原有未提交的 sample-trace 修改，并保留远端main及其他本地历史。

## 验收材料

- [新证据卡](../evidence/rmea/20261115-l4-acceptance/evidence-card.json)
- [接受记录](../evidence/rmea/20261115-l4-acceptance/acceptance.json)
- [精确独立审查](../evidence/rmea/20261115-l4-acceptance/acceptance-review.json)
- [冻结表示合同](../evidence/rmea/20261115-l4-acceptance/review-contract.json)
- [原样命令、输出及退出码](../evidence/rmea/20261115-l4-acceptance/verification.json)
- [此前统计审计](20261115-q-rmea-l4-statistical-audit.md)与[原普通L3审计](20261114-q-rmea-ordinary-l3-scope.md)

本机工程留档 `.codex-artifacts/qattn-l4-registration-20261115/` 保留 `owned-changes.patch`、`verification.json`、可运行的 `rollback.py` 与原始备份。校验器修改位于共享技能 `scripts/endpoint_policy.py`，公共入口按v4显式分派；原v1/v2/v3入口保持原义。旧格式拒绝新卡（exit1）、新格式严格通过（exit0）；旧格式13项测试保持通过。rollback已执行独立副本恢复并验证保留原dirty内容；本机应用默认仅预览，`--apply`须精确hash匹配。没有删除实验数据或变动云资源。
