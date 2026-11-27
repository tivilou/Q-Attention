# 20261127T-091122Z | context-formation-screen | Born词间表征七门1通过，停止确切结构

## 0. 项目工作流总览

原40类关系分类输入是两个有序实体所在句子的冻结骨干词缓存h1024、角色4维和mask。标签仅在原TRAIN训练CE及事后评估使用，未读新VALID/TEST。Q/K角投影经原3比特两层电路形成词对Born权重，混合64维value，经Linear+tanh形成逐词隐藏表示，再按subject/object均值有序拼接为128维，由Linear输出40类logits。AdamW反馈更新本头，骨干不更新。本次改动位于冻结缓存之后的信息形成，而不是把旧分类logits相加。

## 1. 这次想法是什么

[机制解释] 假设先形成词间隐藏表示，能让量子相关信息比末端按类utility加权更有用。gen124只改全字段编码后四效用门失败；本轮实质改变头内计算角色。通用Born注意力已有前作，角色位置不同不自动构成创新。

## 2. 怎么验证

原8192FIT/4096已暴露TRAIN、seed13、logical128/micro4，Q/D/dot/RBF/pairMLP各1024步。所有fit完成后只评final1024，无best选择、补seed或参考重训。Q/D参数99616，三经典分别206568/206632/214952，共用同形value/context/classifier初值和同读出，经典刻意更大，不宣称有效容量匹配。源/spec身份和实际成本见[终态报告](/docs/evidence/rmea/20261127-context-formation-screen/terminal.json)。

## 3. Case Study与边界

没有采集sample-trace.v2、真实句子逐词中间值或预声明微批机制轨迹，均不从聚合结果反推。本确切配置已停止且无继续/replay消费者，Case Study为not_required_closed_exploratory，非ready；不能支撑样本因果判断。更完整的13节说明见[解释](/apps/idea-explanations/?id=q-rmea-born-context-formation)，保留旧技术说明源。

## 4. 结果

[已测事实] 五臂共5120更新/655360样本暴露，各自4096条final评估；campaign墙钟8561.884329795837秒，exit0/stderr空，CPU终态审计和原分配归还通过。

| 实现 | all40 Macro-F1 | CE | 训练外层秒 | 评估秒 |
| --- | ---: | ---: | ---: | ---: |
| born_context | 0.194431284867154 | 0.951032754196575 | 5191.462 | 83.528 |
| dephased_context | 0.236669337526696 | 0.917846464090815 | 5194.165 | 45.272 |
| dot_context | 0.316167850959820 | 0.935815877916271 | 1590.311 | 11.819 |
| rbf_context | 0.470602913950714 | 0.652813309468657 | 1542.314 | 12.083 |
| pair_mlp_context | 0.364636706911204 | 0.768105525314554 | 1741.508 | 13.499 |
| 复用 nativeQ | 0.38930060257417304 | 见原报告 | 未重训 | 未重评 |
| 复用强 Gcov | 0.512054050292468 | 见原报告 | 未重训 | 未重评 |

| 事前必要门 | 结果 |
| --- | --- |
| q_margin_over_dephased_context | FAIL |
| q_margin_over_dot_context | FAIL |
| q_margin_over_rbf_context | FAIL |
| q_margin_over_pair_mlp_context | FAIL |
| q_margin_over_native_quantum | FAIL |
| q_margin_over_strong_gcov | FAIL |
| fixed_q_dephasing_ce_worse | PASS |

fixed-Q双末态去相干CE=1.218354486404434，对原Q的CE差+0.26732173220785904。只此门通过，六效用门全部失败。

## 5. 从结果学到了什么

[已测事实] 技术实现、完整训练、保存恢复和相干敏感性正证据保留；Q的实际效用低于独立D/三经典和两旧参照。单纯把量子核移到词间信息形成，没有在此冻结协议下解决效用差距。

[诊断支持] 全40类均有支持，Q预测类别22/有TP类别19，D为28/23，RBF为37/33；事后聚合描述显示输出覆盖差异，不说明哪个词或优化因素导致差异。未另换指标救援。

[尚未证明] 过度平滑、容量、优化、冻结表征信息边界都不是本轮确定的根因；末态D内部仍有相干，CE敏感不等同独立量子贡献。主代理独立CPU复算不是外部研究者复现。

## 6. 当前证据判断

确切profile科学STOP；新L3/L4-T NOT_ACCEPTED，原限定L4及全部历史保留。不是完整单seed正式实验、独立确认、新颖性清关、新T或硬件优势，也不否定全部量子上下文家族。

## 7. 下一步方向

推荐一次更高层的信息/学习反馈路径复盘：围绕冻结骨干缓存后的量子后处理共同前提执行一次更高层的信息/学习反馈路径复盘：针对既有相关祖先排重，最多比较三个真正改变可学习信息或反馈位置的原任务量子假设，保留一个可证伪线索或NONE。先零新fit/GPU/heldout；不把旧QLASS位置改写、通用Born注意力或编码/宽度/深度/LR/seed/预算救援当新方向。保持原40类任务、独立量子L4-T目标、旧L4和全部正负/无效历史。

假设为“若改变共同的可学习信息/反馈前提，可能产生与这些后处理头不同的任务预测”，尚待排重，不预选新科学候选。最小工作是核对相关祖先与原QLASS计算/反馈路径，最多三种结构比较；0新fit/GPU/heldout。准入需一个不重复关闭前提的任务相关可证伪线索及强同输入经典对照，缺失则NONE/暂停该父方向投资。资源预算仅一次有界源码/证据比较，最多两项必要原作；科研范围已授权。不会以新名字重开本profile。

备选为保留NONE/投资暂缓判断；经典RBF仍低于既有强Gcov，只保留其对照价值，不启动衍生项目或替换量子目标。下一科学模型尚未选定。

## 证据

- [完整终态聚合记录](/docs/evidence/rmea/20261127-context-formation-screen/terminal.json)
- [原冻结协议](/docs/evidence/rmea/20261127-context-formation-screen/protocol.json)
- [结果台](/apps/experiment-results/)
- [版本化解释](/apps/idea-explanations/?id=q-rmea-born-context-formation)
