# 多实体查询前提之后，还缺少什么

结论：多实体查询的有限训练事实已建立，但本轮仍没有合格新机制、非平凡新理论或L5。独立设计意见指出一个更具体的任务侧缺口：同一文本片段对查询A是证据，对查询B却是干扰，能否被独立、盲于原标签和模型输出地识别。

## 三篇原文带来的边界

本轮从研究问题出发核对官方原文，限定阅读范围、哈希与缺口保存在[reading-receipts.json](reading-receipts.json)。没有复现这些论文的实验，也不把不同任务F1与本项目Macro直接比较。

- [PURE，NAACL2021](https://aclanthology.org/2021.naacl-main.5/)在PDF pp3-4用typed subject/object markers重新编码每个实体对；p5的高效近似专门禁止text tokens读取markers，并绑定position，才使原文表示可以共享。原生solid-marker模型没有这个同样的屏蔽条件。相同原tokens不等于相同H；采用这种近似是改模型，需要新证据。
- [PL-Marker，ACL2022](https://aclanthology.org/2022.acl-long.337/)在pp3-4用solid subject与levitated objects组合建模同subject的多个查询。其bidirectional/inverse训练构造不能作为Re-TACRED天然标签逆映射的证明。多查询和早期marker交互已有明确经典来源。
- [ComplEx，ICML2016](https://proceedings.mlr.press/v48/trouillon16.html)在p4式9-11以`Re(sum w_r*e_s*conj(e_o))`建模有向关系，并给出纯实数展开。复数方向性不是量子特有新贡献。其二元知识库补全任务、signed trilinear分数和原生词attention/40类决策不同；文中引用的sign-rank证明并未在本轮独立认证，也没有新rank演示。

## 两个应避免的理论捷径

独立确定性评估器若逐个计算`f(X,q_i)`，重排查询列表只会重排输出。这对任何这种经典或量子函数都成立，不是语义查询变换的定理。

ComplEx的标量例子取`w=i,s=1,o=i`，正反向复幅分别为`+1,-1`，signed实部保留方向，而同一标量的模平方均为1。这是原公式的精确简单算术，不是新定理；尤其不是Q-RMEA无法分方向的反例。原生模型还有query-marked H、显式R、类效应、词归一化及signed decoder，未经证明不能把两者认作相同函数。详细[主裁定与解析边界](primary-audit.json)。

## 新事实包及独立判断

R44固定58,465条train的审计找到5,286个同tokens/不同query/不同label组，覆盖21,798行；这个实测事实与三篇新原文共同进入[候选盲的新包](consultation-packet.json)。它不同于R43旧包，没有主代理偏好的新候选草稿。

咨询复用了带日期、同会话的P0元数据先于LLM-Stats采集的HLE单来源代理快照，没有声称新采集或绝对科研能力排名。top5与API/native-max未映射状态分开；新的A6catalog预检URLError，零上下文调用、零重试。并不证明每个模型离线。按model-consultation-orchestrator回退规则，隔离同宿主子代理只读取并核对同一包，返回[no_qualified_question](consultant-answer.json)。它没有网页/源码/数据/服务器/实验或其他代理权限；不是外部模型、原文或定理认证。

主代理接受有界否定，不重投相同包寻找肯定答案。冻结包沿用E9的一句旧“多查询尚未建立”已过时，独立意见指出了它；新增R44实测项优先，主裁定明确更正，保留原包以免事后改证据。换查询可能改变H与utilities，本轮未测数值变化。对所有39竞争类的严格正margin对应唯一正确argmax；等于0仍需真实tie规则。

## 已固定的最小下一步

[证据归属盲审协议](evidence-ownership-protocol.json)现在仅是设计，不是已取样/标注/运行：在全部10,153个多查询token组中，忽略label值，按固定salt与canonical tokens的SHA256选前32组，每组取按span坐标排序前两个query。不会只挑5,286个混合标签组，也不按易标注程度、输出或想要的结果替换样本。

两位隔离读者需看未改动文本与实体查询，隐藏原example label、预测、mixed-label分层信息；记录支持A、支持B、共同支持、无关或不确定。所有32组及分歧保留。仅模型一致性仍为暂定意见，不能称人类gold。先解决private pack与读者权限，再冻结实现、review/canary和共享调度配方；没有授权把raw数据发送到外部咨询接口或联系其他人。

至少一个明确且双方一致的归属变化也只形成有限语义前提；零个则关闭该32组探针，不宣称总体不可能。正结果仍须连接实体感知经典对照下的完整40类决策预测，才能考虑模型实验。它不是Case Study回放、语义编辑或逆关系测试，不复活已停实验家族。

本轮零模型/encoder前向、训练、private数据/数组读取、COS恢复及新正式实验；原限定L4和历史结果不变。新模型的共同协议/headroom门仍未完成，任何改encoder/机制都不能继承旧L4。
