# 共同词态与 observable value：原文核对后仍无新结构候选

本轮完成了 gen105 后冻结的只读排重。10 个本地源文件身份核验通过，并取得、解析和按范围阅读 QSANN 官方 v2 原文。结论为 **NONE_NEW_STRUCTURAL_CANDIDATE**：没有证据表明已执行过完全相同的原生共同训练配置，但“同态有界 observable value 加任务训练”已有直接先例，且所拟读出仍是本项目已记录的 D 组件。本轮没有提出一个新增的任务信息路径，故不据此启动训练。这个结论不否定一切共同学习，也不把 gen102 扩大为全家族不可能性。

## 实际读到的原文

[Li、Zhao、Wang，Quantum Self-Attention Neural Networks for Text Classification，arXiv:2205.05625v2](https://arxiv.org/pdf/2205.05625v2)，官方 PDF 共13页、804181字节，SHA256 `95561d5444e91b185a60c39838ce7590c33779511803b461bf7b9e67ed2c51ad`。实际读了第1–7页的文字，视觉核对第3、4、5、7页；第8–13页、上游代码、补充材料及出版版本差异未审计。官方落地页的修订日期为2023-09-28，PDF标题页日期为2023-09-29，分别保留。

第3页式(3)、第4页式(4)–(7)把同一输入态送入 query、key 和 value 电路；value 是酉变换后的 Pauli 期望，因此每个坐标有界。其路由是 query/key 期望上的 Gaussian，而非本项目的类条件 Born 重叠。第5页式(8)–(10)、第6页梯度和第7页算法1给出任务损失及量子 ansatz/decoder 的优化，多层输出又成为后续公共输入态。没有把它说成与原生1028→12 preparer、40个秩一类效应、EPS词归一化及任意类Hermitian值完全相同的实现。

网页搜索服务报404/api_route_not_found，没有产生搜索证据；随后一次官方落地页请求确认论文身份，一次官方PDF请求取得v2。没有轮换入口或重复下载。阅读记录见 [reading-receipt.json](reading-receipt.json)。原文、抽取文字和页图只在本地私有目录保留。

## 与已知机制的实际关系

| 固定近邻 | 本轮核对的范围 | 结论 |
| --- | --- | --- |
| gen102 固定teacher value-span | 冻结全部原生表示，拟合独立teacher值；不是任务共同训练 | 原三门0PASS/STOP保持；不改rank、ridge、teacher或seed |
| 同态值 D | `v_rt=Tr(V_r rho_t)`，保留原Born/EPS池化 | 共同训练改变优化协议，没有改变这个组件公式 |
| QSANN v2 | 同态Pauli值、量子注意力、任务优化 | 直接组件先例；不证明原生完整配置等价或已经无效 |
| Q-BAM | 顺序测量和有符号回作用 | 与投影前期望不同；旧停止家族不重开 |
| gen97 / gen92 | 独立h-linear值的协方差 / 冻结值后训练router | 不同于同rho值，但公式差异还不是效用证据 |
| conditional token POVM | 原权重和任意signed值的精确再表达 | 没有施加同态值约束，也没有新的干预 |

完整逐项记录和哈希见 [comparison.json](comparison.json)、[source-pins.json](source-pins.json)。这只排除了指定的本地近邻及一篇直接原文，没有声称穷尽所有文献或所有历史代码。

共同词态会同时收到 route 和 value 的梯度，这是普通链式法则。固定效应、observable和bias，只改变词态时，`dz = sum alpha_t dv_t + sum alpha_t(v_t-z_without_bias) dp_t/(p_t+EPS)`。它可以描述优化路径，却没有自动给出新的学习机制、泛化界或任务改善。D 的一/二阶密度矩表达仍为已知恒等式。

若以后有具体新干预，须保留最强适用Gaussian/real训练对照，并将route/value去相干分成2×2。Hermitian64或real16同函数求值只核技术一致性，不能替代独立训练的经典基线。既有gen105参数差和全部负结果保持，不转成支持此方向的证据。

## 后续工作与等级

下一步只设计至多一个改变实际量子注意力信息路径的任务假设：从现有完整研究和gen98/102/104/105结果出发，逐项说明与最近已关配置的结构差别、最强同预算经典对照、关键消融及小筛选的预测/停止门。可使用一次既有协议下的有界设计咨询，由主模型作最终选择；没有结构差别则返回NONE。这里没有重新建立数据资格、runner或证据框架的要求，也没有自动训练准入。具体边界见 [next-direction.json](next-direction.json)。

本轮为0 GPU、0新任务forward、0原生circuit调用、0fit、0模型API、0新增容量。原限定L4、gen98/102/104/105及所有已关确切profile与全部正负历史保留。原Q-RMEA目标继续ACTIVE；新L3及L4-T仍为NOT_ACCEPTED。尚需同一核心的公平完整训练与适当独立确认、关键消融、实质近邻区别、非平凡正确理论、可区分预测、实现一致性和独立审核。
