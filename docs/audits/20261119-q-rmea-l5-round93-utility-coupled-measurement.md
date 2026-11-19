# Q-RMEA R93：从读出瓶颈提出 utility 耦合测量

本轮不再以寻找高级量子术语作为入口，而是从 R81/R84 的真实开发观察出发，
比较表示拼接、value-aware 概率重分配与矩阵相对熵测量更新。前两项分别作为
竞争解释和强经典对照，第三项成为**待实现、待证伪的新机制草案**，不是新效果。

## 新机制及边界

原头在 inference 中先计算 Born 注意力，再用带符号 utility 读出。新草案让当前
样本 utility 先参与测量算符的一次熵近端更新，再生成注意力，最终仍用原 utility
读出；没有真实标签输入，没有经典残差分支，也没有额外可训练参数。
新增矩阵计算不免费，不继承旧双层纯态电路的实现成本或旧限定L4。

原 epsilon 可精确写成 E0=(P+epsilon I)/(1+8epsilon)，因此零更新在实数算术下
退回原模型。借助已知 Gibbs/矩阵相对熵工具，可证明单类自身聚合分数不下降。
本轮也明确构造了**该性质仍会使真实类别 CE 变差**的40类反例。因此它不是
任务收益定理，更不是 L5 新理论；需要真实任务数据和强经典对照决定价值。

原文新增限定阅读：Gupta/Berant 的 value-aware attention、Tsuda等的矩阵指数
梯度、Chen等的复数量子注意力。逐文版本/哈希/页码已保存。模型拼接与任务通信
原作复用先前记录，不冒充新发现；未宣称跨领域穷尽或新颖性放行。

## 独立审核与修订

首稿在精确机制审核中有两项 must-fix，虽 verdict 为 PASS_WITH_RISK，也没有据此
准入：算子范数归一化在零点附近不稳定，错配 utility 对照没有隔离构造路径。
v002 在任何实验之前改用固定平滑尺度 sqrt(1+tr A^2)，明确放弃首稿的正缩放
不变性；用解析 log(E0) 避免简并本征向量梯度，并把错配严格定义为输入ID决定的
同样本 token utility 置换，仅改测量构造，原样本最终 utility 读出不动。

两版各一次 gpt-6.1-sol/medium 独立请求，初版缺陷和字节均保留。
v002 精确审核 PASS_WITH_RISK、must-fix=0，仅支持继续设计有界 CPU canary。
审核并未运行代码、独立核读文献或确认近邻新颖性。

## 接下来

实现并审核一项 CPU canary：原模型零更新一致性、矩阵正性/归一化、真实电路
两词见证、平滑点梯度、零/近零 utility、单词/同态/奇异S、掩码，以及有害CE反例。
同时保留 simplex utility tilt 与 query-block pinching 对照；后者还改变更新幅度，
不能单独把其差异归为量子相干的因果贡献。全部科学计算走共享调度器。

本轮未实现该模块、未执行 canary、未训练新模型、未发布正式实验。正式协议与
独立验证预算尚未冻结；新全量实验仍交师弟。旧 scoped L4、R81非晋级、R84观察及
所有已停止路线保持。L5 NOT_ACCEPTED。

- [当前 v002 完整机制、证明、反例和对照](../evidence/rmea/20261119-l5-round93/utility-coupled-measurement-v002.md)
- [原文阅读回执](../evidence/rmea/20261119-l5-round93/reading-receipts.json)
- [跨领域迁移卡](../evidence/rmea/20261119-l5-round93/transfer-cards.json)
- [两版独立审核记录](../evidence/rmea/20261119-l5-round93/mechanism-review-record.json)
