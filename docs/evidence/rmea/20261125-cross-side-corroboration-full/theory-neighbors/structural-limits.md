# 跨侧印证门控：原论文近邻与实际结构边界

对象固定为当前运行的plugin SHA6174cb5aba15c106ed432dbe08c6f750dae6b70499a32be34797e9d744f0b131。
此分析不改变五路全量源码、门、预算或结果读取时序，不是新的理论成熟度接受。

## 已实际阅读的近邻

1. **ConQAR**, Ping/Chen, arXiv:1912.11720v1，PDF第3–4页，§4.3–4.4，式(6)、(8)–(15)：
   已有文本密度混合、两个对象的矩阵相乘、trace重叠及由另一对象影响的注意力。
   其注意力作用于卷积特征维度和两个完整对象矩阵，不是这里的主语侧token先验。
   这个区别有待评价意义；它不让“密度矩阵+跨对象注意力”成为本项目首创。
2. **QMSAN**, Chen等，arXiv:2403.02871v3，PDF第6–9、11–14页，式(9)、(10)、(13)、(15)–(21)：
   已有mixed-state query/key的Hilbert–Schmidt重叠、SWAP估计、归一化注意力和value聚合。
   它通过部分迹形成单token的mixed态；当前候选则平均指定宾语侧的纯态，作为
   单侧门控并乘已有relation-query Gaussian。二者不是逐字同一架构，但核心trace工具已知。
3. **QCSAM**, Chen等，arXiv:2503.19002v2，PDF第3–6页，§III.A–E，式(6)、(8)–(20)：
   使用复内积及复系数value态叠加；当前候选只用非负squared-overlap门控和实值value。
   因此“使用phase”这个宽泛说法不构成区别；两者实现的输出对象也不同。
4. **QKSAN**，arXiv:2308.13422v2，复用已存原HTML及
   `.codex-artifacts/relative-return-training-20261122/qksan-reading.json` 的§III范围：
   quantum kernel注意力和条件value处理早已存在。本轮不重新下载或重跑其模型。

以上只做所列原文范围阅读；不认证论文实验分数、代码或全部理论。无穷举或priority clearance。
PDF保留原版版权，不能跟我们的公开结果包再发布；只公开链接和阅读记录。

## 当前门控的准确解释

令phi(k)=|psi(k)><psi(k)|，K(k,l)=Tr(phi(k)phi(l))。
当前宾语summary正是经验均值 mu_O=(1/m)sum_j phi(k_j)，因此

    g_i = <phi(k_i),mu_O>_HS = (1/m)sum_j K(k_i,k_j), i in S.

这是quantum-fidelity kernel下的mean-affinity/对象侧加权degree，不是新算符或新测量。
g在所有relation query间共享。为防止从名称臆造功能，必须保留以下两点：

### A. 无EPS的跨relation交比不变

对严格正的b和g，取alpha_ri=b_ri*g_i/sum_t(b_rt*g_t)。任意两relation和两token，

    alpha_ri*alpha_sj/(alpha_rj*alpha_si)
      = b_ri*b_sj/(b_rj*b_si).

因此该共享门可以按宾语上下文移动全部query的token先验，但不能任意重新定义
relation间的相对token偏好。这不是“不能提高分类”的定理。真实代码有EPS，故不能
把该等式直接当作生产代码精确不变量；EPS占比可控时才有相应近似。

生产代码w_ri=b_ri*g_i+eps。若四个b*g均至少tau>0，log交比相对无EPS值的误差
不超过2*log(1+eps/tau)：四项中两个正号、两个负号，各增量在[0,log(1+eps/tau)]。
当某个g=0时这个界不适用，EPS会主导那项；不能把它删掉以维持漂亮等式。

### B. diffuse object上下文的保真度稀释

若m个宾语态只有一个与subject态重合、其余正交，则g=1/m。
取一个subject token，其base为h、decoded value为+1；m个object token的base
各为h/m、decoded value各为-1。未门控时两边总base相同。保持真实代码每token
都加一次EPS，门控后subject和每一个object的权重恰好都为h/m+eps，因此

    logit = [h/m-h-(m-1)*eps]/[h/m+h+(m+1)*eps]
          = -(m-1)/(m+1).

若改成每个object base均为h，或把所有object先合并后只加一次EPS，公式不同，不能混用。
这是encoded-state/base/value层面的固定构造，尚未证明这些指定量能在同一个native
参数状态共同实现，不是自然文本上的因果或性能预测。m越大，单个真corroborator
可能越难超过固定rival，但编码器也能训练成让多个token匹配；不构成全家族不可能性。

## 不照搬原文的两处断言

- QMSAN第13页式(16)后称相同mixed态SWAP总通过。其式(16)本身却给出
  p=(1+Tr(rho^2))/2。rho=I/2时p=3/4，不是1。只采用正确公式，不继承该段解释。
- QCSAM把squared-overlap描述为丢弃phase。准确说它丢弃overlap的整体复相位，
  但仍可依赖态内部相对phase，例如|+>和|->有相同population却正交。
  这既不证明我们的phase具有独立实证贡献，也不认证其全部复杂度主张。

## 当前结论

现有同一核心的数学解释更清楚，但这些是已知工具和直接结构边界，不满足L4-T的
非平凡新理论门。先让冻结全量研究判定utility/消融/公平替代是否值得独立确认；
不因近邻碰撞中途停止训练，也不把简单恒等式包装成T创新。下一步由全部结果决定。
