# 有序实体的双输入路径与角色/上下文交互筛查

本轮先冻结任务前提及实际分类终点，再用精确反例检查能否区分机制。结论：非零角色×上下文交互，包括固定value后的交互，不能独立支持量子机制。本轮关闭这一判别预测；未选择新方法，L5未接受，旧限定L4不变。

## 原正式源码的完整绑定

缓存中原confirmation协议列出的36个源文件全部存在，逐文件SHA256组成的协议汇总为 `494c9e3e0e7dba468380c692d743419600b92d994d73f680edc40ebc45b0c7ce`，与原fresh5记录完全一致。[完整文件哈希与检查绑定](source-verification.json)。本轮没有取得新数据或执行模型。

`scripts/audit_rmea_tokenization.py` 的 `marked_words` 在subject两侧加@，object两侧加#。原runner先对带标记的输入执行冻结encoder，然后只汇聚原词对应的subwords，排除标记本身。排除标记的池化并不排除它对上下文向量的潜在影响。另一路显式R包含到subject/object中心的截断有符号距离/32，以及两个实体成员bit。故输入是 `H(tokens,ordered_entities)` 和 `R(tokens,ordered_entities)` 两条路径。

实际纯函数fixture：tokens=[u,links,v]，subject=u、object=v时词序列为[@,u,@,links,#,v,#]；交换角色则为[#,u,#,links,@,v,@]。原词映射均为[null,0,null,1,null,2,null]。固定tokens/spans后改label/types/metadata不改变标记。只从AST抽出这个纯函数执行，没有import其模块或encoder；没有测量hidden向量是否、如何变化。

由此，uniform头虽然忽略显式R，却同样接收带有实体标记上下文的H，不能称为按构造entity-blind。固定H只换R是对显式角色路径的干预，不是新自然实体查询；自然实体变更需要重新预处理/编码。此修正不改变原始比较的数值或限定L4支持。

## 任务前提与终点

候选自然配对需真实train行有相同原tokens和不同有序实体对；其存在数量、标签语义和适用性尚未测量。没有假设独立双视图、角色交换后H固定、已知逆关系映射或量子可实现标签分布。不能为交叉路径的输入凭空赋新真标签。

固定同一类对r,s，含原生signed values/decoder/bias的margin记作m(H,R)。四格为AA=m(H_A,R_A)、AB=m(H_A,R_B)、BA=m(H_B,R_A)、BB=m(H_B,R_B)，交互I=BB−BA−AB+AA。总变化恒等于(AB−AA)+(BB−AB)，也等于(BA−AA)+(BB−BA)；对称路径平均亦成立。这些是已知有限差分恒等式，不是新理论。分类正确性仍需其余全部40类，不能用这一类对或换标签的CE相减直接声称Macro改善。

## 两个执行前固定的经典反例

两例都是已有relation_softmax接口允许的有限参数构造，检验逻辑蕴含，不声称是自然RoBERTa特征或已训练模型。有效词数2，实体单词span交换；令class0 query为sqrt(8)e1、key第一坐标实现所需affine score，除sqrt(8)抵消。value map读取独立H坐标；class0 decoder选此值，class1 decoder与两bias为0，使margin等于所列加权值。其余类可合法保留，但这里不推断全类预测。

第一例：score只依赖subject成员bit、系数ln2。交换subject后，第二词概率为1/3和2/3。H_A的词utility为(0,1)，H_B为(0,2)。依AA、AB、BA、BB顺序，margin为[1/3,2/3,2/3,4/3]，I=1/3。scorer没有内容依赖，输出仍有角色/上下文交互，原因是角色权重乘上变化的value。

第二例：所有格固定utility=(0,1)。score是内容项加角色项；subject系数ln2/2，交换角色使第二词相对第一词的角色log-odds为−ln2/2、+ln2/2。H_A、H_B内容log-odds为ln2/2、3ln2/2，得到四格log-odds[0,ln2,ln2,2ln2]。普通softmax输出为[1/2,2/3,2/3,4/5]，I=−1/30。内容score与value可读不同H坐标，所以改变内容score不要求改变value。即使value固定、scorer的content/role logits相加，也能经softmax产生非零输出交互。

[预先固定的输入/判据](design-packet.json)、[精确Fraction结果](design-checks.json)、[筛查判定](theory-audit.json)。没有参数搜索、随机seed、模型/encoder前向或训练。主代理完成核查，未声称独立理论审评。

## 判定与下一步

上述反例排除了“非零交互独特支持量子机制”的当前预测；没有排除所有量子角色/上下文机制，也没有证明所有已匹配模型等价。无需为已经失去判别力的指标再扫描私有数据或启动模型。

下一步把这项完整源码修正、实际margin终点和反例加入一份实质更新的独立设计包，先更新模型能力/精确传输预检，再进行一次有界设计咨询。允许返回没有合格候选，禁止换弱模型购物。只接受能说明可独立测量任务前提与不同终点预测的候选；改方法需新L4，不继承旧证据。rank、spectral/quotient、ordered/joint/complement/outcome/spectrum/partialswap等停止门保持；不扩展泛attention论文队列，不启动正式实验。
