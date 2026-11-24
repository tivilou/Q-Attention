# 句子共同静默：先排除代数重命名

本次Astra/high综合返回NONE。它提出的未决任务问题是：正关系证据的共同缺席能否改善no_relation决策，并保持正类之间的辨识？已有类别错误和CE/Macro分离没有证明这是现有错误的原因。

令39个正关系的单位查询为phi_r，局部效应E_r=(|phi_r><phi_r|+eps I)/(39(1+eps))，eps=1e−6。因为每个E_r不超过I/39，F=I−sum_r E_r为合法静默效应。给定完整文本x，各词态psi_t(x)已经允许依赖整个上下文；这里的制备仍是各词态的张量积。只在有效词上使用F，padding不产生测量。

对这个确切规则：S(x)=Tr[(tensor_t F)(tensor_t |psi_t><psi_t|)]=product_t(1−p_t)，p_t=sum_r(<psi_t|E_r|psi_t>)。这就是相同局部概率下的经典noisy-OR补事件。代数对每个输入及所有可训练参数成立，故任何使用S、原有q/mask/value/bias的相同40类读出也逐点一致；不是只对某个冻结模型成立。可微处梯度也一致。完整上下文化不破坏给定输入后的乘积恒等式。

若要求各词的约化态仍为指定纯态，仅改变联合制备也不能偷偷加入关联：对于某个纯约化态|psi><psi|，其正交投影Q满足Tr[(Q tensor I)rho]=0。由rho为正，rho的支撑局限在span(psi) tensor H，故rho=|psi><psi| tensor sigma。对各词递推得到乘积态。这不禁止重新定义为混合约化态，也不禁止在乘积输入上采用非乘积的联合效应。

独立NumPy计算直接构造64维联合态与F tensor F，对比两个标量静默概率；还核对mask、词顺序、正效应谱以及两个上下文化词共享同一可训练标量时的导数。结果见silence-reduction-check.json。这是公式核对，不是数据实验、语义证据或创新证明，无新增模型拟合。

因此不启动“乘积静默量子模型vs相同概率经典noisy-OR”的训练比较。这样的比较不能识别新量子机制。是否有用的多实例似然仍是经验问题，以上恒等式不否定其任务效用；真正非乘积测量也未被排除。下一项尚须找到有任务证据支持的具体联合事件与公平控制，不能用“量子缺席”“联合测量”名称替代方程或绕过既有ordered-memory、DPP、outcome-contrast负结果。

L3/L4-T仍未接受；旧限定L4和所有历史保持。没有选定新插件、申请GPU、读取新VALID/TEST、写COS或下载模型数据。

[独立算式核对](../evidence/rmea/20261125-product-silence/silence-reduction-check.json)。
