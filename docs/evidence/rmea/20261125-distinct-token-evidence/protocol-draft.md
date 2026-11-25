# 固定下一小筛选：删除同token二次自乘

上一轮Qmeanproduct .504045671350好于Qcov .478786491123，但未胜最强Gcov
.512054050292，且CE/正确数更差；这不构成成功，也未隔离原因。新的假设只删除
均值乘积里相同token位置的自乘，检验不同位置之间的配对证据是否更有用。
这是新开发假设，不重新解释旧八门失败。原先所有确切配置及历史结果保持原状。

固定mu=sum(alpha*v)，M=mu_even*mu_odd，S=sum(alpha**2*v_even*v_odd)，
U=M-S=sum_{t!=u}alpha_t*alpha_u*v_t,even*v_u,odd。U不除以
1-sum(alpha**2)，不加温度或缩放。仍用原32相邻通道及40x32零初始化decoder，
其他部分原样复用gen96共享query初始状态，各81924 trainables。

五臂固定：Qdistinct、Ddistinct、Gdistinct、QpermutedDistinct、QcollisionMass。
后者只用(1-sum(alpha**2))*M。它由主模型加入，用于区分真正的去自乘差异和
单纯attention集中度调制；外部建议没有替代这项主模型判断。

Qdistinct必须在全40类Macro严格胜四个新对照和全部九个已存参照，共十三门。
所有比较身份及值在selection.json固定，包含上一轮全部五臂和四个更早参照，
尤其保留.504045671350的Qmeanproduct和.512054050292的GnativeCov。
不删对照、转成只胜自身基线、只看稀有类、CE或特定关系来宣布成功。

沿用同TRAIN8192/4096、句子组隔离、全部40类、seed13、1024updates/b128/micro4、
原AdamW/FP32/TF32off；五fit结束才统一评估，总5120物理updates。相同预算
900sec/arm/3600sec总上限，调度现有双GPU。旧参照直接复用，无baseline重跑。
八门旧规则和失败状态不改；这次十三门全过也只允许设计完整单seed协议。
任何门失败即暂缓本确切profile，不扫归一化、集中度、通道、pair、rank、token、
scale、LR、seed、budget、cohort或对照救援，不推断整个方法族不可能。

只测试新增U及mass-control独立双token公式/梯度、单token U=0、全同value时
U=(1-sum(alpha**2))*M、padding、零moment共享端点和clone actual8。复用query/
circuit/稳定core及原生命周期覆盖；不重跑串行/暂停/多卡/COS恢复矩阵。

U是已知非对角二次统计身份，适用于任何归一attention，并非新T或量子排他性质。
它对token位置对称；即使有收益，也未证明subject/object绑定、次序或量子优势。
attention集中度与value分布的混合效应须由固定mass-control区分，结果仍是暴露过的
单seed开发集筛选，不能直接接受独立L3/T。0新VALID/TEST或COS、Windows0模型/
特征/逐例数组。尚未实现、审核、分配或运行。
