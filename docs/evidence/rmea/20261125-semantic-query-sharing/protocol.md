# 下一轮选择：共享关系查询的固定小筛选

本轮任务证据：Q比D拟合更好但评估更差，稀有类差距大但不是全部欠拟合/过拟合。
Astra/high给出NONE，只建议D注意力替换Q读出；该组件回放不推进当前机制选择，未选。
主agent独立选择一个新的、尚未证成的共享结构假设，不把外部建议写成支持结论。

所有39个公开关系名按同一正则拆词，以固定binary/sqrt-count矩阵做top7 SVD，
逐行L2归一；null独占第1坐标，正类使用其余7维。40x8固定code乘8x12训练系数，取代40x12自由
query raw坐标。Q/D再通过原pi*tanh和原3qubit浅层circuit；G仍用原Gaussian query。
不增加词输入或独立经典分类器。仅96而非480 query参数，预计各80644 trainables。
使用原seed13自由query到固定code的最小二乘初始化；不声称router初值与旧模型相同。
shared value/decoder/bias和原key保持同样初始化/训练方法。

四新臂Qshared/Dshared/Gshared/Qpermuted：所有结构臂有相同name信息、code秩和相同
共享参数化；Qpermuted沿固定public-name hash顺序循环移位39行，null不动，完整
Gram谱/conditioning不变而语义对应打乱。code不使用训练文本、label目标、支持数或
模型结果，精确数组/词表/奇异值及SHA固定。SVD退化子空间不保证跨库bitwise重建。

原8192fit/4096assessment/全部40类/同组不重叠/seed13/1024updates/b128/micro4，
原AdamW/FP32/TF32off，四fit全结束后最终评估。复用已有nativeQ/D/Gbase/Gcov，
不重跑baseline；Qshared要胜Qpermuted、Dshared、Gshared以及四旧参照，七门全过
才允许完整单seed协议设计。差距可能来自已知共享/优化归纳偏置，不能直接归因
coherence、统计方差或量子理论。新输入结构对控制公平，registered count不等于
有效容量公平。小筛选与L3/理论接受分开，不声明新理论或novelty。

失败后暂缓这一确切rank7/name-token/初始化/1024配置，不扫rank/词法/阈值/LR/seed/
预算/样本/控制救援；不以低支持组或强制正类指标替代主Macro。下一动作是实现此
插件和对实际新语义做一次有界检查，然后由调度器运行四路小筛选。尚未运行。


执行补充：固定原AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1；
FP32/complex64、TF32off。最小二乘在CPU FP64、gelsd driver中使用实际FP32
code及原seed13 FP32 query，然后将系数保存为FP32；全局随机数不用于编码或
投影。编码文件SHA固定并随科学文件上传。原自由query保留为非训练buffer，
live forward仅使用code@query_weights。null码保持独立。query行数/输出40类不变。
registered80644相同不宣称effectivecapacity相同，也不声称原自由query初值保真。
4×1024=4096physical updates，900sec/arm、3600sec总预算，动态双GPU readyqueue。
原core50ce858f与host238f7e68不变。只查新映射、buffer、解析/差分query梯度、
原生替换公式/Gaussian NumPy、padding、clone actual8一次update及一轮语义审核。
无串行/续训/COS生命周期矩阵。短可重启筛选0新COS，已有输入持续保护。
Windows只接收代码、配置、40类聚合报告，无权重/特征/逐例数组。全部历史保留。
