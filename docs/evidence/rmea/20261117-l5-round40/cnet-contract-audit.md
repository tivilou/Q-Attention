# CNet 原文、固定源码与 Q-RMEA 函数合同

本轮核实了直接 Born-attention 先例，并保留具体函数差异。它缩小了可主张的新颖性范围，没有提出或验证新方法。旧限定 L4 保持，L5 未接受。

## 版本和阅读范围

[CNet v1](https://arxiv.org/abs/2610.08592v1) 的 arXiv 水印日期是 2026-10-06，标题页内部日期是10-07；实际查看页1、11。页11把依赖输入的振幅/Born路由列为后续设想。首版 abstract 的 Born 分类头不是 Born attention。这个选读结论不等于全文穷尽否定。

[CNet v3](https://arxiv.org/abs/2610.08592v3) 的水印日期是10-15，内部日期10-16；选读页1、6、11、12、14，实际查看图像页1、11、12、14。Section11 和 Fig2 明确给出复数内积模平方、因果归一化和复数 value 混合。未取 v2，未认定首次出现版本或全球优先权。论文中的语言模型指标、有限差分精度及优先权陈述均是作者报告，本项目未复现。

[公开仓库固定提交](https://github.com/crasmarum/CNet/tree/0073ea943aba3fd6962f06ac8718e9726e4858fe) 日期为11-10，晚于v3。五个文件已通过官方 git-blob API 取得，逐一验证 Git blob SHA1 和 SHA256；这是当前提交的静态阅读，不是论文实验代码版本认证。raw HTTP 超时，随后按用户要求尝试无头浏览器，遇到 ERR_CONNECTION_RESET；最后改用已经可访问的官方 API 读取同一对象，未更换版本。未运行上游代码。

## 实际对应

| 部位 | CNet 论文与当前源码 | 已执行 Q-RMEA 正式源 |
| --- | --- | --- |
| 查询 | 当前 token 的投影复数场 | 与当前句子无关的40个已学习关系 effect |
| key | 各 token 的复数投影 | 冻结1024维词特征＋4角色，经 affine、pi*tanh、三比特两层RY/RZ/CNOT产生单位纯态 |
| 核心分数 | `s=abs(Q dot conj(K))²` | `q=abs(phi^H psi)²` |
| 权重 | `A=s/(sum(s)+delta)`，因果mask；全零分母分支输出零 | `alpha=(q+epsilon)/sum_valid(q+epsilon)`，epsilon=1e-6；有效词非空 |
| 值与路由 | 复数V，后续投影、残差及多个block | real value1024→64，40×64 decoder给出带符号utilities |
| 最终预测 | 序列各位置的 Born 振幅归一化读出 | `z_r=b_r+sum_t alpha_rt*u_rt`，普通40类softmax CE |
| 证据 | 作者报告的语言建模实验，本项目未复现 | 原完整fresh5的限定普通L4：相对uniform的Macro效用＋固定值下局部alignment |

具体源码定位与哈希见[source-verification.json](source-verification.json)。CNet CPU forward/backward 已静态查看；没有独立梯度证明或数值认证。当前代码还出现linear/streaming/decay分支，但相关kernel、论文及性能未审计，不能作为本轮结果。Q-RMEA 正式源 composite仍为494c9e3e0e7dba468380c692d743419600b92d994d73f680edc40ebc45b0c7ce。

## 两个容易混淆的合同

第一，epsilon的位置不同。设同一有效分数向量为q、总和S，n个有效词。CNet的分母保护给出q/(S+delta)：delta>0时行和S/(S+delta)<1；Q-RMEA给出(q+epsilon)/(S+n*epsilon)，行和恒为1。当全部q=0，CNet的显式分支输出零，Q-RMEA得到均匀权重。这是池化接口的已知代数差异；不是训练模型发生该退化的证据，也没有对这种差异赋予新理论或效用。

当S>0时，Q的权重还可写成 `(S/(S+n*epsilon))*(q/S)+(n*epsilon/(S+n*epsilon))*uniform`；这只是已知加性平滑，不是新的量子贡献。不把上述边界转换成新的epsilon扫描实验。

第二，RMS pre-norm不等于单位量子态。当前 `tokennorm.h` 实现 `x/sqrt(sum|x|²/d+eta)`，其范数平方为 `sum|x|²/(sum|x|²/d+eta)`，一般不是1。`gollm.cpp` 中它位于CGelu和Q/K投影之前；后续未约束投影也不保证单位Q/K。论文页11的“unit norm”直观措辞不应代替这个实际合同。当前提交也不是论文运行版本，不能据此推断论文实验错误。

若非零Q/K写成范数与单位方向的乘积，分数包含query范数平方×key范数平方×单位方向重叠。delta=0且行非零时query范数消掉，key范数仍留在token权重中；delta>0时query范数甚至影响行总质量。故共享平方重叠形状不等于两个完整架构相同。

CNet页12对“没有干净实数类似物”的措辞不构成不可能性定理。R39已有精确real16双轴表达，且不是现有real8单平方控制；经典可模拟性既不抹除量子模块，也不产生新颖性或硬件优势。

## 新颖性与下一步

通用Born打分与行归一化已有直接先例；按类查询、加权池化、独立解码则在已审R14的CAML/LAAT原文中已有先例，见[历史审计](../../../audits/20261116-q-rmea-l5-round14.md)。本轮去重后复用该结论，不重复获取它们。具体受限PQC权重族与这些架构有别，但区别本身未建立非平凡理论、任务必要性或独立预测证据；相对matched real/dephased控制的优势也仍未成立。

下一步只做一次有界的同任务近邻门：先复用阅读库，最多两个针对quantum/measurement relation extraction与上下文特征的原始来源发现查询，至多选一篇尚未检查的直接近邻，读其真正任务/学习假设。只有超出已知Born核或label-wise池化、且能落到实际分类终点的机制才继续设计；否则关闭这条迁移。历史停止的rank、ordered/joint/complement等家族保持停止，不借新名字重开。没有新模型前向、训练、私有恢复或正式实验。
