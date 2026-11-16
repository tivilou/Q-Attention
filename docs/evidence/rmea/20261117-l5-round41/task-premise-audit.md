# 同任务原文的任务前提与分类终点

本轮选读了一篇此前未审的经典少样本关系分类方法，检查它能否给 Q-RMEA 提供任务相关理论。结论是这条直接迁移尚缺桥梁；两项固定代数检查进一步排除了过强推论。它们是基本逻辑检查，L5未接受，旧限定L4保持。

## 来源、版本和边界

[FAEA exact v1](https://arxiv.org/abs/2204.12111v1)，arXiv水印2022-04-26，作者Chunliu Dou、Shaojuan Wu、Xiaowang Zhang、Zhiyong Feng、Kewen Wang。原PDF标题为 *Function-words Adaptively Enhanced Attention Networks for Few-Shot Inverse Relation Classification*，metadata标题少了Adaptively；两者明示保留，同一exact ID/版本和五作者核对，未当作逐字标题相同。

取得8页PDF，SHA256为cfcc6ef84a7b4b2a2e2613655fb8e98df34a20e6fa8ba38f5227b4d7b2aa56a5。选读页1-5、7-8，实际查看图像页1、4、5；页6未读，Fig2未作视觉审计，没有代码、训练或数据复现。页4指向的Appendix证明未在该取得版本中定位到；没有取得补充文件，不推断其他版本或位置也没有证明。

网页搜索工具本次返回接口404，没有站点结果；随后两次官方arXiv查询成功。量子/extraction查询返回的是无关物理论文，measurement/relation查询引出了FAEA的经典dot-product分析。所选论文不是量子同任务先例；有限检索也不证明量子方法不存在。其他结果只保留metadata。当前公共证据/审计/所属refs的exact ID、标题和缩写去重无命中；已审CAML/LAAT、QHA、ReadingWhy、CNet继续复用。

## 任务与实现对应

| 部分 | FAEA选读原文 | 已执行Q-RMEA |
| --- | --- | --- |
| 任务 | FewRel episodic support/query，N-way K-shot，描述输入辅助新类分类 | 固定40类Re-TACRED，完整train/valid/test |
| 表示 | BERT768；句子、实体、关系名称与描述；global/local混合 | 冻结RoBERTa1024词特征＋4角色；affine及pi*tanh→单位3qubit态 |
| 信息交互 | support已知标签决定同类/异类图，消息传递后平均prototype | 40个固定已学习二元effect独立对有效词评分；未实现support图或prototype |
| 预测 | query与prototype点积的softmax，Eq12-13 | epsilon归一化attention×带符号utilities＋bias→40实数logits |
| 指标 | 多个episode平均accuracy；FewRel逆关系结果 | 官方固定cohort全40类Macro-F1；固定头permutation CE局部对齐 |

FAEA的source数字和定理陈述属于作者报告，没有成为本项目测量结果。两者都涉及注意力和关系分类，尚无等式/分布假设把其完整模型或理论映射到当前PQC、value/decoder及分类终点。

## 打印定义与两个独立的逻辑边界

原文p4 Eq14将符号解释为向量inner product，却写 `norm(s)=abs(s)/norm2(s)`。按标量字面解释，s非零时值为1，s为零时未定义；这是实际渲染页观察到的定义歧义。没有擅自认定作者本意，也没有将它替换后宣称推翻原定理。该定义本身不足以提供所需的相似度下降解释。

第一项检查明确假设一个待检验的cosine版本：若功能词表示之间的绝对cosine不大于关键词之间的绝对cosine，是否就能推出两者相加后的cosine更小？固定：

```
c_i=(1,0), c_j=(1,1)
f_i=(0,1), f_j=(2,1)
x_i=c_i+f_i=(1,1), x_j=c_j+f_j=(3,2)
cos²(f_i,f_j)=1/5 < cos²(c_i,c_j)=1/2
cos²(x_i,x_j)=25/26 > 1/2
```

所有向量非零、坐标非负，被比较的点积也均非零；所以绝对cosine排序与平方排序相同。混合点积含有 `c_i·f_j=2` 和 `f_i·c_j=1`，原先只约束两个纯分量的相似度，没有约束这些交叉项。这是对这个明确写出的cosine迁移命题的反例；不是对作者未明确意图、未取得的附录、FAEA完整算法或真实句子的反例。它要求今后的任务前提显式控制混合项，不能只假设“功能词相似度更低”。

第二项检查针对分类终点：同类点积增大是否足够保证分类？固定query `q=(1,0)`、真类prototype `p=(1,1)`、另一类 `r=(0,6)`。最初点积为1、0，真类胜。仅作同类对称正更新 `q'=q+p=(2,1), p'=p+q=(2,1)`，r保持原值；此时真类点积从1升到5，但另一类点积变成6，argmax翻为错误类。

这说明成对点积增长本身不足以推出多类分类保证。该更新只是检验逻辑蕴含的固定构造，既不是FAEA完整adaptive support graph，也不是Q-RMEA干预。没有据此断言对方算法降低性能。softmax不改变点积分数的argmax，而Macro-F1还涉及全类/全样本统计；没有从局部相似度直接跳到该终点。

两个向量例子在执行前固定于proof_protocol.json；Fraction精确算术结果已写入[结构化判定](theory-audit.json)，输入、分数和限定一并保留。没有参数搜索、随机seed、私有数组、模型前向或训练。核验是主代理完成，未声称独立理论评审；基本反例也不是L5的新理论。

## 对当前研究的约束与下一步

旧限定L4仍只支持原Q-RMEA相对uniform的效用及固定值局部alignment。它没有测量上述关键词/功能词的可加分解、交叉项、support图或prototype几何；从旧报告推定这些前提成立没有证据。当前量子效应、readout和终点必须共同进入下一份理论设计。

下一步停止扩展generic attention论文队列，先写一份任务前提与分类终点设计包：前提应能由冻结上下文特征、角色和标签独立测量；列出匹配的非线性label-wise竞争解释；给出对实际带符号value/class margin的不同预测，并先找解析反例。只剩Q-realizability、一般容量/平滑或缺失交叉项时不选该机制。已关闭的rank、quotient/spectral、ordered/joint/complement等路线保持关闭；不把本轮基本检查包装为L5，也不启动训练。
