# Q-RMEA L5 Round35：函数相同为何还不足以比较学习动态

本轮得到一个明确的比较边界：real-overlap 的参数尺度可以在保持当前注意力不变的同时改变后续更新；Born 的每关系 effect 又只覆盖完整纯态切空间的一部分。因此，单凭当前权重/相位不变量或一个理想状态空间切核，还没有得到当前方法相对对照的有方向预测。限定 L4 保持，L5 尚未接受，没有选择新方法或启动实验。

## 原文先澄清了什么

| 指定原文 | 已核对的内容 | 对本项目的作用 |
|---|---|---|
| [Jastrzebski et al., 2609.17352v1](https://arxiv.org/abs/2609.17352v1)，选读 pp1–4 | 对多份量子输入施加可训练局部变换，再测量 CYCLE/SWAP；三态及更高阶 Bargmann 关系和几何相位已有明确学习应用。p4 明确其 Gaussian 示例不构成对高效 Gaussian 专门方法的优势证明 | “利用高阶相位关系学习”已有直接近邻。其量子数据/集体测量和 shot 对比不能直接搬给当前词编码模型 |
| [Shirai et al., 2111.02951v2](https://arxiv.org/abs/2111.02951v2)，选读 pp1、3–4 | Eq17 是参数导数内积的量子切核；用初始固定切核替代训练过程需要局部线性近似等条件。给出的实例使用 MSE/SGD | 定义一个导数内积不是新理论；固定初始切核也未解释实际有限 AdamW 的全过程 |
| [Sarao Mannelli et al., 2006.06997v1](https://arxiv.org/abs/2006.06997v1)，选读 pp1–2、9 | 真实模型是实 Gaussian 测量、teacher 的平方强度标签、四次损失与球面 gradient flow；p9 说明对特定对称性和 random-guess phase 的依赖 | 标题 Complex Dynamics 指复杂动态，不能读成“复振幅模型”。平方重叠的外观不足以把相变/恢复理论转移至当前 NLP 分类 |
| [van Laarhoven, 1706.05350v1](https://arxiv.org/abs/1706.05350v1)，选读 pp1–5 | 归一化可让函数对参数缩放不变，却改变梯度和有效学习率；Section5.3 已讨论 Adam 的尺度依赖 | 归一化＋Adam 的尺度解释有明确先例。本轮的有限首步条件需另外核对，不能包装为新的量子机制 |

四篇均为指定范围阅读，未完成全文证明、实验或代码复现。[阅读范围与 PDF 身份](reading-receipts.json)记录精确版本及九页图。首篇内部 Dated16Sep 与 arXiv 水印15Sep、QTK 内部 Dated28Nov 与水印24Nov 分别保留，没有改写版本身份。

## 已知三态几何如何进入一个理想切核

固定准备态 psi_t，令 P_t=|psi_t><psi_t|，effect 态 phi 归一化，q_t=<phi|P_t|phi>。在完整复纯态空间、采用实内积 Re<.,.> 的水平梯度是

    grad q_t = 2(P_t−q_t I)phi，
    <grad q_t,grad q_s>_real
      = 4[Re(<phi|P_t P_s|phi>)−q_t q_s]。

三态乘积对应已知 Bargmann 不变量。把全部态同时复共轭会翻转其虚部，却保持上述实部和理想核不变；“有非零虚部/有方向相位”本身没有推出更好的更新。对 alpha=(q+epsilon)/S，S=sum(q+epsilon)，还须用 J=(I−alpha 1^T)/S 作归一化拉回，即 K_alpha=J K_q J^T，再结合带符号的 value 对比。这些是已知微分/链式工具，不是新的 L5 理论，也不自动给出分类效用。

更关键的是，实际 effect 并未在完整纯态空间直接执行这种梯度。正式源码每关系只有2×3×2=12个实 effect 参数；C^8 单位纯态模去全局相位的 CP7 是14个实维度。固定准备后，水平状态 Jacobian 的秩至多12，至少遗漏两个完整水平切方向。因此它的全空间拉回算子不可能处处等于正数乘14维恒等算子；若以 kappa I 近似，全空间算子范数误差至少 kappa。这个结论仅针对固定 prepare 的 effect 分支，不排除在有限可观测方向上的近似，也没有把共享 prepare 的联合变化一并否定。

实际电路还经过 tanh 角度映射，随后执行裁剪 AdamW。理想态空间的导数核不能省略电路 Jacobian、坐标尺度、矩历史和衰减便称为实际更新预测。

## 原生 real-overlap 的函数不变尺度族

正式共享 head 中，key(x)=Kx+c，query 每关系为 q_r。real-overlap 使用

    p_rt = (normalize(Kx_t+c)^T normalize(q_r))²，
    alpha_rt = (p_rt+epsilon)/sum_valid(p_rt+epsilon)。

保持 value、decoder 和最终 bias 不变，同时令(K,c,q)→s(K,c,q)，s>0。在所有相关向量的范数均高于 normalize 下限时，所有输入的 p、alpha、logit 都精确相同。这里缩放包括 key 的偏置，不能只乘 K 而漏掉 c；也不能把零范数或触发下限的输入纳入精确结论。

本轮读取固定 PyTorch v2.7.1 的 [normalize 源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/functional.py)：分母为 max(norm,eps)，该参考版本默认 eps=1e-12。普通 HTTP 25s 超时后，一次无头 Chromium 导航成功保存了页面快照；后续 eval 55s 超时，改从已保存快照离线提取，未重新导航。浏览器已关闭。快照折叠了空白，其哈希绑定渲染文本，不冒充原 HTTP 文件字节哈希；未据此新增认证 fresh5 的安装分发或浮点路径。

## 有限首步为何仍随尺度变化

下面只分析零初始矩的参考标准 AdamW 首步，不把它外推到20 epoch 全轨迹。设 eta>0，k=1−eta*lambda>0。无论全局裁剪系数如何，单个坐标的适应位移绝对值

    eta*|h_j|/(|h_j|+eps_Adam) <= eta。

因此缩放后的 attention 参数更新可写成 k*s*theta+d_s，||d_s||_infinity<=eta。先做共同正向径向衰减仍保持归一化函数不变；其后的有界非径向扰动相对参数尺度为 O(1/s)。

对一个固定有限输入池，假定原始各 query 与 key(x_t) 均非零。query 扰动范数至多 eta*sqrt(8)，key 输出扰动范数至多 eta*sqrt(8)*(||x_t||_1+1)，均不依赖 s。对非零向量，归一化差可以用2*扰动范数/原范数控制；当 s 足够大，更新前后也不触发归一化下限。故各 p 的首步变化为 O(1/s)。注意力分母至少为有效词数乘 epsilon，因此 alpha 同样连续地以 O(1/s) 变化。固定有限的带符号 utilities 后，注意力部分导致的 logit 变化也趋于0。常数可能很松，尤其含1/epsilon；没有预测实际固定初始化的速率或任务效果。

这个界已经容纳全局裁剪和有限 Adam eps，不需要假设“裁剪在 Adam 中完全抵消”。共同 head 必须固定或施加相同的外部受控变化；完整联合训练的 logit 变化还包括 value/decoder/bias 更新，不能把它们归入该 attention-only 界。该尺度族是说明函数值不足以唯一决定动态的条件结果；现有正式协议锁定了初始化，不能据此宣布原始对照实验无效。

一个精确有理数例子只核验这一边界：两条正交 key、query=(s,s)，初始 alpha=(1/2,1/2)。规定 query-only 梯度为(−1/s,+1/s)，零初始矩、eta=1/1000、lambda=1/100、eps_Adam=1e-8；s=2,4,8时该规定梯度不触发裁剪。令 d(s)=eta/(1+s*eps_Adam)，则更新 query=(ks+d,ks−d)，

    alpha1_after−1/2
      = ks*d / [(k²s²+d²)(1+2epsilon)]
      < eta/[ks(1+2epsilon)]。

三种尺度的初始函数相同，注意力改变量严格递减；精确分数命令已执行，exit0。该规定梯度/固定 key 的例子不是原生 CE、模型前向、完整训练或真实病例。[结构化审计](transfer-audit.json)保留输入身份和全部范围限制。

## 研究决策

本轮没有得到可选的有方向比较命题：高阶关系已有直接近邻，导数内积有 QTK 先例，完整纯态梯度不等于12参数 effect 更新，函数匹配也没有固定 real-overlap 的学习尺度。这些结果说明下一项证据需要什么，并非新的性能或量子必要性证明。

唯一下一步是核查**既有冻结正式产物**是否已记录实际初始参数尺度、首步原始梯度、after 参数及可恢复的优化器/输入身份，从而判断是否能设计一个尺度与更新历史受控的训练→局部对齐假设。先查紧凑报告/源码 schema，暂不恢复私有数组或运行模型；若只能回算通用 Taylor/切核身份，就不选为方法。连续研究授权保留；旧实验不重跑，新全量仍交师弟。已停止的 partial-swap、ordered、joint、outcome、spectrum 等家族没有重开。
