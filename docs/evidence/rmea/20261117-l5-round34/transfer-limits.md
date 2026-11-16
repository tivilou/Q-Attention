# Q-RMEA L5 Round34：实际裁剪 AdamW 与可借用理论的边界

本轮补上了一个实际实现对应：四份缓存的正式实验源码逐字节匹配既有公开统计审计；其中两份 head 源码也匹配此前阅读的开发快照。三篇原文的指定章节和九页图已核对，但尚未得到区别共享 head 对照的理论预测。限定 L4 保持，L5 尚未接受；没有改模型、启动训练或读取私有模型/数据。

## 正式实验究竟做了什么

正式源码组合哈希为 `494c9e3e0e7dba468380c692d743419600b92d994d73f680edc40ebc45b0c7ce`。四份文件的完整哈希、公开审计及定位见 [source-verification.json](source-verification.json)。这次建立的是静态源码的对应，开发 Git revision 与正式源码组合哈希仍是不同身份；并未把源码匹配写成每步训练重放成功。

`rmea_confirmation_run.py` 第111行构造 AdamW，只显式指定 lr 和 weight_decay；第114行每 epoch 随机置换；第117–119行做 minibatch CE/backward；第123行记录**裁剪前**的首步梯度；第125行对整个 head 做全局范数裁剪，再执行 optimizer.step；第127行才取 after 参数。因此，首步原始梯度和参数差之间还隔着共享裁剪系数、Adam 矩与解耦衰减。既有五 seed 协议保持20 epochs、batch128、lr=.001、weight_decay=.01、clip_norm=1。

额外读取了 [PyTorch v2.7.1 AdamW](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adamw.py)、[Adam](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py) 和 [clip_grad](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/utils/clip_grad.py) 的固定 tag 源码。该参考版本默认 betas=(.9,.999)、eps=1e-8，二阶矩包含当前梯度，分母在开平方后加 eps；裁剪系数为 min(1,C/(||g||+1e-6))。这是固定版本参考，尚未新增认证 fresh5 安装分发、foreach/fused 分派或逐步浮点轨迹。

## 三篇原文没有自动给出的结论

| 原文 | 已读到的实质结果 | 与当前方法之间的缺口 |
|---|---|---|
| [Cattaneo & Shigida, 2602.01642v2](https://arxiv.org/abs/2602.01642v2)，pp1–8、22 | 用平滑性与有界导数条件，在小步长下用无记忆迭代近似 Adam；噪声二阶展开再经坐标噪声比平均等简化得到 sharpness 方向预测 | Definition3.1 的 eps 在根号内；没有完成当前全局裁剪、解耦衰减及误差常数的桥接。sharpness 方向也不是 held-out Macro 保证 |
| [Zhu & Huang, 2611.17736v1](https://arxiv.org/abs/2611.17736v1)，pp1、4–5、14–17 | 在显式平滑性、噪声、metric 与 momentum alignment 条件下给出 stationarity 高概率界 | p4 的预条件器使用过去梯度；p14 裁剪真实梯度之外的噪声。两者均不同于当前 total-gradient clipping＋AdamW；这些条件也未为当前训练认证 |
| [Abedi et al., 2202.08232v7](https://arxiv.org/abs/2202.08232v7)，pp1、7–9、11–12 | 几何局部电路/观测量、独立随机参数、平方损失 gradient flow 下，随 qubit 数增大得到 lazy/linearization 控制 | 当前是固定3 qubits、学习的 prepare/effect、词间归一化、带符号仿射 head、CE 和有限 AdamW；没有建立对应的增长 qubit 家族。定理不直接适用也不等于实测发生了 feature learning |

日期身份依据官方 metadata 和 PDF 水印；2611.17736v1 水印为13 Nov2026，没有凭编号后缀猜日期。JOTA 模板与 Quantum 接收页脚分别按观察记录，未额外认证出版社状态。三篇均为选定范围阅读，全文证明、实验与代码尚未复现。访问均为 HTTP200，未触发浏览器；普通无头浏览器回退授权保留。

## 首步近似抵消，不能外推成整段训练无关

这是已知优化器代数，用来约束解释，并非新贡献。记裁剪后梯度 h_t=c_t g_t。参考 AdamW 的标准实数更新为

    m_t=β1 m_(t−1)+(1−β1)h_t，v_t=β2 v_(t−1)+(1−β2)h_t²，
    θ_(t+1)=(1−ηλ)θ_t−η mhat_t/(sqrt(vhat_t)+eps_Adam)。

从零矩状态的首步，适应项是 −η h/(|h|+eps_Adam)。当 |h| 远大于 eps_Adam 时，单个坐标更接近按符号更新；原始梯度较大并不同比例产生较大的坐标更新。有限 eps 下缩放仍影响结果，参数衰减项也需保留。仅凭“Born 梯度较大/较小”推断其训练更快或泛化更好，缺少这一步分析。

但 c_t 是随时间变化的，不能把首步的近似缩放抵消外推到整个动量历史。一个固定输入的标量反例即可区分：g1=2、g2=−1、β1=.9、C=1，使用参考裁剪的 δ=1e-6，则

    未裁剪的第二步一阶矩去掉公共正因子后：β1*g1+g2=4/5>0；
    裁剪后：β1*2/(2+δ)−1/(1+δ)
            = (−1/5+(4/5)δ)/((2+δ)(1+δ))
            = −66666400000/666667666667 <0。

二阶矩分母及 bias correction 都为正，因此这两个历史的第二步**适应项方向**相反。比较的是与一阶矩同向的 normalized direction；参数位移另有负学习率符号，衰减部分不在该比较中。精确分数运算已执行核验。该反例是规定的两个标量梯度，既非 Q-RMEA 产生的梯度，也非真实样本、实验重放或新的算法效果。

## 研究判断与下一步

本轮关闭“借通用 Adam 收敛/隐式偏置/lazy 定理直接把现有结果升级为 L5”的捷径。没有证明这些理论永远不能适配，也没有证明 Q-RMEA 的优化劣势或优势。静态正式源码对应得到加强，科学判别证据仍缺失。

下一步只推荐一个可审查产物：为原生 Born 与 real-overlap 对照定义**函数空间中注意力对带符号 value 的对齐变化**，固定共同的 head 变化与更新历史，检查是否能推出一个有方向、可证伪且区别于通用优化器性质的比较预测。先做理论与近邻原文核对；若只能得到梯度链式法则、参数缩放身份或事后解释，则不选为方法，不启动实验。已有连续研究授权覆盖该设计工作；新全量正式实验仍仅交师弟。
