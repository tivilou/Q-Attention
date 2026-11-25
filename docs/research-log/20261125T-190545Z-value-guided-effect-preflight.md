# Signed-value 引导 effect：有可区分行为，尚未选为任务方法

本轮完成冻结的零 GPU 代数与项目历史检查，并追加一个预声明的构造对照及一次独立静态机制审核。结果保留两面：该表达式确实区别于已核的几个原公式，但“每类分数上升”不足以支持分类收益；构造中的行为也由 real8 算符更新精确复现。新 L3/L4-T 未接受，原量子目标 ACTIVE。

## 做了什么

保留独立 signed utility，定义 B_t=rho_t+EPS I、B=sum B_t、A_c=sum u_ct B_t、z0=phi* A phi/(phi* B phi)、C=A-z0B，再令 q_s 正比 exp(sC)phi。C 固定于原 z0，s 只作非负符号变量。没有选择任务步长、增加模型参数或重开旧筛选。

[推导结论] 对有限 s>=0，分数不低于原 z0，初始导数为 2||Cphi||²/(phi*Bphi)。该结论比较更新前后，不保证随 s 单调。六词反例在 x=exp(2s)=6、100 的分数分别为 0.3535345608、0.3355024618，后者更低。原 Born+EPS 权重始终非负，分数仍在原词 utility 凸包。

[已测事实] 24 个一般纯态 context、每个40类，共960行/3840个代数点；原 ratio、实数化与归一检查最大误差4.22e-15，有限差分导数相对误差7.68e-10，约0.39秒。独立 Fraction 构造12个精确 moment 例、两个严格不等式及40类分类反例，全部 exit0。它们是一般算符例，未认证原 native ansatz 联合可达，也不属于任务结果。

[反例] 40类构造中原正确行保持1，错误竞争行由0升到1.9986546022，其余38行保持常数。各类分数均不低于原值，预测却变错，CE从0.3132616875升至1.3122783009。故单类性质不等于 margin/Macro 改善。强标量控制 alpha_tau 正比 alpha0 exp(tau u) 也已有导数等于 utility 方差的上升性质。

## 哪一点有区别，哪一点仍缺证据

[构造事实] 两个 context 使用同一 query、原 attention、全部64-channel value 和词态 populations：A 的两个词态都为 plus，B 为 plus/minus。原池化、scalar tilt、指定 gen97/98 统计，以及该例上的 gen90、frame whitening、weak/coherent readout 均不能区分两者。当前 C 更新在 A 输出0，在 B 输出 tanh(2)/(1+2EPS)=0.9640256520。这证明指定函数并不相同；没有据此作全近邻新颖性清关。

[控制边界] 该例本身是实数态，real8 更新精确复现；一般复8算符还可精确写成受约束的实16表示。这些是公平控制，不否决量子方法，也不证明复幅必要性、独立量子贡献或硬件优势。独立训练的强 Gaussian/实向量模型也不被限制为原 attention 的函数。

[已有任务证据的条件推论] 已核同一 checkpoint/state 与旧 shared-query 诊断。旧结果有431个原 quantum 错例，其 gold 行即使自由换 query，计算出的数值上界仍低于原 top rival。当前所有 rival 行又不低于原分数，因此这431例在固定 teacher 下仍受阻。这是对旧 FP64 证书的条件推论，没有读取新任务数组、重算例子或提供理想实数区间证书；不覆盖共同重训。旧59个共享冲突也不等于59个可修复例。

## 独立审核与下一步

[独立模型审核] 按现有 owner-preference 队列选择 gpt-6.2-astra/high，先核 ai-model-world metadata 和实际 A6API catalog，并核回传 model/effort。一次独立静态机制审核给出 PASS、无 GPU 准入、无新理论接受；要求澄清“严格增加”为“对每个正 s 均严格高于起点”。原冻结推导和回执保留，唯一文字修正见 derivation-v2.md 与 review-response.md。审核只读限定材料，非独立任务确认或全理论认证。

审核把可证伪假设明确为：内部 residual 几何是否提供超出原 attention/value 的标签相关信息；未来 g_gold-g_rival 的局部 margin 预测须优于 scalar tilt、diagonal 和公平 real-vector 控制，并保护原正确例。目前没有自然任务对齐证据。

**唯一下一步：两篇近邻原文审计。** 先核 reviewer 指定的 Knyazev 预条件 eigensolver 与 Ba 等 fast weights 的准确身份和权威全文，比较广义 Rayleigh residual、外积记忆与 query 更新，决定实质区别是否成立。当前仅是原文审计冻结，不自动准入实现、任务诊断或训练。详见 next-direction.json。

本轮数学检查0新任务行/feature/checkpoint/native-head/circuit/更新/GPU/服务器写入/COS；另有一次限定外部模型咨询。沿用现有 QT 环境，零安装。原限定 L4、gen98 STOP、gen102 STOP 与全部正负历史保留。
