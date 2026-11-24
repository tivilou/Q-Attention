# 位移单光子测量：完整数据资格实验（事前冻结）

动机和证据见hypothesis.md及已完成gen70/71；强RBF12必须保留。新假设：保持Q-RMEA的关系独立测量、词归一化、共享value/decoder，非Gaussian测量相似度的有限距离零点可以学出有用的证据排除结构，而非仅改变局部带宽。没有已定位的语义根因，不预称创新、量子排他性或硬件优势。

每个词/关系仍是原12维线性key/query，初始化和输入与原real12相同。坐标两两形成6个复位移，z=(a_even+i*a_odd)/sqrt(24)。单模d=|z-w|²，k1(d)=exp(-d)(1-d)^2。生产用[(1-d)exp(-d/2)]²避免极大输入时多项式乘指数产生inf*0。六个模式为乘积态，各模式初始|1>，Q概率为六个k1乘积。6个光学模式/六个初始光子和原3qubit资源不同；这个核可高效经典计算。

奇偶消融不是完整Fock基去相干：每个模式施加Delta_parity(rho)=(rho+Pi*rho*Pi)/2，Pi=(-1)^N。由于Pi D(z)|1>和D(-z)|1>仅差全局符号，两态均施加该幂等通道后的Hilbert-Schmidt重叠为[k1(|z-w|²)+k1(|z+w|²)]/2；六模式独立，取六个平均的乘积。只移除奇偶扇区之间的相干，保留同奇偶扇区相干；Q-D差只能支持该消融的贡献。

固定新五路：fock1、parity_fock1、vacuum_born（exp(-sum d)）、gaussian3_born（exp(-3sum d)，与fock1原点局部斜率相同）、rbf12（已实现的原Gaussian无EPS masked softmax）。前四路raw=(probability+1e-6)*mask后按词归一化；rbf12严格沿用原代码。五路同12维key/query、81028注册参数、同共享初始value/decoder/bias、同seed/data/order/优化/精度。注册参数相同不保证有效容量相同。

复用既有完整fair-reference的uniform/quantum/dephased/real12/softmax12五路，按原summary/manifest与训练配置、cache、初始化和order验证；使用既有已接受的完整实验比较接口，不新增跨运行通用复用机制。不得拿本轮新方法只与弱softmax比。

技术准入只做新公式/独立有限维算符及奇偶通道、梯度、掩码/参数/初始化和真实TRAIN前8样本各一更新。复用原训练/多GPU/本地best-latest/续训/10% COS能力，不做新的短小样本效果淘汰赛。旧4192/4000筛选含稀有类缺失且已经暴露；本新方法通过技术准入后直接采用完整原生协议一次评价，避免把代码可行性误作成熟度或用短期分数延长旧配置。

固定完整资格：TRAIN58465/VALID19584/TEST13418，seed13，20epoch/batch128/nativeFP32、原AdamW/lr.001/weight_decay.01/clip1，共每路9140更新、五路45700，在调度分配的既有双GPU动态队列执行。逐epoch VALID all40Macro选择最佳，平手保留最早；最终只按该选中状态报告TEST全40类。全部新五路完成后与复用五路共同报告。主门：fock1 TEST all40Macro严格高于其余九个新/旧条件，九门皆正才允许准备独立确认。CE/每类/耗时/修复损害完整报告为次指标，不增显著性要求。固定全量失败关闭本确切单光子/6模式/scale24/eps/奇偶消融/20epoch配置，不改光子数、模式数、缩放、温度、seed、epoch、metric或控制救援。

修复/损害采用十路训练全部完成后的固定VALID-selected状态进行一次描述性TEST回放，统计相同13418条样本上fock1相对各九路的repaired/harmed/both_correct/both_wrong；不重新选择checkpoint、不更新模型、不用它挽救主门。旧native五路的selected状态仍是分析依赖，保留原已固定的恢复对象，若迁机先从旧run固定恢复再做最终分析。逐例数据留计算节点，只有聚合计数进入Git。

资格单seed且官方划分已暴露，不能接受L3。仅九门全过时进入另行冻结并执行的独立确认，候选固定、所有对照同预算、所有预定重复保留；不选择性补种子。确认前仍不接受L3，更不接受L4-T。完整研究源、环境/输入身份与结果通过Git；可续训的非Git最小状态仅按既有10%节点到COS，本地实时best/latest。无新购买/扩容/师弟服务器访问。

理论边界：Fock重叠零点、核、奇偶twirl和局部斜率均为已知工具。Schuld/Killoran2018已讨论光学核/非Gaussian；Gan/Leykam/Angelakis2022已研究Fock输入、光子数与表达能力/核方法。其后再有“Fock更有表达力”不是新T。须在真实完整效用与独立确认后，对同一贡献另行建立实质近邻区别、非平凡正确理论、可区分预测、实现一致性及适当独立内容审核。
